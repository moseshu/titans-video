from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet3DConditionModel
from diffusers.schedulers import DDPMScheduler
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from world_model_reasoning import WorldModelReasoning
import torch.nn.functional as F
from einops import rearrange, repeat

class TitansVideoGenerator(nn.Module):
    """
    基于Titans的视频生成模型
    支持文生视频和图生视频
    """
    def __init__(
            self,
            pretrained_model_name: str = "stabilityai/stable-video-diffusion-img2vid",
            reasoning_config: dict = None,
    ):
        super().__init__()

        # 加载预训练组件
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name, subfolder="vae"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.image_encoder = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # UNet backbone
        self.unet = UNet3DConditionModel.from_pretrained(
            pretrained_model_name, subfolder="unet"
        )

        # 世界模型推理模块
        reasoning_config = reasoning_config or {}
        self.world_model = WorldModelReasoning(
            dim=self.unet.config.cross_attention_dim,
            **reasoning_config
        )

        # 噪声调度器
        self.scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name, subfolder="scheduler"
        )

        # 冻结预训练权重
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # 条件投影
        self.condition_proj = nn.Linear(
            self.text_encoder.config.hidden_size,
            self.unet.config.cross_attention_dim
        )

    def encode_text(self, text: list[str]) -> torch.Tensor:
        """编码文本条件"""
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.text_encoder.device)

        text_embeds = self.text_encoder(**tokens).last_hidden_state
        return self.condition_proj(text_embeds.mean(dim=1))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """编码图像条件"""
        image_embeds = self.image_encoder(images).last_hidden_state
        return self.condition_proj(image_embeds.mean(dim=1))

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """编码视频到潜在空间"""
        B, T, C, H, W = video.shape
        video_flat = rearrange(video, 'b t c h w -> (b t) c h w')

        latents = self.vae.encode(video_flat).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        latents = rearrange(latents, '(b t) c h w -> b t c h w', b=B)
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """解码潜在表示到视频"""
        B, T, C, H, W = latents.shape
        latents = latents / self.vae.config.scaling_factor

        latents_flat = rearrange(latents, 'b t c h w -> (b t) c h w')
        video_flat = self.vae.decode(latents_flat).sample

        video = rearrange(video_flat, '(b t) c h w -> b t c h w', b=B)
        return video

    def forward(
            self,
            latents: torch.Tensor,  # Noisy Latents [B, T, 4, H, W]
            timesteps: torch.Tensor,
            text: Optional[list[str]] = None,
            images: Optional[torch.Tensor] = None, # 参考图 [B, 3, H, W]
            history_latents: Optional[torch.Tensor] = None,
            entity_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        """
        B, T, C, H, W = latents.shape

        # 1. 编码 Cross-Attention 条件 (Text/Image Embeddings)
        text_embeds = self.encode_text(text) if text is not None else None
        image_embeds = self.encode_image(images) if images is not None else None

        # 准备 Cross-Attention 输入 [B, 1, D]
        if text_embeds is not None and image_embeds is not None:
            condition_embeds = (text_embeds + image_embeds) / 2
        elif text_embeds is not None:
            condition_embeds = text_embeds
        else:
            # 如果都没有，必须有一个 fallback，通常 SVD 依赖 image_embeds
            condition_embeds = image_embeds if image_embeds is not None else torch.zeros(B, 1024, device=latents.device)

        if condition_embeds.ndim == 2:
            condition_embeds = condition_embeds.unsqueeze(1)


        if images is not None:
            with torch.no_grad():
                # 编码参考图: [B, 3, H, W] -> [B, 4, H/8, W/8]
                # 注意：encode_video 需要处理 T 维度，这里 unsqueeze 模拟 T=1
                clean_image_latents = self.encode_video(images.unsqueeze(1))

                # 移除 T 维度，得到 [B, 4, H, W] 用于 SVD Concat
                svd_image_cond_latents = clean_image_latents.squeeze(1)

                # 重复到 T 用于 World Model 输入 [B, T, 4, H, W]
                world_model_input = repeat(clean_image_latents, 'b 1 c h w -> b t c h w', t=T)
        else:
            # 纯文生视频模式 (无参考图)
            world_model_input = torch.zeros_like(latents)
            svd_image_cond_latents = torch.zeros(B, 4, H, W, device=latents.device)

        # 3. 世界模型推理
        reasoned_features, aux = self.world_model(
            current_latents=world_model_input, # 使用 Clean Latents
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            history_latents=history_latents,
            entity_memory=entity_memory
        )

        # 4. 融合策略
        # 这里的策略是：利用物理推理结果，微调"噪声"输入，引导去噪方向
        enhanced_noisy_latents = latents + 0.1 * reasoned_features

        # 5. 构建 SVD UNet 的输入 (8通道)
        # SVD UNet 需要输入: Concat([NoisyLatents, ConditionImageLatents])
        # ConditionImageLatents 需要重复 T 次以匹配时间维度
        svd_image_cond_latents_repeated = repeat(svd_image_cond_latents, 'b c h w -> b t c h w', t=T)

        # 拼接: [B, T, 4, H, W] + [B, T, 4, H, W] -> [B, T, 8, H, W]
        unet_input = torch.cat([enhanced_noisy_latents, svd_image_cond_latents_repeated], dim=2)

        # 6. 生成 added_time_ids (FPS, Motion Bucket, Augmentation)
        # 如果不传这个，SVD 会报错或效果极差
        added_time_ids = self._get_add_time_ids(
            B,
            fps=7,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            dtype=latents.dtype,
            device=latents.device
        )

        # 7. UNet 预测
        noise_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=condition_embeds,
            added_time_ids=added_time_ids, # 必须传入
        ).sample

        return noise_pred, aux

    # --- 辅助函数：生成 SVD 必需的时间条件 ---
    def _get_add_time_ids(self, batch_size, fps, motion_bucket_id, noise_aug_strength, dtype, device):
        # SVD 默认的微调条件
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids


class TitansVideoTrainer:
    """训练器"""
    def __init__(
            self,
            model: TitansVideoGenerator,
            learning_rate: float = 1e-4,
            use_8bit_adam: bool = False,
    ):
        self.model = model

        # 优化器
        if use_8bit_adam:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )

        self.scaler = torch.cuda.amp.GradScaler()

    def train_step(
            self,
            video: torch.Tensor,  # [B, T, 3, H, W]
            text: list[str],
            images: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        单步训练

        Args:
            video: 目标视频
            text: 文本提示
            images: 参考图像（可选，用于图生视频）
        """

        self.model.train()

        with torch.cuda.amp.autocast():
            # 编码视频到潜在空间
            latents = self.model.encode_video(video)

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.model.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            ).long()

            noisy_latents = self.model.scheduler.add_noise(
                latents, noise, timesteps
            )

            # 前向传播
            noise_pred, aux = self.model(
                noisy_latents,
                timesteps,
                text=text,
                images=images,
            )

            # 计算损失
            diffusion_loss = F.mse_loss(noise_pred, noise)

            # 辅助损失
            plausibility = aux['plausibility']
            plausibility_loss = F.binary_cross_entropy(
                plausibility,
                torch.ones_like(plausibility)
            )

            total_loss = diffusion_loss + 0.1 * plausibility_loss

        # 反向传播
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            'total_loss': total_loss.item(),
            'diffusion_loss': diffusion_loss.item(),
            'plausibility_loss': plausibility_loss.item(),
        }

    @torch.no_grad()
    def generate(
            self,
            text: list[str],
            images: Optional[torch.Tensor] = None,
            num_frames: int = 16,
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        生成视频

        Args:
            text: 文本提示
            images: 参考图像（可选）
            num_frames: 帧数
            height, width: 分辨率
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
        """

        self.model.eval()
        device = next(self.model.parameters()).device

        # 初始化潜在表示
        latent_h = height // 8
        latent_w = width // 8
        latents = torch.randn(
            len(text), num_frames, 4, latent_h, latent_w,
            device=device
        )

        # 设置调度器
        self.model.scheduler.set_timesteps(num_inference_steps)

        # 实体记忆（用于长视频一致性）
        entity_memory = None

        # 去噪循环
        for t in self.model.scheduler.timesteps:
            # 扩展用于classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            t_batch = torch.cat([t.unsqueeze(0)] * len(text) * 2)

            # 条件和无条件预测
            noise_pred_cond, aux = self.model(
                latent_model_input[:len(text)],
                t_batch[:len(text)],
                text=text,
                images=images,
                entity_memory=entity_memory,
            )

            noise_pred_uncond, _ = self.model(
                latent_model_input[len(text):],
                t_batch[len(text):],
                text=[""] * len(text),
                images=None,
                entity_memory=entity_memory,
            )

            # 更新实体记忆
            entity_memory = aux['entity_memory']

            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
            )

            # 去噪步骤
            latents = self.model.scheduler.step(
                noise_pred, t, latents
            ).prev_sample

        # 解码
        video = self.model.decode_latents(latents)

        # 归一化到[0, 1]
        video = (video + 1) / 2
        video = video.clamp(0, 1)

        return video
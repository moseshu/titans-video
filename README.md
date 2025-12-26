# Titans video

```bash
# 推理（生成视频）
python train.py --mode inference

# 训练
python train.py --mode train
```



## LLM 文本编码器（Qwen / Mistral / Llama / gemma3）

框架支持 `UMT5/T5 + LLM` 的双编码器：`--text-model` + `--llm-model`。LLM 通过 `transformers.AutoModel` 加载（建议 7B 用 `--llm-dtype bf16`）。

如果你的体系（例如 CogView / Hunyuan / Wan2.2）**不需要 LLM**，只用 `UMT5/T5` 作为文本编码器即可：不传 `--llm-model`（默认就是只用 `--text-model google/umt5-xxl`）。

少数新/多模态 `config.json` 的 hidden size 字段在 `text_config.hidden_size` 等层级里；如果自动推断失败，可手动指定 `--llm-hidden-size <N>`。

```bash
python3 train/videos/video_framework.py \
  --manifest train/videos/mock_data/manifest.jsonl \
  --pretrained-model /path/to/diffusers_weights \
  --text-model google/umt5-xxl \
  --llm-model /path/to/qwen_or_mistral_or_llama \
  --llm-dtype bf16 --llm-max-length 256 \
  --attention-type st --patch-size 2,2,2 \
  --local-files-only
```

## 外部 3D VAE（CogVideoX 示例）

已提供一个可直接用的适配器模块：`train/videos/cogvideox_vae_adapter.py`（类 `CogVideoXVAEWrapper`）。

```bash
python3 train/videos/video_framework.py \
  --manifest train/videos/mock_data/manifest.jsonl \
  --pretrained-model . \
  --vae-backend external \
  --vae-module train.videos.cogvideox_vae_adapter \
  --vae-class CogVideoXVAEWrapper \
  --vae-init-json '{"pretrained_path":"/path/to/CogVideoX-2b","dtype":"float16"}' \
  --attention-type st --patch-size 2,2,2 \
  --text-model google/umt5-xxl \
  --llm-model /path/to/mistral_or_qwen --llm-dtype bf16 --local-files-only
```

## 8×A100 分布式训练（Accelerate）

```bash
accelerate launch --num_processes 8 train/videos/train_video_accelerate.py \
  --manifest /path/to/manifest.jsonl --data-root /path/to/data \
  --pretrained-model /path/to/svd --output-dir outputs_video \
  --max-steps 100000 --batch-size 1 --grad-accum 4 --mixed-precision bf16
```

## SoraDiT 分布式训练（Accelerate，step-based）

适用于 `train/videos/video_framework.py` 同一套模型/codec/LLM 参数（但用 Accelerate 跑 DDP）：

```bash
accelerate launch --num_processes 8 train/videos/train_sora_accelerate.py \
  --manifest /path/to/manifest.jsonl \
  --pretrained-model /path/to/diffusers_weights_or_dummy \
  --output-dir outputs_sora \
  --max-steps 100000 --batch-size 1 --grad-accum 4 --mixed-precision bf16 \
  --lr-schedule cosine --lr-warmup-steps 1000 --min-lr 0.0 \
  --cfg-drop-prob 0.1 \
  --attention-type st --patch-size 2,2,2 \
  --save-limit 3
```

**`--pretrained-model` 是什么？**

- 当 `--vae-backend diffusers` 时：`--pretrained-model` 指向一个本地 diffusers 权重目录（需要包含 `vae/` 子目录），用于加载 VAE。
- 当 `--vae-backend external|identity` 时：`--pretrained-model` 不参与加载（可填 `.`），VAE 的来源由 `--vae-*` 参数决定。

### DDP 示例（UMT5 + Qwen2.5B + 外部 CogVideoX VAE）

```bash
accelerate launch --num_processes 8 train/videos/train_sora_accelerate.py \
  --manifest train/videos/mock_data/manifest.jsonl \
  --pretrained-model . \
  --output-dir outputs_sora \
  --max-steps 20000 --batch-size 1 --grad-accum 4 --mixed-precision bf16 \
  --text-model google/umt5-xxl --text-dtype bf16 --text-max-length 256 \
  --llm-model /path/to/Qwen2.5-3B-or-7B --llm-dtype bf16 --llm-max-length 256 \
  --vae-backend external \
  --vae-module train.videos.cogvideox_vae_adapter \
  --vae-class CogVideoXVAEWrapper \
  --vae-init-json '{"pretrained_path":"/path/to/CogVideoX-2b","dtype":"float16"}' \
  --attention-type st --patch-size 2,2,2 \
  --local-files-only
```

### FSDP（可选）

这个架构**可以用 FSDP**（尤其是当 `SoraDiT` 本体太大单卡放不下时），但一般建议先用 DDP：因为当前训练默认只训练 `SoraDiT`（UMT5/LLM/VAE 都是 frozen），DDP 往往足够。

使用 Accelerate 跑 FSDP 的推荐方式是用配置文件（`accelerate config` 生成），然后：

```bash
accelerate launch --config_file fsdp.yaml train/videos/train_sora_accelerate.py \
  --manifest ... --pretrained-model ... --output-dir outputs_sora ...
```

## SoraDiT 推理（从训练 checkpoint 生成视频）

训练脚本会在 `--output-dir` 下保存 `checkpoint-*/model.pt` / `last.pt`。推理时直接加载其中一个：

```bash
python3 train/videos/infer_sora.py \
  --checkpoint outputs_sora/last.pt \
  --prompt "a cat walking, cinematic" \
  --num-frames 16 --height 512 --width 512 \
  --steps 50 --guidance-scale 1.0 \
  --out out.mp4

# 也可以用“时长 + fps”来换算帧数（num_frames = ceil(seconds * fps)）
python3 train/videos/infer_sora.py \
  --checkpoint outputs_sora/last.pt \
  --prompt "a cat walking, cinematic" \
  --fps 24 --seconds 8 --height 512 --width 512 \
  --out out_8s_24fps.mp4
```

## CogVideoX LoRA 微调（Accelerate）

依赖：`diffusers + accelerate + torch`（不依赖 peft；LoRA 注入逻辑在 `train/videos/lora_utils.py`）。

```bash
accelerate launch --num_processes 8 train/videos/train_cogvideox_lora_accelerate.py \
  --manifest /path/to/manifest.jsonl --data-root /path/to/data \
  --pretrained-model /path/to/cogvideox_diffusers_pipeline \
  --output-dir outputs_cogvideox_lora \
  --height 512 --width 512 --num-frames 16 \
  --max-steps 50000 --batch-size 1 --grad-accum 4 --mixed-precision bf16
```

保存的 LoRA checkpoint：`outputs_cogvideox_lora/lora_0000500.pt`（包含 LoRA 权重与优化器状态）。

## Titans-style Memory VideoDiT（从零训练骨架）

这是一个“KV-cache-free / 固定 memory tokens”的 Video DiT scaffold，用来探索长视频 token 的可扩展训练。
当前仍使用 2D VAE（`AutoencoderKL`）做逐帧 latent 编码；后续可替换成 3D Causal VAE。

```bash
accelerate launch --num_processes 8 train/videos/train_titans_video_dit_accelerate.py \
  --manifest /path/to/manifest.jsonl --data-root /path/to/data \
  --pretrained-vae /path/to/svd_or_sd_diffusers \
  --output-dir outputs_titans_videodit \
  --resolution 512 --num-frames 16 \
  --model-dim 1024 --depth 12 --heads 16 --mem-tokens 64 --patch-size 2,4,4 \
  --max-steps 100000 --batch-size 1 --grad-accum 4 --mixed-precision bf16
```

### 16:9/9:16 分桶（768/1024）

```bash
accelerate launch --num_processes 8 train/videos/train_video_accelerate.py \
  --manifest /path/to/manifest.jsonl --data-root /path/to/data \
  --pretrained-model /path/to/svd --output-dir outputs_video \
  --buckets 576x1024,1024x576,432x768,768x432 \
  --batch-size 1 --grad-accum 4 --mixed-precision bf16
```

### manifest 任务混合（t2v/i2v 50/50）

```jsonl
{"video":"videos/0001.mp4","caption":"a cat running","task":"t2v"}
{"video":"videos/0002.mp4","caption":"a dog running","task":"i2v","cond_image":"refs/0002.png"}
```

## 长视频推理（滑窗续写）

8s@24fps=192帧，12s@24fps=288帧。建议先用更低 fps 生成再插帧到24fps；如需直接生成，可用滑窗：

```bash
python3 train/videos/video_framework.py infer \
  --pretrained-model /path/to/svd --checkpoint outputs_video/checkpoint-5000/model.pt \
  --prompt "..." --height 576 --width 1024 --fps 24 \
  --num-frames 192 --window 32 --overlap 8 --out out_8s.mp4
```

**注意事项：**

1. **依赖安装：**
```bash
pip install torch torchvision diffusers transformers accelerate imageio opencv-python einops
```

2. **数据准备：** 需要准备 `metadata.json` 格式如：
```json
[
  {
    "video_path": "videos/video1.mp4",
    "caption": "A cat playing in the garden"
  }
]
```

3. **显存优化：** 如果显存不足，可以：
    - 减小 `batch_size`
    - 减小 `resolution`
    - 减小 `num_frames`
    - 启用梯度检查点

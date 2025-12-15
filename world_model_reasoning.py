import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple

class PhysicsAwareMemory(nn.Module):
    """
    物理感知记忆模块 - 存储和检索物理规律知识
    """
    def __init__(
        self,
        dim: int,
        num_physics_rules: int = 128,  # 物理规则数量
        memory_depth: int = 3,
    ):
        super().__init__()
        self.dim = dim
        
        # 物理规则嵌入（重力、碰撞、流体等）
        self.physics_embeddings = nn.Parameter(
            torch.randn(num_physics_rules, dim) * 0.02
        )
        
        # 深度记忆网络
        self.memory_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            ) for _ in range(memory_depth)
        ])
        
        # 规则选择注意力
        self.rule_attention = nn.MultiheadAttention(
            dim, num_heads=8, batch_first=True
        )
        
    def forward(
        self, 
        current_state: torch.Tensor,  # [B, T, D]
        context: torch.Tensor,  # [B, T, D] - 历史帧特征
    ) -> torch.Tensor:
        B, T, D = current_state.shape
        
        # 检索相关物理规则
        physics_rules = repeat(
            self.physics_embeddings, 'n d -> b n d', b=B
        )
        
        # 注意力选择相关规则
        attended_rules, _ = self.rule_attention(
            current_state,  # query
            physics_rules,  # key
            physics_rules,  # value
        )
        
        # 融合当前状态、历史上下文和物理规则
        fused = current_state + attended_rules + context
        
        # 通过深度记忆网络推理
        for layer in self.memory_layers:
            fused = fused + layer(fused)
            
        return fused


class CommonSenseReasoning(nn.Module):
    """
    常识推理模块 - 理解日常场景的合理性
    """
    def __init__(
        self,
        dim: int,
        num_scenes: int = 256,  # 场景类型数量
        reasoning_depth: int = 4,
    ):
        super().__init__()
        
        # 场景原型记忆（室内、室外、交通等）
        self.scene_prototypes = nn.Parameter(
            torch.randn(num_scenes, dim) * 0.02
        )
        
        # 因果推理链
        self.causal_chain = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 2, dim),
            ) for _ in range(reasoning_depth)
        ])
        
        # 合理性评分器
        self.plausibility_scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        frame_features: torch.Tensor,  # [B, T, D]
        text_condition: torch.Tensor,  # [B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = frame_features.shape
        
        # 匹配场景原型
        text_expanded = repeat(text_condition, 'b d -> b t d', t=T)
        combined = frame_features + text_expanded
        
        # 检索相关场景知识
        scene_sim = torch.einsum(
            'btd,nd->btn', combined, self.scene_prototypes
        )
        scene_weights = F.softmax(scene_sim, dim=-1)
        scene_context = torch.einsum(
            'btn,nd->btd', scene_weights, self.scene_prototypes
        )
        
        # 因果推理
        reasoned = combined + scene_context
        for layer in self.causal_chain:
            reasoned = reasoned + layer(reasoned)
            
        # 计算合理性分数
        plausibility = self.plausibility_scorer(reasoned)
        
        return reasoned, plausibility


class TemporalConsistencyModule(nn.Module):
    """
    时间一致性模块 - 保持人物和物体的长期一致性
    """
    def __init__(
        self,
        dim: int,
        num_entities: int = 64,  # 最大跟踪实体数
        consistency_depth: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.num_entities = num_entities
        
        # 实体记忆槽
        self.entity_memory = nn.Parameter(
            torch.zeros(1, num_entities, dim)
        )
        
        # 实体追踪注意力
        self.entity_tracker = nn.MultiheadAttention(
            dim, num_heads=8, batch_first=True
        )
        
        # 一致性强化层
        self.consistency_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            ) for _ in range(consistency_depth)
        ])
        
        # 实体更新门控
        self.update_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        current_frame: torch.Tensor,  # [B, T, D]
        entity_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = current_frame.shape
        
        # 初始化实体记忆
        if entity_memory is None:
            entity_memory = repeat(
                self.entity_memory, '1 n d -> b n d', b=B
            )
        
        # 检索相关实体
        attended_entities, attn_weights = self.entity_tracker(
            current_frame,  # query
            entity_memory,  # key
            entity_memory,  # value
        )
        
        # 应用一致性约束
        consistent_frame = current_frame + attended_entities
        for layer in self.consistency_layers:
            consistent_frame = consistent_frame + layer(consistent_frame)
        
        # 更新实体记忆（加权平均）
        frame_summary = current_frame.mean(dim=1, keepdim=True)  # [B, 1, D]
        update_signal = torch.cat([frame_summary, entity_memory.mean(dim=1, keepdim=True)], dim=-1)
        gate = self.update_gate(update_signal)  # [B, 1, D]
        
        new_entity_memory = gate * frame_summary + (1 - gate) * entity_memory.mean(dim=1, keepdim=True)
        
        # 选择性更新最相关的实体槽
        top_k = min(3, self.num_entities)
        topk_indices = attn_weights.mean(dim=1).topk(top_k, dim=-1).indices  # [B, top_k]
        
        for b in range(B):
            for idx in topk_indices[b]:
                entity_memory[b, idx] = 0.7 * entity_memory[b, idx] + 0.3 * new_entity_memory[b, 0]
        
        return consistent_frame, entity_memory



class WorldModelReasoning(nn.Module):
    """
    高效版世界模型推理模块
    """
    def __init__(
            self,
            dim: int,
            num_physics_rules: int = 128,
            num_scenes: int = 256,
            num_entities: int = 64,
            memory_depth: int = 3,
            reasoning_depth: int = 4,
            consistency_depth: int = 3,
    ):
        super().__init__()

        # --- 1. 空间压缩模块 ---
        # 将 64x64 的 latent 压缩到 16x16，节省 16倍 显存
        self.spatial_reducer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=4, stride=4),
            nn.GroupNorm(32, dim),
            nn.SiLU()
        )

        # --- 2. 核心推理模块 ---
        self.physics_memory = PhysicsAwareMemory(dim, num_physics_rules, memory_depth)
        self.commonsense_reasoning = CommonSenseReasoning(dim, num_scenes, reasoning_depth)
        self.consistency_module = TemporalConsistencyModule(dim, num_entities, consistency_depth)

        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim),
            ) for _ in range(2) # 稍微减少层数以换取效率
        ])

        self.adaptive_gate = nn.Sequential(
            nn.Linear(dim * 3, 3),
            nn.Softmax(dim=-1),
        )

        # --- 3. 空间恢复模块  ---
        self.spatial_expander = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=4),
            nn.GroupNorm(32, dim)
        )

    def forward(
            self,
            current_latents: torch.Tensor,  # [B, T, C, H, W]
            text_embeds: torch.Tensor,      # [B, D]
            image_embeds: Optional[torch.Tensor] = None,
            history_latents: Optional[torch.Tensor] = None,
            entity_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        B, T, C, H, W = current_latents.shape

        # --- 步骤 1: 空间压缩 ---
        # 合并 Batch 和 Time 以便进行 2D 卷积
        x_flat = rearrange(current_latents, 'b t c h w -> (b t) c h w')
        x_reduced = self.spatial_reducer(x_flat) # [(B T), C, H/4, W/4]

        # 展平用于 Transformer 风格的推理模块
        # 现在序列长度减少了 16 倍！
        current_flat = rearrange(x_reduced, 'bt c h w -> bt (h w) c')
        # 恢复 Batch 维度用于推理: [B, (T * h_small * w_small), D]
        current_flat = rearrange(current_flat, '(b t) s c -> b (t s) c', b=B)

        # --- 步骤 2: 准备上下文 ---
        # 注意：History Latents 也需要经过同样的压缩，为了简单起见，这里假设 history 已经被压缩
        # 或者我们只用 Global Average Pooling 的 context
        if history_latents is not None:
            # 简化处理：对历史帧做全局平均，作为 context
            context = history_latents.mean(dim=[2, 3, 4]).unsqueeze(1).expand(B, current_flat.shape[1], -1)
        else:
            context = torch.zeros_like(current_flat)

        # --- 步骤 3: 核心推理---

        # 3.1 物理推理
        physics_reasoned = self.physics_memory(current_flat, context)

        # 3.2 常识推理
        condition = text_embeds if image_embeds is None else (text_embeds + image_embeds) / 2
        commonsense_reasoned, plausibility = self.commonsense_reasoning(
            current_flat, condition
        )

        # 3.3 一致性推理
        consistent_reasoned, new_entity_memory = self.consistency_module(
            current_flat, entity_memory
        )

        # --- 步骤 4: 融合 ---
        combined = torch.stack([physics_reasoned, commonsense_reasoned, consistent_reasoned], dim=-1)

        # 计算门控
        gate_input = torch.cat([
            physics_reasoned.mean(dim=1),
            commonsense_reasoned.mean(dim=1),
            consistent_reasoned.mean(dim=1),
        ], dim=-1)

        gates = self.adaptive_gate(gate_input)
        gates = rearrange(gates, 'b g -> b 1 1 g')

        fused = (combined * gates).sum(dim=-1)

        for layer in self.fusion_layers:
            fused = fused + layer(fused)

        # --- 步骤 5: 空间恢复 ---
        # [B, (T * h * w), C] -> [(B T), C, h, w]
        h_small, w_small = H // 4, W // 4
        fused_img = rearrange(fused, 'b (t h w) c -> (b t) c h w', t=T, h=h_small, w=w_small)

        # 上采样回原始尺寸
        reasoned_latents = self.spatial_expander(fused_img)

        # 恢复原始维度结构
        reasoned_latents = rearrange(reasoned_latents, '(b t) c h w -> b t c h w', b=B)

        # 此时 reasoned_latents 的形状应该与输入 current_latents 完全一致

        aux_outputs = {
            'plausibility': plausibility,
            'entity_memory': new_entity_memory,
            'fusion_gates': gates,
        }

        return reasoned_latents, aux_outputs
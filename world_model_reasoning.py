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
    世界模型推理模块 - 整合所有推理能力
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
        
        # 三大推理模块
        self.physics_memory = PhysicsAwareMemory(
            dim, num_physics_rules, memory_depth
        )
        
        self.commonsense_reasoning = CommonSenseReasoning(
            dim, num_scenes, reasoning_depth
        )
        
        self.consistency_module = TemporalConsistencyModule(
            dim, num_entities, consistency_depth
        )
        
        # 多模态融合
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim),
            ) for _ in range(3)
        ])
        
        # 自适应门控
        self.adaptive_gate = nn.Sequential(
            nn.Linear(dim * 3, 3),
            nn.Softmax(dim=-1),
        )
        
    def forward(
        self,
        current_latents: torch.Tensor,  # [B, T, H, W, D] - 当前帧潜在表示
        text_embeds: torch.Tensor,  # [B, D] - 文本条件
        image_embeds: Optional[torch.Tensor] = None,  # [B, D] - 图像条件
        history_latents: Optional[torch.Tensor] = None,  # [B, T_past, H, W, D]
        entity_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        
        B, T, H, W, D = current_latents.shape
        
        # 展平空间维度用于推理
        current_flat = rearrange(current_latents, 'b t h w d -> b (t h w) d')
        
        # 准备历史上下文
        if history_latents is not None:
            history_flat = rearrange(history_latents, 'b t h w d -> b (t h w) d')
            context = history_flat.mean(dim=1, keepdim=True).expand_as(current_flat)
        else:
            context = torch.zeros_like(current_flat)
        
        # 1. 物理推理
        physics_reasoned = self.physics_memory(current_flat, context)
        
        # 2. 常识推理
        condition = text_embeds if image_embeds is None else (text_embeds + image_embeds) / 2
        commonsense_reasoned, plausibility = self.commonsense_reasoning(
            current_flat, condition
        )
        
        # 3. 一致性推理
        consistent_reasoned, new_entity_memory = self.consistency_module(
            current_flat, entity_memory
        )
        
        # 自适应融合三种推理结果
        combined = torch.stack([
            physics_reasoned,
            commonsense_reasoned,
            consistent_reasoned
        ], dim=-1)  # [B, T*H*W, D, 3]
        
        gate_input = torch.cat([
            physics_reasoned.mean(dim=1),
            commonsense_reasoned.mean(dim=1),
            consistent_reasoned.mean(dim=1),
        ], dim=-1)  # [B, D*3]
        
        gates = self.adaptive_gate(gate_input)  # [B, 3]
        gates = rearrange(gates, 'b g -> b 1 1 g')
        
        fused = (combined * gates).sum(dim=-1)  # [B, T*H*W, D]
        
        # 深度融合
        for layer in self.fusion_layers:
            fused = fused + layer(fused)
        
        # 恢复空间维度
        reasoned_latents = rearrange(
            fused, 'b (t h w) d -> b t h w d', t=T, h=H, w=W
        )
        
        aux_outputs = {
            'plausibility': plausibility,
            'entity_memory': new_entity_memory,
            'fusion_gates': gates,
        }
        
        return reasoned_latents, aux_outputs

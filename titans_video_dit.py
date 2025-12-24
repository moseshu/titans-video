"""
Titans-style "Neural Memory" Video DiT backbone for latent diffusion.

Key idea:
- Use "Neural Memory" (Test-Time Training) for global context communication.
- Each layer has a Neural Memory module (weights updated via gradients on the sequence).
- Preserve local detail with lightweight depthwise 3D convolution mixer.

This is a research scaffold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore

from neural_memory import NeuralMemory




def _require(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(msg)




def _sincos_1d(pos, dim: int):
    
    _require(torch is not None, "Need torch installed.")
    assert dim % 2 == 0
    omega = torch.arange(dim // 2, device=pos.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 2)))
    out = pos.float().unsqueeze(-1) * omega.unsqueeze(0)  # [N, dim/2]
    return torch.cat([out.sin(), out.cos()], dim=-1)  # [N, dim]


def sincos_3d_pos_embed(dim: int, t: int, h: int, w: int, device):
    """
    Returns [1, t*h*w, dim] positional embedding.
    """
    
    _require(torch is not None, "Need torch installed.")
    # allocate per-axis dims (even), then pad to `dim`
    per = dim // 3
    dt = per - (per % 2)
    dh = per - (per % 2)
    dw = dim - dt - dh
    dw = dw - (dw % 2)
    used = dt + dh + dw
    _require(dt > 0 and dh > 0 and dw > 0, "model_dim too small for 3D sincos positional embedding.")

    tt = torch.arange(t, device=device)
    hh = torch.arange(h, device=device)
    ww = torch.arange(w, device=device)

    grid_t, grid_h, grid_w = torch.meshgrid(tt, hh, ww, indexing="ij")
    pos_t = grid_t.reshape(-1)
    pos_h = grid_h.reshape(-1)
    pos_w = grid_w.reshape(-1)

    emb_t = _sincos_1d(pos_t, dt)
    emb_h = _sincos_1d(pos_h, dh)
    emb_w = _sincos_1d(pos_w, dw)
    emb = torch.cat([emb_t, emb_h, emb_w], dim=-1)  # [N, dim]
    if used < dim:
        emb = torch.nn.functional.pad(emb, (0, dim - used))
    return emb.unsqueeze(0)  # [1, N, dim]


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        
        _require(torch is not None and nn is not None, "Need torch installed.")
        super().__init__()
        self.dim = int(dim)
        hidden_dim = int(hidden_dim or dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward(self, timesteps):
        
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * (float(__import__("math").log(10000.0)) / half)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class AdaLNZero(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        
        _require(torch is not None and nn is not None, "Need torch installed.")
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_mod = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.to_mod.weight)
        nn.init.zeros_(self.to_mod.bias)

    def forward(self, x, cond):
        
        m = self.to_mod(cond).unsqueeze(1)  # [B,1,6D]
        shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff = m.chunk(6, dim=-1)
        x_msa = self.norm(x) * (1.0 + scale_msa) + shift_msa
        x_ff = self.norm(x) * (1.0 + scale_ff) + shift_ff
        return x_msa, gate_msa, x_ff, gate_ff


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        
        _require(torch is not None and nn is not None, "Need torch installed.")
        super().__init__()
        inner = int(dim) * int(mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class TitansBlock(nn.Module):
    """
    Titans Architecture Block (Bidirectional):
    1. Local Branch: Short-term memory (3D Conv + GELU)
    2. Global Branch: Bidirectional Neural Memory (Fwd + Bwd)
    3. Gated Fusion: Dynamic weighting between Local and Global
    """

    def __init__(
        self,
        dim: int,
        time_dim: int,
        *,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        chunk_size: int = 128,
        use_local_3d_mixer: bool = True,
        use_per_head_memory: bool = False,
    ):
        
        _require(torch is not None and nn is not None, "Need torch installed.")
        _require(NeuralMemory is not None, "NeuralMemory module not found.")
        super().__init__()

        self.adaln = AdaLNZero(dim, time_dim)

        # 1. Local Branch (Short-term)
        self.use_local_3d_mixer = bool(use_local_3d_mixer)
        if self.use_local_3d_mixer:
            self.local = nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim),
                nn.GELU()
            )
        
        # 2. Global Branch (Bidirectional Neural Memory)
        # Forward Memory
        self.memory_fwd = NeuralMemory(
            dim=dim,
            chunk_size=chunk_size,
            heads=heads,
            dim_head=dim_head,
            per_head_learned_parameters=use_per_head_memory,
        )
        # Backward Memory
        self.memory_bwd = NeuralMemory(
            dim=dim,
            chunk_size=chunk_size,
            heads=heads,
            dim_head=dim_head,
            per_head_learned_parameters=use_per_head_memory,
        )

        # 3. Fusion Gate (Sigmoid-Gated)
        self.gate_proj = nn.Linear(dim * 2, dim) # Projects concat([local, global]) -> Gate weights
        self.mix_proj = nn.Linear(dim * 2, dim)  # Projects concat([local, global]) -> Mixed Features

        # Feed Forward
        self.ff = FeedForward(dim, mult=ff_mult)
        
        # Condition (Control) Injection
        self.cond_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_attn = nn.MultiheadAttention(dim, num_heads=heads, batch_first=True)
        self.cond_gate = nn.Linear(time_dim, dim)
        nn.init.zeros_(self.cond_gate.weight)
        nn.init.zeros_(self.cond_gate.bias)


    def forward(
        self,
        tokens,  # [B,N,D]
        t_embed,  # [B,time_dim]
        cond: Optional["torch.Tensor"],  # [B,S,D] or None
        grid_shape: Tuple[int, int, int],  # (T',H',W')
    ):
        
        
        # Apply AdaLN
        x_mixer, gate_mixer, x_ff, gate_ff = self.adaln(tokens, t_embed)

        # 1. Local Branch
        local_out = x_mixer
        if self.use_local_3d_mixer:
            tt, hh, ww = grid_shape
            # Reshape to [B, D, T, H, W] for Conv3d
            x_local = local_out.transpose(1, 2).reshape(local_out.shape[0], local_out.shape[2], tt, hh, ww)
            x_local = self.local(x_local)
            local_out = local_out + x_local.reshape(local_out.shape[0], local_out.shape[2], -1).transpose(1, 2)

        # 2. Global Branch (Bidirectional)
        # Forward pass
        feat_fwd, _ = self.memory_fwd(x_mixer)
        
        # Backward pass (flip, process, flip back)
        x_reversed = torch.flip(x_mixer, dims=[1])
        feat_bwd_rev, _ = self.memory_bwd(x_reversed)
        feat_bwd = torch.flip(feat_bwd_rev, dims=[1])
        
        # Combine Forward and Backward (Simple Sum or Average or Concat -> Proj)
        # Here we choose Sum to keep dim same, or we could concat.
        # Let's sum them for parameter efficiency before fusion, preserving 'global_out' semantics
        global_out = feat_fwd + feat_bwd 
        
        # Add residual from input to global branch?
        # Standard Titans adds input residual. 
        global_out = global_out + x_mixer

        # 3. Fusion (Gated)
        concat = torch.cat([local_out, global_out], dim=-1) # [B, N, 2D]
        gate = torch.sigmoid(self.gate_proj(concat)) # [B, N, D]
        mixed = self.mix_proj(concat) * gate         # [B, N, D]
        
        # Residual Connection
        tokens = tokens + gate_mixer * mixed

        # External Condition (Cross Attention)
        if cond is not None:
            x_cond = self.cond_norm(tokens)
            attn_out, _ = self.cond_attn(x_cond, cond, cond, need_weights=False)
            tokens = tokens + self.cond_gate(t_embed).unsqueeze(1) * attn_out

        # Feed Forward
        tokens = tokens + gate_ff * self.ff(x_ff)

        return tokens


@dataclass
class TitansVideoDiTConfig:
    in_channels: int = 4
    model_dim: int = 1024
    depth: int = 12
    heads: int = 16
    dim_head: int = 64
    ff_mult: int = 4
    patch_size: Tuple[int, int, int] = (2, 4, 4)  # (pt, ph, pw)
    chunk_size: int = 128 # Chunk size for Titans TTT
    use_local_3d_mixer: bool = True
    use_per_head_memory: bool = False


class TitansVideoDiT(nn.Module):
    """
    Diffusion Transformer-style model operating on VAE latents.
    Uses Titans (Neural Memory) blocks.

    Input/Output: latents [B,C,T,H,W]
    """

    def __init__(self, cfg: TitansVideoDiTConfig):
        
        _require(torch is not None and nn is not None, "Need torch installed.")
        super().__init__()
        self.cfg = cfg

        pt, ph, pw = cfg.patch_size
        self.patch = nn.Conv3d(cfg.in_channels, cfg.model_dim, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))
        self.unpatch = nn.ConvTranspose3d(cfg.model_dim, cfg.in_channels, kernel_size=(pt, ph, pw), stride=(pt, ph, pw))

        self.time = TimestepEmbedder(cfg.model_dim)

        self.blocks = nn.ModuleList(
            [
                TitansBlock(
                    cfg.model_dim,
                    cfg.model_dim,
                    heads=cfg.heads,
                    dim_head=cfg.dim_head,
                    ff_mult=cfg.ff_mult,
                    chunk_size=cfg.chunk_size,
                    use_local_3d_mixer=cfg.use_local_3d_mixer,
                    use_per_head_memory=cfg.use_per_head_memory,
                )
                for _ in range(cfg.depth)
            ]
        )

        self.out_norm = nn.LayerNorm(cfg.model_dim, elementwise_affine=False)
        self.out = nn.Linear(cfg.model_dim, cfg.model_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        noisy_latents_bcthw,
        timesteps,
        *,
        encoder_hidden_states=None,
    ):
        
        _require(torch is not None, "Need torch installed.")
        _require(noisy_latents_bcthw.ndim == 5, "Expected latents [B,C,T,H,W]")
        b = noisy_latents_bcthw.shape[0]
        pt, ph, pw = self.cfg.patch_size
        _require(noisy_latents_bcthw.shape[2] % pt == 0, "Latent T must be divisible by patch_size[0]")
        _require(noisy_latents_bcthw.shape[3] % ph == 0, "Latent H must be divisible by patch_size[1]")
        _require(noisy_latents_bcthw.shape[4] % pw == 0, "Latent W must be divisible by patch_size[2]")

        x = self.patch(noisy_latents_bcthw)  # [B,D,T',H',W']
        _, d, tt, hh, ww = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B,N,D]

        tokens = tokens + sincos_3d_pos_embed(d, tt, hh, ww, device=tokens.device).to(tokens.dtype)

        t_embed = self.time(timesteps).to(dtype=tokens.dtype)

        # Condition
        cond = encoder_hidden_states
        if cond is not None:
            cond = cond.to(device=tokens.device, dtype=tokens.dtype)

        # Pass through Titans Blocks
        for blk in self.blocks:
            tokens = blk(tokens, t_embed=t_embed, cond=cond, grid_shape=(tt, hh, ww))

        tokens = self.out(self.out_norm(tokens))
        x = tokens.transpose(1, 2).reshape(b, d, tt, hh, ww)
        x = self.unpatch(x)
        return x

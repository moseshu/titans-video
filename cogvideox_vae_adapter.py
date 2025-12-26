from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CogVideoXVAEWrapper(nn.Module):
    """
    Adapter for Diffusers CogVideoX VAE to the framework's ExternalVideoVAE interface.

    Required by ExternalVideoVAE:
      - encode(video_btchw: [B,T,3,H,W]) -> latents_bcthw
      - decode(latents_bcthw: [B,C,T',h,w]) -> video_btchw
    """

    def __init__(
        self,
        pretrained_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        try:
            from diffusers import AutoencoderKLCogVideoX  # type: ignore
        except Exception as e:
            raise RuntimeError("Need diffusers with AutoencoderKLCogVideoX to use CogVideoXVAEWrapper.") from e

        try:
            self.vae = AutoencoderKLCogVideoX.from_pretrained(str(pretrained_path), subfolder="vae", dtype=dtype)
        except TypeError:
            self.vae = AutoencoderKLCogVideoX.from_pretrained(str(pretrained_path), subfolder="vae", torch_dtype=dtype)
        self.scale_factor = float(getattr(getattr(self.vae, "config", object()), "scaling_factor", 0.18215))
        self.latent_channels = int(getattr(getattr(self.vae, "config", object()), "latent_channels", 16))
        self.dtype = dtype
        if device is not None:
            self.vae.to(device)
        self.vae.requires_grad_(False)

    @torch.no_grad()
    def encode(self, video_btchw: torch.Tensor) -> torch.Tensor:
        x = video_btchw.permute(0, 2, 1, 3, 4).to(dtype=self.dtype)  # [B,3,T,H,W]
        dist = self.vae.encode(x).latent_dist
        latents = dist.sample() * self.scale_factor
        return latents

    @torch.no_grad()
    def decode(self, latents_bcthw: torch.Tensor) -> torch.Tensor:
        x = (latents_bcthw / self.scale_factor).to(dtype=self.dtype)
        video = self.vae.decode(x).sample  # [B,3,T,H,W]
        return video.permute(0, 2, 1, 3, 4).float()  # [B,T,3,H,W]

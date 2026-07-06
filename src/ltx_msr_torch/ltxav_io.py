from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint_loader import load_safetensors_subset
from .ltx_patchify import (
    AudioPatchifyResult,
    VideoPatchifyResult,
    latent_to_pixel_coords,
    patchify_audio,
    symmetric_patchify_video,
)


@dataclass(frozen=True)
class LTXAVProjectedInputs:
    video_tokens: torch.Tensor
    audio_tokens: torch.Tensor
    video_latent_coords: torch.Tensor
    video_pixel_coords: torch.Tensor
    audio_latent_coords: torch.Tensor
    video_patches: torch.Tensor
    audio_patches: torch.Tensor


class LTXAVInputProjection(torch.nn.Module):
    def __init__(
        self,
        *,
        video_in_channels: int = 128,
        video_hidden_dim: int = 4096,
        audio_in_channels: int = 128,
        audio_hidden_dim: int = 2048,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.patchify_proj = torch.nn.Linear(video_in_channels, video_hidden_dim, bias=True, dtype=dtype, device=device)
        self.audio_patchify_proj = torch.nn.Linear(audio_in_channels, audio_hidden_dim, bias=True, dtype=dtype, device=device)

    def forward(
        self,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        *,
        vae_scale_factors: tuple[int, int, int] = (8, 32, 32),
        causal_temporal_positioning: bool = True,
    ) -> LTXAVProjectedInputs:
        video: VideoPatchifyResult = symmetric_patchify_video(video_latents, patch_size=1, start_end=True)
        audio: AudioPatchifyResult = patchify_audio(audio_latents, start_end=True)
        video_pixel_coords = latent_to_pixel_coords(
            video.latent_coords,
            vae_scale_factors,
            causal_fix=causal_temporal_positioning,
        )
        return LTXAVProjectedInputs(
            video_tokens=self.patchify_proj(video.patches),
            audio_tokens=self.audio_patchify_proj(audio.patches),
            video_latent_coords=video.latent_coords,
            video_pixel_coords=video_pixel_coords,
            audio_latent_coords=audio.timings,
            video_patches=video.patches,
            audio_patches=audio.patches,
        )


class LTXAVOutputProjection(torch.nn.Module):
    def __init__(
        self,
        *,
        video_hidden_dim: int = 4096,
        video_out_channels: int = 128,
        audio_hidden_dim: int = 2048,
        audio_out_channels: int = 128,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.proj_out = torch.nn.Linear(video_hidden_dim, video_out_channels, bias=True, dtype=dtype, device=device)
        self.audio_proj_out = torch.nn.Linear(audio_hidden_dim, audio_out_channels, bias=True, dtype=dtype, device=device)

    def forward(self, video_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.proj_out(video_tokens), self.audio_proj_out(audio_tokens)


def load_ltxav_input_projection_state_dict(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    prefix = "model.diffusion_model."
    keys = (
        f"{prefix}patchify_proj.bias",
        f"{prefix}patchify_proj.weight",
        f"{prefix}audio_patchify_proj.bias",
        f"{prefix}audio_patchify_proj.weight",
    )
    raw = load_safetensors_subset(checkpoint_path, keys, device=device)
    return {key[len(prefix) :]: value for key, value in raw.items()}


def load_ltxav_output_projection_state_dict(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    prefix = "model.diffusion_model."
    keys = (
        f"{prefix}proj_out.bias",
        f"{prefix}proj_out.weight",
        f"{prefix}audio_proj_out.bias",
        f"{prefix}audio_proj_out.weight",
    )
    raw = load_safetensors_subset(checkpoint_path, keys, device=device)
    return {key[len(prefix) :]: value for key, value in raw.items()}

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
    grid_mask: torch.Tensor | None = None
    orig_patchified_shape: tuple[int, ...] | None = None
    num_guide_tokens: int = 0
    resolved_guide_entries: tuple[dict[str, object], ...] = ()


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
        keyframe_idxs: torch.Tensor | None = None,
        denoise_mask: torch.Tensor | None = None,
        guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
        vae_scale_factors: tuple[int, int, int] = (8, 32, 32),
        causal_temporal_positioning: bool = False,
    ) -> LTXAVProjectedInputs:
        video: VideoPatchifyResult = symmetric_patchify_video(video_latents, patch_size=1, start_end=True)
        audio: AudioPatchifyResult = patchify_audio(audio_latents, start_end=True)
        video_pixel_coords = latent_to_pixel_coords(
            video.latent_coords,
            vae_scale_factors,
            causal_fix=causal_temporal_positioning,
        )
        video_patches = video.patches
        video_latent_coords = video.latent_coords
        orig_patchified_shape = None
        grid_mask = None
        resolved_entries: tuple[dict[str, object], ...] = ()
        num_guide_tokens = 0
        if keyframe_idxs is not None and keyframe_idxs.shape[2] > 0:
            if denoise_mask is None:
                raise ValueError("denoise_mask is required when keyframe indices are provided")
            orig_patchified_shape = tuple(video_patches.shape)
            expanded_mask = _expand_denoise_mask(denoise_mask, video_latents)
            denoise_patches = symmetric_patchify_video(expanded_mask, patch_size=1, start_end=True).patches
            grid_mask = ~torch.any(denoise_patches < 0, dim=-1)[0]
            video_patches = video_patches[:, grid_mask, :]
            video_latent_coords = video_latent_coords[:, :, grid_mask, :]
            video_pixel_coords = video_pixel_coords[:, :, grid_mask, :]
            kf_grid_mask = grid_mask[-keyframe_idxs.shape[2] :]
            if guide_attention_entries:
                resolved_entries = _resolve_guide_attention_entries(guide_attention_entries, kf_grid_mask)
            keyframe_idxs = keyframe_idxs[..., kf_grid_mask, :]
            if keyframe_idxs.shape[2] > 0:
                video_pixel_coords[:, :, -keyframe_idxs.shape[2] :, :] = keyframe_idxs
            num_guide_tokens = int(keyframe_idxs.shape[2])
        return LTXAVProjectedInputs(
            video_tokens=self.patchify_proj(video_patches),
            audio_tokens=self.audio_patchify_proj(audio.patches),
            video_latent_coords=video_latent_coords,
            video_pixel_coords=video_pixel_coords,
            audio_latent_coords=audio.timings,
            video_patches=video_patches,
            audio_patches=audio.patches,
            grid_mask=grid_mask,
            orig_patchified_shape=orig_patchified_shape,
            num_guide_tokens=num_guide_tokens,
            resolved_guide_entries=resolved_entries,
        )


def _expand_denoise_mask(denoise_mask: torch.Tensor, video_latents: torch.Tensor) -> torch.Tensor:
    if denoise_mask.shape[3] == video_latents.shape[3] and denoise_mask.shape[4] == video_latents.shape[4]:
        return denoise_mask
    return denoise_mask.expand(-1, -1, -1, video_latents.shape[3], video_latents.shape[4])


def _resolve_guide_attention_entries(
    entries: tuple[dict[str, object], ...] | list[dict[str, object]],
    kf_grid_mask: torch.Tensor,
) -> tuple[dict[str, object], ...]:
    total = sum(int(entry["pre_filter_count"]) for entry in entries)
    if total != len(kf_grid_mask):
        raise ValueError(f"guide pre_filter_counts ({total}) != keyframe grid mask length ({len(kf_grid_mask)})")
    resolved: list[dict[str, object]] = []
    offset = 0
    for entry in entries:
        count = int(entry["pre_filter_count"])
        entry_mask = kf_grid_mask[offset : offset + count]
        resolved.append({**entry, "surviving_count": int(entry_mask.sum().item())})
        offset += count
    return tuple(resolved)


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

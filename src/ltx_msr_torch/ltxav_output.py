from __future__ import annotations

import math
from pathlib import Path

import torch

from .checkpoint_loader import load_safetensors_subset
from .ltx_patchify import symmetric_unpatchify_video, unpatchify_audio
from .ltx_timestep import CompressedTimestep


class LTXAVOutputProcessor(torch.nn.Module):
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
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, video_hidden_dim, dtype=dtype, device=device))
        self.norm_out = torch.nn.LayerNorm(
            video_hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
            dtype=dtype,
            device=device,
        )
        self.proj_out = torch.nn.Linear(video_hidden_dim, video_out_channels, dtype=dtype, device=device)
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio_hidden_dim, dtype=dtype, device=device))
        self.audio_norm_out = torch.nn.LayerNorm(
            audio_hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
            dtype=dtype,
            device=device,
        )
        self.audio_proj_out = torch.nn.Linear(audio_hidden_dim, audio_out_channels, dtype=dtype, device=device)

    def _process_video(
        self,
        video_tokens: torch.Tensor,
        embedded_timestep: torch.Tensor | CompressedTimestep,
        *,
        orig_shape: tuple[int, ...] | list[int],
        keyframe_idxs: torch.Tensor | None = None,
        grid_mask: torch.Tensor | None = None,
        orig_patchified_shape: tuple[int, ...] | list[int] | None = None,
        patch_size: int = 1,
    ) -> torch.Tensor:
        if isinstance(embedded_timestep, CompressedTimestep):
            embedded_timestep = embedded_timestep.expand()
        scale_shift = self.scale_shift_table[None, None].to(
            device=embedded_timestep.device,
            dtype=embedded_timestep.dtype,
        ) + embedded_timestep[:, :, None]
        shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
        video_tokens = self.norm_out(video_tokens)
        video_tokens = video_tokens * (1 + scale) + shift
        video_tokens = self.proj_out(video_tokens)

        if keyframe_idxs is not None and keyframe_idxs.shape[2] > 0:
            if grid_mask is None or orig_patchified_shape is None:
                raise ValueError("grid_mask and orig_patchified_shape are required when keyframe indices are provided")
            full_tokens = torch.zeros(orig_patchified_shape, dtype=video_tokens.dtype, device=video_tokens.device)
            full_tokens[:, grid_mask, :] = video_tokens
            video_tokens = full_tokens

        return symmetric_unpatchify_video(
            video_tokens,
            output_height=int(orig_shape[3]),
            output_width=int(orig_shape[4]),
            output_num_frames=int(orig_shape[2]),
            out_channels=int(orig_shape[1]) // math.prod((1, patch_size, patch_size)),
            patch_size=patch_size,
        )

    def _process_audio(
        self,
        audio_tokens: torch.Tensor,
        embedded_timestep: torch.Tensor,
        *,
        ref_audio_seq_len: int = 0,
        channels: int = 8,
        frequency: int = 16,
    ) -> torch.Tensor:
        if ref_audio_seq_len > 0:
            audio_tokens = audio_tokens[:, ref_audio_seq_len:]
            if embedded_timestep.shape[1] > 1:
                embedded_timestep = embedded_timestep[:, ref_audio_seq_len:]
        scale_shift = self.audio_scale_shift_table[None, None].to(
            device=embedded_timestep.device,
            dtype=embedded_timestep.dtype,
        ) + embedded_timestep[:, :, None]
        shift, scale = scale_shift[:, :, 0], scale_shift[:, :, 1]
        audio_tokens = self.audio_norm_out(audio_tokens)
        audio_tokens = audio_tokens * (1 + scale) + shift
        audio_tokens = self.audio_proj_out(audio_tokens)
        return unpatchify_audio(audio_tokens, channels=channels, frequency=frequency)

    def forward(
        self,
        video_tokens: torch.Tensor,
        audio_tokens: torch.Tensor,
        *,
        video_embedded_timestep: torch.Tensor | CompressedTimestep,
        audio_embedded_timestep: torch.Tensor | None,
        orig_shape: tuple[int, ...] | list[int],
        keyframe_idxs: torch.Tensor | None = None,
        grid_mask: torch.Tensor | None = None,
        orig_patchified_shape: tuple[int, ...] | list[int] | None = None,
        ref_audio_seq_len: int = 0,
        audio_channels: int = 8,
        audio_frequency: int = 16,
        patch_size: int = 1,
    ) -> torch.Tensor | list[torch.Tensor]:
        video = self._process_video(
            video_tokens,
            video_embedded_timestep,
            orig_shape=orig_shape,
            keyframe_idxs=keyframe_idxs,
            grid_mask=grid_mask,
            orig_patchified_shape=orig_patchified_shape,
            patch_size=patch_size,
        )
        if audio_tokens.numel() == 0:
            return video
        if audio_embedded_timestep is None:
            raise ValueError("audio_embedded_timestep is required when audio tokens are present")
        audio = self._process_audio(
            audio_tokens,
            audio_embedded_timestep,
            ref_audio_seq_len=ref_audio_seq_len,
            channels=audio_channels,
            frequency=audio_frequency,
        )
        return [video, audio]


def load_ltxav_output_processor_state_dict(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    prefix = "model.diffusion_model."
    keys = (
        f"{prefix}scale_shift_table",
        f"{prefix}proj_out.bias",
        f"{prefix}proj_out.weight",
        f"{prefix}audio_scale_shift_table",
        f"{prefix}audio_proj_out.bias",
        f"{prefix}audio_proj_out.weight",
    )
    raw = load_safetensors_subset(checkpoint_path, keys, device=device)
    return {key[len(prefix) :]: value for key, value in raw.items()}

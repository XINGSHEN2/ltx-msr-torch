from __future__ import annotations

from dataclasses import dataclass

import torch

from .ltx_timestep import AdaLayerNormSingle, CompressedTimestep, compute_prompt_timestep


@dataclass(frozen=True)
class LTXAVPreparedTimesteps:
    video_timestep: torch.Tensor | CompressedTimestep
    audio_timestep: torch.Tensor | None
    video_cross_scale_shift_timestep: torch.Tensor | CompressedTimestep | None
    audio_cross_scale_shift_timestep: torch.Tensor | None
    video_cross_gate_timestep: torch.Tensor | CompressedTimestep | None
    audio_cross_gate_timestep: torch.Tensor | None
    video_prompt_timestep: torch.Tensor | None
    audio_prompt_timestep: torch.Tensor | None
    video_embedded_timestep: torch.Tensor | CompressedTimestep
    audio_embedded_timestep: torch.Tensor | None


def _video_patches_per_frame(
    *,
    orig_shape: tuple[int, ...] | list[int] | None,
    has_spatial_mask: bool | None,
) -> int | None:
    if not has_spatial_mask and orig_shape is not None and len(orig_shape) == 5:
        return int(orig_shape[3]) * int(orig_shape[4])
    return None


def prepare_ltxav_timesteps(
    *,
    timestep: torch.Tensor,
    batch_size: int,
    hidden_dtype: torch.dtype,
    video_adaln_single: AdaLayerNormSingle,
    audio_adaln_single: AdaLayerNormSingle,
    av_ca_video_scale_shift_adaln_single: AdaLayerNormSingle,
    av_ca_a2v_gate_adaln_single: AdaLayerNormSingle,
    av_ca_audio_scale_shift_adaln_single: AdaLayerNormSingle,
    av_ca_v2a_gate_adaln_single: AdaLayerNormSingle,
    video_prompt_adaln_single: AdaLayerNormSingle | None = None,
    audio_prompt_adaln_single: AdaLayerNormSingle | None = None,
    audio_timestep: torch.Tensor | None = None,
    grid_mask: torch.Tensor | None = None,
    orig_shape: tuple[int, ...] | list[int] | None = None,
    has_spatial_mask: bool | None = None,
    ref_audio_seq_len: int = 0,
    target_audio_seq_len: int | None = None,
    embedded_timestep: torch.Tensor | None = None,
    timestep_scale_multiplier: float = 1000.0,
    av_ca_timestep_scale_multiplier: float = 1.0,
) -> LTXAVPreparedTimesteps:
    patches_per_frame = _video_patches_per_frame(orig_shape=orig_shape, has_spatial_mask=has_spatial_mask)
    selected_timestep = timestep[:, grid_mask] if grid_mask is not None else timestep
    timestep_scaled = selected_timestep * timestep_scale_multiplier

    per_frame_path = (
        patches_per_frame is not None
        and (timestep.numel() // batch_size) % patches_per_frame == 0
    )
    if per_frame_path:
        per_frame = timestep.reshape(batch_size, -1, patches_per_frame)[:, :, 0]
        if grid_mask is not None:
            per_frame = per_frame[:, grid_mask[::patches_per_frame]]
        video_ts_input = per_frame * timestep_scale_multiplier
    else:
        video_ts_input = timestep_scaled

    video_timestep, video_embedded = video_adaln_single(
        video_ts_input.flatten(),
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )
    video_timestep = CompressedTimestep(
        video_timestep.view(batch_size, -1, video_timestep.shape[-1]),
        patches_per_frame,
        per_frame=per_frame_path,
    )
    video_embedded = CompressedTimestep(
        video_embedded.view(batch_size, -1, video_embedded.shape[-1]),
        patches_per_frame,
        per_frame=per_frame_path,
    )
    video_prompt = compute_prompt_timestep(
        video_prompt_adaln_single,
        timestep_scaled,
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )

    if ref_audio_seq_len > 0 and audio_timestep is not None:
        if target_audio_seq_len is None:
            raise ValueError("target_audio_seq_len is required when ref_audio_seq_len > 0")
        if audio_timestep.dim() <= 1:
            audio_timestep = audio_timestep.view(-1, 1).expand(batch_size, target_audio_seq_len)
        ref_ts = torch.zeros(
            batch_size,
            ref_audio_seq_len,
            *audio_timestep.shape[2:],
            device=audio_timestep.device,
            dtype=audio_timestep.dtype,
        )
        audio_timestep = torch.cat([ref_ts, audio_timestep], dim=1)

    if audio_timestep is None:
        return LTXAVPreparedTimesteps(
            video_timestep=video_timestep,
            audio_timestep=timestep_scaled,
            video_cross_scale_shift_timestep=None,
            audio_cross_scale_shift_timestep=None,
            video_cross_gate_timestep=None,
            audio_cross_gate_timestep=None,
            video_prompt_timestep=video_prompt,
            audio_prompt_timestep=None,
            video_embedded_timestep=video_embedded,
            audio_embedded_timestep=embedded_timestep,
        )

    audio_timestep_scaled = audio_timestep * timestep_scale_multiplier
    audio_timestep_flat = audio_timestep_scaled.flatten()
    video_timestep_flat = timestep_scaled.flatten()
    av_ca_factor = av_ca_timestep_scale_multiplier / timestep_scale_multiplier

    audio_cross_scale_shift, _ = av_ca_audio_scale_shift_adaln_single(
        audio_timestep_flat,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )
    video_cross_scale_shift, _ = av_ca_video_scale_shift_adaln_single(
        video_timestep_flat,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )
    video_cross_gate, _ = av_ca_a2v_gate_adaln_single(
        audio_timestep_scaled.max().expand_as(video_timestep_flat) * av_ca_factor,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )
    audio_cross_gate, _ = av_ca_v2a_gate_adaln_single(
        timestep_scaled.max().expand_as(audio_timestep_flat) * av_ca_factor,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )

    audio_prepared, audio_embedded = audio_adaln_single(
        audio_timestep_flat,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )
    audio_prompt = compute_prompt_timestep(
        audio_prompt_adaln_single,
        audio_timestep_scaled,
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )

    return LTXAVPreparedTimesteps(
        video_timestep=video_timestep,
        audio_timestep=audio_prepared.view(batch_size, -1, audio_prepared.shape[-1]),
        video_cross_scale_shift_timestep=CompressedTimestep(
            video_cross_scale_shift.view(batch_size, -1, video_cross_scale_shift.shape[-1]),
            patches_per_frame,
        ),
        audio_cross_scale_shift_timestep=audio_cross_scale_shift.view(
            batch_size,
            -1,
            audio_cross_scale_shift.shape[-1],
        ),
        video_cross_gate_timestep=CompressedTimestep(
            video_cross_gate.view(batch_size, -1, video_cross_gate.shape[-1]),
            patches_per_frame,
        ),
        audio_cross_gate_timestep=audio_cross_gate.view(batch_size, -1, audio_cross_gate.shape[-1]),
        video_prompt_timestep=video_prompt,
        audio_prompt_timestep=audio_prompt,
        video_embedded_timestep=video_embedded,
        audio_embedded_timestep=audio_embedded.view(batch_size, -1, audio_embedded.shape[-1]),
    )

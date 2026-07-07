from __future__ import annotations

from dataclasses import dataclass

import torch

from .ltx_rope import precompute_ltx_freqs_cis
from .ltxav_io import LTXAVInputProjection, LTXAVProjectedInputs


@dataclass(frozen=True)
class LTXAVPreparedBlockInputs:
    projected: LTXAVProjectedInputs
    video_context: torch.Tensor
    audio_context: torch.Tensor
    attention_mask: torch.Tensor | None
    video_pe: tuple[torch.Tensor, torch.Tensor, bool]
    audio_pe: tuple[torch.Tensor, torch.Tensor, bool]
    video_cross_pe: tuple[torch.Tensor, torch.Tensor, bool]
    audio_cross_pe: tuple[torch.Tensor, torch.Tensor, bool]
    grid_mask: torch.Tensor | None = None
    orig_patchified_shape: tuple[int, ...] | None = None


def prepare_attention_mask(attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
    if attention_mask is not None and not torch.is_floating_point(attention_mask):
        attention_mask = (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max
    return attention_mask


def split_ltxav_context(
    context: torch.Tensor,
    *,
    video_dim: int,
    audio_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.shape[-1] != video_dim + audio_dim:
        raise ValueError(f"expected context dim {video_dim + audio_dim}, got {context.shape[-1]}")
    return torch.split(context, [video_dim, audio_dim], dim=-1)


def prepare_ltxav_positional_embeddings(
    projected: LTXAVProjectedInputs,
    *,
    frame_rate: float,
    dtype: torch.dtype,
    video_dim: int,
    audio_dim: int,
    audio_cross_dim: int,
    video_heads: int,
    audio_heads: int,
    positional_embedding_theta: float = 10000.0,
    video_max_pos: tuple[int, int, int] = (20, 2048, 2048),
    audio_max_pos: tuple[int] = (20,),
    use_middle_indices_grid: bool = True,
    split_rope: bool = True,
    double_precision_grid: bool = True,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, bool],
    tuple[torch.Tensor, torch.Tensor, bool],
    tuple[torch.Tensor, torch.Tensor, bool],
    tuple[torch.Tensor, torch.Tensor, bool],
]:
    video_coords = projected.video_pixel_coords.to(torch.float32)
    video_coords[:, 0] = video_coords[:, 0] * (1.0 / frame_rate)
    video_pe = precompute_ltx_freqs_cis(
        video_coords,
        dim=video_dim,
        out_dtype=dtype,
        theta=positional_embedding_theta,
        max_pos=video_max_pos,
        use_middle_indices_grid=use_middle_indices_grid,
        num_attention_heads=video_heads,
        split=split_rope,
        double_precision_grid=double_precision_grid,
    )
    audio_pe = precompute_ltx_freqs_cis(
        projected.audio_latent_coords,
        dim=audio_dim,
        out_dtype=dtype,
        theta=positional_embedding_theta,
        max_pos=audio_max_pos,
        use_middle_indices_grid=use_middle_indices_grid,
        num_attention_heads=audio_heads,
        split=split_rope,
        double_precision_grid=double_precision_grid,
    )
    max_pos = (max(video_max_pos[0], audio_max_pos[0]),)
    video_cross_pe = precompute_ltx_freqs_cis(
        video_coords[:, 0:1, :],
        dim=audio_cross_dim,
        out_dtype=dtype,
        theta=positional_embedding_theta,
        max_pos=max_pos,
        use_middle_indices_grid=True,
        num_attention_heads=audio_heads,
        split=split_rope,
        double_precision_grid=double_precision_grid,
    )
    audio_cross_pe = precompute_ltx_freqs_cis(
        projected.audio_latent_coords[:, 0:1, :],
        dim=audio_cross_dim,
        out_dtype=dtype,
        theta=positional_embedding_theta,
        max_pos=max_pos,
        use_middle_indices_grid=True,
        num_attention_heads=audio_heads,
        split=split_rope,
        double_precision_grid=double_precision_grid,
    )
    return video_pe, audio_pe, video_cross_pe, audio_cross_pe


def prepare_ltxav_block_inputs(
    *,
    input_projection: LTXAVInputProjection,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    context: torch.Tensor,
    attention_mask: torch.Tensor | None,
    frame_rate: float,
    video_dim: int = 4096,
    audio_dim: int = 2048,
    audio_cross_dim: int = 2048,
    video_heads: int = 32,
    audio_heads: int = 32,
    keyframe_idxs: torch.Tensor | None = None,
    denoise_mask: torch.Tensor | None = None,
    guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
) -> LTXAVPreparedBlockInputs:
    projected = input_projection(
        video_latents,
        audio_latents,
        keyframe_idxs=keyframe_idxs,
        denoise_mask=denoise_mask,
        guide_attention_entries=guide_attention_entries,
    )
    video_context, audio_context = split_ltxav_context(context, video_dim=video_dim, audio_dim=audio_dim)
    prepared_mask = prepare_attention_mask(attention_mask, projected.video_tokens.dtype)
    video_pe, audio_pe, video_cross_pe, audio_cross_pe = prepare_ltxav_positional_embeddings(
        projected,
        frame_rate=frame_rate,
        dtype=projected.video_tokens.dtype,
        video_dim=video_dim,
        audio_dim=audio_dim,
        audio_cross_dim=audio_cross_dim,
        video_heads=video_heads,
        audio_heads=audio_heads,
    )
    return LTXAVPreparedBlockInputs(
        projected=projected,
        video_context=video_context,
        audio_context=audio_context,
        attention_mask=prepared_mask,
        video_pe=video_pe,
        audio_pe=audio_pe,
        video_cross_pe=video_cross_pe,
        audio_cross_pe=audio_cross_pe,
        grid_mask=projected.grid_mask,
        orig_patchified_shape=projected.orig_patchified_shape,
    )

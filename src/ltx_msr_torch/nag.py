from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LTX2NAGConfig:
    scale: float
    alpha: float
    tau: float
    inplace: bool = True


@dataclass(frozen=True)
class LTX2NAGPatchPlan:
    config: LTX2NAGConfig
    patch_video: bool
    patch_audio: bool
    transformer_block_count: int
    video_patch_targets: tuple[str, ...]
    audio_patch_targets: tuple[str, ...]


def normalized_attention_guidance(
    x_positive: torch.Tensor,
    x_negative: torch.Tensor,
    *,
    scale: float,
    alpha: float,
    tau: float,
    inplace: bool = False,
) -> torch.Tensor:
    """Pure torch equivalent of KJNodes LTX2 normalized_attention_guidance."""
    if inplace:
        x_positive = x_positive.clone()
        x_negative = x_negative.clone()
        nag_guidance = x_negative.mul_(scale - 1).neg_().add_(x_positive, alpha=scale)
    else:
        nag_guidance = (x_positive * scale).sub(x_negative * (scale - 1))

    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(nag_guidance, p=1, dim=-1, keepdim=True)
    ratio = norm_guidance / norm_positive
    ratio = torch.nan_to_num(ratio, nan=10.0)
    adjustment = (norm_positive * tau) / (norm_guidance + 1e-7)
    nag_guidance = nag_guidance * torch.where(ratio > tau, adjustment, 1.0)

    if inplace:
        return nag_guidance.sub_(x_positive).mul_(alpha).add_(x_positive)
    return nag_guidance * alpha + x_positive * (1 - alpha)


def build_ltx2_nag_patch_plan(
    *,
    scale: float,
    alpha: float,
    tau: float,
    inplace: bool,
    transformer_block_count: int,
    has_video_conditioning: bool,
    has_audio_conditioning: bool,
) -> LTX2NAGPatchPlan:
    config = LTX2NAGConfig(scale=scale, alpha=alpha, tau=tau, inplace=inplace)
    if scale == 0:
        return LTX2NAGPatchPlan(
            config=config,
            patch_video=False,
            patch_audio=False,
            transformer_block_count=transformer_block_count,
            video_patch_targets=(),
            audio_patch_targets=(),
        )

    video_targets = (
        tuple(f"diffusion_model.transformer_blocks.{idx}.attn2.forward" for idx in range(transformer_block_count))
        if has_video_conditioning
        else ()
    )
    audio_targets = (
        tuple(f"diffusion_model.transformer_blocks.{idx}.audio_attn2.forward" for idx in range(transformer_block_count))
        if has_audio_conditioning
        else ()
    )
    return LTX2NAGPatchPlan(
        config=config,
        patch_video=bool(video_targets),
        patch_audio=bool(audio_targets),
        transformer_block_count=transformer_block_count,
        video_patch_targets=video_targets,
        audio_patch_targets=audio_targets,
    )

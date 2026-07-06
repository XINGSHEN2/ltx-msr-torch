from __future__ import annotations

from dataclasses import dataclass

import torch

from .lora_loader import LocalICLoRALoadResult, inspect_ic_lora_model_only
from .model_paths import LocalModelPaths, resolve_workflow_model_paths
from .torch_nodes import (
    LocalRandomNoise,
    empty_ltxv_latent_video,
    int_constant,
    manual_sigmas,
    random_noise,
)
from .workflow_config import WorkflowConfig


@dataclass(frozen=True)
class LocalLowLevelState:
    width: int
    height: int
    frame_count: int
    video_length: int
    video_latent: dict[str, torch.Tensor | int]
    sigmas: torch.Tensor
    noise: LocalRandomNoise
    ic_lora: LocalICLoRALoadResult
    model_paths: LocalModelPaths


def build_low_level_state(
    config: WorkflowConfig,
    *,
    device: torch.device | str | None = None,
) -> LocalLowLevelState:
    """Build local replacements for the workflow's low-level nodes."""
    width = int_constant(config.reference.width)
    height = int_constant(config.reference.height)
    video_length = int_constant(config.latent.video_frames)
    sigmas = manual_sigmas(", ".join(str(value) for value in config.sampling.sigmas))
    noise = random_noise(config.sampling.seed)
    video_latent = empty_ltxv_latent_video(
        width=config.latent.width,
        height=config.latent.height,
        length=config.latent.video_frames,
        batch_size=config.latent.batch_size,
        device=device,
    )
    ic_lora = inspect_ic_lora_model_only(
        config.model.lora,
        strength_model=config.model.lora_strength,
    )
    model_paths = resolve_workflow_model_paths(config)
    return LocalLowLevelState(
        width=width,
        height=height,
        frame_count=config.reference.frame_count,
        video_length=video_length,
        video_latent=video_latent,
        sigmas=sigmas,
        noise=noise,
        ic_lora=ic_lora,
        model_paths=model_paths,
    )

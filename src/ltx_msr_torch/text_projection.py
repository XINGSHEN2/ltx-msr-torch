from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint_loader import load_safetensors_subset, strip_prefix_from_state_dict


TEXT_PROJECTION_KEYS: tuple[str, ...] = (
    "text_embedding_projection.audio_aggregate_embed.bias",
    "text_embedding_projection.audio_aggregate_embed.weight",
    "text_embedding_projection.video_aggregate_embed.bias",
    "text_embedding_projection.video_aggregate_embed.weight",
)


@dataclass(frozen=True)
class TextProjectionConfig:
    input_dim: int
    video_dim: int
    audio_dim: int
    dtype: torch.dtype


class DualLinearTextProjection(torch.nn.Module):
    """Local equivalent of ComfyUI LTXAV DualLinearProjection."""

    def __init__(
        self,
        *,
        input_dim: int,
        video_dim: int,
        audio_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.audio_aggregate_embed = torch.nn.Linear(input_dim, audio_dim, bias=True, dtype=dtype, device=device)
        self.video_aggregate_embed = torch.nn.Linear(input_dim, video_dim, bias=True, dtype=dtype, device=device)

    @property
    def config(self) -> TextProjectionConfig:
        return TextProjectionConfig(
            input_dim=self.video_aggregate_embed.in_features,
            video_dim=self.video_aggregate_embed.out_features,
            audio_dim=self.audio_aggregate_embed.out_features,
            dtype=self.video_aggregate_embed.weight.dtype,
        )

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        source_dim = hidden.shape[-1]
        x = hidden.movedim(1, -1)
        x = (x * torch.rsqrt(torch.mean(x**2, dim=2, keepdim=True) + 1e-6)).flatten(start_dim=2)
        if attention_mask is not None:
            x = torch.where(attention_mask.to(device=x.device).bool().unsqueeze(-1), x, torch.zeros_like(x))
        video = self.video_aggregate_embed(x * math.sqrt(self.video_aggregate_embed.out_features / source_dim))
        audio = self.audio_aggregate_embed(x * math.sqrt(self.audio_aggregate_embed.out_features / source_dim))
        return torch.cat((video, audio), dim=-1)


def infer_text_projection_config(state_dict: dict[str, torch.Tensor]) -> TextProjectionConfig:
    video_weight = state_dict["video_aggregate_embed.weight"]
    audio_weight = state_dict["audio_aggregate_embed.weight"]
    if video_weight.shape[1] != audio_weight.shape[1]:
        raise ValueError("video and audio text projection input dims must match")
    return TextProjectionConfig(
        input_dim=int(video_weight.shape[1]),
        video_dim=int(video_weight.shape[0]),
        audio_dim=int(audio_weight.shape[0]),
        dtype=video_weight.dtype,
    )


def load_text_projection_state_dict(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    raw = load_safetensors_subset(checkpoint_path, TEXT_PROJECTION_KEYS, device=device)
    return strip_prefix_from_state_dict(raw, "text_embedding_projection")


def build_text_projection_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> DualLinearTextProjection:
    state_dict = load_text_projection_state_dict(checkpoint_path, device=device)
    config = infer_text_projection_config(state_dict)
    module = DualLinearTextProjection(
        input_dim=config.input_dim,
        video_dim=config.video_dim,
        audio_dim=config.audio_dim,
        dtype=config.dtype,
        device=device,
    )
    module.load_state_dict(state_dict, strict=True)
    module.eval()
    return module

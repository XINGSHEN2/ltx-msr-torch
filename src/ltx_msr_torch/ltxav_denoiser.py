from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class LTXAVModelProtocol(Protocol):
    def __call__(
        self,
        *,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        frame_rate: float,
        attention_mask: torch.Tensor | None = None,
        transformer_options: dict[str, object] | None = None,
        self_attention_mask: torch.Tensor | None = None,
        ref_audio_seq_len: int = 0,
        target_audio_seq_len: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        ...


@dataclass(frozen=True)
class LTXAVDenoiser:
    model: LTXAVModelProtocol
    context: torch.Tensor
    attention_mask: torch.Tensor | None
    frame_rate: float
    transformer_options: dict[str, object] | None = None
    self_attention_mask: torch.Tensor | None = None
    ref_audio_seq_len: int = 0

    def __call__(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_latents, audio_latents = latents
        batch_size = video_latents.shape[0]
        if sigma.ndim == 0:
            sigma = sigma.expand(batch_size)
        video_timestep = sigma.reshape(batch_size, 1).expand(batch_size, _video_token_count(video_latents))
        audio_timestep = sigma.reshape(batch_size, 1).expand(batch_size, _audio_token_count(audio_latents))
        output = self.model(
            video_latents=video_latents,
            audio_latents=audio_latents,
            context=self.context,
            timestep=video_timestep,
            audio_timestep=audio_timestep,
            frame_rate=self.frame_rate,
            attention_mask=self.attention_mask,
            transformer_options=self.transformer_options,
            self_attention_mask=self.self_attention_mask,
            ref_audio_seq_len=self.ref_audio_seq_len,
            target_audio_seq_len=audio_timestep.shape[1],
        )
        if not isinstance(output, list) or len(output) != 2:
            raise TypeError("LTXAV denoiser expects model output [video, audio]")
        return output[0], output[1]


def _video_token_count(video_latents: torch.Tensor) -> int:
    if video_latents.ndim != 5:
        raise ValueError(f"expected video latents [B,C,T,H,W], got {tuple(video_latents.shape)}")
    return int(video_latents.shape[2] * video_latents.shape[3] * video_latents.shape[4])


def _audio_token_count(audio_latents: torch.Tensor) -> int:
    if audio_latents.ndim != 4:
        raise ValueError(f"expected audio latents [B,C,T,F], got {tuple(audio_latents.shape)}")
    return int(audio_latents.shape[2])

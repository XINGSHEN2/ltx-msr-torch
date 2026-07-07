from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .ltx_vae import decode_ltx_audio_latents, decode_ltx_video_latents


class VideoDecoderProtocol(Protocol):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        ...


class AudioDecoderProtocol(Protocol):
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class LTXAVDecodedOutput:
    video: torch.Tensor
    audio: torch.Tensor | None


@torch.no_grad()
def decode_ltxav_latents(
    *,
    video_vae: VideoDecoderProtocol,
    video_latents: torch.Tensor,
    audio_vae: AudioDecoderProtocol | None = None,
    audio_latents: torch.Tensor | None = None,
) -> LTXAVDecodedOutput:
    video = decode_ltx_video_latents(video_vae, video_latents)
    audio = None
    if audio_latents is not None:
        if audio_vae is None:
            raise ValueError("audio_vae is required when audio_latents are provided")
        audio = decode_ltx_audio_latents(audio_vae, audio_latents)
    return LTXAVDecodedOutput(video=video, audio=audio)

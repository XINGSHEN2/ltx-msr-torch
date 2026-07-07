from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .ltxav_denoiser import LTXAVModelProtocol, sample_ltxav_euler
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


@dataclass(frozen=True)
class LTXAVWorkflowSampleOutput:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    decoded: LTXAVDecodedOutput | None


@torch.no_grad()
def decode_ltxav_latents(
    *,
    video_vae: VideoDecoderProtocol,
    video_latents: torch.Tensor,
    audio_vae: AudioDecoderProtocol | None = None,
    audio_latents: torch.Tensor | None = None,
) -> LTXAVDecodedOutput:
    video_latents = _latents_for_module(video_latents, video_vae)
    video = decode_ltx_video_latents(video_vae, video_latents)
    audio = None
    if audio_latents is not None:
        if audio_vae is None:
            raise ValueError("audio_vae is required when audio_latents are provided")
        audio_latents = _latents_for_module(audio_latents, audio_vae)
        audio = decode_ltx_audio_latents(audio_vae, audio_latents)
    return LTXAVDecodedOutput(video=video, audio=audio)


def _latents_for_module(latents: torch.Tensor, module: object) -> torch.Tensor:
    if not isinstance(module, torch.nn.Module):
        return latents
    try:
        reference = next(module.parameters())
    except StopIteration:
        return latents
    return latents.to(device=reference.device, dtype=reference.dtype)


@torch.no_grad()
def sample_ltxav_workflow_latents(
    *,
    model: LTXAVModelProtocol,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    context: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sigmas: torch.Tensor,
    frame_rate: float,
    transformer_options: dict[str, object] | None = None,
    self_attention_mask: torch.Tensor | None = None,
    ref_audio_seq_len: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return sample_ltxav_euler(
        model=model,
        video_latents=video_latents,
        audio_latents=audio_latents,
        context=context,
        attention_mask=attention_mask,
        frame_rate=frame_rate,
        sigmas=sigmas,
        transformer_options=transformer_options,
        self_attention_mask=self_attention_mask,
        ref_audio_seq_len=ref_audio_seq_len,
    )


@torch.no_grad()
def run_ltxav_sample_decode(
    *,
    model: LTXAVModelProtocol,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    context: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sigmas: torch.Tensor,
    frame_rate: float,
    video_vae: VideoDecoderProtocol | None = None,
    audio_vae: AudioDecoderProtocol | None = None,
    transformer_options: dict[str, object] | None = None,
    self_attention_mask: torch.Tensor | None = None,
    ref_audio_seq_len: int = 0,
) -> LTXAVWorkflowSampleOutput:
    sampled_video, sampled_audio = sample_ltxav_workflow_latents(
        model=model,
        video_latents=video_latents,
        audio_latents=audio_latents,
        context=context,
        attention_mask=attention_mask,
        sigmas=sigmas,
        frame_rate=frame_rate,
        transformer_options=transformer_options,
        self_attention_mask=self_attention_mask,
        ref_audio_seq_len=ref_audio_seq_len,
    )
    decoded = None
    if video_vae is not None:
        decoded = decode_ltxav_latents(
            video_vae=video_vae,
            video_latents=sampled_video,
            audio_vae=audio_vae,
            audio_latents=sampled_audio if audio_vae is not None else None,
        )
    return LTXAVWorkflowSampleOutput(
        video_latents=sampled_video,
        audio_latents=sampled_audio,
        decoded=decoded,
    )

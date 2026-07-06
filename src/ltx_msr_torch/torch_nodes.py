from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import torch


def int_constant(value: int) -> int:
    """Local equivalent of KJNodes `INTConstant`."""
    return value


def manual_sigmas(sigmas: str) -> torch.Tensor:
    """Local equivalent of ComfyUI `ManualSigmas`.

    ComfyUI extracts all float-like tokens with:
    `r"[-+]?(?:\\d*\\.*\\d+)"`
    and returns `torch.FloatTensor(values)`.
    """
    values = re.findall(r"[-+]?(?:\d*\.*\d+)", sigmas)
    return torch.FloatTensor([float(value) for value in values])


@dataclass(frozen=True)
class LocalRandomNoise:
    """Local equivalent handle for ComfyUI `Noise_RandomNoise`."""

    seed: int

    def generate_noise(self, input_latent: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate deterministic CPU noise matching the latent shape.

        This is a local fallback. ComfyUI's exact sampler path uses
        `comfy.sample.prepare_noise`, which additionally handles batch indices
        and Comfy-specific devices. The seed and output shape contract are the
        same, which is the part used by the parity configuration.
        """
        latent_image = input_latent["samples"]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.seed))
        return torch.randn(
            latent_image.shape,
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
            generator=generator,
        )


def random_noise(noise_seed: int) -> LocalRandomNoise:
    """Local equivalent of ComfyUI `RandomNoise`."""
    return LocalRandomNoise(seed=noise_seed)


def empty_ltxv_latent_video(
    width: int,
    height: int,
    length: int,
    batch_size: int = 1,
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor | int]:
    """Local equivalent of ComfyUI `EmptyLTXVLatentVideo`."""
    latent = torch.zeros(
        [
            batch_size,
            128,
            ((length - 1) // 8) + 1,
            height // 32,
            width // 32,
        ],
        device=device,
    )
    return {"samples": latent, "downscale_ratio_spacial": 32}


class _FirstStageAudioModel(Protocol):
    latent_frequency_bins: int

    def num_of_latents_from_frames(self, frames_number: int, frame_rate: int) -> int:
        ...


class AudioVAEProtocol(Protocol):
    latent_channels: int
    first_stage_model: _FirstStageAudioModel


def empty_ltxv_latent_audio(
    frames_number: int,
    frame_rate: int,
    batch_size: int,
    audio_vae: AudioVAEProtocol,
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor | str]:
    """Local equivalent of ComfyUI `LTXVEmptyLatentAudio`."""
    if audio_vae is None:
        raise AssertionError("Audio VAE model is required")

    z_channels = audio_vae.latent_channels
    audio_freq = audio_vae.first_stage_model.latent_frequency_bins
    num_audio_latents = audio_vae.first_stage_model.num_of_latents_from_frames(
        frames_number,
        frame_rate,
    )
    audio_latents = torch.zeros(
        (batch_size, z_channels, num_audio_latents, audio_freq),
        device=device,
    )
    return {
        "samples": audio_latents,
        "type": "audio",
    }


def conditioning_set_values(
    conditioning: list[list[object]],
    values: dict[str, object],
    *,
    append: bool = False,
) -> list[list[object]]:
    """Local equivalent of ComfyUI `node_helpers.conditioning_set_values`."""
    output: list[list[object]] = []
    for item in conditioning:
        metadata = item[1].copy()
        for key, new_value in values.items():
            value = new_value
            if append:
                old_value = metadata.get(key)
                if old_value is not None:
                    value = old_value + new_value
            metadata[key] = value
        output.append([item[0], metadata])
    return output


def ltxv_conditioning(
    positive: list[list[object]],
    negative: list[list[object]],
    frame_rate: float,
) -> tuple[list[list[object]], list[list[object]]]:
    """Local equivalent of ComfyUI `LTXVConditioning`."""
    values = {"frame_rate": frame_rate}
    return (
        conditioning_set_values(positive, values),
        conditioning_set_values(negative, values),
    )

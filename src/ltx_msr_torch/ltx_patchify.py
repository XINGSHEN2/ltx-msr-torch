from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VideoPatchifyResult:
    patches: torch.Tensor
    latent_coords: torch.Tensor


@dataclass(frozen=True)
class AudioPatchifyResult:
    patches: torch.Tensor
    timings: torch.Tensor


def latent_to_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: tuple[int, int, int],
    *,
    causal_fix: bool = False,
) -> torch.Tensor:
    shape = [1] * latent_coords.ndim
    shape[1] = -1
    scales = torch.tensor(scale_factors, device=latent_coords.device, dtype=latent_coords.dtype).view(*shape)
    pixel_coords = latent_coords * scales
    if causal_fix:
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)
    return pixel_coords


def video_latent_coords(
    *,
    frames: int,
    height: int,
    width: int,
    batch_size: int,
    patch_size: tuple[int, int, int] = (1, 1, 1),
    start_end: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    frame_coords, height_coords, width_coords = torch.meshgrid(
        torch.arange(0, frames, patch_size[0], device=device),
        torch.arange(0, height, patch_size[1], device=device),
        torch.arange(0, width, patch_size[2], device=device),
        indexing="ij",
    )
    starts = torch.stack((frame_coords, height_coords, width_coords), dim=0)
    starts = starts.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).reshape(batch_size, 3, -1)
    if not start_end:
        return starts
    delta = torch.tensor(patch_size, device=starts.device, dtype=starts.dtype).view(1, 3, 1)
    ends = starts + delta
    return torch.stack((starts, ends), dim=-1)


def symmetric_patchify_video(
    latents: torch.Tensor,
    *,
    patch_size: int = 1,
    start_end: bool = True,
) -> VideoPatchifyResult:
    batch, channels, frames, height, width = latents.shape
    patch = (1, patch_size, patch_size)
    if frames % patch[0] or height % patch[1] or width % patch[2]:
        raise ValueError("latent shape must be divisible by patch size")
    coords = video_latent_coords(
        frames=frames,
        height=height,
        width=width,
        batch_size=batch,
        patch_size=patch,
        start_end=start_end,
        device=latents.device,
    )
    patches = (
        latents.reshape(batch, channels, frames // patch[0], patch[0], height // patch[1], patch[1], width // patch[2], patch[2])
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(batch, -1, channels * patch[0] * patch[1] * patch[2])
    )
    return VideoPatchifyResult(patches=patches, latent_coords=coords)


def symmetric_unpatchify_video(
    patches: torch.Tensor,
    *,
    output_height: int,
    output_width: int,
    output_num_frames: int,
    out_channels: int,
    patch_size: int = 1,
) -> torch.Tensor:
    height = output_height // patch_size
    width = output_width // patch_size
    batch = patches.shape[0]
    return (
        patches.reshape(batch, output_num_frames, height, width, out_channels, patch_size, patch_size)
        .permute(0, 4, 1, 2, 5, 3, 6)
        .reshape(batch, out_channels, output_num_frames, output_height, output_width)
    )


def audio_latent_time_in_seconds(
    start_latent: int,
    end_latent: int,
    *,
    sample_rate: int = 16000,
    hop_length: int = 160,
    audio_latent_downsample_factor: int = 4,
    is_causal: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    audio_latent_frame = torch.arange(start_latent, end_latent, dtype=dtype, device=device)
    audio_mel_frame = audio_latent_frame * audio_latent_downsample_factor
    if is_causal:
        audio_mel_frame = (audio_mel_frame + 1 - audio_latent_downsample_factor).clip(min=0)
    return audio_mel_frame * hop_length / sample_rate


def patchify_audio(
    audio_latents: torch.Tensor,
    *,
    sample_rate: int = 16000,
    hop_length: int = 160,
    audio_latent_downsample_factor: int = 4,
    is_causal: bool = True,
    start_end: bool = True,
    shift: int = 0,
) -> AudioPatchifyResult:
    batch, channels, time, frequency = audio_latents.shape
    patches = audio_latents.permute(0, 2, 1, 3).reshape(batch, time, channels * frequency)
    starts = audio_latent_time_in_seconds(
        shift,
        time + shift,
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio_latent_downsample_factor=audio_latent_downsample_factor,
        is_causal=is_causal,
        dtype=torch.float32,
        device=audio_latents.device,
    ).unsqueeze(0).expand(batch, -1).unsqueeze(1)
    if not start_end:
        return AudioPatchifyResult(patches=patches, timings=starts)
    ends = audio_latent_time_in_seconds(
        shift + 1,
        time + shift + 1,
        sample_rate=sample_rate,
        hop_length=hop_length,
        audio_latent_downsample_factor=audio_latent_downsample_factor,
        is_causal=is_causal,
        dtype=torch.float32,
        device=audio_latents.device,
    ).unsqueeze(0).expand(batch, -1).unsqueeze(1)
    return AudioPatchifyResult(patches=patches, timings=torch.stack((starts, ends), dim=-1))


def unpatchify_audio(
    patches: torch.Tensor,
    *,
    channels: int,
    frequency: int,
) -> torch.Tensor:
    batch, time, _ = patches.shape
    return patches.reshape(batch, time, channels, frequency).permute(0, 2, 1, 3)

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


ScaleFactors = tuple[int, int, int]
LatentShape = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ICLoRAVideoGuidePlan:
    latent_shape: LatentShape
    image_frame_count: int
    scale_factors: ScaleFactors
    frame_idx: int
    latent_idx: int
    num_frames_to_keep: int
    causal_fix: bool
    encode_frame_count: int
    effective_guide_frame_count: int
    target_width: int
    target_height: int
    estimated_guide_latent_shape: tuple[int, int, int]
    estimated_tokens_added: int
    latent_downscale_factor: float
    crop: str
    use_tiled_encode: bool
    tile_size: int
    tile_overlap: int


def plan_iclora_video_guide(
    *,
    latent_shape: Sequence[int],
    image_frame_count: int,
    scale_factors: ScaleFactors = (8, 32, 32),
    frame_idx: int = 0,
    latent_downscale_factor: float = 1.0,
    crop: str = "center",
    use_tiled_encode: bool = False,
    tile_size: int = 256,
    tile_overlap: int = 64,
    existing_keyframes: int = 0,
) -> ICLoRAVideoGuidePlan:
    """Plan the deterministic parts of LTXAddVideoICLoRAGuide.

    The VAE encode itself is model dependent. Everything here mirrors the
    shape, frame-index, and parameter logic around that encode path.
    """
    batch, channels, latent_length, latent_height, latent_width = _latent_shape(latent_shape)

    if image_frame_count <= 0:
        raise ValueError("image_frame_count must be positive")
    if latent_downscale_factor <= 0:
        raise ValueError("latent_downscale_factor must be positive")

    time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
    num_frames_to_keep = ((image_frame_count - 1) // time_scale_factor) * time_scale_factor + 1
    resolved_frame_idx = _resolve_frame_idx(
        latent_length=latent_length,
        guide_length=num_frames_to_keep,
        frame_idx=frame_idx,
        time_scale_factor=time_scale_factor,
        existing_keyframes=existing_keyframes,
    )
    causal_fix = resolved_frame_idx == 0 or num_frames_to_keep == 1
    encode_frame_count = num_frames_to_keep if causal_fix else num_frames_to_keep + 1

    if latent_downscale_factor > 1 and (
        latent_width % latent_downscale_factor != 0
        or latent_height % latent_downscale_factor != 0
    ):
        raise ValueError(
            f"Latent spatial size {latent_width}x{latent_height} must be divisible "
            f"by latent_downscale_factor {latent_downscale_factor}"
        )

    encoded_latent_frames = ((encode_frame_count - 1) // time_scale_factor) + 1
    effective_guide_frame_count = encoded_latent_frames if causal_fix else encoded_latent_frames - 1
    if effective_guide_frame_count <= 0:
        raise ValueError("guide frame count must be positive")

    target_width = int(latent_width * width_scale_factor / latent_downscale_factor)
    target_height = int(latent_height * height_scale_factor / latent_downscale_factor)
    guide_latent_height = int(latent_height / latent_downscale_factor)
    guide_latent_width = int(latent_width / latent_downscale_factor)
    dilated_height = guide_latent_height * int(latent_downscale_factor)
    dilated_width = guide_latent_width * int(latent_downscale_factor)
    estimated_guide_latent_shape = (effective_guide_frame_count, dilated_height, dilated_width)

    latent_idx = (resolved_frame_idx + time_scale_factor - 1) // time_scale_factor
    if latent_idx + effective_guide_frame_count > latent_length:
        raise AssertionError("Conditioning frames exceed the length of the latent sequence.")

    return ICLoRAVideoGuidePlan(
        latent_shape=(batch, channels, latent_length, latent_height, latent_width),
        image_frame_count=image_frame_count,
        scale_factors=scale_factors,
        frame_idx=resolved_frame_idx,
        latent_idx=latent_idx,
        num_frames_to_keep=num_frames_to_keep,
        causal_fix=causal_fix,
        encode_frame_count=encode_frame_count,
        effective_guide_frame_count=effective_guide_frame_count,
        target_width=target_width,
        target_height=target_height,
        estimated_guide_latent_shape=estimated_guide_latent_shape,
        estimated_tokens_added=(
            effective_guide_frame_count * estimated_guide_latent_shape[1] * estimated_guide_latent_shape[2]
        ),
        latent_downscale_factor=latent_downscale_factor,
        crop=crop,
        use_tiled_encode=use_tiled_encode,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )


def _latent_shape(latent_shape: Sequence[int]) -> LatentShape:
    if len(latent_shape) != 5:
        raise ValueError("latent_shape must be [batch, channels, frames, height, width]")
    return tuple(int(value) for value in latent_shape)  # type: ignore[return-value]


def _resolve_frame_idx(
    *,
    latent_length: int,
    guide_length: int,
    frame_idx: int,
    time_scale_factor: int,
    existing_keyframes: int,
) -> int:
    latent_count = latent_length - existing_keyframes
    resolved = frame_idx if frame_idx >= 0 else max((latent_count - 1) * time_scale_factor + 1 + frame_idx, 0)
    if guide_length > 1 and resolved != 0:
        resolved = (resolved - 1) // time_scale_factor * time_scale_factor + 1
    return resolved

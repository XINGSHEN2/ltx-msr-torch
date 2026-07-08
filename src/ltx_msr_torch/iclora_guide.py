from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .ltx_patchify import latent_to_pixel_coords, symmetric_patchify_video
from .ltx_vae import encode_ltx_video_pixels
from .torch_nodes import conditioning_set_values


ScaleFactors = tuple[int, int, int]
LatentShape = tuple[int, int, int, int, int]
Conditioning = list[list[object]]


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


@dataclass(frozen=True)
class ICLoRAGuideAppendResult:
    positive: Conditioning
    negative: Conditioning
    latent: dict[str, torch.Tensor]
    tokens_added: int
    guide_orig_shape: tuple[int, int, int]
    frame_idx: int
    latent_idx: int


@dataclass(frozen=True)
class ICLoRAPreparedGuideResult:
    encoded_pixels: torch.Tensor
    guide_latent: torch.Tensor
    append: ICLoRAGuideAppendResult


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


def get_conditioning_value(conditioning: Conditioning, key: str, default: object = None) -> object:
    for item in conditioning:
        metadata = item[1]
        if isinstance(metadata, dict) and key in metadata:
            return metadata[key]
    return default


def get_noise_mask(latent: dict[str, torch.Tensor]) -> torch.Tensor:
    latent_image = latent["samples"]
    noise_mask = latent.get("noise_mask")
    if noise_mask is not None:
        return noise_mask.clone()
    return torch.ones(
        (latent_image.shape[0], 1, latent_image.shape[2], 1, 1),
        dtype=torch.float32,
        device=latent_image.device,
    )


def get_keyframe_idxs(conditioning: Conditioning, latent_shape: Sequence[int] | None = None) -> tuple[torch.Tensor | None, int]:
    keyframe_idxs = get_conditioning_value(conditioning, "keyframe_idxs")
    if keyframe_idxs is None:
        return None, 0
    if not isinstance(keyframe_idxs, torch.Tensor):
        raise TypeError("keyframe_idxs must be a tensor")
    if latent_shape is not None and len(latent_shape) == 5:
        tokens_per_frame = int(latent_shape[-2]) * int(latent_shape[-1])
        return keyframe_idxs, int(keyframe_idxs.shape[2] // tokens_per_frame)
    entries = get_conditioning_value(conditioning, "guide_attention_entries", [])
    if entries:
        return keyframe_idxs, int(sum(entry["latent_shape"][0] for entry in entries))  # type: ignore[index]
    return keyframe_idxs, 0


def add_guide_attention_entry(
    conditioning: Conditioning,
    *,
    pre_filter_count: int,
    latent_shape: Sequence[int],
    attention_strength: float = 1.0,
    attention_mask: torch.Tensor | None = None,
) -> Conditioning:
    existing = get_conditioning_value(conditioning, "guide_attention_entries", [])
    entries = [*(existing if isinstance(existing, list) else [])]
    entries.append(
        {
            "pre_filter_count": int(pre_filter_count),
            "strength": float(attention_strength),
            "pixel_mask": attention_mask.unsqueeze(0).unsqueeze(0) if attention_mask is not None else None,
            "latent_shape": [int(value) for value in latent_shape],
        }
    )
    return conditioning_set_values(conditioning, {"guide_attention_entries": entries})


def add_keyframe_index(
    conditioning: Conditioning,
    *,
    frame_idx: int,
    guiding_latent: torch.Tensor,
    scale_factors: ScaleFactors = (8, 32, 32),
    latent_downscale_factor: float = 1.0,
    causal_fix: bool | None = None,
) -> Conditioning:
    keyframe_idxs, _ = get_keyframe_idxs(conditioning)
    patchified = symmetric_patchify_video(guiding_latent, patch_size=1, start_end=True)
    if causal_fix is None:
        causal_fix = frame_idx == 0 or guiding_latent.shape[2] == 1
    pixel_coords = latent_to_pixel_coords(patchified.latent_coords, scale_factors, causal_fix=causal_fix)
    pixel_coords[:, 0] += frame_idx
    spatial_end_offset = (latent_downscale_factor - 1) * torch.tensor(
        scale_factors[1:],
        device=pixel_coords.device,
    ).view(1, -1, 1, 1)
    pixel_coords[:, 1:, :, 1:] += spatial_end_offset.to(pixel_coords.dtype)
    if keyframe_idxs is not None:
        pixel_coords = torch.cat([keyframe_idxs, pixel_coords], dim=2)
    return conditioning_set_values(conditioning, {"keyframe_idxs": pixel_coords})


def append_iclora_keyframe(
    *,
    positive: Conditioning,
    negative: Conditioning,
    latent: dict[str, torch.Tensor],
    guiding_latent: torch.Tensor,
    frame_idx: int,
    strength: float,
    scale_factors: ScaleFactors = (8, 32, 32),
    guide_mask: torch.Tensor | None = None,
    latent_downscale_factor: float = 1.0,
    causal_fix: bool | None = None,
) -> ICLoRAGuideAppendResult:
    latent_image = latent["samples"]
    if latent_image.shape[1] != guiding_latent.shape[1]:
        raise ValueError("Adding guide to a combined AV latent is not supported.")
    resolved_frame_idx, latent_idx = _get_latent_index(
        positive,
        latent_length=latent_image.shape[2],
        guide_length=guiding_latent.shape[2],
        frame_idx=frame_idx,
        scale_factors=scale_factors,
        latent_shape=latent_image.shape,
    )
    if latent_idx + guiding_latent.shape[2] > latent_image.shape[2]:
        raise AssertionError("Conditioning frames exceed the length of the latent sequence.")
    noise_mask = get_noise_mask(latent)
    positive = add_keyframe_index(
        positive,
        frame_idx=resolved_frame_idx,
        guiding_latent=guiding_latent,
        scale_factors=scale_factors,
        latent_downscale_factor=latent_downscale_factor,
        causal_fix=causal_fix,
    )
    negative = add_keyframe_index(
        negative,
        frame_idx=resolved_frame_idx,
        guiding_latent=guiding_latent,
        scale_factors=scale_factors,
        latent_downscale_factor=latent_downscale_factor,
        causal_fix=causal_fix,
    )
    if guide_mask is not None:
        target_h = max(noise_mask.shape[3], guide_mask.shape[3])
        target_w = max(noise_mask.shape[4], guide_mask.shape[4])
        if noise_mask.shape[3] == 1 or noise_mask.shape[4] == 1:
            noise_mask = noise_mask.expand(-1, -1, -1, target_h, target_w)
        if guide_mask.shape[3] == 1 or guide_mask.shape[4] == 1:
            guide_mask = guide_mask.expand(-1, -1, -1, target_h, target_w)
        mask = guide_mask - strength
    else:
        mask = torch.full(
            (noise_mask.shape[0], 1, guiding_latent.shape[2], noise_mask.shape[3], noise_mask.shape[4]),
            max(0.0, 1.0 - strength),
            dtype=noise_mask.dtype,
            device=noise_mask.device,
        )
    latent_image = torch.cat([latent_image, guiding_latent], dim=2)
    noise_mask = torch.cat([noise_mask, mask], dim=2)
    guide_orig_shape = tuple(int(value) for value in guiding_latent.shape[2:])
    tokens_added = int(guiding_latent.shape[2] * guiding_latent.shape[3] * guiding_latent.shape[4])
    positive = add_guide_attention_entry(
        positive,
        pre_filter_count=tokens_added,
        latent_shape=guide_orig_shape,
        attention_strength=strength,
    )
    negative = add_guide_attention_entry(
        negative,
        pre_filter_count=tokens_added,
        latent_shape=guide_orig_shape,
        attention_strength=strength,
    )
    return ICLoRAGuideAppendResult(
        positive=positive,
        negative=negative,
        latent={"samples": latent_image, "noise_mask": noise_mask},
        tokens_added=tokens_added,
        guide_orig_shape=guide_orig_shape,
        frame_idx=resolved_frame_idx,
        latent_idx=latent_idx,
    )


def prepare_and_append_iclora_video_guide(
    *,
    video_vae: torch.nn.Module,
    positive: Conditioning,
    negative: Conditioning,
    latent: dict[str, torch.Tensor],
    image: torch.Tensor,
    frame_idx: int,
    strength: float,
    latent_downscale_factor: float = 1.0,
    crop: str = "center",
    scale_factors: ScaleFactors = (8, 32, 32),
) -> ICLoRAPreparedGuideResult:
    latent_image = latent["samples"]
    plan = plan_iclora_video_guide(
        latent_shape=latent_image.shape,
        image_frame_count=image.shape[0],
        scale_factors=scale_factors,
        frame_idx=frame_idx,
        latent_downscale_factor=latent_downscale_factor,
        crop=crop,
    )
    pixels = image[: plan.num_frames_to_keep]
    if not plan.causal_fix:
        pixels = torch.cat([pixels[:1], pixels], dim=0)
    resized = resize_video_pixels(
        pixels,
        width=plan.target_width,
        height=plan.target_height,
        crop=crop,
    )
    guide_latent = encode_ltx_video_pixels(video_vae, resized)
    if not plan.causal_fix:
        guide_latent = guide_latent[:, :, 1:, :, :]
        resized = resized[1:]
    append = append_iclora_keyframe(
        positive=positive,
        negative=negative,
        latent=latent,
        guiding_latent=guide_latent,
        frame_idx=plan.frame_idx,
        strength=strength,
        scale_factors=scale_factors,
        latent_downscale_factor=latent_downscale_factor,
        causal_fix=plan.causal_fix,
    )
    return ICLoRAPreparedGuideResult(
        encoded_pixels=resized,
        guide_latent=guide_latent,
        append=append,
    )


def resize_video_pixels(
    pixels: torch.Tensor,
    *,
    width: int,
    height: int,
    crop: str = "center",
) -> torch.Tensor:
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"expected pixels [T,H,W,3], got {tuple(pixels.shape)}")
    if crop not in {"center", "disabled"}:
        raise ValueError(f"unsupported crop mode: {crop}")
    nchw = pixels.movedim(-1, 1)
    if crop == "center":
        old_width = int(nchw.shape[-1])
        old_height = int(nchw.shape[-2])
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        nchw = nchw.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    resized = F.interpolate(nchw, size=(height, width), mode="bilinear")
    return resized.movedim(1, -1)


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


def _get_latent_index(
    conditioning: Conditioning,
    *,
    latent_length: int,
    guide_length: int,
    frame_idx: int,
    scale_factors: ScaleFactors,
    latent_shape: Sequence[int] | None = None,
) -> tuple[int, int]:
    time_scale_factor = scale_factors[0]
    _, num_keyframes = get_keyframe_idxs(conditioning, latent_shape)
    latent_count = latent_length - num_keyframes
    resolved = frame_idx if frame_idx >= 0 else max((latent_count - 1) * time_scale_factor + 1 + frame_idx, 0)
    if guide_length > 1 and resolved != 0:
        resolved = (resolved - 1) // time_scale_factor * time_scale_factor + 1
    latent_idx = (resolved + time_scale_factor - 1) // time_scale_factor
    return resolved, latent_idx

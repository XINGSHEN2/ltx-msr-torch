from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


TokenRange = tuple[int, int]


@dataclass(frozen=True)
class PromptRelaySegment:
    token_range: TokenRange
    midpoint: int
    window: float
    sigma: float
    strength: float
    window_audio: float
    sigma_audio: float
    strength_audio: float


@dataclass(frozen=True)
class PromptRelayPlan:
    local_prompts: tuple[str, ...]
    latent_frames: int
    tokens_per_frame: int
    specified_latent_lengths: tuple[int, ...] | None
    effective_lengths: tuple[int, ...]
    segments: tuple[PromptRelaySegment, ...]


def split_local_prompts(local_prompts: str) -> tuple[str, ...]:
    locals_list = tuple(part.strip() for part in local_prompts.split("|") if part.strip())
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |)")
    return locals_list


def parse_segment_lengths(
    segment_lengths: str,
    *,
    temporal_stride: int,
    latent_frames: int,
) -> tuple[int, ...] | None:
    if not segment_lengths.strip():
        return None
    pixel_lengths = [int(part.strip()) for part in segment_lengths.split(",") if part.strip()]
    return tuple(convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames))


def convert_to_latent_lengths(
    pixel_lengths: Sequence[int],
    temporal_stride: int,
    latent_frames: int,
) -> list[int]:
    if not pixel_lengths:
        return []
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [length * target_total / total_pixel for length in pixel_lengths]
    result = [int(value) for value in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for index in range(diff):
            result[order[index % len(order)]] += 1

    for index in range(len(result)):
        if result[index] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[index] = 1
    return result


def distribute_segment_lengths(
    num_segments: int,
    latent_frames: int,
    specified_lengths: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if specified_lengths:
        if len(specified_lengths) != num_segments:
            raise ValueError(
                f"Number of segment_lengths ({len(specified_lengths)}) "
                f"must match number of local prompts ({num_segments})"
            )
        lengths = list(specified_lengths)
    else:
        step = -(-latent_frames // num_segments)
        lengths = [step] * num_segments

    effective: list[int] = []
    cursor = 0
    for length in lengths:
        end = min(cursor + length, latent_frames)
        effective.append(max(end - cursor, 0))
        cursor = end
    return tuple(effective)


def build_relay_segments(
    token_ranges: Sequence[TokenRange],
    segment_lengths: Sequence[int],
    *,
    epsilon: float = 1e-3,
    relay_options: dict[str, float | None] | None = None,
) -> tuple[PromptRelaySegment, ...]:
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    opts = relay_options or {}
    video_strength = float(opts.get("video_strength", 1.0) or 0.0)
    video_window_scale = float(opts.get("video_window_scale", 1.0) or 0.0)
    audio_epsilon = opts.get("audio_epsilon")
    audio_strength = float(opts.get("audio_strength", 1.0) or 0.0)
    audio_window_scale = float(opts.get("audio_window_scale", 1.0) or 0.0)
    if audio_epsilon is not None and 0 < audio_epsilon < 1:
        sigma_audio = 1.0 / math.log(1.0 / audio_epsilon)
    else:
        sigma_audio = sigma

    segments: list[PromptRelaySegment] = []
    frame_cursor = 0
    for token_range, length in zip(token_ranges, segment_lengths):
        if length <= 0:
            frame_cursor += length
            continue
        midpoint = (2 * frame_cursor + length) // 2
        base_window = max(length // 2 - 2, 0)
        segments.append(
            PromptRelaySegment(
                token_range=token_range,
                midpoint=midpoint,
                window=max(base_window * video_window_scale, 0.0),
                sigma=sigma,
                strength=video_strength,
                window_audio=max(base_window * audio_window_scale, 0.0),
                sigma_audio=sigma_audio,
                strength_audio=audio_strength,
            )
        )
        frame_cursor += length
    return tuple(segments)


def map_token_indices(
    tokenizer: Callable[[str], dict[str, Sequence[int]]],
    global_prompt: str,
    local_prompts: Iterable[str],
    *,
    add_eos: bool | None = None,
) -> tuple[str, tuple[TokenRange, ...]]:
    locals_tuple = tuple(local_prompts)
    prefixed_locals = tuple(" " + local_prompt for local_prompt in locals_tuple)
    full_prompt = global_prompt + "".join(prefixed_locals)
    has_eos = bool(getattr(tokenizer, "add_eos", False) if add_eos is None else add_eos)
    eos_adj = 1 if has_eos else 0

    prev_len = len(tokenizer(global_prompt)["input_ids"]) - eos_adj
    token_ranges: list[TokenRange] = []
    built = global_prompt
    for prefixed_local in prefixed_locals:
        built += prefixed_local
        cur_len = len(tokenizer(built)["input_ids"]) - eos_adj
        if cur_len <= prev_len:
            raise ValueError(f"Local prompt produced no tokens: '{prefixed_local.strip()}'")
        token_ranges.append((prev_len, cur_len))
        prev_len = cur_len
    return full_prompt, tuple(token_ranges)


def plan_prompt_relay(
    *,
    local_prompts: str,
    latent_shape: Sequence[int],
    patch_size: tuple[int, int, int] = (1, 1, 1),
    temporal_stride: int = 8,
    segment_lengths: str = "",
    token_ranges: Sequence[TokenRange],
    epsilon: float = 1e-3,
    relay_options: dict[str, float | None] | None = None,
) -> PromptRelayPlan:
    locals_list = split_local_prompts(local_prompts)
    latent_frames = int(latent_shape[2])
    tokens_per_frame = (int(latent_shape[3]) // patch_size[1]) * (int(latent_shape[4]) // patch_size[2])
    specified = parse_segment_lengths(
        segment_lengths,
        temporal_stride=temporal_stride,
        latent_frames=latent_frames,
    )
    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, specified)
    segments = build_relay_segments(token_ranges, effective_lengths, epsilon=epsilon, relay_options=relay_options)
    return PromptRelayPlan(
        local_prompts=locals_list,
        latent_frames=latent_frames,
        tokens_per_frame=tokens_per_frame,
        specified_latent_lengths=specified,
        effective_lengths=effective_lengths,
        segments=segments,
    )

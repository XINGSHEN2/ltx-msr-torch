from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .checkpoint_loader import load_safetensors_subset, strip_prefix_from_state_dict


@dataclass(frozen=True)
class VAESectionManifest:
    prefix: str
    key_count: int
    encoder_key_count: int
    decoder_key_count: int
    statistics_key_count: int
    first_keys: tuple[str, ...]
    first_shapes: tuple[tuple[str, tuple[int, ...], torch.dtype], ...]


def inspect_vae_section(
    checkpoint_path: str | Path,
    *,
    prefix: str = "vae",
    first_key_count: int = 8,
) -> VAESectionManifest:
    full_prefix = f"{prefix}."
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = tuple(key for key in handle.keys() if key.startswith(full_prefix))
        first_keys = keys[:first_key_count]
        first_shapes = tuple((key, tuple(handle.get_tensor(key).shape), handle.get_tensor(key).dtype) for key in first_keys)
    return VAESectionManifest(
        prefix=prefix,
        key_count=len(keys),
        encoder_key_count=sum(1 for key in keys if key.startswith(f"{full_prefix}encoder.")),
        decoder_key_count=sum(1 for key in keys if key.startswith(f"{full_prefix}decoder.")),
        statistics_key_count=sum(1 for key in keys if key.startswith(f"{full_prefix}per_channel_statistics.")),
        first_keys=first_keys,
        first_shapes=first_shapes,
    )


def load_vae_section_state_dict(
    checkpoint_path: str | Path,
    *,
    prefix: str = "vae",
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    full_prefix = f"{prefix}."
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = tuple(key for key in handle.keys() if key.startswith(full_prefix))
    raw = load_safetensors_subset(checkpoint_path, keys, device=device)
    return strip_prefix_from_state_dict(raw, prefix)

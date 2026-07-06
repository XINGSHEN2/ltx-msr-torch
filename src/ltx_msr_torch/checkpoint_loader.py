from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from .lora_apply import apply_lora_to_state_dict, match_lora_targets
from .lora_loader import LoRAManifest


CHECKPOINT_SECTIONS: tuple[str, ...] = (
    "model",
    "vae",
    "audio_vae",
    "vocoder",
    "text_embedding_projection",
)


@dataclass(frozen=True)
class CheckpointSection:
    name: str
    key_count: int
    first_keys: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointManifest:
    path: Path
    key_count: int
    sections: tuple[CheckpointSection, ...]
    unknown_keys: tuple[str, ...]


@dataclass(frozen=True)
class LoRAFilteredApplyResult:
    state_dict: dict[str, torch.Tensor]
    report_matched: int
    report_skipped: int


def inspect_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    resolved = Path(path)
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    sections = []
    known: set[str] = set()
    for section in CHECKPOINT_SECTIONS:
        section_keys = tuple(key for key in keys if key == section or key.startswith(f"{section}."))
        sections.append(
            CheckpointSection(
                name=section,
                key_count=len(section_keys),
                first_keys=section_keys[:8],
            )
        )
        known.update(section_keys)
    return CheckpointManifest(
        path=resolved,
        key_count=len(keys),
        sections=tuple(sections),
        unknown_keys=tuple(key for key in keys if key not in known),
    )


def load_safetensors_subset(
    path: str | Path,
    keys: Iterable[str],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    requested = tuple(dict.fromkeys(keys))
    with safe_open(str(path), framework="pt", device=str(device)) as handle:
        available = set(handle.keys())
        missing = [key for key in requested if key not in available]
        if missing:
            raise KeyError(f"checkpoint keys not found: {missing[:8]}")
        return {key: handle.get_tensor(key) for key in requested}


def strip_prefix_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    full_prefix = f"{prefix}."
    return {
        (key[len(full_prefix) :] if key.startswith(full_prefix) else key): value
        for key, value in state_dict.items()
    }


def load_lora_target_checkpoint_subset(
    checkpoint_path: str | Path,
    manifest: LoRAManifest,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        matches = match_lora_targets(set(handle.keys()), manifest)
    target_keys = tuple(match.state_key for match in matches if match.state_key is not None)
    return load_safetensors_subset(checkpoint_path, target_keys, device=device)


def apply_lora_to_checkpoint_subset(
    checkpoint_path: str | Path,
    *,
    lora_path: str | Path,
    manifest: LoRAManifest,
    strength: float,
    device: str | torch.device = "cpu",
) -> LoRAFilteredApplyResult:
    subset = load_lora_target_checkpoint_subset(checkpoint_path, manifest, device=device)
    updated, report = apply_lora_to_state_dict(
        subset,
        lora_path=lora_path,
        manifest=manifest,
        strength=strength,
        strict=True,
    )
    return LoRAFilteredApplyResult(
        state_dict=updated,
        report_matched=report.matched,
        report_skipped=report.skipped,
    )

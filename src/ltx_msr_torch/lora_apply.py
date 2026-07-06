from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open

from .lora_loader import LoRAManifest, LoRAPairManifest


@dataclass(frozen=True)
class LoRATargetMatch:
    pair: LoRAPairManifest
    state_key: str | None


@dataclass(frozen=True)
class LoRAApplyReport:
    matched: int
    skipped: int
    applied_keys: tuple[str, ...]
    skipped_targets: tuple[str, ...]


def target_key_candidates(target_key: str) -> tuple[str, ...]:
    candidates = [target_key]
    if target_key.startswith("diffusion_model."):
        candidates.append(f"model.{target_key}")
    if target_key.startswith("model.diffusion_model."):
        candidates.append(target_key[len("model.") :])
    return tuple(dict.fromkeys(candidates))


def match_lora_targets(
    state_keys: set[str],
    manifest: LoRAManifest,
) -> tuple[LoRATargetMatch, ...]:
    matches: list[LoRATargetMatch] = []
    for pair in manifest.pairs:
        state_key = next((candidate for candidate in target_key_candidates(pair.target_key) if candidate in state_keys), None)
        matches.append(LoRATargetMatch(pair=pair, state_key=state_key))
    return tuple(matches)


def lora_pair_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    target_shape: torch.Size | tuple[int, ...],
    *,
    alpha: float | None,
    strength: float,
) -> torch.Tensor:
    rank = lora_a.shape[0]
    scale = 1.0 if alpha is None else float(alpha) / float(rank)
    delta = torch.mm(lora_b.flatten(start_dim=1), lora_a.flatten(start_dim=1)).reshape(tuple(target_shape))
    return delta * (float(strength) * scale)


def apply_lora_to_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    lora_path: str | Path,
    manifest: LoRAManifest,
    strength: float,
    strict: bool = False,
) -> tuple[dict[str, torch.Tensor], LoRAApplyReport]:
    output = {key: value.clone() for key, value in state_dict.items()}
    matches = match_lora_targets(set(output.keys()), manifest)
    applied: list[str] = []
    skipped: list[str] = []
    with safe_open(str(lora_path), framework="pt", device="cpu") as handle:
        for match in matches:
            if match.state_key is None:
                skipped.append(match.pair.target_key)
                continue
            weight = output[match.state_key]
            lora_a = handle.get_tensor(match.pair.lora_a_key).to(device=weight.device, dtype=torch.float32)
            lora_b = handle.get_tensor(match.pair.lora_b_key).to(device=weight.device, dtype=torch.float32)
            delta = lora_pair_delta(
                lora_a,
                lora_b,
                weight.shape,
                alpha=match.pair.alpha,
                strength=strength,
            ).to(dtype=weight.dtype)
            output[match.state_key] = weight + delta
            applied.append(match.state_key)

    if strict and skipped:
        raise KeyError(f"LoRA targets not found in state_dict: {skipped[:8]}")
    return output, LoRAApplyReport(
        matched=len(applied),
        skipped=len(skipped),
        applied_keys=tuple(applied),
        skipped_targets=tuple(skipped),
    )

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from transformers import Gemma3TextConfig, Gemma3TextModel

from .model_paths import LocalModelPaths, resolve_workflow_model_paths
from .text_encoder_sections import TextEncoderConfigPaths, resolve_text_encoder_config_paths
from .workflow_config import WorkflowConfig, default_workflow_config


COMFY_TEXT_PREFIX = "model."


@dataclass(frozen=True)
class GemmaTextModelCompatibility:
    checkpoint_key_count: int
    hf_key_count: int
    matched_key_count: int
    missing_hf_keys: tuple[str, ...]
    unexpected_checkpoint_keys: tuple[str, ...]

    @property
    def is_exact_match(self) -> bool:
        return not self.missing_hf_keys and not self.unexpected_checkpoint_keys


def load_gemma3_text_config(paths: TextEncoderConfigPaths | None = None) -> Gemma3TextConfig:
    resolved = paths or resolve_text_encoder_config_paths()
    data = json.loads(Path(resolved.gemma_config).read_text())
    text_config = data.get("text_config", data)
    return Gemma3TextConfig(**text_config)


def build_empty_gemma3_text_model(
    config: Gemma3TextConfig | None = None,
    *,
    device: torch.device | str = "meta",
) -> Gemma3TextModel:
    resolved_config = config or load_gemma3_text_config()
    with torch.device(device):
        model = Gemma3TextModel(resolved_config)
    model.eval()
    return model


def map_comfy_text_encoder_key(key: str) -> str | None:
    if not key.startswith(COMFY_TEXT_PREFIX):
        return None
    return key[len(COMFY_TEXT_PREFIX) :]


def iter_comfy_text_encoder_keys(text_encoder_path: str | Path) -> tuple[str, ...]:
    with safe_open(str(text_encoder_path), framework="pt", device="cpu") as handle:
        return tuple(key for key in handle.keys() if key.startswith(COMFY_TEXT_PREFIX))


def map_comfy_text_encoder_keys(keys: Iterable[str]) -> tuple[str, ...]:
    mapped: list[str] = []
    for key in keys:
        mapped_key = map_comfy_text_encoder_key(key)
        if mapped_key is not None:
            mapped.append(mapped_key)
    return tuple(mapped)


def inspect_gemma_text_model_compatibility(
    *,
    paths: LocalModelPaths | None = None,
    config_paths: TextEncoderConfigPaths | None = None,
) -> GemmaTextModelCompatibility:
    model_paths = paths or resolve_workflow_model_paths(default_workflow_config())
    config = load_gemma3_text_config(config_paths)
    model = build_empty_gemma3_text_model(config)
    hf_keys = set(model.state_dict().keys())
    checkpoint_keys = set(map_comfy_text_encoder_keys(iter_comfy_text_encoder_keys(model_paths.text_encoder)))
    return GemmaTextModelCompatibility(
        checkpoint_key_count=len(checkpoint_keys),
        hf_key_count=len(hf_keys),
        matched_key_count=len(checkpoint_keys & hf_keys),
        missing_hf_keys=tuple(sorted(hf_keys - checkpoint_keys)),
        unexpected_checkpoint_keys=tuple(sorted(checkpoint_keys - hf_keys)),
    )


def load_gemma_text_state_dict_subset(
    text_encoder_path: str | Path,
    mapped_keys: Iterable[str],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    requested = tuple(dict.fromkeys(mapped_keys))
    comfy_keys = {f"{COMFY_TEXT_PREFIX}{key}": key for key in requested}
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(text_encoder_path), framework="pt", device=str(device)) as handle:
        available = set(handle.keys())
        missing = [source for source in comfy_keys if source not in available]
        if missing:
            raise KeyError(f"missing text encoder keys: {missing[:8]}")
        for source, target in comfy_keys.items():
            tensors[target] = handle.get_tensor(source)
    return tensors


def workflow_gemma_text_paths(config: WorkflowConfig | None = None) -> tuple[LocalModelPaths, TextEncoderConfigPaths]:
    workflow_config = config or default_workflow_config()
    return resolve_workflow_model_paths(workflow_config), resolve_text_encoder_config_paths()

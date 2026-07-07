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


@dataclass(frozen=True)
class GemmaTextModelLoadReport:
    loaded: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


def load_gemma3_text_config(paths: TextEncoderConfigPaths | None = None) -> Gemma3TextConfig:
    resolved = paths or resolve_text_encoder_config_paths()
    data = json.loads(Path(resolved.gemma_config).read_text())
    text_config = data.get("text_config", data)
    return Gemma3TextConfig(**text_config)


def build_empty_gemma3_text_model(
    config: Gemma3TextConfig | None = None,
    *,
    device: torch.device | str = "meta",
    dtype: torch.dtype | None = None,
    num_layers: int | None = None,
) -> Gemma3TextModel:
    resolved_config = config or load_gemma3_text_config()
    if num_layers is not None:
        if num_layers < 0 or num_layers > resolved_config.num_hidden_layers:
            raise ValueError(f"num_layers must be between 0 and {resolved_config.num_hidden_layers}, got {num_layers}")
        data = resolved_config.to_dict()
        if "layer_types" in data and data["layer_types"] is not None:
            data["layer_types"] = data["layer_types"][:num_layers]
        resolved_config = Gemma3TextConfig(**{**data, "num_hidden_layers": num_layers})
    original_dtype = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Gemma3TextModel(resolved_config)
    finally:
        if dtype is not None:
            torch.set_default_dtype(original_dtype)
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


def _resolve_state_target(model: torch.nn.Module, state_key: str) -> tuple[torch.nn.Module, str, torch.Tensor, bool]:
    parent_name, _, tensor_name = state_key.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if tensor_name in parent._parameters:
        tensor = parent._parameters[tensor_name]
        if tensor is None:
            raise KeyError(f"parameter is None: {state_key}")
        return parent, tensor_name, tensor, True
    if tensor_name in parent._buffers:
        tensor = parent._buffers[tensor_name]
        if tensor is None:
            raise KeyError(f"buffer is None: {state_key}")
        return parent, tensor_name, tensor, False
    raise KeyError(f"state key not found in module: {state_key}")


def _assign_state_tensor(
    model: torch.nn.Module,
    state_key: str,
    tensor: torch.Tensor,
    *,
    assign: bool,
) -> None:
    parent, tensor_name, target, is_parameter = _resolve_state_target(model, state_key)
    if tuple(target.shape) != tuple(tensor.shape):
        raise ValueError(f"shape mismatch for {state_key}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
    if assign or target.is_meta:
        if is_parameter:
            parent._parameters[tensor_name] = torch.nn.Parameter(tensor, requires_grad=target.requires_grad)
        else:
            parent._buffers[tensor_name] = tensor
        return
    with torch.no_grad():
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))


def load_gemma_text_model_weights_streaming(
    model: Gemma3TextModel,
    text_encoder_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    assign: bool = False,
    strict: bool = True,
) -> GemmaTextModelLoadReport:
    local_keys = tuple(model.state_dict())
    source_pairs = tuple((key, f"{COMFY_TEXT_PREFIX}{key}") for key in local_keys)
    with safe_open(str(text_encoder_path), framework="pt", device=str(device)) as handle:
        available = set(handle.keys())
        missing = tuple(source for _, source in source_pairs if source not in available)
        if strict and missing:
            raise KeyError(f"text encoder keys not found: {missing[:8]}")
        loaded = 0
        for local_key, source_key in source_pairs:
            if source_key not in available:
                continue
            _assign_state_tensor(model, local_key, handle.get_tensor(source_key), assign=assign)
            loaded += 1
    return GemmaTextModelLoadReport(loaded=loaded, missing=missing, unexpected=())


def workflow_gemma_text_paths(config: WorkflowConfig | None = None) -> tuple[LocalModelPaths, TextEncoderConfigPaths]:
    workflow_config = config or default_workflow_config()
    return resolve_workflow_model_paths(workflow_config), resolve_text_encoder_config_paths()

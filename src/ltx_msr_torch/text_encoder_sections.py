from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch
from safetensors import safe_open


DEFAULT_GEMMA_CONFIG_DIR = Path("/home/xingshen/ComfyUI/custom_nodes/ComfyUI-LTXVideo/gemma_configs")


@dataclass(frozen=True)
class TextEncoderConfigPaths:
    config_dir: Path
    gemma_config: Path
    tokenizer_json: Path
    tokenizer_model: Path
    tokenizer_config: Path


@dataclass(frozen=True)
class TextEncoderSectionManifest:
    path: Path
    key_count: int
    text_model_key_count: int
    vision_model_key_count: int
    projector_key_count: int
    spiece_key_count: int
    layer_count: int
    first_text_keys: tuple[str, ...]
    first_text_shapes: tuple[tuple[str, tuple[int, ...], torch.dtype], ...]
    config_paths: TextEncoderConfigPaths


def resolve_text_encoder_config_paths(
    config_dir: str | Path = DEFAULT_GEMMA_CONFIG_DIR,
) -> TextEncoderConfigPaths:
    root = Path(config_dir)
    paths = TextEncoderConfigPaths(
        config_dir=root,
        gemma_config=root / "gemma3cfg.json",
        tokenizer_json=root / "tokenizer.json",
        tokenizer_model=root / "tokenizer.model",
        tokenizer_config=root / "tokenizer_config.json",
    )
    missing = [path for path in (paths.gemma_config, paths.tokenizer_json, paths.tokenizer_model, paths.tokenizer_config) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Gemma config files: {missing}")
    return paths


def inspect_text_encoder_section(
    text_encoder_path: str | Path,
    *,
    config_dir: str | Path = DEFAULT_GEMMA_CONFIG_DIR,
    first_key_count: int = 8,
) -> TextEncoderSectionManifest:
    resolved = Path(text_encoder_path)
    layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
        text_keys = tuple(key for key in keys if key.startswith("model."))
        first_text_keys = text_keys[:first_key_count]
        first_text_shapes = tuple((key, tuple(handle.get_tensor(key).shape), handle.get_tensor(key).dtype) for key in first_text_keys)
        layer_indices = {
            int(match.group(1))
            for key in text_keys
            if (match := layer_pattern.match(key)) is not None
        }
    return TextEncoderSectionManifest(
        path=resolved,
        key_count=len(keys),
        text_model_key_count=len(text_keys),
        vision_model_key_count=sum(1 for key in keys if key.startswith("vision_model.")),
        projector_key_count=sum(1 for key in keys if key.startswith("multi_modal_projector.")),
        spiece_key_count=sum(1 for key in keys if key == "spiece_model"),
        layer_count=(max(layer_indices) + 1 if layer_indices else 0),
        first_text_keys=first_text_keys,
        first_text_shapes=first_text_shapes,
        config_paths=resolve_text_encoder_config_paths(config_dir),
    )

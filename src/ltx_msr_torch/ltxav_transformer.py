from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from safetensors import safe_open


DIFFUSION_PREFIX = "model.diffusion_model."


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class LTXAVTransformerConfig:
    image_model: str
    num_layers: int
    in_channels: int
    out_channels: int
    cross_attention_dim: int
    attention_head_dim: int
    num_attention_heads: int
    caption_channels: int
    audio_cross_attention_dim: int
    audio_attention_head_dim: int
    audio_num_attention_heads: int
    use_audio_video_cross_attention: bool
    use_embeddings_connector: bool
    connector_num_layers: int
    connector_num_learnable_registers: int
    cross_attention_adaln: bool
    apply_gated_attention: bool
    rope_type: str
    frequencies_precision: str


@dataclass(frozen=True)
class LTXAVTransformerManifest:
    path: Path
    key_count: int
    group_counts: tuple[tuple[str, int], ...]
    block_count: int
    block_key_count: int
    keys_per_block: int
    config: LTXAVTransformerConfig
    specs: Mapping[str, TensorSpec]


def _dtype_name(dtype: object) -> str:
    return str(dtype).replace("F", "float").replace("B", "b")


def _tensor_spec(handle, key: str) -> TensorSpec:
    tensor_slice = handle.get_slice(key)
    return TensorSpec(shape=tuple(tensor_slice.get_shape()), dtype=_dtype_name(tensor_slice.get_dtype()))


def _count_transformer_blocks(keys: set[str]) -> int:
    pattern = re.compile(rf"^{re.escape(DIFFUSION_PREFIX)}transformer_blocks\.(\d+)\.")
    indices = {int(match.group(1)) for key in keys if (match := pattern.match(key))}
    return max(indices) + 1 if indices else 0


def _load_metadata_transformer_config(metadata: dict[str, str] | None) -> dict[str, object]:
    if not metadata or "config" not in metadata:
        return {}
    return json.loads(metadata["config"]).get("transformer", {})


def infer_ltxav_transformer_config(
    keys: set[str],
    specs: Mapping[str, TensorSpec],
    *,
    metadata_config: Mapping[str, object] | None = None,
) -> LTXAVTransformerConfig:
    prefix = DIFFUSION_PREFIX
    if f"{prefix}adaln_single.emb.timestep_embedder.linear_1.bias" not in keys:
        raise ValueError("checkpoint does not look like an LTXV/LTXAV transformer")
    image_model = "ltxav" if f"{prefix}audio_adaln_single.linear.weight" in keys else "ltxv"
    attn2_key = f"{prefix}transformer_blocks.0.attn2.to_k.weight"
    attn2_shape = specs[attn2_key].shape
    detected = {
        "image_model": image_model,
        "num_layers": _count_transformer_blocks(keys),
        "attention_head_dim": attn2_shape[0] // 32,
        "cross_attention_dim": attn2_shape[1],
    }
    merged = {**detected, **dict(metadata_config or {})}
    return LTXAVTransformerConfig(
        image_model=str(merged["image_model"]),
        num_layers=int(merged["num_layers"]),
        in_channels=int(merged["in_channels"]),
        out_channels=int(merged["out_channels"]),
        cross_attention_dim=int(merged["cross_attention_dim"]),
        attention_head_dim=int(merged["attention_head_dim"]),
        num_attention_heads=int(merged["num_attention_heads"]),
        caption_channels=int(merged["caption_channels"]),
        audio_cross_attention_dim=int(merged["audio_cross_attention_dim"]),
        audio_attention_head_dim=int(merged["audio_attention_head_dim"]),
        audio_num_attention_heads=int(merged["audio_num_attention_heads"]),
        use_audio_video_cross_attention=bool(merged["use_audio_video_cross_attention"]),
        use_embeddings_connector=bool(merged["use_embeddings_connector"]),
        connector_num_layers=int(merged["connector_num_layers"]),
        connector_num_learnable_registers=int(merged["connector_num_learnable_registers"]),
        cross_attention_adaln=bool(merged["cross_attention_adaln"]),
        apply_gated_attention=bool(merged["apply_gated_attention"]),
        rope_type=str(merged["rope_type"]),
        frequencies_precision=str(merged["frequencies_precision"]),
    )


def inspect_ltxav_transformer_manifest(checkpoint_path: str | Path) -> LTXAVTransformerManifest:
    path = Path(checkpoint_path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = tuple(key for key in handle.keys() if key.startswith(DIFFUSION_PREFIX))
        key_set = set(keys)
        groups = Counter(key[len(DIFFUSION_PREFIX) :].split(".", 1)[0] for key in keys)
        spec_keys = (
            f"{DIFFUSION_PREFIX}patchify_proj.weight",
            f"{DIFFUSION_PREFIX}proj_out.weight",
            f"{DIFFUSION_PREFIX}audio_patchify_proj.weight",
            f"{DIFFUSION_PREFIX}audio_proj_out.weight",
            f"{DIFFUSION_PREFIX}transformer_blocks.0.attn2.to_k.weight",
            f"{DIFFUSION_PREFIX}transformer_blocks.0.audio_attn2.to_k.weight",
            f"{DIFFUSION_PREFIX}video_embeddings_connector.learnable_registers",
            f"{DIFFUSION_PREFIX}audio_embeddings_connector.learnable_registers",
        )
        specs = {key: _tensor_spec(handle, key) for key in spec_keys}
        metadata_config = _load_metadata_transformer_config(handle.metadata())
    block_count = _count_transformer_blocks(key_set)
    block_key_count = groups["transformer_blocks"]
    return LTXAVTransformerManifest(
        path=path,
        key_count=len(keys),
        group_counts=tuple(sorted(groups.items())),
        block_count=block_count,
        block_key_count=block_key_count,
        keys_per_block=block_key_count // block_count if block_count else 0,
        config=infer_ltxav_transformer_config(key_set, specs, metadata_config=metadata_config),
        specs=specs,
    )

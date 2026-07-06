from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open


DEFAULT_COMFYUI_LORA_DIRS: tuple[Path, ...] = (
    Path("/home/xingshen/ComfyUI/models/loras"),
)


@dataclass(frozen=True)
class LocalICLoRALoadResult:
    lora_name: str
    lora_path: Path
    metadata: dict[str, str] | None
    latent_downscale_factor: float
    strength_model: float


@dataclass(frozen=True)
class LoRAPairManifest:
    prefix: str
    target_key: str
    lora_a_key: str
    lora_b_key: str
    lora_a_shape: tuple[int, ...]
    lora_b_shape: tuple[int, ...]
    rank: int
    alpha: float | None


@dataclass(frozen=True)
class LoRAManifest:
    path: Path
    metadata: dict[str, str] | None
    key_count: int
    pair_count: int
    pairs: tuple[LoRAPairManifest, ...]
    unpaired_keys: tuple[str, ...]


def resolve_lora_path(
    lora_name: str,
    search_dirs: tuple[Path, ...] = DEFAULT_COMFYUI_LORA_DIRS,
) -> Path:
    """Resolve a ComfyUI-style LoRA name to a filesystem path.

    ComfyUI accepts relative names under registered `loras` folders. This
    local resolver mirrors the needed behavior for the MSR workflow and also
    tolerates Windows-style separators stored in workflow JSON.
    """
    candidate = Path(lora_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for root in search_dirs:
        path = root / lora_name
        if path.exists():
            return path

    normalized = lora_name.replace("\\", "/")
    for root in search_dirs:
        path = root / normalized
        if path.exists():
            return path

    raise FileNotFoundError(f"LoRA not found: {lora_name}")


def read_safetensors_metadata(path: str | Path) -> dict[str, str] | None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.metadata()


def inspect_lora_manifest(path: str | Path) -> LoRAManifest:
    """Inspect LoRA tensor pairs without applying them to a model."""
    resolved = Path(path)
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()
        key_set = set(keys)
        pairs: list[LoRAPairManifest] = []
        loaded_pair_keys: set[str] = set()
        for key in keys:
            suffix = ".lora_A.weight"
            if not key.endswith(suffix):
                continue
            prefix = key[: -len(suffix)]
            b_key = f"{prefix}.lora_B.weight"
            if b_key not in key_set:
                continue
            a_shape = tuple(handle.get_tensor(key).shape)
            b_shape = tuple(handle.get_tensor(b_key).shape)
            if len(a_shape) < 2 or len(b_shape) < 2:
                raise ValueError(f"LoRA pair must have matrix-like tensors: {prefix}")
            if a_shape[0] != b_shape[1]:
                raise ValueError(
                    f"LoRA rank mismatch for {prefix}: A{a_shape} cannot pair with B{b_shape}"
                )
            alpha_key = f"{prefix}.alpha"
            alpha = None
            if alpha_key in key_set:
                alpha = float(handle.get_tensor(alpha_key).item())
                loaded_pair_keys.add(alpha_key)
            pairs.append(
                LoRAPairManifest(
                    prefix=prefix,
                    target_key=f"{prefix}.weight",
                    lora_a_key=key,
                    lora_b_key=b_key,
                    lora_a_shape=a_shape,
                    lora_b_shape=b_shape,
                    rank=int(a_shape[0]),
                    alpha=alpha,
                )
            )
            loaded_pair_keys.add(key)
            loaded_pair_keys.add(b_key)

    return LoRAManifest(
        path=resolved,
        metadata=metadata,
        key_count=len(keys),
        pair_count=len(pairs),
        pairs=tuple(pairs),
        unpaired_keys=tuple(key for key in keys if key not in loaded_pair_keys),
    )


def extract_reference_downscale_factor(
    metadata: dict[str, str] | None,
    *,
    lora_path: str | Path | None = None,
) -> float:
    try:
        if metadata is None:
            raise KeyError("metadata")
        return float(metadata["reference_downscale_factor"])
    except (KeyError, ValueError, TypeError):
        if lora_path is not None:
            logging.warning(
                "Failed to extract reference_downscale_factor from metadata for %s, using 1.0",
                lora_path,
            )
        return 1.0


def inspect_ic_lora_model_only(
    lora_name: str,
    strength_model: float,
    search_dirs: tuple[Path, ...] = DEFAULT_COMFYUI_LORA_DIRS,
) -> LocalICLoRALoadResult:
    """Local metadata-only equivalent of `LTXICLoRALoaderModelOnly`.

    This intentionally does not apply LoRA weights yet. It resolves the LoRA,
    reads metadata, and returns the same `latent_downscale_factor` value used
    by the ComfyUI node.
    """
    lora_path = resolve_lora_path(lora_name, search_dirs=search_dirs)
    metadata = read_safetensors_metadata(lora_path)
    latent_downscale_factor = extract_reference_downscale_factor(
        metadata,
        lora_path=lora_path,
    )
    return LocalICLoRALoadResult(
        lora_name=lora_name,
        lora_path=lora_path,
        metadata=metadata,
        latent_downscale_factor=latent_downscale_factor,
        strength_model=strength_model,
    )

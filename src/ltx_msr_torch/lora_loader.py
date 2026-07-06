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

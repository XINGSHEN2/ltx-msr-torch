from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .lora_loader import resolve_lora_path
from .workflow_config import WorkflowConfig


DEFAULT_COMFYUI_MODEL_DIRS: dict[str, tuple[Path, ...]] = {
    "checkpoints": (Path("/home/xingshen/ComfyUI/models/checkpoints"),),
    "text_encoders": (Path("/home/xingshen/ComfyUI/models/text_encoders"),),
    "loras": (Path("/home/xingshen/ComfyUI/models/loras"),),
}


@dataclass(frozen=True)
class LocalModelPaths:
    checkpoint: Path
    text_encoder: Path
    lora: Path
    audio_vae_checkpoint: Path


def resolve_model_path(
    kind: str,
    name: str,
    model_dirs: dict[str, tuple[Path, ...]] = DEFAULT_COMFYUI_MODEL_DIRS,
) -> Path:
    """Resolve a ComfyUI model filename under a model category."""
    if kind == "loras":
        return resolve_lora_path(name, search_dirs=model_dirs[kind])

    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_dirs = model_dirs.get(kind)
    if search_dirs is None:
        raise KeyError(f"unknown model kind: {kind}")

    for root in search_dirs:
        path = root / name
        if path.exists():
            return path

    normalized = name.replace("\\", "/")
    for root in search_dirs:
        path = root / normalized
        if path.exists():
            return path

    raise FileNotFoundError(f"{kind} model not found: {name}")


def resolve_workflow_model_paths(config: WorkflowConfig) -> LocalModelPaths:
    checkpoint = resolve_model_path("checkpoints", config.model.checkpoint)
    text_encoder = resolve_model_path("text_encoders", config.model.text_encoder)
    lora = resolve_model_path("loras", config.model.lora)
    return LocalModelPaths(
        checkpoint=checkpoint,
        text_encoder=text_encoder,
        lora=lora,
        audio_vae_checkpoint=checkpoint,
    )


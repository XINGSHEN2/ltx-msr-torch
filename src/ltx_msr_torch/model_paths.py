from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .lora_loader import resolve_lora_path
from .runtime_paths import model_dirs as default_model_dirs
from .workflow_config import WorkflowConfig


@dataclass(frozen=True)
class LocalModelPaths:
    checkpoint: Path
    text_encoder: Path
    lora: Path
    audio_vae_checkpoint: Path


def resolve_model_path(
    kind: str,
    name: str,
    model_dirs: dict[str, tuple[Path, ...]] | None = None,
) -> Path:
    """Resolve a ComfyUI model filename under a model category."""
    search_roots = model_dirs if model_dirs is not None else default_model_dirs()
    if kind == "loras":
        return resolve_lora_path(name, search_dirs=search_roots[kind])

    candidate = Path(name)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    search_dirs = search_roots.get(kind)
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

    searched = ", ".join(str(root) for root in search_dirs)
    raise FileNotFoundError(
        f"{kind} model not found: {name}; searched: {searched}. "
        "Set LTX_MSR_MODEL_ROOT to use another model directory."
    )


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

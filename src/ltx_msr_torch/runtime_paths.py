from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _environment_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def model_root() -> Path:
    """Return the model root, defaulting to the checkout's models directory."""
    return _environment_path("LTX_MSR_MODEL_ROOT", PROJECT_ROOT / "models")


def model_dirs() -> dict[str, tuple[Path, ...]]:
    root = model_root()
    return {
        "checkpoints": (root / "checkpoints",),
        "text_encoders": (root / "text_encoders",),
        "loras": (root / "loras",),
    }


def gemma_config_dir() -> Path:
    return _environment_path(
        "LTX_MSR_GEMMA_CONFIG_DIR",
        model_root() / "gemma_configs",
    )


def comfyui_root() -> Path | None:
    value = os.environ.get("COMFYUI_ROOT")
    return Path(value).expanduser() if value else None

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open

from .model_paths import LocalModelPaths


@dataclass(frozen=True)
class SafetensorsInspection:
    path: Path
    key_count: int
    first_keys: tuple[str, ...]
    metadata: dict[str, str] | None


@dataclass(frozen=True)
class WorkflowModelInspection:
    checkpoint: SafetensorsInspection
    text_encoder: SafetensorsInspection
    lora: SafetensorsInspection


def inspect_safetensors_header(
    path: str | Path,
    *,
    first_key_count: int = 8,
) -> SafetensorsInspection:
    resolved = Path(path)
    with safe_open(str(resolved), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()
    return SafetensorsInspection(
        path=resolved,
        key_count=len(keys),
        first_keys=tuple(keys[:first_key_count]),
        metadata=metadata,
    )


def inspect_workflow_model_headers(paths: LocalModelPaths) -> WorkflowModelInspection:
    return WorkflowModelInspection(
        checkpoint=inspect_safetensors_header(paths.checkpoint),
        text_encoder=inspect_safetensors_header(paths.text_encoder),
        lora=inspect_safetensors_header(paths.lora),
    )


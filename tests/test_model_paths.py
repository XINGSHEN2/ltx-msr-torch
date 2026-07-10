from pathlib import Path

from ltx_msr_torch.model_paths import resolve_model_path, resolve_workflow_model_paths
from ltx_msr_torch.runtime_paths import model_root
from ltx_msr_torch.workflow_config import default_workflow_config


def _create_model_layout(root: Path) -> None:
    for relative in (
        "checkpoints/ltx-2.3-22b-distilled-1.1.safetensors",
        "text_encoders/gemma_3_12B_it.safetensors",
        "loras/LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_resolve_model_path_finds_workflow_models(tmp_path, monkeypatch):
    _create_model_layout(tmp_path)
    monkeypatch.setenv("LTX_MSR_MODEL_ROOT", str(tmp_path))

    assert resolve_model_path(
        "checkpoints",
        "ltx-2.3-22b-distilled-1.1.safetensors",
    ).exists()
    assert resolve_model_path(
        "text_encoders",
        "gemma_3_12B_it.safetensors",
    ).exists()
    assert resolve_model_path(
        "loras",
        "LTX-2.3\\LTX-2.3-Licon-MSR-V1.safetensors",
    ).exists()


def test_resolve_workflow_model_paths_uses_configured_root(tmp_path, monkeypatch):
    _create_model_layout(tmp_path)
    monkeypatch.setenv("LTX_MSR_MODEL_ROOT", str(tmp_path))
    paths = resolve_workflow_model_paths(default_workflow_config())

    assert model_root() == tmp_path
    assert paths.checkpoint.name == "ltx-2.3-22b-distilled-1.1.safetensors"
    assert paths.text_encoder.name == "gemma_3_12B_it.safetensors"
    assert paths.lora.name == "LTX-2.3-Licon-MSR-V1.safetensors"
    assert paths.audio_vae_checkpoint == paths.checkpoint

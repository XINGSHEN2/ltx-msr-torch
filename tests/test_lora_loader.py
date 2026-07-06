from pathlib import Path

from ltx_msr_torch.lora_loader import (
    extract_reference_downscale_factor,
    inspect_ic_lora_model_only,
    resolve_lora_path,
)


def test_extract_reference_downscale_factor_matches_comfy_fallback():
    assert extract_reference_downscale_factor({"reference_downscale_factor": "2"}) == 2.0
    assert extract_reference_downscale_factor({}) == 1.0
    assert extract_reference_downscale_factor(None) == 1.0
    assert extract_reference_downscale_factor({"reference_downscale_factor": "bad"}) == 1.0


def test_resolve_lora_path_normalizes_workflow_separator():
    path = resolve_lora_path("LTX-2.3\\LTX-2.3-Licon-MSR-V1.safetensors")

    assert path.exists()
    assert path.name in {
        "LTX-2.3-Licon-MSR-V1.safetensors",
        "LTX-2.3\\LTX-2.3-Licon-MSR-V1.safetensors",
    }


def test_inspect_ic_lora_model_only_reads_metadata_without_loading_weights():
    result = inspect_ic_lora_model_only(
        "LTX-2.3\\LTX-2.3-Licon-MSR-V1.safetensors",
        strength_model=1.0,
    )

    assert result.lora_path.exists()
    assert result.strength_model == 1.0
    assert result.latent_downscale_factor == 1.0
    assert result.metadata is not None


def test_resolve_lora_path_raises_for_missing_lora(tmp_path: Path):
    try:
        resolve_lora_path("missing.safetensors", search_dirs=(tmp_path,))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")

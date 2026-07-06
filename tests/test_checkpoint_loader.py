import torch
from safetensors.torch import save_file

from ltx_msr_torch.checkpoint_loader import (
    apply_lora_to_checkpoint_subset,
    inspect_checkpoint_manifest,
    load_safetensors_subset,
    strip_prefix_from_state_dict,
)
from ltx_msr_torch.lora_loader import LoRAManifest, LoRAPairManifest, inspect_lora_manifest, resolve_lora_path
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def test_inspect_checkpoint_manifest_finds_workflow_sections():
    paths = resolve_workflow_model_paths(default_workflow_config())
    manifest = inspect_checkpoint_manifest(paths.checkpoint)
    counts = {section.name: section.key_count for section in manifest.sections}

    assert manifest.key_count == 5947
    assert counts["model"] == 4444
    assert counts["vae"] == 170
    assert counts["audio_vae"] == 102
    assert counts["vocoder"] == 1227
    assert counts["text_embedding_projection"] == 4
    assert manifest.unknown_keys == ()


def test_load_safetensors_subset_reads_only_requested_keys():
    paths = resolve_workflow_model_paths(default_workflow_config())
    state = load_safetensors_subset(
        paths.checkpoint,
        [
            "text_embedding_projection.audio_aggregate_embed.bias",
            "text_embedding_projection.video_aggregate_embed.bias",
        ],
    )

    assert set(state) == {
        "text_embedding_projection.audio_aggregate_embed.bias",
        "text_embedding_projection.video_aggregate_embed.bias",
    }
    assert all(isinstance(value, torch.Tensor) for value in state.values())


def test_strip_prefix_from_state_dict_removes_raw_checkpoint_model_prefix():
    state = {"model.diffusion_model.block.weight": torch.zeros(1), "vae.block.weight": torch.ones(1)}

    stripped = strip_prefix_from_state_dict(state, "model")

    assert "diffusion_model.block.weight" in stripped
    assert "vae.block.weight" in stripped


def test_workflow_lora_targets_all_match_checkpoint_subset():
    paths = resolve_workflow_model_paths(default_workflow_config())
    lora_path = resolve_lora_path(default_workflow_config().model.lora)
    manifest = inspect_lora_manifest(lora_path)
    result = apply_lora_to_checkpoint_subset(
        paths.checkpoint,
        lora_path=lora_path,
        manifest=manifest,
        strength=0.0,
    )

    assert result.report_matched == 480
    assert result.report_skipped == 0
    assert len(result.state_dict) == 480


def test_apply_lora_to_checkpoint_subset_updates_synthetic_file(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.safetensors"
    lora_path = tmp_path / "lora.safetensors"
    save_file({"model.diffusion_model.block.weight": torch.zeros(2, 2)}, str(checkpoint_path))
    save_file(
        {
            "diffusion_model.block.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "diffusion_model.block.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        },
        str(lora_path),
    )
    pair = LoRAPairManifest(
        prefix="diffusion_model.block",
        target_key="diffusion_model.block.weight",
        lora_a_key="diffusion_model.block.lora_A.weight",
        lora_b_key="diffusion_model.block.lora_B.weight",
        lora_a_shape=(1, 2),
        lora_b_shape=(2, 1),
        rank=1,
        alpha=None,
    )
    manifest = LoRAManifest(
        path=lora_path,
        metadata=None,
        key_count=2,
        pair_count=1,
        pairs=(pair,),
        unpaired_keys=(),
    )

    result = apply_lora_to_checkpoint_subset(
        checkpoint_path,
        lora_path=lora_path,
        manifest=manifest,
        strength=1.0,
    )

    assert result.report_matched == 1
    assert torch.equal(
        result.state_dict["model.diffusion_model.block.weight"],
        torch.tensor([[3.0, 6.0], [4.0, 8.0]]),
    )

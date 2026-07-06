import torch
from safetensors.torch import save_file

from ltx_msr_torch.lora_apply import (
    apply_lora_to_state_dict,
    lora_pair_delta,
    match_lora_targets,
    target_key_candidates,
)
from ltx_msr_torch.lora_loader import LoRAManifest, LoRAPairManifest


def test_target_key_candidates_support_raw_checkpoint_prefix():
    assert target_key_candidates("diffusion_model.block.weight") == (
        "diffusion_model.block.weight",
        "model.diffusion_model.block.weight",
    )


def test_match_lora_targets_accepts_model_prefixed_checkpoint_keys():
    pair = LoRAPairManifest(
        prefix="diffusion_model.block",
        target_key="diffusion_model.block.weight",
        lora_a_key="diffusion_model.block.lora_A.weight",
        lora_b_key="diffusion_model.block.lora_B.weight",
        lora_a_shape=(1, 2),
        lora_b_shape=(3, 1),
        rank=1,
        alpha=None,
    )
    manifest = LoRAManifest(
        path=None,  # type: ignore[arg-type]
        metadata=None,
        key_count=2,
        pair_count=1,
        pairs=(pair,),
        unpaired_keys=(),
    )

    matches = match_lora_targets({"model.diffusion_model.block.weight"}, manifest)

    assert matches[0].state_key == "model.diffusion_model.block.weight"


def test_lora_pair_delta_matches_comfy_formula_without_alpha():
    lora_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lora_b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    delta = lora_pair_delta(lora_a, lora_b, (2, 2), alpha=None, strength=0.5)

    assert torch.equal(delta, torch.mm(lora_b, lora_a) * 0.5)


def test_lora_pair_delta_scales_alpha_by_rank():
    lora_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    lora_b = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
    delta = lora_pair_delta(lora_a, lora_b, (2, 2), alpha=1.0, strength=2.0)

    assert torch.equal(delta, torch.mm(lora_b, lora_a))


def test_apply_lora_to_state_dict_adds_matmul_delta(tmp_path):
    lora_path = tmp_path / "test.safetensors"
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

    updated, report = apply_lora_to_state_dict(
        {"model.diffusion_model.block.weight": torch.zeros(2, 2)},
        lora_path=lora_path,
        manifest=manifest,
        strength=1.0,
    )

    assert report.matched == 1
    assert report.skipped == 0
    assert torch.equal(
        updated["model.diffusion_model.block.weight"],
        torch.tensor([[3.0, 6.0], [4.0, 8.0]]),
    )

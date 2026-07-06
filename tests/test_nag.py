import torch

from ltx_msr_torch.nag import build_ltx2_nag_patch_plan, normalized_attention_guidance


def test_normalized_attention_guidance_matches_formula_without_clipping():
    pos = torch.tensor([[[2.0, 1.0]]])
    neg = torch.tensor([[[1.0, 1.0]]])

    out = normalized_attention_guidance(pos, neg, scale=2.0, alpha=0.25, tau=10.0)
    guided = pos * 2.0 - neg
    expected = guided * 0.25 + pos * 0.75

    assert torch.allclose(out, expected)


def test_normalized_attention_guidance_clips_by_tau():
    pos = torch.tensor([[[1.0, 1.0]]])
    neg = torch.tensor([[[-10.0, -10.0]]])

    out = normalized_attention_guidance(pos, neg, scale=2.0, alpha=1.0, tau=2.5)

    assert torch.allclose(torch.norm(out, p=1, dim=-1), torch.tensor([[5.0]]), atol=1e-5)


def test_normalized_attention_guidance_inplace_mode_matches_non_inplace_result():
    pos = torch.randn(2, 3, 4)
    neg = torch.randn(2, 3, 4)

    out = normalized_attention_guidance(pos, neg, scale=11.0, alpha=0.25, tau=2.5, inplace=False)
    out_inplace = normalized_attention_guidance(pos, neg, scale=11.0, alpha=0.25, tau=2.5, inplace=True)

    assert torch.allclose(out, out_inplace)


def test_build_ltx2_nag_patch_plan_uses_workflow_targets():
    plan = build_ltx2_nag_patch_plan(
        scale=11.0,
        alpha=0.25,
        tau=2.5,
        inplace=True,
        transformer_block_count=48,
        has_video_conditioning=True,
        has_audio_conditioning=False,
    )

    assert plan.patch_video is True
    assert plan.patch_audio is False
    assert len(plan.video_patch_targets) == 48
    assert plan.video_patch_targets[0] == "diffusion_model.transformer_blocks.0.attn2.forward"


def test_build_ltx2_nag_patch_plan_disables_when_scale_zero():
    plan = build_ltx2_nag_patch_plan(
        scale=0.0,
        alpha=0.25,
        tau=2.5,
        inplace=True,
        transformer_block_count=48,
        has_video_conditioning=True,
        has_audio_conditioning=True,
    )

    assert plan.patch_video is False
    assert plan.patch_audio is False
    assert plan.video_patch_targets == ()
    assert plan.audio_patch_targets == ()


def test_build_ltx2_nag_patch_plan_uses_kjnodes_audio_target_names():
    plan = build_ltx2_nag_patch_plan(
        scale=11.0,
        alpha=0.25,
        tau=2.5,
        inplace=True,
        transformer_block_count=2,
        has_video_conditioning=False,
        has_audio_conditioning=True,
    )

    assert plan.audio_patch_targets == (
        "diffusion_model.transformer_blocks.0.audio_attn2.forward",
        "diffusion_model.transformer_blocks.1.audio_attn2.forward",
    )

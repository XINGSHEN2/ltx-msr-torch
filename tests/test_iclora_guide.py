from ltx_msr_torch.iclora_guide import plan_iclora_video_guide


def test_plan_iclora_video_guide_matches_project_sample_shape():
    plan = plan_iclora_video_guide(
        latent_shape=(1, 128, 46, 22, 40),
        image_frame_count=41,
        scale_factors=(8, 32, 32),
        frame_idx=0,
        latent_downscale_factor=1.0,
    )

    assert plan.num_frames_to_keep == 41
    assert plan.causal_fix is True
    assert plan.encode_frame_count == 41
    assert plan.effective_guide_frame_count == 6
    assert plan.target_width == 1280
    assert plan.target_height == 704
    assert plan.latent_idx == 0
    assert plan.estimated_guide_latent_shape == (6, 22, 40)
    assert plan.estimated_tokens_added == 5280


def test_plan_iclora_video_guide_aligns_nonzero_frame_idx():
    plan = plan_iclora_video_guide(
        latent_shape=(1, 128, 46, 22, 40),
        image_frame_count=41,
        scale_factors=(8, 32, 32),
        frame_idx=10,
        latent_downscale_factor=1.0,
    )

    assert plan.frame_idx == 9
    assert plan.latent_idx == 2
    assert plan.causal_fix is False
    assert plan.encode_frame_count == 42
    assert plan.effective_guide_frame_count == 5


def test_plan_iclora_video_guide_rejects_overflow():
    try:
        plan_iclora_video_guide(
            latent_shape=(1, 128, 5, 22, 40),
            image_frame_count=41,
            scale_factors=(8, 32, 32),
            frame_idx=1,
            latent_downscale_factor=1.0,
        )
    except AssertionError as error:
        assert "Conditioning frames exceed" in str(error)
    else:
        raise AssertionError("expected guide overflow to raise")

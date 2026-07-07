import torch

from ltx_msr_torch.iclora_guide import (
    append_iclora_keyframe,
    prepare_and_append_iclora_video_guide,
    plan_iclora_video_guide,
)


class _FakeGuideVideoVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.seen = None

    def encode(self, pixels):
        self.seen = pixels
        batch, _, frames, height, width = pixels.shape
        return torch.ones(batch, 128, ((frames - 1) // 8) + 1, height // 32, width // 32)


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


def test_append_iclora_keyframe_appends_latent_and_conditioning_metadata():
    positive = [[torch.zeros(1, 2, 4), {}]]
    negative = [[torch.zeros(1, 2, 4), {}]]
    latent = {"samples": torch.zeros(1, 128, 3, 2, 2)}
    guide = torch.ones(1, 128, 1, 2, 2)

    result = append_iclora_keyframe(
        positive=positive,
        negative=negative,
        latent=latent,
        guiding_latent=guide,
        frame_idx=0,
        strength=0.75,
        scale_factors=(8, 32, 32),
    )

    assert result.latent["samples"].shape == (1, 128, 4, 2, 2)
    assert torch.equal(result.latent["samples"][:, :, 3:], guide)
    assert result.latent["noise_mask"].shape == (1, 1, 4, 1, 1)
    assert torch.allclose(result.latent["noise_mask"][:, :, :3], torch.ones(1, 1, 3, 1, 1))
    assert torch.allclose(result.latent["noise_mask"][:, :, 3:], torch.full((1, 1, 1, 1, 1), 0.25))
    positive_meta = result.positive[0][1]
    negative_meta = result.negative[0][1]
    assert positive_meta["keyframe_idxs"].shape == (1, 3, 4, 2)
    assert negative_meta["keyframe_idxs"].shape == (1, 3, 4, 2)
    assert positive_meta["guide_attention_entries"] == [
        {"pre_filter_count": 4, "strength": 1.0, "pixel_mask": None, "latent_shape": [1, 2, 2]}
    ]
    assert result.tokens_added == 4
    assert result.guide_orig_shape == (1, 2, 2)


def test_append_iclora_keyframe_extends_existing_keyframe_metadata():
    positive = [[torch.zeros(1, 2, 4), {}]]
    negative = [[torch.zeros(1, 2, 4), {}]]
    latent = {"samples": torch.zeros(1, 128, 4, 2, 2)}
    first = append_iclora_keyframe(
        positive=positive,
        negative=negative,
        latent=latent,
        guiding_latent=torch.ones(1, 128, 1, 2, 2),
        frame_idx=0,
        strength=1.0,
    )

    second = append_iclora_keyframe(
        positive=first.positive,
        negative=first.negative,
        latent=first.latent,
        guiding_latent=torch.ones(1, 128, 1, 2, 2) * 2,
        frame_idx=8,
        strength=1.0,
    )

    assert second.positive[0][1]["keyframe_idxs"].shape == (1, 3, 8, 2)
    assert len(second.positive[0][1]["guide_attention_entries"]) == 2
    assert second.latent["samples"].shape == (1, 128, 6, 2, 2)


def test_append_iclora_keyframe_aligns_nonzero_frame_index():
    result = append_iclora_keyframe(
        positive=[[torch.zeros(1, 2, 4), {}]],
        negative=[[torch.zeros(1, 2, 4), {}]],
        latent={"samples": torch.zeros(1, 128, 8, 1, 1)},
        guiding_latent=torch.ones(1, 128, 2, 1, 1),
        frame_idx=10,
        strength=1.0,
    )

    assert result.frame_idx == 9
    assert result.latent_idx == 2
    assert result.positive[0][1]["keyframe_idxs"][0, 0, 0, 0].item() == 9


def test_prepare_and_append_iclora_video_guide_encodes_and_appends():
    vae = _FakeGuideVideoVAE()
    result = prepare_and_append_iclora_video_guide(
        video_vae=vae,
        positive=[[torch.zeros(1, 2, 4), {}]],
        negative=[[torch.zeros(1, 2, 4), {}]],
        latent={"samples": torch.zeros(1, 128, 8, 2, 2)},
        image=torch.zeros(9, 20, 30, 3),
        frame_idx=0,
        strength=1.0,
    )

    assert vae.seen.shape == (1, 3, 9, 64, 64)
    assert result.encoded_pixels.shape == (9, 64, 64, 3)
    assert result.guide_latent.shape == (1, 128, 2, 2, 2)
    assert result.append.latent["samples"].shape == (1, 128, 10, 2, 2)
    assert result.append.tokens_added == 8

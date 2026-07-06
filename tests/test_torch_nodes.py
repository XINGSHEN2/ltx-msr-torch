import torch

from ltx_msr_torch.torch_nodes import (
    empty_ltxv_latent_audio,
    empty_ltxv_latent_video,
    int_constant,
    manual_sigmas,
    random_noise,
)


def test_int_constant_returns_value_unchanged():
    assert int_constant(1280) == 1280


def test_manual_sigmas_matches_comfy_regex_and_dtype():
    sigmas = manual_sigmas("1.0, 0.99375, -0.25, +.5, text")

    assert sigmas.dtype == torch.float32
    assert torch.equal(sigmas, torch.FloatTensor([1.0, 0.99375, -0.25, 0.5]))


def test_empty_ltxv_latent_video_shape_matches_comfy_formula():
    latent = empty_ltxv_latent_video(
        width=1280,
        height=720,
        length=361,
        batch_size=1,
        device="cpu",
    )

    assert latent["downscale_ratio_spacial"] == 32
    assert tuple(latent["samples"].shape) == (1, 128, 46, 22, 40)
    assert latent["samples"].device.type == "cpu"
    assert float(latent["samples"].sum()) == 0.0


class _FakeFirstStageAudioModel:
    latent_frequency_bins = 32

    def num_of_latents_from_frames(self, frames_number, frame_rate):
        assert frames_number == 361
        assert frame_rate == 24
        return 42


class _FakeAudioVAE:
    latent_channels = 128
    first_stage_model = _FakeFirstStageAudioModel()


def test_empty_ltxv_latent_audio_uses_audio_vae_config():
    latent = empty_ltxv_latent_audio(
        frames_number=361,
        frame_rate=24,
        batch_size=1,
        audio_vae=_FakeAudioVAE(),
        device="cpu",
    )

    assert latent["type"] == "audio"
    assert tuple(latent["samples"].shape) == (1, 128, 42, 32)
    assert float(latent["samples"].sum()) == 0.0


def test_random_noise_preserves_seed_and_shape_contract():
    latent = {"samples": torch.zeros((1, 4, 2, 3, 5), dtype=torch.float32)}
    noise = random_noise(123)

    first = noise.generate_noise(latent)
    second = random_noise(123).generate_noise(latent)

    assert noise.seed == 123
    assert tuple(first.shape) == (1, 4, 2, 3, 5)
    assert torch.equal(first, second)


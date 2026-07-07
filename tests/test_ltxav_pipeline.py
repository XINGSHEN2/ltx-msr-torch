import torch

from ltx_msr_torch.ltxav_pipeline import decode_ltxav_latents


class _FakeVideoVAE:
    def __init__(self):
        self.calls = []

    def decode(self, latents):
        self.calls.append(latents)
        return latents + 1


class _FakeAudioVAE:
    def __init__(self):
        self.calls = []

    def decode(self, latents):
        self.calls.append(latents)
        return latents.mean(dim=(2, 3))


def test_decode_ltxav_latents_decodes_video_and_audio():
    video_vae = _FakeVideoVAE()
    audio_vae = _FakeAudioVAE()
    video_latents = torch.ones(1, 2, 3, 4, 5)
    audio_latents = torch.ones(1, 2, 6, 3)

    output = decode_ltxav_latents(
        video_vae=video_vae,
        video_latents=video_latents,
        audio_vae=audio_vae,
        audio_latents=audio_latents,
    )

    assert torch.equal(output.video, video_latents + 1)
    assert output.audio is not None
    assert output.audio.shape == (1, 2)
    assert video_vae.calls == [video_latents]
    assert audio_vae.calls == [audio_latents]


def test_decode_ltxav_latents_allows_video_only():
    video_latents = torch.ones(1, 2, 3, 4, 5)

    output = decode_ltxav_latents(video_vae=_FakeVideoVAE(), video_latents=video_latents)

    assert torch.equal(output.video, video_latents + 1)
    assert output.audio is None


def test_decode_ltxav_latents_requires_audio_vae_for_audio_latents():
    try:
        decode_ltxav_latents(
            video_vae=_FakeVideoVAE(),
            video_latents=torch.ones(1, 2, 3, 4, 5),
            audio_latents=torch.ones(1, 2, 6, 3),
        )
    except ValueError as exc:
        assert "audio_vae" in str(exc)
    else:
        raise AssertionError("expected ValueError")

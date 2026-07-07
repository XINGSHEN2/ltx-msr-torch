import torch

from ltx_msr_torch.ltxav_pipeline import decode_ltxav_latents, run_ltxav_sample_decode, sample_ltxav_workflow_latents


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


class _TorchVideoVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.seen_dtype = None

    def decode(self, latents):
        self.seen_dtype = latents.dtype
        return latents


class _FakeDenoiserModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [kwargs["video_latents"] * 0.5, kwargs["audio_latents"] * 0.25]


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


def test_decode_ltxav_latents_casts_to_decoder_parameter_dtype():
    vae = _TorchVideoVAE()

    output = decode_ltxav_latents(video_vae=vae, video_latents=torch.ones(1, 2, 1, 1, 1, dtype=torch.bfloat16))

    assert vae.seen_dtype == torch.float32
    assert output.video.dtype == torch.float32


def test_sample_ltxav_workflow_latents_runs_tuple_euler_sampler():
    model = _FakeDenoiserModel()
    video = torch.ones(1, 2, 1, 1, 1) * 4
    audio = torch.ones(1, 2, 1, 3) * 8
    context = torch.randn(1, 2, 4)
    mask = torch.ones(1, 2, dtype=torch.long)

    sampled_video, sampled_audio = sample_ltxav_workflow_latents(
        model=model,
        video_latents=video,
        audio_latents=audio,
        context=context,
        attention_mask=mask,
        sigmas=torch.tensor([2.0, 1.0, 0.0]),
        frame_rate=24.0,
    )

    assert sampled_video.shape == video.shape
    assert sampled_audio.shape == audio.shape
    assert len(model.calls) == 2
    assert torch.equal(model.calls[0]["context"], context)
    assert torch.equal(model.calls[0]["attention_mask"], mask)


def test_run_ltxav_sample_decode_samples_and_decodes():
    output = run_ltxav_sample_decode(
        model=_FakeDenoiserModel(),
        video_latents=torch.ones(1, 2, 1, 1, 1) * 4,
        audio_latents=torch.ones(1, 2, 1, 3) * 8,
        context=torch.randn(1, 2, 4),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        sigmas=torch.tensor([2.0, 1.0]),
        frame_rate=24.0,
        video_vae=_FakeVideoVAE(),
        audio_vae=_FakeAudioVAE(),
    )

    assert output.video_latents.shape == (1, 2, 1, 1, 1)
    assert output.audio_latents.shape == (1, 2, 1, 3)
    assert output.decoded is not None
    assert output.decoded.video.shape == output.video_latents.shape
    assert output.decoded.audio is not None
    assert output.decoded.audio.shape == (1, 2)

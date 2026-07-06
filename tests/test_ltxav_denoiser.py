import torch

from ltx_msr_torch.ltxav_denoiser import LTXAVDenoiser, sample_ltxav_euler


class _FakeLTXAVModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [kwargs["video_latents"] * 0.5, kwargs["audio_latents"] * 0.25]


def test_ltxav_denoiser_builds_per_token_timesteps_and_returns_tuple():
    model = _FakeLTXAVModel()
    context = torch.randn(1, 5, 16)
    mask = torch.ones(1, 5, dtype=torch.long)
    denoiser = LTXAVDenoiser(model=model, context=context, attention_mask=mask, frame_rate=25.0)
    video = torch.ones(1, 2, 2, 3, 4)
    audio = torch.ones(1, 2, 6, 3)

    output = denoiser((video, audio), torch.tensor([0.7]))

    assert isinstance(output, tuple)
    assert torch.equal(output[0], video * 0.5)
    assert torch.equal(output[1], audio * 0.25)
    call = model.calls[0]
    assert call["context"] is context
    assert call["attention_mask"] is mask
    assert call["frame_rate"] == 25.0
    assert call["timestep"].shape == (1, 24)
    assert call["audio_timestep"].shape == (1, 6)
    assert torch.equal(call["timestep"], torch.full((1, 24), 0.7))
    assert torch.equal(call["audio_timestep"], torch.full((1, 6), 0.7))
    assert call["target_audio_seq_len"] == 6


def test_ltxav_denoiser_supports_scalar_sigma():
    model = _FakeLTXAVModel()
    denoiser = LTXAVDenoiser(model=model, context=torch.randn(2, 5, 16), attention_mask=None, frame_rate=24.0)

    denoiser((torch.ones(2, 2, 1, 1, 1), torch.ones(2, 2, 3, 3)), torch.tensor(0.4))

    call = model.calls[0]
    assert call["timestep"].shape == (2, 1)
    assert call["audio_timestep"].shape == (2, 3)
    assert torch.equal(call["timestep"], torch.full((2, 1), 0.4))


def test_sample_ltxav_euler_runs_adapter_and_tuple_sampler():
    model = _FakeLTXAVModel()
    video = torch.ones(1, 2, 1, 1, 1) * 4
    audio = torch.ones(1, 2, 1, 1) * 8

    out = sample_ltxav_euler(
        model=model,
        video_latents=video,
        audio_latents=audio,
        context=torch.randn(1, 5, 16),
        attention_mask=None,
        frame_rate=25.0,
        sigmas=torch.tensor([2.0, 1.0, 0.0]),
    )

    assert isinstance(out, tuple)
    assert torch.allclose(out[0], torch.ones_like(video) * 1.5)
    assert torch.allclose(out[1], torch.ones_like(audio) * 1.25)
    assert len(model.calls) == 2

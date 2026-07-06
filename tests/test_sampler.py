import torch

from ltx_msr_torch.sampler import (
    append_dims,
    build_sampler_plan,
    euler_step,
    euler_step_latents,
    sample_euler,
    sample_euler_latents,
    to_d,
    to_d_latents,
)


def test_append_dims_matches_k_diffusion_shape_rule():
    value = torch.tensor([1.0, 2.0])

    assert append_dims(value, 4).shape == (2, 1, 1, 1)


def test_to_d_converts_denoised_to_derivative():
    x = torch.tensor([[3.0, 5.0]])
    denoised = torch.tensor([[1.0, 1.0]])
    sigma = torch.tensor([2.0])

    assert torch.equal(to_d(x, sigma, denoised), torch.tensor([[1.0, 2.0]]))


def test_euler_step_matches_k_diffusion_formula():
    x = torch.tensor([[3.0, 5.0]])
    denoised = torch.tensor([[1.0, 1.0]])
    sigma = torch.tensor(2.0)
    sigma_next = torch.tensor(1.0)

    assert torch.equal(euler_step(x, denoised, sigma, sigma_next), torch.tensor([[2.0, 3.0]]))


def test_sample_euler_runs_deterministic_steps():
    sigmas = torch.tensor([2.0, 1.0, 0.0])

    def denoiser(x, sigma):
        assert sigma.shape == (1,)
        return x * 0.5

    out = sample_euler(denoiser, torch.tensor([[4.0]]), sigmas)

    assert torch.allclose(out, torch.tensor([[1.5]]))


def test_to_d_latents_supports_video_audio_tuple():
    x = (torch.tensor([[3.0]]), torch.tensor([[5.0]]))
    denoised = (torch.tensor([[1.0]]), torch.tensor([[1.0]]))
    derivative = to_d_latents(x, torch.tensor([2.0]), denoised)

    assert isinstance(derivative, tuple)
    assert torch.equal(derivative[0], torch.tensor([[1.0]]))
    assert torch.equal(derivative[1], torch.tensor([[2.0]]))


def test_euler_step_latents_updates_each_latent():
    x = (torch.tensor([[3.0]]), torch.tensor([[5.0]]))
    denoised = (torch.tensor([[1.0]]), torch.tensor([[1.0]]))

    out = euler_step_latents(x, denoised, torch.tensor(2.0), torch.tensor(1.0))

    assert isinstance(out, tuple)
    assert torch.equal(out[0], torch.tensor([[2.0]]))
    assert torch.equal(out[1], torch.tensor([[3.0]]))


def test_sample_euler_latents_runs_video_audio_tuple():
    sigmas = torch.tensor([2.0, 1.0, 0.0])

    def denoiser(x, sigma):
        assert sigma.shape == (1,)
        assert isinstance(x, tuple)
        return x[0] * 0.5, x[1] * 0.25

    out = sample_euler_latents(denoiser, (torch.tensor([[4.0]]), torch.tensor([[8.0]])), sigmas)

    assert isinstance(out, tuple)
    assert torch.allclose(out[0], torch.tensor([[1.5]]))
    assert torch.allclose(out[1], torch.tensor([[1.25]]))


def test_build_sampler_plan_uses_workflow_sigmas():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    plan = build_sampler_plan(sampler_name="euler", cfg=1.0, sigmas=sigmas)

    assert plan.sampler_name == "euler"
    assert plan.cfg == 1.0
    assert plan.step_count == 2
    assert plan.sigma_count == 3
    assert plan.first_sigma == 1.0
    assert plan.last_sigma == 0.0

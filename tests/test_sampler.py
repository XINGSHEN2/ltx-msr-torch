import torch

from ltx_msr_torch.sampler import append_dims, build_sampler_plan, euler_step, sample_euler, to_d


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


def test_build_sampler_plan_uses_workflow_sigmas():
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    plan = build_sampler_plan(sampler_name="euler", cfg=1.0, sigmas=sigmas)

    assert plan.sampler_name == "euler"
    assert plan.cfg == 1.0
    assert plan.step_count == 2
    assert plan.sigma_count == 3
    assert plan.first_sigma == 1.0
    assert plan.last_sigma == 0.0

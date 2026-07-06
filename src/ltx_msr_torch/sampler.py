from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SamplerPlan:
    sampler_name: str
    cfg: float
    step_count: int
    sigma_count: int
    first_sigma: float
    last_sigma: float


def append_dims(value: torch.Tensor, target_ndim: int) -> torch.Tensor:
    dims_to_append = target_ndim - value.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {value.ndim} dims but target_ndim is {target_ndim}")
    return value[(...,) + (None,) * dims_to_append]


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    return (x - denoised) / append_dims(sigma, x.ndim)


def euler_step(
    x: torch.Tensor,
    denoised: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    derivative = to_d(x, sigma, denoised)
    return x + derivative * append_dims(sigma_next - sigma, x.ndim)


@torch.no_grad()
def sample_euler(
    denoiser: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    s_in = x.new_ones([x.shape[0]])
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        denoised = denoiser(x, sigma * s_in)
        x = euler_step(x, denoised, sigma, sigmas[index + 1])
    return x


def build_sampler_plan(
    *,
    sampler_name: str,
    cfg: float,
    sigmas: torch.Tensor,
) -> SamplerPlan:
    return SamplerPlan(
        sampler_name=sampler_name,
        cfg=float(cfg),
        step_count=max(int(sigmas.numel()) - 1, 0),
        sigma_count=int(sigmas.numel()),
        first_sigma=float(sigmas[0].item()) if sigmas.numel() else 0.0,
        last_sigma=float(sigmas[-1].item()) if sigmas.numel() else 0.0,
    )

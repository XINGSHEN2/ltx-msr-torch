from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch


Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Latents: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]
LatentDenoiser: TypeAlias = Callable[[Latents, torch.Tensor], Latents]


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


def _latent_zip_map(left: Latents, right: Latents, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Latents:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return fn(left, right)
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(fn(left_value, right_value) for left_value, right_value in zip(left, right))  # type: ignore[return-value]
    raise TypeError("latent structures must both be tensors or same-length tuples")


def _latent_map(value: Latents, fn: Callable[[torch.Tensor], torch.Tensor]) -> Latents:
    if isinstance(value, torch.Tensor):
        return fn(value)
    return tuple(fn(item) for item in value)


def to_d_latents(x: Latents, sigma: torch.Tensor, denoised: Latents) -> Latents:
    return _latent_zip_map(x, denoised, lambda x_item, denoised_item: to_d(x_item, sigma, denoised_item))


def euler_step_latents(
    x: Latents,
    denoised: Latents,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> Latents:
    derivative = to_d_latents(x, sigma, denoised)
    return _latent_zip_map(
        x,
        derivative,
        lambda x_item, derivative_item: x_item + derivative_item * append_dims(sigma_next - sigma, x_item.ndim),
    )


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


@torch.no_grad()
def sample_euler_latents(
    denoiser: LatentDenoiser,
    x: Latents,
    sigmas: torch.Tensor,
) -> Latents:
    first = x if isinstance(x, torch.Tensor) else x[0]
    s_in = first.new_ones([first.shape[0]])
    for index in range(len(sigmas) - 1):
        sigma = sigmas[index]
        denoised = denoiser(x, sigma * s_in)
        x = euler_step_latents(x, denoised, sigma, sigmas[index + 1])
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

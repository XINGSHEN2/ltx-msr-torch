"""Small PyTorch replacements for the ComfyUI helpers used by LTX VAE code."""

from __future__ import annotations

import os

import torch


class _NoInitMixin:
    """Match ComfyUI's disable_weight_init layers without its runtime."""

    def reset_parameters(self) -> None:
        return None


class _Linear(_NoInitMixin, torch.nn.Linear):
    pass


class _Conv1d(_NoInitMixin, torch.nn.Conv1d):
    pass


class _Conv2d(_NoInitMixin, torch.nn.Conv2d):
    pass


class _Conv3d(_NoInitMixin, torch.nn.Conv3d):
    pass


class _ConvTranspose1d(_NoInitMixin, torch.nn.ConvTranspose1d):
    pass


class _GroupNorm(_NoInitMixin, torch.nn.GroupNorm):
    pass


class _LayerNorm(_NoInitMixin, torch.nn.LayerNorm):
    pass


class disable_weight_init:
    Linear = _Linear
    Conv1d = _Conv1d
    Conv2d = _Conv2d
    Conv3d = _Conv3d
    ConvTranspose1d = _ConvTranspose1d
    GroupNorm = _GroupNorm
    LayerNorm = _LayerNorm


def cast_to(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    copy: bool = False,
) -> torch.Tensor:
    """Match the non-streaming path of ``comfy.model_management.cast_to``."""
    target_device = torch.device(device) if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype
    if not copy and tensor.device == target_device and tensor.dtype == target_dtype:
        return tensor
    return tensor.to(device=target_device, dtype=target_dtype, copy=copy)


def get_total_memory(device: torch.device | str) -> int:
    device = torch.device(device)
    if device.type == "cuda":
        return int(torch.cuda.mem_get_info(device)[1])
    if device.type == "mps":
        return _host_memory_bytes()
    return _host_memory_bytes()


def _host_memory_bytes() -> int:
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    return page_size * pages


def intermediate_device() -> torch.device:
    # The previous standalone path initialized ComfyUI with ``--cpu``, for
    # which decoded chunk accumulation happens on CPU.
    return torch.device("cpu")


def torch_cat_if_needed(tensors: list[torch.Tensor | None], dim: int) -> torch.Tensor | None:
    present = [tensor for tensor in tensors if tensor is not None and tensor.shape[dim] > 0]
    if len(present) > 1:
        return torch.cat(present, dim=dim)
    if present:
        return present[0]
    return None

from __future__ import annotations

import math

import numpy as np
import torch


def _log_base(x: float, base: float) -> float:
    return float(np.log(x) / np.log(base))


def get_fractional_positions(indices_grid: torch.Tensor, max_pos: list[int] | tuple[int, ...]) -> torch.Tensor:
    n_pos_dims = indices_grid.shape[1]
    if n_pos_dims != len(max_pos):
        raise AssertionError(f"Number of position dimensions ({n_pos_dims}) must match max_pos length ({len(max_pos)})")
    return torch.stack([indices_grid[:, i] / max_pos[i] for i in range(n_pos_dims)], axis=-1)


def generate_freq_grid_np(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    theta = positional_embedding_theta
    n_elem = 2 * positional_embedding_max_pos_count
    pow_indices = np.power(
        theta,
        np.linspace(
            _log_base(1, theta),
            _log_base(theta, theta),
            inner_dim // n_elem,
            dtype=np.float64,
        ),
    )
    return torch.tensor(pow_indices * math.pi / 2, dtype=torch.float32, device=device)


def generate_freq_grid_pytorch(
    positional_embedding_theta: float,
    positional_embedding_max_pos_count: int,
    inner_dim: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    theta = positional_embedding_theta
    n_elem = 2 * positional_embedding_max_pos_count
    indices = theta ** (
        torch.linspace(
            math.log(1, theta),
            math.log(theta, theta),
            inner_dim // n_elem,
            device=device,
            dtype=torch.float32,
        )
    )
    return indices.to(dtype=torch.float32) * math.pi / 2


def generate_freqs(
    indices: torch.Tensor,
    indices_grid: torch.Tensor,
    max_pos: list[int] | tuple[int, ...],
    use_middle_indices_grid: bool,
) -> torch.Tensor:
    if use_middle_indices_grid:
        if not (len(indices_grid.shape) == 4 and indices_grid.shape[-1] == 2):
            raise AssertionError("middle-index grid expects [B, C, T, 2] coordinates")
        indices_grid = (indices_grid[..., 0] + indices_grid[..., 1]) / 2.0
    elif len(indices_grid.shape) == 4:
        indices_grid = indices_grid[..., 0]
    fractional_positions = get_fractional_positions(indices_grid, max_pos)
    indices = indices.to(device=fractional_positions.device)
    return (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)


def interleaved_freqs_cis(freqs: torch.Tensor, pad_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
    sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(cos_freq[:, :, :pad_size])
        cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
        sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
    return cos_freq, sin_freq


def split_freqs_cis(
    freqs: torch.Tensor,
    pad_size: int,
    num_attention_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_freq = freqs.cos()
    sin_freq = freqs.sin()
    if pad_size != 0:
        cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
        sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])
        cos_freq = torch.concatenate([cos_padding, cos_freq], axis=-1)
        sin_freq = torch.concatenate([sin_padding, sin_freq], axis=-1)
    batch, tokens, half_head_dim = cos_freq.shape
    cos_freq = cos_freq.reshape(batch, tokens, num_attention_heads, half_head_dim // num_attention_heads)
    sin_freq = sin_freq.reshape(batch, tokens, num_attention_heads, half_head_dim // num_attention_heads)
    return torch.swapaxes(cos_freq, 1, 2), torch.swapaxes(sin_freq, 1, 2)


def precompute_ltx_freqs_cis(
    indices_grid: torch.Tensor,
    *,
    dim: int,
    out_dtype: torch.dtype,
    theta: float = 10000.0,
    max_pos: list[int] | tuple[int, ...] = (20, 2048, 2048),
    use_middle_indices_grid: bool = False,
    num_attention_heads: int = 32,
    split: bool = True,
    double_precision_grid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    generator = generate_freq_grid_np if double_precision_grid else generate_freq_grid_pytorch
    indices = generator(theta, indices_grid.shape[1], dim, indices_grid.device)
    freqs = generate_freqs(indices, indices_grid, max_pos, use_middle_indices_grid)
    if split:
        expected_freqs = dim // 2
        pad_size = expected_freqs - freqs.shape[-1]
        cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, num_attention_heads)
    else:
        n_elem = 2 * indices_grid.shape[1]
        cos_freq, sin_freq = interleaved_freqs_cis(freqs, dim % n_elem)
    return cos_freq.to(out_dtype), sin_freq.to(out_dtype), split

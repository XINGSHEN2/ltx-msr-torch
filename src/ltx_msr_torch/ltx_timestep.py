from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint_loader import load_safetensors_subset


ADALN_BASE_PARAMS_COUNT = 6
ADALN_CROSS_ATTN_PARAMS_COUNT = 9


@dataclass(frozen=True)
class AdaLayerNormSingleConfig:
    embedding_dim: int
    embedding_coefficient: int
    timestep_channels: int = 256
    sample_proj_bias: bool = True


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    if len(timesteps.shape) != 1:
        raise AssertionError("Timesteps should be a 1d-array")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = scale * timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(torch.nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: float = 1) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        *,
        out_dim: int | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias, dtype=dtype, device=device)
        self.cond_proj = (
            torch.nn.Linear(cond_proj_dim, in_channels, bias=False, dtype=dtype, device=device)
            if cond_proj_dim is not None
            else None
        )
        self.act = torch.nn.SiLU()
        self.linear_2 = torch.nn.Linear(
            time_embed_dim,
            out_dim if out_dim is not None else time_embed_dim,
            bias=sample_proj_bias,
            dtype=dtype,
            device=device,
        )

    def forward(self, sample: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is not None:
            if self.cond_proj is None:
                raise ValueError("condition was provided but cond_proj_dim is not configured")
            sample = sample + self.cond_proj(condition)
        return self.linear_2(self.act(self.linear_1(sample)))


class PixArtAlphaCombinedTimestepSizeEmbeddings(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.outdim = embedding_dim // 3
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        resolution=None,
        aspect_ratio=None,
        batch_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del resolution, aspect_ratio, batch_size
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))


class AdaLayerNormSingle(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        *,
        embedding_coefficient: int = ADALN_BASE_PARAMS_COUNT,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = AdaLayerNormSingleConfig(
            embedding_dim=embedding_dim,
            embedding_coefficient=embedding_coefficient,
        )
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim, dtype=dtype, device=device)
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        *,
        batch_size: int | None = None,
        hidden_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
        embedded_timestep = self.emb(timestep, **kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class CompressedTimestep:
    def __init__(self, tensor: torch.Tensor, patches_per_frame: int | None, *, per_frame: bool = False) -> None:
        self.batch_size, n_tokens, self.feature_dim = tensor.shape
        if per_frame:
            self.patches_per_frame = patches_per_frame
            self.num_frames = n_tokens
            self.data = tensor
        elif patches_per_frame is not None and n_tokens >= patches_per_frame and n_tokens % patches_per_frame == 0:
            self.patches_per_frame = patches_per_frame
            self.num_frames = n_tokens // patches_per_frame
            self.data = tensor.view(self.batch_size, self.num_frames, patches_per_frame, self.feature_dim)[
                :, :, 0, :
            ].contiguous()
        else:
            self.patches_per_frame = 1
            self.num_frames = n_tokens
            self.data = tensor

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def expand(self) -> torch.Tensor:
        if self.patches_per_frame == 1:
            return self.data
        expanded = self.data.unsqueeze(2).expand(
            self.batch_size,
            self.num_frames,
            self.patches_per_frame,
            self.feature_dim,
        )
        return expanded.reshape(self.batch_size, -1, self.feature_dim)

    def expand_for_computation(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        indices: slice = slice(None, None),
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]
        if self.patches_per_frame == 1:
            token_count = self.data.shape[1]
            reshaped = self.data.reshape(batch_size, token_count, num_ada_params, -1)[:, :, indices, :]
            table_values = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(
                device=self.data.device,
                dtype=self.data.dtype,
            )
            return (table_values + reshaped).unbind(dim=2)

        frame_values = self.data.reshape(batch_size, self.num_frames, num_ada_params, -1)[:, :, indices, :]
        table_values = scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(
            device=self.data.device,
            dtype=self.data.dtype,
        )
        frame_ada = (table_values + frame_values).unbind(dim=2)
        return tuple(
            value.unsqueeze(2).expand(batch_size, self.num_frames, self.patches_per_frame, -1).reshape(
                batch_size,
                -1,
                value.shape[-1],
            )
            for value in frame_ada
        )


def compute_prompt_timestep(
    adaln_module: AdaLayerNormSingle | None,
    timestep_scaled: torch.Tensor,
    *,
    batch_size: int,
    hidden_dtype: torch.dtype,
) -> torch.Tensor | None:
    if adaln_module is None:
        return None
    ts_input = (
        timestep_scaled.max(dim=1, keepdim=True).values.flatten()
        if timestep_scaled.dim() > 1
        else timestep_scaled.flatten()
    )
    prompt_ts, _ = adaln_module(
        ts_input,
        {"resolution": None, "aspect_ratio": None},
        batch_size=batch_size,
        hidden_dtype=hidden_dtype,
    )
    return prompt_ts.view(batch_size, 1, prompt_ts.shape[-1])


def load_adaln_single_state_dict(
    checkpoint_path: str | Path,
    prefix: str,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    full_prefix = f"model.diffusion_model.{prefix}."
    keys = (
        f"{full_prefix}emb.timestep_embedder.linear_1.bias",
        f"{full_prefix}emb.timestep_embedder.linear_1.weight",
        f"{full_prefix}emb.timestep_embedder.linear_2.bias",
        f"{full_prefix}emb.timestep_embedder.linear_2.weight",
        f"{full_prefix}linear.bias",
        f"{full_prefix}linear.weight",
    )
    raw = load_safetensors_subset(checkpoint_path, keys, device=device)
    return {key[len(full_prefix) :]: value for key, value in raw.items()}

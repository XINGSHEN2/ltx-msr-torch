from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint_loader import load_safetensors_subset
from .ltx_attention import CrossAttention, FeedForward, rms_norm
from .ltx_rope import generate_freq_grid_np, interleaved_freqs_cis, split_freqs_cis


@dataclass(frozen=True)
class Embeddings1DConnectorConfig:
    inner_dim: int
    attention_head_dim: int
    num_attention_heads: int
    num_layers: int
    num_learnable_registers: int
    apply_gated_attention: bool
    split_rope: bool
    double_precision_rope: bool


class BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        *,
        apply_gated_attention: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
        )
        self.ff = FeedForward(dim, dim_out=dim, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool] | None = None,
    ) -> torch.Tensor:
        norm_hidden_states = rms_norm(hidden_states).squeeze(1)
        hidden_states = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe) + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        hidden_states = self.ff(rms_norm(hidden_states)) + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class Embeddings1DConnector(torch.nn.Module):
    def __init__(
        self,
        *,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        num_layers: int = 8,
        num_learnable_registers: int = 128,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: tuple[int, ...] = (1,),
        apply_gated_attention: bool = True,
        split_rope: bool = True,
        double_precision_rope: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.split_rope = split_rope
        self.double_precision_rope = double_precision_rope
        self.num_learnable_registers = num_learnable_registers
        self.config = Embeddings1DConnectorConfig(
            inner_dim=self.inner_dim,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            num_learnable_registers=num_learnable_registers,
            apply_gated_attention=apply_gated_attention,
            split_rope=split_rope,
            double_precision_rope=double_precision_rope,
        )
        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                BasicTransformerBlock1D(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    apply_gated_attention=apply_gated_attention,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        if num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.empty(num_learnable_registers, self.inner_dim, dtype=dtype, device=device)
            )

    def get_fractional_positions(self, indices_grid: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [indices_grid[:, i] / self.positional_embedding_max_pos[i] for i in range(1)],
            dim=-1,
        )

    def generate_freq_grid(self, spacing: str, dtype: torch.dtype, device: torch.device | str) -> torch.Tensor:
        dim = self.inner_dim
        theta = self.positional_embedding_theta
        n_elem = 2
        if spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, n_elem, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(1, theta, dim // n_elem, device=device, dtype=dtype)
        elif spacing == "sqrt":
            indices = torch.linspace(1, theta**2, dim // n_elem, device=device, dtype=dtype).sqrt()
        else:
            indices = torch.linspace(1, theta, dim // n_elem, device=device, dtype=dtype)
        return indices * math.pi / 2

    def precompute_freqs(self, indices_grid: torch.Tensor, spacing: str = "exp") -> torch.Tensor:
        source_dtype = indices_grid.dtype
        dtype = torch.float32 if source_dtype in (torch.bfloat16, torch.float16) else source_dtype
        fractional_positions = self.get_fractional_positions(indices_grid)
        if self.double_precision_rope:
            indices = generate_freq_grid_np(
                self.positional_embedding_theta,
                indices_grid.shape[1],
                self.inner_dim,
            ).to(device=fractional_positions.device)
        else:
            indices = self.generate_freq_grid(spacing, dtype, fractional_positions.device)
        if spacing == "exp_2":
            return (indices * fractional_positions.unsqueeze(-1)).transpose(-1, -2).flatten(2)
        return (indices * (fractional_positions.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)

    def precompute_freqs_cis(self, indices_grid: torch.Tensor, spacing: str = "exp") -> tuple[torch.Tensor, torch.Tensor, bool]:
        freqs = self.precompute_freqs(indices_grid, spacing)
        if self.split_rope:
            expected_freqs = self.inner_dim // 2
            pad_size = expected_freqs - freqs.shape[-1]
            cos_freq, sin_freq = split_freqs_cis(freqs, pad_size, self.num_attention_heads)
        else:
            cos_freq, sin_freq = interleaved_freqs_cis(freqs, self.inner_dim % 2)
        dtype = self.dtype or cos_freq.dtype
        return cos_freq.to(dtype), sin_freq.to(dtype), self.split_rope

    def _replace_padding_with_registers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.num_learnable_registers:
            return hidden_states, attention_mask
        if hidden_states.shape[1] % self.num_learnable_registers != 0:
            raise AssertionError(
                f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by "
                f"num_learnable_registers {self.num_learnable_registers}."
            )
        duplications = hidden_states.shape[1] // self.num_learnable_registers
        registers = torch.tile(self.learnable_registers, (duplications, 1)).to(hidden_states.device)
        attention_mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()
        non_zero_hidden_states = hidden_states[:, attention_mask_binary.squeeze().bool(), :]
        non_zero_count = non_zero_hidden_states.shape[1]
        pad_length = hidden_states.shape[1] - non_zero_count
        adjusted_hidden_states = torch.nn.functional.pad(non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0)
        flipped_mask = torch.flip(attention_mask_binary, dims=[1])
        hidden_states = flipped_mask * adjusted_hidden_states + (1 - flipped_mask) * registers
        attention_mask = torch.full_like(attention_mask, 0.0, dtype=attention_mask.dtype, device=attention_mask.device)
        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if attention_mask is not None:
            hidden_states, attention_mask = self._replace_padding_with_registers(hidden_states, attention_mask)
        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)[None, None, :]
        freqs_cis = self.precompute_freqs_cis(indices_grid)
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)
        hidden_states = rms_norm(hidden_states)
        return hidden_states, attention_mask


def connector_prefix(kind: str) -> str:
    if kind not in {"video", "audio"}:
        raise ValueError("kind must be 'video' or 'audio'")
    return f"model.diffusion_model.{kind}_embeddings_connector."


def load_embeddings_connector_state_dict(
    checkpoint_path: str | Path,
    kind: str,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    prefix = connector_prefix(kind)
    from safetensors import safe_open

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = tuple(key for key in handle.keys() if key.startswith(prefix))
    raw = load_safetensors_subset(checkpoint_path, keys, device=device)
    return {key[len(prefix) :]: value for key, value in raw.items()}


def build_embeddings_connector_from_checkpoint(
    checkpoint_path: str | Path,
    kind: str,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = "cpu",
    strict: bool = True,
) -> Embeddings1DConnector:
    if kind == "video":
        attention_head_dim = 128
    elif kind == "audio":
        attention_head_dim = 64
    else:
        raise ValueError("kind must be 'video' or 'audio'")
    state_dict = load_embeddings_connector_state_dict(checkpoint_path, kind, device=device or "cpu")
    resolved_dtype = dtype or state_dict["learnable_registers"].dtype
    connector = Embeddings1DConnector(
        attention_head_dim=attention_head_dim,
        num_attention_heads=32,
        num_layers=8,
        num_learnable_registers=128,
        apply_gated_attention=True,
        split_rope=True,
        double_precision_rope=True,
        dtype=resolved_dtype,
        device=device,
    )
    connector.load_state_dict(state_dict, strict=strict)
    connector.eval()
    return connector

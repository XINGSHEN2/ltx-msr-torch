from __future__ import annotations

from dataclasses import dataclass

import torch

from .ltx_attention import CrossAttention, FeedForward, rms_norm
from .ltx_timestep import ADALN_BASE_PARAMS_COUNT, ADALN_CROSS_ATTN_PARAMS_COUNT


def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: CrossAttention,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=attention_mask) * q_gate


@dataclass(frozen=True)
class BasicTransformerBlockConfig:
    dim: int
    heads: int
    dim_head: int
    context_dim: int
    cross_attention_adaln: bool


class BasicTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        *,
        context_dim: int,
        cross_attention_adaln: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = BasicTransformerBlockConfig(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            context_dim=context_dim,
            cross_attention_adaln=cross_attention_adaln,
        )
        self.cross_attention_adaln = cross_attention_adaln
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            context_dim=None,
            dtype=dtype,
            device=device,
        )
        self.ff = FeedForward(dim, dim_out=dim, dtype=dtype, device=device)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dtype=dtype,
            device=device,
        )
        num_ada_params = ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.scale_shift_table = torch.nn.Parameter(torch.empty(num_ada_params, dim, device=device, dtype=dtype))
        if cross_attention_adaln:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, dim, device=device, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        timestep: torch.Tensor,
        pe: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool] | None = None,
        self_attention_mask: torch.Tensor | None = None,
        prompt_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        timestep_values = timestep.reshape(batch_size, timestep.shape[1], self.scale_shift_table.shape[0], -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None, :6].to(device=x.device, dtype=x.dtype)
            + timestep_values[:, :, :6, :]
        ).unbind(dim=2)
        x = x + self.attn1(
            rms_norm(x) * (1 + scale_msa) + shift_msa,
            pe=pe,
            mask=self_attention_mask,
        ) * gate_msa

        if self.cross_attention_adaln:
            if prompt_timestep is None:
                raise ValueError("prompt_timestep is required when cross_attention_adaln=True")
            shift_q, scale_q, gate = (
                self.scale_shift_table[None, None, 6:9].to(device=x.device, dtype=x.dtype)
                + timestep_values[:, :, 6:9, :]
            ).unbind(dim=2)
            x = x + apply_cross_attention_adaln(
                x,
                context,
                self.attn2,
                shift_q,
                scale_q,
                gate,
                self.prompt_scale_shift_table,
                prompt_timestep,
                attention_mask,
            )
        else:
            x = x + self.attn2(x, context=context, mask=attention_mask)

        y = rms_norm(x)
        y = torch.addcmul(y, y, scale_mlp).add_(shift_mlp)
        return x.addcmul(self.ff(y), gate_mlp)

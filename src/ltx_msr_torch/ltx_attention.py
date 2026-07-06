from __future__ import annotations

from dataclasses import dataclass

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    output = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    if weight is not None:
        output = output * weight
    return output


def apply_interleaved_rotary_emb(input_tensor: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor) -> torch.Tensor:
    reshaped = input_tensor.reshape(*input_tensor.shape[:-1], -1, 2)
    first, second = reshaped.unbind(dim=-1)
    rotated = torch.stack((-second, first), dim=-1).reshape_as(input_tensor)
    return input_tensor * cos_freqs + rotated * sin_freqs


def apply_split_rotary_emb(input_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    needs_reshape = False
    if input_tensor.ndim != 4 and cos.ndim == 4:
        batch, heads, tokens, _ = cos.shape
        input_tensor = input_tensor.reshape(batch, tokens, heads, -1).swapaxes(1, 2)
        needs_reshape = True
    split_input = input_tensor.reshape(*input_tensor.shape[:-1], 2, -1)
    first_half = split_input[..., :1, :]
    second_half = split_input[..., 1:, :]
    output = split_input * cos.unsqueeze(-2)
    output[..., :1, :].addcmul_(-sin.unsqueeze(-2), second_half)
    output[..., 1:, :].addcmul_(sin.unsqueeze(-2), first_half)
    output = output.reshape(*output.shape[:-2], -1)
    return output.swapaxes(1, 2).reshape(batch, tokens, -1) if needs_reshape else output


def apply_rotary_emb(
    input_tensor: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool],
) -> torch.Tensor:
    cos_freqs, sin_freqs = freqs_cis[0], freqs_cis[1]
    split_pe = freqs_cis[2] if len(freqs_cis) > 2 else False
    if split_pe:
        return apply_split_rotary_emb(input_tensor, cos_freqs, sin_freqs)
    return apply_interleaved_rotary_emb(input_tensor, cos_freqs, sin_freqs)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, query_tokens, inner_dim = q.shape
    key_tokens = k.shape[1]
    head_dim = inner_dim // heads
    qh = q.reshape(batch, query_tokens, heads, head_dim).transpose(1, 2)
    kh = k.reshape(batch, key_tokens, heads, head_dim).transpose(1, 2)
    vh = v.reshape(batch, key_tokens, heads, head_dim).transpose(1, 2)
    attn = torch.matmul(qh, kh.transpose(-2, -1)) * (head_dim**-0.5)
    if mask is not None:
        attn = attn + mask.to(dtype=attn.dtype, device=attn.device)
    probs = torch.softmax(attn, dim=-1)
    return torch.matmul(probs, vh).transpose(1, 2).reshape(batch, query_tokens, inner_dim)


class GELUApprox(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        mult: float = 4,
        dropout: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = torch.nn.Sequential(
            GELUApprox(dim, inner_dim, dtype=dtype, device=device),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(inner_dim, dim_out, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class CrossAttentionConfig:
    query_dim: int
    context_dim: int
    heads: int
    dim_head: int
    apply_gated_attention: bool

    @property
    def inner_dim(self) -> int:
        return self.heads * self.dim_head


class CrossAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        *,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        apply_gated_attention: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        resolved_context_dim = query_dim if context_dim is None else context_dim
        inner_dim = dim_head * heads
        self.config = CrossAttentionConfig(
            query_dim=query_dim,
            context_dim=resolved_context_dim,
            heads=heads,
            dim_head=dim_head,
            apply_gated_attention=apply_gated_attention,
        )
        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=1e-5, dtype=dtype, device=device)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=1e-5, dtype=dtype, device=device)
        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_k = torch.nn.Linear(resolved_context_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_v = torch.nn.Linear(resolved_context_dim, inner_dim, bias=True, dtype=dtype, device=device)
        self.to_gate_logits = (
            torch.nn.Linear(query_dim, heads, bias=True, dtype=dtype, device=device)
            if apply_gated_attention
            else None
        )
        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim, dtype=dtype, device=device),
            torch.nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool] | None = None,
    ) -> torch.Tensor:
        q = self.q_norm(self.to_q(x))
        resolved_context = x if context is None else context
        k = self.k_norm(self.to_k(resolved_context))
        v = self.to_v(resolved_context)
        if pe is not None:
            q = apply_rotary_emb(q, pe)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe)
        out = scaled_dot_product_attention(q, k, v, self.config.heads, mask=mask)
        if self.to_gate_logits is not None:
            gates = 2.0 * torch.sigmoid(self.to_gate_logits(x))
            out = out.reshape(out.shape[0], out.shape[1], self.config.heads, self.config.dim_head)
            out = (out * gates.unsqueeze(-1)).reshape(out.shape[0], out.shape[1], self.config.inner_dim)
        return self.to_out(out)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


GEMMA3_12B_SLIDING_PATTERN: tuple[int | bool, ...] = (1024, 1024, 1024, 1024, 1024, False)


@dataclass(frozen=True)
class ComfyGemma3TextConfig:
    vocab_size: int = 262208
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: tuple[float, float] = (1000000.0, 10000.0)
    head_dim: int = 256
    sliding_window: int = 1024
    sliding_attention: tuple[int | bool, ...] = GEMMA3_12B_SLIDING_PATTERN
    rope_scale: tuple[float, float] = (8.0, 1.0)
    rms_norm_add: bool = True
    transformer_type: str = "gemma3"
    mlp_activation: str = "gelu_pytorch_tanh"
    qkv_bias: bool = False
    q_norm: str | None = "gemma3"
    k_norm: str | None = "gemma3"
    final_norm: bool = True


@dataclass(frozen=True)
class ComfyGemmaTextModelOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


def comfy_gemma3_config_from_hf(config: Any, *, num_layers: int | None = None) -> ComfyGemma3TextConfig:
    layer_count = int(num_layers if num_layers is not None else getattr(config, "num_hidden_layers", 48))
    if layer_count < 0 or layer_count > int(getattr(config, "num_hidden_layers", 48)):
        raise ValueError(f"num_layers must be between 0 and {getattr(config, 'num_hidden_layers', 48)}, got {layer_count}")

    rope_theta = float(getattr(config, "rope_theta", 1000000.0))
    local_theta = float(getattr(config, "rope_local_base_freq", 10000.0))
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_factor = float(rope_scaling.get("factor", 8.0)) if isinstance(rope_scaling, dict) else 8.0
    sliding_window = int(getattr(config, "sliding_window", 1024))
    sliding_pattern = int(getattr(config, "sliding_window_pattern", 6))
    sliding_attention = tuple([sliding_window] * max(sliding_pattern - 1, 0) + [False])

    return ComfyGemma3TextConfig(
        vocab_size=int(getattr(config, "vocab_size", 262208)),
        hidden_size=int(getattr(config, "hidden_size", 3840)),
        intermediate_size=int(getattr(config, "intermediate_size", 15360)),
        num_hidden_layers=layer_count,
        num_attention_heads=int(getattr(config, "num_attention_heads", 16)),
        num_key_value_heads=int(getattr(config, "num_key_value_heads", 8)),
        max_position_embeddings=int(getattr(config, "max_position_embeddings", 131072)),
        rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        rope_theta=(rope_theta, local_theta),
        head_dim=int(getattr(config, "head_dim", 256)),
        sliding_window=sliding_window,
        sliding_attention=sliding_attention,
        rope_scale=(rope_factor, 1.0),
    )


class ComfyCastLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=input.device, dtype=input.dtype)
        bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
        return F.linear(input, weight, bias)


class ComfyEmbedding(torch.nn.Embedding):
    def forward(self, input: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        weight = self.weight.to(device=input.device)
        output = F.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
        return output.to(dtype=out_dtype) if out_dtype is not None else output


class ScaledEmbedding(ComfyEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.scale = scale

    def forward(self, input: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        return super().forward(input, out_dtype=out_dtype) * self.scale


class ComfyRMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1e-6,
        add: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.add = add
        self.weight = torch.nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight + 1.0 if self.add else self.weight
        return F.rms_norm(x, weight.shape, weight=weight.to(device=x.device, dtype=x.dtype), eps=self.eps)


def precompute_freqs_cis(
    head_dim: int,
    position_ids: torch.Tensor,
    theta: tuple[float, ...] | list[float] | float,
    rope_scale: tuple[float, ...] | list[float] | float | None = None,
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    theta_values = list(theta) if isinstance(theta, (list, tuple)) else [float(theta)]
    scale_values = list(rope_scale) if isinstance(rope_scale, (list, tuple)) else rope_scale
    out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for index, theta_value in enumerate(theta_values):
        theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
        inv_freq = 1.0 / (float(theta_value) ** (theta_numerator / head_dim))
        if scale_values is not None:
            inv_freq = inv_freq / (float(scale_values[index]) if isinstance(scale_values, list) else float(scale_values))
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)
        sin_split = sin.shape[-1] // 2
        out.append((cos, sin[..., :sin_split], -sin[..., sin_split:]))
    return out[0] if len(out) == 1 else out


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    org_dtype = xq.dtype
    cos, sin, nsin = freqs_cis

    q_embed = xq * cos
    q_split = q_embed.shape[-1] // 2
    q_embed[..., :q_split].addcmul_(xq[..., q_split:], nsin)
    q_embed[..., q_split:].addcmul_(xq[..., :q_split], sin)

    k_embed = xk * cos
    k_split = k_embed.shape[-1] // 2
    k_embed[..., :k_split].addcmul_(xk[..., k_split:], nsin)
    k_embed[..., k_split:].addcmul_(xk[..., :k_split], sin)

    return q_embed.to(org_dtype), k_embed.to(org_dtype)


class GemmaAttention(torch.nn.Module):
    def __init__(
        self,
        config: ComfyGemma3TextConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.inner_size = self.num_heads * self.head_dim
        self.q_proj = ComfyCastLinear(config.hidden_size, self.inner_size, bias=config.qkv_bias, device=device, dtype=dtype)
        self.k_proj = ComfyCastLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.v_proj = ComfyCastLinear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias, device=device, dtype=dtype)
        self.o_proj = ComfyCastLinear(self.inner_size, config.hidden_size, bias=False, device=device, dtype=dtype)
        self.q_norm = ComfyRMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.k_norm = ComfyRMSNorm(self.head_dim, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)

        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq, xk = apply_rope(xq, xk, freqs_cis=freqs_cis)

        repeat = self.num_heads // self.num_kv_heads
        xk = xk.repeat_interleave(repeat, dim=1)
        xv = xv.repeat_interleave(repeat, dim=1)
        output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask, dropout_p=0.0)
        output = output.transpose(1, 2).reshape(batch_size, seq_length, self.inner_size)
        return self.o_proj(output)


class GemmaMLP(torch.nn.Module):
    def __init__(
        self,
        config: ComfyGemma3TextConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gate_proj = ComfyCastLinear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = ComfyCastLinear(config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = ComfyCastLinear(config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ComfyGemma3TextConfig,
        index: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = GemmaAttention(config, device=device, dtype=dtype)
        self.mlp = GemmaMLP(config, device=device, dtype=dtype)
        self.input_layernorm = ComfyRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_attention_layernorm = ComfyRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.pre_feedforward_layernorm = ComfyRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.post_feedforward_layernorm = ComfyRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.sliding_attention = config.sliding_attention[index % len(config.sliding_attention)]

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        freqs_cis: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        block_mask = attention_mask
        if self.sliding_attention:
            if x.shape[1] > int(self.sliding_attention):
                sliding_mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min / 4, device=x.device, dtype=x.dtype)
                sliding_mask.tril_(diagonal=-int(self.sliding_attention))
                block_mask = block_mask + sliding_mask if block_mask is not None else sliding_mask
            block_freqs = freqs_cis[1]
        else:
            block_freqs = freqs_cis[0]

        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(hidden_states=x, attention_mask=block_mask, freqs_cis=block_freqs)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return residual + x


class ComfyGemma3TextModel(torch.nn.Module):
    """Pure torch implementation of ComfyUI's Gemma3_12B text path used by LTXAV."""

    def __init__(
        self,
        config: ComfyGemma3TextConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = ScaledEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.hidden_size**0.5,
            device=device,
            dtype=dtype,
        )
        self.layers = torch.nn.ModuleList(
            GemmaTransformerBlock(config, index=i, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        )
        self.norm = ComfyRMSNorm(config.hidden_size, eps=config.rms_norm_eps, add=config.rms_norm_add, device=device, dtype=dtype)
        self.eval()

    def get_input_embeddings(self) -> ScaledEmbedding:
        return self.embed_tokens

    def compute_freqs_cis(self, position_ids: torch.Tensor, device: torch.device | str) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        freqs = precompute_freqs_cis(
            self.config.head_dim,
            position_ids,
            self.config.rope_theta,
            self.config.rope_scale,
            device=device,
        )
        if not isinstance(freqs, list):
            raise TypeError("Gemma3 text config must produce local and global RoPE frequencies")
        return freqs

    def _attention_mask(self, attention_mask: torch.Tensor | None, *, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(device=device, dtype=dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(
                attention_mask.shape[0], 1, seq_len, attention_mask.shape[-1]
            )
            mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(dtype).min / 4)
        if seq_len > 1:
            causal_mask = torch.empty(seq_len, seq_len, dtype=dtype, device=device).fill_(torch.finfo(dtype).min / 4).triu_(1)
            mask = mask + causal_mask if mask is not None else causal_mask
        return mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        embeds: torch.Tensor | None = None,
        intermediate_output: str | None = None,
        final_layer_norm_intermediate: bool = False,
    ) -> ComfyGemmaTextModelOutput:
        if embeds is not None:
            x = embeds
        elif input_ids is not None:
            x = self.embed_tokens(input_ids, out_dtype=torch.float32)
        else:
            raise ValueError("input_ids or embeds must be provided")

        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        freqs_cis = self.compute_freqs_cis(position_ids, x.device)
        mask = self._attention_mask(attention_mask, seq_len=seq_len, dtype=x.dtype, device=x.device)

        collect_all = output_hidden_states or intermediate_output == "all"
        all_hidden: list[torch.Tensor] = []
        for layer in self.layers:
            if collect_all:
                all_hidden.append(x.clone())
            x = layer(x, attention_mask=mask, freqs_cis=freqs_cis)

        x = self.norm(x)
        if collect_all:
            all_hidden.append(x.clone())
            if final_layer_norm_intermediate:
                all_hidden = [self.norm(hidden) for hidden in all_hidden]

        return ComfyGemmaTextModelOutput(
            last_hidden_state=x,
            hidden_states=tuple(all_hidden) if collect_all else None,
        )


def build_comfy_gemma3_text_model(
    config: Any,
    *,
    device: torch.device | str = "meta",
    dtype: torch.dtype | None = None,
    num_layers: int | None = None,
) -> ComfyGemma3TextModel:
    comfy_config = config if isinstance(config, ComfyGemma3TextConfig) else comfy_gemma3_config_from_hf(config, num_layers=num_layers)
    if isinstance(config, ComfyGemma3TextConfig) and num_layers is not None:
        comfy_config = ComfyGemma3TextConfig(**{**config.__dict__, "num_hidden_layers": num_layers})
    with torch.device(device):
        model = ComfyGemma3TextModel(comfy_config, device=device, dtype=dtype)
    model.eval()
    return model

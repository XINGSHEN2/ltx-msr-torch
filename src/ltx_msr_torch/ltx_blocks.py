from __future__ import annotations

from dataclasses import dataclass

import torch

from .ltx_attention import CrossAttention, FeedForward, rms_norm
from .nag import LTX2NAGConfig
from .prompt_relay import PromptRelayPlan, build_promptrelay_mask
from .ltx_timestep import ADALN_BASE_PARAMS_COUNT, ADALN_CROSS_ATTN_PARAMS_COUNT, CompressedTimestep


def _combine_attention_masks(
    attention_mask: torch.Tensor | None,
    promptrelay_plan: PromptRelayPlan | None,
    *,
    query_tokens: int,
    key_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    transformer_options: dict[str, object] | None = None,
) -> torch.Tensor | None:
    promptrelay_mask = None
    if promptrelay_plan is not None:
        promptrelay_mask = build_promptrelay_mask(
            promptrelay_plan,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            dtype=dtype,
            device=device,
            transformer_options=transformer_options,
        )
    if promptrelay_mask is None:
        return attention_mask
    if attention_mask is None:
        return promptrelay_mask
    return attention_mask.to(device=device, dtype=dtype) + promptrelay_mask


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


@dataclass(frozen=True)
class BasicAVTransformerBlockConfig:
    video_dim: int
    audio_dim: int
    video_heads: int
    audio_heads: int
    video_dim_head: int
    audio_dim_head: int
    video_context_dim: int
    audio_context_dim: int
    apply_gated_attention: bool
    cross_attention_adaln: bool


class BasicAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        video_dim: int,
        audio_dim: int,
        video_heads: int,
        audio_heads: int,
        video_dim_head: int,
        audio_dim_head: int,
        video_context_dim: int,
        audio_context_dim: int,
        apply_gated_attention: bool = False,
        cross_attention_adaln: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = BasicAVTransformerBlockConfig(
            video_dim=video_dim,
            audio_dim=audio_dim,
            video_heads=video_heads,
            audio_heads=audio_heads,
            video_dim_head=video_dim_head,
            audio_dim_head=audio_dim_head,
            video_context_dim=video_context_dim,
            audio_context_dim=audio_context_dim,
            apply_gated_attention=apply_gated_attention,
            cross_attention_adaln=cross_attention_adaln,
        )
        self.cross_attention_adaln = cross_attention_adaln
        self.attn1 = CrossAttention(
            query_dim=video_dim,
            heads=video_heads,
            dim_head=video_dim_head,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
        )
        self.audio_attn1 = CrossAttention(
            query_dim=audio_dim,
            heads=audio_heads,
            dim_head=audio_dim_head,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
        )
        self.attn2 = CrossAttention(
            query_dim=video_dim,
            context_dim=video_context_dim,
            heads=video_heads,
            dim_head=video_dim_head,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
        )
        self.audio_attn2 = CrossAttention(
            query_dim=audio_dim,
            context_dim=audio_context_dim,
            heads=audio_heads,
            dim_head=audio_dim_head,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
        )
        self.audio_to_video_attn = CrossAttention(
            query_dim=video_dim,
            context_dim=audio_dim,
            heads=audio_heads,
            dim_head=audio_dim_head,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
        )
        self.video_to_audio_attn = CrossAttention(
            query_dim=audio_dim,
            context_dim=video_dim,
            heads=audio_heads,
            dim_head=audio_dim_head,
            apply_gated_attention=apply_gated_attention,
            dtype=dtype,
            device=device,
        )
        self.ff = FeedForward(video_dim, dim_out=video_dim, dtype=dtype, device=device)
        self.audio_ff = FeedForward(audio_dim, dim_out=audio_dim, dtype=dtype, device=device)

        num_ada_params = ADALN_CROSS_ATTN_PARAMS_COUNT if cross_attention_adaln else ADALN_BASE_PARAMS_COUNT
        self.scale_shift_table = torch.nn.Parameter(torch.empty(num_ada_params, video_dim, device=device, dtype=dtype))
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(num_ada_params, audio_dim, device=device, dtype=dtype))
        if cross_attention_adaln:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video_dim, device=device, dtype=dtype))
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio_dim, device=device, dtype=dtype))
        self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio_dim, device=device, dtype=dtype))
        self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video_dim, device=device, dtype=dtype))

    @staticmethod
    def get_ada_values(
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor | CompressedTimestep,
        indices: slice = slice(None, None),
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(timestep, CompressedTimestep):
            return timestep.expand_for_computation(scale_shift_table, batch_size, indices)
        num_ada_params = scale_shift_table.shape[0]
        values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return values

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: CrossAttention,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        nag_context: torch.Tensor | None = None,
        nag_config: LTX2NAGConfig | None = None,
        promptrelay_plan: PromptRelayPlan | None = None,
        transformer_options: dict[str, object] | None = None,
    ) -> torch.Tensor:
        combined_mask = _combine_attention_masks(
            attention_mask,
            promptrelay_plan,
            query_tokens=x.shape[1],
            key_tokens=context.shape[1],
            dtype=x.dtype,
            device=x.device,
            transformer_options=transformer_options,
        )
        if self.cross_attention_adaln:
            if prompt_scale_shift_table is None or prompt_timestep is None:
                raise ValueError("prompt timestep and table are required when cross_attention_adaln=True")
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            batch_size = x.shape[0]
            shift_kv, scale_kv = (
                prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
                + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
            ).unbind(dim=2)
            attn_input = rms_norm(x) * (1 + scale_q) + shift_q
            encoder_hidden_states = context * (1 + scale_kv) + shift_kv
            if nag_context is not None and nag_config is not None and nag_config.scale != 0:
                return attn.forward_nag(
                    attn_input,
                    context=encoder_hidden_states,
                    nag_context=nag_context,
                    nag_config=nag_config,
                    mask=combined_mask,
                ) * gate
            return attn(attn_input, context=encoder_hidden_states, mask=combined_mask) * gate
        if nag_context is not None and nag_config is not None and nag_config.scale != 0:
            return attn.forward_nag(
                rms_norm(x),
                context=context,
                nag_context=nag_context,
                nag_config=nag_config,
                mask=combined_mask,
            )
        return attn(rms_norm(x), context=context, mask=combined_mask)

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        *,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        video_pe: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool] | None = None,
        audio_pe: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool] | None = None,
        video_cross_pe: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool] | None = None,
        audio_cross_pe: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, bool] | None = None,
        video_cross_scale_shift_timestep: torch.Tensor | None = None,
        audio_cross_scale_shift_timestep: torch.Tensor | None = None,
        video_cross_gate_timestep: torch.Tensor | None = None,
        audio_cross_gate_timestep: torch.Tensor | None = None,
        transformer_options: dict[str, object] | None = None,
        self_attention_mask: torch.Tensor | None = None,
        video_prompt_timestep: torch.Tensor | None = None,
        audio_prompt_timestep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        options = transformer_options or {}
        nag_config = options.get("nag_config")
        if nag_config is not None and not isinstance(nag_config, LTX2NAGConfig):
            raise TypeError("transformer_options['nag_config'] must be LTX2NAGConfig")
        nag_video_context = options.get("nag_video_context")
        nag_audio_context = options.get("nag_audio_context")
        promptrelay_plan = options.get("promptrelay_plan")
        if promptrelay_plan is not None and not isinstance(promptrelay_plan, PromptRelayPlan):
            raise TypeError("transformer_options['promptrelay_plan'] must be PromptRelayPlan")
        vx, ax = x
        run_vx = bool(options.get("run_vx", True))
        run_ax = bool(options.get("run_ax", True)) and ax.numel() > 0
        run_a2v = run_vx and bool(options.get("a2v_cross_attn", True)) and ax.numel() > 0
        run_v2a = run_ax and bool(options.get("v2a_cross_attn", True))

        if run_vx:
            shift_msa, scale_msa = self.get_ada_values(self.scale_shift_table, vx.shape[0], video_timestep, slice(0, 2))
            attn1_out = self.attn1(
                rms_norm(vx) * (1 + scale_msa) + shift_msa,
                pe=video_pe,
                mask=self_attention_mask,
            )
            gate_msa = self.get_ada_values(self.scale_shift_table, vx.shape[0], video_timestep, slice(2, 3))[0]
            vx = vx.addcmul(attn1_out, gate_msa)
            vx = vx + self._apply_text_cross_attention(
                vx,
                video_context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video_timestep,
                video_prompt_timestep,
                attention_mask,
                nag_context=nag_video_context if isinstance(nag_video_context, torch.Tensor) else None,
                nag_config=nag_config,
                promptrelay_plan=promptrelay_plan,
                transformer_options=options,
            )

        if run_ax:
            shift_msa, scale_msa = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], audio_timestep, slice(0, 2))
            attn1_out = self.audio_attn1(rms_norm(ax) * (1 + scale_msa) + shift_msa, pe=audio_pe)
            gate_msa = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], audio_timestep, slice(2, 3))[0]
            ax = ax.addcmul(attn1_out, gate_msa)
            ax = ax + self._apply_text_cross_attention(
                ax,
                audio_context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio_timestep,
                audio_prompt_timestep,
                attention_mask,
                nag_context=nag_audio_context if isinstance(nag_audio_context, torch.Tensor) else None,
                nag_config=nag_config,
                promptrelay_plan=promptrelay_plan,
                transformer_options=options,
            )

        if run_a2v or run_v2a:
            if (
                video_cross_scale_shift_timestep is None
                or audio_cross_scale_shift_timestep is None
                or video_cross_gate_timestep is None
                or audio_cross_gate_timestep is None
            ):
                raise ValueError("AV cross-attention timesteps are required when AV cross-attention is enabled")
            vx_norm = rms_norm(vx)
            ax_norm = rms_norm(ax)
            if run_a2v:
                audio_scale, audio_shift = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4], ax.shape[0], audio_cross_scale_shift_timestep
                )[:2]
                video_scale, video_shift = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_video[:4], vx.shape[0], video_cross_scale_shift_timestep
                )[:2]
                a2v_out = self.audio_to_video_attn(
                    vx_norm * (1 + video_scale) + video_shift,
                    context=ax_norm * (1 + audio_scale) + audio_shift,
                    pe=video_cross_pe,
                    k_pe=audio_cross_pe,
                )
                gate = self.get_ada_values(self.scale_shift_table_a2v_ca_video[4:], vx.shape[0], video_cross_gate_timestep)[0]
                vx = vx.addcmul(a2v_out, gate)
            if run_v2a:
                audio_scale, audio_shift = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_audio[:4], ax.shape[0], audio_cross_scale_shift_timestep
                )[2:4]
                video_scale, video_shift = self.get_ada_values(
                    self.scale_shift_table_a2v_ca_video[:4], vx.shape[0], video_cross_scale_shift_timestep
                )[2:4]
                v2a_out = self.video_to_audio_attn(
                    ax_norm * (1 + audio_scale) + audio_shift,
                    context=vx_norm * (1 + video_scale) + video_shift,
                    pe=audio_cross_pe,
                    k_pe=video_cross_pe,
                )
                gate = self.get_ada_values(self.scale_shift_table_a2v_ca_audio[4:], ax.shape[0], audio_cross_gate_timestep)[0]
                ax = ax.addcmul(v2a_out, gate)

        if run_vx:
            shift_mlp, scale_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], video_timestep, slice(3, 5))
            ff_out = self.ff(rms_norm(vx) * (1 + scale_mlp) + shift_mlp)
            gate_mlp = self.get_ada_values(self.scale_shift_table, vx.shape[0], video_timestep, slice(5, 6))[0]
            vx = vx.addcmul(ff_out, gate_mlp)

        if run_ax:
            shift_mlp, scale_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], audio_timestep, slice(3, 5))
            ff_out = self.audio_ff(rms_norm(ax) * (1 + scale_mlp) + shift_mlp)
            gate_mlp = self.get_ada_values(self.audio_scale_shift_table, ax.shape[0], audio_timestep, slice(5, 6))[0]
            ax = ax.addcmul(ff_out, gate_mlp)

        return vx, ax

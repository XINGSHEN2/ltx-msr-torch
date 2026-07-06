from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint_loader import load_safetensors_subset
from .ltx_blocks import BasicAVTransformerBlock
from .ltx_timestep import ADALN_CROSS_ATTN_PARAMS_COUNT, AdaLayerNormSingle
from .ltxav_io import LTXAVInputProjection
from .ltxav_output import LTXAVOutputProcessor
from .ltxav_prepare import prepare_ltxav_block_inputs
from .ltxav_timestep import prepare_ltxav_timesteps
from .ltxav_transformer import LTXAVTransformerManifest


@dataclass(frozen=True)
class LTXAVModelConfig:
    video_in_channels: int = 128
    audio_in_channels: int = 128
    video_dim: int = 4096
    audio_dim: int = 2048
    video_heads: int = 32
    audio_heads: int = 32
    video_dim_head: int = 128
    audio_dim_head: int = 64
    num_layers: int = 48
    video_context_dim: int = 4096
    audio_context_dim: int = 2048
    video_out_channels: int = 128
    audio_out_channels: int = 128
    audio_channels: int = 8
    audio_frequency: int = 16
    cross_attention_adaln: bool = True
    apply_gated_attention: bool = True
    timestep_scale_multiplier: float = 1000.0
    av_ca_timestep_scale_multiplier: float = 1.0


def ltxav_model_config_from_manifest(manifest: LTXAVTransformerManifest) -> LTXAVModelConfig:
    config = manifest.config
    audio_in_channels = manifest.specs["model.diffusion_model.audio_patchify_proj.weight"].shape[1]
    audio_out_channels = manifest.specs["model.diffusion_model.audio_proj_out.weight"].shape[0]
    return LTXAVModelConfig(
        video_in_channels=config.in_channels,
        audio_in_channels=audio_in_channels,
        video_dim=config.num_attention_heads * config.attention_head_dim,
        audio_dim=config.audio_num_attention_heads * config.audio_attention_head_dim,
        video_heads=config.num_attention_heads,
        audio_heads=config.audio_num_attention_heads,
        video_dim_head=config.attention_head_dim,
        audio_dim_head=config.audio_attention_head_dim,
        num_layers=config.num_layers,
        video_context_dim=config.cross_attention_dim,
        audio_context_dim=config.audio_cross_attention_dim,
        video_out_channels=config.out_channels,
        audio_out_channels=audio_out_channels,
        cross_attention_adaln=config.cross_attention_adaln,
        apply_gated_attention=config.apply_gated_attention,
    )


class LTXAVModel(torch.nn.Module):
    def __init__(
        self,
        config: LTXAVModelConfig = LTXAVModelConfig(),
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_projection = LTXAVInputProjection(
            video_in_channels=config.video_in_channels,
            video_hidden_dim=config.video_dim,
            audio_in_channels=config.audio_in_channels,
            audio_hidden_dim=config.audio_dim,
            dtype=dtype,
            device=device,
        )
        coefficient = ADALN_CROSS_ATTN_PARAMS_COUNT if config.cross_attention_adaln else 6
        self.video_adaln_single = AdaLayerNormSingle(
            config.video_dim,
            embedding_coefficient=coefficient,
            dtype=dtype,
            device=device,
        )
        self.audio_adaln_single = AdaLayerNormSingle(
            config.audio_dim,
            embedding_coefficient=coefficient,
            dtype=dtype,
            device=device,
        )
        self.video_prompt_adaln_single = (
            AdaLayerNormSingle(config.video_dim, embedding_coefficient=2, dtype=dtype, device=device)
            if config.cross_attention_adaln
            else None
        )
        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(config.audio_dim, embedding_coefficient=2, dtype=dtype, device=device)
            if config.cross_attention_adaln
            else None
        )
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            config.video_dim,
            embedding_coefficient=4,
            dtype=dtype,
            device=device,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            config.video_dim,
            embedding_coefficient=1,
            dtype=dtype,
            device=device,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            config.audio_dim,
            embedding_coefficient=4,
            dtype=dtype,
            device=device,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            config.audio_dim,
            embedding_coefficient=1,
            dtype=dtype,
            device=device,
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    video_dim=config.video_dim,
                    audio_dim=config.audio_dim,
                    video_heads=config.video_heads,
                    audio_heads=config.audio_heads,
                    video_dim_head=config.video_dim_head,
                    audio_dim_head=config.audio_dim_head,
                    video_context_dim=config.video_context_dim,
                    audio_context_dim=config.audio_context_dim,
                    apply_gated_attention=config.apply_gated_attention,
                    cross_attention_adaln=config.cross_attention_adaln,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.output_processor = LTXAVOutputProcessor(
            video_hidden_dim=config.video_dim,
            video_out_channels=config.video_out_channels,
            audio_hidden_dim=config.audio_dim,
            audio_out_channels=config.audio_out_channels,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        *,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        frame_rate: float,
        attention_mask: torch.Tensor | None = None,
        transformer_options: dict[str, object] | None = None,
        self_attention_mask: torch.Tensor | None = None,
        ref_audio_seq_len: int = 0,
        target_audio_seq_len: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        config = self.config
        prepared = prepare_ltxav_block_inputs(
            input_projection=self.input_projection,
            video_latents=video_latents,
            audio_latents=audio_latents,
            context=context,
            attention_mask=attention_mask,
            frame_rate=frame_rate,
            video_dim=config.video_dim,
            audio_dim=config.audio_dim,
            audio_cross_dim=config.audio_heads * config.audio_dim_head,
            video_heads=config.video_heads,
            audio_heads=config.audio_heads,
        )
        timesteps = prepare_ltxav_timesteps(
            timestep=timestep,
            batch_size=video_latents.shape[0],
            hidden_dtype=prepared.projected.video_tokens.dtype,
            video_adaln_single=self.video_adaln_single,
            audio_adaln_single=self.audio_adaln_single,
            av_ca_video_scale_shift_adaln_single=self.av_ca_video_scale_shift_adaln_single,
            av_ca_a2v_gate_adaln_single=self.av_ca_a2v_gate_adaln_single,
            av_ca_audio_scale_shift_adaln_single=self.av_ca_audio_scale_shift_adaln_single,
            av_ca_v2a_gate_adaln_single=self.av_ca_v2a_gate_adaln_single,
            video_prompt_adaln_single=self.video_prompt_adaln_single,
            audio_prompt_adaln_single=self.audio_prompt_adaln_single,
            audio_timestep=audio_timestep,
            orig_shape=tuple(video_latents.shape),
            has_spatial_mask=False,
            ref_audio_seq_len=ref_audio_seq_len,
            target_audio_seq_len=target_audio_seq_len,
            timestep_scale_multiplier=config.timestep_scale_multiplier,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
        )
        video_tokens = prepared.projected.video_tokens
        audio_tokens = prepared.projected.audio_tokens
        for block in self.transformer_blocks:
            video_tokens, audio_tokens = block(
                (video_tokens, audio_tokens),
                video_context=prepared.video_context,
                audio_context=prepared.audio_context,
                attention_mask=prepared.attention_mask,
                video_timestep=timesteps.video_timestep,
                audio_timestep=timesteps.audio_timestep,
                video_pe=prepared.video_pe,
                audio_pe=prepared.audio_pe,
                video_cross_pe=prepared.video_cross_pe,
                audio_cross_pe=prepared.audio_cross_pe,
                video_cross_scale_shift_timestep=timesteps.video_cross_scale_shift_timestep,
                audio_cross_scale_shift_timestep=timesteps.audio_cross_scale_shift_timestep,
                video_cross_gate_timestep=timesteps.video_cross_gate_timestep,
                audio_cross_gate_timestep=timesteps.audio_cross_gate_timestep,
                transformer_options=transformer_options,
                self_attention_mask=self_attention_mask,
                video_prompt_timestep=timesteps.video_prompt_timestep,
                audio_prompt_timestep=timesteps.audio_prompt_timestep,
            )
        return self.output_processor(
            video_tokens,
            audio_tokens,
            video_embedded_timestep=timesteps.video_embedded_timestep,
            audio_embedded_timestep=timesteps.audio_embedded_timestep,
            orig_shape=tuple(video_latents.shape),
            ref_audio_seq_len=ref_audio_seq_len,
            audio_channels=config.audio_channels,
            audio_frequency=config.audio_frequency,
        )


def ltxav_model_checkpoint_key(model_state_key: str) -> str:
    prefixes = {
        "input_projection.": "",
        "output_processor.": "",
        "video_adaln_single.": "adaln_single.",
        "video_prompt_adaln_single.": "prompt_adaln_single.",
    }
    for local_prefix, checkpoint_prefix in prefixes.items():
        if model_state_key.startswith(local_prefix):
            return f"model.diffusion_model.{checkpoint_prefix}{model_state_key[len(local_prefix):]}"
    return f"model.diffusion_model.{model_state_key}"


def load_ltxav_model_state_dict(
    model: LTXAVModel,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    local_keys = tuple(model.state_dict().keys())
    checkpoint_keys = tuple(ltxav_model_checkpoint_key(key) for key in local_keys)
    raw = load_safetensors_subset(checkpoint_path, checkpoint_keys, device=device)
    return {
        local_key: raw[checkpoint_key]
        for local_key, checkpoint_key in zip(local_keys, checkpoint_keys)
    }


def load_ltxav_model_weights(
    model: LTXAVModel,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    state_dict = load_ltxav_model_state_dict(model, checkpoint_path, device=device)
    return model.load_state_dict(state_dict, strict=strict)

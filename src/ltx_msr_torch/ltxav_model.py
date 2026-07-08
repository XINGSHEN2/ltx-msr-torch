from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from safetensors import safe_open

from .checkpoint_loader import load_safetensors_subset
from .lora_apply import lora_pair_delta, target_key_candidates
from .lora_loader import LoRAManifest
from .ltx_attention import GuideAttentionMask
from .ltx_blocks import BasicAVTransformerBlock
from .ltx_timestep import ADALN_CROSS_ATTN_PARAMS_COUNT, AdaLayerNormSingle
from .ltxav_io import LTXAVInputProjection
from .ltxav_output import LTXAVOutputProcessor
from .ltxav_prepare import prepare_ltxav_block_inputs
from .ltxav_timestep import prepare_ltxav_timesteps
from .ltxav_transformer import LTXAVTransformerManifest, inspect_ltxav_transformer_manifest


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
    causal_temporal_positioning: bool = False
    use_middle_indices_grid: bool = False


@dataclass(frozen=True)
class LTXAVModelLoadReport:
    loaded: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]


@dataclass(frozen=True)
class LTXAVModelLoRAReport:
    matched: int
    skipped: int
    applied_keys: tuple[str, ...]
    skipped_targets: tuple[str, ...]


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
        timestep_scale_multiplier=config.timestep_scale_multiplier,
        av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
        causal_temporal_positioning=config.causal_temporal_positioning,
        use_middle_indices_grid=config.use_middle_indices_grid,
    )


def create_ltxav_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = "meta",
    num_layers: int | None = None,
) -> LTXAVModel:
    manifest = inspect_ltxav_transformer_manifest(checkpoint_path)
    config = ltxav_model_config_from_manifest(manifest)
    if num_layers is not None:
        if num_layers < 0 or num_layers > config.num_layers:
            raise ValueError(f"num_layers must be between 0 and {config.num_layers}, got {num_layers}")
        config = replace(config, num_layers=num_layers)
    return LTXAVModel(config, dtype=dtype, device=device)


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
        keyframe_idxs: torch.Tensor | None = None,
        grid_mask: torch.Tensor | None = None,
        orig_patchified_shape: tuple[int, ...] | list[int] | None = None,
        output_orig_shape: tuple[int, ...] | list[int] | None = None,
        denoise_mask: torch.Tensor | None = None,
        guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
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
            keyframe_idxs=keyframe_idxs,
            denoise_mask=denoise_mask,
            guide_attention_entries=guide_attention_entries,
            causal_temporal_positioning=config.causal_temporal_positioning,
            use_middle_indices_grid=config.use_middle_indices_grid,
        )
        grid_mask = grid_mask if grid_mask is not None else prepared.grid_mask
        orig_patchified_shape = orig_patchified_shape if orig_patchified_shape is not None else prepared.orig_patchified_shape
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
            grid_mask=grid_mask,
            orig_shape=tuple(video_latents.shape),
            has_spatial_mask=_has_spatial_denoise_mask(denoise_mask),
            ref_audio_seq_len=ref_audio_seq_len,
            target_audio_seq_len=target_audio_seq_len,
            timestep_scale_multiplier=config.timestep_scale_multiplier,
            av_ca_timestep_scale_multiplier=config.av_ca_timestep_scale_multiplier,
        )
        video_tokens = prepared.projected.video_tokens
        audio_tokens = prepared.projected.audio_tokens
        debug_trace = _debug_trace_from_options(transformer_options)
        if debug_trace is not None:
            _append_model_debug_trace(debug_trace, stage="input", index=-1, video=video_tokens, audio=audio_tokens)
        guide_self_attention_mask = self_attention_mask
        if guide_self_attention_mask is None:
            guide_self_attention_mask = self._build_guide_self_attention_mask(
                video_tokens,
                num_guide_tokens=prepared.projected.num_guide_tokens,
                resolved_entries=prepared.projected.resolved_guide_entries,
            )
        for block_index, block in enumerate(self.transformer_blocks):
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
                self_attention_mask=guide_self_attention_mask,
                video_prompt_timestep=timesteps.video_prompt_timestep,
                audio_prompt_timestep=timesteps.audio_prompt_timestep,
            )
            if debug_trace is not None:
                _append_model_debug_trace(
                    debug_trace,
                    stage="block",
                    index=block_index,
                    video=video_tokens,
                    audio=audio_tokens,
                )
        return self.output_processor(
            video_tokens,
            audio_tokens,
            video_embedded_timestep=timesteps.video_embedded_timestep,
            audio_embedded_timestep=timesteps.audio_embedded_timestep,
            orig_shape=tuple(output_orig_shape or video_latents.shape),
            keyframe_idxs=keyframe_idxs,
            grid_mask=grid_mask,
            orig_patchified_shape=orig_patchified_shape,
            ref_audio_seq_len=ref_audio_seq_len,
            audio_channels=config.audio_channels,
            audio_frequency=config.audio_frequency,
        )

    @staticmethod
    def _build_guide_self_attention_mask(
        video_tokens: torch.Tensor,
        *,
        num_guide_tokens: int,
        resolved_entries: tuple[dict[str, object], ...],
    ) -> GuideAttentionMask | None:
        if num_guide_tokens == 0 or not resolved_entries:
            return None
        needs_mask = any(
            float(entry["strength"]) != 1.0 or entry.get("pixel_mask") is not None
            for entry in resolved_entries
        )
        if not needs_mask:
            return None

        total_tokens = int(video_tokens.shape[1])
        guide_start = total_tokens - int(num_guide_tokens)
        weights: list[torch.Tensor] = []
        tracked = 0
        for entry in resolved_entries:
            surviving = int(entry["surviving_count"])
            if surviving == 0:
                continue
            strength = float(entry["strength"])
            pixel_mask = entry.get("pixel_mask")
            latent_shape = entry.get("latent_shape")
            if isinstance(pixel_mask, torch.Tensor) and latent_shape is not None:
                f_lat, h_lat, w_lat = (int(value) for value in latent_shape)  # type: ignore[arg-type]
                per_token = LTXAVModel._downsample_mask_to_latent(
                    pixel_mask.to(device=video_tokens.device, dtype=video_tokens.dtype),
                    f_lat,
                    h_lat,
                    w_lat,
                )
                if per_token.shape[0] > 1:
                    per_token = per_token[:1]
                n_weights = min(int(per_token.shape[1]), surviving)
                entry_weights = per_token[:, :n_weights] * strength
            else:
                entry_weights = torch.full(
                    (1, surviving),
                    strength,
                    device=video_tokens.device,
                    dtype=video_tokens.dtype,
                )
            weights.append(entry_weights)
            tracked += int(entry_weights.shape[1])

        if not weights:
            return None
        tracked_weights = torch.cat(weights, dim=1)
        if bool((tracked_weights == 1.0).all().item()):
            return None
        return GuideAttentionMask(total_tokens, guide_start, tracked, tracked_weights)

    @staticmethod
    def _downsample_mask_to_latent(mask: torch.Tensor, f_lat: int, h_lat: int, w_lat: int) -> torch.Tensor:
        batch = int(mask.shape[0])
        f_pix = int(mask.shape[2])
        spatial = mask.permute(0, 2, 1, 3, 4).reshape(batch * f_pix, 1, mask.shape[3], mask.shape[4])
        spatial_down = torch.nn.functional.interpolate(spatial, size=(h_lat, w_lat), mode="area")
        spatial_down = spatial_down.reshape(batch, f_pix, 1, h_lat, w_lat).permute(0, 2, 1, 3, 4)

        first_frame = spatial_down[:, :, :1, :, :]
        if f_pix > 1 and f_lat > 1:
            remaining_pix = f_pix - 1
            remaining_lat = f_lat - 1
            group = remaining_pix // remaining_lat
            if group < 1:
                rest_flat = spatial_down[:, :, 1:, :, :].permute(0, 3, 4, 1, 2).reshape(batch * h_lat * w_lat, 1, -1)
                rest_up = torch.nn.functional.interpolate(rest_flat, size=remaining_lat, mode="nearest")
                rest = rest_up.reshape(batch, h_lat, w_lat, 1, remaining_lat).permute(0, 3, 4, 1, 2)
            else:
                usable = remaining_lat * group
                rest = spatial_down[:, :, 1 : 1 + usable, :, :].reshape(
                    batch,
                    1,
                    remaining_lat,
                    group,
                    h_lat,
                    w_lat,
                ).mean(dim=3)
            latent_mask = torch.cat([first_frame, rest], dim=2)
        elif f_lat > 1:
            latent_mask = first_frame.expand(-1, -1, f_lat, -1, -1)
        else:
            latent_mask = first_frame
        return latent_mask.reshape(batch, f_lat * h_lat * w_lat)


def _has_spatial_denoise_mask(denoise_mask: torch.Tensor | None) -> bool:
    if denoise_mask is None:
        return False
    for frame_idx in range(int(denoise_mask.shape[2])):
        frame_mask = denoise_mask[0, 0, frame_idx]
        if frame_mask.numel() > 0 and frame_mask.min() != frame_mask.max():
            return True
    return False


def _debug_trace_from_options(transformer_options: dict[str, object] | None) -> list[dict[str, object]] | None:
    if not isinstance(transformer_options, dict):
        return None
    trace = transformer_options.get("ltx_msr_debug_trace")
    return trace if isinstance(trace, list) else None


def _append_model_debug_trace(
    trace: list[dict[str, object]],
    *,
    stage: str,
    index: int,
    video: torch.Tensor,
    audio: torch.Tensor,
) -> None:
    trace.append(
        {
            "stage": stage,
            "index": int(index),
            "video": _trace_tensor_sample(video),
            "audio": _trace_tensor_sample(audio),
        }
    )


def _trace_tensor_sample(tensor: torch.Tensor) -> dict[str, object]:
    token_count = int(tensor.shape[1])
    dim_count = int(tensor.shape[2])
    token_indices = _trace_indices(token_count, max_count=8, device=tensor.device)
    dim_indices = _trace_indices(dim_count, max_count=16, device=tensor.device)
    sample = tensor[0].index_select(0, token_indices).index_select(1, dim_indices).detach().cpu()
    sample_float = sample.float()
    return {
        "shape": tuple(int(value) for value in tensor.shape),
        "token_indices": tuple(int(value) for value in token_indices.detach().cpu().tolist()),
        "dim_indices": tuple(int(value) for value in dim_indices.detach().cpu().tolist()),
        "sample": sample,
        "sample_mean": float(sample_float.mean().item()),
        "sample_std": float(sample_float.std().item()),
        "sample_absmax": float(sample_float.abs().max().item()),
    }


def _trace_indices(length: int, *, max_count: int, device: torch.device) -> torch.Tensor:
    if length <= max_count:
        return torch.arange(length, device=device, dtype=torch.long)
    raw = torch.linspace(0, length - 1, steps=max_count, device=device)
    return raw.round().to(dtype=torch.long).unique(sorted=True)


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


def ltxav_model_local_key(checkpoint_key: str, local_keys: set[str]) -> str | None:
    reverse_prefixes = {
        "model.diffusion_model.": "",
        "model.diffusion_model.adaln_single.": "video_adaln_single.",
        "model.diffusion_model.prompt_adaln_single.": "video_prompt_adaln_single.",
    }
    for checkpoint_prefix, local_prefix in sorted(reverse_prefixes.items(), key=lambda item: len(item[0]), reverse=True):
        if checkpoint_key.startswith(checkpoint_prefix):
            candidate = f"{local_prefix}{checkpoint_key[len(checkpoint_prefix):]}"
            if candidate in local_keys:
                return candidate
            input_candidate = f"input_projection.{candidate}"
            if input_candidate in local_keys:
                return input_candidate
            output_candidate = f"output_processor.{candidate}"
            if output_candidate in local_keys:
                return output_candidate
    return checkpoint_key if checkpoint_key in local_keys else None


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


def missing_ltxav_model_checkpoint_keys(
    model: LTXAVModel,
    checkpoint_path: str | Path,
) -> tuple[str, ...]:
    checkpoint_keys = tuple(ltxav_model_checkpoint_key(key) for key in model.state_dict())
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
    return tuple(key for key in checkpoint_keys if key not in available)


def _resolve_state_target(model: torch.nn.Module, state_key: str) -> tuple[torch.nn.Module, str, torch.Tensor, bool]:
    parent_name, _, tensor_name = state_key.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if tensor_name in parent._parameters:
        tensor = parent._parameters[tensor_name]
        if tensor is None:
            raise KeyError(f"parameter is None: {state_key}")
        return parent, tensor_name, tensor, True
    if tensor_name in parent._buffers:
        tensor = parent._buffers[tensor_name]
        if tensor is None:
            raise KeyError(f"buffer is None: {state_key}")
        return parent, tensor_name, tensor, False
    raise KeyError(f"state key not found in module: {state_key}")


def _assign_state_tensor(
    model: torch.nn.Module,
    state_key: str,
    tensor: torch.Tensor,
    *,
    assign: bool,
) -> None:
    parent, tensor_name, target, is_parameter = _resolve_state_target(model, state_key)
    if tuple(target.shape) != tuple(tensor.shape):
        raise ValueError(f"shape mismatch for {state_key}: expected {tuple(target.shape)}, got {tuple(tensor.shape)}")
    if assign or target.is_meta:
        if is_parameter:
            parent._parameters[tensor_name] = torch.nn.Parameter(tensor, requires_grad=target.requires_grad)
        else:
            parent._buffers[tensor_name] = tensor
        return
    with torch.no_grad():
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))


def load_ltxav_model_weights_streaming(
    model: LTXAVModel,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    assign: bool = False,
    strict: bool = True,
) -> LTXAVModelLoadReport:
    local_keys = tuple(model.state_dict().keys())
    checkpoint_pairs = tuple((local_key, ltxav_model_checkpoint_key(local_key)) for local_key in local_keys)
    with safe_open(str(checkpoint_path), framework="pt", device=str(device)) as handle:
        available = set(handle.keys())
        missing = tuple(checkpoint_key for _, checkpoint_key in checkpoint_pairs if checkpoint_key not in available)
        if strict and missing:
            raise KeyError(f"checkpoint keys not found: {missing[:8]}")
        loaded = 0
        for local_key, checkpoint_key in checkpoint_pairs:
            if checkpoint_key not in available:
                continue
            _assign_state_tensor(model, local_key, handle.get_tensor(checkpoint_key), assign=assign)
            loaded += 1
    return LTXAVModelLoadReport(loaded=loaded, missing=missing, unexpected=())


def apply_lora_to_ltxav_model(
    model: LTXAVModel,
    *,
    lora_path: str | Path,
    manifest: LoRAManifest,
    strength: float,
    strict: bool = False,
) -> LTXAVModelLoRAReport:
    local_keys = set(model.state_dict().keys())
    applied: list[str] = []
    skipped: list[str] = []
    with safe_open(str(lora_path), framework="pt", device="cpu") as handle:
        for pair in manifest.pairs:
            local_key = None
            for candidate in target_key_candidates(pair.target_key):
                local_key = ltxav_model_local_key(candidate, local_keys)
                if local_key is not None:
                    break
            if local_key is None:
                skipped.append(pair.target_key)
                continue
            _, _, target, _ = _resolve_state_target(model, local_key)
            if target.is_meta:
                raise ValueError(f"cannot apply LoRA to meta tensor: {local_key}")
            compute_dtype = _comfy_lowvram_lora_compute_dtype(target.device, target.dtype)
            lora_a = handle.get_tensor(pair.lora_a_key).to(device=target.device, dtype=compute_dtype)
            lora_b = handle.get_tensor(pair.lora_b_key).to(device=target.device, dtype=compute_dtype)
            delta = lora_pair_delta(
                lora_a,
                lora_b,
                target.shape,
                alpha=pair.alpha,
                strength=strength,
            ).to(dtype=compute_dtype)
            with torch.no_grad():
                patched = target.to(dtype=compute_dtype).add_(delta).to(dtype=target.dtype)
                target.copy_(patched)
            applied.append(local_key)
    if strict and skipped:
        raise KeyError(f"LoRA targets not found in LTXAV model: {skipped[:8]}")
    return LTXAVModelLoRAReport(
        matched=len(applied),
        skipped=len(skipped),
        applied_keys=tuple(applied),
        skipped_targets=tuple(skipped),
    )


def _comfy_lowvram_lora_compute_dtype(device: torch.device, target_dtype: torch.dtype) -> torch.dtype:
    # The MSR workflow loads LTXAV through ComfyUI DynamicVRAM/LowVramPatch.
    # In that path LoRA is calculated inside the layer's weight_function with
    # intermediate_dtype=weight.dtype, not model_management.lora_compute_dtype.
    if device.type == "cuda" and target_dtype in (torch.float16, torch.bfloat16):
        return target_dtype
    return torch.float32


def load_ltxav_model_weights(
    model: LTXAVModel,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    state_dict = load_ltxav_model_state_dict(model, checkpoint_path, device=device)
    return model.load_state_dict(state_dict, strict=strict)

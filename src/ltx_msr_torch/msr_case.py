from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dtypes import torch_dtype_from_cli
from .gemma_text_model import build_empty_gemma3_text_model, load_gemma_text_model_weights_streaming
from .gemma_tokenizer import GemmaTokenizer
from .iclora_guide import get_conditioning_value, prepare_and_append_iclora_video_guide
from .local_state import build_low_level_state
from .lora_loader import inspect_lora_manifest, resolve_lora_path
from .ltx_embeddings_connector import build_embeddings_connector_from_checkpoint
from .ltxav_model import apply_lora_to_ltxav_model, create_ltxav_model_from_checkpoint, load_ltxav_model_weights_streaming
from .ltxav_pipeline import (
    decode_ltxav_latents,
    sample_ltxav_workflow_latents_comfy_legacy_packed,
    sample_ltxav_workflow_latents_comfy_packed,
)
from .ltx_vae import load_ltxav_decoders_from_checkpoint
from .msr_debug import debug_msr_first_step
from .msr_reference import create_msr_reference_video_from_paths
from .nag import LTX2NAGConfig
from .prompt_relay import plan_prompt_relay
from .prompt_utils import parse_reference_prompt_file
from .text_conditioning import (
    attention_mask_tensor,
    build_text_conditioning_inputs_from_plan,
    connect_ltxav_text_embeddings,
    encode_ltx_text_conditioning,
)
from .text_projection import build_text_projection_from_checkpoint
from .torch_nodes import empty_ltxv_latent_audio
from .video_io import write_av_mp4
from .workflow_extract import extract_workflow_config


@dataclass
class PersistentMSRRuntime:
    """Models pinned across requests for a split-device MSR service."""

    workflow_path: Path
    config: Any
    state: Any
    dtype: Any
    text_device: Any
    model_device: Any
    layers: int
    tokenizer: Any
    gemma: Any
    gemma_report: Any
    projection: Any
    video_connector: Any
    audio_connector: Any
    decoders: Any
    model: Any
    model_report: Any
    lora_name: str
    lora_strength: float
    lora_report: Any
    apply_lora: bool

    def validate_request(self, args: argparse.Namespace) -> None:
        import torch

        workflow_path = Path(args.workflow).resolve()
        dtype = torch_dtype_from_cli(args.dtype)
        model_device = torch.device(args.device)
        lora_name, lora_strength = _requested_lora_settings(args, self.config)
        apply_lora = not bool(getattr(args, "no_apply_lora", False))
        mismatches: list[str] = []
        if workflow_path != self.workflow_path:
            mismatches.append(f"workflow={workflow_path}")
        if dtype != self.dtype:
            mismatches.append(f"dtype={dtype}")
        if model_device != self.model_device:
            mismatches.append(f"model_device={model_device}")
        if int(args.layers) != self.layers:
            mismatches.append(f"layers={args.layers}")
        if lora_name != self.lora_name:
            mismatches.append(f"lora_name={lora_name}")
        if lora_strength != self.lora_strength:
            mismatches.append(f"lora_strength={lora_strength}")
        if apply_lora != self.apply_lora:
            mismatches.append(f"apply_lora={apply_lora}")
        if mismatches:
            raise ValueError(
                "request is incompatible with the resident MSR runtime: "
                + ", ".join(mismatches)
            )

    def clear_transient_memory(self) -> None:
        import gc
        import torch

        gc.collect()
        for device in {self.text_device, self.model_device}:
            if device.type != "cuda":
                continue
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def load_persistent_msr_runtime(
    args: argparse.Namespace,
    *,
    text_device: str,
) -> PersistentMSRRuntime:
    """Load the text stack and generation stack once on separate devices."""
    import torch

    workflow_path = Path(args.workflow).resolve()
    config = extract_workflow_config(workflow_path)
    state = build_low_level_state(config, device="cpu")
    dtype = torch_dtype_from_cli(args.dtype)
    model_device = torch.device(args.device)
    resolved_text_device = torch.device(text_device)
    if model_device == resolved_text_device:
        raise ValueError(
            "persistent MSR runtime requires different text and model devices; "
            f"both resolved to {model_device}"
        )

    tokenizer = GemmaTokenizer.from_config_paths()
    gemma = build_empty_gemma3_text_model(
        device=resolved_text_device,
        dtype=dtype,
        num_layers=48,
    )
    gemma_report = load_gemma_text_model_weights_streaming(
        gemma,
        state.model_paths.text_encoder,
        device=resolved_text_device,
    )
    gemma.eval()
    projection = build_text_projection_from_checkpoint(
        state.model_paths.checkpoint,
        device=resolved_text_device,
    )
    video_connector = build_embeddings_connector_from_checkpoint(
        state.model_paths.checkpoint,
        "video",
        dtype=dtype,
        device=resolved_text_device,
    )
    audio_connector = build_embeddings_connector_from_checkpoint(
        state.model_paths.checkpoint,
        "audio",
        dtype=dtype,
        device=resolved_text_device,
    )

    decoders = load_ltxav_decoders_from_checkpoint(
        state.model_paths.checkpoint,
        dtype=dtype,
        device=model_device,
    )
    decoders.video_vae.eval()
    decoders.audio_vae.eval()
    model = create_ltxav_model_from_checkpoint(
        state.model_paths.checkpoint,
        dtype=dtype,
        device="meta",
        num_layers=int(args.layers),
    )
    model_report = load_ltxav_model_weights_streaming(
        model,
        state.model_paths.checkpoint,
        device=model_device,
        assign=True,
    )
    lora_name, lora_strength = _requested_lora_settings(args, config)
    apply_lora = not bool(getattr(args, "no_apply_lora", False))
    lora_report = None
    if apply_lora:
        lora_path = resolve_lora_path(lora_name)
        lora_manifest = inspect_lora_manifest(lora_path)
        lora_report = apply_lora_to_ltxav_model(
            model,
            lora_path=lora_path,
            manifest=lora_manifest,
            strength=lora_strength,
        )
    model.eval()

    return PersistentMSRRuntime(
        workflow_path=workflow_path,
        config=config,
        state=state,
        dtype=dtype,
        text_device=resolved_text_device,
        model_device=model_device,
        layers=int(args.layers),
        tokenizer=tokenizer,
        gemma=gemma,
        gemma_report=gemma_report,
        projection=projection,
        video_connector=video_connector,
        audio_connector=audio_connector,
        decoders=decoders,
        model=model,
        model_report=model_report,
        lora_name=lora_name,
        lora_strength=lora_strength,
        lora_report=lora_report,
        apply_lora=apply_lora,
    )


def generate_msr_case(
    args: argparse.Namespace,
    *,
    runtime: PersistentMSRRuntime | None = None,
) -> int:
    import torch

    workflow_path = Path(args.workflow).resolve()
    if runtime is not None:
        runtime.validate_request(args)
        config = runtime.config
        state = runtime.state
        dtype = runtime.dtype
        device = runtime.model_device
        text_device = runtime.text_device
        tokenizer = runtime.tokenizer
    else:
        config = extract_workflow_config(workflow_path)
        state = build_low_level_state(config, device="cpu")
        dtype = torch_dtype_from_cli(args.dtype)
        device = torch.device(args.device)
        text_device = device
        tokenizer = GemmaTokenizer.from_config_paths()
    case_dir = Path(args.case_dir)
    width = int(args.width) if args.width is not None else int(config.latent.width)
    height = int(args.height) if args.height is not None else int(config.latent.height)
    reference_frames = int(args.reference_frames) if args.reference_frames is not None else int(config.reference.frame_count)
    video_frames = int(args.video_frames) if args.video_frames is not None else int(config.latent.video_frames)
    if width < 32 or height < 32:
        raise ValueError("width and height must be at least 32")
    reference_width = int(args.reference_width) if args.reference_width is not None else int(config.reference.width)
    reference_height = int(args.reference_height) if args.reference_height is not None else int(config.reference.height)
    if reference_width <= 0 or reference_height <= 0:
        raise ValueError("reference-width and reference-height must be positive")
    if reference_frames <= 0 or video_frames <= 0:
        raise ValueError("reference-frames and video-frames must be positive")

    sigmas = _limited_sigmas(state.sigmas, args.max_sigmas).to(device=device)
    latent_frames = ((video_frames - 1) // 8) + 1
    latent_height = height // 32
    latent_width = width // 32

    prompt_path = Path(args.prompt_file) if args.prompt_file is not None else None
    workflow_reference_inputs = _resolve_workflow_licon_images(workflow_path, case_dir)
    subject_1 = Path(args.subject_1) if args.subject_1 is not None else workflow_reference_inputs.get("1", case_dir / "1.jpg")
    subject_2 = Path(args.subject_2) if args.subject_2 is not None else workflow_reference_inputs.get("2", case_dir / "2.jpg")
    subject_3 = (
        Path(args.subject_3)
        if getattr(args, "subject_3", None) is not None
        else workflow_reference_inputs.get("3")
    )
    subject_4 = (
        Path(args.subject_4)
        if getattr(args, "subject_4", None) is not None
        else workflow_reference_inputs.get("4")
    )
    background = (
        Path(args.background)
        if args.background is not None
        else workflow_reference_inputs.get("background", case_dir / "bg.png")
    )

    full_prompt = getattr(args, "full_prompt", None)
    if full_prompt is not None:
        global_prompt = str(full_prompt)
        local_prompts = ""
        promptrelay_plan = None
        token_plan = tokenizer.tokenize_with_weights(str(full_prompt))
    elif args.global_prompt is not None and args.local_prompts is not None:
        global_prompt = str(args.global_prompt)
        local_prompts = str(args.local_prompts)
        relay_plan = tokenizer.plan_prompt_relay_tokens(
            global_prompt=global_prompt,
            local_prompts=local_prompts,
        )
        promptrelay_plan = plan_prompt_relay(
            local_prompts=local_prompts,
            latent_shape=(1, 128, latent_frames, latent_height, latent_width),
            patch_size=(1, 1, 1),
            temporal_stride=8,
            segment_lengths=config.prompt.segment_lengths,
            token_ranges=relay_plan.token_ranges,
            epsilon=config.prompt.epsilon,
        )
        token_plan = tokenizer.tokenize_with_weights(relay_plan.full_prompt)
    elif prompt_path is not None:
        global_prompt, local_prompts = parse_reference_prompt_file(prompt_path)
        relay_plan = tokenizer.plan_prompt_relay_tokens(
            global_prompt=global_prompt,
            local_prompts=local_prompts,
        )
        promptrelay_plan = plan_prompt_relay(
            local_prompts=local_prompts,
            latent_shape=(1, 128, latent_frames, latent_height, latent_width),
            patch_size=(1, 1, 1),
            temporal_stride=8,
            segment_lengths=config.prompt.segment_lengths,
            token_ranges=relay_plan.token_ranges,
            epsilon=config.prompt.epsilon,
        )
        token_plan = tokenizer.tokenize_with_weights(relay_plan.full_prompt)
    else:
        global_prompt = config.prompt.global_prompt
        local_prompts = config.prompt.local_prompts
        relay_plan = tokenizer.plan_prompt_relay_tokens(
            global_prompt=global_prompt,
            local_prompts=local_prompts,
        )
        promptrelay_plan = plan_prompt_relay(
            local_prompts=local_prompts,
            latent_shape=(1, 128, latent_frames, latent_height, latent_width),
            patch_size=(1, 1, 1),
            temporal_stride=8,
            segment_lengths=config.prompt.segment_lengths,
            token_ranges=relay_plan.token_ranges,
            epsilon=config.prompt.epsilon,
        )
        token_plan = tokenizer.tokenize_with_weights(relay_plan.full_prompt)
    text_inputs = build_text_conditioning_inputs_from_plan(token_plan)
    input_ids = torch.tensor(text_inputs.token_ids, device=text_device, dtype=torch.long)
    text_mask = attention_mask_tensor(text_inputs, device=text_device)

    if runtime is not None:
        gemma = runtime.gemma
        gemma_report = runtime.gemma_report
        projection = runtime.projection
        video_connector = runtime.video_connector
        audio_connector = runtime.audio_connector
    else:
        gemma = build_empty_gemma3_text_model(device=text_device, dtype=dtype, num_layers=48)
        gemma_report = load_gemma_text_model_weights_streaming(
            gemma,
            state.model_paths.text_encoder,
            device=text_device,
        )
        gemma.eval()
        projection = None
        video_connector = None
        audio_connector = None
    with torch.inference_mode():
        gemma_output = gemma(
            input_ids=input_ids,
            attention_mask=text_mask,
            output_hidden_states=True,
        )
    all_layer_hidden = torch.stack(gemma_output.hidden_states, dim=1)
    if runtime is None:
        projection = build_text_projection_from_checkpoint(
            state.model_paths.checkpoint,
            device=text_device,
        )
        video_connector = build_embeddings_connector_from_checkpoint(
            state.model_paths.checkpoint,
            "video",
            dtype=dtype,
            device=text_device,
        )
        audio_connector = build_embeddings_connector_from_checkpoint(
            state.model_paths.checkpoint,
            "audio",
            dtype=dtype,
            device=text_device,
        )
    assert projection is not None
    assert video_connector is not None
    assert audio_connector is not None
    encoded = encode_ltx_text_conditioning(
        all_layer_hidden.to(dtype=projection.config.dtype),
        attention_mask=text_mask,
        projection=projection,
    )
    conditioning = encoded.conditioning.to(device=text_device, dtype=dtype)
    context_output = connect_ltxav_text_embeddings(
        conditioning,
        attention_mask=None,
        video_connector=video_connector,
        audio_connector=audio_connector,
    )
    negative_prompt = args.negative_prompt if args.negative_prompt is not None else config.prompt.negative_prompt
    negative_plan = tokenizer.tokenize_with_weights(negative_prompt)
    negative_inputs = build_text_conditioning_inputs_from_plan(negative_plan)
    negative_input_ids = torch.tensor(negative_inputs.token_ids, device=text_device, dtype=torch.long)
    negative_mask = attention_mask_tensor(negative_inputs, device=text_device)
    with torch.inference_mode():
        negative_output = gemma(
            input_ids=negative_input_ids,
            attention_mask=negative_mask,
            output_hidden_states=True,
        )
    negative_hidden = torch.stack(negative_output.hidden_states, dim=1)
    negative_encoded = encode_ltx_text_conditioning(
        negative_hidden.to(dtype=projection.config.dtype),
        attention_mask=negative_mask,
        projection=projection,
    )
    negative_conditioning = negative_encoded.conditioning.to(device=text_device, dtype=dtype)
    negative_context_output = connect_ltxav_text_embeddings(
        negative_conditioning,
        attention_mask=None,
        video_connector=video_connector,
        audio_connector=audio_connector,
    )
    del (
        gemma_output,
        all_layer_hidden,
        negative_output,
        negative_hidden,
        conditioning,
        negative_conditioning,
    )
    if runtime is None:
        del gemma, projection, video_connector, audio_connector
    if text_device.type == "cuda":
        with torch.cuda.device(text_device):
            torch.cuda.empty_cache()

    if runtime is not None:
        decoders = runtime.decoders
    else:
        decoders = load_ltxav_decoders_from_checkpoint(
            state.model_paths.checkpoint,
            dtype=dtype,
            device=device,
        )
        decoders.video_vae.eval()
        decoders.audio_vae.eval()

    reference = create_msr_reference_video_from_paths(
        subjects=[subject_1, subject_2, subject_3, subject_4],
        background=background,
        width=reference_width,
        height=reference_height,
        frame_count=reference_frames,
    )
    target_latent = {
        "samples": torch.zeros(
            (1, 128, latent_frames, latent_height, latent_width),
            dtype=torch.float32,
            device=device,
        )
    }
    seed_conditioning = [[torch.zeros(1, 1, context_output.context.shape[-1], device=device, dtype=dtype), {}]]
    guide = prepare_and_append_iclora_video_guide(
        video_vae=decoders.video_vae,
        positive=seed_conditioning,
        negative=seed_conditioning,
        latent=target_latent,
        image=reference,
        frame_idx=config.ic_lora_guide.frame_idx,
        strength=config.ic_lora_guide.strength,
        latent_downscale_factor=state.ic_lora.latent_downscale_factor,
        crop=config.ic_lora_guide.crop,
    )
    video_samples = guide.append.latent["samples"].to(device=device, dtype=torch.float32)
    denoise_mask = guide.append.latent["noise_mask"].to(device=device, dtype=torch.float32)

    audio_latent = empty_ltxv_latent_audio(
        frames_number=video_frames,
        frame_rate=int(config.latent.frame_rate),
        batch_size=int(config.latent.batch_size),
        audio_vae=decoders.audio_vae,
        device=device,
    )
    audio_latents = audio_latent["samples"].to(device=device, dtype=torch.float32)
    seed = int(args.seed) if args.seed is not None else int(config.sampling.seed)
    video_noise, audio_noise = _generate_av_noise_like(
        seed=seed,
        video_latents=video_samples.detach().cpu(),
        audio_latents=audio_latents.detach().cpu(),
    )
    video_noise = video_noise.to(device=device, dtype=torch.float32)
    audio_noise = audio_noise.to(device=device, dtype=torch.float32)
    audio_latent_image = audio_latents
    audio_denoise_mask = torch.ones_like(audio_latent_image, dtype=torch.float32)

    lora_name, lora_strength = _requested_lora_settings(args, config)
    if runtime is not None:
        model = runtime.model
        model_report = runtime.model_report
        lora_report = runtime.lora_report
    else:
        model = create_ltxav_model_from_checkpoint(
            state.model_paths.checkpoint,
            dtype=dtype,
            device="meta",
            num_layers=int(args.layers),
        )
        model_report = load_ltxav_model_weights_streaming(
            model,
            state.model_paths.checkpoint,
            device=device,
            assign=True,
        )
        lora_report = None
        if not args.no_apply_lora:
            lora_path = resolve_lora_path(lora_name)
            lora_manifest = inspect_lora_manifest(lora_path)
            lora_report = apply_lora_to_ltxav_model(
                model,
                lora_path=lora_path,
                manifest=lora_manifest,
                strength=lora_strength,
            )
        model.eval()

    keyframe_idxs = get_conditioning_value(guide.append.positive, "keyframe_idxs")
    guide_attention_entries = get_conditioning_value(guide.append.positive, "guide_attention_entries", [])
    transformer_options = {
        "a2v_cross_attn": bool(args.enable_av_cross),
        "v2a_cross_attn": bool(args.enable_av_cross),
    }
    if promptrelay_plan is not None:
        transformer_options["promptrelay_plan"] = promptrelay_plan
    if not bool(getattr(args, "disable_nag", False)):
        transformer_options.update(
            {
                "nag_config": LTX2NAGConfig(
                    scale=config.nag.scale,
                    alpha=config.nag.alpha,
                    tau=config.nag.tau,
                    inplace=config.nag.inplace,
                ),
                "nag_video_context": negative_context_output.video_context.to(device=device, dtype=dtype),
                "nag_audio_context": negative_context_output.audio_context.to(device=device, dtype=dtype),
            }
        )
    sampler_kwargs = {
        "model": model,
        "video_noise": video_noise,
        "audio_noise": audio_noise,
        "video_latent_image": video_samples,
        "audio_latent_image": audio_latent_image,
        "context": context_output.context.to(device=device, dtype=dtype),
        "raw_conditioning": encoded.conditioning.to(device=device),
        "negative_raw_conditioning": negative_encoded.conditioning.to(device=device),
        "attention_mask": context_output.attention_mask.to(device=device) if context_output.attention_mask is not None else None,
        "sigmas": sigmas,
        "frame_rate": float(config.latent.frame_rate),
        "transformer_options": transformer_options,
        "keyframe_idxs": keyframe_idxs.to(device=device) if isinstance(keyframe_idxs, torch.Tensor) else None,
        "denoise_mask": denoise_mask,
        "audio_denoise_mask": audio_denoise_mask,
        "guide_attention_entries": guide_attention_entries,
    }
    if bool(getattr(args, "debug_first_step", False)):
        debug_msr_first_step(
            sampler_kwargs=sampler_kwargs,
            negative_context=negative_context_output.context.to(device=device, dtype=dtype),
            negative_attention_mask=negative_context_output.attention_mask.to(device=device)
            if negative_context_output.attention_mask is not None
            else None,
            cfg=float(config.sampling.cfg),
            dump_path=getattr(args, "debug_first_step_dump", None),
        )
        return 0
    sampling_kwargs = {
        key: value
        for key, value in sampler_kwargs.items()
        if key not in {"raw_conditioning", "negative_raw_conditioning"}
    }
    if args.sampler_impl == "legacy-packed":
        sampled_video_latents, sampled_audio_latents = sample_ltxav_workflow_latents_comfy_legacy_packed(**sampling_kwargs)
    else:
        sampled_video_latents, sampled_audio_latents = sample_ltxav_workflow_latents_comfy_packed(
            **sampling_kwargs,
            negative_context=negative_context_output.context.to(device=device, dtype=dtype),
            negative_attention_mask=negative_context_output.attention_mask.to(device=device)
            if negative_context_output.attention_mask is not None
            else None,
            cfg=float(config.sampling.cfg),
        )
    crop_guides_removed = max(int(sampled_video_latents.shape[2]) - latent_frames, 0)
    sampled_video = sampled_video_latents[:, :, :latent_frames]
    decoded = decode_ltxav_latents(
        video_vae=decoders.video_vae,
        video_latents=sampled_video,
        audio_vae=decoders.audio_vae,
        audio_latents=sampled_audio_latents,
    )
    output_path = write_av_mp4(
        decoded.video,
        decoded.audio,
        args.output_video,
        fps=float(config.latent.frame_rate),
        sample_rate=int(getattr(decoders.audio_vae, "output_sample_rate", 48000)),
    )

    print(f"msr_case_case_dir={case_dir}")
    print(f"msr_case_prompt_file={prompt_path}")
    print(f"msr_case_subject_1={subject_1}")
    print(f"msr_case_subject_2={subject_2}")
    print(f"msr_case_subject_3={subject_3}")
    print(f"msr_case_subject_4={subject_4}")
    print(f"msr_case_background={background}")
    print(f"msr_case_workflow={workflow_path}")
    print(f"msr_case_output_video={output_path}")
    print(f"msr_case_width={width}")
    print(f"msr_case_height={height}")
    print(f"msr_case_reference_width={reference_width}")
    print(f"msr_case_reference_height={reference_height}")
    print(f"msr_case_video_frames={video_frames}")
    print(f"msr_case_reference_frames={reference_frames}")
    print(f"msr_case_seed={seed}")
    print(f"msr_case_sampler_path={args.sampler_impl}")
    print("msr_case_transformer_options=cond_or_uncond,sigmas,sample_sigmas")
    print(f"msr_case_workflow_sampler={config.sampling.sampler}")
    print(f"msr_case_workflow_cfg={config.sampling.cfg}")
    print(f"msr_case_latent_shape={tuple(video_samples.shape)}")
    print(f"msr_case_target_latent_frames={latent_frames}")
    print(f"msr_case_crop_guides_removed={crop_guides_removed}")
    print(f"msr_case_video_noise_shape={tuple(video_noise.shape)}")
    print(f"msr_case_audio_latent_shape={tuple(audio_latent_image.shape)}")
    print(f"msr_case_audio_noise_shape={tuple(audio_noise.shape)}")
    print(f"msr_case_audio_denoise_mask_shape={tuple(audio_denoise_mask.shape)}")
    print(f"msr_case_context_shape={tuple(context_output.context.shape)}")
    print(f"msr_case_text_real_tokens={text_inputs.real_token_count}")
    print(f"msr_case_negative_text_real_tokens={negative_inputs.real_token_count}")
    print(f"msr_case_negative_prompt={negative_prompt}")
    print(f"msr_case_promptrelay_segments={len(promptrelay_plan.segments) if promptrelay_plan is not None else 0}")
    print(f"msr_case_promptrelay_effective_lengths={promptrelay_plan.effective_lengths if promptrelay_plan is not None else ()}")
    print(f"msr_case_promptrelay_tokens_per_frame={promptrelay_plan.tokens_per_frame if promptrelay_plan is not None else 0}")
    print(f"msr_case_gemma_loaded_key_count={gemma_report.loaded}")
    print(f"msr_case_ltxav_layers={int(args.layers)}")
    print(f"msr_case_ltxav_loaded_key_count={model_report.loaded}")
    print(f"msr_case_sigmas={tuple(float(value) for value in sigmas.detach().cpu())}")
    print(f"msr_case_lora_applied={not args.no_apply_lora}")
    print(f"msr_case_lora_name={lora_name}")
    print(f"msr_case_lora_strength={lora_strength}")
    print(f"msr_case_nag_enabled={not bool(getattr(args, 'disable_nag', False))}")
    print(f"msr_case_nag_scale={config.nag.scale if not bool(getattr(args, 'disable_nag', False)) else 0.0}")
    print(f"msr_case_nag_alpha={config.nag.alpha if not bool(getattr(args, 'disable_nag', False)) else 0.0}")
    print(f"msr_case_nag_tau={config.nag.tau if not bool(getattr(args, 'disable_nag', False)) else 0.0}")
    if lora_report is not None:
        print(f"msr_case_lora_matched={lora_report.matched}")
        print(f"msr_case_lora_skipped={lora_report.skipped}")
    print(f"msr_case_guide_latent_shape={tuple(guide.guide_latent.shape)}")
    print(f"msr_case_keyframe_shape={tuple(keyframe_idxs.shape) if isinstance(keyframe_idxs, torch.Tensor) else None}")
    print(f"msr_case_sampled_video_finite={bool(torch.isfinite(sampled_video_latents).all().item())}")
    print(f"msr_case_sampled_audio_finite={bool(torch.isfinite(sampled_audio_latents).all().item())}")
    print(f"msr_case_decoded_video_shape={tuple(decoded.video.shape)}")
    print(f"msr_case_decoded_audio_shape={tuple(decoded.audio.shape) if decoded.audio is not None else None}")
    return 0


def _requested_lora_settings(args: argparse.Namespace, config: Any) -> tuple[str, float]:
    lora_name = str(getattr(args, "lora_name", None) or config.model.lora)
    requested_strength = getattr(args, "lora_strength", None)
    lora_strength = float(
        config.model.lora_strength
        if requested_strength is None
        else requested_strength
    )
    return lora_name, lora_strength


def _resolve_workflow_licon_images(workflow_path: Path, case_dir: Path) -> dict[str, Path]:
    workflow = json.loads(workflow_path.read_text())
    nodes = {str(node["id"]): node for node in workflow.get("nodes", [])}
    links = {int(link[0]): link for link in workflow.get("links", [])}
    licon_nodes = [node for node in workflow.get("nodes", []) if node.get("type") == "LiconMSR"]
    if len(licon_nodes) != 1:
        return {}

    resolved: dict[str, Path] = {}
    for input_def in licon_nodes[0].get("inputs") or []:
        name = str(input_def.get("name"))
        if name not in {"1", "2", "3", "4", "background"}:
            continue
        link_id = input_def.get("link")
        if link_id is None:
            continue
        link = links.get(int(link_id))
        if link is None:
            continue
        source = nodes.get(str(link[1]))
        if source is None or source.get("type") != "LoadImage":
            continue
        if int(source.get("mode", 0)) in {2, 4}:
            continue
        widgets = source.get("widgets_values") or []
        if not widgets:
            continue
        resolved[name] = _resolve_workflow_image_name(str(widgets[0]), case_dir)
    return resolved


def _resolve_workflow_image_name(image_name: str, case_dir: Path) -> Path:
    image_path = Path(image_name)
    candidates = [image_path] if image_path.is_absolute() else [case_dir / image_name]
    if image_name.startswith("bg "):
        candidates.append(case_dir / "bg.png")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve workflow image {image_name!r}; tried {candidates}")


def _limited_sigmas(sigmas, max_sigmas: int):
    import torch

    if max_sigmas <= 0 or max_sigmas >= int(sigmas.numel()):
        return sigmas
    if max_sigmas < 2:
        raise ValueError("max-sigmas must be at least 2, or <=0 for all sigmas")
    return sigmas[torch.tensor([*range(max_sigmas - 1), int(sigmas.numel()) - 1])]


def _generate_av_noise_like(
    *,
    seed: int,
    video_latents,
    audio_latents,
):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    video_noise = torch.randn(
        video_latents.shape,
        dtype=torch.float32,
        layout=video_latents.layout,
        device="cpu",
        generator=generator,
    )
    audio_noise = torch.randn(
        audio_latents.shape,
        dtype=torch.float32,
        layout=audio_latents.layout,
        device="cpu",
        generator=generator,
    )
    return video_noise, audio_noise

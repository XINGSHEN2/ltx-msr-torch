from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint_loader import apply_lora_to_checkpoint_subset, inspect_checkpoint_manifest
from .comfy_api_prompt import build_case_api_prompt, save_api_prompt
from .comfy_client import load_api_prompt, queue_prompt, wait_for_history
from .gemma_tokenizer import GemmaTokenizer
from .gemma_text_model import (
    build_empty_gemma3_text_model,
    inspect_gemma_text_model_compatibility,
    load_gemma3_text_config,
    load_gemma_text_model_weights_streaming,
)
from .local_state import build_low_level_state
from .lora_apply import target_key_candidates
from .ltx_embeddings_connector import build_embeddings_connector_from_checkpoint
from .ltxav_model import (
    apply_lora_to_ltxav_model,
    create_ltxav_model_from_checkpoint,
    load_ltxav_model_weights_streaming,
    ltxav_model_local_key,
    missing_ltxav_model_checkpoint_keys,
)
from .ltxav_transformer import inspect_ltxav_transformer_manifest
from .ltx_vae import (
    build_ltx_audio_vae_from_checkpoint,
    build_ltx_video_vae_from_checkpoint,
    load_ltxav_decoders_from_checkpoint,
    load_ltx_audio_vae_state_dict,
    load_ltx_video_vae_state_dict,
    missing_ltx_vae_keys,
)
from .ltxav_denoiser import LTXAVDenoiser, sample_ltxav_euler
from .ltxav_pipeline import decode_ltxav_latents, run_ltxav_sample_decode
from .lora_apply import match_lora_targets
from .lora_loader import inspect_lora_manifest, resolve_lora_path
from .model_inspect import inspect_workflow_model_headers
from .msr_reference import create_msr_reference_video_from_paths
from .prompt_utils import parse_reference_prompt_file
from .text_encoder_sections import inspect_text_encoder_section
from .text_conditioning import (
    build_text_conditioning_inputs_from_plan,
    attention_mask_tensor,
    connect_ltxav_text_embeddings,
)
from .text_projection import build_text_projection_from_checkpoint
from .vae_sections import inspect_vae_section
from .workflow_extract import extract_workflow_config
from .workflow_config import default_workflow_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ltx-msr-torch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_reference = subparsers.add_parser(
        "build-reference",
        help="Build the LiconMSR-compatible fixed-frame reference tensor.",
    )
    for index in range(1, 5):
        build_reference.add_argument(f"--subject-{index}", default=None)
    build_reference.add_argument("--background", required=True)
    build_reference.add_argument("--output", required=True)
    build_reference.add_argument("--width", type=int, default=None)
    build_reference.add_argument("--height", type=int, default=None)
    build_reference.add_argument("--frame-count", type=int, default=None)

    inspect_config = subparsers.add_parser(
        "inspect-config",
        help="Print the extracted parity config.",
    )
    inspect_config.add_argument(
        "--workflow",
        default=None,
        help="Optional ComfyUI workflow JSON to extract config from.",
    )

    build_api_prompt = subparsers.add_parser(
        "build-api-prompt",
        help="Build a ComfyUI API prompt from the MSR workflow and a sample case.",
    )
    build_api_prompt.add_argument(
        "--workflow",
        default="/home/xingshen/ComfyUI/custom_nodes/ComfyUI-Licon-MSR/LTX-2.3_MSR_sample_workflow_V2.json",
    )
    build_api_prompt.add_argument("--case-dir", required=True)
    build_api_prompt.add_argument("--output", required=True)
    build_api_prompt.add_argument(
        "--output-prefix",
        default="LTX-2/MSR_torch_parity",
        help="ComfyUI output filename prefix for the SaveVideo node.",
    )

    submit_api_prompt = subparsers.add_parser(
        "submit-api-prompt",
        help="Submit an API prompt JSON to a running ComfyUI server.",
    )
    submit_api_prompt.add_argument("--prompt", required=True)
    submit_api_prompt.add_argument("--server", default="127.0.0.1:8188")
    submit_api_prompt.add_argument("--wait", action="store_true")
    submit_api_prompt.add_argument("--timeout-seconds", type=float, default=None)

    inspect_local = subparsers.add_parser(
        "inspect-local-state",
        help="Print local torch replacements for parity-critical low-level nodes.",
    )
    inspect_local.add_argument("--device", default="cpu")

    inspect_models = subparsers.add_parser(
        "inspect-model-headers",
        help="Inspect workflow safetensors headers without loading full weights.",
    )

    inspect_lora = subparsers.add_parser(
        "inspect-lora-manifest",
        help="Inspect workflow LoRA A/B tensor pairs without applying weights.",
    )

    inspect_checkpoint = subparsers.add_parser(
        "inspect-checkpoint",
        help="Inspect workflow checkpoint sections and key counts.",
    )

    inspect_text_projection = subparsers.add_parser(
        "inspect-text-projection",
        help="Load and inspect the workflow text embedding projection module.",
    )

    inspect_vae = subparsers.add_parser(
        "inspect-vae-sections",
        help="Inspect video/audio VAE checkpoint sections.",
    )

    inspect_text_encoder = subparsers.add_parser(
        "inspect-text-encoder",
        help="Inspect workflow Gemma text encoder checkpoint and config files.",
    )

    inspect_tokenizer = subparsers.add_parser(
        "inspect-tokenizer",
        help="Tokenize a sample case prompt and print PromptRelay token ranges.",
    )
    inspect_tokenizer.add_argument("--case-dir", default="sample_cases/validition_v1_01")

    inspect_text_conditioning = subparsers.add_parser(
        "inspect-text-conditioning",
        help="Inspect local LTXAV text conditioning mask, trim, and projection shape.",
    )
    inspect_text_conditioning.add_argument("--case-dir", default="sample_cases/validition_v1_01")
    inspect_text_conditioning.add_argument("--device", default="cpu")

    inspect_gemma_model = subparsers.add_parser(
        "inspect-gemma-text-model",
        help="Inspect local Transformers Gemma3 text model compatibility with workflow weights.",
    )
    smoke_gemma_text_forward = subparsers.add_parser(
        "smoke-gemma-text-forward",
        help="Load a prefix of the real Gemma text encoder weights and run a minimal torch forward pass.",
    )
    smoke_gemma_text_forward.add_argument("--layers", type=int, default=1)
    smoke_gemma_text_forward.add_argument("--tokens", type=int, default=8)
    smoke_gemma_text_forward.add_argument("--device", default="cpu")
    smoke_gemma_text_forward.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    smoke_gemma_text_forward.add_argument("--case-dir", default="sample_cases/validition_v1_01")
    smoke_text_conditioning = subparsers.add_parser(
        "smoke-text-conditioning",
        help="Run Gemma, text projection, and LTXAV embedding connectors with real workflow weights.",
    )
    smoke_text_conditioning.add_argument("--layers", type=int, default=48)
    smoke_text_conditioning.add_argument("--min-length", type=int, default=128)
    smoke_text_conditioning.add_argument("--device", default="cpu")
    smoke_text_conditioning.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    smoke_text_conditioning.add_argument("--case-dir", default="sample_cases/validition_v1_01")

    inspect_transformer = subparsers.add_parser(
        "inspect-ltxav-transformer",
        help="Inspect local LTXAV diffusion transformer config and key manifest.",
    )
    inspect_ltxav_model = subparsers.add_parser(
        "inspect-ltxav-model",
        help="Build the local torch LTXAV model on meta and verify checkpoint key coverage.",
    )
    inspect_ltxav_decoders = subparsers.add_parser(
        "inspect-ltxav-decoders",
        help="Build local torch LTX video/audio decoders and verify checkpoint key coverage.",
    )
    inspect_ltxav_pipeline = subparsers.add_parser(
        "inspect-ltxav-pipeline",
        help="Verify the local torch LTXAV model, sampler adapter, and decoders can be wired together.",
    )
    smoke_ltxav_model_forward = subparsers.add_parser(
        "smoke-ltxav-model-forward",
        help="Load a prefix of the real LTXAV transformer weights and run a minimal torch forward pass.",
    )
    smoke_ltxav_model_forward.add_argument("--layers", type=int, default=1)
    smoke_ltxav_model_forward.add_argument("--device", default="cpu")
    smoke_ltxav_model_forward.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    smoke_ltxav_model_forward.add_argument("--enable-av-cross", action="store_true")
    smoke_ltxav_sampling = subparsers.add_parser(
        "smoke-ltxav-sampling",
        help="Run minimal pure-torch LTXAV sampling with real transformer weights.",
    )
    smoke_ltxav_sampling.add_argument("--layers", type=int, default=1)
    smoke_ltxav_sampling.add_argument("--device", default="cpu")
    smoke_ltxav_sampling.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    smoke_ltxav_sampling.add_argument("--decode", action="store_true")
    smoke_ltxav_sampling.add_argument("--apply-lora", action="store_true")
    smoke_ltxav_sampling.add_argument("--enable-av-cross", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build-reference":
        return _build_reference(args)
    if args.command == "inspect-config":
        return _inspect_config(args)
    if args.command == "build-api-prompt":
        return _build_api_prompt(args)
    if args.command == "submit-api-prompt":
        return _submit_api_prompt(args)
    if args.command == "inspect-local-state":
        return _inspect_local_state(args)
    if args.command == "inspect-model-headers":
        return _inspect_model_headers()
    if args.command == "inspect-lora-manifest":
        return _inspect_lora_manifest()
    if args.command == "inspect-checkpoint":
        return _inspect_checkpoint()
    if args.command == "inspect-text-projection":
        return _inspect_text_projection()
    if args.command == "inspect-vae-sections":
        return _inspect_vae_sections()
    if args.command == "inspect-text-encoder":
        return _inspect_text_encoder()
    if args.command == "inspect-tokenizer":
        return _inspect_tokenizer(args)
    if args.command == "inspect-text-conditioning":
        return _inspect_text_conditioning(args)
    if args.command == "inspect-gemma-text-model":
        return _inspect_gemma_text_model()
    if args.command == "smoke-gemma-text-forward":
        return _smoke_gemma_text_forward(args)
    if args.command == "smoke-text-conditioning":
        return _smoke_text_conditioning(args)
    if args.command == "inspect-ltxav-transformer":
        return _inspect_ltxav_transformer()
    if args.command == "inspect-ltxav-model":
        return _inspect_ltxav_model()
    if args.command == "inspect-ltxav-decoders":
        return _inspect_ltxav_decoders()
    if args.command == "inspect-ltxav-pipeline":
        return _inspect_ltxav_pipeline()
    if args.command == "smoke-ltxav-model-forward":
        return _smoke_ltxav_model_forward(args)
    if args.command == "smoke-ltxav-sampling":
        return _smoke_ltxav_sampling(args)
    raise AssertionError(f"unhandled command: {args.command}")


def _build_reference(args: argparse.Namespace) -> int:
    config = default_workflow_config().reference
    width = args.width or config.width
    height = args.height or config.height
    frame_count = args.frame_count or config.frame_count
    subjects = [getattr(args, f"subject_{index}") for index in range(1, 5)]

    tensor = create_msr_reference_video_from_paths(
        subjects=subjects,
        background=args.background,
        width=width,
        height=height,
        frame_count=frame_count,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch

    torch.save(tensor, output)
    print(
        f"saved {output} shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} min={tensor.min().item():.6f} max={tensor.max().item():.6f}"
    )
    return 0


def _inspect_config(args: argparse.Namespace) -> int:
    if args.workflow:
        config = extract_workflow_config(args.workflow)
    else:
        config = default_workflow_config()
    print(config)
    return 0


def _build_api_prompt(args: argparse.Namespace) -> int:
    prompt = build_case_api_prompt(
        workflow_path=args.workflow,
        case_dir=args.case_dir,
        output_prefix=args.output_prefix,
    )
    save_api_prompt(prompt, args.output)
    print(f"saved API prompt {args.output} nodes={len(prompt)}")
    return 0


def _submit_api_prompt(args: argparse.Namespace) -> int:
    prompt = load_api_prompt(args.prompt)
    response = queue_prompt(prompt, server=args.server)
    print(response)
    prompt_id = response.get("prompt_id")
    if args.wait and prompt_id:
        history = wait_for_history(
            prompt_id,
            server=args.server,
            timeout_seconds=args.timeout_seconds,
        )
        print(history)
    return 0


def _inspect_local_state(args: argparse.Namespace) -> int:
    config = default_workflow_config()
    state = build_low_level_state(config, device=args.device)
    print(f"reference_width={state.width}")
    print(f"reference_height={state.height}")
    print(f"reference_frame_count={state.frame_count}")
    print(f"video_length={state.video_length}")
    print(f"video_latent_shape={tuple(state.video_latent['samples'].shape)}")
    print(f"video_latent_downscale_ratio_spacial={state.video_latent['downscale_ratio_spacial']}")
    print(f"sigmas={state.sigmas.tolist()}")
    print(f"noise_seed={state.noise.seed}")
    print(f"iclora_guide_frame_idx={state.ic_lora_guide.frame_idx}")
    print(f"iclora_guide_latent_idx={state.ic_lora_guide.latent_idx}")
    print(f"iclora_guide_num_frames_to_keep={state.ic_lora_guide.num_frames_to_keep}")
    print(f"iclora_guide_causal_fix={state.ic_lora_guide.causal_fix}")
    print(f"iclora_guide_encode_frame_count={state.ic_lora_guide.encode_frame_count}")
    print(f"iclora_guide_estimated_shape={state.ic_lora_guide.estimated_guide_latent_shape}")
    print(f"iclora_guide_estimated_tokens={state.ic_lora_guide.estimated_tokens_added}")
    print(f"iclora_guide_target_size={state.ic_lora_guide.target_width}x{state.ic_lora_guide.target_height}")
    print(f"nag_scale={state.nag_patch.config.scale}")
    print(f"nag_alpha={state.nag_patch.config.alpha}")
    print(f"nag_tau={state.nag_patch.config.tau}")
    print(f"nag_inplace={state.nag_patch.config.inplace}")
    print(f"nag_transformer_block_count={state.nag_patch.transformer_block_count}")
    print(f"nag_video_patch_count={len(state.nag_patch.video_patch_targets)}")
    print(f"nag_audio_patch_count={len(state.nag_patch.audio_patch_targets)}")
    print(f"sampler_name={state.sampler.sampler_name}")
    print(f"sampler_cfg={state.sampler.cfg}")
    print(f"sampler_step_count={state.sampler.step_count}")
    print(f"sampler_sigma_count={state.sampler.sigma_count}")
    print(f"sampler_first_sigma={state.sampler.first_sigma}")
    print(f"sampler_last_sigma={state.sampler.last_sigma}")
    print(f"checkpoint={state.model_paths.checkpoint}")
    print(f"text_encoder={state.model_paths.text_encoder}")
    print(f"lora={state.model_paths.lora}")
    print(f"lora_strength={state.ic_lora.strength_model}")
    print(f"lora_reference_downscale_factor={state.ic_lora.latent_downscale_factor}")
    return 0


def _inspect_model_headers() -> int:
    state = build_low_level_state(default_workflow_config(), device="cpu")
    inspection = inspect_workflow_model_headers(state.model_paths)
    for label in ("checkpoint", "text_encoder", "lora"):
        item = getattr(inspection, label)
        print(f"{label}_path={item.path}")
        print(f"{label}_key_count={item.key_count}")
        print(f"{label}_first_keys={list(item.first_keys)}")
        print(f"{label}_metadata_keys={sorted((item.metadata or {}).keys())}")
    return 0


def _inspect_lora_manifest() -> int:
    config = default_workflow_config()
    state = build_low_level_state(config, device="cpu")
    path = resolve_lora_path(config.model.lora)
    manifest = inspect_lora_manifest(path)
    from safetensors import safe_open

    with safe_open(str(state.model_paths.checkpoint), framework="pt", device="cpu") as handle:
        checkpoint_keys = set(handle.keys())
    checkpoint_matches = match_lora_targets(checkpoint_keys, manifest)
    checkpoint_match_count = sum(1 for match in checkpoint_matches if match.state_key is not None)
    ranks = sorted({pair.rank for pair in manifest.pairs})
    print(f"lora_path={manifest.path}")
    print(f"lora_key_count={manifest.key_count}")
    print(f"lora_pair_count={manifest.pair_count}")
    print(f"lora_checkpoint_target_matches={checkpoint_match_count}")
    print(f"lora_ranks={ranks}")
    print(f"lora_unpaired_key_count={len(manifest.unpaired_keys)}")
    print(f"lora_metadata_keys={sorted((manifest.metadata or {}).keys())}")
    apply_result = apply_lora_to_checkpoint_subset(
        state.model_paths.checkpoint,
        lora_path=path,
        manifest=manifest,
        strength=0.0,
    )
    print(f"lora_checkpoint_subset_apply_matched={apply_result.report_matched}")
    print(f"lora_checkpoint_subset_apply_skipped={apply_result.report_skipped}")
    for index, pair in enumerate(manifest.pairs[:8]):
        print(
            f"pair_{index}={pair.target_key} "
            f"A={pair.lora_a_shape} B={pair.lora_b_shape} rank={pair.rank} alpha={pair.alpha}"
        )
    return 0


def _inspect_checkpoint() -> int:
    state = build_low_level_state(default_workflow_config(), device="cpu")
    manifest = inspect_checkpoint_manifest(state.model_paths.checkpoint)
    print(f"checkpoint_path={manifest.path}")
    print(f"checkpoint_key_count={manifest.key_count}")
    for section in manifest.sections:
        print(f"checkpoint_section_{section.name}_key_count={section.key_count}")
        print(f"checkpoint_section_{section.name}_first_keys={list(section.first_keys)}")
    print(f"checkpoint_unknown_key_count={len(manifest.unknown_keys)}")
    return 0


def _inspect_text_projection() -> int:
    state = build_low_level_state(default_workflow_config(), device="cpu")
    module = build_text_projection_from_checkpoint(state.model_paths.checkpoint)
    config = module.config
    print(f"text_projection_input_dim={config.input_dim}")
    print(f"text_projection_video_dim={config.video_dim}")
    print(f"text_projection_audio_dim={config.audio_dim}")
    print(f"text_projection_dtype={config.dtype}")
    print(f"text_projection_video_weight_shape={tuple(module.video_aggregate_embed.weight.shape)}")
    print(f"text_projection_audio_weight_shape={tuple(module.audio_aggregate_embed.weight.shape)}")
    return 0


def _inspect_vae_sections() -> int:
    state = build_low_level_state(default_workflow_config(), device="cpu")
    for prefix in ("vae", "audio_vae"):
        manifest = inspect_vae_section(state.model_paths.checkpoint, prefix=prefix)
        print(f"{prefix}_key_count={manifest.key_count}")
        print(f"{prefix}_encoder_key_count={manifest.encoder_key_count}")
        print(f"{prefix}_decoder_key_count={manifest.decoder_key_count}")
        print(f"{prefix}_statistics_key_count={manifest.statistics_key_count}")
        print(f"{prefix}_first_shapes={manifest.first_shapes}")
    return 0


def _inspect_text_encoder() -> int:
    state = build_low_level_state(default_workflow_config(), device="cpu")
    manifest = inspect_text_encoder_section(state.model_paths.text_encoder)
    print(f"text_encoder_path={manifest.path}")
    print(f"text_encoder_key_count={manifest.key_count}")
    print(f"text_encoder_text_model_key_count={manifest.text_model_key_count}")
    print(f"text_encoder_vision_model_key_count={manifest.vision_model_key_count}")
    print(f"text_encoder_projector_key_count={manifest.projector_key_count}")
    print(f"text_encoder_spiece_key_count={manifest.spiece_key_count}")
    print(f"text_encoder_layer_count={manifest.layer_count}")
    print(f"text_encoder_config_dir={manifest.config_paths.config_dir}")
    print(f"text_encoder_first_text_shapes={manifest.first_text_shapes}")
    return 0


def _inspect_tokenizer(args: argparse.Namespace) -> int:
    global_prompt, local_prompts = parse_reference_prompt_file(Path(args.case_dir) / "prompt.txt")
    tokenizer = GemmaTokenizer.from_config_paths()
    plan = tokenizer.plan_prompt_relay_tokens(
        global_prompt=global_prompt,
        local_prompts=local_prompts,
    )
    token_weight_plan = tokenizer.tokenize_with_weights(plan.full_prompt)
    print(f"tokenizer_full_prompt_token_count={len(plan.input_ids)}")
    print(f"tokenizer_padded_token_count={len(token_weight_plan.padded_input_ids)}")
    print(f"tokenizer_attention_real_count={sum(token_weight_plan.attention_mask)}")
    print(f"tokenizer_local_prompt_count={len(plan.local_prompts)}")
    print(f"tokenizer_token_ranges={plan.token_ranges}")
    print(f"tokenizer_first_token_ids={plan.input_ids[:16]}")
    print(f"tokenizer_first_padded_token_ids={token_weight_plan.padded_input_ids[:16]}")
    return 0


def _inspect_text_conditioning(args: argparse.Namespace) -> int:
    global_prompt, local_prompts = parse_reference_prompt_file(Path(args.case_dir) / "prompt.txt")
    tokenizer = GemmaTokenizer.from_config_paths()
    relay_plan = tokenizer.plan_prompt_relay_tokens(
        global_prompt=global_prompt,
        local_prompts=local_prompts,
    )
    token_plan = tokenizer.tokenize_with_weights(relay_plan.full_prompt)
    inputs = build_text_conditioning_inputs_from_plan(token_plan)
    mask = attention_mask_tensor(inputs, device=args.device)
    state = build_low_level_state(default_workflow_config(), device="cpu")
    projection = build_text_projection_from_checkpoint(state.model_paths.checkpoint, device=args.device)
    output_dim = projection.config.video_dim + projection.config.audio_dim
    print(f"text_conditioning_token_count={inputs.real_token_count}")
    print(f"text_conditioning_attention_mask_shape={tuple(mask.shape)}")
    print(f"text_conditioning_projection_dtype={projection.config.dtype}")
    print(f"text_conditioning_projection_input_dim={projection.config.input_dim}")
    print(f"text_conditioning_expected_output_shape={(1, inputs.real_token_count, output_dim)}")
    print("text_conditioning_extra={'unprocessed_ltxav_embeds': True}")
    return 0


def _inspect_gemma_text_model() -> int:
    config = load_gemma3_text_config()
    compatibility = inspect_gemma_text_model_compatibility()
    print(f"gemma_text_hidden_size={config.hidden_size}")
    print(f"gemma_text_intermediate_size={config.intermediate_size}")
    print(f"gemma_text_layer_count={config.num_hidden_layers}")
    print(f"gemma_text_attention_heads={config.num_attention_heads}")
    print(f"gemma_text_key_value_heads={config.num_key_value_heads}")
    print(f"gemma_text_head_dim={config.head_dim}")
    print(f"gemma_text_vocab_size={config.vocab_size}")
    print(f"gemma_text_checkpoint_key_count={compatibility.checkpoint_key_count}")
    print(f"gemma_text_hf_key_count={compatibility.hf_key_count}")
    print(f"gemma_text_matched_key_count={compatibility.matched_key_count}")
    print(f"gemma_text_exact_key_match={compatibility.is_exact_match}")
    print(f"gemma_text_missing_hf_key_count={len(compatibility.missing_hf_keys)}")
    print(f"gemma_text_unexpected_checkpoint_key_count={len(compatibility.unexpected_checkpoint_keys)}")
    return 0


def _smoke_gemma_text_forward(args: argparse.Namespace) -> int:
    import torch

    state = build_low_level_state(default_workflow_config(), device="cpu")
    global_prompt, local_prompts = parse_reference_prompt_file(Path(args.case_dir) / "prompt.txt")
    tokenizer = GemmaTokenizer.from_config_paths()
    relay_plan = tokenizer.plan_prompt_relay_tokens(
        global_prompt=global_prompt,
        local_prompts=local_prompts,
    )
    token_ids = relay_plan.input_ids[: args.tokens]
    if not token_ids:
        raise ValueError("prompt produced no Gemma token ids")
    device = torch.device(args.device)
    dtype = _torch_dtype_from_cli(args.dtype)
    model = build_empty_gemma3_text_model(device=device, dtype=dtype, num_layers=args.layers)
    report = load_gemma_text_model_weights_streaming(
        model,
        state.model_paths.text_encoder,
        device=device,
    )
    model.eval()
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hidden_states = output.hidden_states
    last_hidden = output.last_hidden_state
    print(f"gemma_forward_smoke_text_encoder={state.model_paths.text_encoder}")
    print(f"gemma_forward_smoke_layers={model.config.num_hidden_layers}")
    print(f"gemma_forward_smoke_loaded_key_count={report.loaded}")
    print(f"gemma_forward_smoke_device={device}")
    print(f"gemma_forward_smoke_token_count={input_ids.shape[1]}")
    print(f"gemma_forward_smoke_hidden_state_count={len(hidden_states)}")
    print(f"gemma_forward_smoke_last_hidden_shape={tuple(last_hidden.shape)}")
    print(f"gemma_forward_smoke_last_hidden_finite={bool(torch.isfinite(last_hidden).all().item())}")
    return 0


def _smoke_text_conditioning(args: argparse.Namespace) -> int:
    import math
    import torch

    state = build_low_level_state(default_workflow_config(), device="cpu")
    global_prompt, local_prompts = parse_reference_prompt_file(Path(args.case_dir) / "prompt.txt")
    tokenizer = GemmaTokenizer.from_config_paths()
    relay_plan = tokenizer.plan_prompt_relay_tokens(
        global_prompt=global_prompt,
        local_prompts=local_prompts,
    )
    minimum = max(args.min_length, math.ceil(len(relay_plan.input_ids) / 128) * 128)
    token_plan = tokenizer.tokenize_with_weights(relay_plan.full_prompt, min_length=minimum)
    token_ids = list(token_plan.padded_input_ids)
    token_mask = list(token_plan.attention_mask)
    remainder = len(token_ids) % 128
    if remainder:
        pad_count = 128 - remainder
        token_ids = [tokenizer.pad_token_id] * pad_count + token_ids
        token_mask = [0] * pad_count + token_mask
    device = torch.device(args.device)
    dtype = _torch_dtype_from_cli(args.dtype)
    model = build_empty_gemma3_text_model(device=device, dtype=dtype, num_layers=args.layers)
    report = load_gemma_text_model_weights_streaming(
        model,
        state.model_paths.text_encoder,
        device=device,
    )
    input_ids = torch.tensor([token_ids], device=device, dtype=torch.long)
    attention_mask = torch.tensor([token_mask], device=device, dtype=torch.long)
    with torch.inference_mode():
        gemma_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    all_layer_hidden = torch.stack(gemma_output.hidden_states, dim=1)
    projection = build_text_projection_from_checkpoint(state.model_paths.checkpoint, device=device)
    conditioning = projection(all_layer_hidden).to(dtype=torch.float32)
    video_connector = build_embeddings_connector_from_checkpoint(
        state.model_paths.checkpoint,
        "video",
        dtype=dtype,
        device=device,
    )
    audio_connector = build_embeddings_connector_from_checkpoint(
        state.model_paths.checkpoint,
        "audio",
        dtype=dtype,
        device=device,
    )
    context_output = connect_ltxav_text_embeddings(
        conditioning.to(device=device, dtype=dtype),
        attention_mask=attention_mask.to(device=device),
        video_connector=video_connector,
        audio_connector=audio_connector,
    )
    print(f"text_conditioning_smoke_checkpoint={state.model_paths.checkpoint}")
    print(f"text_conditioning_smoke_text_encoder={state.model_paths.text_encoder}")
    print(f"text_conditioning_smoke_gemma_layers={model.config.num_hidden_layers}")
    print(f"text_conditioning_smoke_gemma_loaded_key_count={report.loaded}")
    print(f"text_conditioning_smoke_token_count={input_ids.shape[1]}")
    print(f"text_conditioning_smoke_real_token_count={sum(token_mask)}")
    print(f"text_conditioning_smoke_hidden_shape={tuple(all_layer_hidden.shape)}")
    print(f"text_conditioning_smoke_conditioning_shape={tuple(conditioning.shape)}")
    print(f"text_conditioning_smoke_context_shape={tuple(context_output.context.shape)}")
    print(f"text_conditioning_smoke_attention_mask_shape={tuple(context_output.attention_mask.shape)}")
    print(f"text_conditioning_smoke_context_finite={bool(torch.isfinite(context_output.context).all().item())}")
    return 0


def _inspect_ltxav_transformer() -> int:
    state = build_low_level_state(default_workflow_config(), device="cpu")
    manifest = inspect_ltxav_transformer_manifest(state.model_paths.checkpoint)
    config = manifest.config
    print(f"ltxav_transformer_key_count={manifest.key_count}")
    print(f"ltxav_transformer_block_count={manifest.block_count}")
    print(f"ltxav_transformer_block_key_count={manifest.block_key_count}")
    print(f"ltxav_transformer_keys_per_block={manifest.keys_per_block}")
    print(f"ltxav_transformer_image_model={config.image_model}")
    print(f"ltxav_transformer_in_channels={config.in_channels}")
    print(f"ltxav_transformer_out_channels={config.out_channels}")
    print(f"ltxav_transformer_cross_attention_dim={config.cross_attention_dim}")
    print(f"ltxav_transformer_audio_cross_attention_dim={config.audio_cross_attention_dim}")
    print(f"ltxav_transformer_attention_head_dim={config.attention_head_dim}")
    print(f"ltxav_transformer_audio_attention_head_dim={config.audio_attention_head_dim}")
    print(f"ltxav_transformer_connector_num_layers={config.connector_num_layers}")
    print(f"ltxav_transformer_rope_type={config.rope_type}")
    print(f"ltxav_transformer_frequencies_precision={config.frequencies_precision}")
    print(f"ltxav_transformer_group_counts={manifest.group_counts}")
    print(f"ltxav_transformer_specs={manifest.specs}")
    return 0


def _inspect_ltxav_model() -> int:
    state = build_low_level_state(default_workflow_config(), device="cpu")
    model = create_ltxav_model_from_checkpoint(state.model_paths.checkpoint, device="meta")
    missing = missing_ltxav_model_checkpoint_keys(model, state.model_paths.checkpoint)
    lora_path = resolve_lora_path(default_workflow_config().model.lora)
    lora_manifest = inspect_lora_manifest(lora_path)
    local_keys = set(model.state_dict())
    mapped_lora_targets = sum(
        1
        for pair in lora_manifest.pairs
        if any(ltxav_model_local_key(candidate, local_keys) is not None for candidate in target_key_candidates(pair.target_key))
    )
    config = model.config
    print(f"ltxav_model_checkpoint={state.model_paths.checkpoint}")
    print(f"ltxav_model_num_layers={config.num_layers}")
    print(f"ltxav_model_video_dim={config.video_dim}")
    print(f"ltxav_model_audio_dim={config.audio_dim}")
    print(f"ltxav_model_state_key_count={len(model.state_dict())}")
    print(f"ltxav_model_missing_checkpoint_key_count={len(missing)}")
    if missing:
        print(f"ltxav_model_missing_checkpoint_keys={list(missing[:8])}")
    print(f"ltxav_model_lora_path={lora_path}")
    print(f"ltxav_model_lora_pair_count={lora_manifest.pair_count}")
    print(f"ltxav_model_lora_mapped_target_count={mapped_lora_targets}")
    print(f"ltxav_model_first_weight_is_meta={model.input_projection.patchify_proj.weight.is_meta}")
    return 0


def _inspect_ltxav_decoders() -> int:
    state = build_low_level_state(default_workflow_config(), device="cpu")
    video_vae = build_ltx_video_vae_from_checkpoint(state.model_paths.checkpoint, device="cpu")
    audio_vae = build_ltx_audio_vae_from_checkpoint(state.model_paths.checkpoint, device="cpu")
    video_state = load_ltx_video_vae_state_dict(state.model_paths.checkpoint)
    audio_state = load_ltx_audio_vae_state_dict(state.model_paths.checkpoint)
    video_missing = missing_ltx_vae_keys(video_vae, video_state)
    audio_missing = missing_ltx_vae_keys(audio_vae, audio_state)
    print(f"ltxav_decoders_checkpoint={state.model_paths.checkpoint}")
    print(f"ltxav_video_vae_state_key_count={len(video_vae.state_dict())}")
    print(f"ltxav_video_vae_missing_key_count={len(video_missing)}")
    print(f"ltxav_audio_vae_state_key_count={len(audio_vae.state_dict())}")
    print(f"ltxav_audio_vae_missing_key_count={len(audio_missing)}")
    print(f"ltxav_audio_vae_sample_rate={audio_vae.sample_rate}")
    print(f"ltxav_audio_vae_output_sample_rate={audio_vae.output_sample_rate}")
    print(f"ltxav_audio_vae_latent_channels={audio_vae.latent_channels}")
    print(f"ltxav_audio_vae_latent_frequency_bins={audio_vae.latent_frequency_bins}")
    if video_missing:
        print(f"ltxav_video_vae_missing_keys={list(video_missing[:8])}")
    if audio_missing:
        print(f"ltxav_audio_vae_missing_keys={list(audio_missing[:8])}")
    return 0


def _inspect_ltxav_pipeline() -> int:
    import torch

    state = build_low_level_state(default_workflow_config(), device="cpu")
    model = create_ltxav_model_from_checkpoint(state.model_paths.checkpoint, device="meta")
    decoders = load_ltxav_decoders_from_checkpoint(state.model_paths.checkpoint, device="cpu")
    video_latents = torch.zeros((1, 128, 1, 1, 1), dtype=torch.float32)
    audio_latents = torch.zeros((1, 8, 1, 16), dtype=torch.float32)
    context = torch.zeros((1, 1, model.config.video_context_dim + model.config.audio_context_dim), dtype=torch.float32)
    attention_mask = torch.ones((1, 1), dtype=torch.long)
    denoiser = LTXAVDenoiser(
        model=model,
        context=context,
        attention_mask=attention_mask,
        frame_rate=float(default_workflow_config().latent.frame_rate),
        transformer_options={"run_vx": False, "run_ax": False, "a2v_cross_attn": False, "v2a_cross_attn": False},
    )
    print(f"ltxav_pipeline_model_layers={model.config.num_layers}")
    print(f"ltxav_pipeline_model_is_meta={model.input_projection.patchify_proj.weight.is_meta}")
    print(f"ltxav_pipeline_video_decoder_loaded={not decoders.video_vae.decoder.conv_in.conv.weight.is_meta}")
    print(f"ltxav_pipeline_audio_decoder_loaded={decoders.audio_vae.output_sample_rate == 48000}")
    print(f"ltxav_pipeline_video_latent_shape={tuple(video_latents.shape)}")
    print(f"ltxav_pipeline_audio_latent_shape={tuple(audio_latents.shape)}")
    print(f"ltxav_pipeline_context_shape={tuple(context.shape)}")
    print(f"ltxav_pipeline_denoiser_type={denoiser.__class__.__name__}")
    print(f"ltxav_pipeline_sample_entry={sample_ltxav_euler.__name__}")
    print(f"ltxav_pipeline_decode_entry={decode_ltxav_latents.__name__}")
    return 0


def _torch_dtype_from_cli(value: str):
    import torch

    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _smoke_ltxav_model_forward(args: argparse.Namespace) -> int:
    import torch

    state = build_low_level_state(default_workflow_config(), device="cpu")
    dtype = _torch_dtype_from_cli(args.dtype)
    device = torch.device(args.device)
    model = create_ltxav_model_from_checkpoint(
        state.model_paths.checkpoint,
        dtype=dtype,
        device="meta",
        num_layers=args.layers,
    )
    report = load_ltxav_model_weights_streaming(
        model,
        state.model_paths.checkpoint,
        device=device,
        assign=True,
    )
    model.eval()
    options = {
        "a2v_cross_attn": bool(args.enable_av_cross),
        "v2a_cross_attn": bool(args.enable_av_cross),
    }
    video_latents = torch.zeros((1, model.config.video_in_channels, 1, 1, 1), device=device, dtype=dtype)
    audio_latents = torch.zeros((1, model.config.audio_channels, 1, model.config.audio_frequency), device=device, dtype=dtype)
    context = torch.zeros(
        (1, 1, model.config.video_context_dim + model.config.audio_context_dim),
        device=device,
        dtype=dtype,
    )
    attention_mask = torch.ones((1, 1), device=device, dtype=torch.long)
    timestep = torch.full((1, 1), 0.1, device=device, dtype=dtype)
    audio_timestep = torch.full((1, 1), 0.1, device=device, dtype=dtype)
    with torch.inference_mode():
        output = model(
            video_latents=video_latents,
            audio_latents=audio_latents,
            context=context,
            timestep=timestep,
            audio_timestep=audio_timestep,
            frame_rate=float(default_workflow_config().latent.frame_rate),
            attention_mask=attention_mask,
            transformer_options=options,
        )
    video_output = output[0] if isinstance(output, list) else output
    audio_output = output[1] if isinstance(output, list) else None
    print(f"ltxav_forward_smoke_checkpoint={state.model_paths.checkpoint}")
    print(f"ltxav_forward_smoke_layers={model.config.num_layers}")
    print(f"ltxav_forward_smoke_loaded_key_count={report.loaded}")
    print(f"ltxav_forward_smoke_device={device}")
    print(f"ltxav_forward_smoke_dtype={dtype}")
    print(f"ltxav_forward_smoke_av_cross={bool(args.enable_av_cross)}")
    print(f"ltxav_forward_smoke_video_shape={tuple(video_output.shape)}")
    print(f"ltxav_forward_smoke_video_finite={bool(torch.isfinite(video_output).all().item())}")
    if audio_output is not None:
        print(f"ltxav_forward_smoke_audio_shape={tuple(audio_output.shape)}")
        print(f"ltxav_forward_smoke_audio_finite={bool(torch.isfinite(audio_output).all().item())}")
    return 0


def _smoke_ltxav_sampling(args: argparse.Namespace) -> int:
    import torch

    state = build_low_level_state(default_workflow_config(), device="cpu")
    dtype = _torch_dtype_from_cli(args.dtype)
    device = torch.device(args.device)
    model = create_ltxav_model_from_checkpoint(
        state.model_paths.checkpoint,
        dtype=dtype,
        device="meta",
        num_layers=args.layers,
    )
    report = load_ltxav_model_weights_streaming(
        model,
        state.model_paths.checkpoint,
        device=device,
        assign=True,
    )
    lora_report = None
    if args.apply_lora:
        lora_path = resolve_lora_path(default_workflow_config().model.lora)
        lora_manifest = inspect_lora_manifest(lora_path)
        lora_report = apply_lora_to_ltxav_model(
            model,
            lora_path=lora_path,
            manifest=lora_manifest,
            strength=default_workflow_config().model.lora_strength,
        )
    model.eval()
    video_latents = torch.zeros((1, model.config.video_in_channels, 1, 1, 1), device=device, dtype=dtype)
    audio_latents = torch.zeros((1, model.config.audio_channels, 1, model.config.audio_frequency), device=device, dtype=dtype)
    context = torch.zeros((1, 1, model.config.video_context_dim + model.config.audio_context_dim), device=device, dtype=dtype)
    attention_mask = torch.ones((1, 1), device=device, dtype=torch.long)
    sigmas = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    video_vae = None
    audio_vae = None
    if args.decode:
        decoders = load_ltxav_decoders_from_checkpoint(state.model_paths.checkpoint, device=device)
        video_vae = decoders.video_vae
        audio_vae = decoders.audio_vae
    output = run_ltxav_sample_decode(
        model=model,
        video_latents=video_latents,
        audio_latents=audio_latents,
        context=context,
        attention_mask=attention_mask,
        sigmas=sigmas,
        frame_rate=float(default_workflow_config().latent.frame_rate),
        video_vae=video_vae,
        audio_vae=audio_vae,
        transformer_options={
            "a2v_cross_attn": bool(args.enable_av_cross),
            "v2a_cross_attn": bool(args.enable_av_cross),
        },
    )
    print(f"ltxav_sampling_smoke_checkpoint={state.model_paths.checkpoint}")
    print(f"ltxav_sampling_smoke_layers={model.config.num_layers}")
    print(f"ltxav_sampling_smoke_loaded_key_count={report.loaded}")
    print(f"ltxav_sampling_smoke_device={device}")
    print(f"ltxav_sampling_smoke_dtype={dtype}")
    print(f"ltxav_sampling_smoke_decode={bool(args.decode)}")
    print(f"ltxav_sampling_smoke_lora_applied={bool(args.apply_lora)}")
    if lora_report is not None:
        print(f"ltxav_sampling_smoke_lora_matched={lora_report.matched}")
        print(f"ltxav_sampling_smoke_lora_skipped={lora_report.skipped}")
    print(f"ltxav_sampling_smoke_video_latent_shape={tuple(output.video_latents.shape)}")
    print(f"ltxav_sampling_smoke_audio_latent_shape={tuple(output.audio_latents.shape)}")
    print(f"ltxav_sampling_smoke_video_finite={bool(torch.isfinite(output.video_latents).all().item())}")
    print(f"ltxav_sampling_smoke_audio_finite={bool(torch.isfinite(output.audio_latents).all().item())}")
    if output.decoded is not None:
        print(f"ltxav_sampling_smoke_decoded_video_shape={tuple(output.decoded.video.shape)}")
        print(f"ltxav_sampling_smoke_decoded_video_finite={bool(torch.isfinite(output.decoded.video).all().item())}")
        if output.decoded.audio is not None:
            print(f"ltxav_sampling_smoke_decoded_audio_shape={tuple(output.decoded.audio.shape)}")
            print(f"ltxav_sampling_smoke_decoded_audio_finite={bool(torch.isfinite(output.decoded.audio).all().item())}")
    return 0

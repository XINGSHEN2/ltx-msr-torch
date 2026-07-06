from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint_loader import apply_lora_to_checkpoint_subset, inspect_checkpoint_manifest
from .comfy_api_prompt import build_case_api_prompt, save_api_prompt
from .comfy_client import load_api_prompt, queue_prompt, wait_for_history
from .local_state import build_low_level_state
from .lora_apply import match_lora_targets
from .lora_loader import inspect_lora_manifest, resolve_lora_path
from .model_inspect import inspect_workflow_model_headers
from .msr_reference import create_msr_reference_video_from_paths
from .text_projection import build_text_projection_from_checkpoint
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

from __future__ import annotations

import argparse
from pathlib import Path

from .msr_reference import create_msr_reference_video_from_paths
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

    args = parser.parse_args(argv)
    if args.command == "build-reference":
        return _build_reference(args)
    if args.command == "inspect-config":
        return _inspect_config(args)
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

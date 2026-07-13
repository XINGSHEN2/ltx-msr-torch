#!/usr/bin/env python3
"""Validate an ltx-msr-torch deployment, with an optional GPU smoke test."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


class Reporter:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.warned = 0

    def pass_(self, message: str) -> None:
        self.passed += 1
        print(f"[PASS] {message}")

    def fail(self, message: str) -> None:
        self.failed += 1
        print(f"[FAIL] {message}")

    def warn(self, message: str) -> None:
        self.warned += 1
        print(f"[WARN] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Python dependencies, ffmpeg, model files, Gemma configs, and "
            "the bundled standalone VAE implementation. CUDA is opt-in."
        )
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Model root; overrides LTX_MSR_MODEL_ROOT.",
    )
    parser.add_argument(
        "--gemma-config-dir",
        type=Path,
        default=None,
        help="Gemma config directory; overrides LTX_MSR_GEMMA_CONFIG_DIR.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Require CUDA and BF16 support (disabled by default for CPU-only servers).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also run a small end-to-end generation (loads all model types).",
    )
    parser.add_argument("--device", default="cuda", help="Device for --smoke (default: cuda).")
    parser.add_argument(
        "--output-video",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "environment_smoke.mp4",
        help="Output path for --smoke.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print tracebacks for failed checks.",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    if args.model_root is not None:
        os.environ["LTX_MSR_MODEL_ROOT"] = str(args.model_root.expanduser().resolve())
    if args.gemma_config_dir is not None:
        os.environ["LTX_MSR_GEMMA_CONFIG_DIR"] = str(
            args.gemma_config_dir.expanduser().resolve()
        )

    # This is intentional: the validation must succeed without importing ComfyUI.
    os.environ.pop("COMFYUI_ROOT", None)
    sys.path.insert(0, str(SOURCE_ROOT))


def check_python(reporter: Reporter) -> None:
    version = sys.version_info
    message = f"Python {version.major}.{version.minor}.{version.micro} ({sys.executable})"
    if version >= (3, 10):
        reporter.pass_(message)
    else:
        reporter.fail(f"{message}; Python >= 3.10 is required")


def check_dependencies(reporter: Reporter, *, verbose: bool) -> dict[str, object]:
    modules = {
        "einops": "einops",
        "numpy": "numpy",
        "Pillow": "PIL",
        "safetensors": "safetensors",
        "tokenizers": "tokenizers",
        "torch": "torch",
        "torchaudio": "torchaudio",
        "transformers": "transformers",
    }
    imported: dict[str, object] = {}
    failures: list[str] = []
    for display_name, module_name in modules.items():
        try:
            imported[module_name] = importlib.import_module(module_name)
        except Exception as exc:  # Import failures can include binary ABI errors.
            failures.append(f"{display_name}: {type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()

    if failures:
        reporter.fail("Python dependencies: " + " | ".join(failures))
    else:
        versions = ", ".join(
            f"{name}={getattr(imported[module], '__version__', 'unknown')}"
            for name, module in modules.items()
        )
        reporter.pass_(f"Python dependencies imported ({versions})")
    return imported


def check_cuda(reporter: Reporter, torch_module: object | None, *, required: bool) -> None:
    if torch_module is None:
        if required:
            reporter.fail("CUDA check could not run because torch was not imported")
        return

    if not required:
        reporter.pass_("hardware-neutral mode active; CUDA/NVIDIA is not required or inspected")
        return

    torch = torch_module
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    if not torch.cuda.is_available():
        reporter.fail(f"CUDA unavailable (torch={torch.__version__}, torch CUDA={cuda_version})")
        return

    device_lines: list[str] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            memory = f", free={free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB"
        except Exception:
            memory = f", total={properties.total_memory / 2**30:.1f} GiB"
        device_lines.append(f"cuda:{index}={properties.name}{memory}")
    reporter.pass_(
        f"CUDA available (torch CUDA={cuda_version}; " + "; ".join(device_lines) + ")"
    )

    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    if bf16_supported:
        reporter.pass_("CUDA BF16 is supported")
    else:
        reporter.fail("CUDA BF16 is unavailable; the GPU deployment uses --dtype bf16")


def check_media_tools(reporter: Reporter) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        reporter.pass_(f"ffmpeg found: {ffmpeg}")
    else:
        reporter.fail("ffmpeg was not found in PATH")

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        reporter.pass_(f"ffprobe found: {ffprobe}")
    else:
        reporter.warn("ffprobe was not found; generation still works, but output probing is disabled")


def check_project_imports(reporter: Reporter, *, verbose: bool) -> None:
    try:
        package = importlib.import_module("ltx_msr_torch")
        vae = importlib.import_module("ltx_msr_torch.vae")
        video_module = vae.VideoVAE.__module__
        audio_module = vae.AudioVAE.__module__
        if not video_module.startswith("ltx_msr_torch.vae."):
            raise RuntimeError(f"VideoVAE resolved outside this package: {video_module}")
        if not audio_module.startswith("ltx_msr_torch.vae."):
            raise RuntimeError(f"AudioVAE resolved outside this package: {audio_module}")
        comfy_modules = sorted(name for name in sys.modules if name == "comfy" or name.startswith("comfy."))
        if comfy_modules:
            raise RuntimeError(f"ComfyUI modules were imported: {comfy_modules[:5]}")
        location = Path(package.__file__).resolve()
        reporter.pass_(
            f"standalone package import succeeded ({location}); local VideoVAE and AudioVAE are active"
        )
    except Exception as exc:
        reporter.fail(f"standalone package import failed: {type(exc).__name__}: {exc}")
        if verbose:
            traceback.print_exc()


def check_models(reporter: Reporter, *, verbose: bool) -> None:
    try:
        from ltx_msr_torch.model_inspect import inspect_workflow_model_headers
        from ltx_msr_torch.model_paths import resolve_workflow_model_paths
        from ltx_msr_torch.workflow_config import default_workflow_config

        paths = resolve_workflow_model_paths(default_workflow_config())
        inspections = inspect_workflow_model_headers(paths)
        for label, inspection in (
            ("checkpoint", inspections.checkpoint),
            ("text encoder", inspections.text_encoder),
            ("MSR LoRA", inspections.lora),
        ):
            if inspection.key_count <= 0:
                raise RuntimeError(f"{label} contains no tensors: {inspection.path}")
            size_gib = inspection.path.stat().st_size / 2**30
            reporter.pass_(
                f"{label}: {inspection.path} ({size_gib:.2f} GiB, {inspection.key_count} tensors)"
            )
    except Exception as exc:
        reporter.fail(f"model files are incomplete or invalid: {type(exc).__name__}: {exc}")
        if verbose:
            traceback.print_exc()


def check_gemma_configs(reporter: Reporter, *, verbose: bool) -> None:
    try:
        from ltx_msr_torch.gemma_tokenizer import GemmaTokenizer
        from ltx_msr_torch.text_encoder_sections import resolve_text_encoder_config_paths

        paths = resolve_text_encoder_config_paths()
        tokenizer = GemmaTokenizer.from_config_paths(paths)
        token_count = len(tokenizer.encode("environment check").input_ids)
        if token_count <= 0:
            raise RuntimeError("tokenizer produced no tokens")
        reporter.pass_(
            f"Gemma configs and tokenizer are usable: {paths.config_dir} ({token_count} test tokens)"
        )
    except Exception as exc:
        reporter.fail(f"Gemma configs/tokenizer check failed: {type(exc).__name__}: {exc}")
        if verbose:
            traceback.print_exc()


def run_smoke_test(reporter: Reporter, args: argparse.Namespace) -> None:
    output = args.output_video.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "ltx_msr_torch",
        "generate-msr-case",
        "--workflow",
        str(PROJECT_ROOT / "sample_cases" / "LTX-2.3_MSR_sample_workflow_V2.json"),
        "--case-dir",
        str(PROJECT_ROOT / "sample_cases" / "validition_v1_01"),
        "--output-video",
        str(output),
        "--width",
        "256",
        "--height",
        "256",
        "--reference-width",
        "256",
        "--reference-height",
        "256",
        "--reference-frames",
        "9",
        "--video-frames",
        "17",
        "--layers",
        "1",
        "--max-sigmas",
        "2",
        "--dtype",
        "bf16",
        "--device",
        args.device,
    ]
    environment = os.environ.copy()
    environment.pop("COMFYUI_ROOT", None)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(SOURCE_ROOT)
    )

    print("[INFO] Running end-to-end smoke test (this loads the model weights)...")
    print("[INFO] " + " ".join(str(part) for part in command))
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
        if completed.returncode != 0:
            reporter.fail(f"end-to-end smoke test exited with code {completed.returncode}")
            return
        if not output.is_file() or output.stat().st_size == 0:
            reporter.fail(f"smoke test returned success but output is missing or empty: {output}")
            return
        reporter.pass_(f"end-to-end generation succeeded: {output} ({output.stat().st_size / 2**20:.1f} MiB)")
    except Exception as exc:
        reporter.fail(f"end-to-end smoke test could not start: {type(exc).__name__}: {exc}")
        if args.verbose:
            traceback.print_exc()


def main() -> int:
    args = parse_args()
    configure_environment(args)
    reporter = Reporter()

    print("ltx-msr-torch environment verification")
    print(f"project: {PROJECT_ROOT}")
    print(f"model root: {os.environ.get('LTX_MSR_MODEL_ROOT', PROJECT_ROOT / 'models')}")
    print(
        "Gemma configs: "
        + os.environ.get(
            "LTX_MSR_GEMMA_CONFIG_DIR",
            str(Path(os.environ.get("LTX_MSR_MODEL_ROOT", PROJECT_ROOT / "models")) / "gemma_configs"),
        )
    )
    print("ComfyUI runtime: deliberately disabled for this check")

    check_python(reporter)
    imported = check_dependencies(reporter, verbose=args.verbose)
    check_cuda(reporter, imported.get("torch"), required=args.require_cuda)
    check_media_tools(reporter)
    check_project_imports(reporter, verbose=args.verbose)
    check_models(reporter, verbose=args.verbose)
    check_gemma_configs(reporter, verbose=args.verbose)

    if args.smoke:
        if reporter.failed:
            reporter.warn("skipping --smoke because prerequisite checks failed")
        else:
            run_smoke_test(reporter, args)

    print(
        f"\nSummary: {reporter.passed} passed, {reporter.failed} failed, "
        f"{reporter.warned} warnings"
    )
    if reporter.failed:
        print("RESULT: environment is NOT ready")
        return 1
    print("RESULT: environment is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

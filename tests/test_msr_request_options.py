from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

import ltx_msr_torch.cli as cli
from ltx_msr_torch.msr_case import (
    _ceil_div,
    _crop_decoded_video,
    _requested_cfg_guide,
    _requested_dimensions,
    _requested_frame_rate,
    _requested_ic_lora_guide_strength,
    _requested_output_path,
)


def workflow_config():
    return SimpleNamespace(
        latent=SimpleNamespace(width=1280, height=1920, frame_rate=24),
        sampling=SimpleNamespace(cfg=1.0),
        ic_lora_guide=SimpleNamespace(strength=1.0),
    )


def test_generate_msr_case_cli_exposes_service_request_options(monkeypatch) -> None:
    captured = {}

    def fake_generate(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "generate_msr_case", fake_generate)

    result = cli.main(
        [
            "generate-msr-case",
            "--resolution",
            "720*1280",
            "--fps",
            "24",
            "--global-prompt",
            "global",
            "--local-prompt",
            "local one | local two",
            "--negative-prompt",
            "watermark",
            "--image-1",
            "/tmp/one.png",
            "--background",
            "/tmp/background.png",
            "--ic-lora-guide-strength",
            "0.75",
            "--cfg-guide",
            "1.5",
            "--output-dir",
            "/tmp/outputs",
        ]
    )

    assert result == 0
    assert captured["resolution"] == "720*1280"
    assert captured["frame_rate"] == 24
    assert captured["global_prompt"] == "global"
    assert captured["local_prompts"] == "local one | local two"
    assert captured["negative_prompt"] == "watermark"
    assert captured["subject_1"] == "/tmp/one.png"
    assert captured["subject_2"] is None
    assert captured["background"] == "/tmp/background.png"
    assert captured["ic_lora_guide_strength"] == 0.75
    assert captured["cfg_guide"] == 1.5
    assert captured["output_dir"] == "/tmp/outputs"


def test_request_option_defaults_and_overrides() -> None:
    config = workflow_config()
    defaults = Namespace()

    assert _requested_dimensions(defaults, config) == (1280, 1920)
    assert _requested_frame_rate(defaults, config) == 24
    assert _requested_ic_lora_guide_strength(defaults, config) == 1.0
    assert _requested_cfg_guide(defaults, config) == 1.0
    assert _requested_output_path(defaults) == "outputs/msr_case_01_torch.mp4"

    requested = Namespace(
        resolution="720x1280",
        width=None,
        height=None,
        frame_rate=30,
        ic_lora_guide_strength=0.6,
        cfg_guide=1.2,
        output_dir="/tmp/results",
        output_video="custom.mp4",
    )
    assert _requested_dimensions(requested, config) == (720, 1280)
    assert _requested_frame_rate(requested, config) == 30
    assert _requested_ic_lora_guide_strength(requested, config) == 0.6
    assert _requested_cfg_guide(requested, config) == 1.2
    assert _requested_output_path(requested) == str(Path("/tmp/results/custom.mp4"))


def test_requested_resolution_uses_ceil_latents_and_exact_output_crop() -> None:
    assert _ceil_div(720, 32) == 23
    assert _ceil_div(1280, 32) == 40
    assert _ceil_div(24 - 1, 8) + 1 == 4

    decoded = torch.arange(1 * 3 * 5 * 12 * 10, dtype=torch.float32).reshape(
        1, 3, 5, 12, 10
    )
    cropped = _crop_decoded_video(
        decoded,
        frame_count=4,
        width=8,
        height=10,
    )

    assert cropped.shape == (4, 10, 8, 3)

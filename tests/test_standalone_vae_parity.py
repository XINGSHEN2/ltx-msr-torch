import gc
import json
import os
import sys

import pytest
import torch

from ltx_msr_torch.ltx_vae import (
    build_ltx_audio_vae_from_checkpoint,
    build_ltx_video_vae_from_checkpoint,
    encode_ltx_video_pixels,
    load_checkpoint_config,
    load_ltx_audio_vae_state_dict,
    load_ltx_video_vae_state_dict,
)
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def _enable_reference_comfy_imports() -> None:
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        pytest.skip("COMFYUI_ROOT is required only for reference parity tests")
    if root not in sys.path:
        sys.path.insert(0, root)
    import comfy.options

    sys.argv = ["ltx_msr_torch_vae_parity", "--cpu"]
    comfy.options.args_parsing = True


def test_standalone_video_vae_matches_comfy_reference_exactly():
    paths = resolve_workflow_model_paths(default_workflow_config())
    state = load_ltx_video_vae_state_dict(paths.checkpoint)
    pixels = torch.linspace(0.0, 1.0, 9 * 32 * 32 * 3, dtype=torch.float32).reshape(9, 32, 32, 3)
    latents = torch.linspace(-1.0, 1.0, 128, dtype=torch.float32).reshape(1, 128, 1, 1, 1)

    local = build_ltx_video_vae_from_checkpoint(paths.checkpoint, device="cpu")
    local.load_state_dict(state, strict=True)
    local_encoded = encode_ltx_video_pixels(local, pixels)
    local_decoded = local.decode(latents)
    del local
    gc.collect()

    _enable_reference_comfy_imports()
    from comfy.ldm.lightricks.vae.causal_video_autoencoder import VideoVAE

    reference = VideoVAE(config=load_checkpoint_config(paths.checkpoint)["vae"])
    reference.load_state_dict(state, strict=True)
    model_input = pixels.movedim(-1, 1).movedim(1, 0).unsqueeze(0) * 2.0 - 1.0
    reference_encoded = reference.encode(model_input)
    reference_decoded = reference.decode(latents)

    assert torch.equal(local_encoded, reference_encoded)
    assert torch.equal(local_decoded, reference_decoded)


def test_standalone_audio_vae_matches_comfy_reference_exactly():
    paths = resolve_workflow_model_paths(default_workflow_config())
    state = load_ltx_audio_vae_state_dict(paths.checkpoint)
    latents = torch.linspace(-1.0, 1.0, 1 * 8 * 4 * 16, dtype=torch.float32).reshape(1, 8, 4, 16)

    local = build_ltx_audio_vae_from_checkpoint(paths.checkpoint, device="cpu")
    local.load_state_dict(state, strict=True)
    local_decoded = local.decode(latents)
    del local
    gc.collect()

    _enable_reference_comfy_imports()
    from comfy.ldm.lightricks.vae.audio_vae import AudioVAE

    reference = AudioVAE({"config": json.dumps(load_checkpoint_config(paths.checkpoint))})
    reference.load_state_dict(state, strict=True)
    reference_decoded = reference.decode(latents)

    assert torch.equal(local_decoded, reference_decoded)

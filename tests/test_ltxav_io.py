import torch

from ltx_msr_torch.ltxav_io import (
    LTXAVInputProjection,
    LTXAVOutputProjection,
    load_ltxav_input_projection_state_dict,
    load_ltxav_output_projection_state_dict,
)
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def test_load_ltxav_projection_state_dicts_read_workflow_weights():
    paths = resolve_workflow_model_paths(default_workflow_config())
    input_state = load_ltxav_input_projection_state_dict(paths.checkpoint)
    output_state = load_ltxav_output_projection_state_dict(paths.checkpoint)

    assert input_state["patchify_proj.weight"].shape == (4096, 128)
    assert input_state["audio_patchify_proj.weight"].shape == (2048, 128)
    assert output_state["proj_out.weight"].shape == (128, 4096)
    assert output_state["audio_proj_out.weight"].shape == (128, 2048)
    assert input_state["patchify_proj.weight"].dtype == torch.bfloat16


def test_ltxav_input_projection_forward_shapes():
    module = LTXAVInputProjection(
        video_in_channels=2,
        video_hidden_dim=4,
        audio_in_channels=6,
        audio_hidden_dim=3,
        dtype=torch.float32,
    )
    video = torch.randn(1, 2, 2, 3, 4)
    audio = torch.randn(1, 2, 5, 3)

    output = module(video, audio)

    assert output.video_patches.shape == (1, 24, 2)
    assert output.video_tokens.shape == (1, 24, 4)
    assert output.video_latent_coords.shape == (1, 3, 24, 2)
    assert output.video_pixel_coords.shape == (1, 3, 24, 2)
    assert output.audio_patches.shape == (1, 5, 6)
    assert output.audio_tokens.shape == (1, 5, 3)
    assert output.audio_latent_coords.shape == (1, 1, 5, 2)


def test_ltxav_output_projection_forward_shapes():
    module = LTXAVOutputProjection(
        video_hidden_dim=4,
        video_out_channels=2,
        audio_hidden_dim=3,
        audio_out_channels=6,
        dtype=torch.float32,
    )
    video_tokens = torch.randn(1, 24, 4)
    audio_tokens = torch.randn(1, 5, 3)

    video_out, audio_out = module(video_tokens, audio_tokens)

    assert video_out.shape == (1, 24, 2)
    assert audio_out.shape == (1, 5, 6)


def test_ltxav_input_projection_meta_shapes_match_workflow():
    module = LTXAVInputProjection(device="meta")
    output = LTXAVOutputProjection(device="meta")

    assert tuple(module.patchify_proj.weight.shape) == (4096, 128)
    assert tuple(module.audio_patchify_proj.weight.shape) == (2048, 128)
    assert tuple(output.proj_out.weight.shape) == (128, 4096)
    assert tuple(output.audio_proj_out.weight.shape) == (128, 2048)

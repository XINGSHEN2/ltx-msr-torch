import math

import torch

from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.text_projection import (
    DualLinearTextProjection,
    build_text_projection_from_checkpoint,
    infer_text_projection_config,
    load_text_projection_state_dict,
)
from ltx_msr_torch.workflow_config import default_workflow_config


def test_dual_linear_text_projection_matches_comfy_formula():
    module = DualLinearTextProjection(input_dim=3, video_dim=2, audio_dim=3, dtype=torch.float32)
    with torch.no_grad():
        module.video_aggregate_embed.weight.fill_(1.0)
        module.video_aggregate_embed.bias.zero_()
        module.audio_aggregate_embed.weight.fill_(2.0)
        module.audio_aggregate_embed.bias.zero_()
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

    output = module(hidden)
    x = hidden.movedim(1, -1)
    x = (x * torch.rsqrt(torch.mean(x**2, dim=2, keepdim=True) + 1e-6)).flatten(start_dim=2)
    expected_video = torch.nn.functional.linear(x * math.sqrt(2 / 2), module.video_aggregate_embed.weight)
    expected_audio = torch.nn.functional.linear(x * math.sqrt(3 / 2), module.audio_aggregate_embed.weight)

    assert torch.allclose(output, torch.cat((expected_video, expected_audio), dim=-1))


def test_load_text_projection_state_dict_reads_workflow_weights():
    paths = resolve_workflow_model_paths(default_workflow_config())
    state_dict = load_text_projection_state_dict(paths.checkpoint)
    config = infer_text_projection_config(state_dict)

    assert set(state_dict) == {
        "audio_aggregate_embed.bias",
        "audio_aggregate_embed.weight",
        "video_aggregate_embed.bias",
        "video_aggregate_embed.weight",
    }
    assert config.input_dim == 188160
    assert config.video_dim == 4096
    assert config.audio_dim == 2048
    assert config.dtype == torch.bfloat16


def test_build_text_projection_from_checkpoint_loads_module():
    paths = resolve_workflow_model_paths(default_workflow_config())
    module = build_text_projection_from_checkpoint(paths.checkpoint)

    assert module.video_aggregate_embed.weight.shape == (4096, 188160)
    assert module.audio_aggregate_embed.weight.shape == (2048, 188160)
    assert module.config.dtype == torch.bfloat16

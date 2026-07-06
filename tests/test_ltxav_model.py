import torch
from safetensors.torch import save_file

from ltx_msr_torch.ltxav_model import (
    LTXAVModel,
    LTXAVModelConfig,
    load_ltxav_model_state_dict,
    ltxav_model_checkpoint_key,
)


def test_ltxav_model_forward_runs_small_pipeline():
    config = LTXAVModelConfig(
        video_in_channels=2,
        audio_in_channels=6,
        video_dim=12,
        audio_dim=4,
        video_heads=2,
        audio_heads=2,
        video_dim_head=6,
        audio_dim_head=2,
        num_layers=1,
        video_context_dim=12,
        audio_context_dim=4,
        video_out_channels=2,
        audio_out_channels=6,
        audio_channels=2,
        audio_frequency=3,
        cross_attention_adaln=True,
        apply_gated_attention=True,
    )
    model = LTXAVModel(config, dtype=torch.float32)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.02, 0.02)

    video_latents = torch.randn(1, 2, 2, 2, 2)
    audio_latents = torch.randn(1, 2, 4, 3)
    context = torch.randn(1, 5, 16)
    output = model(
        video_latents=video_latents,
        audio_latents=audio_latents,
        context=context,
        timestep=torch.full((1, 8), 0.1),
        audio_timestep=torch.full((1, 4), 0.1),
        frame_rate=25.0,
        attention_mask=torch.ones(1, 5, dtype=torch.long),
    )

    assert isinstance(output, list)
    assert output[0].shape == video_latents.shape
    assert output[1].shape == audio_latents.shape


def test_ltxav_model_checkpoint_key_maps_wrapped_modules():
    assert ltxav_model_checkpoint_key("input_projection.patchify_proj.weight") == (
        "model.diffusion_model.patchify_proj.weight"
    )
    assert ltxav_model_checkpoint_key("output_processor.audio_proj_out.bias") == (
        "model.diffusion_model.audio_proj_out.bias"
    )
    assert ltxav_model_checkpoint_key("video_adaln_single.linear.weight") == (
        "model.diffusion_model.adaln_single.linear.weight"
    )
    assert ltxav_model_checkpoint_key("transformer_blocks.0.attn1.to_q.weight") == (
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
    )


def test_load_ltxav_model_state_dict_maps_from_safetensors(tmp_path):
    config = LTXAVModelConfig(
        video_in_channels=2,
        audio_in_channels=6,
        video_dim=12,
        audio_dim=4,
        video_heads=2,
        audio_heads=2,
        video_dim_head=6,
        audio_dim_head=2,
        num_layers=1,
        video_context_dim=12,
        audio_context_dim=4,
        video_out_channels=2,
        audio_out_channels=6,
        audio_channels=2,
        audio_frequency=3,
    )
    model = LTXAVModel(config, dtype=torch.float32)
    local_state = model.state_dict()
    checkpoint_state = {
        ltxav_model_checkpoint_key(key): torch.full_like(value, index + 1, dtype=value.dtype)
        for index, (key, value) in enumerate(local_state.items())
    }
    path = tmp_path / "small_ltxav.safetensors"
    save_file(checkpoint_state, path)

    loaded = load_ltxav_model_state_dict(model, path)

    assert set(loaded) == set(local_state)
    for index, key in enumerate(local_state):
        assert torch.equal(loaded[key], torch.full_like(local_state[key], index + 1, dtype=local_state[key].dtype))

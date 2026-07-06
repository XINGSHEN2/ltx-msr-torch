import torch
from safetensors.torch import save_file

from ltx_msr_torch.ltxav_model import (
    LTXAVModel,
    LTXAVModelConfig,
    create_ltxav_model_from_checkpoint,
    load_ltxav_model_state_dict,
    load_ltxav_model_weights_streaming,
    ltxav_model_config_from_manifest,
    ltxav_model_checkpoint_key,
    missing_ltxav_model_checkpoint_keys,
)
from ltx_msr_torch.ltxav_transformer import inspect_ltxav_transformer_manifest
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


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


def test_ltxav_model_config_from_workflow_manifest():
    manifest = inspect_ltxav_transformer_manifest(resolve_workflow_model_paths(default_workflow_config()).checkpoint)

    config = ltxav_model_config_from_manifest(manifest)

    assert config.video_in_channels == 128
    assert config.audio_in_channels == 128
    assert config.video_dim == 4096
    assert config.audio_dim == 2048
    assert config.num_layers == 48
    assert config.video_context_dim == 4096
    assert config.audio_context_dim == 2048
    assert config.video_out_channels == 128
    assert config.audio_out_channels == 128


def test_create_ltxav_model_from_checkpoint_on_meta_device():
    paths = resolve_workflow_model_paths(default_workflow_config())

    model = create_ltxav_model_from_checkpoint(paths.checkpoint, device="meta")

    assert model.config.num_layers == 48
    assert len(model.transformer_blocks) == 48
    assert tuple(model.input_projection.patchify_proj.weight.shape) == (4096, 128)
    assert tuple(model.transformer_blocks[0].audio_attn1.to_q.weight.shape) == (2048, 2048)
    assert tuple(model.output_processor.audio_proj_out.weight.shape) == (128, 2048)


def test_ltxav_model_checkpoint_mapping_keys_exist_for_workflow_checkpoint():
    paths = resolve_workflow_model_paths(default_workflow_config())
    model = create_ltxav_model_from_checkpoint(paths.checkpoint, device="meta")

    missing = missing_ltxav_model_checkpoint_keys(model, paths.checkpoint)

    assert missing == ()


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


def test_load_ltxav_model_weights_streaming_copies_into_existing_model(tmp_path):
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
        ltxav_model_checkpoint_key(key): torch.full_like(value, 0.25, dtype=value.dtype)
        for key, value in local_state.items()
    }
    path = tmp_path / "small_ltxav_stream.safetensors"
    save_file(checkpoint_state, path)

    report = load_ltxav_model_weights_streaming(model, path)

    assert report.loaded == len(local_state)
    assert report.missing == ()
    assert torch.allclose(model.input_projection.patchify_proj.weight, torch.full_like(model.input_projection.patchify_proj.weight, 0.25))


def test_load_ltxav_model_weights_streaming_assigns_meta_model(tmp_path):
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
    model = LTXAVModel(config, dtype=torch.float32, device="meta")
    local_state = model.state_dict()
    checkpoint_state = {
        ltxav_model_checkpoint_key(key): torch.full(value.shape, 0.5, dtype=value.dtype)
        for key, value in local_state.items()
    }
    path = tmp_path / "small_ltxav_meta_stream.safetensors"
    save_file(checkpoint_state, path)

    report = load_ltxav_model_weights_streaming(model, path, assign=True)

    assert report.loaded == len(local_state)
    assert not model.input_projection.patchify_proj.weight.is_meta
    assert torch.allclose(model.input_projection.patchify_proj.weight, torch.full_like(model.input_projection.patchify_proj.weight, 0.5))

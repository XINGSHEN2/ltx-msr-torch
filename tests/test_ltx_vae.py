import torch

from ltx_msr_torch.ltx_vae import (
    build_ltx_audio_vae_from_checkpoint,
    build_ltx_video_vae_from_checkpoint,
    load_checkpoint_config,
    load_ltxav_decoders_from_checkpoint,
    load_ltx_audio_vae_state_dict,
    load_ltx_video_vae_state_dict,
    missing_ltx_vae_keys,
)
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def test_load_checkpoint_config_contains_video_audio_and_vocoder_sections():
    paths = resolve_workflow_model_paths(default_workflow_config())
    config = load_checkpoint_config(paths.checkpoint)

    assert "vae" in config
    assert "audio_vae" in config
    assert "vocoder" in config
    assert config["vae"]["latent_channels"] == 128


def test_build_ltx_video_vae_and_state_keys_match_checkpoint():
    paths = resolve_workflow_model_paths(default_workflow_config())
    model = build_ltx_video_vae_from_checkpoint(paths.checkpoint, dtype=torch.bfloat16, device="cpu")
    state = load_ltx_video_vae_state_dict(paths.checkpoint)

    missing = missing_ltx_vae_keys(model, state)

    assert missing == ()
    assert tuple(model.decoder.conv_in.conv.weight.shape) == (1024, 128, 3, 3, 3)
    assert state["decoder.conv_in.conv.weight"].shape == model.decoder.conv_in.conv.weight.shape


def test_build_ltx_audio_vae_and_state_keys_match_checkpoint():
    paths = resolve_workflow_model_paths(default_workflow_config())
    model = build_ltx_audio_vae_from_checkpoint(paths.checkpoint, dtype=torch.bfloat16, device="cpu")
    state = load_ltx_audio_vae_state_dict(paths.checkpoint)

    missing = missing_ltx_vae_keys(model, state)

    assert missing == ()
    assert model.latent_channels == 8
    assert model.latent_frequency_bins == 16
    assert "autoencoder.decoder.conv_in.conv.weight" in state
    assert "vocoder.vocoder.conv_pre.weight" in state


def test_load_ltxav_decoders_from_checkpoint_strict_loads_weights():
    paths = resolve_workflow_model_paths(default_workflow_config())

    decoders = load_ltxav_decoders_from_checkpoint(paths.checkpoint, device="cpu")

    assert tuple(decoders.video_vae.decoder.conv_in.conv.weight.shape) == (1024, 128, 3, 3, 3)
    assert decoders.audio_vae.output_sample_rate == 48000

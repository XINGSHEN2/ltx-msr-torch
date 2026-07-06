import torch

from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.vae_sections import inspect_vae_section, load_vae_section_state_dict
from ltx_msr_torch.workflow_config import default_workflow_config


def test_inspect_video_vae_section_matches_checkpoint_layout():
    paths = resolve_workflow_model_paths(default_workflow_config())
    manifest = inspect_vae_section(paths.checkpoint, prefix="vae")

    assert manifest.key_count == 170
    assert manifest.encoder_key_count == 84
    assert manifest.decoder_key_count == 84
    assert manifest.statistics_key_count == 2
    assert manifest.first_shapes[0] == ("vae.decoder.conv_in.conv.bias", (1024,), torch.bfloat16)


def test_inspect_audio_vae_section_matches_checkpoint_layout():
    paths = resolve_workflow_model_paths(default_workflow_config())
    manifest = inspect_vae_section(paths.checkpoint, prefix="audio_vae")

    assert manifest.key_count == 102
    assert manifest.encoder_key_count > 0
    assert manifest.decoder_key_count > 0
    assert manifest.statistics_key_count == 2
    assert manifest.first_shapes[0] == ("audio_vae.decoder.conv_in.conv.bias", (512,), torch.bfloat16)


def test_load_vae_section_state_dict_strips_prefix():
    paths = resolve_workflow_model_paths(default_workflow_config())
    state_dict = load_vae_section_state_dict(paths.checkpoint, prefix="vae")

    assert len(state_dict) == 170
    assert "decoder.conv_in.conv.bias" in state_dict
    assert "vae.decoder.conv_in.conv.bias" not in state_dict
    assert state_dict["decoder.conv_in.conv.weight"].shape == (1024, 128, 3, 3, 3)

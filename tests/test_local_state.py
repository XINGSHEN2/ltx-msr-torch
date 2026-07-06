from ltx_msr_torch.local_state import build_low_level_state
from ltx_msr_torch.workflow_config import default_workflow_config


def test_build_low_level_state_uses_workflow_parameters():
    state = build_low_level_state(default_workflow_config(), device="cpu")

    assert state.width == 1920
    assert state.height == 1280
    assert state.frame_count == 41
    assert state.video_length == 145
    assert state.noise.seed == 337096718960207
    assert tuple(state.sigmas.shape) == (9,)
    assert tuple(state.video_latent["samples"].shape) == (1, 128, 19, 60, 40)
    assert state.ic_lora.latent_downscale_factor == 1.0
    assert state.nag_patch.config.scale == 11.0
    assert state.nag_patch.config.alpha == 0.25
    assert state.nag_patch.config.tau == 2.5
    assert state.nag_patch.transformer_block_count == 48
    assert len(state.nag_patch.video_patch_targets) == 48
    assert len(state.nag_patch.audio_patch_targets) == 0
    assert state.ic_lora.lora_path.exists()
    assert state.model_paths.checkpoint.exists()
    assert state.model_paths.text_encoder.exists()
    assert state.model_paths.lora.exists()

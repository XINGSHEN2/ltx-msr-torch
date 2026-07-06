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


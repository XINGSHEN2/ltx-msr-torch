import torch

from ltx_msr_torch.ltx_timestep import CompressedTimestep
from ltx_msr_torch.ltxav_output import LTXAVOutputProcessor, load_ltxav_output_processor_state_dict
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def test_ltxav_output_processor_weight_shapes_match_workflow():
    state = load_ltxav_output_processor_state_dict(resolve_workflow_model_paths(default_workflow_config()).checkpoint)

    assert state["scale_shift_table"].shape == (2, 4096)
    assert state["proj_out.weight"].shape == (128, 4096)
    assert state["audio_scale_shift_table"].shape == (2, 2048)
    assert state["audio_proj_out.weight"].shape == (128, 2048)


def test_ltxav_output_processor_unpatchifies_video_and_audio():
    processor = LTXAVOutputProcessor(
        video_hidden_dim=4,
        video_out_channels=2,
        audio_hidden_dim=4,
        audio_out_channels=6,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for parameter in processor.parameters():
            parameter.uniform_(-0.05, 0.05)
    video_tokens = torch.randn(1, 8, 4)
    audio_tokens = torch.randn(1, 3, 4)
    video_embedded = CompressedTimestep(torch.randn(1, 2, 4), patches_per_frame=4, per_frame=True)
    audio_embedded = torch.randn(1, 3, 4)

    output = processor(
        video_tokens,
        audio_tokens,
        video_embedded_timestep=video_embedded,
        audio_embedded_timestep=audio_embedded,
        orig_shape=(1, 2, 2, 2, 2),
        audio_channels=2,
        audio_frequency=3,
    )

    assert isinstance(output, list)
    assert output[0].shape == (1, 2, 2, 2, 2)
    assert output[1].shape == (1, 2, 3, 3)


def test_ltxav_output_processor_trims_reference_audio_tokens():
    processor = LTXAVOutputProcessor(
        video_hidden_dim=4,
        video_out_channels=1,
        audio_hidden_dim=4,
        audio_out_channels=4,
        dtype=torch.float32,
    )
    with torch.no_grad():
        for parameter in processor.parameters():
            parameter.uniform_(-0.05, 0.05)
    output = processor(
        torch.randn(1, 2, 4),
        torch.randn(1, 4, 4),
        video_embedded_timestep=torch.randn(1, 2, 4),
        audio_embedded_timestep=torch.randn(1, 4, 4),
        orig_shape=(1, 1, 2, 1, 1),
        ref_audio_seq_len=1,
        audio_channels=2,
        audio_frequency=2,
    )

    assert isinstance(output, list)
    assert output[1].shape == (1, 2, 3, 2)

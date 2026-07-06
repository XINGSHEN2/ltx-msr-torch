import torch

from ltx_msr_torch.ltx_timestep import AdaLayerNormSingle, CompressedTimestep
from ltx_msr_torch.ltxav_timestep import prepare_ltxav_timesteps


def _adaln(dim: int, coefficient: int) -> AdaLayerNormSingle:
    module = AdaLayerNormSingle(embedding_dim=dim, embedding_coefficient=coefficient, dtype=torch.float32)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.05, 0.05)
    return module


def test_prepare_ltxav_timesteps_shapes_and_compresses_video_timesteps():
    prepared = prepare_ltxav_timesteps(
        timestep=torch.tensor([[0.1, 0.1, 0.1, 0.2, 0.2, 0.2]], dtype=torch.float32),
        batch_size=1,
        hidden_dtype=torch.float32,
        video_adaln_single=_adaln(4, 9),
        audio_adaln_single=_adaln(4, 9),
        av_ca_video_scale_shift_adaln_single=_adaln(4, 4),
        av_ca_a2v_gate_adaln_single=_adaln(4, 1),
        av_ca_audio_scale_shift_adaln_single=_adaln(4, 4),
        av_ca_v2a_gate_adaln_single=_adaln(4, 1),
        video_prompt_adaln_single=_adaln(4, 2),
        audio_prompt_adaln_single=_adaln(4, 2),
        audio_timestep=torch.tensor([[0.3, 0.4]], dtype=torch.float32),
        orig_shape=(1, 1, 2, 1, 3),
        has_spatial_mask=False,
        timestep_scale_multiplier=1000.0,
        av_ca_timestep_scale_multiplier=1.0,
    )

    assert isinstance(prepared.video_timestep, CompressedTimestep)
    assert prepared.video_timestep.data.shape == (1, 2, 36)
    assert prepared.video_timestep.expand().shape == (1, 6, 36)
    assert isinstance(prepared.video_embedded_timestep, CompressedTimestep)
    assert prepared.video_embedded_timestep.data.shape == (1, 2, 4)

    assert prepared.audio_timestep is not None
    assert prepared.audio_timestep.shape == (1, 2, 36)
    assert prepared.audio_embedded_timestep is not None
    assert prepared.audio_embedded_timestep.shape == (1, 2, 4)

    assert isinstance(prepared.video_cross_scale_shift_timestep, CompressedTimestep)
    assert prepared.video_cross_scale_shift_timestep.data.shape == (1, 2, 16)
    assert prepared.audio_cross_scale_shift_timestep is not None
    assert prepared.audio_cross_scale_shift_timestep.shape == (1, 2, 16)
    assert isinstance(prepared.video_cross_gate_timestep, CompressedTimestep)
    assert prepared.video_cross_gate_timestep.data.shape == (1, 2, 4)
    assert prepared.audio_cross_gate_timestep is not None
    assert prepared.audio_cross_gate_timestep.shape == (1, 2, 4)

    assert prepared.video_prompt_timestep is not None
    assert prepared.video_prompt_timestep.shape == (1, 1, 8)
    assert prepared.audio_prompt_timestep is not None
    assert prepared.audio_prompt_timestep.shape == (1, 1, 8)


def test_prepare_ltxav_timesteps_expands_reference_audio_timestep():
    prepared = prepare_ltxav_timesteps(
        timestep=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        batch_size=1,
        hidden_dtype=torch.float32,
        video_adaln_single=_adaln(4, 9),
        audio_adaln_single=_adaln(4, 9),
        av_ca_video_scale_shift_adaln_single=_adaln(4, 4),
        av_ca_a2v_gate_adaln_single=_adaln(4, 1),
        av_ca_audio_scale_shift_adaln_single=_adaln(4, 4),
        av_ca_v2a_gate_adaln_single=_adaln(4, 1),
        audio_timestep=torch.tensor([0.4], dtype=torch.float32),
        orig_shape=(1, 1, 2, 1, 1),
        has_spatial_mask=False,
        ref_audio_seq_len=1,
        target_audio_seq_len=2,
    )

    assert prepared.audio_timestep is not None
    assert prepared.audio_timestep.shape == (1, 3, 36)
    assert prepared.audio_cross_scale_shift_timestep is not None
    assert prepared.audio_cross_scale_shift_timestep.shape == (1, 3, 16)

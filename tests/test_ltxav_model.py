import torch

from ltx_msr_torch.ltxav_model import LTXAVModel, LTXAVModelConfig


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

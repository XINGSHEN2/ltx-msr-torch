import torch

from ltx_msr_torch.ltxav_io import LTXAVInputProjection
from ltx_msr_torch.ltxav_prepare import (
    prepare_attention_mask,
    prepare_ltxav_block_inputs,
    prepare_ltxav_positional_embeddings,
    split_ltxav_context,
)


def test_prepare_attention_mask_matches_comfy_shape():
    mask = torch.tensor([[0, 1, 1]], dtype=torch.long)
    prepared = prepare_attention_mask(mask, torch.float32)

    assert prepared.shape == (1, 1, 1, 3)
    assert prepared[0, 0, 0, 0] < -1e30
    assert prepared[0, 0, 0, 1].item() == 0.0


def test_split_ltxav_context_splits_last_dim():
    context = torch.randn(1, 5, 6)
    video, audio = split_ltxav_context(context, video_dim=4, audio_dim=2)

    assert torch.equal(video, context[:, :, :4])
    assert torch.equal(audio, context[:, :, 4:])


def test_prepare_ltxav_positional_embeddings_shapes_small_dims():
    projection = LTXAVInputProjection(
        video_in_channels=2,
        video_hidden_dim=12,
        audio_in_channels=6,
        audio_hidden_dim=4,
        dtype=torch.float32,
    )
    projected = projection(torch.randn(1, 2, 2, 2, 2), torch.randn(1, 2, 4, 3))
    video_pe, audio_pe, video_cross_pe, audio_cross_pe = prepare_ltxav_positional_embeddings(
        projected,
        frame_rate=25,
        dtype=torch.float32,
        video_dim=12,
        audio_dim=4,
        audio_cross_dim=4,
        video_heads=2,
        audio_heads=2,
        video_max_pos=(20, 2048, 2048),
        audio_max_pos=(20,),
    )

    assert video_pe[0].shape == (1, 2, 8, 3)
    assert audio_pe[0].shape == (1, 2, 4, 1)
    assert video_cross_pe[0].shape == (1, 2, 8, 1)
    assert audio_cross_pe[0].shape == (1, 2, 4, 1)


def test_prepare_ltxav_block_inputs_combines_projection_context_mask_and_rope():
    projection = LTXAVInputProjection(
        video_in_channels=2,
        video_hidden_dim=12,
        audio_in_channels=6,
        audio_hidden_dim=4,
        dtype=torch.float32,
    )
    video_latents = torch.randn(1, 2, 2, 2, 2)
    audio_latents = torch.randn(1, 2, 4, 3)
    context = torch.randn(1, 5, 16)
    mask = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)

    prepared = prepare_ltxav_block_inputs(
        input_projection=projection,
        video_latents=video_latents,
        audio_latents=audio_latents,
        context=context,
        attention_mask=mask,
        frame_rate=25,
        video_dim=12,
        audio_dim=4,
        audio_cross_dim=4,
        video_heads=2,
        audio_heads=2,
    )

    assert prepared.projected.video_tokens.shape == (1, 8, 12)
    assert prepared.projected.audio_tokens.shape == (1, 4, 4)
    assert torch.equal(prepared.video_context, context[:, :, :12])
    assert torch.equal(prepared.audio_context, context[:, :, 12:])
    assert prepared.attention_mask.shape == (1, 1, 1, 5)
    assert prepared.video_pe[0].shape == (1, 2, 8, 3)
    assert prepared.audio_pe[0].shape == (1, 2, 4, 1)

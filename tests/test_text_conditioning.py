import torch

from ltx_msr_torch.gemma_tokenizer import GemmaTokenizer
from ltx_msr_torch.text_conditioning import (
    attention_mask_tensor,
    build_text_conditioning_inputs,
    build_text_conditioning_inputs_from_plan,
    encode_ltx_text_conditioning,
    estimate_gemma_memory_mb,
    trim_left_padding_from_layer_hidden,
)
from ltx_msr_torch.text_projection import DualLinearTextProjection


def test_build_text_conditioning_inputs_matches_comfy_left_pad_mask_rule():
    pairs = (((0, 1.0), (0, 1.0), (2, 1.0), (10, 1.0), (0, 1.0)),)
    inputs = build_text_conditioning_inputs(pairs)

    assert inputs.token_ids == ((0, 0, 2, 10, 0),)
    assert inputs.attention_mask == ((0, 0, 1, 1, 1),)
    assert inputs.num_tokens == (3,)
    assert inputs.real_token_count == 3


def test_build_text_conditioning_inputs_from_gemma_plan_uses_real_token_count():
    tokenizer = GemmaTokenizer.from_config_paths()
    token_plan = tokenizer.tokenize_with_weights("参考图1：红色水枪")
    inputs = build_text_conditioning_inputs_from_plan(token_plan)

    assert inputs.token_ids[0] == token_plan.padded_input_ids
    assert inputs.attention_mask[0] == token_plan.attention_mask
    assert inputs.real_token_count == len(token_plan.input_ids)


def test_trim_left_padding_from_layer_hidden_uses_sequence_axis():
    hidden = torch.arange(1 * 2 * 5 * 3, dtype=torch.float32).reshape(1, 2, 5, 3)
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)

    trimmed = trim_left_padding_from_layer_hidden(hidden, mask)

    assert torch.equal(trimmed, hidden[:, :, -3:])


def test_encode_ltx_text_conditioning_projects_trimmed_hidden():
    projection = DualLinearTextProjection(input_dim=4, video_dim=2, audio_dim=1, dtype=torch.float32)
    with torch.no_grad():
        projection.video_aggregate_embed.weight.fill_(1.0)
        projection.video_aggregate_embed.bias.zero_()
        projection.audio_aggregate_embed.weight.fill_(1.0)
        projection.audio_aggregate_embed.bias.zero_()
    hidden = torch.ones((1, 2, 4, 2), dtype=torch.float32)
    mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    output = encode_ltx_text_conditioning(hidden, attention_mask=mask, projection=projection)

    assert output.conditioning.shape == (1, 3, 3)
    assert output.conditioning.dtype == torch.float32
    assert torch.equal(output.attention_mask, mask.flatten().unsqueeze(0))
    assert output.extra == {"unprocessed_ltxav_embeds": True}


def test_attention_mask_tensor_and_memory_estimate_match_workflow_floor():
    pairs = (((0, 1.0), (0, 1.0), (2, 1.0), (10, 1.0)),)
    inputs = build_text_conditioning_inputs(pairs)

    assert torch.equal(attention_mask_tensor(inputs), torch.tensor([[0, 0, 1, 1]], dtype=torch.long))
    assert estimate_gemma_memory_mb(pairs) == 642 * 6.0
    assert estimate_gemma_memory_mb(pairs, bf16=True) == 642 * 3.0

import sys

import torch

from ltx_msr_torch.ltx_blocks import BasicAVTransformerBlock, BasicTransformerBlock, apply_cross_attention_adaln
from ltx_msr_torch.ltx_timestep import ADALN_CROSS_ATTN_PARAMS_COUNT


def _enable_comfy_cpu_import():
    sys.path.insert(0, "/home/xingshen/ComfyUI")
    import comfy.options

    sys.argv = ["ltx_blocks_test", "--cpu"]
    comfy.options.args_parsing = True


def _fill_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.1, 0.1)


def test_basic_transformer_block_weight_shapes_match_video_block0():
    block = BasicTransformerBlock(
        dim=4096,
        heads=32,
        dim_head=128,
        context_dim=4096,
        cross_attention_adaln=True,
        device="meta",
    )

    assert tuple(block.scale_shift_table.shape) == (ADALN_CROSS_ATTN_PARAMS_COUNT, 4096)
    assert tuple(block.prompt_scale_shift_table.shape) == (2, 4096)
    assert tuple(block.attn1.to_q.weight.shape) == (4096, 4096)
    assert tuple(block.attn2.to_k.weight.shape) == (4096, 4096)
    assert tuple(block.ff.net[0].proj.weight.shape) == (16384, 4096)


def test_basic_transformer_block_matches_comfy_without_cross_adaln():
    _enable_comfy_cpu_import()
    import comfy.ops
    from comfy.ldm.lightricks.model import BasicTransformerBlock as ComfyBasicTransformerBlock

    local = BasicTransformerBlock(dim=4, heads=2, dim_head=2, context_dim=6, dtype=torch.float32)
    comfy = ComfyBasicTransformerBlock(
        4,
        2,
        2,
        context_dim=6,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    _fill_module(local)
    comfy.load_state_dict(local.state_dict(), strict=True)
    x = torch.randn(1, 3, 4)
    context = torch.randn(1, 5, 6)
    timestep = torch.randn(1, 3, 6 * 4)

    local_out = local(x.clone(), context=context, timestep=timestep)
    comfy_out = comfy(x.clone(), context=context, timestep=timestep, transformer_options={})

    assert torch.allclose(local_out, comfy_out, atol=1e-5, rtol=1e-5)


def test_basic_transformer_block_matches_comfy_with_cross_adaln():
    _enable_comfy_cpu_import()
    import comfy.ops
    from comfy.ldm.lightricks.model import BasicTransformerBlock as ComfyBasicTransformerBlock

    local = BasicTransformerBlock(
        dim=4,
        heads=2,
        dim_head=2,
        context_dim=4,
        cross_attention_adaln=True,
        dtype=torch.float32,
    )
    comfy = ComfyBasicTransformerBlock(
        4,
        2,
        2,
        context_dim=4,
        cross_attention_adaln=True,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    _fill_module(local)
    comfy.load_state_dict(local.state_dict(), strict=True)
    x = torch.randn(1, 3, 4)
    context = torch.randn(1, 5, 4)
    timestep = torch.randn(1, 3, 9 * 4)
    prompt_timestep = torch.randn(1, 1, 2 * 4)

    local_out = local(x.clone(), context=context, timestep=timestep, prompt_timestep=prompt_timestep)
    comfy_out = comfy(
        x.clone(),
        context=context,
        timestep=timestep,
        prompt_timestep=prompt_timestep,
        transformer_options={},
    )

    assert torch.allclose(local_out, comfy_out, atol=1e-5, rtol=1e-5)


def test_apply_cross_attention_adaln_output_shape():
    block = BasicTransformerBlock(
        dim=4,
        heads=2,
        dim_head=2,
        context_dim=4,
        cross_attention_adaln=True,
        dtype=torch.float32,
    )
    x = torch.randn(1, 3, 4)
    context = torch.randn(1, 5, 4)
    q_shift = torch.randn(1, 3, 4)
    q_scale = torch.randn(1, 3, 4)
    q_gate = torch.randn(1, 3, 4)
    prompt_timestep = torch.randn(1, 1, 8)

    output = apply_cross_attention_adaln(
        x,
        context,
        block.attn2,
        q_shift,
        q_scale,
        q_gate,
        block.prompt_scale_shift_table,
        prompt_timestep,
    )

    assert output.shape == x.shape


def test_basic_av_transformer_block_weight_shapes_match_ltxav_block0():
    block = BasicAVTransformerBlock(
        video_dim=4096,
        audio_dim=2048,
        video_heads=32,
        audio_heads=32,
        video_dim_head=128,
        audio_dim_head=64,
        video_context_dim=4096,
        audio_context_dim=2048,
        apply_gated_attention=True,
        cross_attention_adaln=True,
        device="meta",
    )

    assert tuple(block.scale_shift_table.shape) == (9, 4096)
    assert tuple(block.audio_scale_shift_table.shape) == (9, 2048)
    assert tuple(block.scale_shift_table_a2v_ca_video.shape) == (5, 4096)
    assert tuple(block.scale_shift_table_a2v_ca_audio.shape) == (5, 2048)
    assert tuple(block.attn1.to_gate_logits.weight.shape) == (32, 4096)
    assert tuple(block.audio_attn1.to_q.weight.shape) == (2048, 2048)
    assert tuple(block.audio_to_video_attn.to_q.weight.shape) == (2048, 4096)
    assert tuple(block.audio_to_video_attn.to_out[0].weight.shape) == (4096, 2048)
    assert tuple(block.video_to_audio_attn.to_k.weight.shape) == (2048, 4096)


def test_basic_av_transformer_block_matches_comfy_without_av_cross_attention():
    _enable_comfy_cpu_import()
    import comfy.ops
    from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock as ComfyBasicAVTransformerBlock

    local = BasicAVTransformerBlock(
        video_dim=4,
        audio_dim=4,
        video_heads=2,
        audio_heads=2,
        video_dim_head=2,
        audio_dim_head=2,
        video_context_dim=4,
        audio_context_dim=4,
        apply_gated_attention=True,
        cross_attention_adaln=True,
        dtype=torch.float32,
    )
    comfy = ComfyBasicAVTransformerBlock(
        v_dim=4,
        a_dim=4,
        v_heads=2,
        a_heads=2,
        vd_head=2,
        ad_head=2,
        v_context_dim=4,
        a_context_dim=4,
        apply_gated_attention=True,
        cross_attention_adaln=True,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    _fill_module(local)
    comfy.load_state_dict(local.state_dict(), strict=True)
    vx = torch.randn(1, 3, 4)
    ax = torch.randn(1, 2, 4)
    v_context = torch.randn(1, 5, 4)
    a_context = torch.randn(1, 5, 4)
    v_timestep = torch.randn(1, 3, 9 * 4)
    a_timestep = torch.randn(1, 2, 9 * 4)
    v_prompt_timestep = torch.randn(1, 1, 8)
    a_prompt_timestep = torch.randn(1, 1, 8)
    options = {"a2v_cross_attn": False, "v2a_cross_attn": False}

    local_out = local(
        (vx.clone(), ax.clone()),
        video_context=v_context,
        audio_context=a_context,
        video_timestep=v_timestep,
        audio_timestep=a_timestep,
        transformer_options=options,
        video_prompt_timestep=v_prompt_timestep,
        audio_prompt_timestep=a_prompt_timestep,
    )
    comfy_out = comfy(
        (vx.clone(), ax.clone()),
        v_context=v_context,
        a_context=a_context,
        v_timestep=v_timestep,
        a_timestep=a_timestep,
        transformer_options=options,
        v_prompt_timestep=v_prompt_timestep,
        a_prompt_timestep=a_prompt_timestep,
    )

    assert torch.allclose(local_out[0], comfy_out[0], atol=1e-5, rtol=1e-5)
    assert torch.allclose(local_out[1], comfy_out[1], atol=1e-5, rtol=1e-5)

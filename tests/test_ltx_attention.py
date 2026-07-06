import sys

import torch

from ltx_msr_torch.ltx_attention import (
    CrossAttention,
    FeedForward,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
    rms_norm,
    scaled_dot_product_attention,
)


def _enable_comfy_cpu_import():
    sys.path.insert(0, "/home/xingshen/ComfyUI")
    import comfy.options

    sys.argv = ["ltx_attention_test", "--cpu"]
    comfy.options.args_parsing = True


def test_rms_norm_matches_formula():
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 0.0, -4.0]]])
    weight = torch.tensor([1.0, 2.0, 3.0])

    output = rms_norm(x, weight, eps=1e-6)
    expected = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * weight

    assert torch.allclose(output, expected)


def test_rotary_embeddings_match_comfy_functions():
    _enable_comfy_cpu_import()
    from comfy.ldm.lightricks.model import (
        apply_interleaved_rotary_emb as comfy_interleaved,
        apply_split_rotary_emb as comfy_split,
    )

    x = torch.randn(1, 4, 8)
    cos = torch.randn(1, 4, 8)
    sin = torch.randn(1, 4, 8)
    assert torch.allclose(apply_interleaved_rotary_emb(x, cos, sin), comfy_interleaved(x, cos, sin))

    split_x = torch.randn(1, 4, 8)
    split_cos = torch.randn(1, 2, 4, 2)
    split_sin = torch.randn(1, 2, 4, 2)
    assert torch.allclose(apply_split_rotary_emb(split_x, split_cos, split_sin), comfy_split(split_x, split_cos, split_sin))


def test_feed_forward_matches_comfy_with_same_weights():
    _enable_comfy_cpu_import()
    import comfy.ops
    from comfy.ldm.lightricks.model import FeedForward as ComfyFeedForward

    local = FeedForward(4, 4, mult=2, dtype=torch.float32)
    comfy = ComfyFeedForward(4, 4, mult=2, dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for parameter in local.parameters():
            parameter.uniform_(-0.1, 0.1)
    comfy.load_state_dict(local.state_dict(), strict=True)
    x = torch.randn(1, 3, 4)

    assert torch.allclose(local(x), comfy(x))


def test_scaled_dot_product_attention_shapes_and_masking():
    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    mask = torch.zeros(1, 1, 3, 4)
    mask[..., -1] = torch.finfo(torch.float32).min

    output = scaled_dot_product_attention(q, k, v, heads=2, mask=mask)

    assert output.shape == (1, 3, 8)


def test_cross_attention_forward_and_gating_shapes():
    module = CrossAttention(
        query_dim=4,
        context_dim=6,
        heads=2,
        dim_head=3,
        apply_gated_attention=True,
        dtype=torch.float32,
    )
    x = torch.randn(1, 5, 4)
    context = torch.randn(1, 7, 6)

    output = module(x, context=context)

    assert output.shape == x.shape
    assert module.config.inner_dim == 6
    assert module.to_gate_logits is not None


def test_cross_attention_weight_shapes_match_ltxav_block_keys():
    module = CrossAttention(query_dim=4096, context_dim=4096, heads=32, dim_head=128, apply_gated_attention=True, device="meta")

    assert tuple(module.to_q.weight.shape) == (4096, 4096)
    assert tuple(module.to_k.weight.shape) == (4096, 4096)
    assert tuple(module.to_v.weight.shape) == (4096, 4096)
    assert tuple(module.to_out[0].weight.shape) == (4096, 4096)
    assert tuple(module.to_gate_logits.weight.shape) == (32, 4096)
    assert tuple(module.q_norm.weight.shape) == (4096,)

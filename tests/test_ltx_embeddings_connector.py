import sys
import importlib.util
import types
from pathlib import Path

import torch

from ltx_msr_torch.ltx_embeddings_connector import (
    BasicTransformerBlock1D,
    Embeddings1DConnector,
    load_embeddings_connector_state_dict,
)
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def _enable_comfy_cpu_import():
    sys.path.insert(0, "/home/xingshen/ComfyUI")
    import comfy.options

    sys.argv = ["ltx_connector_test", "--cpu"]
    comfy.options.args_parsing = True


def _load_comfy_embeddings_connector():
    package_name = "comfy_ltxvideo_test"
    root = Path("/home/xingshen/ComfyUI/custom_nodes/ComfyUI-LTXVideo")
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root)]
        sys.modules[package_name] = package
    module_name = f"{package_name}.embeddings_connector"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, root / "embeddings_connector.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fill_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.1, 0.1)


def test_basic_transformer_block_1d_matches_comfy():
    _enable_comfy_cpu_import()
    import comfy.ops
    ComfyBasicTransformerBlock1D = _load_comfy_embeddings_connector().BasicTransformerBlock1D

    local = BasicTransformerBlock1D(4, 2, 2, apply_gated_attention=True, dtype=torch.float32)
    comfy = ComfyBasicTransformerBlock1D(
        4,
        2,
        2,
        apply_gated_attention=True,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    _fill_module(local)
    comfy.load_state_dict(local.state_dict(), strict=True)
    x = torch.randn(1, 4, 4)

    assert torch.allclose(local(x), comfy(x), atol=1e-5, rtol=1e-5)


def test_embeddings_connector_matches_comfy_small_without_padding_mask():
    _enable_comfy_cpu_import()
    import comfy.ops
    ComfyEmbeddings1DConnector = _load_comfy_embeddings_connector().Embeddings1DConnector

    local = Embeddings1DConnector(
        attention_head_dim=2,
        num_attention_heads=2,
        num_layers=2,
        num_learnable_registers=0,
        apply_gated_attention=True,
        split_rope=True,
        double_precision_rope=True,
        dtype=torch.float32,
    )
    comfy = ComfyEmbeddings1DConnector(
        attention_head_dim=2,
        num_attention_heads=2,
        num_layers=2,
        num_learnable_registers=0,
        apply_gated_attention=True,
        split_rope=True,
        double_precision_rope=True,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    _fill_module(local)
    comfy.load_state_dict(local.state_dict(), strict=True)
    x = torch.randn(1, 4, 4)

    local_out = local(x)
    comfy_out = comfy(x)

    assert torch.allclose(local_out[0], comfy_out[0], atol=1e-5, rtol=1e-5)
    assert local_out[1] is None and comfy_out[1] is None


def test_embeddings_connector_replaces_left_padding_with_registers():
    connector = Embeddings1DConnector(
        attention_head_dim=2,
        num_attention_heads=2,
        num_layers=0,
        num_learnable_registers=2,
        dtype=torch.float32,
    )
    with torch.no_grad():
        connector.learnable_registers.copy_(torch.tensor([[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]))
    hidden = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]])
    mask = torch.tensor([[[[-10000.0, -10000.0, 0.0, 0.0]]]])

    replaced, new_mask = connector._replace_padding_with_registers(hidden, mask)

    assert torch.equal(replaced[:, :2], hidden[:, 2:])
    assert torch.equal(replaced[0, 2], connector.learnable_registers[0])
    assert torch.equal(replaced[0, 3], connector.learnable_registers[1])
    assert torch.equal(new_mask, torch.zeros_like(mask))


def test_load_embeddings_connector_state_dict_reads_workflow_shapes():
    paths = resolve_workflow_model_paths(default_workflow_config())
    video = load_embeddings_connector_state_dict(paths.checkpoint, "video")
    audio = load_embeddings_connector_state_dict(paths.checkpoint, "audio")

    assert len(video) == 129
    assert len(audio) == 129
    assert video["learnable_registers"].shape == (128, 4096)
    assert audio["learnable_registers"].shape == (128, 2048)
    assert video["transformer_1d_blocks.0.attn1.to_q.weight"].shape == (4096, 4096)
    assert audio["transformer_1d_blocks.0.attn1.to_q.weight"].shape == (2048, 2048)
    assert video["learnable_registers"].dtype == torch.bfloat16


def test_embeddings_connector_meta_shapes_match_workflow_video_audio():
    video = Embeddings1DConnector(
        attention_head_dim=128,
        num_attention_heads=32,
        num_layers=8,
        apply_gated_attention=True,
        device="meta",
    )
    audio = Embeddings1DConnector(
        attention_head_dim=64,
        num_attention_heads=32,
        num_layers=8,
        apply_gated_attention=True,
        device="meta",
    )

    assert tuple(video.learnable_registers.shape) == (128, 4096)
    assert tuple(video.transformer_1d_blocks[0].attn1.to_gate_logits.weight.shape) == (32, 4096)
    assert tuple(audio.learnable_registers.shape) == (128, 2048)
    assert tuple(audio.transformer_1d_blocks[0].ff.net[0].proj.weight.shape) == (8192, 2048)

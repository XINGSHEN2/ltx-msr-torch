import sys

import torch

from ltx_msr_torch.ltx_timestep import (
    ADALN_CROSS_ATTN_PARAMS_COUNT,
    AdaLayerNormSingle,
    CompressedTimestep,
    compute_prompt_timestep,
    get_timestep_embedding,
    load_adaln_single_state_dict,
)
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def _enable_comfy_cpu_import():
    sys.path.insert(0, "/home/xingshen/ComfyUI")
    import comfy.options

    sys.argv = ["ltx_timestep_test", "--cpu"]
    comfy.options.args_parsing = True


def test_get_timestep_embedding_matches_comfy_formula():
    _enable_comfy_cpu_import()
    from comfy.ldm.lightricks.model import get_timestep_embedding as comfy_embedding

    timesteps = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
    local = get_timestep_embedding(timesteps, 7, flip_sin_to_cos=True, downscale_freq_shift=0)
    comfy = comfy_embedding(timesteps, 7, flip_sin_to_cos=True, downscale_freq_shift=0)

    assert torch.allclose(local, comfy)


def test_adaln_single_forward_shape_and_prompt_timestep():
    module = AdaLayerNormSingle(embedding_dim=4, embedding_coefficient=3, dtype=torch.float32)
    timestep = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)

    output, embedded = module(timestep.flatten(), batch_size=1, hidden_dtype=torch.float32)
    prompt = compute_prompt_timestep(module, timestep, batch_size=1, hidden_dtype=torch.float32)

    assert output.shape == (3, 12)
    assert embedded.shape == (3, 4)
    assert prompt.shape == (1, 1, 12)


def test_compressed_timestep_keeps_one_row_per_frame_and_expands():
    tensor = torch.arange(1 * 6 * 4, dtype=torch.float32).reshape(1, 6, 4)
    compressed = CompressedTimestep(tensor, patches_per_frame=3)

    assert compressed.data.shape == (1, 2, 4)
    assert torch.equal(compressed.data, tensor.reshape(1, 2, 3, 4)[:, :, 0, :])

    expanded = compressed.expand()

    assert expanded.shape == tensor.shape
    assert torch.equal(expanded.reshape(1, 2, 3, 4)[:, :, 0, :], compressed.data)
    assert torch.equal(expanded.reshape(1, 2, 3, 4)[:, :, 1, :], compressed.data)


def test_compressed_timestep_expand_for_computation_matches_full_repeated_timestep():
    table = torch.randn(6, 4)
    per_frame = torch.randn(1, 2, 6 * 4)
    full = per_frame.unsqueeze(2).expand(1, 2, 3, 6 * 4).reshape(1, 6, 6 * 4)
    compressed = CompressedTimestep(per_frame, patches_per_frame=3, per_frame=True)

    local = compressed.expand_for_computation(table, batch_size=1, indices=slice(0, 3))
    expected = (
        table[None, None, :3]
        + full.reshape(1, 6, 6, 4)[:, :, :3, :]
    ).unbind(dim=2)

    assert len(local) == len(expected)
    for local_value, expected_value in zip(local, expected):
        assert torch.allclose(local_value, expected_value)


def test_adaln_single_matches_comfy_module_with_same_weights():
    _enable_comfy_cpu_import()
    from comfy.ldm.lightricks.model import AdaLayerNormSingle as ComfyAdaLayerNormSingle
    import comfy.ops

    local = AdaLayerNormSingle(embedding_dim=4, embedding_coefficient=3, dtype=torch.float32)
    comfy = ComfyAdaLayerNormSingle(
        4,
        embedding_coefficient=3,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for parameter in local.parameters():
            parameter.uniform_(-0.1, 0.1)
    comfy.load_state_dict(local.state_dict(), strict=True)
    timestep = torch.tensor([0.1, 0.2], dtype=torch.float32)

    local_out = local(timestep, batch_size=1, hidden_dtype=torch.float32)
    comfy_out = comfy(timestep, {"resolution": None, "aspect_ratio": None}, batch_size=1, hidden_dtype=torch.float32)

    assert torch.allclose(local_out[0], comfy_out[0])
    assert torch.allclose(local_out[1], comfy_out[1])


def test_load_adaln_single_state_dict_reads_video_and_audio_workflow_weights():
    paths = resolve_workflow_model_paths(default_workflow_config())
    video = load_adaln_single_state_dict(paths.checkpoint, "adaln_single")
    audio = load_adaln_single_state_dict(paths.checkpoint, "audio_adaln_single")

    assert video["linear.weight"].shape == (ADALN_CROSS_ATTN_PARAMS_COUNT * 4096, 4096)
    assert audio["linear.weight"].shape == (ADALN_CROSS_ATTN_PARAMS_COUNT * 2048, 2048)
    assert video["emb.timestep_embedder.linear_1.weight"].shape == (4096, 256)
    assert audio["emb.timestep_embedder.linear_1.weight"].shape == (2048, 256)
    assert video["linear.weight"].dtype == torch.bfloat16


def test_adaln_single_loads_real_subset_and_runs_forward_on_cpu():
    paths = resolve_workflow_model_paths(default_workflow_config())
    state = load_adaln_single_state_dict(paths.checkpoint, "audio_adaln_single")
    module = AdaLayerNormSingle(
        embedding_dim=2048,
        embedding_coefficient=ADALN_CROSS_ATTN_PARAMS_COUNT,
        dtype=torch.bfloat16,
    )
    module.load_state_dict(state, strict=True)
    timestep = torch.tensor([1000.0], dtype=torch.bfloat16)

    output, embedded = module(timestep, batch_size=1, hidden_dtype=torch.bfloat16)

    assert output.shape == (1, ADALN_CROSS_ATTN_PARAMS_COUNT * 2048)
    assert embedded.shape == (1, 2048)
    assert output.dtype == torch.bfloat16

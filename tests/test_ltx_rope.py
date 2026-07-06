import sys

import torch

from ltx_msr_torch.ltx_patchify import latent_to_pixel_coords, symmetric_patchify_video
from ltx_msr_torch.ltx_rope import (
    generate_freq_grid_np,
    generate_freqs,
    get_fractional_positions,
    precompute_ltx_freqs_cis,
    split_freqs_cis,
)


def _enable_comfy_cpu_import():
    sys.path.insert(0, "/home/xingshen/ComfyUI")
    import comfy.options

    sys.argv = ["ltx_rope_test", "--cpu"]
    comfy.options.args_parsing = True


def test_get_fractional_positions_scales_each_axis():
    grid = torch.tensor([[[0.0, 10.0], [0.0, 32.0], [0.0, 64.0]]])
    fractional = get_fractional_positions(grid, (20, 2048, 2048))

    assert torch.allclose(fractional[0, 1], torch.tensor([0.5, 32.0 / 2048.0, 64.0 / 2048.0]))


def test_generate_freqs_matches_comfy_function():
    _enable_comfy_cpu_import()
    from comfy.ldm.lightricks.model import generate_freqs as comfy_generate_freqs

    grid = torch.tensor([[[0.0, 10.0], [0.0, 32.0], [0.0, 64.0]]])
    indices = generate_freq_grid_np(10000.0, 3, 384)
    local = generate_freqs(indices, grid, (20, 2048, 2048), False)
    comfy = comfy_generate_freqs(indices, grid, (20, 2048, 2048), False)

    assert torch.allclose(local, comfy)


def test_split_freqs_cis_matches_comfy_function():
    _enable_comfy_cpu_import()
    from comfy.ldm.lightricks.model import split_freqs_cis as comfy_split_freqs_cis

    freqs = torch.randn(1, 5, 192)
    local = split_freqs_cis(freqs, pad_size=64, num_attention_heads=32)
    comfy = comfy_split_freqs_cis(freqs, pad_size=64, num_attention_heads=32)

    assert torch.allclose(local[0], comfy[0])
    assert torch.allclose(local[1], comfy[1])


def test_precompute_ltx_freqs_cis_for_video_patch_coords():
    latents = torch.zeros((1, 128, 2, 4, 4), dtype=torch.bfloat16)
    patchified = symmetric_patchify_video(latents, patch_size=1, start_end=True)
    pixel_coords = latent_to_pixel_coords(patchified.latent_coords, (8, 32, 32), causal_fix=True)
    pixel_coords = pixel_coords.to(torch.float32)
    pixel_coords[:, 0] = pixel_coords[:, 0] * (1.0 / 25)

    cos, sin, split = precompute_ltx_freqs_cis(
        pixel_coords,
        dim=4096,
        out_dtype=torch.bfloat16,
        max_pos=(20, 2048, 2048),
        use_middle_indices_grid=True,
        num_attention_heads=32,
        split=True,
        double_precision_grid=True,
    )

    assert split is True
    assert cos.shape == (1, 32, 32, 64)
    assert sin.shape == (1, 32, 32, 64)
    assert cos.dtype == torch.bfloat16
    assert sin.dtype == torch.bfloat16


def test_precompute_ltx_freqs_cis_supports_audio_temporal_coords():
    coords = torch.tensor([[[[0.0, 0.01], [0.01, 0.05], [0.05, 0.09]]]], dtype=torch.float32)
    cos, sin, split = precompute_ltx_freqs_cis(
        coords,
        dim=2048,
        out_dtype=torch.float32,
        max_pos=(20,),
        use_middle_indices_grid=True,
        num_attention_heads=32,
        split=True,
        double_precision_grid=True,
    )

    assert split is True
    assert cos.shape == (1, 32, 3, 32)
    assert sin.shape == (1, 32, 3, 32)

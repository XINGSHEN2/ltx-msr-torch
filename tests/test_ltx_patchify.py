import sys

import torch

from ltx_msr_torch.ltx_patchify import (
    audio_latent_time_in_seconds,
    latent_to_pixel_coords,
    patchify_audio,
    symmetric_patchify_video,
    symmetric_unpatchify_video,
    unpatchify_audio,
    video_latent_coords,
)


def test_symmetric_patchify_video_round_trip_patch_size_one():
    latents = torch.arange(1 * 2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)
    result = symmetric_patchify_video(latents, patch_size=1, start_end=True)
    restored = symmetric_unpatchify_video(
        result.patches,
        output_height=4,
        output_width=5,
        output_num_frames=3,
        out_channels=2,
        patch_size=1,
    )

    assert result.patches.shape == (1, 60, 2)
    assert result.latent_coords.shape == (1, 3, 60, 2)
    assert torch.equal(restored, latents)


def test_symmetric_patchify_video_matches_comfy_patchifier():
    sys.path.insert(0, "/home/xingshen/ComfyUI")
    from comfy.ldm.lightricks.symmetric_patchifier import SymmetricPatchifier

    latents = torch.arange(1 * 2 * 2 * 4 * 6, dtype=torch.float32).reshape(1, 2, 2, 4, 6)
    local = symmetric_patchify_video(latents, patch_size=2, start_end=True)
    comfy_patches, comfy_coords = SymmetricPatchifier(2, start_end=True).patchify(latents)
    comfy_restored = SymmetricPatchifier(2, start_end=True).unpatchify(
        comfy_patches,
        output_height=4,
        output_width=6,
        output_num_frames=2,
        out_channels=2,
    )
    local_restored = symmetric_unpatchify_video(
        local.patches,
        output_height=4,
        output_width=6,
        output_num_frames=2,
        out_channels=2,
        patch_size=2,
    )

    assert torch.equal(local.patches, comfy_patches)
    assert torch.equal(local.latent_coords, comfy_coords)
    assert torch.equal(local_restored, comfy_restored)


def test_video_latent_coords_and_pixel_coords_match_causal_fix():
    coords = video_latent_coords(frames=2, height=2, width=2, batch_size=1, start_end=True)
    pixels = latent_to_pixel_coords(coords, (8, 32, 32), causal_fix=True)

    assert coords.shape == (1, 3, 8, 2)
    assert pixels[0, 0, 0, 0].item() == 0
    assert pixels[0, 0, -1, 0].item() == 1
    assert pixels[0, 1, 1, 1].item() == 32


def test_audio_patchify_round_trip_and_timings():
    audio = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)
    result = patchify_audio(audio, start_end=True)
    restored = unpatchify_audio(result.patches, channels=2, frequency=3)

    assert result.patches.shape == (1, 4, 6)
    assert result.timings.shape == (1, 1, 4, 2)
    assert torch.equal(restored, audio)
    assert torch.allclose(result.timings[0, 0, :, 0], torch.tensor([0.0, 0.01, 0.05, 0.09]))
    assert torch.allclose(result.timings[0, 0, :, 1], torch.tensor([0.01, 0.05, 0.09, 0.13]))


def test_audio_patchify_matches_comfy_audio_patchifier():
    sys.path.insert(0, "/home/xingshen/ComfyUI")
    from comfy.ldm.lightricks.symmetric_patchifier import AudioPatchifier

    audio = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)
    local = patchify_audio(audio, start_end=True)
    comfy_patches, comfy_timings = AudioPatchifier(1, start_end=True).patchify(audio)

    assert torch.equal(local.patches, comfy_patches)
    assert torch.equal(local.timings, comfy_timings)


def test_audio_latent_time_in_seconds_supports_shift_and_noncausal():
    causal = audio_latent_time_in_seconds(1, 4)
    noncausal = audio_latent_time_in_seconds(1, 4, is_causal=False)

    assert torch.allclose(causal, torch.tensor([0.01, 0.05, 0.09]))
    assert torch.allclose(noncausal, torch.tensor([0.04, 0.08, 0.12]))

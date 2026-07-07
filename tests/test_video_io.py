from pathlib import Path
import shutil

import torch

from ltx_msr_torch.video_io import decoded_video_to_frames, frames_to_uint8_rgb, write_video_mp4


def test_decoded_video_to_frames_accepts_bcthw_and_maps_signed_range():
    video = torch.zeros(1, 3, 2, 4, 5)
    video[:, 0] = -1.0
    video[:, 1] = 0.0
    video[:, 2] = 1.0

    frames = decoded_video_to_frames(video)

    assert frames.shape == (2, 4, 5, 3)
    assert torch.allclose(frames[0, 0, 0], torch.tensor([0.0, 0.5, 1.0]))


def test_frames_to_uint8_rgb_expands_single_channel():
    frames = frames_to_uint8_rgb(torch.ones(2, 4, 5, 1) * 0.25)

    assert frames.shape == (2, 4, 5, 3)
    assert frames.dtype == torch.uint8
    assert int(frames[0, 0, 0, 0]) == 64


def test_write_video_mp4_writes_small_file(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        return
    frames = torch.zeros(2, 16, 16, 3)
    frames[1, :, :, 0] = 1.0

    output = write_video_mp4(frames, tmp_path / "small.mp4", fps=24.0)

    assert output.exists()
    assert output.stat().st_size > 0

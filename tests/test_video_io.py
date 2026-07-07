from pathlib import Path
import shutil
import wave

import torch

from ltx_msr_torch.video_io import (
    audio_to_int16_pcm,
    decoded_audio_to_samples,
    decoded_video_to_frames,
    frames_to_uint8_rgb,
    write_audio_wav,
    write_av_mp4,
    write_video_mp4,
)


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


def test_decoded_audio_to_samples_accepts_bcs_layout():
    audio = torch.zeros(1, 2, 4)
    audio[:, 0] = -2.0
    audio[:, 1] = 0.5

    samples = decoded_audio_to_samples(audio)
    pcm = audio_to_int16_pcm(audio)

    assert samples.shape == (4, 2)
    assert torch.allclose(samples[0], torch.tensor([-1.0, 0.5]))
    assert pcm.shape == (4, 2)
    assert pcm.dtype == torch.int16


def test_write_video_mp4_writes_small_file(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        return
    frames = torch.zeros(2, 16, 16, 3)
    frames[1, :, :, 0] = 1.0

    output = write_video_mp4(frames, tmp_path / "small.mp4", fps=24.0)

    assert output.exists()
    assert output.stat().st_size > 0


def test_write_audio_wav_writes_small_file(tmp_path: Path):
    audio = torch.zeros(1, 2, 16)
    audio[:, 0] = 0.25

    output = write_audio_wav(audio, tmp_path / "small.wav", sample_rate=48000)

    with wave.open(str(output), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getframerate() == 48000
        assert handle.getnframes() == 16


def test_write_av_mp4_writes_muxed_file(tmp_path: Path):
    if shutil.which("ffmpeg") is None:
        return
    frames = torch.zeros(2, 16, 16, 3)
    frames[1, :, :, 1] = 1.0
    audio = torch.zeros(1, 2, 2048)

    output = write_av_mp4(frames, audio, tmp_path / "muxed.mp4", fps=24.0, sample_rate=48000)

    assert output.exists()
    assert output.stat().st_size > 0

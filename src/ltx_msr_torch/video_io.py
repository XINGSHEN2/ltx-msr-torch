from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import torch


def decoded_video_to_frames(video: torch.Tensor) -> torch.Tensor:
    """Normalize decoded video tensors to [T,H,W,3] float32 in [0,1]."""
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"expected batch size 1 for decoded video, got {video.shape[0]}")
        if video.shape[1] in (1, 3):
            video = video[0].movedim(0, -1)
        elif video.shape[-1] in (1, 3):
            video = video[0]
        else:
            raise ValueError(f"cannot infer channel axis for decoded video shape {tuple(video.shape)}")
    elif video.ndim == 4:
        if video.shape[0] in (1, 3):
            video = video.movedim(0, -1)
        elif video.shape[-1] not in (1, 3):
            raise ValueError(f"cannot infer channel axis for decoded video shape {tuple(video.shape)}")
    else:
        raise ValueError(f"expected decoded video with 4 or 5 dims, got {tuple(video.shape)}")

    video = video.detach().to(dtype=torch.float32, device="cpu")
    if video.shape[-1] == 1:
        video = video.expand(*video.shape[:-1], 3)
    if float(video.min()) < 0.0:
        video = (video + 1.0) * 0.5
    return video.clamp(0.0, 1.0)


def frames_to_uint8_rgb(frames: torch.Tensor) -> torch.Tensor:
    frames = decoded_video_to_frames(frames)
    return (frames * 255.0).round().to(torch.uint8).contiguous()


def decoded_audio_to_samples(audio: torch.Tensor) -> torch.Tensor:
    """Normalize decoded audio tensors to [samples, channels] float32 in [-1,1]."""
    if audio.ndim == 3:
        if audio.shape[0] != 1:
            raise ValueError(f"expected batch size 1 for decoded audio, got {audio.shape[0]}")
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.ndim == 2:
        if audio.shape[0] in (1, 2) and audio.shape[1] > audio.shape[0]:
            audio = audio.movedim(0, 1)
    else:
        raise ValueError(f"expected decoded audio with 1, 2, or 3 dims, got {tuple(audio.shape)}")
    audio = audio.detach().to(dtype=torch.float32, device="cpu")
    return audio.clamp(-1.0, 1.0).contiguous()


def audio_to_int16_pcm(audio: torch.Tensor) -> torch.Tensor:
    samples = decoded_audio_to_samples(audio)
    return (samples * 32767.0).round().to(torch.int16).contiguous()


def write_audio_wav(
    audio: torch.Tensor,
    output: str | Path,
    *,
    sample_rate: int,
) -> Path:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    pcm = audio_to_int16_pcm(audio)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.numpy().tobytes())
    return output_path


def write_video_mp4(
    frames: torch.Tensor,
    output: str | Path,
    *,
    fps: float,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Write [T,H,W,3] or decoder-layout video frames to an mp4 with ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write mp4 output")
    if fps <= 0:
        raise ValueError("fps must be positive")

    rgb = frames_to_uint8_rgb(frames)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"expected RGB frames [T,H,W,3], got {tuple(rgb.shape)}")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count, height, width, _ = rgb.shape
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        str(output_path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(rgb.numpy().tobytes())
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {process.returncode}: {stderr.decode('utf-8', errors='replace')}"
        )
    if frame_count <= 0 or not output_path.exists():
        raise RuntimeError(f"failed to write video output: {output_path}")
    return output_path


def write_av_mp4(
    frames: torch.Tensor,
    audio: torch.Tensor,
    output: str | Path,
    *,
    fps: float,
    sample_rate: int,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Write decoded video and audio tensors to a muxed mp4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to mux mp4 output")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ltx_msr_mux_") as tmp:
        tmp_path = Path(tmp)
        video_path = write_video_mp4(
            frames,
            tmp_path / "video.mp4",
            fps=fps,
            crf=crf,
            preset=preset,
        )
        audio_path = write_audio_wav(audio, tmp_path / "audio.wav", sample_rate=sample_rate)
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mux failed with code {process.returncode}: "
            f"{process.stderr.decode('utf-8', errors='replace')}"
        )
    if not output_path.exists():
        raise RuntimeError(f"failed to write muxed video output: {output_path}")
    return output_path

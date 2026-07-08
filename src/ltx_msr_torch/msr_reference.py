from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence


def load_image_tensor(path: str | Path) -> torch.Tensor:
    """Load an image as a ComfyUI-style image tensor: [1, H, W, C], float32."""
    av_tensor = _load_image_tensor_with_av(path)
    if av_tensor is not None:
        return av_tensor
    return _load_image_tensor_with_pil(path)


def create_msr_reference_video(
    subjects: Iterable[torch.Tensor | np.ndarray | Image.Image | None],
    background: torch.Tensor | np.ndarray | Image.Image,
    width: int,
    height: int,
    frame_count: int,
) -> torch.Tensor:
    """Create the fixed-frame MSR reference video used by `LiconMSR`.

    The frame order matches the ComfyUI node:

    1 -> 2 -> 3 -> 4 -> background

    Disconnected subject inputs are skipped. Background is required and always
    appended last.
    """
    if background is None:
        raise ValueError("background is required")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")

    prepared: list[np.ndarray] = []
    for image in subjects:
        if image is not None:
            prepared.append(_prepare_image(image, width, height))
    prepared.append(_prepare_image(background, width, height))

    frames = _expand_frames(prepared, frame_count)
    array = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(array)


def create_msr_reference_video_from_paths(
    subjects: Iterable[str | Path | None],
    background: str | Path,
    width: int,
    height: int,
    frame_count: int,
) -> torch.Tensor:
    subject_tensors = [
        load_image_tensor(path) if path is not None else None
        for path in subjects
    ]
    return create_msr_reference_video(
        subject_tensors,
        load_image_tensor(background),
        width,
        height,
        frame_count,
    )


def _tensor_to_rgb_array(image: torch.Tensor | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    if isinstance(image, torch.Tensor):
        if image.ndim == 4:
            image = image[0]
        image = image.detach().cpu().numpy()

    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]

    return np.ascontiguousarray(array)


def _load_image_tensor_with_av(path: str | Path) -> torch.Tensor | None:
    """Match ComfyUI LoadImage's preferred PyAV decode path for still images."""
    try:
        import av
    except ImportError:
        return None

    try:
        with av.open(str(path), mode="r") as container:
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if video_stream is None:
                return None

            frames: list[torch.Tensor] = []
            image_format = "gbrpf32le"
            process_image_format = lambda value: value
            checked_format = False

            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    if not checked_format:
                        if frame.format.name in ("yuvj420p", "yuvj422p", "yuvj444p", "rgb24", "rgba", "pal8"):
                            image_format = "rgb24"
                            process_image_format = lambda value: value.float() / 255.0
                        checked_format = True

                    image = frame.to_ndarray(format=image_format)
                    rotation = getattr(frame, "rotation", 0)
                    if rotation:
                        image = np.rot90(image, k=int(round(rotation // 90)), axes=(0, 1)).copy()
                    frames.append(torch.from_numpy(np.ascontiguousarray(image[..., :3])))

            if not frames:
                return None
            return process_image_format(torch.stack(frames)).to(dtype=torch.float32)
    except Exception:
        return None


def _load_image_tensor_with_pil(path: str | Path) -> torch.Tensor:
    image = Image.open(path)
    frames: list[torch.Tensor] = []
    width = None
    height = None
    for frame in ImageSequence.Iterator(image):
        frame = ImageOps.exif_transpose(frame).convert("RGB")
        if width is None:
            width, height = frame.size
        if frame.size != (width, height):
            continue
        array = np.asarray(frame, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(array))
    if not frames:
        raise ValueError(f"failed to load image frames from {path}")
    return torch.stack(frames)


def _prepare_image(
    image: torch.Tensor | np.ndarray | Image.Image,
    width: int,
    height: int,
) -> np.ndarray:
    array = _tensor_to_rgb_array(image)
    if array.shape[1] == width and array.shape[0] == height:
        return np.ascontiguousarray(array)

    try:
        import cv2

        return cv2.resize(array, (width, height), interpolation=cv2.INTER_LANCZOS4)
    except ImportError:
        pil_image = Image.fromarray(array).convert("RGB")
        return np.asarray(
            pil_image.resize((width, height), Image.Resampling.LANCZOS)
        )


def _expand_frames(images: list[np.ndarray], frame_count: int) -> list[np.ndarray]:
    if not images:
        raise ValueError("at least one image is required")

    base_count = frame_count // len(images)
    remainder = frame_count % len(images)
    frames: list[np.ndarray] = []
    for index, image in enumerate(images):
        repeats = base_count + (1 if index < remainder else 0)
        frames.extend([image] * repeats)
    return frames

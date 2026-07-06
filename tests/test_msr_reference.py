import numpy as np
import torch

from ltx_msr_torch.msr_reference import create_msr_reference_video


def _image(value: float) -> torch.Tensor:
    return torch.full((1, 2, 2, 3), value)


def test_reference_frame_distribution_keeps_background_last():
    video = create_msr_reference_video(
        subjects=[_image(0.1), None, _image(0.3), None],
        background=_image(0.9),
        width=2,
        height=2,
        frame_count=7,
    )

    assert tuple(video.shape) == (7, 2, 2, 3)
    values = [float(frame[0, 0, 0]) for frame in video]
    assert np.allclose(values, [0.09803922, 0.09803922, 0.09803922, 0.29803923, 0.29803923, 0.8980392, 0.8980392])


def test_reference_requires_background():
    try:
        create_msr_reference_video([], None, width=2, height=2, frame_count=1)
    except ValueError as exc:
        assert "background" in str(exc)
    else:
        raise AssertionError("expected ValueError")


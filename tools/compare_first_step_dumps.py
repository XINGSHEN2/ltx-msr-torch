from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


KEY_MAP = {
    "packed_noise": ("packed_noise", "noise"),
    "packed_latent_image": ("packed_latent_image", "latent_image"),
    "packed_denoise_mask": ("packed_denoise_mask", "denoise_mask"),
    "packed_start": ("x",),
    "packed_denoised": ("denoised",),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("torch_dump")
    parser.add_argument("comfy_dump")
    args = parser.parse_args()

    torch_values = _load_values(Path(args.torch_dump))
    comfy_values = _load_values(Path(args.comfy_dump))
    for torch_key, comfy_keys in KEY_MAP.items():
        _print_compare(torch_key, torch_values.get(torch_key), _first_present(comfy_values, comfy_keys))
    return 0


def _load_values(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["values"]


def _print_compare(label: str, left: Any, right: Any) -> None:
    if left is None and right is None:
        print(f"{label}: both None")
        return
    if left is None or right is None:
        print(f"{label}: one side None left={left is None} right={right is None}")
        return
    if isinstance(left, list) or isinstance(right, list):
        print(f"{label}: nested/list left={_shape(left)} right={_shape(right)}")
        return
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        print(f"{label}: non tensor left={type(left).__name__} right={type(right).__name__}")
        return
    l = left.float()
    r = right.float()
    print(f"{label}: left_shape={tuple(left.shape)} right_shape={tuple(right.shape)}")
    if tuple(left.shape) != tuple(right.shape):
        return
    diff = (l - r).abs()
    denom = r.abs().mean().clamp_min(1e-8)
    print(
        f"{label}: mean_abs={float(diff.mean().item()):.9g} "
        f"max_abs={float(diff.max().item()):.9g} "
        f"rel_mean={float((diff.mean() / denom).item()):.9g} "
        f"left_mean={float(l.mean().item()):.9g} right_mean={float(r.mean().item()):.9g} "
        f"left_std={float(l.std().item()):.9g} right_std={float(r.std().item()):.9g}"
    )


def _first_present(values: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


if __name__ == "__main__":
    raise SystemExit(main())

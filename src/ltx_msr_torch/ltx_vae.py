from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .checkpoint_loader import load_safetensors_subset, strip_prefix_from_state_dict
from .vae import AudioVAE, VideoVAE


@dataclass(frozen=True)
class LTXVAELoadReport:
    loaded: int
    missing: tuple[str, ...]


@dataclass(frozen=True)
class LTXAVDecoders:
    video_vae: torch.nn.Module
    audio_vae: torch.nn.Module


def load_checkpoint_config(checkpoint_path: str | Path) -> dict[str, object]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata is None or "config" not in metadata:
        raise ValueError("checkpoint metadata does not contain config")
    raw_config = metadata["config"]
    return json.loads(raw_config) if isinstance(raw_config, str) else raw_config


def build_ltx_video_vae_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    config = load_checkpoint_config(checkpoint_path)["vae"]
    model = VideoVAE(config=config)
    if dtype is not None or device is not None:
        model = model.to(device=device, dtype=dtype)
    return model


def build_ltx_audio_vae_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.nn.Module:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    model = AudioVAE(metadata)
    if dtype is not None or device is not None:
        model = model.to(device=device, dtype=dtype)
    return model


def load_ltx_video_vae_state_dict(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = tuple(key for key in handle.keys() if key.startswith("vae."))
    raw = load_safetensors_subset(checkpoint_path, keys, device=device)
    return strip_prefix_from_state_dict(raw, "vae")


def load_ltx_audio_vae_state_dict(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = tuple(
            key
            for key in handle.keys()
            if key.startswith("audio_vae.") or key.startswith("vocoder.")
        )
    raw = load_safetensors_subset(checkpoint_path, keys, device=device)
    output: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if key.startswith("audio_vae."):
            output[f"autoencoder.{key[len('audio_vae.'):]}"] = value
        elif key.startswith("vocoder."):
            output[key] = value
    return output


def missing_ltx_vae_keys(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> tuple[str, ...]:
    available = set(state_dict)
    return tuple(key for key in model.state_dict() if key not in available)


def load_ltx_video_vae_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    state_dict = load_ltx_video_vae_state_dict(checkpoint_path, device=device)
    return model.load_state_dict(state_dict, strict=strict)


def load_ltx_audio_vae_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
    device: str | torch.device = "cpu",
) -> torch.nn.modules.module._IncompatibleKeys:
    state_dict = load_ltx_audio_vae_state_dict(checkpoint_path, device=device)
    return model.load_state_dict(state_dict, strict=strict)


def load_ltxav_decoders_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = "cpu",
    strict: bool = True,
) -> LTXAVDecoders:
    video_vae = build_ltx_video_vae_from_checkpoint(checkpoint_path, dtype=dtype, device=device)
    audio_vae = build_ltx_audio_vae_from_checkpoint(checkpoint_path, dtype=dtype, device=device)
    load_ltx_video_vae_weights(video_vae, checkpoint_path, strict=strict, device=device or "cpu")
    load_ltx_audio_vae_weights(audio_vae, checkpoint_path, strict=strict, device=device or "cpu")
    return LTXAVDecoders(video_vae=video_vae, audio_vae=audio_vae)


@torch.no_grad()
def decode_ltx_video_latents(video_vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    return video_vae.decode(latents)


@torch.no_grad()
def encode_ltx_video_pixels(video_vae: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"expected pixels [T,H,W,3], got {tuple(pixels.shape)}")
    reference = next(video_vae.parameters(), None) if isinstance(video_vae, torch.nn.Module) else None
    device = reference.device if reference is not None else pixels.device
    dtype = reference.dtype if reference is not None else pixels.dtype
    model_input = pixels.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
    model_input = (model_input * 2.0 - 1.0).to(device=device, dtype=dtype)
    return video_vae.encode(model_input)


@torch.no_grad()
def decode_ltx_audio_latents(audio_vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    return audio_vae.decode(latents)

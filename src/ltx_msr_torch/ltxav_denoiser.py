from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch

from .ltx_patchify import patchify_audio, symmetric_patchify_video
from .sampler import append_dims
from .sampler import sample_euler
from .sampler import sample_euler_latents


class LTXAVModelProtocol(Protocol):
    def __call__(
        self,
        *,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        frame_rate: float,
        attention_mask: torch.Tensor | None = None,
        transformer_options: dict[str, object] | None = None,
        self_attention_mask: torch.Tensor | None = None,
        ref_audio_seq_len: int = 0,
        target_audio_seq_len: int | None = None,
        keyframe_idxs: torch.Tensor | None = None,
        grid_mask: torch.Tensor | None = None,
        orig_patchified_shape: tuple[int, ...] | list[int] | None = None,
        output_orig_shape: tuple[int, ...] | list[int] | None = None,
        denoise_mask: torch.Tensor | None = None,
        guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        ...


@dataclass(frozen=True)
class LTXAVDenoiser:
    model: LTXAVModelProtocol
    context: torch.Tensor
    attention_mask: torch.Tensor | None
    frame_rate: float
    transformer_options: dict[str, object] | None = None
    self_attention_mask: torch.Tensor | None = None
    ref_audio_seq_len: int = 0
    keyframe_idxs: torch.Tensor | None = None
    grid_mask: torch.Tensor | None = None
    orig_patchified_shape: tuple[int, ...] | list[int] | None = None
    output_orig_shape: tuple[int, ...] | list[int] | None = None
    denoise_mask: torch.Tensor | None = None
    audio_denoise_mask: torch.Tensor | None = None
    video_latent_image: torch.Tensor | None = None
    sample_sigmas: torch.Tensor | None = None
    guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None

    def __call__(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_latents, audio_latents = latents
        batch_size = video_latents.shape[0]
        if sigma.ndim == 0:
            sigma = sigma.expand(batch_size)
        expanded_denoise_mask = None
        video_model_latents = video_latents
        if self.denoise_mask is not None:
            expanded_denoise_mask = _expand_denoise_mask(self.denoise_mask, video_latents).to(
                device=video_latents.device,
                dtype=video_latents.dtype,
            )
            if self.video_latent_image is not None:
                latent_image = self.video_latent_image.to(device=video_latents.device, dtype=video_latents.dtype)
                video_model_latents = video_latents * expanded_denoise_mask + latent_image * (1.0 - expanded_denoise_mask)
            timestep_mask = expanded_denoise_mask[:, :1] * sigma.reshape(batch_size, 1, 1, 1, 1).to(
                device=video_latents.device,
                dtype=expanded_denoise_mask.dtype,
            )
            video_timestep = symmetric_patchify_video(timestep_mask, patch_size=1, start_end=True).patches[..., 0]
        else:
            video_timestep = sigma.reshape(batch_size, 1).expand(batch_size, _video_token_count(video_latents))
        if self.audio_denoise_mask is not None:
            audio_timestep = _audio_timestep_from_mask(self.audio_denoise_mask, audio_latents, sigma)
        elif self.denoise_mask is not None:
            audio_timestep = _audio_timestep_from_mask(torch.ones_like(audio_latents), audio_latents, sigma)
        else:
            audio_timestep = sigma.reshape(batch_size, 1).expand(batch_size, _audio_token_count(audio_latents))
        model_dtype = self.context.dtype if torch.is_floating_point(self.context) else video_latents.dtype
        transformer_options = _step_transformer_options(
            self.transformer_options,
            sigma=sigma,
            sample_sigmas=self.sample_sigmas,
        )
        output = self.model(
            video_latents=video_model_latents.to(dtype=model_dtype),
            audio_latents=audio_latents.to(dtype=model_dtype),
            context=self.context,
            timestep=video_timestep,
            audio_timestep=audio_timestep,
            frame_rate=self.frame_rate,
            attention_mask=self.attention_mask,
            transformer_options=transformer_options,
            self_attention_mask=self.self_attention_mask,
            ref_audio_seq_len=self.ref_audio_seq_len,
            target_audio_seq_len=audio_timestep.shape[1],
            keyframe_idxs=self.keyframe_idxs,
            grid_mask=self.grid_mask,
            orig_patchified_shape=self.orig_patchified_shape,
            output_orig_shape=self.output_orig_shape,
            denoise_mask=self.denoise_mask,
            guide_attention_entries=self.guide_attention_entries,
        )
        if not isinstance(output, list) or len(output) != 2:
            raise TypeError("LTXAV denoiser expects model output [video, audio]")
        video_denoised = video_model_latents - output[0].to(dtype=torch.float32) * append_dims(
            sigma.to(video_latents.device), video_latents.ndim
        )
        if expanded_denoise_mask is not None and self.video_latent_image is not None:
            latent_image = self.video_latent_image.to(device=video_latents.device, dtype=video_denoised.dtype)
            mask = expanded_denoise_mask.to(dtype=video_denoised.dtype)
            video_denoised = video_denoised * mask + latent_image * (1.0 - mask)
        audio_denoised = audio_latents - output[1].to(dtype=torch.float32) * append_dims(
            sigma.to(audio_latents.device), audio_latents.ndim
        )
        return video_denoised, audio_denoised


@dataclass(frozen=True)
class PackedLatents:
    tensor: torch.Tensor
    shapes: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class LTXAVConditioning:
    context: torch.Tensor
    attention_mask: torch.Tensor | None
    transformer_options: dict[str, object] | None = None


@dataclass(frozen=True)
class ComfyLTXAVApplyModel:
    model: LTXAVModelProtocol
    latent_shapes: tuple[tuple[int, ...], tuple[int, ...]]
    frame_rate: float
    self_attention_mask: torch.Tensor | None = None
    ref_audio_seq_len: int = 0
    keyframe_idxs: torch.Tensor | None = None
    grid_mask: torch.Tensor | None = None
    orig_patchified_shape: tuple[int, ...] | list[int] | None = None
    output_orig_shape: tuple[int, ...] | list[int] | None = None
    video_denoise_mask: torch.Tensor | None = None
    audio_denoise_mask: torch.Tensor | None = None
    guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None

    def apply_model(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        *,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        transformer_options: dict[str, object] | None = None,
    ) -> torch.Tensor:
        video_latents, audio_latents = unpack_latents(x, self.latent_shapes)
        batch_size = video_latents.shape[0]
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        video_timestep = _video_timestep_from_mask(self.video_denoise_mask, video_latents, timestep)
        audio_timestep = _audio_timestep_from_mask(
            self.audio_denoise_mask if self.audio_denoise_mask is not None else torch.ones_like(audio_latents),
            audio_latents,
            timestep,
        )
        model_dtype = context.dtype if torch.is_floating_point(context) else video_latents.dtype
        output = self.model(
            video_latents=video_latents.to(dtype=model_dtype),
            audio_latents=audio_latents.to(dtype=model_dtype),
            context=context,
            timestep=video_timestep,
            audio_timestep=audio_timestep,
            frame_rate=self.frame_rate,
            attention_mask=attention_mask,
            transformer_options=transformer_options,
            self_attention_mask=self.self_attention_mask,
            ref_audio_seq_len=self.ref_audio_seq_len,
            target_audio_seq_len=audio_timestep.shape[1],
            keyframe_idxs=self.keyframe_idxs,
            grid_mask=self.grid_mask,
            orig_patchified_shape=self.orig_patchified_shape,
            output_orig_shape=self.output_orig_shape,
            denoise_mask=self.video_denoise_mask,
            guide_attention_entries=self.guide_attention_entries,
        )
        if not isinstance(output, list) or len(output) != 2:
            raise TypeError("LTXAV apply_model expects model output [video, audio]")
        video_denoised = video_latents - output[0].to(dtype=torch.float32) * append_dims(
            timestep.to(video_latents.device), video_latents.ndim
        )
        audio_denoised = audio_latents - output[1].to(dtype=torch.float32) * append_dims(
            timestep.to(audio_latents.device), audio_latents.ndim
        )
        return pack_latents((video_denoised, audio_denoised)).tensor


@dataclass(frozen=True)
class ComfyCFGGuider:
    inner_model: ComfyLTXAVApplyModel
    positive: list[LTXAVConditioning]
    negative: list[LTXAVConditioning] | None
    cfg: float
    model_options: dict[str, object]

    def __call__(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        *,
        model_options: dict[str, object] | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        options = _merge_model_options(self.model_options, model_options)
        return sampling_function(
            self.inner_model,
            x,
            timestep,
            self.negative,
            self.positive,
            self.cfg,
            model_options=options,
            seed=seed,
        )


@dataclass(frozen=True)
class ComfyKSamplerX0Inpaint:
    inner_model: ComfyCFGGuider
    sigmas: torch.Tensor
    noise: torch.Tensor
    latent_image: torch.Tensor
    denoise_mask: torch.Tensor | None

    def __call__(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if self.denoise_mask is not None:
            denoise_mask = self.denoise_mask.to(device=x.device, dtype=x.dtype)
            latent_mask = 1.0 - denoise_mask
            latent_image = self.latent_image.to(device=x.device, dtype=x.dtype)
            x = x * denoise_mask + latent_image * latent_mask
        out = self.inner_model(x, sigma, model_options={"sigmas": sigma}, seed=None)
        if self.denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out


@dataclass(frozen=True)
class LTXAVPackedDenoiser:
    denoiser: LTXAVDenoiser
    latent_shapes: tuple[tuple[int, ...], ...]

    def __call__(self, latents: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        video_latents, audio_latents = unpack_latents(latents, self.latent_shapes)
        video_denoised, audio_denoised = self.denoiser((video_latents, audio_latents), sigma)
        return pack_latents((video_denoised, audio_denoised)).tensor


def pack_latents(latents: tuple[torch.Tensor, ...]) -> PackedLatents:
    if not latents:
        raise ValueError("at least one latent tensor is required")
    batch_size = latents[0].shape[0]
    shapes: list[tuple[int, ...]] = []
    flat: list[torch.Tensor] = []
    for latent in latents:
        if latent.shape[0] != batch_size:
            raise ValueError("all latent tensors must use the same batch size")
        shapes.append(tuple(int(dim) for dim in latent.shape))
        flat.append(latent.reshape(batch_size, 1, -1))
    return PackedLatents(tensor=torch.cat(flat, dim=-1), shapes=tuple(shapes))


def unpack_latents(latents: torch.Tensor, shapes: tuple[tuple[int, ...], ...]) -> tuple[torch.Tensor, ...]:
    if latents.ndim != 3 or latents.shape[1] != 1:
        raise ValueError(f"expected packed latents [B,1,N], got {tuple(latents.shape)}")
    outputs: list[torch.Tensor] = []
    offset = 0
    for shape in shapes:
        count = 1
        for dim in shape[1:]:
            count *= int(dim)
        chunk = latents[:, :, offset : offset + count]
        if chunk.shape[-1] != count:
            raise ValueError("packed latent tensor is shorter than the provided shapes")
        outputs.append(chunk.reshape(shape))
        offset += count
    if offset != latents.shape[-1]:
        raise ValueError("packed latent tensor has trailing values")
    return tuple(outputs)


def calc_cond_batch(
    model: ComfyLTXAVApplyModel,
    conds: list[list[LTXAVConditioning] | None],
    x_in: torch.Tensor,
    timestep: torch.Tensor,
    model_options: dict[str, object],
) -> list[torch.Tensor]:
    out_conds = []
    out_counts = []
    for cond_index, cond in enumerate(conds):
        out = torch.zeros_like(x_in)
        count = torch.ones_like(x_in) * 1e-37
        if cond is not None:
            for item in cond:
                transformer_options = _step_transformer_options(
                    _merge_transformer_options(model_options.get("transformer_options"), item.transformer_options),
                    sigma=timestep,
                    sample_sigmas=model_options.get("sample_sigmas") if isinstance(model_options.get("sample_sigmas"), torch.Tensor) else None,
                    cond_or_uncond=[cond_index],
                )
                pred = model.apply_model(
                    x_in,
                    timestep,
                    context=item.context,
                    attention_mask=item.attention_mask,
                    transformer_options=transformer_options,
                )
                out += pred
                count += torch.ones_like(pred)
        out_conds.append(out / count)
        out_counts.append(count)
    return out_conds


def cfg_function(
    cond_pred: torch.Tensor,
    uncond_pred: torch.Tensor,
    cond_scale: float,
) -> torch.Tensor:
    return uncond_pred + (cond_pred - uncond_pred) * cond_scale


def sampling_function(
    model: ComfyLTXAVApplyModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    uncond: list[LTXAVConditioning] | None,
    cond: list[LTXAVConditioning],
    cond_scale: float,
    *,
    model_options: dict[str, object],
    seed: int | None = None,
) -> torch.Tensor:
    if math.isclose(cond_scale, 1.0) and not bool(model_options.get("disable_cfg1_optimization", False)):
        uncond_ = None
    else:
        uncond_ = uncond
    out = calc_cond_batch(model, [cond, uncond_], x, timestep, model_options)
    return cfg_function(out[0], out[1], cond_scale)


def _video_token_count(video_latents: torch.Tensor) -> int:
    if video_latents.ndim != 5:
        raise ValueError(f"expected video latents [B,C,T,H,W], got {tuple(video_latents.shape)}")
    return int(video_latents.shape[2] * video_latents.shape[3] * video_latents.shape[4])


def _audio_token_count(audio_latents: torch.Tensor) -> int:
    if audio_latents.ndim != 4:
        raise ValueError(f"expected audio latents [B,C,T,F], got {tuple(audio_latents.shape)}")
    return int(audio_latents.shape[2])


def _expand_denoise_mask(denoise_mask: torch.Tensor, video_latents: torch.Tensor) -> torch.Tensor:
    if denoise_mask.shape[3] == video_latents.shape[3] and denoise_mask.shape[4] == video_latents.shape[4]:
        return denoise_mask
    return denoise_mask.expand(-1, -1, -1, video_latents.shape[3], video_latents.shape[4])


def _video_timestep_from_mask(
    denoise_mask: torch.Tensor | None,
    video_latents: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    batch_size = video_latents.shape[0]
    if denoise_mask is None:
        return sigma.reshape(batch_size, 1).expand(batch_size, _video_token_count(video_latents))
    expanded = _expand_denoise_mask(denoise_mask, video_latents).to(
        device=video_latents.device,
        dtype=video_latents.dtype,
    )
    timestep_mask = expanded[:, :1] * sigma.reshape(batch_size, 1, 1, 1, 1).to(
        device=video_latents.device,
        dtype=expanded.dtype,
    )
    return symmetric_patchify_video(timestep_mask, patch_size=1, start_end=True).patches[..., 0]


@torch.no_grad()
def sample_ltxav_euler(
    *,
    model: LTXAVModelProtocol,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    context: torch.Tensor,
    attention_mask: torch.Tensor | None,
    frame_rate: float,
    sigmas: torch.Tensor,
    transformer_options: dict[str, object] | None = None,
    self_attention_mask: torch.Tensor | None = None,
    ref_audio_seq_len: int = 0,
    keyframe_idxs: torch.Tensor | None = None,
    grid_mask: torch.Tensor | None = None,
    orig_patchified_shape: tuple[int, ...] | list[int] | None = None,
    output_orig_shape: tuple[int, ...] | list[int] | None = None,
    denoise_mask: torch.Tensor | None = None,
    audio_denoise_mask: torch.Tensor | None = None,
    video_latent_image: torch.Tensor | None = None,
    guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_latents = video_latents.to(dtype=torch.float32)
    audio_latents = audio_latents.to(dtype=torch.float32)
    sigmas = sigmas.to(device=video_latents.device, dtype=torch.float32)
    denoiser = LTXAVDenoiser(
        model=model,
        context=context,
        attention_mask=attention_mask,
        frame_rate=frame_rate,
        transformer_options=transformer_options,
        self_attention_mask=self_attention_mask,
        ref_audio_seq_len=ref_audio_seq_len,
        keyframe_idxs=keyframe_idxs,
        grid_mask=grid_mask,
        orig_patchified_shape=orig_patchified_shape,
        output_orig_shape=output_orig_shape,
        denoise_mask=denoise_mask,
        audio_denoise_mask=audio_denoise_mask,
        video_latent_image=video_latent_image,
        sample_sigmas=sigmas,
        guide_attention_entries=guide_attention_entries,
    )
    sampled = sample_euler_latents(denoiser, (video_latents, audio_latents), sigmas)
    if not isinstance(sampled, tuple):
        raise TypeError("sample_ltxav_euler expected tuple latents")
    return sampled


@torch.no_grad()
def sample_ltxav_euler_comfy(
    *,
    model: LTXAVModelProtocol,
    video_noise: torch.Tensor,
    audio_noise: torch.Tensor,
    video_latent_image: torch.Tensor,
    audio_latent_image: torch.Tensor,
    context: torch.Tensor,
    attention_mask: torch.Tensor | None,
    frame_rate: float,
    sigmas: torch.Tensor,
    transformer_options: dict[str, object] | None = None,
    self_attention_mask: torch.Tensor | None = None,
    ref_audio_seq_len: int = 0,
    keyframe_idxs: torch.Tensor | None = None,
    grid_mask: torch.Tensor | None = None,
    orig_patchified_shape: tuple[int, ...] | list[int] | None = None,
    output_orig_shape: tuple[int, ...] | list[int] | None = None,
    denoise_mask: torch.Tensor | None = None,
    audio_denoise_mask: torch.Tensor | None = None,
    guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sigmas.numel() == 0:
        return video_latent_image, audio_latent_image
    video_noise = video_noise.to(dtype=torch.float32)
    audio_noise = audio_noise.to(dtype=torch.float32)
    video_latent_image = video_latent_image.to(device=video_noise.device, dtype=torch.float32)
    audio_latent_image = audio_latent_image.to(device=audio_noise.device, dtype=torch.float32)
    sigmas = sigmas.to(device=video_noise.device, dtype=torch.float32)
    first_sigma = sigmas[0]
    video_latents = _const_noise_scaling(first_sigma, video_noise, video_latent_image)
    audio_latents = _const_noise_scaling(first_sigma.to(audio_noise.device), audio_noise, audio_latent_image)
    return sample_ltxav_euler(
        model=model,
        video_latents=video_latents,
        audio_latents=audio_latents,
        context=context,
        attention_mask=attention_mask,
        frame_rate=frame_rate,
        sigmas=sigmas,
        transformer_options=transformer_options,
        self_attention_mask=self_attention_mask,
        ref_audio_seq_len=ref_audio_seq_len,
        keyframe_idxs=keyframe_idxs,
        grid_mask=grid_mask,
        orig_patchified_shape=orig_patchified_shape,
        output_orig_shape=output_orig_shape,
        denoise_mask=denoise_mask,
        audio_denoise_mask=audio_denoise_mask,
        video_latent_image=video_latent_image,
        guide_attention_entries=guide_attention_entries,
    )


@torch.no_grad()
def sample_ltxav_euler_comfy_packed(
    *,
    model: LTXAVModelProtocol,
    video_noise: torch.Tensor,
    audio_noise: torch.Tensor,
    video_latent_image: torch.Tensor,
    audio_latent_image: torch.Tensor,
    context: torch.Tensor,
    attention_mask: torch.Tensor | None,
    frame_rate: float,
    sigmas: torch.Tensor,
    negative_context: torch.Tensor | None = None,
    negative_attention_mask: torch.Tensor | None = None,
    cfg: float = 1.0,
    transformer_options: dict[str, object] | None = None,
    self_attention_mask: torch.Tensor | None = None,
    ref_audio_seq_len: int = 0,
    keyframe_idxs: torch.Tensor | None = None,
    grid_mask: torch.Tensor | None = None,
    orig_patchified_shape: tuple[int, ...] | list[int] | None = None,
    output_orig_shape: tuple[int, ...] | list[int] | None = None,
    denoise_mask: torch.Tensor | None = None,
    audio_denoise_mask: torch.Tensor | None = None,
    guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sigmas.numel() == 0:
        return video_latent_image, audio_latent_image
    video_noise = video_noise.to(dtype=torch.float32)
    audio_noise = audio_noise.to(dtype=torch.float32)
    video_latent_image = video_latent_image.to(device=video_noise.device, dtype=torch.float32)
    audio_latent_image = audio_latent_image.to(device=audio_noise.device, dtype=torch.float32)
    sigmas = sigmas.to(device=video_noise.device, dtype=torch.float32)
    first_sigma = sigmas[0]
    video_latents = _const_noise_scaling(first_sigma, video_noise, video_latent_image)
    audio_latents = _const_noise_scaling(first_sigma.to(audio_noise.device), audio_noise, audio_latent_image)
    packed_noise = pack_latents((video_noise, audio_noise))
    packed_latent_image = pack_latents((video_latent_image, audio_latent_image))
    packed_mask = _pack_denoise_masks(
        video_mask=denoise_mask,
        audio_mask=audio_denoise_mask,
        latent_shapes=packed_latent_image.shapes,
        device=video_noise.device,
    )
    positive = LTXAVConditioning(
        context=context,
        attention_mask=attention_mask,
        transformer_options=transformer_options,
    )
    negative = None
    if negative_context is not None:
        negative = [
            LTXAVConditioning(
                context=negative_context,
                attention_mask=negative_attention_mask,
                transformer_options=transformer_options,
            )
        ]
    apply_model = ComfyLTXAVApplyModel(
        model=model,
        latent_shapes=(packed_latent_image.shapes[0], packed_latent_image.shapes[1]),
        frame_rate=frame_rate,
        self_attention_mask=self_attention_mask,
        ref_audio_seq_len=ref_audio_seq_len,
        keyframe_idxs=keyframe_idxs,
        grid_mask=grid_mask,
        orig_patchified_shape=orig_patchified_shape,
        output_orig_shape=output_orig_shape,
        video_denoise_mask=denoise_mask,
        audio_denoise_mask=audio_denoise_mask,
        guide_attention_entries=guide_attention_entries,
    )
    guider = ComfyCFGGuider(
        inner_model=apply_model,
        positive=[positive],
        negative=negative,
        cfg=cfg,
        model_options={"transformer_options": transformer_options or {}, "sample_sigmas": sigmas},
    )
    inpaint = ComfyKSamplerX0Inpaint(
        inner_model=guider,
        sigmas=sigmas,
        noise=packed_noise.tensor,
        latent_image=packed_latent_image.tensor,
        denoise_mask=packed_mask,
    )
    start = _const_noise_scaling(first_sigma, packed_noise.tensor, packed_latent_image.tensor)
    sampled = sample_euler(inpaint, start, sigmas)
    video_sampled, audio_sampled = unpack_latents(sampled, packed_latent_image.shapes)
    return video_sampled, audio_sampled


@torch.no_grad()
def sample_ltxav_euler_comfy_legacy_packed(
    *,
    model: LTXAVModelProtocol,
    video_noise: torch.Tensor,
    audio_noise: torch.Tensor,
    video_latent_image: torch.Tensor,
    audio_latent_image: torch.Tensor,
    context: torch.Tensor,
    attention_mask: torch.Tensor | None,
    frame_rate: float,
    sigmas: torch.Tensor,
    transformer_options: dict[str, object] | None = None,
    self_attention_mask: torch.Tensor | None = None,
    ref_audio_seq_len: int = 0,
    keyframe_idxs: torch.Tensor | None = None,
    grid_mask: torch.Tensor | None = None,
    orig_patchified_shape: tuple[int, ...] | list[int] | None = None,
    output_orig_shape: tuple[int, ...] | list[int] | None = None,
    denoise_mask: torch.Tensor | None = None,
    audio_denoise_mask: torch.Tensor | None = None,
    guide_attention_entries: tuple[dict[str, object], ...] | list[dict[str, object]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sigmas.numel() == 0:
        return video_latent_image, audio_latent_image
    video_noise = video_noise.to(dtype=torch.float32)
    audio_noise = audio_noise.to(dtype=torch.float32)
    video_latent_image = video_latent_image.to(device=video_noise.device, dtype=torch.float32)
    audio_latent_image = audio_latent_image.to(device=audio_noise.device, dtype=torch.float32)
    sigmas = sigmas.to(device=video_noise.device, dtype=torch.float32)
    first_sigma = sigmas[0]
    video_latents = _const_noise_scaling(first_sigma, video_noise, video_latent_image)
    audio_latents = _const_noise_scaling(first_sigma.to(audio_noise.device), audio_noise, audio_latent_image)
    packed = pack_latents((video_latents, audio_latents))
    denoiser = LTXAVDenoiser(
        model=model,
        context=context,
        attention_mask=attention_mask,
        frame_rate=frame_rate,
        transformer_options=transformer_options,
        self_attention_mask=self_attention_mask,
        ref_audio_seq_len=ref_audio_seq_len,
        keyframe_idxs=keyframe_idxs,
        grid_mask=grid_mask,
        orig_patchified_shape=orig_patchified_shape,
        output_orig_shape=output_orig_shape,
        denoise_mask=denoise_mask,
        audio_denoise_mask=audio_denoise_mask,
        video_latent_image=video_latent_image,
        sample_sigmas=sigmas,
        guide_attention_entries=guide_attention_entries,
    )
    packed_denoiser = LTXAVPackedDenoiser(denoiser=denoiser, latent_shapes=packed.shapes)
    sampled = sample_euler(packed_denoiser, packed.tensor, sigmas)
    video_sampled, audio_sampled = unpack_latents(sampled, packed.shapes)
    return video_sampled, audio_sampled


def _const_noise_scaling(sigma: torch.Tensor, noise: torch.Tensor, latent_image: torch.Tensor) -> torch.Tensor:
    sigma = sigma.reshape([1] + [1] * (noise.ndim - 1)).to(device=noise.device, dtype=noise.dtype)
    return sigma * noise + (1.0 - sigma) * latent_image


def _step_transformer_options(
    transformer_options: dict[str, object] | None,
    *,
    sigma: torch.Tensor,
    sample_sigmas: torch.Tensor | None,
    cond_or_uncond: list[int] | None = None,
) -> dict[str, object] | None:
    options = dict(transformer_options or {})
    # ComfyUI's CFGGuider/calc_cond_batch marks the positive batch as 0 and
    # passes both current and full sigma schedules through transformer_options.
    options["cond_or_uncond"] = [0] if cond_or_uncond is None else cond_or_uncond
    options["sigmas"] = sigma
    if sample_sigmas is not None:
        options["sample_sigmas"] = sample_sigmas
    return options


def _merge_model_options(
    base: dict[str, object] | None,
    override: dict[str, object] | None,
) -> dict[str, object]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if key == "transformer_options":
            merged[key] = _merge_transformer_options(merged.get(key), value)
        else:
            merged[key] = value
    return merged


def _merge_transformer_options(
    base: object,
    override: object,
) -> dict[str, object]:
    merged = dict(base) if isinstance(base, dict) else {}
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _pack_denoise_masks(
    *,
    video_mask: torch.Tensor | None,
    audio_mask: torch.Tensor | None,
    latent_shapes: tuple[tuple[int, ...], ...],
    device: torch.device,
) -> torch.Tensor | None:
    if video_mask is None and audio_mask is None:
        return None
    masks: list[torch.Tensor] = []
    if video_mask is None:
        masks.append(torch.ones(latent_shapes[0], device=device, dtype=torch.float32))
    else:
        masks.append(video_mask.to(device=device, dtype=torch.float32).expand(latent_shapes[0]))
    if audio_mask is None:
        masks.append(torch.ones(latent_shapes[1], device=device, dtype=torch.float32))
    else:
        masks.append(audio_mask.to(device=device, dtype=torch.float32).expand(latent_shapes[1]))
    return pack_latents(tuple(masks)).tensor


def _audio_timestep_from_mask(
    audio_denoise_mask: torch.Tensor,
    audio_latents: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    batch_size = audio_latents.shape[0]
    mask = audio_denoise_mask.to(device=audio_latents.device, dtype=audio_latents.dtype)
    if mask.shape != audio_latents.shape:
        if mask.shape[0] != batch_size or mask.shape[2] != audio_latents.shape[2]:
            raise ValueError(
                f"audio_denoise_mask shape {tuple(mask.shape)} is incompatible with audio latents {tuple(audio_latents.shape)}"
            )
        mask = mask.expand_as(audio_latents)
    timestep_mask = mask[:, :1, :, :1] * sigma.reshape(batch_size, 1, 1, 1).to(
        device=audio_latents.device,
        dtype=mask.dtype,
    )
    return patchify_audio(timestep_mask, start_end=True).patches[..., 0]

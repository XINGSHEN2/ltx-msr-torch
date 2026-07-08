from __future__ import annotations

from pathlib import Path

from .ltxav_denoiser import (
    ComfyCFGGuider,
    ComfyKSamplerX0Inpaint,
    ComfyLTXAVApplyModel,
    LTXAVConditioning,
    _const_noise_scaling,
    pack_latents,
    unpack_latents,
)


def debug_msr_first_step(
    *,
    sampler_kwargs: dict[str, object],
    negative_context,
    negative_attention_mask,
    cfg: float,
    dump_path: str | Path | None = None,
) -> None:
    import torch

    model = sampler_kwargs["model"]
    video_noise = sampler_kwargs["video_noise"]
    audio_noise = sampler_kwargs["audio_noise"]
    video_latent_image = sampler_kwargs["video_latent_image"]
    audio_latent_image = sampler_kwargs["audio_latent_image"]
    sigmas = sampler_kwargs["sigmas"]
    if not isinstance(video_noise, torch.Tensor) or not isinstance(audio_noise, torch.Tensor):
        raise TypeError("debug expects tensor noise")
    if not isinstance(video_latent_image, torch.Tensor) or not isinstance(audio_latent_image, torch.Tensor):
        raise TypeError("debug expects tensor latent images")
    if not isinstance(sigmas, torch.Tensor):
        raise TypeError("debug expects tensor sigmas")

    first_sigma = sigmas[0].to(device=video_noise.device, dtype=torch.float32)
    packed_noise = pack_latents((video_noise.to(dtype=torch.float32), audio_noise.to(dtype=torch.float32)))
    packed_latent_image = pack_latents(
        (
            video_latent_image.to(device=video_noise.device, dtype=torch.float32),
            audio_latent_image.to(device=audio_noise.device, dtype=torch.float32),
        )
    )
    start = _const_noise_scaling(first_sigma, packed_noise.tensor, packed_latent_image.tensor)
    positive = LTXAVConditioning(
        context=sampler_kwargs["context"],
        attention_mask=sampler_kwargs["attention_mask"],
        transformer_options=sampler_kwargs["transformer_options"],
    )
    negative = [
        LTXAVConditioning(
            context=negative_context,
            attention_mask=negative_attention_mask,
            transformer_options=sampler_kwargs["transformer_options"],
        )
    ]
    apply_model = ComfyLTXAVApplyModel(
        model=model,
        latent_shapes=(packed_latent_image.shapes[0], packed_latent_image.shapes[1]),
        frame_rate=float(sampler_kwargs["frame_rate"]),
        self_attention_mask=sampler_kwargs.get("self_attention_mask"),
        ref_audio_seq_len=int(sampler_kwargs.get("ref_audio_seq_len", 0)),
        keyframe_idxs=sampler_kwargs["keyframe_idxs"],
        grid_mask=sampler_kwargs.get("grid_mask"),
        orig_patchified_shape=sampler_kwargs.get("orig_patchified_shape"),
        output_orig_shape=sampler_kwargs.get("output_orig_shape"),
        video_denoise_mask=sampler_kwargs["denoise_mask"],
        audio_denoise_mask=sampler_kwargs["audio_denoise_mask"],
        guide_attention_entries=sampler_kwargs["guide_attention_entries"],
    )
    guider = ComfyCFGGuider(
        inner_model=apply_model,
        positive=[positive],
        negative=negative,
        cfg=cfg,
        model_options={
            "transformer_options": sampler_kwargs["transformer_options"] or {},
            "sample_sigmas": sigmas,
        },
    )
    packed_mask = _debug_pack_denoise_masks(
        video_mask=sampler_kwargs["denoise_mask"],
        audio_mask=sampler_kwargs["audio_denoise_mask"],
        latent_shapes=packed_latent_image.shapes,
        device=video_noise.device,
    )
    transformer_options = sampler_kwargs["transformer_options"]
    if not isinstance(transformer_options, dict):
        transformer_options = {}
        sampler_kwargs["transformer_options"] = transformer_options
    model_debug_trace: list[dict[str, object]] = []
    transformer_options["ltx_msr_debug_trace"] = model_debug_trace
    inpaint = ComfyKSamplerX0Inpaint(
        inner_model=guider,
        sigmas=sigmas,
        noise=packed_noise.tensor,
        latent_image=packed_latent_image.tensor,
        denoise_mask=packed_mask,
    )
    sigma_batch = first_sigma.expand(start.shape[0])
    with torch.inference_mode():
        denoised = inpaint(start, sigma_batch)
    start_video, start_audio = unpack_latents(start, packed_latent_image.shapes)
    denoised_video, denoised_audio = unpack_latents(denoised, packed_latent_image.shapes)
    print(f"debug_first_sigma={float(first_sigma.detach().cpu())}")
    print(f"debug_packed_shape={tuple(start.shape)}")
    context = sampler_kwargs["context"]
    attention_mask = sampler_kwargs["attention_mask"]
    if isinstance(context, torch.Tensor):
        print(f"debug_context_shape={tuple(context.shape)}")
        print(f"debug_context_dtype={context.dtype}")
    if isinstance(attention_mask, torch.Tensor):
        print(f"debug_attention_mask_shape={tuple(attention_mask.shape)}")
    else:
        print("debug_attention_mask=None")
    _print_guide_entries(sampler_kwargs["guide_attention_entries"])
    _print_mask_summary("debug_video_denoise_mask", sampler_kwargs["denoise_mask"])
    _print_mask_summary("debug_audio_denoise_mask", sampler_kwargs["audio_denoise_mask"])
    _print_mask_summary("debug_packed_denoise_mask", packed_mask)
    _print_tensor_stats("debug_start_video", start_video)
    _print_tensor_stats("debug_start_audio", start_audio)
    _print_tensor_stats("debug_denoised_video", denoised_video)
    _print_tensor_stats("debug_denoised_audio", denoised_audio)
    if dump_path is not None:
        _save_debug_dump(
            dump_path,
            first_sigma=first_sigma,
            sigmas=sigmas,
            packed_shapes=packed_latent_image.shapes,
            packed_noise=packed_noise.tensor,
            packed_latent_image=packed_latent_image.tensor,
            packed_start=start,
            packed_denoise_mask=packed_mask,
            packed_denoised=denoised,
            video_noise=video_noise,
            audio_noise=audio_noise,
            video_latent_image=video_latent_image,
            audio_latent_image=audio_latent_image,
            video_denoise_mask=sampler_kwargs["denoise_mask"],
            audio_denoise_mask=sampler_kwargs["audio_denoise_mask"],
            video_start=start_video,
            audio_start=start_audio,
            video_denoised=denoised_video,
            audio_denoised=denoised_audio,
            context=sampler_kwargs["context"],
            raw_conditioning=sampler_kwargs.get("raw_conditioning"),
            negative_raw_conditioning=sampler_kwargs.get("negative_raw_conditioning"),
            attention_mask=sampler_kwargs["attention_mask"],
            negative_context=negative_context,
            negative_attention_mask=negative_attention_mask,
            keyframe_idxs=sampler_kwargs["keyframe_idxs"],
            model_debug_trace=model_debug_trace,
        )
        print(f"debug_dump_path={Path(dump_path)}")


def _debug_pack_denoise_masks(
    *,
    video_mask,
    audio_mask,
    latent_shapes: tuple[tuple[int, ...], ...],
    device,
):
    import torch

    if video_mask is None and audio_mask is None:
        return None
    masks = []
    if video_mask is None:
        masks.append(torch.ones(latent_shapes[0], device=device, dtype=torch.float32))
    else:
        masks.append(video_mask.to(device=device, dtype=torch.float32).expand(latent_shapes[0]))
    if audio_mask is None:
        masks.append(torch.ones(latent_shapes[1], device=device, dtype=torch.float32))
    else:
        masks.append(audio_mask.to(device=device, dtype=torch.float32).expand(latent_shapes[1]))
    return pack_latents(tuple(masks)).tensor


def _print_tensor_stats(label: str, tensor) -> None:
    import torch

    value = tensor.detach().float()
    print(f"{label}_shape={tuple(tensor.shape)}")
    print(f"{label}_finite={bool(torch.isfinite(value).all().item())}")
    print(f"{label}_mean={float(value.mean().item())}")
    print(f"{label}_std={float(value.std().item())}")
    print(f"{label}_absmax={float(value.abs().max().item())}")


def _print_guide_entries(entries) -> None:
    import torch

    if not entries:
        print("debug_guide_entries_count=0")
        return
    print(f"debug_guide_entries_count={len(entries)}")
    for index, entry in enumerate(entries):
        pixel_mask = entry.get("pixel_mask") if isinstance(entry, dict) else None
        pixel_mask_shape = tuple(pixel_mask.shape) if isinstance(pixel_mask, torch.Tensor) else None
        print(
            f"debug_guide_entry_{index}=pre_filter_count:{entry.get('pre_filter_count')},"
            f"strength:{entry.get('strength')},latent_shape:{entry.get('latent_shape')},"
            f"pixel_mask_shape:{pixel_mask_shape}"
        )


def _print_mask_summary(label: str, tensor) -> None:
    import torch

    if tensor is None:
        print(f"{label}=None")
        return
    value = tensor.detach().float()
    print(f"{label}_shape={tuple(tensor.shape)}")
    print(f"{label}_min={float(value.min().item())}")
    print(f"{label}_max={float(value.max().item())}")
    unique, counts = torch.unique(value.cpu(), return_counts=True)
    if unique.numel() <= 12:
        pairs = ",".join(f"{float(u)}:{int(c)}" for u, c in zip(unique, counts, strict=True))
    else:
        pairs = f"{unique.numel()} unique values"
    print(f"{label}_values={pairs}")


def _save_debug_dump(path: str | Path, **values) -> None:
    import torch

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ltx_msr_first_step_debug_v1",
        "values": {name: _to_cpu(value) for name, value in values.items()},
    }
    torch.save(payload, output)


def _to_cpu(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value

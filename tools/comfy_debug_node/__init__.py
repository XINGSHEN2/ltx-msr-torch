from __future__ import annotations

from pathlib import Path

import torch


class _StopAfterFirstStep(Exception):
    pass


class LTXMSRDebugFirstStep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "dump_path": (
                    "STRING",
                    {"default": "/tmp/ltx_msr_comfy_first_step.pt", "multiline": False},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "debug/ltx_msr_torch"
    OUTPUT_NODE = True

    def execute(self, noise, guider, sampler, sigmas, latent_image, dump_path):
        import comfy.model_management
        import comfy.sample
        import comfy.utils

        latent = latent_image.copy()
        latent_samples = latent["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent_samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = latent_samples
        denoise_mask = latent.get("noise_mask")
        noise_tensor = noise.generate_noise(latent)
        packed_noise, _ = _pack_if_nested(noise_tensor)
        packed_latent_image, latent_shapes = _pack_if_nested(latent_samples)
        packed_denoise_mask, _ = _pack_if_nested(denoise_mask)
        output_path = Path(dump_path)
        model_debug_trace = []
        restore_model_debug_trace = _install_model_debug_trace(
            guider.model_patcher.model.diffusion_model,
            model_debug_trace,
        )

        def callback(step, denoised, x, total_steps):
            payload = {
                "format": "ltx_msr_comfy_first_step_debug_v1",
                "values": {
                    "step": int(step),
                    "total_steps": int(total_steps),
                    "sigmas": _to_cpu(sigmas),
                    "latent_shapes": latent_shapes,
                    "noise": _to_cpu(noise_tensor),
                    "latent_image": _to_cpu(latent_samples),
                    "denoise_mask": _to_cpu(denoise_mask),
                    "packed_noise": _to_cpu(packed_noise),
                    "packed_latent_image": _to_cpu(packed_latent_image),
                    "packed_denoise_mask": _to_cpu(packed_denoise_mask),
                    "guider_original_conds": _to_debug_value(getattr(guider, "original_conds", None)),
                    "guider_processed_conds": _to_debug_value(getattr(guider, "conds", None)),
                    "x": _to_cpu(x),
                    "denoised": _to_cpu(denoised),
                    "model_debug_trace": _to_cpu(model_debug_trace),
                },
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, output_path)
            raise _StopAfterFirstStep()

        try:
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            guider.sample(
                noise_tensor,
                latent_samples,
                sampler,
                sigmas,
                denoise_mask=denoise_mask,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise.seed,
            )
        except _StopAfterFirstStep:
            print(f"LTXMSRDebugFirstStep saved {output_path}")
        finally:
            restore_model_debug_trace()

        latent = latent.copy()
        latent["samples"] = latent_samples.to(comfy.model_management.intermediate_device())
        return (latent,)


def _to_cpu(value):
    if getattr(value, "is_nested", False):
        return [_to_cpu(item) for item in value.unbind()]
    if isinstance(value, torch.Tensor):
        if getattr(value, "is_nested", False):
            return [_to_cpu(item) for item in value.unbind()]
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(_to_cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    return value


def _to_debug_value(value):
    if getattr(value, "is_nested", False):
        return [_to_debug_value(item) for item in value.unbind()]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (tuple, list)):
        return type(value)(_to_debug_value(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _to_debug_value(item) for key, item in value.items()}
    if hasattr(value, "cond") and value.__class__.__module__ == "comfy.conds":
        return {
            "__class__": value.__class__.__name__,
            "cond": _to_debug_value(value.cond),
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"__class__": f"{value.__class__.__module__}.{value.__class__.__name__}"}


def _pack_if_nested(value):
    import comfy.utils

    if value is None:
        return None, []
    if getattr(value, "is_nested", False):
        return comfy.utils.pack_latents(value.unbind())
    if isinstance(value, torch.Tensor):
        return value, [tuple(value.shape)]
    return value, []


def _install_model_debug_trace(model, trace):
    originals = []
    original_process = getattr(model, "_process_transformer_blocks", None)
    if original_process is not None:
        originals.append((model, "_process_transformer_blocks", original_process))

        def process_wrapper(x, *args, **kwargs):
            if isinstance(x, (tuple, list)) and len(x) >= 2:
                _append_model_debug_trace(trace, stage="input", index=-1, video=x[0], audio=x[1])
            return original_process(x, *args, **kwargs)

        model._process_transformer_blocks = process_wrapper

    for index, block in enumerate(getattr(model, "transformer_blocks", [])):
        original_forward = block.forward
        originals.append((block, "forward", original_forward))

        def forward_wrapper(*args, __index=index, __original_forward=original_forward, **kwargs):
            output = __original_forward(*args, **kwargs)
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                _append_model_debug_trace(
                    trace,
                    stage="block",
                    index=__index,
                    video=output[0],
                    audio=output[1],
                )
            return output

        block.forward = forward_wrapper

    def restore():
        for target, name, value in originals:
            setattr(target, name, value)

    return restore


def _append_model_debug_trace(trace, *, stage, index, video, audio):
    trace.append(
        {
            "stage": stage,
            "index": int(index),
            "video": _trace_tensor_sample(video),
            "audio": _trace_tensor_sample(audio),
        }
    )


def _trace_tensor_sample(tensor):
    token_count = int(tensor.shape[1])
    dim_count = int(tensor.shape[2])
    token_indices = _trace_indices(token_count, max_count=8, device=tensor.device)
    dim_indices = _trace_indices(dim_count, max_count=16, device=tensor.device)
    sample = tensor[0].index_select(0, token_indices).index_select(1, dim_indices).detach().cpu()
    sample_float = sample.float()
    return {
        "shape": tuple(int(value) for value in tensor.shape),
        "token_indices": tuple(int(value) for value in token_indices.detach().cpu().tolist()),
        "dim_indices": tuple(int(value) for value in dim_indices.detach().cpu().tolist()),
        "sample": sample,
        "sample_mean": float(sample_float.mean().item()),
        "sample_std": float(sample_float.std().item()),
        "sample_absmax": float(sample_float.abs().max().item()),
    }


def _trace_indices(length, *, max_count, device):
    if length <= max_count:
        return torch.arange(length, device=device, dtype=torch.long)
    raw = torch.linspace(0, length - 1, steps=max_count, device=device)
    return raw.round().to(dtype=torch.long).unique(sorted=True)


NODE_CLASS_MAPPINGS = {
    "LTXMSRDebugFirstStep": LTXMSRDebugFirstStep,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXMSRDebugFirstStep": "LTX MSR Debug First Step",
}

from __future__ import annotations


def torch_dtype_from_cli(value: str):
    import torch

    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .gemma_tokenizer import GemmaTokenWeightPlan
from .text_projection import DualLinearTextProjection


@dataclass(frozen=True)
class TextConditioningInputs:
    token_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    num_tokens: tuple[int, ...]

    @property
    def real_token_count(self) -> int:
        return sum(self.num_tokens)


@dataclass(frozen=True)
class LTXTextConditioningOutput:
    conditioning: torch.Tensor
    pooled: torch.Tensor | None
    attention_mask: torch.Tensor
    extra: dict[str, object]


def build_text_conditioning_inputs(
    token_weight_pairs: Sequence[Sequence[tuple[int, float]]],
    *,
    pad_token_id: int = 0,
) -> TextConditioningInputs:
    token_ids: list[tuple[int, ...]] = []
    attention_masks: list[tuple[int, ...]] = []
    num_tokens: list[int] = []

    for batch in token_weight_pairs:
        ids = tuple(int(token_id) for token_id, _ in batch)
        left_pad = bool(ids and ids[0] == pad_token_id)
        mask: list[int] = []
        for token_id in ids:
            if left_pad and token_id == pad_token_id:
                mask.append(0)
            else:
                mask.append(1)
                left_pad = False
        token_ids.append(ids)
        mask_tuple = tuple(mask)
        attention_masks.append(mask_tuple)
        num_tokens.append(sum(mask_tuple))

    return TextConditioningInputs(
        token_ids=tuple(token_ids),
        attention_mask=tuple(attention_masks),
        num_tokens=tuple(num_tokens),
    )


def build_text_conditioning_inputs_from_plan(plan: GemmaTokenWeightPlan) -> TextConditioningInputs:
    return build_text_conditioning_inputs(plan.token_weight_pairs, pad_token_id=0)


def attention_mask_tensor(
    inputs: TextConditioningInputs,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return torch.tensor(inputs.attention_mask, dtype=torch.long, device=device)


def trim_left_padding_from_layer_hidden(
    all_layer_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    real_tokens = int(torch.sum(attention_mask).item())
    if real_tokens <= 0:
        raise ValueError("attention mask contains no real tokens")
    return all_layer_hidden[:, :, -real_tokens:]


def encode_ltx_text_conditioning(
    all_layer_hidden: torch.Tensor,
    *,
    attention_mask: torch.Tensor,
    projection: DualLinearTextProjection,
    pooled: torch.Tensor | None = None,
) -> LTXTextConditioningOutput:
    trimmed = trim_left_padding_from_layer_hidden(all_layer_hidden, attention_mask)
    conditioning = projection(trimmed)
    return LTXTextConditioningOutput(
        conditioning=conditioning.to(device=all_layer_hidden.device, dtype=torch.float),
        pooled=pooled,
        attention_mask=attention_mask.flatten().unsqueeze(dim=0),
        extra={"unprocessed_ltxav_embeds": True},
    )


def estimate_gemma_memory_mb(
    token_weight_pairs: Sequence[Sequence[tuple[int, float]]],
    *,
    pad_token_id: int = 0,
    bf16: bool = False,
) -> float:
    inputs = build_text_conditioning_inputs(token_weight_pairs, pad_token_id=pad_token_id)
    num_tokens = max(sum(len(batch) for batch in inputs.token_ids) - min(_leading_pads(batch, pad_token_id) for batch in inputs.token_ids), 642)
    constant = 3.0 if bf16 else 6.0
    return float(num_tokens * constant)


def _leading_pads(token_ids: Sequence[int], pad_token_id: int) -> int:
    count = 0
    for token_id in token_ids:
        if token_id != pad_token_id:
            break
        count += 1
    return count

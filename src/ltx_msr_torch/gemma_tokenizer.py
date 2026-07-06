from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer

from .prompt_relay import TokenRange, map_token_indices, split_local_prompts
from .text_encoder_sections import TextEncoderConfigPaths, resolve_text_encoder_config_paths


@dataclass(frozen=True)
class TokenizedText:
    text: str
    input_ids: tuple[int, ...]
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class PromptRelayTokenPlan:
    full_prompt: str
    input_ids: tuple[int, ...]
    token_ranges: tuple[TokenRange, ...]
    local_prompts: tuple[str, ...]


@dataclass(frozen=True)
class GemmaTokenWeightPlan:
    text: str
    input_ids: tuple[int, ...]
    padded_input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_weight_pairs: tuple[tuple[tuple[int, float], ...], ...]
    embedding_key: str = "gemma3_12b"

    @property
    def comfy_tokens(self) -> dict[str, tuple[tuple[tuple[int, float], ...], ...]]:
        return {self.embedding_key: self.token_weight_pairs}


class GemmaTokenizer:
    pad_token_id = 0
    min_length = 1024
    embedding_key = "gemma3_12b"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self.add_eos = False

    @classmethod
    def from_config_paths(cls, paths: TextEncoderConfigPaths | None = None) -> "GemmaTokenizer":
        resolved = paths or resolve_text_encoder_config_paths()
        return cls(Tokenizer.from_file(str(resolved.tokenizer_json)))

    @classmethod
    def from_file(cls, tokenizer_json: str | Path) -> "GemmaTokenizer":
        return cls(Tokenizer.from_file(str(tokenizer_json)))

    def __call__(self, text: str) -> dict[str, Sequence[int]]:
        return {"input_ids": self.encode(text).input_ids}

    def encode(self, text: str) -> TokenizedText:
        encoded = self._tokenizer.encode(text)
        return TokenizedText(
            text=text,
            input_ids=tuple(encoded.ids),
            tokens=tuple(encoded.tokens),
        )

    def tokenize_with_weights(self, text: str, *, min_length: int | None = None) -> GemmaTokenWeightPlan:
        tokenized = self.encode(text)
        target_length = min_length if min_length is not None else self.min_length
        pad_count = max(0, target_length - len(tokenized.input_ids))
        padded_ids = (self.pad_token_id,) * pad_count + tokenized.input_ids
        attention_mask = (0,) * pad_count + (1,) * len(tokenized.input_ids)
        pairs = tuple((token_id, 1.0) for token_id in padded_ids)
        return GemmaTokenWeightPlan(
            text=text,
            input_ids=tokenized.input_ids,
            padded_input_ids=padded_ids,
            attention_mask=attention_mask,
            token_weight_pairs=(pairs,),
            embedding_key=self.embedding_key,
        )

    def plan_prompt_relay_tokens(
        self,
        *,
        global_prompt: str,
        local_prompts: str,
    ) -> PromptRelayTokenPlan:
        locals_list = split_local_prompts(local_prompts)
        full_prompt, ranges = map_token_indices(self, global_prompt, locals_list)
        tokenized = self.encode(full_prompt)
        return PromptRelayTokenPlan(
            full_prompt=full_prompt,
            input_ids=tokenized.input_ids,
            token_ranges=ranges,
            local_prompts=locals_list,
        )

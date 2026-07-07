import torch
from safetensors.torch import save_file
from transformers import Gemma3TextConfig

from ltx_msr_torch.gemma_text_model import (
    build_empty_gemma3_text_model,
    inspect_gemma_text_model_compatibility,
    iter_comfy_text_encoder_keys,
    load_gemma3_text_config,
    load_gemma_text_state_dict_subset,
    load_gemma_text_model_weights_streaming,
    map_comfy_text_encoder_key,
    map_comfy_text_encoder_keys,
)
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def test_load_gemma3_text_config_matches_workflow_model():
    config = load_gemma3_text_config()

    assert config.hidden_size == 3840
    assert config.intermediate_size == 15360
    assert config.num_hidden_layers == 48
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.head_dim == 256
    assert config.vocab_size == 262208
    assert config.sliding_window == 1024


def test_comfy_text_encoder_key_mapping_strips_model_prefix():
    assert map_comfy_text_encoder_key("model.layers.0.self_attn.q_proj.weight") == "layers.0.self_attn.q_proj.weight"
    assert map_comfy_text_encoder_key("vision_model.encoder.layers.0.weight") is None
    assert map_comfy_text_encoder_keys(
        ["model.embed_tokens.weight", "spiece_model", "model.norm.weight"]
    ) == ("embed_tokens.weight", "norm.weight")


def test_empty_hf_gemma3_text_model_has_workflow_key_count():
    model = build_empty_gemma3_text_model()
    state_keys = tuple(model.state_dict().keys())

    assert len(state_keys) == 626
    assert state_keys[0] == "embed_tokens.weight"
    assert "layers.47.self_attn.q_proj.weight" in state_keys
    assert "norm.weight" in state_keys


def test_empty_hf_gemma3_text_model_can_limit_layers():
    model = build_empty_gemma3_text_model(num_layers=1)
    state_keys = tuple(model.state_dict().keys())

    assert model.config.num_hidden_layers == 1
    assert "layers.0.self_attn.q_proj.weight" in state_keys
    assert "layers.1.self_attn.q_proj.weight" not in state_keys


def test_gemma_text_model_keys_exactly_match_workflow_checkpoint():
    compatibility = inspect_gemma_text_model_compatibility()

    assert compatibility.checkpoint_key_count == 626
    assert compatibility.hf_key_count == 626
    assert compatibility.matched_key_count == 626
    assert compatibility.is_exact_match


def test_load_gemma_text_state_dict_subset_reads_mapped_weights():
    paths = resolve_workflow_model_paths(default_workflow_config())
    state = load_gemma_text_state_dict_subset(
        paths.text_encoder,
        [
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.47.post_feedforward_layernorm.weight",
            "norm.weight",
        ],
    )

    assert state["embed_tokens.weight"].shape == (262208, 3840)
    assert state["layers.0.self_attn.q_proj.weight"].shape == (4096, 3840)
    assert state["layers.47.post_feedforward_layernorm.weight"].shape == (3840,)
    assert state["norm.weight"].shape == (3840,)
    assert state["embed_tokens.weight"].dtype == torch.bfloat16


def test_load_gemma_text_model_weights_streaming_assigns_meta_model(tmp_path):
    config = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        sliding_window=16,
    )
    model = build_empty_gemma3_text_model(config, device="meta")
    local_state = model.state_dict()
    checkpoint_state = {
        f"model.{key}": torch.full(value.shape, 0.125, dtype=torch.float32)
        for key, value in local_state.items()
    }
    path = tmp_path / "small_gemma_text.safetensors"
    save_file(checkpoint_state, path)

    report = load_gemma_text_model_weights_streaming(model, path, assign=True)

    assert report.loaded == len(local_state)
    assert report.missing == ()
    assert not model.embed_tokens.weight.is_meta
    assert torch.allclose(model.embed_tokens.weight, torch.full_like(model.embed_tokens.weight, 0.125))


def test_iter_comfy_text_encoder_keys_only_returns_text_model_keys():
    paths = resolve_workflow_model_paths(default_workflow_config())
    keys = iter_comfy_text_encoder_keys(paths.text_encoder)

    assert len(keys) == 626
    assert keys[0].startswith("model.")
    assert all(key.startswith("model.") for key in keys)

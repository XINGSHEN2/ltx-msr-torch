import torch

from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.text_encoder_sections import (
    inspect_text_encoder_section,
    resolve_text_encoder_config_paths,
)
from ltx_msr_torch.workflow_config import default_workflow_config


def test_resolve_text_encoder_config_paths_finds_gemma_files():
    paths = resolve_text_encoder_config_paths()

    assert paths.gemma_config.name == "gemma3cfg.json"
    assert paths.tokenizer_json.exists()
    assert paths.tokenizer_model.exists()
    assert paths.tokenizer_config.exists()


def test_inspect_text_encoder_section_matches_workflow_file():
    paths = resolve_workflow_model_paths(default_workflow_config())
    manifest = inspect_text_encoder_section(paths.text_encoder)

    assert manifest.key_count == 1066
    assert manifest.text_model_key_count == 626
    assert manifest.vision_model_key_count == 437
    assert manifest.projector_key_count == 2
    assert manifest.spiece_key_count == 1
    assert manifest.layer_count == 48
    assert manifest.first_text_shapes[0] == ("model.embed_tokens.weight", (262208, 3840), torch.bfloat16)

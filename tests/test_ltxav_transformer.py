from ltx_msr_torch.ltxav_transformer import inspect_ltxav_transformer_manifest
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def test_inspect_ltxav_transformer_manifest_matches_workflow_checkpoint():
    paths = resolve_workflow_model_paths(default_workflow_config())
    manifest = inspect_ltxav_transformer_manifest(paths.checkpoint)
    config = manifest.config

    assert manifest.key_count == 4444
    assert manifest.block_count == 48
    assert manifest.block_key_count == 4128
    assert manifest.keys_per_block == 86
    assert config.image_model == "ltxav"
    assert config.num_layers == 48
    assert config.in_channels == 128
    assert config.out_channels == 128
    assert config.cross_attention_dim == 4096
    assert config.audio_cross_attention_dim == 2048
    assert config.attention_head_dim == 128
    assert config.audio_attention_head_dim == 64
    assert config.num_attention_heads == 32
    assert config.audio_num_attention_heads == 32
    assert config.caption_channels == 3840
    assert config.use_audio_video_cross_attention
    assert config.use_embeddings_connector
    assert config.connector_num_layers == 8
    assert config.connector_num_learnable_registers == 128
    assert config.cross_attention_adaln
    assert config.apply_gated_attention
    assert config.rope_type == "split"
    assert config.frequencies_precision == "float64"


def test_ltxav_transformer_manifest_records_projection_and_connector_shapes():
    paths = resolve_workflow_model_paths(default_workflow_config())
    manifest = inspect_ltxav_transformer_manifest(paths.checkpoint)
    specs = manifest.specs

    assert specs["model.diffusion_model.patchify_proj.weight"].shape == (4096, 128)
    assert specs["model.diffusion_model.proj_out.weight"].shape == (128, 4096)
    assert specs["model.diffusion_model.audio_patchify_proj.weight"].shape == (2048, 128)
    assert specs["model.diffusion_model.audio_proj_out.weight"].shape == (128, 2048)
    assert specs["model.diffusion_model.transformer_blocks.0.attn2.to_k.weight"].shape == (4096, 4096)
    assert specs["model.diffusion_model.transformer_blocks.0.audio_attn2.to_k.weight"].shape == (2048, 2048)
    assert specs["model.diffusion_model.video_embeddings_connector.learnable_registers"].shape == (128, 4096)
    assert specs["model.diffusion_model.audio_embeddings_connector.learnable_registers"].shape == (128, 2048)
    assert {spec.dtype for spec in specs.values()} == {"bfloat16"}


def test_ltxav_transformer_manifest_group_counts_cover_av_sections():
    paths = resolve_workflow_model_paths(default_workflow_config())
    manifest = inspect_ltxav_transformer_manifest(paths.checkpoint)
    groups = dict(manifest.group_counts)

    assert groups["transformer_blocks"] == 4128
    assert groups["video_embeddings_connector"] == 129
    assert groups["audio_embeddings_connector"] == 129
    assert groups["patchify_proj"] == 2
    assert groups["audio_patchify_proj"] == 2
    assert groups["proj_out"] == 2
    assert groups["audio_proj_out"] == 2

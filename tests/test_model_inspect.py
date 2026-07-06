from ltx_msr_torch.model_inspect import inspect_workflow_model_headers
from ltx_msr_torch.model_paths import resolve_workflow_model_paths
from ltx_msr_torch.workflow_config import default_workflow_config


def test_inspect_workflow_model_headers_reads_expected_files():
    paths = resolve_workflow_model_paths(default_workflow_config())
    inspection = inspect_workflow_model_headers(paths)

    assert inspection.checkpoint.key_count > 1000
    assert "config" in (inspection.checkpoint.metadata or {})
    assert inspection.text_encoder.key_count > 1000
    assert inspection.lora.key_count > 0


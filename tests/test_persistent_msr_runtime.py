from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ltx_msr_torch.msr_case as msr_case
from ltx_msr_torch.msr_case import PersistentMSRRuntime


def runtime_and_args(tmp_path: Path) -> tuple[PersistentMSRRuntime, Namespace]:
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}")
    config = SimpleNamespace(
        model=SimpleNamespace(lora="resident.safetensors", lora_strength=1.0)
    )
    args = Namespace(
        workflow=str(workflow),
        dtype="bf16",
        device="cuda:3",
        layers=48,
        lora_name=None,
        lora_strength=None,
        no_apply_lora=False,
    )
    runtime = PersistentMSRRuntime(
        workflow_path=workflow.resolve(),
        config=config,
        state=None,
        dtype=torch.bfloat16,
        text_device=torch.device("cuda:2"),
        model_device=torch.device("cuda:3"),
        layers=48,
        tokenizer=None,
        gemma=None,
        gemma_report=None,
        projection=None,
        video_connector=None,
        audio_connector=None,
        decoders=None,
        model=None,
        model_report=None,
        lora_name="resident.safetensors",
        lora_strength=1.0,
        lora_report=None,
        apply_lora=True,
    )
    return runtime, args


def test_resident_runtime_accepts_matching_request(tmp_path: Path) -> None:
    runtime, args = runtime_and_args(tmp_path)

    runtime.validate_request(args)


@pytest.mark.parametrize(
    ("field", "value"),
    (("layers", 1), ("lora_strength", 0.5), ("device", "cuda:2")),
)
def test_resident_runtime_rejects_model_configuration_changes(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    runtime, args = runtime_and_args(tmp_path)
    setattr(args, field, value)

    with pytest.raises(ValueError, match="incompatible with the resident MSR runtime"):
        runtime.validate_request(args)


def test_load_resident_runtime_places_stacks_on_separate_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = tmp_path / "workflow.json"
    workflow.write_text("{}")
    config = SimpleNamespace(
        model=SimpleNamespace(lora="resident.safetensors", lora_strength=1.0)
    )
    state = SimpleNamespace(
        model_paths=SimpleNamespace(
            checkpoint=tmp_path / "checkpoint.safetensors",
            text_encoder=tmp_path / "text.safetensors",
        )
    )
    calls: dict[str, object] = {}

    class FakeModule:
        def eval(self):
            return self

    fake_gemma = FakeModule()
    fake_model = FakeModule()
    fake_decoders = SimpleNamespace(video_vae=FakeModule(), audio_vae=FakeModule())

    monkeypatch.setattr(msr_case, "extract_workflow_config", lambda path: config)
    monkeypatch.setattr(msr_case, "build_low_level_state", lambda value, device: state)
    monkeypatch.setattr(
        msr_case,
        "GemmaTokenizer",
        SimpleNamespace(from_config_paths=lambda: "tokenizer"),
    )

    def build_gemma(**kwargs):
        calls["gemma_device"] = kwargs["device"]
        return fake_gemma

    monkeypatch.setattr(msr_case, "build_empty_gemma3_text_model", build_gemma)
    monkeypatch.setattr(msr_case, "load_gemma_text_model_weights_streaming", lambda *args, **kwargs: "gemma-report")
    monkeypatch.setattr(msr_case, "build_text_projection_from_checkpoint", lambda *args, **kwargs: "projection")
    monkeypatch.setattr(
        msr_case,
        "build_embeddings_connector_from_checkpoint",
        lambda *args, **kwargs: f"{args[1]}-connector",
    )

    def load_decoders(*args, **kwargs):
        calls["decoder_device"] = kwargs["device"]
        return fake_decoders

    monkeypatch.setattr(msr_case, "load_ltxav_decoders_from_checkpoint", load_decoders)
    monkeypatch.setattr(msr_case, "create_ltxav_model_from_checkpoint", lambda *args, **kwargs: fake_model)

    def load_model(*args, **kwargs):
        calls["model_device"] = kwargs["device"]
        return "model-report"

    monkeypatch.setattr(msr_case, "load_ltxav_model_weights_streaming", load_model)
    monkeypatch.setattr(msr_case, "resolve_lora_path", lambda name: tmp_path / name)
    monkeypatch.setattr(msr_case, "inspect_lora_manifest", lambda path: "manifest")
    monkeypatch.setattr(msr_case, "apply_lora_to_ltxav_model", lambda *args, **kwargs: "lora-report")

    args = Namespace(
        workflow=str(workflow),
        dtype="bf16",
        device="cuda:3",
        layers=48,
        lora_name=None,
        lora_strength=None,
        no_apply_lora=False,
    )
    runtime = msr_case.load_persistent_msr_runtime(args, text_device="cuda:2")

    assert calls == {
        "gemma_device": torch.device("cuda:2"),
        "decoder_device": torch.device("cuda:3"),
        "model_device": torch.device("cuda:3"),
    }
    assert runtime.gemma is fake_gemma
    assert runtime.model is fake_model
    assert runtime.decoders is fake_decoders
    assert runtime.text_device == torch.device("cuda:2")
    assert runtime.model_device == torch.device("cuda:3")

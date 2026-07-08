from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .workflow_config import (
    IcLoraGuideConfig,
    LatentConfig,
    ModelConfig,
    NagConfig,
    OutputConfig,
    PromptConfig,
    ReferenceConfig,
    SamplingConfig,
    WorkflowConfig,
)


def extract_workflow_config(path: str | Path) -> WorkflowConfig:
    """Extract parity-critical settings from a ComfyUI workflow JSON."""
    workflow = json.loads(Path(path).read_text())
    nodes = workflow.get("nodes", [])
    graph = _WorkflowGraph(workflow)

    checkpoint_node = _single(nodes, "LowVRAMCheckpointLoader")
    text_encoder_node = _single(nodes, "LTXAVTextEncoderLoader")
    lora_node = _single(nodes, "LTXICLoRALoaderModelOnly")
    reference_node = _single(nodes, "LiconMSR")
    latent_node = _single(nodes, "EmptyLTXVLatentVideo")
    audio_node = _single(nodes, "LTXVEmptyLatentAudio")
    noise_node = _single(nodes, "RandomNoise")
    sampler_node = _single(nodes, "KSamplerSelect")
    sigmas_node = _single(nodes, "ManualSigmas")
    cfg_node = _single(nodes, "CFGGuider")
    nag_node = _single(nodes, "LTX2_NAG")
    guide_node = _single(nodes, "LTXAddVideoICLoRAGuide")
    prompt_relay_node = _single(nodes, "PromptRelayEncode")
    negative_node = _single(nodes, "CLIPTextEncode")
    create_video_node = _single(nodes, "CreateVideo")
    save_video_node = _single(nodes, "SaveVideo")

    checkpoint = _widget(checkpoint_node, 0, "ltx-2.3-22b-distilled-1.1.safetensors")
    text_encoder = _widget(text_encoder_node, 0, "gemma_3_12B_it.safetensors")
    lora = str(_widget(lora_node, 0, "LTX-2.3/LTX-2.3-Licon-MSR-V1.safetensors")).replace("\\", "/")

    return WorkflowConfig(
        model=ModelConfig(
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            lora=lora,
            lora_strength=float(_widget(lora_node, 1, 1.0)),
        ),
        reference=ReferenceConfig(
            width=int(graph.input_value(reference_node, "width", 0, 1920)),
            height=int(graph.input_value(reference_node, "height", 1, 1280)),
            frame_count=int(_widget(reference_node, 2, 41)),
        ),
        latent=LatentConfig(
            width=int(graph.input_value(latent_node, "width", 0, 1280)),
            height=int(graph.input_value(latent_node, "height", 1, 1920)),
            video_frames=int(graph.input_value(latent_node, "length", 2, 145)),
            batch_size=int(_widget(latent_node, 3, 1)),
            audio_frames=int(graph.input_value(audio_node, "frames_number", 0, 241)),
            frame_rate=int(_widget(audio_node, 1, 24)),
        ),
        sampling=SamplingConfig(
            seed=int(_widget(noise_node, 0, 0)),
            sampler=_widget(sampler_node, 0, "euler"),
            cfg=float(_widget(cfg_node, 0, 1.0)),
            sigmas=_parse_sigmas(_widget(sigmas_node, 0, "")),
        ),
        nag=NagConfig(
            scale=float(_widget(nag_node, 0, 11.0)),
            alpha=float(_widget(nag_node, 1, 0.25)),
            tau=float(_widget(nag_node, 2, 2.5)),
            inplace=bool(_widget(nag_node, 3, True)),
        ),
        ic_lora_guide=IcLoraGuideConfig(
            frame_idx=int(_widget(guide_node, 0, 0)),
            strength=float(_widget(guide_node, 1, 1.0)),
            latent_downscale_factor=float(_widget(guide_node, 2, 1.0)),
            crop=_widget(guide_node, 3, "center"),
            use_tiled_encode=bool(_widget(guide_node, 4, False)),
            tile_size=int(_widget(guide_node, 5, 256)),
            tile_overlap=int(_widget(guide_node, 6, 64)),
        ),
        prompt=PromptConfig(
            global_prompt=_widget(prompt_relay_node, 0, ""),
            local_prompts=_widget(prompt_relay_node, 1, ""),
            segment_lengths=_widget(prompt_relay_node, 2, ""),
            epsilon=float(_widget(prompt_relay_node, 3, 0.0022)),
            negative_prompt=_widget(negative_node, 0, ""),
        ),
        output=OutputConfig(
            fps=int(_widget(create_video_node, 0, 24)),
            filename_prefix=_widget(save_video_node, 0, "LTX-2/MSR_verify_iclora_guide"),
            format=_widget(save_video_node, 1, "auto"),
            codec=_widget(save_video_node, 2, "auto"),
        ),
    )


def _single(nodes: list[dict[str, Any]], node_type: str) -> dict[str, Any] | None:
    matches = [node for node in nodes if node.get("type") == node_type]
    if not matches:
        return None
    return matches[0]


def _widget(node: dict[str, Any] | None, index: int, default: Any) -> Any:
    if node is None:
        return default
    widgets = node.get("widgets_values") or []
    if index >= len(widgets):
        return default
    return widgets[index]


def _parse_sigmas(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if isinstance(value, list):
        return tuple(float(part) for part in value)
    return ()


class _WorkflowGraph:
    def __init__(self, workflow: dict[str, Any]) -> None:
        self.nodes = {str(node["id"]): node for node in workflow.get("nodes", [])}
        self.links = {int(link[0]): link for link in workflow.get("links", [])}

    def input_value(
        self,
        node: dict[str, Any] | None,
        input_name: str,
        widget_index: int,
        default: Any,
    ) -> Any:
        linked = self.linked_input_value(node, input_name)
        if linked is not None:
            return linked
        return _widget(node, widget_index, default)

    def linked_input_value(self, node: dict[str, Any] | None, input_name: str) -> Any | None:
        if node is None:
            return None
        for input_def in node.get("inputs") or []:
            if input_def.get("name") != input_name:
                continue
            link_id = input_def.get("link")
            if link_id is None:
                return None
            link = self.links.get(int(link_id))
            if link is None:
                return None
            source = self.nodes.get(str(link[1]))
            if source is None:
                return None
            source_type = source.get("type")
            if source_type in {"INTConstant", "FloatConstant", "StringConstant"}:
                return _widget(source, 0, None)
            return None
        return None

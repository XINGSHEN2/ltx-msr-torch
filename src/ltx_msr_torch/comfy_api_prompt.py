from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .prompt_utils import parse_reference_prompt_file


WIDGET_INPUT_NAMES: dict[str, tuple[str, ...]] = {
    "RandomNoise": ("noise_seed",),
    "KSamplerSelect": ("sampler_name",),
    "INTConstant": ("value",),
    "ManualSigmas": ("sigmas",),
    "LoadImage": ("image",),
    "LowVRAMCheckpointLoader": ("ckpt_name",),
    "LTXVAudioVAELoader": ("ckpt_name",),
    "LTXAVTextEncoderLoader": ("text_encoder", "ckpt_name", "device"),
    "EmptyLTXVLatentVideo": ("width", "height", "length", "batch_size"),
    "LTXVEmptyLatentAudio": ("frames_number", "frame_rate", "batch_size"),
    "CLIPTextEncode": ("text",),
    "LTXVConditioning": ("frame_rate",),
    "LiconMSR": ("width", "height", "frame_count"),
    "LTXICLoRALoaderModelOnly": ("lora_name", "strength_model"),
    "PromptRelayEncode": (
        "global_prompt",
        "local_prompts",
        "segment_lengths",
        "epsilon",
    ),
    "LTXAddVideoICLoRAGuide": (
        "frame_idx",
        "strength",
        "latent_downscale_factor",
        "crop",
        "use_tiled_encode",
        "tile_size",
        "tile_overlap",
    ),
    "LTX2_NAG": ("nag_scale", "nag_alpha", "nag_tau", "inplace"),
    "CFGGuider": ("cfg",),
    "CreateVideo": ("fps",),
    "SaveVideo": ("filename_prefix", "format", "codec"),
}


def load_ui_workflow(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def build_api_prompt_from_ui_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """Convert a ComfyUI frontend workflow JSON into API prompt JSON.

    This supports the inspected LTX 2.3 MSR workflow and common ComfyUI widget
    wiring. It is intentionally conservative: unknown widget-only nodes must be
    added to `WIDGET_INPUT_NAMES` before they can be represented faithfully.
    """
    link_sources = _link_sources(workflow.get("links", []))
    api_prompt: dict[str, Any] = {}

    for node in workflow.get("nodes", []):
        if node.get("mode") == 2:
            continue
        node_id = str(node["id"])
        class_type = node["type"]
        inputs = _node_inputs(node, link_sources)
        api_prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }
    return api_prompt


def build_case_api_prompt(
    workflow_path: str | Path,
    case_dir: str | Path,
    output_prefix: str = "LTX-2/MSR_torch_parity",
) -> dict[str, Any]:
    workflow = load_ui_workflow(workflow_path)
    api_prompt = build_api_prompt_from_ui_workflow(workflow)

    case_dir = Path(case_dir)
    global_prompt, local_prompts = parse_reference_prompt_file(case_dir / "prompt.txt")

    _set_load_image(api_prompt, "29", _comfy_input_relative(case_dir / "1.jpg"))
    _set_load_image(api_prompt, "40", _comfy_input_relative(case_dir / "2.jpg"))
    _set_load_image(api_prompt, "30", _comfy_input_relative(case_dir / "bg.png"))

    # The sample has two subjects. Disconnect the extra subject slots from
    # LiconMSR so the frame distribution is 1 -> 2 -> background.
    for optional_input in ("3", "4"):
        api_prompt["28"]["inputs"].pop(optional_input, None)

    api_prompt["99"]["inputs"]["global_prompt"] = global_prompt
    api_prompt["99"]["inputs"]["local_prompts"] = local_prompts
    api_prompt["20"]["inputs"]["filename_prefix"] = output_prefix
    return _prune_to_outputs(api_prompt, output_node_ids=["20"])


def save_api_prompt(prompt: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + "\n")


def _node_inputs(
    node: dict[str, Any],
    link_sources: dict[int, tuple[str, int]],
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    consumed_widgets: set[int] = set()

    for input_def in node.get("inputs") or []:
        name = input_def["name"]
        link_id = input_def.get("link")
        if link_id is not None:
            inputs[name] = list(link_sources[int(link_id)])

    widget_names = WIDGET_INPUT_NAMES.get(node["type"], ())
    for index, name in enumerate(widget_names):
        if index in consumed_widgets:
            continue
        if name in inputs:
            continue
        widgets = node.get("widgets_values") or []
        if index < len(widgets):
            inputs[name] = copy.deepcopy(widgets[index])

    return inputs


def _link_sources(links: list[list[Any]]) -> dict[int, tuple[str, int]]:
    return {
        int(link[0]): (str(link[1]), int(link[2]))
        for link in links
    }


def _set_load_image(api_prompt: dict[str, Any], node_id: str, filename: str) -> None:
    api_prompt[node_id]["inputs"]["image"] = filename


def _prune_to_outputs(
    api_prompt: dict[str, Any],
    output_node_ids: list[str],
) -> dict[str, Any]:
    reachable: set[str] = set()
    stack = list(output_node_ids)
    while stack:
        node_id = stack.pop()
        if node_id in reachable or node_id not in api_prompt:
            continue
        reachable.add(node_id)
        for value in api_prompt[node_id].get("inputs", {}).values():
            if _is_link(value):
                stack.append(value[0])
    return {
        node_id: node
        for node_id, node in api_prompt.items()
        if node_id in reachable
    }


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _comfy_input_relative(path: Path) -> str:
    comfy_input_literal = Path("/home/xingshen/ComfyUI/input")
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(comfy_input_literal))
    except ValueError:
        pass

    comfy_input = comfy_input_literal.resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(comfy_input))
    except ValueError:
        # Fall back to the symlink path used by ComfyUI input if this is the
        # downloaded LTX MSR examples directory.
        marker = Path(
            "/mnt/AINAS0/user/xingshen/LTX-2.3-Multiple-Subject-Reference/examples-hf"
        )
        try:
            rel = resolved.relative_to(marker)
            return str(Path("ltx_msr_" + rel.parts[0]) / Path(*rel.parts[1:]))
        except ValueError:
            return str(resolved)

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from ltx_msr_torch.prompt_utils import parse_reference_prompt_file


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
    """Convert the inspected MSR frontend workflow JSON into API prompt JSON.

    This is a development-only ComfyUI parity tool. The pure torch runtime does
    not import this module.
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
    use_case_prompt: bool = False,
) -> dict[str, Any]:
    workflow = load_ui_workflow(workflow_path)
    api_prompt = build_api_prompt_from_ui_workflow(workflow)

    _inject_case_inputs(api_prompt, case_dir, use_case_prompt=use_case_prompt)
    api_prompt["20"]["inputs"]["filename_prefix"] = output_prefix
    return _prune_to_outputs(api_prompt, output_node_ids=["20"])


def build_case_first_step_debug_api_prompt(
    workflow_path: str | Path,
    case_dir: str | Path,
    dump_path: str | Path = "/tmp/ltx_msr_comfy_first_step.pt",
    debug_node_id: str = "100000",
    use_case_prompt: bool = False,
) -> dict[str, Any]:
    workflow = load_ui_workflow(workflow_path)
    api_prompt = build_api_prompt_from_ui_workflow(workflow)

    _inject_case_inputs(api_prompt, case_dir, use_case_prompt=use_case_prompt)
    sampler_inputs = api_prompt["16"]["inputs"]
    api_prompt[debug_node_id] = {
        "class_type": "LTXMSRDebugFirstStep",
        "inputs": {
            "noise": sampler_inputs["noise"],
            "guider": sampler_inputs["guider"],
            "sampler": sampler_inputs["sampler"],
            "sigmas": sampler_inputs["sigmas"],
            "latent_image": sampler_inputs["latent_image"],
            "dump_path": str(dump_path),
        },
    }
    return _prune_to_outputs(api_prompt, output_node_ids=[debug_node_id])


def _inject_case_inputs(api_prompt: dict[str, Any], case_dir: str | Path, *, use_case_prompt: bool = False) -> None:
    case_dir = Path(case_dir)
    for node in api_prompt.values():
        if node.get("class_type") != "LoadImage":
            continue
        image_name = str(node.get("inputs", {}).get("image", ""))
        if not image_name:
            continue
        try:
            resolved_image = _resolve_workflow_image_name(image_name, case_dir)
        except FileNotFoundError:
            continue
        node["inputs"]["image"] = _comfy_input_relative(resolved_image)

    for optional_input in ("3", "4"):
        api_prompt["28"]["inputs"].pop(optional_input, None)

    if use_case_prompt:
        global_prompt, local_prompts = parse_reference_prompt_file(case_dir / "prompt.txt")
        api_prompt["99"]["inputs"]["global_prompt"] = global_prompt
        api_prompt["99"]["inputs"]["local_prompts"] = local_prompts


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


def _resolve_workflow_image_name(image_name: str, case_dir: Path) -> Path:
    image_path = Path(image_name)
    candidates = [image_path] if image_path.is_absolute() else [case_dir / image_name]
    if image_name.startswith("bg "):
        candidates.append(case_dir / "bg.png")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve workflow image {image_name!r}; tried {candidates}")


def _comfy_input_relative(path: Path) -> str:
    configured_input = os.environ.get("COMFYUI_INPUT_DIR")
    configured_root = os.environ.get("COMFYUI_ROOT")
    if configured_input:
        comfy_input_literal = Path(configured_input).expanduser()
    elif configured_root:
        comfy_input_literal = Path(configured_root).expanduser() / "input"
    else:
        comfy_input_literal = Path("ComfyUI/input")
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
        return str(resolved)

from pathlib import Path

from tools.comfy_api_prompt import build_case_api_prompt


WORKFLOW = Path(
    "/home/xingshen/ComfyUI/custom_nodes/ComfyUI-Licon-MSR/LTX-2.3_MSR_sample_workflow_V2.json"
)
CASE_01 = Path(
    "/mnt/AINAS0/user/xingshen/LTX-2.3-Multiple-Subject-Reference/examples-hf/validition_v1/01"
)


def test_build_case_api_prompt_injects_case_01_inputs():
    prompt = build_case_api_prompt(WORKFLOW, CASE_01, output_prefix="LTX-2/test")

    assert prompt["29"]["inputs"]["image"] == "ltx_msr_validition_v1/01/1.jpg"
    assert prompt["40"]["inputs"]["image"] == "ltx_msr_validition_v1/01/2.jpg"
    assert prompt["30"]["inputs"]["image"] == "ltx_msr_validition_v1/01/bg.png"
    assert prompt["28"]["inputs"]["1"] == ["29", 0]
    assert prompt["28"]["inputs"]["2"] == ["40", 0]
    assert "3" not in prompt["28"]["inputs"]
    assert "4" not in prompt["28"]["inputs"]
    assert prompt["28"]["inputs"]["frame_count"] == 41
    assert prompt["20"]["inputs"]["filename_prefix"] == "LTX-2/test"
    assert "参考图1" in prompt["99"]["inputs"]["global_prompt"]
    assert "石板小路" in prompt["99"]["inputs"]["local_prompts"]
    assert "33" not in prompt
    assert "95" not in prompt
    assert "104" not in prompt
    assert "105" not in prompt

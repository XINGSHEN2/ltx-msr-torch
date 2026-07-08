from pathlib import Path

from tools.comfy_api_prompt import build_case_api_prompt
from ltx_msr_torch.msr_reference import create_msr_reference_video_from_paths
from ltx_msr_torch.prompt_utils import parse_reference_prompt_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CASE = PROJECT_ROOT / "sample_cases" / "validition_v1_01"
WORKFLOW = Path(
    "/home/xingshen/ComfyUI/custom_nodes/ComfyUI-Licon-MSR/LTX-2.3_MSR_sample_workflow_V2.json"
)


def test_project_sample_case_assets_are_usable():
    global_prompt, local_prompt = parse_reference_prompt_file(SAMPLE_CASE / "prompt.txt")
    assert "参考图1" in global_prompt
    assert "石板小路" in local_prompt

    reference = create_msr_reference_video_from_paths(
        subjects=[SAMPLE_CASE / "1.jpg", SAMPLE_CASE / "2.jpg", None, None],
        background=SAMPLE_CASE / "bg.png",
        width=192,
        height=128,
        frame_count=41,
    )
    assert tuple(reference.shape) == (41, 128, 192, 3)
    assert reference.dtype.is_floating_point
    assert float(reference.min()) >= 0.0
    assert float(reference.max()) <= 1.0


def test_project_sample_case_builds_api_prompt():
    prompt = build_case_api_prompt(
        WORKFLOW,
        SAMPLE_CASE,
        output_prefix="LTX-2/test_project_sample",
    )

    assert prompt["29"]["inputs"]["image"].endswith("sample_cases/validition_v1_01/1.jpg")
    assert prompt["40"]["inputs"]["image"].endswith("sample_cases/validition_v1_01/2.jpg")
    assert prompt["30"]["inputs"]["image"].endswith("sample_cases/validition_v1_01/bg.png")
    assert prompt["20"]["inputs"]["filename_prefix"] == "LTX-2/test_project_sample"

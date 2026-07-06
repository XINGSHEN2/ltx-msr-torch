from ltx_msr_torch.prompt_utils import parse_reference_prompt


def test_parse_reference_prompt_splits_reference_and_motion_lines():
    global_prompt, local_prompts = parse_reference_prompt(
        "参考图1：A\n"
        "参考图2：B\n"
        "动作一。\n"
        "动作二。\n"
    )

    assert global_prompt == "参考图1：A\n参考图2：B"
    assert local_prompts == "动作一。\n动作二。"


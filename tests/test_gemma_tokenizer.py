from ltx_msr_torch.gemma_tokenizer import GemmaTokenizer
from ltx_msr_torch.prompt_utils import parse_reference_prompt_file


def test_gemma_tokenizer_loads_workflow_tokenizer():
    tokenizer = GemmaTokenizer.from_config_paths()
    tokenized = tokenizer.encode("参考图1：红色水枪")

    assert tokenized.input_ids[0] == 2
    assert tokenized.tokens[0] == "<bos>"
    assert "参考" in tokenized.tokens


def test_gemma_tokenizer_plans_prompt_relay_token_ranges():
    tokenizer = GemmaTokenizer.from_config_paths()
    plan = tokenizer.plan_prompt_relay_tokens(
        global_prompt="global prompt",
        local_prompts="local one | local two",
    )

    assert plan.full_prompt == "global prompt local one local two"
    assert len(plan.local_prompts) == 2
    assert len(plan.token_ranges) == 2
    assert plan.token_ranges[0][0] < plan.token_ranges[0][1] <= plan.token_ranges[1][0]
    assert len(plan.input_ids) >= plan.token_ranges[-1][1]


def test_gemma_tokenizer_tokenizes_project_sample_prompt():
    global_prompt, local_prompts = parse_reference_prompt_file(
        "/home/xingshen/yiwu/ltx-msr-torch/sample_cases/validition_v1_01/prompt.txt"
    )
    tokenizer = GemmaTokenizer.from_config_paths()
    plan = tokenizer.plan_prompt_relay_tokens(
        global_prompt=global_prompt,
        local_prompts=local_prompts,
    )

    assert len(plan.input_ids) > 20
    assert len(plan.token_ranges) == 1


def test_gemma_tokenizer_builds_comfy_style_token_weight_plan():
    tokenizer = GemmaTokenizer.from_config_paths()
    raw = tokenizer.encode("参考图1：红色水枪")
    plan = tokenizer.tokenize_with_weights("参考图1：红色水枪")

    assert len(plan.padded_input_ids) == 1024
    assert len(plan.attention_mask) == 1024
    assert plan.padded_input_ids[: 1024 - len(raw.input_ids)] == (0,) * (1024 - len(raw.input_ids))
    assert plan.padded_input_ids[-len(raw.input_ids) :] == raw.input_ids
    assert sum(plan.attention_mask) == len(raw.input_ids)
    assert plan.comfy_tokens["gemma3_12b"][0][-len(raw.input_ids) :][0][0] == raw.input_ids[0]
    assert {weight for _, weight in plan.comfy_tokens["gemma3_12b"][0]} == {1.0}


def test_gemma_tokenizer_pads_project_sample_prompt_like_workflow():
    global_prompt, local_prompts = parse_reference_prompt_file(
        "/home/xingshen/yiwu/ltx-msr-torch/sample_cases/validition_v1_01/prompt.txt"
    )
    tokenizer = GemmaTokenizer.from_config_paths()
    relay_plan = tokenizer.plan_prompt_relay_tokens(
        global_prompt=global_prompt,
        local_prompts=local_prompts,
    )
    token_plan = tokenizer.tokenize_with_weights(relay_plan.full_prompt)

    assert len(relay_plan.input_ids) == len(token_plan.input_ids)
    assert len(token_plan.padded_input_ids) == 1024
    assert sum(token_plan.attention_mask) == len(token_plan.input_ids)
    assert token_plan.padded_input_ids[-len(token_plan.input_ids) :] == token_plan.input_ids

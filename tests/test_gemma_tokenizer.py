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

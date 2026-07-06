from ltx_msr_torch.prompt_relay import (
    build_relay_segments,
    convert_to_latent_lengths,
    distribute_segment_lengths,
    map_token_indices,
    plan_prompt_relay,
    split_local_prompts,
)


def test_split_local_prompts_matches_promptrelay_separator():
    assert split_local_prompts(" one | two || three ") == ("one", "two", "three")


def test_convert_to_latent_lengths_uses_largest_remainder():
    assert convert_to_latent_lengths([80, 40, 40], temporal_stride=8, latent_frames=19) == [9, 5, 5]
    assert convert_to_latent_lengths([0, 0], temporal_stride=8, latent_frames=19) == [1, 1]


def test_distribute_segment_lengths_caps_to_latent_frames():
    assert distribute_segment_lengths(3, 19) == (7, 7, 5)
    assert distribute_segment_lengths(3, 5, [4, 4, 4]) == (4, 1, 0)


def test_build_relay_segments_matches_promptrelay_math():
    segments = build_relay_segments([(2, 5), (5, 9)], [7, 5], epsilon=0.0022)

    assert len(segments) == 2
    assert segments[0].token_range == (2, 5)
    assert segments[0].midpoint == 3
    assert segments[0].window == 1.0
    assert round(segments[0].sigma, 6) == round(1.0 / 6.119297918617867, 6)
    assert segments[1].midpoint == 9
    assert segments[1].window == 0.0


def test_map_token_indices_uses_incremental_lengths():
    def tokenizer(text: str) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}

    full_prompt, ranges = map_token_indices(tokenizer, "global prompt", ["local one", "local two"])

    assert full_prompt == "global prompt local one local two"
    assert ranges == ((2, 4), (4, 6))


def test_plan_prompt_relay_builds_effective_segments():
    plan = plan_prompt_relay(
        local_prompts="first | second | third",
        latent_shape=(1, 128, 19, 60, 40),
        patch_size=(1, 2, 2),
        temporal_stride=8,
        segment_lengths="48, 48, 48",
        token_ranges=[(10, 12), (12, 15), (15, 18)],
        epsilon=0.0022,
    )

    assert plan.local_prompts == ("first", "second", "third")
    assert plan.tokens_per_frame == 600
    assert plan.specified_latent_lengths == (7, 6, 6)
    assert plan.effective_lengths == (7, 6, 6)
    assert tuple(segment.midpoint for segment in plan.segments) == (3, 10, 16)

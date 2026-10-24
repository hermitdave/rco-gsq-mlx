"""Grouping tests, including the MLX vs HF naming difference."""

import numpy as np

from rco_mlx.grouping import (
    attach_param_counts,
    build_layer_groups,
    build_moe_per_expert_groups,
    get_group_param_count,
    get_layer_to_group_mapping,
    parse_group_spec,
)


def test_no_patterns_gives_one_group_per_layer():
    names = ["a", "b", "c"]
    groups = build_layer_groups(names)
    assert len(groups) == 3
    assert all(len(g.layer_names) == 1 for g in groups)


def test_parse_group_spec():
    spec = "mlp.gate_proj,mlp.down_proj self_attn.k_proj,self_attn.q_proj"
    groups = parse_group_spec(spec)
    assert groups == [["mlp.gate_proj", "mlp.down_proj"], ["self_attn.k_proj", "self_attn.q_proj"]]
    assert parse_group_spec("") == []
    assert parse_group_spec(None) == []


def test_hf_style_grouping():
    names = [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp.gate_proj",
        "model.layers.1.mlp.down_proj",
        "model.layers.1.self_attn.q_proj",
    ]
    patterns = [["mlp.gate_proj", "mlp.down_proj"], ["self_attn.q_proj"]]
    groups = build_layer_groups(names, patterns)

    mapping = get_layer_to_group_mapping(groups)
    # gate_proj and down_proj in the same block share a group
    assert mapping["model.layers.0.mlp.gate_proj"] == mapping["model.layers.0.mlp.down_proj"]
    assert mapping["model.layers.1.mlp.gate_proj"] == mapping["model.layers.1.mlp.down_proj"]
    # but different blocks do not
    assert mapping["model.layers.0.mlp.gate_proj"] != mapping["model.layers.1.mlp.gate_proj"]
    # q_proj is its own group
    assert mapping["model.layers.0.self_attn.q_proj"] != mapping["model.layers.0.mlp.gate_proj"]


def test_mlx_style_grouping_uses_a_different_block_pattern():
    """MLX names modules ``backbone.layers.N.mixer.*`` -- the same logic applies."""
    names = [
        "backbone.layers.0.mixer.q_proj",
        "backbone.layers.0.mixer.out_proj",
        "backbone.layers.1.mixer.q_proj",
        "backbone.layers.1.mixer.out_proj",
    ]
    # default pattern matches "layers.N." so both schemes work
    patterns = [["mixer.q_proj", "mixer.out_proj"]]
    groups = build_layer_groups(names, patterns)
    mapping = get_layer_to_group_mapping(groups)

    assert mapping["backbone.layers.0.mixer.q_proj"] == mapping["backbone.layers.0.mixer.out_proj"]
    assert mapping["backbone.layers.1.mixer.q_proj"] == mapping["backbone.layers.1.mixer.out_proj"]
    assert mapping["backbone.layers.0.mixer.q_proj"] != mapping["backbone.layers.1.mixer.q_proj"]


def test_ungrouped_layers_become_their_own_groups():
    names = ["model.layers.0.mlp.gate_proj", "embed_tokens", "lm_head"]
    patterns = [["mlp.gate_proj"]]
    groups = build_layer_groups(names, patterns)

    assert len(groups) == 3
    assert any(g.pattern == "embed_tokens" for g in groups)
    assert any(g.pattern == "lm_head" for g in groups)


def test_moe_per_expert_grouping():
    names = [
        "model.layers.0.mlp.experts.0.gate_proj",
        "model.layers.0.mlp.experts.0.up_proj",
        "model.layers.0.mlp.experts.0.down_proj",
        "model.layers.0.mlp.experts.1.gate_proj",
        "model.layers.0.mlp.experts.1.down_proj",
        "model.layers.0.self_attn.q_proj",
    ]
    groups = build_moe_per_expert_groups(names)

    mapping = get_layer_to_group_mapping(groups)
    # expert 0's three projections share a group
    assert mapping["model.layers.0.mlp.experts.0.gate_proj"] == mapping[
        "model.layers.0.mlp.experts.0.up_proj"
    ]
    # expert 1 is a separate group
    assert mapping["model.layers.0.mlp.experts.0.gate_proj"] != mapping[
        "model.layers.0.mlp.experts.1.gate_proj"
    ]
    # the attention layer is not an expert
    assert mapping["model.layers.0.self_attn.q_proj"] not in (
        mapping["model.layers.0.mlp.experts.0.gate_proj"],
        mapping["model.layers.0.mlp.experts.1.gate_proj"],
    )


def test_attach_param_counts_returns_fractions():
    names = ["a", "b", "c", "d"]
    groups = build_layer_groups(names)
    param_counts = {"a": 10, "b": 30, "c": 40, "d": 20}

    fracs = attach_param_counts(groups, param_counts)

    assert np.isclose(fracs.sum(), 1.0)
    assert np.allclose(fracs, [0.1, 0.3, 0.4, 0.2])
    assert groups[1].param_count == 30


def test_attach_param_counts_rejects_zero_total():
    groups = build_layer_groups(["a"])
    try:
        attach_param_counts(groups, {"a": 0})
    except ValueError as exc:
        assert "zero" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for a zero total")


def test_get_group_param_count_sums_members():
    from rco_mlx.grouping import LayerGroup

    group = LayerGroup(0, ["a", "b"], "pat")
    assert get_group_param_count(group, {"a": 10, "b": 5}) == 15

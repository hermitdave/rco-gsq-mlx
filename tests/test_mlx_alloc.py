"""MLX config emission tests.

These check the exact shape MLX's loader expects: top-level defaults plus
per-path entries, module paths without ``.weight``, and the mirrored
``quantization_config``.
"""

import json

import pytest

from rco_mlx.mlx_alloc import (
    MlxQuantConfig,
    allocation_to_config,
    read_config,
    recover_bits_from_packed_shape,
    write_config,
)


def test_config_shape_matches_mlx():
    cfg = MlxQuantConfig(default_bits=4, group_size=64)
    cfg.set("backbone.layers.1.mixer.switch_mlp.fc1", 2)
    cfg.set("backbone.embeddings", 8)

    block = cfg.to_dict()

    assert block["group_size"] == 64
    assert block["bits"] == 4
    assert block["mode"] == "affine"
    assert block["backbone.layers.1.mixer.switch_mlp.fc1"] == {
        "group_size": 64,
        "bits": 2,
        "mode": "affine",
    }
    assert block["backbone.embeddings"]["bits"] == 8


def test_paths_are_stripped_of_weight_suffix():
    cfg = MlxQuantConfig()
    cfg.set("backbone.layers.0.mixer.q_proj.weight", 6)
    assert "backbone.layers.0.mixer.q_proj" in cfg.paths
    assert "backbone.layers.0.mixer.q_proj.weight" not in cfg.paths


def test_quantization_config_mirrors_quantization():
    cfg = MlxQuantConfig()
    cfg.set("lm_head", 8)

    full = cfg.to_json()
    assert full["quantization"] == full["quantization_config"]


def test_bits_for_falls_back_to_the_default():
    cfg = MlxQuantConfig(default_bits=4)
    cfg.set("declared", 6)

    assert cfg.bits_for("declared") == 6
    assert cfg.bits_for("undeclared") == 4


def test_allocation_to_config_lists_every_module():
    assignment = {
        "backbone.layers.0.mixer.q_proj": 4,
        "backbone.layers.1.mixer.q_proj": 2,
        "backbone.embeddings": 8,
    }
    cfg = allocation_to_config(assignment, default_bits=4)

    # a module at the default width is still listed explicitly
    assert "backbone.layers.0.mixer.q_proj" in cfg.paths
    assert len(cfg) == 3


def test_rejects_unsupported_bits():
    cfg = MlxQuantConfig()
    with pytest.raises(ValueError, match="not supported by MLX"):
        cfg.set("x", 1)
    with pytest.raises(ValueError, match="not supported by MLX"):
        cfg.set("x", 7)


def test_rejects_unsupported_group_size():
    cfg = MlxQuantConfig()
    with pytest.raises(ValueError, match="not supported by MLX"):
        cfg.set("x", 4, group_size=16)


def test_write_and_read_round_trip(tmp_path):
    cfg = MlxQuantConfig(default_bits=6, group_size=128)
    cfg.set("backbone.layers.0.mixer.q_proj", 4)
    cfg.set("backbone.embeddings", 8)

    base = {"model_type": "qwen3_5", "num_hidden_layers": 36}
    path = tmp_path / "config.json"
    merged = write_config(cfg, path, base_config=base)

    assert merged["model_type"] == "qwen3_5"
    assert merged["quantization"]["bits"] == 6

    loaded = json.loads(path.read_text())
    assert loaded["quantization"] == loaded["quantization_config"]

    block = read_config(path)
    assert block["backbone.layers.0.mixer.q_proj"]["bits"] == 4


def test_recover_bits_from_packed_shape():
    # 64 packed cols over 1024 inputs -> 64*32/1024 = 2 bits
    assert recover_bits_from_packed_shape(64, 1024) == pytest.approx(2.0)
    assert recover_bits_from_packed_shape(128, 1024) == pytest.approx(4.0)
    assert recover_bits_from_packed_shape(256, 1024) == pytest.approx(8.0)


def test_recover_bits_rejects_bad_shapes():
    with pytest.raises(ValueError):
        recover_bits_from_packed_shape(0, 1024)
    with pytest.raises(ValueError):
        recover_bits_from_packed_shape(64, 0)

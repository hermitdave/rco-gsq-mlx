"""Tests for model_adapters."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rco_mlx.adapters import FlintAdapter, LFM2Adapter, get, register


def _write_fake_lfm2(root: Path):
    """Write a minimal LFM2 config + one shard."""
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
        "hidden_size": 2048,
        "intermediate_size": 10752,
        "num_hidden_layers": 6,
        "layer_types": ["conv", "conv", "full_attention", "conv", "conv", "full_attention"],
    }
    (root / "config.json").write_text(json.dumps(cfg))

    # Write a tiny safetensors shard
    try:
        import torch
        from safetensors.torch import save_file
        tensors = {
            "model.embed_tokens.weight": torch.zeros((1000, 2048), dtype=torch.float32),
            "model.embedding_norm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.layers.0.conv.conv.weight": torch.zeros((2048, 1, 3), dtype=torch.float32),
            "model.layers.0.conv.in_proj.weight": torch.zeros((6144, 2048), dtype=torch.float32),
            "model.layers.0.conv.out_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.0.feed_forward.w1.weight": torch.zeros((10752, 2048), dtype=torch.float32),
            "model.layers.0.feed_forward.w2.weight": torch.zeros((2048, 10752), dtype=torch.float32),
            "model.layers.0.feed_forward.w3.weight": torch.zeros((10752, 2048), dtype=torch.float32),
            "model.layers.0.ffn_norm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.layers.0.operator_norm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.layers.2.self_attn.q_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.2.self_attn.k_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.2.self_attn.v_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.2.self_attn.out_proj.weight": torch.zeros((2048, 2048), dtype=torch.float32),
            "model.layers.2.self_attn.q_layernorm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.layers.2.self_attn.k_layernorm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.layers.2.feed_forward.w1.weight": torch.zeros((10752, 2048), dtype=torch.float32),
            "model.layers.2.feed_forward.w2.weight": torch.zeros((2048, 10752), dtype=torch.float32),
            "model.layers.2.feed_forward.w3.weight": torch.zeros((10752, 2048), dtype=torch.float32),
            "model.layers.2.ffn_norm.weight": torch.zeros((2048,), dtype=torch.float32),
            "model.layers.2.operator_norm.weight": torch.zeros((2048,), dtype=torch.float32),
        }
        save_file(tensors, str(root / "model-00001-of-00001.safetensors"))
    except ImportError:
        pytest.skip("safetensors/torch not installed")


def test_lfm2_adapter_enumerate(tmp_path: Path):
    _write_fake_lfm2(tmp_path)
    adapter = LFM2Adapter()
    layers = adapter.enumerate_layers(tmp_path)
    assert len(layers) == 6
    assert layers[0]["type"] == "conv"
    assert layers[2]["type"] == "full_attention"
    assert layers[2]["param_count"] > 0


def test_lfm2_adapter_quant_predicate(tmp_path: Path):
    _write_fake_lfm2(tmp_path)
    adapter = LFM2Adapter()
    conv_layer = {"index": 0, "type": "conv", "weight_keys": [], "param_count": 0}
    attn_layer = {"index": 2, "type": "full_attention", "weight_keys": [], "param_count": 0}
    assert adapter.quant_predicate(conv_layer) == {"bits": 3, "group_size": 64}
    assert adapter.quant_predicate(attn_layer) == {"bits": 4, "group_size": 64}


def test_lfm2_adapter_calibration(tmp_path: Path):
    _write_fake_lfm2(tmp_path)
    adapter = LFM2Adapter()
    cal = adapter.load_calibration(tmp_path, n_samples=16, seq_len=128)
    assert cal.shape == (16, 128, 2048)
    assert cal.dtype == np.float32


def test_lfm2_adapter_grouping():
    adapter = LFM2Adapter()
    spec = adapter.grouping_spec()
    assert spec is not None
    assert "self_attn" in spec
    assert "feed_forward" in spec


def test_lfm2_adapter_key_map():
    adapter = LFM2Adapter()
    assert adapter.map_key_to_mlx("model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert adapter.map_key_to_mlx("model.layers.0.conv.in_proj.weight") == "model.layers.0.conv.in_proj.weight"
    assert adapter.map_key_to_mlx("model.visual.encoder") is None


def test_flint_adapter_registered():
    cls = get("flint")
    assert cls is FlintAdapter


def test_lfm2_adapter_registered():
    cls = get("lfm2")
    assert cls is LFM2Adapter


def test_unknown_adapter():
    with pytest.raises(KeyError):
        get("nonexistent_model_type")

#!/usr/bin/env python3
"""
LFM2.5 adapters (dense and MoE variants).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("lfm2")
class LFM2Adapter(ModelAdapter):
    """Adapter for LFM2.5 dense layers."""

    model_type = "lfm2"
    layer_type_key = "layer_types"
    default_group_size = 64
    default_bits = 4

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        layer_types = cfg.get("layer_types", [])
        hidden_size = cfg.get("hidden_size", 2048)

        shards = sorted(source_dir.glob("*.safetensors"))
        param_counts = {}
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            for key in f.keys():
                param_counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx, ltype in enumerate(layer_types):
            prefix = f"model.layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]
            layers.append({
                "index": idx,
                "type": ltype,
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })

        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        ltype = layer_descriptor["type"]

        if ltype in ("embedding_norm", "norm", "layernorm"):
            return None
        if ltype == "full_attention":
            return {"bits": 4, "group_size": self.default_group_size}
        if ltype == "conv":
            return {"bits": 3, "group_size": self.default_group_size}
        return {"bits": 3, "group_size": self.default_group_size}

    def grouping_spec(self) -> Optional[str]:
        return (
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj,"
            "self_attn.q_layernorm,self_attn.k_layernorm "
            "conv.conv,conv.in_proj,conv.out_proj "
            "feed_forward.w1,feed_forward.w2,feed_forward.w3 "
            "ffn_norm,operator_norm"
        )

    def load_calibration(
        self, source_dir: Path, n_samples: int = 512, seq_len: int = 2048
    ) -> np.ndarray:
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        if key.startswith("model.visual."):
            return None
        if key.startswith("model.language_model."):
            return "language_model.model." + key[len("model.language_model."):]
        return key


@register("lfm2_moe")
class LFM2MoEAdapter(ModelAdapter):
    """Adapter for LFM2.5-MoE layers."""

    model_type = "lfm2_moe"
    layer_type_key = "layer_types"
    default_group_size = 64
    default_bits = 4

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        layer_types = cfg.get("layer_types", [])
        hidden_size = cfg.get("hidden_size", 2048)
        num_experts = cfg.get("num_experts", 64)

        shards = sorted(source_dir.glob("*.safetensors"))
        param_counts = {}
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            for key in f.keys():
                param_counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx, ltype in enumerate(layer_types):
            prefix = f"model.layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]
            layers.append({
                "index": idx,
                "type": ltype,
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })

        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        ltype = layer_descriptor["type"]

        if ltype in ("embedding_norm", "norm", "layernorm"):
            return None
        if ltype == "full_attention":
            return {"bits": 4, "group_size": self.default_group_size}
        if ltype == "conv":
            return {
                "bits": 4,
                "group_size": self.default_group_size,
                "skip_conv1d": True,
            }
        return {"bits": 4, "group_size": self.default_group_size}

    def grouping_spec(self) -> Optional[str]:
        return (
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.out_proj "
            "self_attn.q_layernorm,self_attn.k_layernorm "
            "conv.conv,conv.in_proj,conv.out_proj "
            "feed_forward.gate_proj,feed_forward.up_proj,feed_forward.down_proj "
            "ffn_norm,operator_norm"
        )

    def non_quantized_prefixes(self) -> List[str]:
        return [
            "embed_tokens",
            "lm_head",
            "norm",
            "layernorm",
            "embedding_norm",
            "feed_forward.expert_bias",
            "conv.conv",
        ]

    def load_calibration(
        self, source_dir: Path, n_samples: int = 512, seq_len: int = 2048
    ) -> np.ndarray:
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        if key.startswith("model.visual."):
            return None

        if ".feed_forward.gate." in key or key.endswith(".feed_forward.expert_bias"):
            return None

        if ".ffn_norm." in key or ".operator_norm." in key:
            return None

        if ".switch_mlp." in key:
            mlx = key.replace(".switch_mlp.w1.", ".switch_mlp.gate_proj.")
            mlx = mlx.replace(".switch_mlp.w2.", ".switch_mlp.down_proj.")
            mlx = mlx.replace(".switch_mlp.w3.", ".switch_mlp.up_proj.")
            if ".feed_forward.switch_mlp." not in mlx:
                mlx = mlx.replace(".switch_mlp.", ".feed_forward.switch_mlp.")
            return mlx

        if ".feed_forward.experts." in key:
            return key

        if ".feed_forward.w1." in key:
            return key.replace(".feed_forward.w1.", ".feed_forward.gate_proj.")
        if ".feed_forward.w2." in key:
            return key.replace(".feed_forward.w2.", ".feed_forward.down_proj.")
        if ".feed_forward.w3." in key:
            return key.replace(".feed_forward.w3.", ".feed_forward.up_proj.")

        return key

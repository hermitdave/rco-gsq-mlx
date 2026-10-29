#!/usr/bin/env python3
"""
Flint adapter (hybrid MoE).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("flint")
class FlintAdapter(ModelAdapter):
    """Adapter for Flint hybrid MoE models."""

    model_type = "flint"
    layer_type_key = "layer_types"
    default_group_size = 64
    default_bits = 4

    _NUM_EXPERTS = 256
    _NUM_EXPERTS_PER_TOK = 6

    _BF16_PREFIXES = (
        "embed_tokens",
        "lm_head",
        "norm",
        "layernorm",
        "post_attention_layernorm",
        "softmax_offset",
        "q_norm",
        "k_norm",
    )

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        num_layers = cfg.get("num_hidden_layers", 49)
        layer_types = cfg.get("layer_types", [])
        mlp_only_layers = set(cfg.get("mlp_only_layers", []))
        no_rope_indices = set(cfg.get("no_rope_layer_indices", []))

        shards = sorted(source_dir.glob("*.safetensors"))
        param_counts: Dict[str, int] = {}
        if shards:
            with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
                for key in f.keys():
                    param_counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx in range(num_layers):
            prefix = f"model.layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]

            if idx < len(layer_types):
                ltype = layer_types[idx]
            elif idx in no_rope_indices:
                ltype = "full_attention"
            else:
                ltype = "sliding_attention"

            if idx in mlp_only_layers:
                ltype = "dense_mlp"

            layers.append({
                "index": idx,
                "type": ltype,
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })

        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        idx = layer_descriptor["index"]
        ltype = layer_descriptor["type"]

        if ltype == "dense_mlp":
            return {"bits": 4, "group_size": self.default_group_size}
        if ltype in ("full_attention", "sliding_attention"):
            return {"bits": 4, "group_size": self.default_group_size}
        return {"bits": 4, "group_size": self.default_group_size}

    def grouping_spec(self) -> Optional[str]:
        return (
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj "
            "mlp.gate_proj "
            "mlp.shared_expert.gate_proj,mlp.shared_expert.up_proj,mlp.shared_expert.down_proj "
            "mlp.experts"
        )

    def non_quantized_prefixes(self) -> List[str]:
        return list(self._BF16_PREFIXES) + [
            "mlp.gate_proj",
        ]

    def load_calibration(
        self,
        source_dir: Path,
        n_samples: int = 512,
        seq_len: int = 2048,
    ) -> np.ndarray:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)
        hidden_dim = cfg.get("hidden_size", 2048)
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        for prefix in self._BF16_PREFIXES:
            if f".{prefix}." in f".{key}." or key.startswith(f"{prefix}.") or key == prefix:
                return None

        if ".mlp.gate_proj." in key:
            return None

        if ".mlp.experts." in key:
            parts = key.split(".")
            new_parts = parts[:4] + ["switch_mlp"] + parts[5:]
            return ".".join(new_parts)

        if ".mlp.shared_experts." in key:
            return key.replace(".shared_experts.", ".shared_expert.")

        return key

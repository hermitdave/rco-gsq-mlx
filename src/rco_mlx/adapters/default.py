#!/usr/bin/env python3
"""
Default adapter for standard dense transformer models (Qwen, Llama, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from safetensors import safe_open

from .base import ModelAdapter, register


@register("default")
class DefaultAdapter(ModelAdapter):
    """Fallback adapter for standard dense transformer models.

    Used when no architecture-specific adapter is registered. Quantizes
    every nn.Linear except norms, embeddings, and lm_head.
    """

    model_type = "default"
    default_group_size = 64
    default_bits = 4

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)
        hidden_size = cfg.get("hidden_size", 2048)
        num_layers = cfg.get("num_hidden_layers", 0)

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
            layers.append({
                "index": idx,
                "type": "block",
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })

        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        return {"bits": self.default_bits, "group_size": self.default_group_size}

    def grouping_spec(self) -> Optional[str]:
        return None

    def non_quantized_prefixes(self) -> List[str]:
        return [
            "embed_tokens",
            "lm_head",
            "norm",
            "layernorm",
            "embedding_norm",
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
        # Translate HF safetensors keys to MLX module paths.
        # Ornith-1.5-9B uses a nested language_model wrapper:
        #   lm_head.weight                  -> language_model.lm_head.weight
        #   model.language_model.layers.0.  -> language_model.model.layers.0.
        #   model.language_model.embed_tokens -> language_model.model.embed_tokens
        if key == "lm_head.weight":
            return "language_model.lm_head.weight"
        if key.startswith("model.language_model."):
            return "language_model.model." + key[len("model.language_model."):]
        if key.startswith("language_model.model."):
            return key
        # MTP weights stay as-is (they may not be in the RTN model)
        if key.startswith("mtp."):
            return key
        return key

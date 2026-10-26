#!/usr/bin/env python3
"""
Model adapter for the GSQ + RCO quantization pipeline.

Each supported model gets a small adapter class that describes:
  - how to enumerate its layers and tensors
  - which tensors to quantize vs keep BF16
  - how to group layers for RCO allocation
  - how to load calibration data

The pipeline (pipeline_rco_mlx.py) calls the adapter; it does not need to
know which model it is running on.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from safetensors import safe_open

__all__ = [
    "ModelAdapter",
    "LFM2Adapter",
    "FlintAdapter",
    "register",
    "get",
]

_REGISTRY: Dict[str, type] = {}


def register(name: str):
    """Decorator to register a model adapter."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get(name: str) -> type:
    """Look up a registered adapter by model type name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"No adapter registered for '{name}'. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


class ModelAdapter(ABC):
    """Base class for model adapters."""

    # Subclasses set these
    model_type: str = ""
    layer_type_key: str = "layer_types"  # config key for per-layer type list
    default_group_size: int = 64
    default_bits: int = 4

    @abstractmethod
    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        """Return a list of layer descriptors.

        Each descriptor is a dict with at least:
            index      : int           # 0-based layer index
            type       : str           # e.g. "full_attention", "conv", "moe"
            weight_keys: List[str]     # tensor keys for this layer
            param_count: int           # total params in this layer
        """

    @abstractmethod
    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        """Return a quantization spec for this layer, or None to keep BF16.

        Return value: {"bits": int, "group_size": int} or None.
        """

    @abstractmethod
    def grouping_spec(self) -> Optional[str]:
        """Return a grouping spec string for RCO, or None for per-layer groups."""

    @abstractmethod
    def load_calibration(
        self,
        source_dir: Path,
        n_samples: int = 512,
        seq_len: int = 2048,
    ) -> np.ndarray:
        """Load calibration activations for GSQ training.

        Returns (n_samples, hidden_dim) float32 numpy array.
        """

    @abstractmethod
    def map_key_to_mlx(self, key: str) -> Optional[str]:
        """Map a source checkpoint key to MLX module path, or None to skip."""

    def build_layer_groups(self, layers: List[Dict]) -> List[Dict]:
        """Group layers for RCO allocation. Default: one group per layer."""
        return [
            {"group_id": i, "layer_indices": [l["index"]], "pattern": str(l["index"])}
            for i, l in enumerate(layers)
        ]

    def non_quantized_prefixes(self) -> List[str]:
        """Key prefixes that should always stay BF16."""
        return [
            "embed_tokens",
            "lm_head",
            "norm",
            "layernorm",
            "embedding_norm",
        ]


# ---------------------------------------------------------------------------
# LFM2.5 adapter
# ---------------------------------------------------------------------------

@register("lfm2")
class LFM2Adapter(ModelAdapter):
    model_type = "lfm2"
    layer_type_key = "layer_types"
    default_group_size = 64
    default_bits = 4

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        import json
        from safetensors import safe_open

        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        layer_types = cfg.get("layer_types", [])
        hidden_size = cfg.get("hidden_size", 2048)

        # Read first shard to get weight keys and shapes
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

        # Embeddings and norms stay BF16
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
        """Load random calibration data (no tokenizer needed for shape)."""
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)
        hidden_dim = cfg.get("hidden_size", 2048)
        return np.random.randn(n_samples, seq_len, hidden_dim).astype(np.float32)

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        if key.startswith("model.visual."):
            return None
        if key.startswith("model.language_model."):
            return "language_model.model." + key[len("model.language_model."):]
        return key


# ---------------------------------------------------------------------------
# Flint adapter (hybrid MoE)
# ---------------------------------------------------------------------------

@register("flint")
class FlintAdapter(ModelAdapter):
    model_type = "flint"
    layer_type_key = "layer_types"
    default_group_size = 64
    default_bits = 4

    # Flint-specific constants from config
    _NUM_EXPERTS = 256
    _NUM_EXPERTS_PER_TOK = 6

    # Keys that should always stay BF16
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

        # Read first shard to discover weight keys
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

            # Determine layer type from config
            if idx < len(layer_types):
                ltype = layer_types[idx]
            elif idx in no_rope_indices:
                ltype = "full_attention"
            else:
                ltype = "sliding_attention"

            # Override: mlp_only_layers are dense (no attention weights)
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

        # mlp_only_layers: dense MLP, no attention
        if ltype == "dense_mlp":
            return {"bits": 4, "group_size": self.default_group_size}

        # Attention components
        if ltype in ("full_attention", "sliding_attention"):
            return {"bits": 4, "group_size": self.default_group_size}

        # Fallback
        return {"bits": 4, "group_size": self.default_group_size}

    def grouping_spec(self) -> Optional[str]:
        return (
            # Attention projections grouped together
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj "
            # MoE router (critical, keep high precision)
            "mlp.gate_proj "
            # Shared expert
            "mlp.shared_expert.gate_proj,mlp.shared_expert.up_proj,mlp.shared_expert.down_proj "
            # MoE routed experts (256 experts, stacked in MLX as one tensor)
            "mlp.experts"
        )

    def non_quantized_prefixes(self) -> List[str]:
        return list(self._BF16_PREFIXES) + [
            "mlp.gate_proj",  # router — keep BF16 / high precision
        ]

    def load_calibration(
        self,
        source_dir: Path,
        n_samples: int = 512,
        seq_len: int = 2048,
    ) -> np.ndarray:
        """Load calibration data.

        For GSQ training we need (n_samples, seq_len, hidden_dim) activations.
        Without a working tokenizer/datasets pipeline, we generate random
        hidden states matching the model's input shape. This is a stand-in
        until real calibration data can be collected.
        """
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)
        hidden_dim = cfg.get("hidden_size", 2048)
        return np.random.randn(n_samples, seq_len, hidden_dim).astype(np.float32)

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        # Skip BF16 prefixes — handle both top-level (model.embed_tokens)
        # and nested (model.layers.0.mlp.gate) by matching path components
        for prefix in self._BF16_PREFIXES:
            if f".{prefix}." in f".{key}." or key.startswith(f"{prefix}.") or key == prefix:
                return None

        # Skip router gate — we keep it BF16 / high precision
        if ".mlp.gate_proj." in key:
            return None

        # Stack MoE experts into a single MLX tensor path
        # HF: model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight
        # MLX: model.layers.{l}.mlp.experts.{gate,up,down}_proj.weight
        # (experts are stacked along dim 0 in MLX)
        if ".mlp.experts." in key:
            # Strip the expert index
            parts = key.split(".")
            # model.layers.{l}.mlp.experts.{e}.{proj}.weight
            new_parts = parts[:4] + parts[5:]  # remove experts.{e}
            return ".".join(new_parts)

        # Shared expert: rename shared_experts → shared_expert (singular)
        if ".mlp.shared_experts." in key:
            return key.replace(".shared_experts.", ".shared_expert.")

        # Everything else maps through unchanged
        return key


# ---------------------------------------------------------------------------
# OLMoE adapter (AllenAI, 1B active / 7B total)
# ---------------------------------------------------------------------------

@register("olmoe")
class OLMoEAdapter(ModelAdapter):
    """Adapter for OLMoE-1B-7B (model_type: olmoe).

    Architecture: 16 identical MoE layers, 64 routed experts, top-8 routing.
    No shared expert, no sliding window, no MTP.
    Router is ``mlp.gate`` (not SwiGLU's ``gate_proj``).
    """

    model_type = "olmoe"
    default_group_size = 64
    default_bits = 4

    # Keys that must stay BF16 regardless of RCO assignment
    _BF16_PREFIXES = (
        "embed_tokens",
        "lm_head",
        "norm",           # model.norm (final RMSNorm)
        "layernorm",      # input_layernorm, post_attention_layernorm
        "q_norm",         # per-head Q RMSNorm
        "k_norm",         # per-head K RMSNorm
        "mlp.gate",       # router logits — extremely sensitive, must stay BF16
    )

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        num_layers = cfg.get("num_hidden_layers", 16)
        num_experts = cfg.get("num_experts", 64)

        # Read first shard for weight keys + shapes
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
                "type": "moe",
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })

        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        # All MoE layers: quantize attention + experts, keep norms/router BF16
        return {
            "bits": self.default_bits,
            "group_size": self.default_group_size,
        }

    def grouping_spec(self) -> Optional[str]:
        return (
            # Attention projections share a group
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
            # All experts for a given projection share a group
            " model.layers.N.mlp.experts.N.up_proj"
            " model.layers.N.mlp.experts.N.down_proj"
            " model.layers.N.mlp.experts.N.gate_proj"
        )

    def non_quantized_prefixes(self) -> List[str]:
        return list(self._BF16_PREFIXES)

    def load_calibration(
        self,
        source_dir: Path,
        n_samples: int = 512,
        seq_len: int = 2048,
    ) -> np.ndarray:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)
        hidden_dim = cfg.get("hidden_size", 2048)
        return np.random.randn(n_samples, seq_len, hidden_dim).astype(np.float32)

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        # Skip BF16 prefixes
        for prefix in self._BF16_PREFIXES:
            if key.startswith(f"model.{prefix}.") or key == f"model.{prefix}":
                return None

        # Experts: keep the experts.N path so the pipeline's _stack_experts
        # post-processing can stack them into switch_mlp tensors.
        # MLX-LM's olmoe loader also accepts both formats.
        if ".mlp.experts." in key:
            return key

        # Everything else (attention projs) maps through unchanged
        return key

#!/usr/bin/env python3
"""
Model adapter for the RCO quantization pipeline.

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
    "LFM2MoEAdapter",
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
        """Load calibration activations for error measurement.

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
        """Calibration must come from a real text dataset, not random noise."""
        raise RuntimeError(
            "DefaultAdapter does not generate synthetic calibration. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        if key.startswith("model.visual."):
            return None
        if key.startswith("model.language_model."):
            return "language_model.model." + key[len("model.language_model."):]
        return key


# ---------------------------------------------------------------------------
# LFM2.5-MoE adapter
# ---------------------------------------------------------------------------

@register("lfm2_moe")
class LFM2MoEAdapter(ModelAdapter):
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
            # ShortConv has Conv1d (conv.conv) + Linear (conv.in_proj, conv.out_proj)
            # MLX affine quant only supports Linear; skip Conv1d
            return {"bits": 4, "group_size": self.default_group_size, "skip_conv1d": True}
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
            "feed_forward.gate",
            "feed_forward.expert_bias",
            "conv.conv",  # Conv1d not supported by MLX affine quant
        ]

    def load_calibration(
        self, source_dir: Path, n_samples: int = 512, seq_len: int = 2048
    ) -> np.ndarray:
        """Calibration must come from a real text dataset, not random noise."""
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        if key.startswith("model.visual."):
            return None

        # Router + expert bias stay BF16
        if ".feed_forward.gate." in key or key.endswith(".feed_forward.expert_bias"):
            return None

        # Norms stay BF16
        if ".ffn_norm." in key or ".operator_norm." in key:
            return None

        # Stacked MoE experts (produced by _stack_experts)
        # Config may have keys with or without the feed_forward prefix:
        #   model.layers.{l}.switch_mlp.{w1,w2,w3}
        #   model.layers.{l}.feed_forward.switch_mlp.{w1,w2,w3}
        # MLX-LM expects: model.layers.{l}.feed_forward.switch_mlp.{gate,down,up}_proj
        if ".switch_mlp." in key:
            mlx = key.replace(".switch_mlp.w1.", ".switch_mlp.gate_proj.")
            mlx = mlx.replace(".switch_mlp.w2.", ".switch_mlp.down_proj.")
            mlx = mlx.replace(".switch_mlp.w3.", ".switch_mlp.up_proj.")
            # Ensure feed_forward prefix is present
            if ".feed_forward.switch_mlp." not in mlx:
                mlx = mlx.replace(".switch_mlp.", ".feed_forward.switch_mlp.")
            return mlx

        # Individual MoE experts: keep original path so MLX-LM's sanitize() can stack them
        # MLX-LM expects: model.layers.{l}.feed_forward.experts.{e}.{proj}.weight
        # and stacks them into switch_mlp.{proj}.weight during sanitize
        if ".feed_forward.experts." in key:
            return key

        # Dense FFN renames (w1/w2/w3 -> gate/up/down)
        if ".feed_forward.w1." in key:
            return key.replace(".feed_forward.w1.", ".feed_forward.gate_proj.")
        if ".feed_forward.w2." in key:
            return key.replace(".feed_forward.w2.", ".feed_forward.down_proj.")
        if ".feed_forward.w3." in key:
            return key.replace(".feed_forward.w3.", ".feed_forward.up_proj.")

        return key


# ---------------------------------------------------------------------------
# Flint adapter (hybrid MoE)
# ---------------------------------------------------------------------------

@register("qwen3_5_moe")
class Qwen3NextMoEAdapter(ModelAdapter):
    """Adapter for Qwen3.5 MoE / Groot models (model_type: qwen3_5_moe).

    Architecture: 40 layers (10 full_attention + 30 linear_attention),
    128 routed experts + 1 shared, top-8 routing.
    Experts are stored as stacked SwitchLinear: weight shape (128, in, out).
    """

    model_type = "qwen3_5_moe"
    default_group_size = 64
    default_bits = 4

    # Keys that must stay BF16 regardless of RCO assignment
    _BF16_PREFIXES = (
        "embed_tokens",
        "lm_head",
        "norm",
        "layernorm",
        "embedding_norm",
        "mlp.gate",  # router logits
        "dt_bias",
        "A_log",
        "shared_expert_gate",
    )

    def enumerate_layers(self, source_dir: Path) -> List[Dict]:
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)

        # Groot puts model config under text_config
        text_cfg = cfg.get("text_config", cfg)
        num_layers = text_cfg.get("num_hidden_layers", 40)
        layer_types = text_cfg.get("layer_types", [])

        shards = sorted(source_dir.glob("*.safetensors"))
        param_counts: Dict[str, int] = {}
        # Read ALL shards for weight keys (later layers may be in later shards)
        for shard_path in shards:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.endswith(".weight"):
                        param_counts[key] = int(f.get_tensor(key).numel())

        layers = []
        for idx in range(num_layers):
            # Groot uses model.language_model.layers.{idx}.
            prefix = f"model.language_model.layers.{idx}."
            weight_keys = [k for k in param_counts if k.startswith(prefix)]
            layer_type = layer_types[idx] if idx < len(layer_types) else "block"
            layers.append({
                "index": idx,
                "type": layer_type,
                "weight_keys": weight_keys,
                "param_count": sum(param_counts[k] for k in weight_keys),
            })
        return layers

    def quant_predicate(self, layer_descriptor: Dict) -> Optional[Dict]:
        return {
            "bits": self.default_bits,
            "group_size": self.default_group_size,
        }

    def grouping_spec(self) -> Optional[str]:
        return (
            "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
            " model.layers.N.mlp.switch_mlp.up_proj"
            " model.layers.N.mlp.switch_mlp.down_proj"
            " model.layers.N.mlp.switch_mlp.gate_proj"
            " linear_attn.in_proj_qkv"
            " linear_attn.in_proj_z"
            " linear_attn.in_proj_b"
            " linear_attn.in_proj_a"
            " linear_attn.out_proj"
        )

    def non_quantized_prefixes(self) -> List[str]:
        return list(self._BF16_PREFIXES)

    def load_calibration(
        self, source_dir: Path, n_samples: int = 512, seq_len: int = 2048
    ) -> np.ndarray:
        """Calibration must come from a real text dataset, not random noise."""
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

    def map_key_to_mlx(self, key: str) -> Optional[str]:
        # Groot stores stacked experts as gate_up_proj + down_proj
        # MLX-LM expects switch_mlp.gate_proj, up_proj, down_proj
        if ".mlp.experts.down_proj." in key:
            return key.replace(".mlp.experts.down_proj.", ".mlp.switch_mlp.down_proj.")
        if ".mlp.experts.gate_up_proj." in key:
            # Map to gate_proj; up_proj is derived from the same tensor during load
            return key.replace(".mlp.experts.gate_up_proj.", ".mlp.switch_mlp.gate_proj.")
        # Shared expert
        if ".mlp.shared_expert." in key:
            mlx = key.replace(".mlp.shared_expert.gate_proj.", ".mlp.shared_expert.gate_proj.")
            mlx = mlx.replace(".mlp.shared_expert.up_proj.", ".mlp.shared_expert.up_proj.")
            mlx = mlx.replace(".mlp.shared_expert.down_proj.", ".mlp.shared_expert.down_proj.")
            return mlx
        # Linear attention
        if ".linear_attn." in key:
            return key
        # Self attention
        if ".self_attn." in key:
            return key
        return key


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

        For error measurement we need (n_samples, seq_len, hidden_dim) activations.
        Without a working tokenizer/datasets pipeline, we generate random
        hidden states matching the model's input shape. This is a stand-in
        until real calibration data can be collected.
        """
        with open(source_dir / "config.json") as f:
            cfg = json.load(f)
        hidden_dim = cfg.get("hidden_size", 2048)
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

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
        raise RuntimeError(
            "Calibration requires a real dataset. "
            "Use --calibration-dataset fineweb|wikitext|c4, or pass --calibration <npy>."
        )

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


# ---------------------------------------------------------------------------
# Default adapter (standard dense models: Qwen, Llama, etc.)
# ---------------------------------------------------------------------------

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

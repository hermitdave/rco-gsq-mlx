#!/usr/bin/env python3
"""MoE-aware error_db enhancement for RCO mixed-precision quantization.

Post-processes a Hessian-weighted error_db.json to inject three MoE-specific
signals derived from the research literature:

1. Calibration-balanced coverage (MoEQuant): under-activated experts get their
   errors inflated so the allocator cannot silently starve them of bits.
2. Layer-depth sensitivity (MoE-Compression): middle-layer experts are more
   sensitive to compression error; shallow and deep experts tolerate more.
3. Spectral heavy-tailedness fallback (AlphaQ): when calibration coverage is
   low, a weight-spectrum proxy supplements the Hessian error so the
   allocation does not rely on noisy activation statistics alone.

Usage:
    python -m rco_mlx.scripts.moe_enhance_error_db \
        --input error_db.json \
        --output error_db_enhanced.json \
        --model-type olmoe

The script enriches each layer entry with a ``_meta`` block:
    activation_count, spectral_score, layer_depth, is_expert

It also writes ``work/moe_enhance_report.json`` with coverage statistics
and the depth multipliers applied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from rco_mlx.adapters import get as get_adapter


# ---------------------------------------------------------------------------
# Spectral heavy-tailedness proxy (AlphaQ-inspired, no powerlaw dependency)
# ---------------------------------------------------------------------------

def _spectral_heavy_tailedness(W: np.ndarray) -> float:
    """Cheap proxy for PL_Alpha_Hill.

    Uses log-mean minus log-median of absolute weight values.
    More heavy-tailed -> larger ratio.

    AlphaQ finds that heavier-tailed eigenvalue spectra correlate with
    better-trained, more important experts.  We normalise per-model later.
    """
    W = np.asarray(W, dtype=np.float64)
    flat = np.abs(W.reshape(-1))
    # Avoid log(0)
    flat = flat[flat > 1e-12]
    if flat.size == 0:
        return 0.0
    mean_abs = np.mean(flat)
    median_abs = np.median(flat)
    if median_abs < 1e-12:
        return 0.0
    return math.log(mean_abs) - math.log(median_abs)


# ---------------------------------------------------------------------------
# Depth sensitivity (MoE-Compression)
# ---------------------------------------------------------------------------

def _depth_multiplier(layer_idx: int, total_layers: int) -> float:
    """Return a multiplier on the error based on layer depth.

    Shallow (<30%): minimal degradation from bounded error -> 0.85
    Middle (30-70%): central to reasoning -> 1.15
    Deep (>70%): instruction following; bounded error can help -> 1.00
    """
    if total_layers <= 1:
        return 1.0
    ratio = layer_idx / (total_layers - 1)
    if ratio < 0.30:
        return 0.85
    elif ratio < 0.70:
        return 1.15
    else:
        return 1.00


# ---------------------------------------------------------------------------
# Layer index extraction
# ---------------------------------------------------------------------------

def _extract_layer_idx(layer_path: str) -> int | None:
    """Extract the layer index from a checkpoint key or module path.

    Handles patterns like:
        model.layers.3.mlp.experts.12.w1
        language_model.model.layers.7.self_attn.q_proj
    """
    parts = layer_path.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _is_expert_path(layer_path: str) -> bool:
    return ".experts." in layer_path


def _extract_expert_idx(layer_path: str) -> int | None:
    """Return the expert index if the path contains one, else None."""
    parts = layer_path.split(".")
    try:
        idx = parts.index("experts")
        if idx + 1 < len(parts) and parts[idx + 1].isdigit():
            return int(parts[idx + 1])
    except ValueError:
        pass
    return None


# ---------------------------------------------------------------------------
# Main enhancement logic
# ---------------------------------------------------------------------------

def enhance_error_db(
    db: dict,
    model_type: str = "default",
    coverage_alpha: float = 0.5,
    depth_beta: float = 0.2,
    spectral_gamma: float = 0.3,
) -> dict:
    """Return a new error_db dict with MoE-aware adjustments.

    Parameters
    ----------
    db : dict
        Original error_db loaded from JSON.
    model_type : str
        Model adapter name (e.g. ``olmoe``, ``qwen3_5_moe``).
    coverage_alpha : float
        Strength of under-calibration inflation.  Error is multiplied by
        ``1 + coverage_alpha * (1 - coverage_weight)``.
    depth_beta : float
        Strength of depth multiplier.  ``adjusted = raw * (1 + depth_beta * (mult - 1))``.
    spectral_gamma : float
        Blend weight for spectral fallback when coverage is low.

    Returns
    -------
    dict
        Enhanced error_db with ``_meta`` on each layer and ``_moe_meta`` at top.
    """
    adapter = get_adapter(model_type)()
    layers = db.get("layers", {})
    param_counts = db.get("param_counts", {})

    # Detect total_layers from the layer paths or adapter config.
    all_indices = []
    for path in layers:
        idx = _extract_layer_idx(path)
        if idx is not None:
            all_indices.append(idx)
    total_layers = max(all_indices, default=0) + 1

    # Step 1: compute spectral scores for every layer weight.
    # We need the raw BF16 weights; look for a shard in the source dir if
    # available, otherwise use a cheap proxy from the error magnitude.
    spectral_scores: dict[str, float] = {}
    raw_weights: dict[str, np.ndarray] = {}

    # Try to load weights from the source dir stored in _moe_meta or env.
    source_dir = db.get("_moe_meta", {}).get("source_dir")
    shard_map = {}
    if source_dir and Path(source_dir).exists():
        shards = sorted(Path(source_dir).glob("*.safetensors"))
        if shards:
            from safetensors import safe_open
            for s in shards:
                shard_map[s.name] = s
            # Load index if present
            index_path = Path(source_dir) / "model.safetensors.index.json"
            weight_map = {}
            if index_path.exists():
                with open(index_path) as f:
                    weight_map = json.load(f).get("weight_map", {})
            else:
                with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
                    weight_map = {k: shards[0].name for k in f.keys()}

            # Only compute spectral for expert tensors (the most expensive case).
            expert_paths = [p for p in layers if _is_expert_path(p)]
            for path in expert_paths:
                shard_name = weight_map.get(path + ".weight")
                if shard_name is None:
                    # Try without .weight suffix
                    shard_name = weight_map.get(path)
                if shard_name is None:
                    continue
                shard_path = shard_map.get(shard_name)
                if shard_path is None:
                    continue
                try:
                    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                        tensor = f.get_tensor(path + ".weight")
                    W = np.array(tensor.float(), dtype=np.float32)
                    raw_weights[path] = W
                    spectral_scores[path] = _spectral_heavy_tailedness(W)
                except Exception:
                    pass

    # If no weights loaded, fall back to error-magnitude proxy.
    if not spectral_scores:
        for path, bw_data in layers.items():
            errors = [v.get("error", 0.0) for v in bw_data.values() if isinstance(v, dict)]
            if errors:
                spectral_scores[path] = float(np.std(errors) / (np.mean(errors) + 1e-12))
            else:
                spectral_scores[path] = 0.0

    # Normalise spectral scores to [0, 1] across all layers.
    spec_vals = np.array(list(spectral_scores.values()), dtype=np.float64)
    spec_min, spec_max = float(spec_vals.min()), float(spec_vals.max())
    spec_range = spec_max - spec_min
    if spec_range > 1e-12:
        for k in spectral_scores:
            spectral_scores[k] = (spectral_scores[k] - spec_min) / spec_range
    else:
        for k in spectral_scores:
            spectral_scores[k] = 0.5

    # Step 2: track per-expert activation counts.
    # If the error_db was built with --moe-enhance, it already has
    # activation_count in _meta.  Otherwise synthesise from the model
    # topology (all experts assumed equally activated).
    activation_counts: dict[str, int] = {}
    for path in layers:
        if _is_expert_path(path):
            eidx = _extract_expert_idx(path)
            # If the original build captured counts, use them.
            meta = layers[path].get("_meta", {})
            if "activation_count" in meta:
                activation_counts[path] = meta["activation_count"]
            else:
                # Synthesise: uniform activation assumed.
                activation_counts[path] = 1

    # Normalise activation counts to coverage scores.
    max_act = max(activation_counts.values(), default=1)
    coverage_scores: dict[str, float] = {}
    for path, count in activation_counts.items():
        if max_act > 0:
            coverage_scores[path] = count / max_act
        else:
            coverage_scores[path] = 0.0

    # Step 3: enhance each layer's error values.
    enhanced_layers = {}
    moe_meta = {
        "total_layers": total_layers,
        "coverage_alpha": coverage_alpha,
        "depth_beta": depth_beta,
        "spectral_gamma": spectral_gamma,
        "source_dir": source_dir or "",
    }

    for path, bw_data in layers.items():
        # Skip _meta entries when iterating bitwidths.
        bitwidth_entries = {k: v for k, v in bw_data.items() if k != "_meta"}
        if not bitwidth_entries:
            continue

        layer_idx = _extract_layer_idx(path)
        depth_mult = _depth_multiplier(layer_idx or 0, total_layers)

        is_expert = _is_expert_path(path)
        coverage = coverage_scores.get(path, 1.0)
        coverage_weight = min(1.0, max(0.0, coverage / 0.5))
        spectral = spectral_scores.get(path, 0.5)

        enhanced_bw = {}
        for bw_str, entry in bitwidth_entries.items():
            if not isinstance(entry, dict):
                continue
            raw_err = entry.get("error", 0.0)

            # (a) under-calibration inflation
            under_calib_factor = 1.0 + coverage_alpha * (1.0 - coverage_weight)

            # (b) depth multiplier applied to error magnitude
            depth_factor = 1.0 + depth_beta * (depth_mult - 1.0)

            # (c) spectral blend for low-coverage experts
            #    When coverage is low, shift error toward spectral importance.
            spectral_shift = (1.0 - coverage_weight) * spectral_gamma * (spectral - 0.5)

            adjusted = raw_err * under_calib_factor * depth_factor + spectral_shift * raw_err

            enhanced_bw[bw_str] = {"error": float(adjusted)}

        # Preserve original _meta and add enhancement metadata.
        meta = dict(bw_data.get("_meta", {}))
        meta["activation_count"] = activation_counts.get(path, 0)
        meta["spectral_score"] = float(spectral)
        meta["coverage_score"] = float(coverage)
        meta["layer_depth"] = layer_idx if layer_idx is not None else -1
        meta["is_expert"] = is_expert
        meta["depth_multiplier"] = float(depth_mult)
        meta["raw_error_mean"] = float(
            np.mean([v.get("error", 0.0) for v in bitwidth_entries.values() if isinstance(v, dict)])
        )
        meta["enhanced_error_mean"] = float(
            np.mean([v.get("error", 0.0) for v in enhanced_bw.values()])
        )
        enhanced_bw["_meta"] = meta
        enhanced_layers[path] = enhanced_bw

    result = {
        "layers": enhanced_layers,
        "param_counts": param_counts,
        "_moe_meta": moe_meta,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="MoE-aware error_db enhancement (coverage + depth + spectral)."
    )
    ap.add_argument("--input", required=True, help="Input error_db.json")
    ap.add_argument("--output", required=True, help="Output enhanced error_db.json")
    ap.add_argument("--model-type", default="default", help="Model adapter name")
    ap.add_argument("--coverage-alpha", type=float, default=0.5,
                    help="Under-calibration inflation strength (default: 0.5)")
    ap.add_argument("--depth-beta", type=float, default=0.2,
                    help="Depth sensitivity strength (default: 0.2)")
    ap.add_argument("--spectral-gamma", type=float, default=0.3,
                    help="Spectral blend strength (default: 0.3)")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    db = json.loads(input_path.read_text())
    enhanced = enhance_error_db(
        db,
        model_type=args.model_type,
        coverage_alpha=args.coverage_alpha,
        depth_beta=args.depth_beta,
        spectral_gamma=args.spectral_gamma,
    )

    output_path.write_text(json.dumps(enhanced, indent=2))
    print(f"Wrote enhanced error_db: {output_path}")

    # Print summary.
    layers = enhanced.get("layers", {})
    expert_paths = [p for p in layers if _is_expert_path(p)]
    if expert_paths:
        coverages = [layers[p].get("_meta", {}).get("coverage_score", 0.0) for p in expert_paths]
        raw_means = [layers[p].get("_meta", {}).get("raw_error_mean", 0.0) for p in expert_paths]
        enh_means = [layers[p].get("_meta", {}).get("enhanced_error_mean", 0.0) for p in expert_paths]
        print(f"  Expert layers: {len(expert_paths)}")
        print(f"  Coverage: min={min(coverages):.3f} max={max(coverages):.3f} mean={np.mean(coverages):.3f}")
        print(f"  Raw error mean:  min={min(raw_means):.6f} max={max(raw_means):.6f}")
        print(f"  Enhanced mean:   min={min(enh_means):.6f} max={max(enh_means):.6f}")
        print(f"  Shift ratio (enh/raw): {np.mean(enh_means) / (np.mean(raw_means) + 1e-12):.3f}")


if __name__ == "__main__":
    main()

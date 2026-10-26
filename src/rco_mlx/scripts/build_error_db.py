#!/usr/bin/env python3
"""Build a SLQ-style error_db.json from a BF16 safetensors checkpoint.

For each quantizable linear layer, quantize at each bitwidth in {2,3,4,6,8}
using MLX affine, then measure relative squared Frobenius error:

    error = ||W - Q(W)||² / ||W||²

The result is an error_db.json compatible with rco-mlx's pipeline.

Usage:
    python -m rco_mlx.scripts.build_error_db \\
        --source ~/Projects/rco-mlx/.cache/olmoe-1b-7b-0125 \\
        --output ~/Projects/rco-mlx/.cache/olmoe-test-output/error_db.json \\
        --model-type olmoe \\
        --bitwidths 2,3,4,6,8 \\
        --group-size 64

Port and MLX integration by **Hermit Dave**, produced with **Hermes Agent**
(Nous Research).
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open


def get_adapter(model_type: str):
    from rco_mlx.model_adapters import get
    return get(model_type)()


def is_quantizable(adapter, layer_path: str) -> bool:
    """Check if a layer path should be quantized."""
    for prefix in adapter.non_quantized_prefixes():
        if f".{prefix}." in f".{layer_path}." or layer_path.startswith(f"{prefix}.") or layer_path == prefix:
            return False
    return True


def quantize_and_reconstruct(weight: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """Quantize with MLX affine and dequantize back to float32."""
    w = mx.array(weight, dtype=mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q, scales, biases)
    reconstructed = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
    mx.eval(reconstructed)
    return np.array(reconstructed)


def compute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Relative squared Frobenius error."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    norm_diff = np.sum(diff * diff)
    norm_orig = np.sum(original.astype(np.float64) * original.astype(np.float64))
    return float(norm_diff / max(norm_orig, 1e-12))


def main():
    ap = argparse.ArgumentParser(description="Build SLQ-style error_db.json from BF16 checkpoint")
    ap.add_argument("--source", required=True, help="BF16 safetensors directory")
    ap.add_argument("--output", required=True, help="Output error_db.json path")
    ap.add_argument("--model-type", required=True, help="Model type for adapter")
    ap.add_argument("--bitwidths", default="2,3,4,6,8", help="Bitwidths to sweep")
    ap.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    args = ap.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(args.model_type)
    bitwidths = [int(b) for b in args.bitwidths.split(",")]

    # Discover shards
    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        print(f"No safetensors in {source}")
        sys.exit(1)

    # Build filename -> Path lookup
    shard_map = {s.name: s for s in shards}

    # Read weight keys from index or first shard
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        all_keys = sorted(weight_map.keys())
    else:
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            all_keys = sorted(f.keys())
        weight_map = {k: shards[0].name for k in all_keys}

    # Filter to quantizable .weight keys
    quant_keys = []
    for k in all_keys:
        if not k.endswith(".weight"):
            continue
        base = k[:-7]
        if is_quantizable(adapter, base):
            quant_keys.append((base, weight_map[k]))

    print(f"Quantizable tensors: {len(quant_keys)}")

    # Build error database
    db = {"layers": {}, "param_counts": {}}
    total = len(quant_keys)
    all_errors = []

    for idx, (base, shard_name) in enumerate(quant_keys):
        shard_path = shard_map.get(shard_name)
        if shard_path is None:
            print(f"  SKIP {base}: shard {shard_name} not found")
            continue

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(base + ".weight").float().numpy()

        if tensor.ndim != 2:
            print(f"  SKIP {base}: not 2D ({tensor.ndim}D)")
            continue

        param_count = int(tensor.size)
        db["param_counts"][base] = param_count

        layer_errors = {}
        for bw in bitwidths:
            try:
                reconstructed = quantize_and_reconstruct(tensor, bw, args.group_size)
                err = compute_error(tensor, reconstructed)
                layer_errors[str(bw)] = {"error": err}
                all_errors.append(err)
            except Exception as e:
                print(f"  ERROR {base} @ {bw}-bit: {e}")
                continue

        if layer_errors:
            db["layers"][base] = layer_errors

        if (idx + 1) % 100 == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] {base} — "
                  f"errors: " + ", ".join(f"{bw}={layer_errors[str(bw)]['error']:.6f}" 
                                          for bw in bitwidths if str(bw) in layer_errors))

    output.write_text(json.dumps(db, indent=2))
    print(f"\nWrote error_db.json: {len(db['layers'])} layers, {len(all_errors)} measurements")
    if all_errors:
        print(f"Error range: [{min(all_errors):.6f}, {max(all_errors):.6f}], mean={np.mean(all_errors):.6f}")


if __name__ == "__main__":
    main()

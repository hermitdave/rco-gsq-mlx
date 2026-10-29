#!/usr/bin/env python3
"""
Streaming / layer-by-layer Hessian error_db builder.

Two-phase approach:
  Phase 1: Load a small RTN quant (8-bit) to capture per-layer activations.
  Phase 2: Load BF16 weights one tensor at a time from safetensors and
           measure quant error against the pre-captured Hessian diagonals.

This avoids loading the full BF16 model into memory, making it feasible
for very large models on memory-constrained hardware.

Usage:
    python -m rco_mlx.scripts.build_error_db_streaming \\
        --source ~/Projects/ml/models/lfm25-24b-a2b-bf16 \\
        --rtn-source ~/Projects/ml/models/lfm25-24b-a2b-rtn-8bit \\
        --output error_db.json \\
        --model-type lfm2_moe \\
        --calibration /tmp/lfm25_calib.npy \\
        --bitwidths 2,3,4,6,8 \\
        --group-size 64
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from rco_mlx.adapters import get


# ---------------------------------------------------------------------------
# Activation capture (same as build_error_db.py)
# ---------------------------------------------------------------------------
class _CaptureLinear(nn.Linear):
    def __call__(self, x):
        self._last_input = x
        return super().__call__(x)


class _CaptureQuantized(nn.QuantizedLinear):
    def __init__(self, source: nn.QuantizedLinear):
        super().__init__(
            input_dims=source.weight.shape[1],
            output_dims=source.weight.shape[0],
            bias="bias" in source,
            group_size=source.group_size,
            bits=source.bits,
            mode=source.mode,
        )
        for k in source.keys():
            self[k] = source[k]

    def __call__(self, x):
        self._last_input = x
        return super().__call__(x)


def wrap_linear_layers(model):
    wrapped = []
    for name, _ in model.named_modules():
        if not name:
            continue
        parts = [p for p in name.split(".") if p]
        obj = model
        for part in parts:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        if isinstance(obj, nn.Linear) and not isinstance(obj, _CaptureLinear):
            wrapper = _CaptureLinear(
                input_dims=obj.weight.shape[1],
                output_dims=obj.weight.shape[0],
                bias="bias" in obj,
            )
            wrapper.weight = obj.weight
            if "bias" in obj:
                wrapper.bias = obj.bias
            _set_module_by_path(model, name, wrapper)
            wrapped.append(name)
        elif isinstance(obj, nn.QuantizedLinear) and not isinstance(obj, _CaptureQuantized):
            wrapper = _CaptureQuantized(obj)
            _set_module_by_path(model, name, wrapper)
            wrapped.append(name)
    return wrapped


def _set_module_by_path(model, path: str, new_module):
    parts = [p for p in path.split(".") if p]
    obj = model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)


def _collect_activations(model, wrapped_paths, calibration_ids, batch_size=4):
    n_samples = calibration_ids.shape[0]
    accum = {p: None for p in wrapped_paths}
    total_tokens = 0

    for i in range(0, n_samples, batch_size):
        batch = calibration_ids[i : i + batch_size]
        input_ids = mx.array(batch)
        mx.eval(input_ids)
        logits = model(input_ids)
        mx.eval(logits)

        for path in wrapped_paths:
            module = _get_module_by_path(model, path)
            x = module._last_input
            if x is None:
                continue
            # Explicit copy to break any view that retains the parent buffer.
            # MLX slices are views; appending them keeps the full upstream
            # activation alive. mx.array() copies without the parent.
            x = mx.array(x)
            x_np = np.array(x.astype(mx.float32))
            ss = np.sum(x_np ** 2, axis=tuple(range(x_np.ndim - 1)))
            if accum[path] is None:
                accum[path] = ss
            else:
                accum[path] += ss
            total_tokens += x_np.shape[0] * (x_np.ndim - 1)

    hessian = {}
    for path, ss in accum.items():
        if ss is not None:
            hessian[path] = ss / total_tokens
    return hessian


def _get_module_by_path(model, path: str):
    parts = [p for p in path.split(".") if p]
    obj = model
    for part in parts:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Quant + error
# ---------------------------------------------------------------------------
def quantize_and_reconstruct(weight: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    w = mx.array(weight, dtype=mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q, scales, biases)
    reconstructed = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
    mx.eval(reconstructed)
    return np.array(reconstructed)


def compute_hessian_weighted_error(original: np.ndarray, reconstructed: np.ndarray, h_diag: np.ndarray) -> float:
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    if diff.ndim == 2:
        h = h_diag.reshape(1, -1)
    else:
        h = h_diag
    weighted_diff = np.sum(diff ** 2 * h)
    weighted_orig = np.sum(original.astype(np.float64) ** 2 * h)
    return float(weighted_diff / max(weighted_orig, 1e-12))


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------
def is_quantizable(adapter, layer_path: str) -> bool:
    for prefix in adapter.non_quantized_prefixes():
        if f".{prefix}." in f".{layer_path}." or layer_path.startswith(f"{prefix}.") or layer_path == prefix:
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Streaming layer-by-layer Hessian error_db builder")
    ap.add_argument("--source", required=True, help="BF16 safetensors directory (for weight measurement)")
    ap.add_argument("--rtn-source", required=True, help="RTN quant directory (for activation capture)")
    ap.add_argument("--output", required=True, help="Output error_db.json path")
    ap.add_argument("--model-type", default="default", help="Model type for adapter (default: 'default' for standard dense models)")
    ap.add_argument("--calibration", required=True, help="Path to calibration token IDs .npy")
    ap.add_argument("--bitwidths", default="2,3,4,6,8", help="Bitwidths to sweep")
    ap.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    ap.add_argument("--batch-size", type=int, default=4, help="Calibration batch size")
    args = ap.parse_args()

    source = Path(args.source).expanduser().resolve()
    rtn_source = Path(args.rtn_source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    adapter = get(args.model_type)()
    bitwidths = [int(b) for b in args.bitwidths.split(",")]

    # --- Phase 1: capture activations from RTN quant ---
    print("[phase 1] loading RTN quant for activation capture ...")
    from mlx_lm.utils import load_model
    model, _ = load_model(Path(str(rtn_source)), lazy=False, strict=False)
    model.eval()

    wrapped_paths = wrap_linear_layers(model)
    print(f"  Wrapped {len(wrapped_paths)} linear layers")

    calibration_ids = np.load(args.calibration)
    print(f"  Running calibration (batch_size={args.batch_size}) ...")
    hessian_map = _collect_activations(model, wrapped_paths, calibration_ids, batch_size=args.batch_size)
    print(f"  Captured Hessian for {len(hessian_map)} layers")

    # Free RTN model
    del model
    mx.eval(mx.zeros(1))  # flush

    # --- Phase 2: measure error per tensor from BF16 shards ---
    print("[phase 2] measuring per-tensor error from BF16 shards ...")

    shards = sorted(source.glob("*.safetensors"))
    shard_map = {s.name: s for s in shards}

    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        all_keys = sorted(weight_map.keys())
    else:
        with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
            all_keys = sorted(f.keys())
        weight_map = {k: shards[0].name for k in all_keys}

    quant_keys = []
    for k in all_keys:
        if not k.endswith(".weight"):
            continue
        base = k[:-7]
        if is_quantizable(adapter, base):
            quant_keys.append((base, weight_map[k]))

    print(f"  Quantizable tensors: {len(quant_keys)}")

    db = {"layers": {}, "param_counts": {}}
    total = len(quant_keys)

    for idx, (base, shard_name) in enumerate(quant_keys):
        shard_path = shard_map.get(shard_name)
        if shard_path is None:
            print(f"  SKIP {base}: shard not found")
            continue

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            tensor = f.get_tensor(base + ".weight").float().numpy()

        if tensor.ndim != 2:
            print(f"  SKIP {base}: not 2D ({tensor.ndim}D)")
            continue

        param_count = int(tensor.size)
        db["param_counts"][base] = param_count

        # Map base path to wrapped linear path for Hessian lookup
        # The wrapped path is the MLX module path from the RTN model
        # We need to translate BF16 key -> MLX key -> wrapped path
        mlx_key = adapter.map_key_to_mlx(base + ".weight")
        if mlx_key is None:
            print(f"  SKIP {base}: mapped to None (BF16 prefix)")
            continue

        # Strip .weight for module path lookup
        module_path = mlx_key[:-7] if mlx_key.endswith(".weight") else mlx_key

        # Try direct match first, then fall back to suffix match
        h_diag = hessian_map.get(module_path)
        if h_diag is None:
            # Try matching by last component
            for wrapped_path in hessian_map:
                if wrapped_path.endswith(module_path):
                    h_diag = hessian_map[wrapped_path]
                    break

        if h_diag is None:
            print(f"  SKIP {base}: no Hessian (looked up '{module_path}')")
            continue

        layer_errors = {}
        for bw in bitwidths:
            try:
                reconstructed = quantize_and_reconstruct(tensor, bw, args.group_size)
                err = compute_hessian_weighted_error(tensor, reconstructed, h_diag)
                layer_errors[str(bw)] = {"error": err}
            except Exception as e:
                print(f"  ERR {base} @ {bw}-bit: {e}")

        if layer_errors:
            db["layers"][base] = layer_errors

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [{idx+1}/{total}] {base}")

    output.write_text(json.dumps(db, indent=2))
    print(f"\nDone. Wrote {output}")
    print(f"  Layers with errors: {len(db['layers'])}")


if __name__ == "__main__":
    main()

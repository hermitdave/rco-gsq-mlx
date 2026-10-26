#!/usr/bin/env python3
"""
Build a Hessian-aware error_db.json from a BF16 safetensors checkpoint.

Runs calibration data through the model, captures per-layer input
activations, computes the empirical Fisher diagonal ``diag(X^T X) / N``,
and uses it to weight quantization error:

    error = (W - Q(W))^T H_diag (W - Q(W)) / (W^T H_diag W)

This is a closer approximation to GPTQ's sensitivity metric than the
plain Frobenius error used by ``build_error_db.py`` or the weight-norm
proxy used by ``build_error_db_gptq.py``.

Usage:
    python -m rco_mlx.scripts.build_error_db_hessian \\
        --source ~/Projects/rco-mlx/.cache/olmoe-1b-7b-0125 \\
        --output ~/Projects/rco-mlx/.cache/olmoe-error_db_hessian.json \\
        --model-type olmoe \\
        --calibration /tmp/olmoe_calib_input_ids.npy \\
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
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

from rco_mlx.model_adapters import get


# ---------------------------------------------------------------------------
# Activation-capturing linear layer
# ---------------------------------------------------------------------------
class _CaptureLinear(nn.Linear):
    """Drop-in replacement for nn.Linear that stores the last input."""

    def __call__(self, x):
        self._last_input = x
        return super().__call__(x)


# ---------------------------------------------------------------------------
# Model patching helpers
# ---------------------------------------------------------------------------
def _set_module_by_path(model, path: str, new_module):
    """Replace a module by dot-separated path, handling list indices."""
    parts = [p for p in path.split(".") if p]
    obj = model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = new_module
    else:
        setattr(obj, last, new_module)


def wrap_linear_layers(model):
    """Wrap every nn.Linear in the model to capture activations.

    Returns the list of module paths that were wrapped.
    """
    wrapped = []
    for name, _ in model.named_modules():
        # Resolve the current module to check its type
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
    return wrapped


def _get_module(model, path: str):
    parts = [p for p in path.split(".") if p]
    obj = model
    for part in parts:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Calibration + Hessian computation
# ---------------------------------------------------------------------------
def _collect_activations(model, wrapped_paths, calibration_ids, batch_size=4):
    """Run calibration data and collect per-layer activation statistics.

    For each wrapped linear layer we compute the diagonal of the empirical
    Fisher information matrix::

        H_diag[l] = mean(x^2, axis=0)   # shape: (in_features,)

    Returns:
        dict: layer_path -> hessian_diag (1-D numpy array)
    """
    n_samples = calibration_ids.shape[0]
    # Accumulators: path -> sum_of_squares, count
    accum = {p: None for p in wrapped_paths}
    total_tokens = 0

    for i in range(0, n_samples, batch_size):
        batch = calibration_ids[i : i + batch_size]
        input_ids = mx.array(batch)
        mx.eval(input_ids)
        # Forward pass — activations are captured inside each _CaptureLinear
        logits = model(input_ids)
        mx.eval(logits)

        for path in wrapped_paths:
            module = _get_module(model, path)
            x = module._last_input
            if x is None:
                continue
            # x shape: (batch, seq, in_features) or (batch, in_features)
            x_np = np.array(x)
            # Sum of squares over all axes except the last (in_features)
            ss = np.sum(x_np ** 2, axis=tuple(range(x_np.ndim - 1)))
            if accum[path] is None:
                accum[path] = ss
            else:
                accum[path] += ss
            total_tokens += x_np.shape[0] * (x_np.shape[1] if x_np.ndim > 1 else 1)

    hessian = {}
    for path, ss in accum.items():
        if ss is not None:
            hessian[path] = ss / total_tokens
    return hessian


# ---------------------------------------------------------------------------
# Quantization + error computation
# ---------------------------------------------------------------------------
def quantize_and_reconstruct(weight: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    w = mx.array(weight, dtype=mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q, scales, biases)
    reconstructed = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
    mx.eval(reconstructed)
    return np.array(reconstructed)


def compute_hessian_weighted_error(original: np.ndarray, reconstructed: np.ndarray, h_diag: np.ndarray | None) -> float:
    """Compute Hessian-weighted relative error.

    If h_diag is provided:
        error = sum((W - Q(W))^2 * H) / sum(W^2 * H)
    Otherwise falls back to plain Frobenius relative error.
    """
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    if h_diag is not None:
        if diff.ndim == 2:
            h = h_diag.reshape(1, -1)
        else:
            h = h_diag
        weighted_diff = np.sum(diff ** 2 * h)
        weighted_orig = np.sum(original.astype(np.float64) ** 2 * h)
        return float(weighted_diff / max(weighted_orig, 1e-12))
    else:
        norm_diff = np.sum(diff * diff)
        norm_orig = np.sum(original.astype(np.float64) ** 2)
        return float(norm_diff / max(norm_orig, 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build Hessian-aware error_db.json from BF16 checkpoint")
    ap.add_argument("--source", required=True, help="BF16 safetensors directory")
    ap.add_argument("--output", required=True, help="Output error_db.json path")
    ap.add_argument("--model-type", required=True, help="Model type for adapter")
    ap.add_argument("--calibration", required=True, help="Calibration token IDs .npy")
    ap.add_argument("--bitwidths", default="2,3,4,6,8", help="Bitwidths to sweep")
    ap.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    ap.add_argument("--batch-size", type=int, default=4, help="Calibration batch size")
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

    print(f"Quantizable tensors: {len(quant_keys)}")

    calib_path = Path(args.calibration)
    if not calib_path.exists():
        sys.exit(f"Calibration file not found: {calib_path}")
    calibration_ids = np.load(str(calib_path))
    print(f"Calibration data: {calibration_ids.shape}")

    # Load model and wrap linear layers
    print("Loading BF16 model...")
    from mlx_lm.utils import load as mlx_load
    model, _ = mlx_load(str(source))
    model.eval()

    print("Wrapping linear layers for activation capture...")
    wrapped_paths = wrap_linear_layers(model)
    print(f"  Wrapped {len(wrapped_paths)} layers")

    # Collect activations and compute Hessian diagonals
    print(f"Running calibration ({calibration_ids.shape[0]} sequences, batch_size={args.batch_size})...")
    hessian_map = _collect_activations(model, wrapped_paths, calibration_ids, batch_size=args.batch_size)
    print(f"  Collected Hessian for {len(hessian_map)} layers")

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

        # Get Hessian diagonal for this layer
        h_diag = hessian_map.get(base)

        layer_errors = {}
        for bw in bitwidths:
            try:
                reconstructed = quantize_and_reconstruct(tensor, bw, args.group_size)
                err = compute_hessian_weighted_error(tensor, reconstructed, h_diag)
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


def get_adapter(model_type: str):
    return get(model_type)()


def is_quantizable(adapter, layer_path: str) -> bool:
    for prefix in adapter.non_quantized_prefixes():
        if f".{prefix}." in f".{layer_path}." or layer_path.startswith(f"{prefix}.") or layer_path == prefix:
            return False
    return True


if __name__ == "__main__":
    main()

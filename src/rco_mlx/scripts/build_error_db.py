#!/usr/bin/env python3
"""Build a Hessian-weighted error_db.json from a BF16 safetensors checkpoint.

For each quantizable linear layer, quantize at each bitwidth in {2,3,4,6,8}
using MLX affine, then measure quantization error weighted by the empirical
Fisher diagonal estimated from calibration activations:

    error = (W - Q(W))^T H_diag (W - Q(W)) / (W^T H_diag W)

The Hessian diagonal ``H_diag`` is estimated by running calibration data through
the model and capturing per-layer inputs via ``_CaptureModule`` wrappers:

    H_diag[l] = mean(x^2, axis=all-but-last)

This gives better per-layer sensitivity estimates than plain Frobenius error,
leading to better mixed-precision allocation decisions.

Usage:
    python -m rco_mlx.scripts.build_error_db \\
        --source ~/Projects/rco-mlx/.cache/olmoe-1b-7b-0125 \\
        --output error_db.json \\
        --model-type olmoe \\
        --calibration /tmp/olmoe_calib_input_ids.npy

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
# Activation-capturing module wrappers (for Hessian mode)
# ---------------------------------------------------------------------------
class _CaptureLinear(nn.Linear):
    """Drop-in replacement for nn.Linear that stores the last input."""

    def __call__(self, x):
        self._last_input = x
        return super().__call__(x)


class _CaptureQuantized(nn.QuantizedLinear):
    """Drop-in replacement for nn.QuantizedLinear that stores the last input."""

    def __init__(self, source: nn.QuantizedLinear):
        super().__init__(
            input_dims=source.weight.shape[1],
            output_dims=source.weight.shape[0],
            bias="bias" in source,
            group_size=source.group_size,
            bits=source.bits,
            mode=source.mode,
        )
        # Copy the quantized state (packed weight, scales, biases, bias)
        for k in source.keys():
            self[k] = source[k]

    def __call__(self, x):
        self._last_input = x
        return super().__call__(x)


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


def _get_module(model, path: str):
    parts = [p for p in path.split(".") if p]
    obj = model
    for part in parts:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def wrap_linear_layers(model):
    """Wrap every nn.Linear to capture activations. Returns wrapped paths."""
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


def _collect_activations(model, wrapped_paths, calibration_ids, batch_size=4):
    """Run calibration data and compute per-layer Hessian diagonals.

    Returns:
        dict: layer_path -> hessian_diag (1-D numpy array, shape: in_features)
    """
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
            module = _get_module(model, path)
            x = module._last_input
            if x is None:
                continue
            # Explicit copy to break any view that retains the parent buffer.
            # MLX slices are views; appending them keeps the full upstream
            # activation alive. mx.array() copies without the parent.
            x = mx.array(x)
            x_np = np.array(x.astype(mx.float32))
            # Sum of squares over all axes except last (in_features)
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
# Quantization + error
# ---------------------------------------------------------------------------
def quantize_and_reconstruct(weight: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """Quantize with MLX affine and dequantize back to float32."""
    w = mx.array(weight, dtype=mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q, scales, biases)
    reconstructed = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
    mx.eval(reconstructed)
    return np.array(reconstructed)


def compute_kl_sensitivity(model, calibration_ids, bitwidths, group_size=64,
                           kl_samples=8, kl_bits=4):
    """Compute per-layer KL divergence sensitivity.

    For each quantizable linear layer, quantize it to kl_bits and measure the KL
    divergence of the final output logits vs. full-precision. Lower KL means
    the layer is more robust to quantization (can use fewer bits).

    Uses stratified sampling to keep runtime reasonable:
    - Every Nth layer (N = ceil(num_layers / max_layers))
    - All MoE/expert layers
    - First and last layer

    Discovers layers by walking the model tree directly (no safetensors key mapping).

    Args:
        model: MLX model
        calibration_ids: numpy array of token IDs (n_samples, seq_len)
        bitwidths: list of bitwidths to sweep
        group_size: quantization group size
        kl_samples: number of calibration samples to use for KL measurement
        kl_bits: reference bitwidth for KL measurement

    Returns:
        dict: mlx_path -> kl_score (float, lower = more robust)
    """
    import mlx.nn as nn

    n_samples = min(kl_samples, calibration_ids.shape[0])
    kl_ids = calibration_ids[:n_samples]
    input_ids = mx.array(kl_ids)

    # Get full-precision logits
    model.eval()
    fp_logits = model(input_ids)
    mx.eval(fp_logits)
    fp_probs = mx.softmax(fp_logits, axis=-1)
    fp_log_probs = mx.log(fp_probs + 1e-10)

    # Discover all quantizable modules by walking the model tree
    quant_modules = []
    for name, module in model.named_modules():
        # Skip root and non-quantizable types
        if not name:
            continue
        if isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            if hasattr(module, 'weight') and module.weight is not None:
                shape = module.weight.shape
                if len(shape) == 2 and min(shape) >= 4:
                    quant_modules.append((name, module, shape))
        # Handle SwitchLinear (MoE experts)
        elif type(module).__name__ == 'SwitchLinear':
            if hasattr(module, 'weight') and module.weight is not None:
                shape = module.weight.shape
                if len(shape) == 3 and min(shape) >= 4:
                    quant_modules.append((name, module, shape))

    print(f"[KL] Discovered {len(quant_modules)} quantizable modules")

    # Stratified sampling: select representative layers
    all_names = [name for name, _, _ in quant_modules]
    selected_indices = set()

    # Always include first and last
    if all_names:
        selected_indices.add(0)
        selected_indices.add(len(all_names) - 1)

    # Include all MoE/expert layers
    for i, name in enumerate(all_names):
        if 'expert' in name or 'switch_mlp' in name or 'feed_forward' in name:
            selected_indices.add(i)

    # Sample every Nth layer to get max ~30 layers
    max_kl_layers = 30
    n_total = len(quant_modules)
    if n_total > max_kl_layers:
        step = max(1, n_total // max_kl_layers)
        for i in range(0, n_total, step):
            selected_indices.add(i)
    else:
        selected_indices.update(range(n_total))

    selected_modules = [(all_names[i], quant_modules[i][1], quant_modules[i][2])
                        for i in sorted(selected_indices)]
    print(f"[KL] Selected {len(selected_modules)}/{len(quant_modules)} modules for KL measurement")

    kl_scores = {}

    for idx, (name, module, shape) in enumerate(selected_modules):
        original_weight = module.weight

        try:
            # Quantize this module's weights
            # Convert bfloat16 to float32 before numpy to avoid PEP 3118 error
            w_np = np.array(original_weight.astype(mx.float32))
            recon = quantize_and_reconstruct(w_np, kl_bits, group_size)
            recon_mx = mx.array(recon, dtype=original_weight.dtype)

            # Temporarily replace weights
            module.weight = recon_mx
            mx.eval(module.weight)

            # Run forward pass
            q_logits = model(input_ids)
            mx.eval(q_logits)
            q_probs = mx.softmax(q_logits, axis=-1)
            q_log_probs = mx.log(q_probs + 1e-10)

            # Compute KL(fp || q) averaged over tokens
            kl = mx.mean(mx.sum(fp_probs * (fp_log_probs - q_log_probs), axis=-1))
            kl_val = float(np.array(kl))
            kl_scores[name] = kl_val

            # Restore original weights
            module.weight = original_weight
            mx.eval(module.weight)

            # Clear cache between layers
            mx.clear_cache()

            print(f"  KL [{idx+1}/{len(selected_modules)}] {name} = {kl_val:.6f}")

        except Exception as e:
            print(f"  KL ERROR {name}: {e}")
            try:
                module.weight = original_weight
                mx.eval(module.weight)
            except Exception:
                pass
            continue

    return kl_scores


def compute_hessian_weighted_error(original: np.ndarray, reconstructed: np.ndarray, h_diag: np.ndarray) -> float:
    """Compute Hessian-weighted relative error.

    error = sum((W - Q(W))^2 * H) / sum(W^2 * H)
    """
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
def get_adapter(model_type: str):
    return get(model_type)()


def is_quantizable(adapter, layer_path: str) -> bool:
    for prefix in adapter.non_quantized_prefixes():
        if f".{prefix}." in f".{layer_path}." or layer_path.startswith(f"{prefix}.") or layer_path == prefix:
            return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build SLQ-style error_db.json from BF16 checkpoint")
    ap.add_argument("--source", required=True, help="BF16 safetensors directory")
    ap.add_argument("--output", required=True, help="Output error_db.json path")
    ap.add_argument("--model-type", default="default", help="Model type for adapter (default: 'default' for standard dense models)")
    ap.add_argument("--calibration", required=True,
                    help="Path to calibration token IDs .npy file (required for Hessian-weighted error)")
    ap.add_argument("--bitwidths", default="2,3,4,6,8", help="Bitwidths to sweep")
    ap.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    ap.add_argument("--batch-size", type=int, default=4, help="Calibration batch size")
    ap.add_argument("--rtn-source", default=None,
                    help="Path to an 8-bit RTN quant directory. When provided, use the "
                         "streaming two-phase approach: load the small RTN model for "
                         "activation capture, then measure per-tensor error from BF16 "
                         "shards one at a time. This avoids loading the full BF16 model "
                         "into memory.")
    ap.add_argument("--kl-sensitivity", action="store_true",
                    help="Compute per-layer KL divergence sensitivity scores (forward-only, "
                         "no weight reconstruction). Stores kl_sensitivity in error_db.")
    ap.add_argument("--kl-samples", type=int, default=8,
                    help="Number of calibration samples for KL sensitivity (default: 8)")
    ap.add_argument("--kl-bits", type=int, default=4,
                    help="Reference bitwidth for KL sensitivity measurement (default: 4)")
    args = ap.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(args.model_type)
    bitwidths = [int(b) for b in args.bitwidths.split(",")]

    # Streaming mode: delegate to build_error_db_streaming when an RTN source
    # is provided. This avoids loading the full BF16 model into memory.
    if args.rtn_source:
        from rco_mlx.scripts.build_error_db_streaming import main as streaming_main
        # Re-invoke the streaming script with the same arguments + rtn_source
        import sys
        sys.argv = [
            sys.argv[0],
            "--source", str(source),
            "--rtn-source", str(Path(args.rtn_source).expanduser().resolve()),
            "--output", str(output),
            "--model-type", args.model_type,
            "--calibration", args.calibration,
            "--bitwidths", args.bitwidths,
            "--group-size", str(args.group_size),
            "--batch-size", str(args.batch_size),
        ]
        streaming_main()
        return

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

    # Estimate BF16 memory footprint and warn if it may exceed available RAM.
    # On Apple Silicon unified memory this is advisory: the model shares RAM
    # with everything else, and the kernel will kill the process if it runs out.
    est_bf16_gb = 0.0
    with safe_open(str(shards[0]), framework="pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".weight"):
                est_bf16_gb += f.get_tensor(k).numel() * 2 / 1e9
    est_peak_gb = est_bf16_gb * 2.5  # Hessian + graph + intermediates
    print(f"Estimated BF16 model size: {est_bf16_gb:.1f} GiB")
    print(f"Estimated peak memory:    ~{est_peak_gb:.1f} GiB")
    if est_peak_gb > 50:
        print("WARNING: estimated peak exceeds 50 GiB. Consider --rtn-source for "
              "streaming two-phase activation capture.")

    quant_keys = []
    for k in all_keys:
        if not k.endswith(".weight"):
            continue
        base = k[:-7]
        if is_quantizable(adapter, base):
            quant_keys.append((base, weight_map[k]))

    print(f"Quantizable tensors: {len(quant_keys)}")

    # Load model and collect Hessian diagonals
    calib_path = Path(args.calibration)
    if not calib_path.exists():
        sys.exit(f"Calibration file not found: {calib_path}")
    calibration_ids = np.load(str(calib_path))
    print(f"Calibration data: {calibration_ids.shape}")

    print("Loading BF16 model for activation capture...")
    from mlx_lm.utils import load as mlx_load
    model, _ = mlx_load(str(source))
    model.eval()

    print("Wrapping linear layers...")
    wrapped_paths = wrap_linear_layers(model)
    print(f"  Wrapped {len(wrapped_paths)} layers")

    print(f"Running calibration (batch_size={args.batch_size})...")
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

        h_diag = hessian_map.get(base)
        if h_diag is None:
            print(f"  SKIP {base}: no Hessian captured")
            continue

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
            print(f"  [{idx+1}/{total}] {base} [Hessian] — "
                  f"errors: " + ", ".join(f"{bw}={layer_errors[str(bw)]['error']:.6f}"
                                          for bw in bitwidths if str(bw) in layer_errors))

    # Optional: compute per-layer KL divergence sensitivity
    if args.kl_sensitivity:
        print(f"\n[KL] Computing per-layer KL divergence sensitivity "
              f"(samples={args.kl_samples}, bits={args.kl_bits})...")
        kl_scores = compute_kl_sensitivity(
            model=model,
            calibration_ids=calibration_ids,
            bitwidths=bitwidths,
            group_size=args.group_size,
            kl_samples=args.kl_samples,
            kl_bits=args.kl_bits,
        )
        db["kl_sensitivity"] = kl_scores
        if kl_scores:
            kl_vals = list(kl_scores.values())
            print(f"[KL] Computed {len(kl_scores)} scores: "
                  f"range=[{min(kl_vals):.6f}, {max(kl_vals):.6f}], "
                  f"mean={np.mean(kl_vals):.6f}")

    output.write_text(json.dumps(db, indent=2))
    print(f"\nWrote error_db.json (Hessian-weighted): {len(db['layers'])} layers, {len(all_errors)} measurements")
    if all_errors:
        print(f"Error range: [{min(all_errors):.6f}, {max(all_errors):.6f}], mean={np.mean(all_errors):.6f}")


if __name__ == "__main__":
    main()

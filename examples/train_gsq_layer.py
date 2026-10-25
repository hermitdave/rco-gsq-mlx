"""GSQ training loop for MLX.

Trains a Gumbel-Softmax quantizer on a single linear layer's weights using
calibration data, then extracts hard weights for inference.

Usage:
    python train_gsq_layer.py \\
        --weights ~/Projects/ml/quantization/work/layer_weights.npz \\
        --calibration ~/Projects/ml/quantization/work/calibration.npy \\
        --bits 2 \\
        --group-size 64 \\
        --target-bpw 4.0 \\
        --output work/gsq_layer.npz

The weights file should contain 'weight' (float32, 2D), and optionally
'scales' (float32, 2D) for RTN initialization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from rco_mlx.gsq_quantizer import (
    CosineLRScheduler,
    GumbelQuantizer2Bit,
    GumbelQuantizerInt,
    Lion,
    effective_bits,
)


def load_or_init_weights(weights_path: Path, group_size: int):
    """Load BF16 weights and compute RTN initialization.

    Returns (weight, scales, group_size).
    """
    data = np.load(weights_path, allow_pickle=False)
    weight = mx.array(data["weight"].astype(np.float32))

    if "scales" in data and "biases" in data:
        scales = mx.array(data["scales"].astype(np.float32))
        return weight, scales, group_size

    # RTN initialization
    w = np.array(weight)
    out_features, in_features = w.shape
    n_groups = in_features // group_size
    w_grouped = w.reshape(out_features, n_groups, group_size)
    scales = np.max(np.abs(w_grouped), axis=2) / 7.0  # 8-bit range
    scales = np.maximum(scales, 1e-8)
    scales_expanded = np.repeat(scales[:, :, np.newaxis], group_size, axis=2)
    Q = np.round(w / scales_expanded.reshape(out_features, in_features))
    Q = np.clip(Q, -7, 7).astype(np.float32)
    scales_2d = scales.astype(np.float32)
    return weight, Q, scales_2d


def train_layer(
    weight: mx.array,
    scales: mx.array,
    calibration: mx.array,
    bits: int,
    group_size: int,
    target_bpw: float,
    num_epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    warmup_steps: int = 50,
    std: float = 0.01,
    strength: float = 1.0,
    key: int = 42,
):
    """Train a GSQ quantizer on one layer.

    Parameters
    ----------
    weight : mx.array (float32, 2D)
        Full-precision weights.
    scales : mx.array (float32, 2D)
        Per-group scales from RTN initialization.
    calibration : mx.array (float32, 2D)
        Calibration activations: (n_samples, in_features).
    bits : int
        Target bit width.
    group_size : int
        Quantization group size.
    target_bpw : float
        Target bits per weight (for reporting).
    num_epochs : int
        Training epochs over calibration data.
    batch_size : int
        Microbatch size.
    lr : float
        Learning rate.
    warmup_steps : int
        LR warmup steps.
    std : float
        Logit initialization std.
    strength : float
        Init signal strength.
    key : int
        RNG seed.

    Returns
    -------
    hard_weights : mx.array
        Dequantized weights (for measurement).
    report : dict
        Training statistics.
    """
    out_features, in_features = weight.shape
    n_groups = in_features // group_size
    assert in_features % group_size == 0

    # Create quantizer
    if bits == 2:
        quantizer = GumbelQuantizer2Bit(
            Q=scales * mx.round(weight / mx.repeat(scales, group_size, axis=1)),
            scales=scales,
            group_size=group_size,
            std=std,
            strength=strength,
            dtype=weight.dtype,
        )
    else:
        quantizer = GumbelQuantizerInt(
            Q=scales * mx.round(weight / mx.repeat(scales, group_size, axis=1)),
            scales=scales,
            group_size=group_size,
            std=std,
            strength=strength,
            dtype=weight.dtype,
            bits=bits,
        )

    # Optimizer
    optimizer = Lion(
        params=[quantizer.quant_logits, quantizer.scales],
        lr=lr,
        weight_decay=0.0,
    )

    n_samples = calibration.shape[0]
    n_steps = max(1, (n_samples + batch_size - 1) // batch_size) * num_epochs
    scheduler = CosineLRScheduler(optimizer, n_steps, warmup_steps)

    initial_temp, final_temp = 2.0, 0.05
    initial_scale, final_scale = 100.0, 500.0

    rng = mx.random.key(key)
    losses = []
    step = 0
    t0 = time.time()
    temp = initial_temp
    sc = initial_scale

    for epoch in range(num_epochs):
        perm = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            x = calibration[batch_idx]  # (B, in_features)

            # Temperature / scale annealing
            t = step / max(n_steps - 1, 1)
            temp = initial_temp + (final_temp - initial_temp) * t
            sc = initial_scale + (final_scale - initial_scale) * t

            # Forward
            rng, subkey = mx.random.split(rng)
            soft_w, _ = quantizer.forward(temp, sc, subkey)

            # MSE: soft-quantized output vs full-precision output
            y_soft = x @ soft_w.T  # (B, out_features)
            y_fp = x @ weight.T
            loss = mx.mean((y_soft - y_fp) ** 2)

            # Backward
            mx.eval(loss)
            loss_grad_fn = mx.grad(lambda ql, sc: mx.mean(
                (x @ quantizer.forward(temp, sc, subkey)[0].T - y_fp) ** 2
            ))
            # Note: we need to pass the parameters that require gradients
            # The quantizer stores them as attributes; we need to use them
            # in a closure for mx.grad to trace.

            # Manual gradient: use the analytical formulas from the PyTorch port
            # This avoids the noise-replay issue entirely.
            def loss_fn(ql, sc):
                out, sq = quantizer.forward(temp, sc, subkey)
                y = x @ out.T
                return mx.mean((y - y_fp) ** 2)

            grads = mx.grad(loss_fn)(quantizer.quant_logits, quantizer.scales)
            quantizer.quant_logits.grad = grads[0]
            quantizer.scales.grad = grads[1]

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += float(loss)
            n_batches += 1
            step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        achieved_bpw = effective_bits(bits, group_size)
        print(
            f"  epoch {epoch+1:3d}/{num_epochs}  "
            f"loss={avg_loss:.4e}  "
            f"temp={temp:.4f}  "
            f"effective={achieved_bpw:.3f} bpw  "
            f"target={target_bpw:.1f} bpw"
        )

    # Hard weights
    hard_w, hard_scales = quantizer.get_hard_weights()

    report = {
        "bits": bits,
        "group_size": group_size,
        "target_bpw": target_bpw,
        "achieved_bpw": effective_bits(bits, group_size),
        "num_epochs": num_epochs,
        "final_loss": losses[-1] if losses else None,
        "loss_history": losses,
        "elapsed_s": time.time() - t0,
    }
    return hard_w, hard_scales, report


def main():
    ap = argparse.ArgumentParser(
        description="Train a GSQ quantizer on a single layer (MLX port)"
    )
    ap.add_argument("--weights", required=True, help="NPZ with 'weight' (float32 2D)")
    ap.add_argument("--calibration", required=True, help="NPY calibration activations (n_samples, in_features)")
    ap.add_argument("--bits", type=int, required=True, choices=[2, 3, 4, 5, 6])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--target-bpw", type=float, default=4.0)
    ap.add_argument("--num-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--std", type=float, default=0.01)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="work/gsq_layer.npz")
    args = ap.parse_args()

    weight_path = Path(args.weights)
    cal_path = Path(args.calibration)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights from {weight_path}")
    weight, scales, gs = load_or_init_weights(weight_path, args.group_size)
    print(f"  weight shape: {weight.shape}, dtype: {weight.dtype}")
    print(f"  scales shape: {scales.shape}")
    print(f"  group_size: {gs}")

    print(f"Loading calibration from {cal_path}")
    calibration = mx.array(np.load(cal_path, allow_pickle=False).astype(np.float32))
    print(f"  calibration shape: {calibration.shape}")

    print(f"Training GSQ {args.bits}-bit quantizer …")
    hard_w, hard_scales, report = train_layer(
        weight=weight,
        scales=scales,
        calibration=calibration,
        bits=args.bits,
        group_size=gs,
        target_bpw=args.target_bpw,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        std=args.std,
        strength=args.strength,
        key=args.seed,
    )

    # Save
    np.savez(
        out_path,
        weight=np.array(hard_w),
        scales=np.array(hard_scales),
        bits=args.bits,
        group_size=gs,
        **{k: (v if not isinstance(v, list) else np.array(v)) for k, v in report.items() if k not in ("loss_history",)},
    )
    report_path = out_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

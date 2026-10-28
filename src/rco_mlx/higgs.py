"""HIGGS pre-calibration for RCO quantization.

Implements the HIGGS algorithm from arXiv:2501.09107:
- Weight classification into salient/non-salient via adaptive LASSO
- Soft-thresholding: ŵ_i = sign(w_i) * ReLU(|w_i| - λ'/|w_i|)
- Salient weights protected, non-salient weights shrunk toward zero
- No calibration data required

This is a pre-processing step applied to BF16 weights before RCO quantization.
Per the paper, the classification guides bitwidth assignment; we apply the
soft-thresholding as a practical pre-calibration because MLX quantizes whole
tensors uniformly, so we reshape the distribution rather than split masks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Core HIGGS algorithm
# ---------------------------------------------------------------------------

def higgs_classify(
    weight: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Classify weights into salient (True) and non-salient (False) using HIGGS.

    Uses adaptive LASSO with pseudo activations (identity matrix) to determine
    a threshold λ' such that alpha percent of weights are classified as salient.

    A weight w_i is salient if |w_i|^2 > λ', which is equivalent to
    |w_i| > λ'/|w_i| from equation (5) in the paper.

    Args:
        weight: 1D or 2D numpy array of weights
        alpha: percentage of weights to classify as salient (default 5%)

    Returns:
        Boolean mask: True = salient, False = non-salient
    """
    w_flat = weight.flatten()
    n = len(w_flat)
    n_salient_target = int(np.ceil(alpha * n))

    if n_salient_target == 0:
        return np.zeros(n, dtype=bool)
    if n_salient_target >= n:
        return np.ones(n, dtype=bool)

    # Compute λ' directly from percentile: λ' is at (1-α) percentile of |w|^2
    abs_w_sq = w_flat ** 2
    lambda_prime = np.percentile(abs_w_sq, 100.0 * (1.0 - alpha))
    salient_mask = abs_w_sq > lambda_prime

    return salient_mask


def apply_higgs(
    weight: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply HIGGS pre-calibration: soft-thresholding non-salient weights.

    Per the paper, the soft-thresholding ŵ = sign(w) * ReLU(|w| - λ'/|w|)
    is derived from adaptive LASSO with pseudo activations. We apply it as
    a pre-processing step because MLX quantizes whole tensors uniformly;
    shrinking non-salient weights toward zero preserves their distribution
    while making them less prone to quantization error.

    Args:
        weight: original weight array (numpy)
        alpha: salient weight percentage (default 5%)

    Returns:
        Pre-calibrated weight array (same shape as input)
    """
    w = weight.copy()
    w_flat = w.flatten()

    # Compute λ' directly from percentile
    abs_w_sq = w_flat ** 2
    lambda_prime = np.percentile(abs_w_sq, 100.0 * (1.0 - alpha))

    # Soft-thresholding: ŵ = sign(w) * ReLU(|w| - λ'/|w|)
    abs_w = np.abs(w_flat)
    threshold = lambda_prime / (abs_w + 1e-8)
    shrinkage = np.minimum(abs_w, threshold)
    w_flat = np.sign(w_flat) * shrinkage
    w = w_flat.reshape(w.shape)

    return w


# ---------------------------------------------------------------------------
# Layer-by-layer pre-calibration
# ---------------------------------------------------------------------------

def precalibrate_model(
    source_dir: Path,
    output_dir: Path,
    alpha: float = 0.05,
    salient_bits: int = 4,
    non_salient_bits: int = 3,
    adapter=None,
) -> dict:
    """Apply HIGGS pre-calibration to all quantizable weights in a model.

    Args:
        source_dir: BF16 model directory
        output_dir: output directory for pre-calibrated model
        alpha: salient weight percentage (default 5%)
        salient_bits: bitwidth for salient weights
        non_salient_bits: bitwidth for non-salient weights
        adapter: model adapter for key mapping

    Returns:
        Report dict with statistics
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_dir = Path(source_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all shards
    shards = sorted(source_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards in {source_dir}")

    # Load config
    with open(source_dir / "config.json") as f:
        config = json.load(f)

    # Copy non-weight files
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        src = source_dir / fname
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / fname)

    # Process each shard
    total_params = 0
    salient_counts = []
    layer_stats = {}

    for shard_idx, shard_path in enumerate(shards):
        print(f"  Processing shard {shard_idx+1}/{len(shards)}: {shard_path.name}")

        tensors = {}
        with safe_open(str(shard_path), framework="numpy") as f:
            for key in f.keys():
                weight = f.get_tensor(key)

                # Skip non-quantizable tensors
                if weight.ndim not in (2, 3):
                    tensors[key] = weight
                    continue

                # Skip biases, norms, embeddings, lm_head
                skip = False
                for prefix in ["bias", "norm", "embed", "lm_head", "mtp"]:
                    if prefix in key:
                        skip = True
                        break
                if skip:
                    tensors[key] = weight
                    continue

                if adapter is not None:
                    mapped = adapter.map_key_to_mlx(key)
                    if mapped is None:
                        tensors[key] = weight
                        continue

                # Compute salient count for reporting
                salient_mask = higgs_classify(weight, alpha=alpha)
                n_salient = int(np.sum(salient_mask))
                salient_counts.append(n_salient)

                # Soft-thresholding
                calibrated = apply_higgs(weight, alpha=alpha)

                tensors[key] = calibrated
                total_params += weight.size

                layer_stats[key] = {
                    "total": weight.size,
                    "salient": n_salient,
                    "ratio": float(n_salient) / weight.size,
                }

        # Save shard
        out_path = output_dir / shard_path.name
        # safetensors.torch.save_file expects torch.Tensor, convert from mlx/numpy
        import torch
        torch_tensors = {}
        for k, v in tensors.items():
            if hasattr(v, 'tolist'):
                # mlx array or numpy -> torch tensor
                torch_tensors[k] = torch.tensor(v.tolist() if hasattr(v, 'tolist') else np.array(v).tolist())
            elif isinstance(v, np.ndarray):
                torch_tensors[k] = torch.from_numpy(v)
            else:
                torch_tensors[k] = v
        save_file(torch_tensors, str(out_path))

    # Generate index
    index = {"metadata": {"total_size": total_params}, "weight_map": {}}
    for shard_idx, shard_path in enumerate(shards):
        with safe_open(str(shard_path), framework="numpy") as f:
            for key in f.keys():
                index["weight_map"][key] = shard_path.name

    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    report = {
        "method": "HIGGS",
        "alpha": alpha,
        "salient_bits": salient_bits,
        "non_salient_bits": non_salient_bits,
        "total_params": total_params,
        "total_tensors": len(salient_counts),
        "mean_salient_ratio": float(np.mean([s / t for s, t in zip(salient_counts, [v["total"] for v in layer_stats.values()])])) if layer_stats else 0.0,
        "layer_stats": layer_stats,
    }

    return report


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_higgs(
    source_dir: Path,
    precalibrated_dir: Path,
    adapter=None,
) -> dict:
    """Validate HIGGS pre-calibration: compare weight distributions.

    Args:
        source_dir: original BF16 model
        precalibrated_dir: HIGGS pre-calibrated model
        adapter: model adapter

    Returns:
        Validation report
    """
    from safetensors import safe_open

    source_dir = Path(source_dir).expanduser()
    precalibrated_dir = Path(precalibrated_dir).expanduser()

    src_shards = sorted(source_dir.glob("*.safetensors"))
    pre_shards = sorted(precalibrated_dir.glob("*.safetensors"))

    if not src_shards or not pre_shards:
        return {"error": "No shards found"}

    layer_diffs = []

    with safe_open(str(src_shards[0]), framework="numpy") as f_src, \
         safe_open(str(pre_shards[0]), framework="numpy") as f_pre:

        for key in f_src.keys():
            w_src = f_src.get_tensor(key)
            try:
                w_pre = f_pre.get_tensor(key)
            except Exception:
                continue

            if w_src.ndim not in (2, 3) or w_pre.ndim != w_src.ndim:
                continue

            # KL divergence approximation between distributions
            mu_src, sigma_src = np.mean(w_src), np.std(w_src)
            mu_pre, sigma_pre = np.mean(w_pre), np.std(w_pre)

            kl = np.log(sigma_pre / sigma_src) + (sigma_src**2 + (mu_src - mu_pre)**2) / (2 * sigma_pre**2) - 0.5

            layer_diffs.append({
                "key": key,
                "mean_diff": float(np.mean(np.abs(w_src - w_pre))),
                "std_src": float(sigma_src),
                "std_pre": float(sigma_pre),
                "kl": float(kl),
            })

    return {
        "layers_compared": len(layer_diffs),
        "mean_kl": float(np.mean([d["kl"] for d in layer_diffs])) if layer_diffs else 0.0,
        "mean_abs_diff": float(np.mean([d["mean_diff"] for d in layer_diffs])) if layer_diffs else 0.0,
        "layer_diffs": layer_diffs[:10],  # sample
    }

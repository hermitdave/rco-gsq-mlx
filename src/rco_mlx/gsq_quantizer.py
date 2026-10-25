"""Gumbel-Softmax quantizers ported to MLX.

Ports the core GSQ algorithm (ISTA-DASLab/GSQ, Apache-2.0) from PyTorch to
MLX.  The quantizers are pure functions: given logits, scales, and a
temperature they return a soft-quantized weight tensor whose gradient flows
through MLX's autograd.

The downstream training loop is in ``gsq_train.py``.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np

__all__ = [
    "GumbelQuantizer2Bit",
    "GumbelQuantizerInt",
    "effective_bits",
    "PACKS_PER_WORD",
]

# Exact-width packing: values per 32-bit word, matching Humming layout.
PACKS_PER_WORD = {2: 16, 3: 10, 4: 8, 5: 6, 6: 5, 8: 4}


def effective_bits(bits: int, group_size: int, scale_bits: int = 16) -> float:
    """Bits per weight including per-group scale overhead."""
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(
            f"bits={bits} is not supported by MLX (supported: 2,3,4,5,6,8)"
        )
    if group_size not in (32, 64, 128):
        raise ValueError(
            f"group_size={group_size} is not supported by MLX (supported: 32,64,128)"
        )
    return bits + scale_bits / group_size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gumbel_noise(shape, key):
    """Sample Gumbel(0,1) noise."""
    u = mx.random.uniform(shape=shape, key=key)
    return -mx.log(-mx.log(u + 1e-8) + 1e-8)


def _soft_quantize_2bit(logits, scales, values, idx, temperature, scale_factor, key):
    """Soft 2-bit Gumbel quantize.

    Returns (soft_output, soft_quant) where soft_output is the dequantized
    weight tensor and soft_quant is the (4, rows, cols) soft assignment
    probabilities (needed for the manual backward in the trainer).
    """
    noise = _gumbel_noise(logits.shape, key)
    soft_quant = mx.softmax((logits * scale_factor + noise) / temperature, axis=0)
    soft_output = (soft_quant * values.reshape(4, 1, 1)).sum(axis=0)
    scale_per_col = scales[:, idx]
    return soft_output * scale_per_col, soft_quant


def _soft_quantize_int(
    logits, scales, shift_values, valid_mask, idx, init, temperature, scale_factor, key
):
    """Soft N-bit Gumbel quantize (N = len(shift_values))."""
    masked_logits = mx.where(valid_mask, logits * scale_factor, -1e9)
    noise = _gumbel_noise(masked_logits.shape, key)
    soft_quant = mx.softmax((masked_logits + noise) / temperature, axis=0)
    soft_output = (soft_quant * shift_values.reshape(-1, 1, 1)).sum(axis=0)
    scale_per_col = scales[:, idx]
    return (init + soft_output) * scale_per_col, soft_quant


# ---------------------------------------------------------------------------
# 2-bit quantizer
# ---------------------------------------------------------------------------

class GumbelQuantizer2Bit:
    """2-bit Gumbel-Softmax quantizer.

    Parameters
    ----------
    Q : mx.array (float32, shape (out_features, in_features))
        Initial quantized weights (from RTN or GPTQ), divided by per-group
        scales in the constructor.
    scales : mx.array (float32, shape (out_features, n_groups))
        Per-group scales.  ``n_groups = in_features // group_size``.
    group_size : int
        Number of input channels per group.
    std : float
        Std dev of the random perturbation on initialization.
    strength : float
        Weight of the initialization signal vs. the random prior.
    dtype : mx.Dtype
        Working dtype for values (typically bfloat16 or float16).
    logits_dtype : mx.Dtype, optional
        Dtype for the learnable logits.  Defaults to ``dtype``.
    """

    def __init__(
        self,
        Q: mx.array,
        scales: mx.array,
        group_size: int,
        std: float,
        strength: float,
        dtype: mx.Dtype,
        logits_dtype: Optional[mx.Dtype] = None,
    ):
        self.dtype = dtype
        self.logits_dtype = logits_dtype or dtype
        self.group_size = group_size
        self.idx = mx.arange(Q.shape[1]) // group_size  # (in_features,)

        scale_per_col = scales[:, self.idx]
        Q_init = Q / scale_per_col

        values = mx.array([-2.0, -1.0, 0.0, 1.0], dtype=Q.dtype)

        logits = -0.5 * (Q_init[None, ...] - values.reshape(4, 1, 1)) ** 2
        logits = logits - logits.mean(axis=0, keepdims=True)
        quant_logits = std * (
            mx.random.normal(logits.shape) + logits * strength
        )

        self.values = values.astype(self.dtype)
        self.quant_logits = quant_logits.astype(self.logits_dtype)
        self.scales = scales.astype(mx.float32)

    def forward(self, temperature: float, scale_factor: float = 1.0, key=None):
        """Soft-quantize the weights.

        Parameters
        ----------
        temperature : float
            Gumbel temperature (annealed 2.0 → 0.05 during training).
        scale_factor : float
            Logit scale (annealed 100 → 500 during training).
        key : mx.random.Key, optional
            RNG key for Gumbel noise.  Generated from a default key if None.

        Returns
        -------
        soft_output : mx.array
            Dequantized weight tensor, same shape as the original weights.
        soft_quant : mx.array
            (4, rows, cols) soft assignment probabilities.
        """
        if key is None:
            key = mx.random.key(0)
        out, sq = _soft_quantize_2bit(
            self.quant_logits, self.scales, self.values,
            self.idx, temperature, scale_factor, key,
        )
        return out, sq

    def get_hard_weights(self):
        """Discrete (hard) weights for inference."""
        hard_idx = mx.argmax(self.quant_logits, axis=0)
        scale_per_col = self.scales[:, self.idx].astype(self.dtype)
        return self.values[hard_idx] * scale_per_col, self.scales.astype(self.dtype)


# ---------------------------------------------------------------------------
# N-bit integer quantizer  (3, 4, 5, 6-bit)
# ---------------------------------------------------------------------------

class GumbelQuantizerInt:
    """N-bit Gumbel-Softmax quantizer with learned shift values.

    Parameters
    ----------
    Q : mx.array
        Initial quantized weights.
    scales : mx.array
        Per-group scales.
    group_size : int
        Channels per group.
    std : float
        Random perturbation std on initialization.
    strength : float
        Weight of init signal.
    dtype : mx.Dtype
    logits_dtype : mx.Dtype, optional
    bits : int
        Bit width (3, 4, 5, or 6).
    """

    def __init__(
        self,
        Q: mx.array,
        scales: mx.array,
        group_size: int,
        std: float,
        strength: float,
        dtype: mx.Dtype,
        logits_dtype: Optional[mx.Dtype] = None,
        bits: int = 3,
    ):
        self.dtype = dtype
        self.logits_dtype = logits_dtype or dtype
        self.bits = bits
        self.group_size = group_size
        self.idx = mx.arange(Q.shape[1]) // group_size

        scale_per_col = scales[:, self.idx]
        Q_init = (Q / scale_per_col).astype(self.dtype)

        self.min_grid = -(2 ** (bits - 1))
        self.max_grid = 2 ** (bits - 1) - 1

        # Shift values relative to the init weight
        self.shift_values = mx.array(
            [-2.0, -1.0, 0.0, 1.0, 2.0], dtype=Q.dtype
        )  # 5 values for bits >= 3

        cand = Q_init[None, ...] + self.shift_values.reshape(-1, 1, 1)
        valid_mask = (cand >= self.min_grid) & (cand <= self.max_grid)

        logits = -0.5 * (
            mx.zeros_like(Q_init).reshape(1, *Q_init.shape)
            - self.shift_values.reshape(-1, 1, 1)
        ) ** 2
        mean = (logits * valid_mask).sum(axis=0, keepdims=True) / valid_mask.sum(
            axis=0, keepdims=True
        )
        logits = logits - mean
        quant_logits = std * (mx.random.normal(logits.shape) + logits * strength)

        self.init = Q_init
        self.valid_mask = valid_mask
        self.quant_logits = quant_logits.astype(self.logits_dtype)
        self.scales = scales.astype(mx.float32)
        self.shift_values = self.shift_values.astype(self.dtype)

    def forward(self, temperature: float, scale_factor: float = 1.0, key=None):
        """Soft-quantize the weights.

        Returns
        -------
        soft_output : mx.array
            Dequantized weight tensor.
        soft_quant : mx.array
            (n_codes, rows, cols) soft assignment probabilities.
        """
        if key is None:
            key = mx.random.key(0)
        out, sq = _soft_quantize_int(
            self.quant_logits,
            self.scales,
            self.shift_values,
            self.valid_mask,
            self.idx,
            self.init,
            temperature,
            scale_factor,
            key,
        )
        return out, sq

    def get_hard_weights(self):
        """Discrete (hard) weights for inference."""
        masked = mx.where(self.valid_mask, self.quant_logits, -1e9)
        hard_shift_idx = mx.argmax(masked, axis=0)
        scale_per_col = self.scales[:, self.idx].astype(self.dtype)
        output = (self.init + self.shift_values[hard_shift_idx]) * scale_per_col
        return output, self.scales.astype(self.dtype)


# ---------------------------------------------------------------------------
# Lion optimizer (ported from PyTorch)
# ---------------------------------------------------------------------------

class Lion:
    """Lion optimizer, ported from ``lion_pytorch``."""

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.state: dict = {}
        for i, p in enumerate(self.params):
            self.state[i] = {"exp_avg": mx.zeros_like(p)}

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            s = self.state[i]
            m = s["exp_avg"]
            m = self.beta1 * m + (1 - self.beta1) * g
            if self.weight_decay != 0:
                p -= self.lr * self.weight_decay * p
            update = mx.sign(self.beta2 * m + (1 - self.beta2) * g)
            p -= self.lr * update
            s["exp_avg"] = m

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# Cosine LR scheduler with warmup
# ---------------------------------------------------------------------------

class CosineLRScheduler:
    """Cosine decay with linear warmup."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.step_count = 0
        self.initial_lrs = [pg["lr"] for pg in self._param_groups()]

    def _param_groups(self):
        # Lion stores lr as an attribute, not in param_groups
        if hasattr(self.optimizer, "param_groups"):
            return self.optimizer.param_groups
        # Lion-style: we fake param_groups
        return [{"lr": self.optimizer.lr}]

    def step(self):
        progress = self.step_count / max(self.total_steps - 1, 1)
        if self.step_count < self.warmup_steps:
            lr_frac = self.min_lr + (1 - self.min_lr) * (
                self.step_count / max(self.warmup_steps, 1)
            )
        else:
            lr_frac = self.min_lr + 0.5 * (1 - self.min_lr) * (
                1 + mx.cos(mx.pi * progress)
            )
            lr_frac = float(lr_frac)

        if hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr_frac
        if hasattr(self.optimizer, "param_groups"):
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_frac

        self.step_count += 1

"""GSQ / Humming checkpoint -> MLX affine conversion.

GSQ (ISTA-DASLab) ships its 2-bit checkpoints in the **Humming** on-disk
layout: exact-width packing (no sub-byte values padded into a wider container),
one bf16 scale per group, and no zero point.

    <layer>.weight        I32   [out_features, in_features / 16]   2-bit, LSB-first
    <layer>.weight_scale  BF16  [out_features, in_features / 128]  group_size = 128

MLX affine mode stores the same shape class:

    <module>.weight    U32   [out_features, in_features * bits / 32]
    <module>.scales    BF16  [out_features, in_features / group_size]
    <module>.biases    BF16  same as scales

So the *packing* coincides (16 two-bit values per 32-bit word, both LSB-first),
but the *codebook* does not: GSQ's levels are learned, MLX affine's are a
uniform grid with a bias. A bit-level transcode would silently change every
weight.

The conversion here is therefore an honest **dequantize-then-requantize**:

    1. decode the packed GSQ weights back to bf16,
    2. re-quantize with ``mx.quantize`` at the target width.

Step 2 is lossy. For 2-bit the two grids are both 4-level, so the error is
small, but it is not zero and it is not a format transcode. If you need the
GSQ grid exactly, use the Humming kernels (``pip install humming-kernels``)
rather than MLX.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "dequantize_exact_width",
    "quantize_to_mlx_affine",
    "gsq_to_mlx_affine",
    "PACKS_PER_WORD",
]

# Values packed per 32-bit word, by bit width. Both Humming and MLX pack
# LSB-first into a uint32, so these coincide for 2-bit.
PACKS_PER_WORD = {2: 16, 3: 10, 4: 8, 5: 6, 6: 5, 8: 4}


def _unpack_exact_width(packed: np.ndarray, bits: int) -> np.ndarray:
    """Unpack LSB-first sub-byte codes into integer values.

    ``packed`` is a ``(rows, cols)`` uint32/int32 array where each word holds
    ``PACKS_PER_WORD[bits]`` codes along the last axis.
    """
    if bits not in PACKS_PER_WORD:
        raise ValueError(f"unsupported bit width {bits}; supported: {sorted(PACKS_PER_WORD)}")
    per_word = PACKS_PER_WORD[bits]
    packed = np.asarray(packed)
    if packed.ndim != 2:
        raise ValueError(f"packed must be 2-D, got shape {packed.shape}")

    words = packed.astype(np.uint32)
    out = np.empty((words.shape[0], words.shape[1] * per_word), dtype=np.uint16)
    mask = np.uint32((1 << bits) - 1)
    for slot in range(per_word):
        out[:, slot::per_word] = (words >> np.uint32(bits * slot)) & mask
    return out


def dequantize_exact_width(
    packed: np.ndarray,
    scales: np.ndarray,
    bits: int,
    group_size: int = 128,
    offset: float = 0.0,
) -> np.ndarray:
    """Decode an exact-width packed, group-scaled tensor back to float.

    ``w[row, col] = scales[row, col // group_size] * (code[row, col] + offset)``

    ``offset`` handles a shifted grid. GSQ's 2-bit grid ``{-2, -1, 0, 1}``
    corresponds to ``offset = -2`` with codes ``{0, 1, 2, 3}``; MLX affine's
    uniform grid corresponds to ``offset = -(2**bits - 1) / 2``. Default 0
    assumes codes already represent signed levels.
    """
    codes = _unpack_exact_width(packed, bits)
    scales = np.asarray(scales)
    if scales.ndim != 2:
        raise ValueError(f"scales must be 2-D, got shape {scales.shape}")
    if scales.shape[0] != codes.shape[0]:
        raise ValueError(
            f"row mismatch: codes {codes.shape[0]} vs scales {scales.shape[0]}"
        )
    if codes.shape[1] % group_size != 0:
        raise ValueError(
            f"code columns ({codes.shape[1]}) not divisible by group_size ({group_size})"
        )

    # broadcast each scale across its group
    n_groups = codes.shape[1] // group_size
    if scales.shape[1] != n_groups:
        raise ValueError(
            f"expected {n_groups} scales per row, got {scales.shape[1]}"
        )
    scale_per_col = np.repeat(scales, group_size, axis=1)
    return scale_per_col * (codes.astype(np.float32) + np.float32(offset))


def quantize_to_mlx_affine(
    weights_bf16,
    bits: int,
    group_size: int = 64,
):
    """Quantize a bf16 weight array to MLX affine ``(q, scales, biases)``.

    Returns NumPy arrays. Use ``mx.array`` on each to build the MLX tensors;
    scales and biases must be cast to the *model's* dtype (bf16 for a bf16
    model) before saving -- fp16 scales on a bf16 model is a known corruption.
    """
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - mlx is optional
        raise ImportError(
            "quantize_to_mlx_affine needs MLX; install with `pip install mlx`"
        ) from exc

    w = np.asarray(weights_bf16, dtype=np.float32)
    arr = mx.array(w)
    q, scales, biases = mx.quantize(arr, group_size=group_size, bits=bits)
    mx.eval(q, scales, biases)
    return np.array(q), np.array(scales), np.array(biases)


def gsq_to_mlx_affine(
    packed: np.ndarray,
    scales: np.ndarray,
    src_bits: int = 2,
    src_group_size: int = 128,
    src_offset: float = -2.0,
    dst_bits: Optional[int] = None,
    dst_group_size: int = 64,
):
    """Full GSQ/Humming tensor -> MLX affine tensor conversion.

    Dequantizes the source tensor, then re-quantizes at the destination width.
    ``dst_bits`` defaults to ``src_bits``.

    Returns ``(q, scales, biases)`` as NumPy arrays.
    """
    dst_bits = src_bits if dst_bits is None else dst_bits
    bf16 = dequantize_exact_width(packed, scales, src_bits, src_group_size, src_offset)
    return quantize_to_mlx_affine(bf16, dst_bits, dst_group_size)

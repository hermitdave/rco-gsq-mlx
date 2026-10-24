"""GSQ / Humming packing tests.

The packing is checked by round-tripping: build a packed tensor from known
codes, dequantize, and confirm the recovered values. The unpack order is
LSB-first, 16 two-bit codes per 32-bit word -- the same layout MLX uses.
"""

import numpy as np
import pytest

from rco_mlx.gsq_format import (
    PACKS_PER_WORD,
    dequantize_exact_width,
    _unpack_exact_width,
)


def _pack(codes, bits):
    """Inverse of ``_unpack_exact_width`` for test construction."""
    per_word = PACKS_PER_WORD[bits]
    codes = np.asarray(codes, dtype=np.uint32)
    rows, cols = codes.shape
    assert cols % per_word == 0
    packed = np.zeros((rows, cols // per_word), dtype=np.uint32)
    for slot in range(per_word):
        packed |= (codes[:, slot::per_word] & np.uint32((1 << bits) - 1)) << np.uint32(
            bits * slot
        )
    return packed


def test_packs_per_word_are_lsb_first_32bit():
    assert PACKS_PER_WORD[2] == 16
    assert PACKS_PER_WORD[4] == 8
    assert PACKS_PER_WORD[8] == 4


def test_unpack_round_trips_2bit():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 4, size=(3, 32), dtype=np.uint16)
    packed = _pack(codes, 2)

    recovered = _unpack_exact_width(packed, 2)
    assert np.array_equal(recovered, codes)


def test_unpack_round_trips_4bit():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 16, size=(2, 16), dtype=np.uint16)
    packed = _pack(codes, 4)

    recovered = _unpack_exact_width(packed, 4)
    assert np.array_equal(recovered, codes)


def test_dequantize_applies_the_group_scale():
    # 2 rows, 64 columns, group_size 32 -> 2 scales per row
    codes = np.zeros((2, 64), dtype=np.uint16)
    codes[0, :32] = 1
    codes[0, 32:] = 3
    codes[1, :] = 2
    packed = _pack(codes, 2)

    scales = np.array([[1.0, 10.0], [5.0, 5.0]], dtype=np.float32)
    out = dequantize_exact_width(packed, scales, bits=2, group_size=32)

    assert out.shape == (2, 64)
    assert np.allclose(out[0, 0], 1.0)
    assert np.allclose(out[0, 32], 30.0)
    assert np.allclose(out[1, 0], 10.0)


def test_dequantize_offset_shifts_the_grid():
    """GSQ's {-2,-1,0,1} grid is codes {0,1,2,3} with offset -2."""
    codes = np.tile(np.array([[0, 1, 2, 3]], dtype=np.uint16), (1, 8))  # 32 columns
    packed = _pack(codes, 2)
    scales = np.array([[1.0]], dtype=np.float32)

    out = dequantize_exact_width(packed, scales, bits=2, group_size=32, offset=-2.0)
    assert np.allclose(out[0], np.array([-2.0, -1.0, 0.0, 1.0] * 8))


def test_dequantize_rejects_wrong_scale_count():
    codes = np.zeros((1, 32), dtype=np.uint16)
    packed = _pack(codes, 2)
    with pytest.raises(ValueError, match="expected .* scales per row"):
        dequantize_exact_width(packed, np.zeros((1, 3), dtype=np.float32), bits=2, group_size=16)


def test_dequantize_rejects_unsupported_bits():
    with pytest.raises(ValueError, match="unsupported bit width"):
        _unpack_exact_width(np.zeros((1, 4), dtype=np.uint32), 7)


def test_unpack_rejects_non_2d():
    with pytest.raises(ValueError, match="packed must be 2-D"):
        _unpack_exact_width(np.zeros((2, 2, 2), dtype=np.uint32), 2)


def test_group_size_must_divide_columns():
    codes = np.zeros((1, 32), dtype=np.uint16)
    packed = _pack(codes, 2)
    # 32 columns are not divisible by 24
    with pytest.raises(ValueError, match="not divisible by group_size"):
        dequantize_exact_width(packed, np.zeros((1, 2), dtype=np.float32), bits=2, group_size=24)

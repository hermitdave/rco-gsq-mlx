"""Tests for gsq_quantizer."""

import numpy as np
import pytest

import mlx.core as mx

from rco_mlx.gsq_quantizer import (
    GumbelQuantizer2Bit,
    GumbelQuantizerInt,
    effective_bits,
    PACKS_PER_WORD,
)


class TestEffectiveBits:
    def test_2bit_g64(self):
        assert effective_bits(2, 64) == pytest.approx(2.25)

    def test_4bit_g64(self):
        assert effective_bits(4, 64) == pytest.approx(4.25)

    def test_8bit_g32(self):
        assert effective_bits(8, 32) == pytest.approx(8.5)

    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            effective_bits(1, 64)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError):
            effective_bits(4, 16)


class TestGumbelQuantizer2Bit:
    @pytest.fixture
    def quantizer(self):
        rng = np.random.default_rng(0)
        out_f, in_f = 32, 64
        Q = mx.array(rng.standard_normal((out_f, in_f)).astype(np.float32))
        scales = mx.array(
            np.abs(rng.standard_normal((out_f, in_f // 64))).astype(np.float32) + 0.5
        )
        return GumbelQuantizer2Bit(
            Q=Q, scales=scales, group_size=64, std=0.01, strength=1.0,
            dtype=mx.float32,
        )

    def test_forward_shape(self, quantizer):
        out, sq = quantizer.forward(temperature=1.0, scale_factor=1.0)
        assert out.shape == quantizer.quant_logits.shape[1:]  # (rows, cols)

    def test_hard_weights_shape(self, quantizer):
        out, scales = quantizer.get_hard_weights()
        assert out.shape == (32, 64)

    def test_hard_values_in_grid(self, quantizer):
        out, _ = quantizer.get_hard_weights()
        scale_per_col = quantizer.scales[:, quantizer.idx]
        raw = np.array(out) / np.array(scale_per_col)
        # All values should be in {-2, -1, 0, 1}
        unique = set(np.unique(np.round(raw).astype(int)))
        assert unique.issubset({-2, -1, 0, 1})

    def test_temperature_anneal(self, quantizer):
        out_hot, sq_hot = quantizer.forward(temperature=2.0, scale_factor=1.0, key=mx.random.key(1))
        out_cold, sq_cold = quantizer.forward(temperature=0.1, scale_factor=1.0, key=mx.random.key(1))
        # At low temp the soft assignment should be sharper (higher max probability)
        max_prob_hot = float(mx.max(sq_hot))
        max_prob_cold = float(mx.max(sq_cold))
        assert max_prob_cold >= max_prob_hot * 0.9  # cold should be sharper


class TestGumbelQuantizerInt:
    @pytest.fixture
    def quantizer_3bit(self):
        rng = np.random.default_rng(0)
        out_f, in_f = 32, 64
        Q = mx.array(rng.standard_normal((out_f, in_f)).astype(np.float32))
        scales = mx.array(
            np.abs(rng.standard_normal((out_f, in_f // 64))).astype(np.float32) + 0.5
        )
        return GumbelQuantizerInt(
            Q=Q, scales=scales, group_size=64, std=0.01, strength=1.0,
            dtype=mx.float32, bits=3,
        )

    def test_forward_shape(self, quantizer_3bit):
        out, sq = quantizer_3bit.forward(temperature=1.0, scale_factor=1.0)
        assert out.shape == (32, 64)

    def test_hard_values_in_range(self, quantizer_3bit):
        out, _ = quantizer_3bit.get_hard_weights()
        scale_per_col = quantizer_3bit.scales[:, quantizer_3bit.idx]
        raw = np.array(out) / np.array(scale_per_col)
        # 3-bit: values in [-4, 3]
        assert int(raw.min()) >= -4
        assert int(raw.max()) <= 3


class TestPacksPerWord:
    def test_standard_widths(self):
        assert PACKS_PER_WORD[2] == 16
        assert PACKS_PER_WORD[3] == 10
        assert PACKS_PER_WORD[4] == 8
        assert PACKS_PER_WORD[8] == 4

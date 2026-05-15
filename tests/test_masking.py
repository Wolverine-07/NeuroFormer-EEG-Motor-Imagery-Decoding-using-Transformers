"""
Unit Tests for Masking functions.

Tests verify:
  1. Padding mask correctly identifies pad tokens
  2. Causal mask is lower-triangular
  3. Target mask combines padding + causal correctly
  4. Masks have correct shapes for broadcasting with attention
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.transformer.utils import (
    generate_padding_mask,
    generate_causal_mask,
    generate_source_mask,
    generate_target_mask,
    LayerNorm,
)


class TestPaddingMask:
    """Tests for padding mask generation."""

    def test_shape(self):
        """Padding mask should have shape (batch, 1, 1, seq_len)."""
        seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        mask = generate_padding_mask(seq, pad_idx=0)
        assert mask.shape == (2, 1, 1, 5)

    def test_pad_positions_are_false(self):
        """Padding positions should be False (0)."""
        seq = torch.tensor([[1, 2, 3, 0, 0]])
        mask = generate_padding_mask(seq, pad_idx=0)

        assert mask[0, 0, 0, 0].item() == True   # token 1 — not padded
        assert mask[0, 0, 0, 1].item() == True   # token 2 — not padded
        assert mask[0, 0, 0, 2].item() == True   # token 3 — not padded
        assert mask[0, 0, 0, 3].item() == False  # pad
        assert mask[0, 0, 0, 4].item() == False  # pad

    def test_no_padding(self):
        """When no padding, all positions should be True."""
        seq = torch.tensor([[1, 2, 3, 4, 5]])
        mask = generate_padding_mask(seq, pad_idx=0)
        assert mask.all()


class TestCausalMask:
    """Tests for causal (look-ahead) mask."""

    def test_shape(self):
        """Causal mask should have shape (1, 1, size, size)."""
        mask = generate_causal_mask(5)
        assert mask.shape == (1, 1, 5, 5)

    def test_lower_triangular(self):
        """Causal mask should be lower triangular."""
        mask = generate_causal_mask(4)
        expected = torch.tensor([[
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]
        ]], dtype=torch.float)
        assert torch.equal(mask, expected)

    def test_first_position(self):
        """First position should only attend to itself."""
        mask = generate_causal_mask(6)
        assert mask[0, 0, 0, 0] == 1  # Can attend to self
        assert mask[0, 0, 0, 1] == 0  # Cannot attend to future

    def test_last_position(self):
        """Last position should attend to all positions."""
        size = 5
        mask = generate_causal_mask(size)
        assert mask[0, 0, size-1, :].sum() == size


class TestTargetMask:
    """Tests for combined target mask (padding + causal)."""

    def test_shape(self):
        """Target mask should have shape (batch, 1, tgt_len, tgt_len)."""
        tgt = torch.tensor([[1, 2, 3, 0, 0]])
        mask = generate_target_mask(tgt, pad_idx=0)
        assert mask.shape == (1, 1, 5, 5)

    def test_combines_padding_and_causal(self):
        """Mask should block both future and padded positions."""
        tgt = torch.tensor([[1, 2, 0]])  # Position 2 is padded
        mask = generate_target_mask(tgt, pad_idx=0)

        # Position 0 attends to position 0 only (causal)
        assert mask[0, 0, 0, 0] == True
        assert mask[0, 0, 0, 1] == False  # future
        assert mask[0, 0, 0, 2] == False  # future + padded

        # Position 1 attends to positions 0 and 1 (causal), not 2 (padded)
        assert mask[0, 0, 1, 0] == True
        assert mask[0, 0, 1, 1] == True
        assert mask[0, 0, 1, 2] == False  # padded


class TestLayerNorm:
    """Tests for custom LayerNorm."""

    def test_output_shape(self):
        """LayerNorm should preserve input shape."""
        norm = LayerNorm(64)
        x = torch.randn(2, 10, 64)
        output = norm(x)
        assert output.shape == (2, 10, 64)

    def test_normalized_stats(self):
        """After normalization, mean ≈ 0 and std ≈ 1."""
        norm = LayerNorm(64)
        x = torch.randn(2, 10, 64) * 5 + 3  # Non-zero mean, non-unit variance

        output = norm(x)

        # Mean should be close to beta (0) and std close to gamma (1)
        # after initialization
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)
        assert torch.allclose(std, torch.ones_like(std), atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

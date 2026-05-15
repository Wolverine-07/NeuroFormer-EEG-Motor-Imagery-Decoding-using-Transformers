"""
Unit Tests for Transformer Attention Mechanism.

Tests verify:
  1. Output tensor shapes are correct
  2. Attention weights sum to 1 (valid probability distribution)
  3. Masking works correctly (masked positions get ~0 attention weight)
  4. Multi-head attention produces correct shapes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.transformer.attention import scaled_dot_product_attention, MultiHeadAttention


class TestScaledDotProductAttention:
    """Tests for scaled_dot_product_attention (Paper Section 3.2.1)."""

    def test_output_shape(self):
        """Output shape should be (batch, heads, seq_q, d_v)."""
        batch, heads, seq_q, seq_k, d_k = 2, 4, 10, 10, 16
        Q = torch.randn(batch, heads, seq_q, d_k)
        K = torch.randn(batch, heads, seq_k, d_k)
        V = torch.randn(batch, heads, seq_k, d_k)

        output, weights = scaled_dot_product_attention(Q, K, V)

        assert output.shape == (batch, heads, seq_q, d_k)
        assert weights.shape == (batch, heads, seq_q, seq_k)

    def test_attention_weights_sum_to_one(self):
        """Attention weights should sum to 1 across the key dimension."""
        Q = torch.randn(2, 4, 10, 16)
        K = torch.randn(2, 4, 10, 16)
        V = torch.randn(2, 4, 10, 16)

        _, weights = scaled_dot_product_attention(Q, K, V)

        # Sum along key dimension (last dim) should be ~1.0
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_masking(self):
        """Masked positions should get ~0 attention weight."""
        batch, heads, seq_len, d_k = 1, 1, 4, 8
        Q = torch.randn(batch, heads, seq_len, d_k)
        K = torch.randn(batch, heads, seq_len, d_k)
        V = torch.randn(batch, heads, seq_len, d_k)

        # Mask: position 3 is masked (0 = masked, 1 = attend)
        mask = torch.ones(1, 1, seq_len, seq_len)
        mask[:, :, :, 3] = 0  # Block attention to position 3

        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Attention to position 3 should be ~0
        assert torch.allclose(weights[:, :, :, 3], torch.zeros(1, 1, seq_len), atol=1e-6)

    def test_causal_mask(self):
        """With causal mask, future positions should get ~0 weight."""
        seq_len, d_k = 4, 8
        Q = torch.randn(1, 1, seq_len, d_k)
        K = torch.randn(1, 1, seq_len, d_k)
        V = torch.randn(1, 1, seq_len, d_k)

        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))

        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Position 0 should only attend to position 0
        assert weights[0, 0, 0, 1].item() < 1e-6
        assert weights[0, 0, 0, 2].item() < 1e-6
        assert weights[0, 0, 0, 3].item() < 1e-6


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention (Paper Section 3.2.2)."""

    def test_output_shape(self):
        """Output shape should be (batch, seq_len, d_model)."""
        batch, seq_len, d_model, num_heads = 2, 10, 64, 4
        mha = MultiHeadAttention(d_model, num_heads)

        x = torch.randn(batch, seq_len, d_model)
        output = mha(x, x, x)

        assert output.shape == (batch, seq_len, d_model)

    def test_cross_attention_shapes(self):
        """Cross-attention with different Q and K/V lengths."""
        batch, d_model, num_heads = 2, 64, 4
        q_len, kv_len = 8, 12

        mha = MultiHeadAttention(d_model, num_heads)

        Q = torch.randn(batch, q_len, d_model)
        K = torch.randn(batch, kv_len, d_model)
        V = torch.randn(batch, kv_len, d_model)

        output = mha(Q, K, V)
        assert output.shape == (batch, q_len, d_model)

    def test_d_model_divisibility_check(self):
        """Should raise error if d_model not divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(d_model=100, num_heads=3)

    def test_attention_weights_stored(self):
        """Attention weights should be stored for visualization."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        mha(x, x, x)

        assert mha.attention_weights is not None
        assert mha.attention_weights.shape == (2, 4, 10, 10)

    def test_gradient_flow(self):
        """Gradients should flow through the attention mechanism."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64, requires_grad=True)

        output = mha(x, x, x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

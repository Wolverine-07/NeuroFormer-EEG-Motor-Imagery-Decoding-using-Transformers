"""
Unit Tests for Encoder and Decoder components.

Tests verify:
  1. Encoder layer and stack produce correct shapes
  2. Decoder layer and stack produce correct shapes
  3. Full Transformer forward pass works end-to-end
  4. Greedy decode produces valid output
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from src.transformer.encoder import EncoderLayer, Encoder
from src.transformer.decoder import DecoderLayer, Decoder
from src.transformer.transformer import Transformer
from src.training.trainer import greedy_decode


class TestEncoder:
    """Tests for Encoder components."""

    def test_encoder_layer_shape(self):
        """EncoderLayer should preserve input shape."""
        batch, seq_len, d_model = 2, 10, 64
        layer = EncoderLayer(d_model=64, num_heads=4, d_ff=128)

        x = torch.randn(batch, seq_len, d_model)
        output = layer(x)

        assert output.shape == (batch, seq_len, d_model)

    def test_encoder_stack_shape(self):
        """Encoder stack should preserve input shape."""
        batch, seq_len, d_model = 2, 10, 64
        encoder = Encoder(d_model=64, num_heads=4, d_ff=128, num_layers=3)

        x = torch.randn(batch, seq_len, d_model)
        output = encoder(x)

        assert output.shape == (batch, seq_len, d_model)

    def test_encoder_with_mask(self):
        """Encoder should work with padding mask."""
        batch, seq_len, d_model = 2, 10, 64
        encoder = Encoder(d_model=64, num_heads=4, d_ff=128, num_layers=2)

        x = torch.randn(batch, seq_len, d_model)
        mask = torch.ones(batch, 1, 1, seq_len)
        mask[0, :, :, -3:] = 0  # Mask last 3 positions of first sample

        output = encoder(x, mask)
        assert output.shape == (batch, seq_len, d_model)


class TestDecoder:
    """Tests for Decoder components."""

    def test_decoder_layer_shape(self):
        """DecoderLayer should produce correct shape."""
        batch, tgt_len, src_len, d_model = 2, 8, 10, 64
        layer = DecoderLayer(d_model=64, num_heads=4, d_ff=128)

        x = torch.randn(batch, tgt_len, d_model)
        memory = torch.randn(batch, src_len, d_model)

        output = layer(x, memory)
        assert output.shape == (batch, tgt_len, d_model)

    def test_decoder_stack_shape(self):
        """Decoder stack should produce correct shape."""
        batch, tgt_len, src_len, d_model = 2, 8, 10, 64
        decoder = Decoder(d_model=64, num_heads=4, d_ff=128, num_layers=3)

        x = torch.randn(batch, tgt_len, d_model)
        memory = torch.randn(batch, src_len, d_model)

        output = decoder(x, memory)
        assert output.shape == (batch, tgt_len, d_model)


class TestTransformer:
    """Tests for the full Transformer model."""

    def setup_method(self):
        """Create a tiny transformer for testing."""
        self.model = Transformer.build_model(
            src_vocab_size=20,
            tgt_vocab_size=20,
            variant="tiny",
        )

    def test_forward_shape(self):
        """Forward pass should produce (batch, tgt_len, vocab_size)."""
        src = torch.randint(3, 20, (2, 10))
        tgt = torch.randint(3, 20, (2, 8))

        output = self.model(src, tgt)
        assert output.shape == (2, 8, 20)

    def test_encode_shape(self):
        """Encode should produce (batch, src_len, d_model)."""
        src = torch.randint(3, 20, (2, 10))
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        memory = self.model.encode(src, src_mask)
        assert memory.shape == (2, 10, self.model.d_model)

    def test_build_model_variants(self):
        """Factory should create models with correct configs."""
        base = Transformer.build_model(20, 20, "base")
        assert base.d_model == 512

        big = Transformer.build_model(20, 20, "big")
        assert big.d_model == 1024

        tiny = Transformer.build_model(20, 20, "tiny")
        assert tiny.d_model == 64

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters."""
        count = self.model.count_parameters()
        assert count > 0
        # Tiny model should be small
        assert count < 1_000_000

    def test_greedy_decode(self):
        """Greedy decode should produce a valid sequence."""
        src = torch.randint(3, 20, (1, 10))

        decoded = greedy_decode(
            self.model, src, max_len=12, start_token=1, end_token=2
        )

        assert decoded.shape[0] == 1  # Batch size 1
        assert decoded.shape[1] >= 1  # At least the start token
        assert decoded[0, 0].item() == 1  # Starts with SOS

    def test_gradient_flow_full_model(self):
        """Gradients should flow through the entire model."""
        src = torch.randint(3, 20, (2, 10))
        tgt = torch.randint(3, 20, (2, 8))

        output = self.model(src, tgt)
        loss = output.sum()
        loss.backward()

        # Check that all parameters received gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

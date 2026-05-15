"""
Transformer Decoder — "Attention Is All You Need" Section 3.1

From the paper:
    "The decoder is also composed of a stack of N = 6 identical layers. In
     addition to the two sub-layers in each encoder layer, the decoder inserts
     a third sub-layer, which performs multi-head attention over the output of
     the encoder stack."

Architecture of each decoder layer:
    Input (shifted right target sequence)
      → Masked Multi-Head Self-Attention  (prevents attending to future positions)
      → Add & Norm
      → Multi-Head Cross-Attention        (attends to encoder output)
      → Add & Norm
      → Position-wise Feed-Forward
      → Add & Norm
    Output

    "We also modify the self-attention sub-layer in the decoder stack to prevent
     positions from attending to subsequent positions. This masking, combined
     with the fact that the output embeddings are offset by one position, ensures
     that the predictions for position i can depend only on the known outputs
     at positions less than i."

The decoder generates the output sequence one element at a time (auto-regressively).
At each step, it consumes the previously generated symbols as additional input.
"""

import torch
import torch.nn as nn

from src.transformer.attention import MultiHeadAttention
from src.transformer.feed_forward import PositionwiseFeedForward
from src.transformer.utils import LayerNorm, SublayerConnection, clone_modules


class DecoderLayer(nn.Module):
    """
    Single Decoder Layer (Paper Section 3.1).

    Each decoder layer has three sub-layers:
    1. Masked Multi-Head Self-Attention (with causal mask)
    2. Multi-Head Cross-Attention (encoder-decoder attention)
    3. Position-wise Feed-Forward Network

    The cross-attention layer is the key difference from the encoder:
        "The decoder inserts a third sub-layer, which performs multi-head
         attention over the output of the encoder stack. In this, queries
         come from the previous decoder layer, and the keys and values
         come from the output of the encoder."

    Args:
        d_model: Model dimension (paper default: 512)
        num_heads: Number of attention heads (paper default: 8)
        d_ff: Feed-forward inner dimension (paper default: 2048)
        dropout: Dropout rate (paper default: 0.1)
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()

        # Sub-layer 1: Masked Self-Attention
        # Uses causal mask to prevent attending to future positions
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Cross-Attention (encoder-decoder attention)
        # Q from decoder, K and V from encoder output
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 3: Position-wise Feed-Forward
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Three sublayer connections, one per sub-layer
        self.sublayer_connections = clone_modules(
            SublayerConnection(d_model, dropout), 3
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for a single decoder layer.

        Args:
            x: Decoder input (batch, tgt_len, d_model)
            encoder_output: Output from the encoder (batch, src_len, d_model)
            src_mask: Source mask for cross-attention (batch, 1, 1, src_len)
            tgt_mask: Target mask for self-attention (batch, 1, tgt_len, tgt_len)
                      Combines padding mask + causal mask

        Returns:
            output: (batch, tgt_len, d_model)
        """
        # Sub-layer 1: Masked Self-Attention
        # Q = K = V = x, with causal mask to prevent attending to future
        x = self.sublayer_connections[0](
            x, lambda x: self.self_attention(x, x, x, mask=tgt_mask)
        )

        # Sub-layer 2: Cross-Attention (Encoder-Decoder Attention)
        # Q = decoder output, K = V = encoder output
        # This is how the decoder "reads" the encoder's representation
        # The paper: "queries come from the previous decoder layer, and
        # the memory keys and values come from the output of the encoder"
        x = self.sublayer_connections[1](
            x, lambda x: self.cross_attention(x, encoder_output, encoder_output, mask=src_mask)
        )

        # Sub-layer 3: Feed-Forward
        x = self.sublayer_connections[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    """
    Transformer Decoder — Stack of N identical layers (Paper Section 3.1).

    "The decoder is also composed of a stack of N = 6 identical layers."

    The decoder generates output auto-regressively: at each time step,
    it takes the encoder output and all previously generated tokens to
    produce the next token.

    Args:
        d_model: Model dimension (paper default: 512)
        num_heads: Number of attention heads (paper default: 8)
        d_ff: Feed-forward inner dimension (paper default: 2048)
        num_layers: Number of decoder layers N (paper default: 6)
        dropout: Dropout rate (paper default: 0.1)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Create N identical decoder layers
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.layers = clone_modules(decoder_layer, num_layers)

        # Final layer normalization
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Pass through N decoder layers sequentially.

        Args:
            x: Embedded target input (batch, tgt_len, d_model)
            encoder_output: Output from encoder (batch, src_len, d_model)
            src_mask: Source padding mask (batch, 1, 1, src_len)
            tgt_mask: Target mask (batch, 1, tgt_len, tgt_len) — padding + causal

        Returns:
            Decoder output (batch, tgt_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)

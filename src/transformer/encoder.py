"""
Transformer Encoder — "Attention Is All You Need" Section 3.1

From the paper:
    "The encoder is composed of a stack of N = 6 identical layers. Each layer
     has two sub-layers. The first is a multi-head self-attention mechanism,
     and the second is a simple, position-wise fully connected feed-forward
     network. We employ a residual connection around each of the two sub-layers,
     followed by layer normalization."

Architecture of each encoder layer:
    Input
      → Multi-Head Self-Attention
      → Add & Norm (residual + layer norm)
      → Position-wise Feed-Forward
      → Add & Norm (residual + layer norm)
    Output

The encoder maps an input sequence of symbol representations (x₁, ..., xₙ)
to a sequence of continuous representations z = (z₁, ..., zₙ).
"""

import torch
import torch.nn as nn

from src.transformer.attention import MultiHeadAttention
from src.transformer.feed_forward import PositionwiseFeedForward
from src.transformer.utils import LayerNorm, SublayerConnection, clone_modules


class EncoderLayer(nn.Module):
    """
    Single Encoder Layer (Paper Section 3.1).

    Each encoder layer consists of two sub-layers:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network

    Each sub-layer is wrapped with a residual connection and layer normalization:
        output = LayerNorm(x + Sublayer(x))

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

        # Sub-layer 1: Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Position-wise Feed-Forward Network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Two sublayer connections (residual + layer norm), one per sub-layer
        self.sublayer_connections = clone_modules(
            SublayerConnection(d_model, dropout), 2
        )

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for a single encoder layer.

        Args:
            x: Input (batch, seq_len, d_model)
            src_mask: Source mask (batch, 1, 1, seq_len) — masks padding tokens

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Sub-layer 1: Self-Attention
        # In self-attention, Q = K = V = x (the layer input)
        x = self.sublayer_connections[0](
            x, lambda x: self.self_attention(x, x, x, mask=src_mask)
        )

        # Sub-layer 2: Feed-Forward
        x = self.sublayer_connections[1](x, self.feed_forward)

        return x


class Encoder(nn.Module):
    """
    Transformer Encoder — Stack of N identical layers (Paper Section 3.1).

    "The encoder is composed of a stack of N = 6 identical layers."

    After the stack, a final layer normalization is applied.
    (This is part of the Pre-LN convention used in our implementation.)

    Args:
        d_model: Model dimension (paper default: 512)
        num_heads: Number of attention heads (paper default: 8)
        d_ff: Feed-forward inner dimension (paper default: 2048)
        num_layers: Number of encoder layers N (paper default: 6)
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

        # Create N identical encoder layers
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.layers = clone_modules(encoder_layer, num_layers)

        # Final layer normalization (applied after the last encoder layer)
        self.norm = LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Pass the input through N encoder layers sequentially.

        Each layer performs:
            x = EncoderLayer(x, src_mask)

        The output of one layer becomes the input to the next.

        Args:
            x: Embedded input (batch, seq_len, d_model)
               (embeddings + positional encoding have already been applied)
            src_mask: Source mask (batch, 1, 1, src_len)

        Returns:
            Encoder output (batch, seq_len, d_model)
            This is the "continuous representation z" mentioned in the paper.
        """
        for layer in self.layers:
            x = layer(x, src_mask)

        # Final normalization
        return self.norm(x)

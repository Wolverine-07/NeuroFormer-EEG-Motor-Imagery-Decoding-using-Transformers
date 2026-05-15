"""
Position-wise Feed-Forward Networks — "Attention Is All You Need" Section 3.3

From the paper:
    "In addition to attention sub-layers, each of the layers in our encoder and
     decoder contains a fully connected feed-forward network, which is applied
     to each position separately and identically. This consists of two linear
     transformations with a ReLU activation in between."

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    "While the linear transformations are the same across different positions,
     they use different parameters from layer to layer. Another way of describing
     this is as two convolutions with kernel size 1."

    The paper uses:
        d_model = 512 (input/output dimension)
        d_ff = 2048 (inner-layer dimension)
"""

import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Paper Section 3.3, Equation 2).

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    This is applied to each position independently and identically.
    It can be thought of as two 1×1 convolutions, or equivalently,
    a bottleneck that expands the representation to d_ff dimensions
    and then projects it back to d_model.

    The "position-wise" means the same transformation is applied at
    every position in the sequence, but with the same parameters
    (unlike attention, which mixes information across positions).

    Args:
        d_model: Input and output dimension (paper default: 512)
        d_ff: Inner-layer dimension (paper default: 2048)
        dropout: Dropout rate (paper default: 0.1)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # W_1: (d_model → d_ff) — expand
        self.linear_1 = nn.Linear(d_model, d_ff)

        # W_2: (d_ff → d_model) — project back
        self.linear_2 = nn.Linear(d_ff, d_model)

        # Dropout applied after ReLU (common practice, also used in
        # the reference implementation)
        self.dropout = nn.Dropout(p=dropout)

        # ReLU activation: max(0, x)
        # The paper uses ReLU. Later work (GPT, etc.) uses GELU,
        # but we stay faithful to the original paper.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))

"""
Attention Mechanisms — "Attention Is All You Need" (Vaswani et al., 2017)

Implements Section 3.2 of the paper:
  - Scaled Dot-Product Attention
  - Multi-Head Attention

References:
  Paper: https://arxiv.org/abs/1706.03762
  Section 3.2.1: Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
  Section 3.2.2: Multi-Head Attention
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention (Paper Section 3.2.1, Equation 1).

    Computes attention as:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    The scaling factor 1/sqrt(d_k) counteracts the effect of large dot products
    pushing the softmax into regions with extremely small gradients. As the paper
    states: "We suspect that for large values of d_k, the dot products grow large
    in magnitude, pushing the softmax function into regions where it has extremely
    small gradients."

    Args:
        query:   (batch, heads, seq_len_q, d_k)
        key:     (batch, heads, seq_len_k, d_k)
        value:   (batch, heads, seq_len_v, d_v)  — seq_len_v == seq_len_k
        mask:    Optional broadcastable mask. Positions with True/1 are MASKED
                 (will be filled with -inf before softmax). Shape should be
                 broadcastable to (batch, heads, seq_len_q, seq_len_k).
        dropout: Optional dropout module applied to attention weights.

    Returns:
        output:           (batch, heads, seq_len_q, d_v)
        attention_weights: (batch, heads, seq_len_q, seq_len_k)
    """
    d_k = query.size(-1)

    # Q K^T / sqrt(d_k)  →  (batch, heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask: fill masked positions with -inf so softmax gives them ~0 weight.
    # The paper uses masking in two places:
    #   1. Padding mask — ignore <pad> tokens in the input
    #   2. Causal mask — in the decoder, prevent attending to future positions
    #      "We need to prevent leftward information flow in the decoder to
    #       preserve the auto-regressive property." (Section 3.1)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # softmax over the key dimension (last dimension)
    attention_weights = F.softmax(scores, dim=-1)

    # Apply dropout to attention weights (Paper Section 5.4 mentions residual
    # dropout; attention dropout is a common addition from the reference implementation)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Weighted sum of values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Paper Section 3.2.2).

    Instead of performing a single attention function with d_model-dimensional
    keys, values, and queries, the paper proposes to linearly project the queries,
    keys, and values h times with different, learned linear projections to d_k,
    d_k, and d_v dimensions respectively. On each of these projected versions,
    the attention function is performed in parallel, yielding d_v-dimensional
    output values.

    From the paper:
        "Multi-head attention allows the model to jointly attend to information
         from different representation subspaces at different positions."

    The paper uses h=8 parallel heads. For each head:
        d_k = d_v = d_model / h = 512 / 8 = 64

    Args:
        d_model: Model dimension (paper default: 512)
        num_heads: Number of parallel attention heads (paper default: 8)
        dropout: Dropout rate for attention weights (paper default: 0.1)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
            f"Paper uses d_model=512, num_heads=8 → d_k=64."
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # d_k = d_v = d_model / h (paper notation)

        # Linear projections: W_i^Q, W_i^K, W_i^V, W^O
        # The paper states: "the projections are parameter matrices
        # W_i^Q ∈ R^{d_model × d_k}, W_i^K ∈ R^{d_model × d_k},
        # W_i^V ∈ R^{d_model × d_v}, W^O ∈ R^{h·d_v × d_model}"
        #
        # We implement all h heads' projections as a single large linear layer
        # for computational efficiency, then reshape to separate heads.
        self.W_q = nn.Linear(d_model, d_model)  # Projects to h × d_k = d_model
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # Output projection W^O

        self.dropout = nn.Dropout(p=dropout)
        self.attention_weights = None  # Stored for visualization (Phase 4)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention.

        In the Transformer, this is called in three different ways:
        1. Encoder self-attention:  Q=K=V = encoder input
        2. Decoder self-attention:  Q=K=V = decoder input (with causal mask)
        3. Decoder cross-attention: Q = decoder, K=V = encoder output

        Args:
            query: (batch, seq_len_q, d_model)
            key:   (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)  — seq_len_v == seq_len_k
            mask:  Optional mask, broadcastable to (batch, 1, seq_len_q, seq_len_k)

        Returns:
            output: (batch, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # 1. Linear projections: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        # 2. Reshape to separate heads: → (batch, seq_len, h, d_k)
        # 3. Transpose for attention: → (batch, h, seq_len, d_k)
        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 4. Apply attention on all heads in parallel
        # attn_output: (batch, h, seq_len_q, d_k)
        # attn_weights: (batch, h, seq_len_q, seq_len_k)
        attn_output, attn_weights = scaled_dot_product_attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # Store attention weights for visualization (detached from computation graph)
        self.attention_weights = attn_weights.detach()

        # 5. Concatenate heads: (batch, h, seq_len_q, d_k) → (batch, seq_len_q, h * d_k)
        # The paper: "These are concatenated and once again projected"
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # 6. Final linear projection W^O
        output = self.W_o(attn_output)

        return output

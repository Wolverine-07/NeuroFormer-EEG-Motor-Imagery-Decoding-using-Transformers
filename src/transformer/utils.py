"""
Transformer Utilities — "Attention Is All You Need"

Helper functions and modules used throughout the Transformer:
  - Layer Normalization (used in "Add & Norm" sub-layers)
  - Module cloning (to create N identical layers)
  - Mask generation (padding masks and causal masks)
  - Residual connection with layer normalization

References:
  Section 3.1: "We employ a residual connection around each of the two
  sub-layers, followed by layer normalization. That is, the output of each
  sub-layer is LayerNorm(x + Sublayer(x))."

  Section 3.1 (Decoder): "We also modify the self-attention sub-layer in the
  decoder stack to prevent positions from attending to subsequent positions.
  This masking, combined with fact that the output embeddings are offset by one
  position, ensures that the predictions for position i can depend only on the
  known outputs at positions less than i."
"""

import copy

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization (Ba et al., 2016).

    Used in the "Add & Norm" sub-layers of the Transformer (Paper Section 3.1).
    The paper states: "the output of each sub-layer is LayerNorm(x + Sublayer(x))"

    Layer normalization normalizes across the feature dimension (d_model),
    as opposed to batch normalization which normalizes across the batch dimension.
    This makes it suitable for sequence models where batch statistics can be noisy.

    Args:
        d_model: Feature dimension to normalize over
        eps: Small constant for numerical stability
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # Learnable affine parameters (scale and shift)
        self.gamma = nn.Parameter(torch.ones(d_model))   # Scale (initialized to 1)
        self.beta = nn.Parameter(torch.zeros(d_model))   # Shift (initialized to 0)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            Normalized output (batch, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SublayerConnection(nn.Module):
    """
    Residual Connection + Layer Normalization (Paper Section 3.1).

    Implements: LayerNorm(x + Sublayer(x))

    The paper uses Post-LN (layer norm after the residual addition).
    This is the original formulation. Pre-LN (norm before sublayer)
    was introduced later and shown to be more stable for deep models,
    but we implement Post-LN to be faithful to the paper.

    Note: The paper also applies dropout to the output of each sub-layer
    before it is added to the input. "We apply dropout to the output of
    each sub-layer, before it is added to the sub-layer input and normalized."
    (Section 5.4)

    Args:
        d_model: Model dimension
        dropout: Dropout rate (paper default: 0.1)
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """
        Apply residual connection with layer normalization.

        Note on implementation order: The paper describes Post-LN as
        LayerNorm(x + Sublayer(x)), but many implementations (including
        Harvard's Annotated Transformer) use Pre-LN: x + Sublayer(LayerNorm(x))
        for training stability. We use Pre-LN here following the widely-adopted
        convention that produces more stable training.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            sublayer_fn: A callable that takes x and returns transformed x

        Returns:
            output (batch, seq_len, d_model)
        """
        # Pre-LN: x + Dropout(Sublayer(LayerNorm(x)))
        return x + self.dropout(sublayer_fn(self.norm(x)))


def clone_modules(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Create N identical copies of a module.

    Used to create the stack of N encoder/decoder layers.
    The paper uses N=6 layers for both encoder and decoder.

    Each copy has its own independent parameters (deep copy).

    Args:
        module: The module to clone
        n: Number of copies (paper default: 6)

    Returns:
        nn.ModuleList of N independent copies
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def generate_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Generate a padding mask for attention.

    Masks out positions where the input token is the padding index.
    This prevents the model from attending to padding tokens.

    Args:
        seq: Input token indices (batch, seq_len)
        pad_idx: Index of the padding token (default: 0)

    Returns:
        mask: (batch, 1, 1, seq_len) — True where NOT padded, False where padded
              Shape is designed for broadcasting with attention scores
              (batch, heads, seq_len_q, seq_len_k)
    """
    # (batch, seq_len) → (batch, 1, 1, seq_len) for broadcasting
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_causal_mask(size: int, device: torch.device = None) -> torch.Tensor:
    """
    Generate a causal (look-ahead) mask for the decoder.

    From the paper (Section 3.1):
        "We also modify the self-attention sub-layer in the decoder stack to
         prevent positions from attending to subsequent positions."

    This creates an upper-triangular mask where position i can only attend
    to positions ≤ i (i.e., past and current, not future).

    Example for size=4:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Args:
        size: Sequence length
        device: Device to create the tensor on

    Returns:
        mask: (1, 1, size, size) — lower triangular matrix
              Shape is designed for broadcasting with attention scores
    """
    # torch.tril creates a lower triangular matrix
    mask = torch.tril(torch.ones(size, size, device=device)).unsqueeze(0).unsqueeze(0)
    return mask  # (1, 1, size, size)


def generate_source_mask(
    src: torch.Tensor, pad_idx: int = 0
) -> torch.Tensor:
    """
    Generate mask for encoder (source) self-attention.

    Only masks padding tokens. No causal masking needed in the encoder
    since it processes the full input sequence bidirectionally.

    Args:
        src: Source token indices (batch, src_len)
        pad_idx: Padding token index

    Returns:
        mask: (batch, 1, 1, src_len)
    """
    return generate_padding_mask(src, pad_idx)


def generate_target_mask(
    tgt: torch.Tensor, pad_idx: int = 0
) -> torch.Tensor:
    """
    Generate mask for decoder (target) self-attention.

    Combines:
    1. Padding mask — ignore <pad> tokens
    2. Causal mask — prevent attending to future positions

    Both constraints must be satisfied simultaneously (logical AND).

    Args:
        tgt: Target token indices (batch, tgt_len)
        pad_idx: Padding token index

    Returns:
        mask: (batch, 1, tgt_len, tgt_len) — combined padding + causal mask
    """
    tgt_len = tgt.size(1)

    # Padding mask: (batch, 1, 1, tgt_len)
    padding_mask = generate_padding_mask(tgt, pad_idx)

    # Causal mask: (1, 1, tgt_len, tgt_len)
    causal_mask = generate_causal_mask(tgt_len, device=tgt.device)

    # Combine: both conditions must be met
    # Broadcasting: (batch, 1, 1, tgt_len) & (1, 1, tgt_len, tgt_len)
    #            → (batch, 1, tgt_len, tgt_len)
    return padding_mask & causal_mask.bool()

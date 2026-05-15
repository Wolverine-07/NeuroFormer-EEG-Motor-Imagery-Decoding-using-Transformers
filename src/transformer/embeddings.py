"""
Embeddings and Positional Encoding — "Attention Is All You Need" Sections 3.4 & 3.5

Section 3.4 — Embeddings:
    "Similarly to other sequence transduction models, we use learned embeddings
     to convert the input tokens and output tokens to vectors of dimension d_model.
     We also use the usual learned linear transformation and softmax function to
     convert the decoder output to predicted next-token probabilities."

    "In the embedding layers, we multiply those weights by sqrt(d_model)."

Section 3.5 — Positional Encoding:
    "Since our model contains no recurrence and no convolution, in order for the
     model to make use of the order of the sequence, we must inject some information
     about the relative or absolute position of the tokens in the sequence."

    The paper uses sinusoidal positional encodings:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    "We chose this function because we hypothesized it would allow the model to
     easily learn to attend by relative positions, since for any fixed offset k,
     PE(pos+k) can be represented as a linear function of PE(pos)."
"""

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Learned token embeddings (Paper Section 3.4).

    Converts token indices to dense vectors of dimension d_model.
    The embeddings are scaled by sqrt(d_model) as stated in the paper:
        "In the embedding layers, we multiply those weights by sqrt(d_model)."

    This scaling ensures that the embedding values are in a similar range
    as the positional encodings, which are added to them.

    Args:
        vocab_size: Size of the vocabulary
        d_model: Embedding dimension (paper default: 512)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices (batch, seq_len)

        Returns:
            Scaled embeddings (batch, seq_len, d_model)
        """
        # Scale by sqrt(d_model) as specified in the paper
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Paper Section 3.5, Equations 3-4).

    Since the Transformer has no recurrence or convolution, it has no inherent
    notion of token order. Positional encodings are added to the input embeddings
    to provide position information.

    The paper uses sine and cosine functions of different frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    where pos is the position and i is the dimension index.

    Key properties of sinusoidal encoding (from the paper):
    1. Each dimension of the PE corresponds to a sinusoid with wavelengths
       forming a geometric progression from 2π to 10000·2π.
    2. For any fixed offset k, PE(pos+k) can be represented as a linear
       function of PE(pos) — this allows the model to learn relative positions.
    3. The encodings are deterministic (not learned), which means they can
       generalize to sequence lengths longer than those seen during training.

    The paper also experimented with learned positional embeddings and found
    "nearly identical results" (Table 3, row E), so the choice of sinusoidal
    was made for its ability to extrapolate to longer sequences.

    Args:
        d_model: Model dimension (paper default: 512)
        max_len: Maximum sequence length to pre-compute encodings for
        dropout: Dropout rate applied to the sum of embeddings + PE (paper default: 0.1)
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encodings for all positions up to max_len
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: (max_len, 1) — column vector
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term: 10000^(2i/d_model)
        # Using log-space for numerical stability:
        #   10000^(2i/d_model) = exp(2i * log(10000) / d_model)
        #   1 / 10000^(2i/d_model) = exp(-2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )

        # Apply sin to even indices (2i) and cos to odd indices (2i+1)
        pe[:, 0::2] = torch.sin(position * div_term)  # PE(pos, 2i) = sin(...)
        pe[:, 1::2] = torch.cos(position * div_term)  # PE(pos, 2i+1) = cos(...)

        # Add batch dimension: (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)

        # Register as a buffer (not a parameter — not learned, but saved with model)
        # This is important: positional encodings are fixed, not trained.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings and apply dropout.

        From the paper: "We apply dropout to the sums of the embeddings and
        the positional encodings in both the encoder and decoder stacks."

        Args:
            x: Embeddings (batch, seq_len, d_model)

        Returns:
            Embeddings + positional encoding with dropout (batch, seq_len, d_model)
        """
        seq_len = x.size(1)

        # Add positional encoding (broadcasting over batch dimension)
        # pe is (1, max_len, d_model), we slice to (1, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)

"""
Full Transformer Model — "Attention Is All You Need" (Vaswani et al., 2017)

This module assembles all components into the complete Encoder-Decoder
Transformer architecture as described in the paper.

Architecture Overview (Paper Figure 1):

    Source Tokens                          Target Tokens (shifted right)
         │                                        │
    [Token Embedding]                       [Token Embedding]
         │                                        │
    [+ Positional Encoding]                [+ Positional Encoding]
         │                                        │
    ┌────▼────────────┐                    ┌──────▼──────────────┐
    │   Encoder       │                    │   Decoder           │
    │   (N=6 layers)  │──── memory ────►   │   (N=6 layers)     │
    │                 │                    │                     │
    │  Self-Attention │                    │  Masked Self-Attn   │
    │  + FFN          │                    │  + Cross-Attention  │
    │  + Add & Norm   │                    │  + FFN              │
    └─────────────────┘                    │  + Add & Norm       │
                                           └──────┬──────────────┘
                                                  │
                                           [Linear Projection]
                                                  │
                                           [Log Softmax]
                                                  │
                                           Output Probabilities

From the paper (Section 3.4):
    "Similarly to other sequence transduction models, we use learned embeddings
     to convert the input tokens and output tokens to vectors of dimension d_model.
     We also use the usual learned linear transformation and softmax function to
     convert the decoder output to predicted next-token probabilities. In our
     model, we share the same weight matrix between the two embedding layers and
     the pre-softmax linear transformation."

Paper hyperparameters (Table 3 — Base model):
    d_model = 512
    d_ff    = 2048
    h       = 8  (number of attention heads)
    N       = 6  (number of encoder/decoder layers)
    P_drop  = 0.1
"""

import torch
import torch.nn as nn

from src.transformer.encoder import Encoder
from src.transformer.decoder import Decoder
from src.transformer.embeddings import TokenEmbedding, PositionalEncoding
from src.transformer.utils import generate_source_mask, generate_target_mask


class Transformer(nn.Module):
    """
    Full Transformer Model — Encoder-Decoder Architecture.

    Faithful implementation of the architecture from "Attention Is All You Need."
    All default hyperparameters match the "base model" from Table 3 of the paper.

    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension (paper default: 512)
        num_heads: Number of attention heads (paper default: 8)
        d_ff: Feed-forward inner dimension (paper default: 2048)
        num_layers: Number of encoder/decoder layers (paper default: 6)
        dropout: Dropout rate (paper default: 0.1)
        max_len: Maximum sequence length for positional encoding
        pad_idx: Padding token index for mask generation
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        # --- Embeddings (Paper Section 3.4) ---
        # Source and target embeddings
        self.src_embedding = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model)

        # Positional encoding (Paper Section 3.5)
        # Shared positional encoding for both source and target
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        # --- Encoder (Paper Section 3.1) ---
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)

        # --- Decoder (Paper Section 3.1) ---
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # --- Output Projection (Paper Section 3.4) ---
        # Linear layer to project decoder output to vocabulary size
        # "We also use the usual learned linear transformation and softmax
        #  function to convert the decoder output to predicted next-token
        #  probabilities."
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # --- Weight Initialization ---
        # The paper states (Section 3.4): "In our model, we share the same
        # weight matrix between the two embedding layers and the pre-softmax
        # linear transformation."
        # We implement this weight sharing when src_vocab == tgt_vocab.
        if src_vocab_size == tgt_vocab_size:
            self.src_embedding.embedding.weight = self.tgt_embedding.embedding.weight
            self.output_projection.weight = self.tgt_embedding.embedding.weight

        # Xavier uniform initialization (standard for transformers)
        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize parameters using Xavier uniform initialization.

        This is important for training stability. The paper doesn't
        explicitly specify initialization, but Xavier uniform is the
        standard used in most Transformer implementations.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode the source sequence.

        Pipeline: tokens → embedding → + positional encoding → encoder stack

        Args:
            src: Source token indices (batch, src_len)
            src_mask: Source mask (batch, 1, 1, src_len)

        Returns:
            Encoder output / memory (batch, src_len, d_model)
        """
        # Embed + positional encoding
        src_embedded = self.positional_encoding(self.src_embedding(src))

        # Pass through encoder stack
        return self.encoder(src_embedded, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode the target sequence given encoder output.

        Pipeline: tokens → embedding → + positional encoding → decoder stack

        Args:
            tgt: Target token indices (batch, tgt_len)
            encoder_output: Output from encoder (batch, src_len, d_model)
            src_mask: Source mask for cross-attention
            tgt_mask: Target mask (padding + causal)

        Returns:
            Decoder output (batch, tgt_len, d_model)
        """
        # Embed + positional encoding
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt))

        # Pass through decoder stack with encoder output
        return self.decoder(tgt_embedded, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Full forward pass: encode source, decode target.

        During training:
            - src: full source sequence
            - tgt: target sequence shifted right (teacher forcing)
            - Output: log-probabilities for each position

        Args:
            src: Source token indices (batch, src_len)
            tgt: Target token indices (batch, tgt_len)
            src_mask: Source padding mask. If None, generated automatically.
            tgt_mask: Target mask (padding + causal). If None, generated automatically.

        Returns:
            Output logits (batch, tgt_len, tgt_vocab_size)
        """
        # Generate masks if not provided
        if src_mask is None:
            src_mask = generate_source_mask(src, self.pad_idx)
        if tgt_mask is None:
            tgt_mask = generate_target_mask(tgt, self.pad_idx)

        # Encode
        encoder_output = self.encode(src, src_mask)

        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        # Project to vocabulary and return logits
        # (batch, tgt_len, d_model) → (batch, tgt_len, tgt_vocab_size)
        output = self.output_projection(decoder_output)

        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def build_model(
        src_vocab_size: int,
        tgt_vocab_size: int,
        variant: str = "base",
        **kwargs,
    ) -> "Transformer":
        """
        Factory method to build Transformer with paper-specified hyperparameters.

        Paper Table 3 defines two model sizes:
            Base:  d_model=512, d_ff=2048, h=8, N=6, dropout=0.1
            Big:   d_model=1024, d_ff=4096, h=16, N=6, dropout=0.3

        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            variant: "base" or "big" (paper Table 3)
            **kwargs: Override any hyperparameter

        Returns:
            Configured Transformer model
        """
        configs = {
            "base": {
                "d_model": 512,
                "num_heads": 8,
                "d_ff": 2048,
                "num_layers": 6,
                "dropout": 0.1,
            },
            "big": {
                "d_model": 1024,
                "num_heads": 16,
                "d_ff": 4096,
                "num_layers": 6,
                "dropout": 0.3,
            },
            "tiny": {
                # For testing and debugging
                "d_model": 64,
                "num_heads": 4,
                "d_ff": 128,
                "num_layers": 2,
                "dropout": 0.1,
            },
        }

        if variant not in configs:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(configs.keys())}")

        config = configs[variant]
        config.update(kwargs)

        return Transformer(src_vocab_size, tgt_vocab_size, **config)

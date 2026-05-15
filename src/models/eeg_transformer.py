"""
EEG-Transformer — CNN-Transformer hybrid for motor imagery classification.

This is the main application model that combines:
1. Our from-scratch Transformer encoder (Phase 1)
2. The CNN tokenizer (Phase 3.1)
3. A classification head

Architecture:
    Raw EEG (batch, 64 channels, ~513 time points)
      → CNN Tokenizer → (batch, n_patches, d_model)
      → + Learnable Positional Encoding
      → [CLS] token prepended
      → Transformer Encoder (N layers, h heads)
      → [CLS] token output → Classification Head
      → 4-class probabilities

The key insight is that we only use the Transformer ENCODER (not decoder).
This is a classification task, not sequence-to-sequence, so we don't need
auto-regressive generation. This is the same approach used by BERT and
Vision Transformers for classification.

The [CLS] token is a special learnable token prepended to the sequence.
Its output representation is used for classification because it attends
to all other tokens through self-attention, aggregating information
from the entire input. (Same idea as in BERT.)
"""

import torch
import torch.nn as nn

from src.transformer.encoder import Encoder
from src.eeg.tokenizer import EEGTokenizer


class EEGTransformer(nn.Module):
    """
    CNN-Transformer hybrid for EEG motor imagery classification.

    Reuses the Transformer encoder from our from-scratch implementation
    (Phase 1) with a CNN front-end for EEG-specific feature extraction.

    The model is deliberately smaller than the paper's base configuration
    because EEG datasets are much smaller than NLP corpora. Using the
    full 512-dim, 6-layer model would massively overfit.

    Args:
        n_channels: Number of EEG channels
        n_times: Number of time points per trial
        n_classes: Number of output classes (4 for our task)
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward inner dimension
        num_layers: Number of Transformer encoder layers
        dropout: Dropout rate
        temporal_filters: CNN temporal filter count
        temporal_kernel: CNN temporal kernel size
        patch_size: Tokenizer patch size
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 513,
        n_classes: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2,
        temporal_filters: int = 16,
        temporal_kernel: int = 64,
        patch_size: int = 16,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_classes = n_classes

        # CNN tokenizer: raw EEG → patch embeddings
        self.tokenizer = EEGTokenizer(
            n_channels=n_channels,
            n_times=n_times,
            d_model=d_model,
            temporal_filters=temporal_filters,
            temporal_kernel=temporal_kernel,
            patch_size=patch_size,
            dropout=dropout,
        )

        # Learnable [CLS] token — aggregates information for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional encoding for the patch sequence
        # (we use learned instead of sinusoidal here because EEG sequences
        # have fixed length, and learned PE works slightly better for
        # fixed-length classification tasks)
        n_positions = self.tokenizer.num_patches + 1  # +1 for CLS
        self.pos_encoding = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer encoder (reusing our from-scratch implementation!)
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Classification head
        # The CLS token output goes through layer norm → linear → output
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with truncated normal (ViT-style)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Raw EEG input (batch, n_channels, n_times)
            return_attention: If True, also return attention weights from
                            all encoder layers (for visualization)

        Returns:
            logits: (batch, n_classes) class logits
            attention_weights: (optional) list of attention weight tensors
        """
        batch_size = x.size(0)

        # 1. Tokenize: raw EEG → patch embeddings
        tokens = self.tokenizer(x)  # (batch, n_patches, d_model)

        # 2. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch, 1+n_patches, d_model)

        # 3. Add positional encoding
        tokens = tokens + self.pos_encoding[:, :tokens.size(1), :]
        tokens = self.pos_dropout(tokens)

        # 4. Transformer encoder
        encoded = self.encoder(tokens)  # (batch, 1+n_patches, d_model)

        # 5. Take [CLS] token output for classification
        cls_output = encoded[:, 0, :]  # (batch, d_model)

        # 6. Classification head
        logits = self.classifier(cls_output)  # (batch, n_classes)

        if return_attention:
            # Collect attention weights from all encoder layers
            attn_weights = []
            for layer in self.encoder.layers:
                if hasattr(layer.self_attention, 'attention_weights'):
                    attn_weights.append(layer.self_attention.attention_weights)
            return logits, attn_weights

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps from all layers for a single input.

        Useful for interpretability analysis — see which time periods
        and spatial patterns the model focuses on.

        Args:
            x: Raw EEG (1, n_channels, n_times)

        Returns:
            List of attention weight tensors, one per layer
            Each tensor has shape (1, n_heads, seq_len, seq_len)
        """
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.forward(x, return_attention=True)
        return attn_weights

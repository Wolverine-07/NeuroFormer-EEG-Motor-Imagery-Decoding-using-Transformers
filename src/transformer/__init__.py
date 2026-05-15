"""
Transformer from Scratch — "Attention Is All You Need" (Vaswani et al., 2017)

A faithful, from-scratch PyTorch implementation of the Transformer architecture.
"""

from src.transformer.attention import MultiHeadAttention, scaled_dot_product_attention
from src.transformer.encoder import Encoder, EncoderLayer
from src.transformer.decoder import Decoder, DecoderLayer
from src.transformer.embeddings import TokenEmbedding, PositionalEncoding
from src.transformer.feed_forward import PositionwiseFeedForward
from src.transformer.transformer import Transformer
from src.transformer.utils import (
    LayerNorm,
    SublayerConnection,
    clone_modules,
    generate_padding_mask,
    generate_causal_mask,
    generate_source_mask,
    generate_target_mask,
)

__all__ = [
    "Transformer",
    "Encoder",
    "EncoderLayer",
    "Decoder",
    "DecoderLayer",
    "MultiHeadAttention",
    "scaled_dot_product_attention",
    "PositionwiseFeedForward",
    "TokenEmbedding",
    "PositionalEncoding",
    "LayerNorm",
    "SublayerConnection",
    "clone_modules",
    "generate_padding_mask",
    "generate_causal_mask",
    "generate_source_mask",
    "generate_target_mask",
]

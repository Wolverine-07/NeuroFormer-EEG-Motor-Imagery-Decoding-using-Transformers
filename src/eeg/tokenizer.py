"""
EEG Tokenizer — CNN-based feature extraction for Transformer input.

Raw EEG signals aren't discrete tokens like words, so we can't feed them
directly into a Transformer. This module converts continuous EEG signals
into a sequence of "tokens" (patch embeddings) that the Transformer can
process.

Architecture (inspired by EEGNet + patch embedding):

    Raw EEG (channels × time)
      → Temporal Conv (extract time-domain features like frequency filters)
      → Spatial Conv (learn spatial filters across channels, like CSP)
      → Non-overlapping patch segmentation
      → Linear projection to d_model
      → Token sequence ready for Transformer

This is similar to how Vision Transformers (ViT) use patch embedding
to convert images into token sequences.
"""

import torch
import torch.nn as nn


class EEGTokenizer(nn.Module):
    """
    CNN-based tokenizer that converts raw EEG to a sequence of embeddings.

    The design follows a common pattern in EEG deep learning:
    1. Temporal convolution captures frequency-domain features
       (similar to bandpass filters but learned from data)
    2. Spatial convolution captures spatial patterns across electrodes
       (similar to Common Spatial Pattern but end-to-end learned)
    3. Patch segmentation splits the resulting feature maps into
       non-overlapping chunks that become Transformer tokens

    Args:
        n_channels: Number of EEG channels (e.g., 64 for PhysioNet)
        n_times: Number of time points per trial
        d_model: Output embedding dimension for Transformer
        temporal_filters: Number of temporal convolution filters
        spatial_filters_per_temporal: Multiplier for spatial filter count
        temporal_kernel: Kernel size for temporal convolution (samples)
        pool_size: Average pooling factor after spatial conv
        dropout: Dropout rate
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 513,
        d_model: int = 128,
        temporal_filters: int = 16,
        spatial_filters_per_temporal: int = 2,
        temporal_kernel: int = 64,
        pool_size: int = 4,
        patch_size: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        n_spatial = temporal_filters * spatial_filters_per_temporal

        # --- Stage 1: Temporal convolution ---
        # Learns frequency-domain filters from the data
        # Conv2d with (1, temporal_kernel) acts as a temporal filter
        # across all channels simultaneously
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                1, temporal_filters,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
        )

        # --- Stage 2: Spatial convolution (depthwise) ---
        # Learns spatial filters — which electrode combinations are informative
        # Groups=temporal_filters makes this depthwise: each temporal filter
        # gets its own set of spatial filters
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                temporal_filters, n_spatial,
                kernel_size=(n_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(n_spatial),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_size)),
            nn.Dropout(dropout),
        )

        # --- Stage 3: Separable convolution for refinement ---
        # Additional feature refinement before creating patches
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                n_spatial, n_spatial,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=n_spatial,
                bias=False,
            ),
            nn.Conv2d(n_spatial, n_spatial, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_spatial),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(dropout),
        )

        # Calculate the time dimension after convolutions
        # We need to do a dummy forward pass to figure out the exact size
        self._n_spatial = n_spatial
        self._feature_time = self._calc_feature_time(n_channels, n_times)
        self._n_patches = self._feature_time // patch_size

        # --- Stage 4: Patch projection ---
        # Project each patch of the feature map to d_model dimensions
        # patch input size = n_spatial * patch_size (flattened)
        self.patch_projection = nn.Linear(n_spatial * patch_size, d_model)

    def _calc_feature_time(self, n_channels: int, n_times: int) -> int:
        """Calculate time dimension after CNN stages using a dummy forward."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            x = self.temporal_conv(dummy)
            x = self.spatial_conv(x)
            x = self.separable_conv(x)
            return x.shape[-1]

    @property
    def num_patches(self) -> int:
        """Number of tokens/patches produced."""
        return self._n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert raw EEG to a sequence of token embeddings.

        Args:
            x: Raw EEG (batch, n_channels, n_times)

        Returns:
            tokens: (batch, n_patches, d_model)
        """
        batch_size = x.size(0)

        # Add channel dim for Conv2d: (batch, 1, channels, times)
        x = x.unsqueeze(1)

        # CNN feature extraction
        x = self.temporal_conv(x)   # (batch, temp_filters, channels, times)
        x = self.spatial_conv(x)    # (batch, spatial_filters, 1, times')
        x = self.separable_conv(x)  # (batch, spatial_filters, 1, times'')

        # Squeeze the spatial dimension (it's 1 after spatial conv)
        x = x.squeeze(2)  # (batch, n_spatial, feature_time)

        # Truncate to fit exact number of patches
        usable_time = self._n_patches * self.patch_size
        x = x[:, :, :usable_time]

        # Reshape into patches: (batch, n_spatial, n_patches, patch_size)
        x = x.reshape(batch_size, self._n_spatial, self._n_patches, self.patch_size)

        # Rearrange: (batch, n_patches, n_spatial * patch_size)
        x = x.permute(0, 2, 1, 3).reshape(
            batch_size, self._n_patches, self._n_spatial * self.patch_size
        )

        # Project to d_model
        tokens = self.patch_projection(x)  # (batch, n_patches, d_model)

        return tokens

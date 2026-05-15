"""
Baseline Models for comparison.

EEGNet — a compact CNN baseline widely used in BCI research.

Reference:
    Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural
    Network for EEG-Based Brain-Computer Interfaces."
    Journal of Neural Engineering, 15(5).

Having a baseline is essential for scientific rigor — if our Transformer
doesn't outperform a simple CNN, we can't claim the attention mechanism
provides any benefit for this task.
"""

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet — lightweight CNN baseline for EEG classification.

    Architecture (from the paper):
      1. Temporal conv → BN
      2. Depthwise spatial conv → BN → ELU → AvgPool → Dropout
      3. Separable conv → BN → ELU → AvgPool → Dropout
      4. Flatten → Linear → output

    This model has very few parameters (~2-5K depending on config),
    making it a fair comparison point for our larger Transformer.

    Args:
        n_channels: Number of EEG channels
        n_times: Number of time points
        n_classes: Number of output classes
        temporal_filters: F1 in the paper (default: 8)
        spatial_filters: D in the paper — multiplier for spatial (default: 2)
        temporal_kernel: Kernel size for temporal conv (default: 64)
        dropout: Dropout rate (default: 0.25)
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 513,
        n_classes: int = 4,
        temporal_filters: int = 8,
        spatial_filters: int = 2,
        temporal_kernel: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()

        F1 = temporal_filters
        D = spatial_filters
        F2 = F1 * D

        # Block 1: temporal filtering
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial conv
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                      groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: separable convolution
        self.block2 = nn.Sequential(
            # Depthwise
            nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8),
                      groups=F2, bias=False),
            # Pointwise
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Classifier
        # Compute flattened feature size
        self._feature_size = self._get_feature_size(n_channels, n_times, F1, D)
        self.classifier = nn.Linear(self._feature_size, n_classes)

    def _get_feature_size(self, n_ch, n_t, F1, D):
        """Calculate flattened feature size with a dummy forward pass."""
        with torch.no_grad():
            x = torch.zeros(1, 1, n_ch, n_t)
            x = self.block1(x)
            x = self.block2(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw EEG (batch, n_channels, n_times)

        Returns:
            logits: (batch, n_classes)
        """
        x = x.unsqueeze(1)  # Add channel dim: (batch, 1, ch, time)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

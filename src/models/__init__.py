"""Models for EEG classification."""

from src.models.eeg_transformer import EEGTransformer
from src.models.baselines import EEGNet

__all__ = ["EEGTransformer", "EEGNet"]

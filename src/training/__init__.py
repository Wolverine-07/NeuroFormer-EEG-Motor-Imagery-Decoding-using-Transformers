"""Training infrastructure for NeuroFormer."""

from src.training.trainer import Trainer, greedy_decode
from src.training.scheduler import NoamScheduler
from src.training.losses import LabelSmoothingLoss
from src.training.metrics import (
    compute_classification_metrics,
    compute_sequence_accuracy,
    paired_ttest,
    compute_confidence_interval,
    cohens_d,
)

__all__ = [
    "Trainer",
    "greedy_decode",
    "NoamScheduler",
    "LabelSmoothingLoss",
    "compute_classification_metrics",
    "compute_sequence_accuracy",
    "paired_ttest",
    "compute_confidence_interval",
    "cohens_d",
]

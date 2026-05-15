"""
Evaluation Metrics for classification and sequence tasks.
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
) -> Dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Optional list of class names

    Returns:
        Dictionary with accuracy, f1, kappa, confusion matrix, and report
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        ),
    }
    return metrics


def compute_sequence_accuracy(
    logits, target, pad_idx: int = 0
) -> float:
    """
    Compute token-level accuracy for sequence tasks, ignoring padding.

    Args:
        logits: Model output (batch, seq_len, vocab_size) — can be tensor or numpy
        target: Target token indices (batch, seq_len)
        pad_idx: Padding token index to ignore

    Returns:
        Accuracy as a float
    """
    import torch

    if isinstance(logits, torch.Tensor):
        predictions = logits.argmax(dim=-1)
        non_pad_mask = target != pad_idx
        correct = (predictions == target) & non_pad_mask
        accuracy = correct.sum().float() / non_pad_mask.sum().float()
        return accuracy.item()
    else:
        predictions = np.argmax(logits, axis=-1)
        non_pad_mask = target != pad_idx
        correct = (predictions == target) & non_pad_mask
        return correct.sum() / non_pad_mask.sum()

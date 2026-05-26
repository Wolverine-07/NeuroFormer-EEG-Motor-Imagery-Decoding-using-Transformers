"""
Evaluation Metrics for classification and sequence tasks.

Includes:
  - Classification metrics (accuracy, F1, Cohen's kappa)
  - Sequence accuracy (token-level, ignoring padding)
  - Statistical tests: paired t-test, confidence intervals, Cohen's d
"""

import math
from typing import Dict, List, Tuple

# Optional scipy import for statistical tests
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_stats = None
    _SCIPY_AVAILABLE = False

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

def paired_ttest(scores_a: List[float], scores_b: List[float]) -> Dict[str, float]:
    """
    Perform a paired t-test between two lists of scores (e.g., accuracy per subject).
    
    Args:
        scores_a: First list of scores
        scores_b: Second list of scores
        
    Returns:
        Dictionary with t_statistic, p_value, mean_difference, and is_significant (alpha=0.05)
    """
    if not _SCIPY_AVAILABLE:
        print("Warning: scipy is not installed. Returning empty t-test results.")
        return {"t_statistic": 0.0, "p_value": 1.0, "mean_difference": 0.0, "is_significant": 0.0}
    
    mean_diff = float(np.mean(scores_a) - np.mean(scores_b))
    result = _scipy_stats.ttest_rel(scores_a, scores_b)
    
    return {
        "t_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "mean_difference": mean_diff,
        "is_significant": float(result.pvalue < 0.05)
    }

def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute the confidence interval for a list of scores.
    
    Args:
        scores: List of scores
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    a = 1.0 * np.array(scores)
    n = len(a)
    m, se = np.mean(a), _scipy_stats.sem(a) if _SCIPY_AVAILABLE else np.std(a, ddof=1) / math.sqrt(n)
    
    if _SCIPY_AVAILABLE:
        h = se * _scipy_stats.t.ppf((1 + confidence) / 2., n-1)
    else:
        # Fallback to normal distribution approximation if scipy is not available
        z = 1.96 if confidence == 0.95 else 2.576 # approx
        h = se * z
        
    return float(m), float(m-h), float(m+h)

def cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Compute Cohen's d effect size between two sets of scores.
    
    Args:
        scores_a: First list of scores
        scores_b: Second list of scores
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(scores_a), len(scores_b)
    var1, var2 = np.var(scores_a, ddof=1), np.var(scores_b, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var)
    
    # Calculate Cohen's d
    if pooled_std == 0:
        return 0.0
        
    return float((np.mean(scores_a) - np.mean(scores_b)) / pooled_std)

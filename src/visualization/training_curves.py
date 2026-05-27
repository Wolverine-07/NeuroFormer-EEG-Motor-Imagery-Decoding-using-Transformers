"""
Training curve and results visualization.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Use non-interactive backend only when not already set (e.g., by Jupyter %matplotlib inline)
if matplotlib.get_backend() == "agg" or not matplotlib.is_interactive():
    try:
        matplotlib.use("Agg")
    except Exception:
        pass


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float] = None,
    train_accs: List[float] = None,
    val_accs: List[float] = None,
    title: str = "Training Curves",
    save_path: Optional[str] = None,
):
    """Plot training/validation loss and accuracy curves."""
    n_plots = 1 + (1 if train_accs else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4.5))

    if n_plots == 1:
        axes = [axes]

    # Loss
    axes[0].plot(train_losses, label="Train", linewidth=1.5)
    if val_losses:
        axes[0].plot(val_losses, label="Validation", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if train_accs and n_plots > 1:
        axes[1].plot(train_accs, label="Train", linewidth=1.5)
        if val_accs:
            axes[1].plot(val_accs, label="Validation", linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    normalize: bool = True,
):
    """Plot confusion matrix with counts and percentages."""
    cm = np.array(cm)
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else None)

    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            if normalize:
                text = f"{val:.2f}\n({count})"
            else:
                text = f"{count}"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_subject_accuracies(
    subject_metrics: Dict[int, Dict],
    model_name: str = "EEG-Transformer",
    save_path: Optional[str] = None,
):
    """
    Bar chart showing per-subject accuracy.

    Useful for seeing which subjects the model handles well vs poorly.
    Large inter-subject variance is typical in BCI.
    """
    subjects = sorted(subject_metrics.keys())
    accs = [subject_metrics[s]["accuracy"] for s in subjects]

    fig, ax = plt.subplots(figsize=(max(12, len(subjects) * 0.3), 5))

    colors = ["#2ecc71" if a > np.mean(accs) else "#e74c3c" for a in accs]
    ax.bar(range(len(subjects)), accs, color=colors, alpha=0.8, edgecolor="gray", linewidth=0.5)

    # Mean line
    mean_acc = np.mean(accs)
    ax.axhline(y=mean_acc, color="navy", linestyle="--", linewidth=1.5,
               label=f"Mean: {mean_acc:.3f}")
    ax.axhline(y=0.25, color="gray", linestyle=":", linewidth=1,
               label="Chance level (0.25)")

    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([f"S{s}" for s in subjects], rotation=90, fontsize=6)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Subject Accuracy — {model_name}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict],
    metric: str = "accuracy",
    save_path: Optional[str] = None,
):
    """
    Compare metrics across different models (e.g., EEG-Transformer vs EEGNet).

    Args:
        results: {model_name: {"accuracy": mean, "accuracy_std": std, ...}}
        metric: Which metric to plot
    """
    models = list(results.keys())
    means = [results[m][metric] for m in models]
    stds = [results[m].get(f"{metric}_std", 0) for m in models]

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(models, means, yerr=stds, capsize=5,
                  color=["#3498db", "#e67e22", "#2ecc71", "#9b59b6"][:len(models)],
                  edgecolor="gray", alpha=0.85)

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}")

    if metric == "accuracy":
        ax.axhline(y=0.25, color="gray", linestyle=":", label="Chance")
        ax.set_ylim(0, 1.05)
        ax.legend()

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

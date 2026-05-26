"""Visualization utilities for EEG-Transformer analysis."""

from src.visualization.attention_maps import (
    plot_attention_heatmap,
    plot_all_heads,
    plot_cls_attention_over_patches,
)
from src.visualization.training_curves import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_subject_accuracies,
    plot_model_comparison,
)
from src.visualization.eeg_plots import (
    plot_eeg_trial,
    plot_class_averaged_signals,
    plot_psd_comparison,
    plot_topographic_map,
    plot_channel_importance,
)

__all__ = [
    "plot_attention_heatmap",
    "plot_all_heads",
    "plot_cls_attention_over_patches",
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_subject_accuracies",
    "plot_model_comparison",
    "plot_eeg_trial",
    "plot_class_averaged_signals",
    "plot_psd_comparison",
    "plot_topographic_map",
    "plot_channel_importance",
]

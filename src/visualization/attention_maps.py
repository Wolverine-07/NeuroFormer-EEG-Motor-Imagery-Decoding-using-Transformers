"""
Attention map visualization for EEG-Transformer.

Visualize which time segments and spatial patterns the model attends to.
This is crucial for interpretability in neuroscience — we need to verify
the model is learning biologically meaningful features (e.g., attending
to motor cortex channels C3/C4 during motor imagery).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Optional

matplotlib.use("Agg")  # non-interactive backend for saving plots


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: (n_heads, seq_len, seq_len) or (seq_len, seq_len)
        layer_idx: Which layer these weights are from (for title)
        head_idx: If None, average across heads. Otherwise, show specific head.
        title: Additional title text
        save_path: If provided, save figure to this path
        figsize: Figure size
    """
    if attention_weights.ndim == 3:
        if head_idx is not None:
            attn = attention_weights[head_idx]
            subtitle = f"Layer {layer_idx}, Head {head_idx}"
        else:
            attn = attention_weights.mean(axis=0)
            subtitle = f"Layer {layer_idx}, Averaged over heads"
    else:
        attn = attention_weights
        subtitle = f"Layer {layer_idx}"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn, cmap="viridis", aspect="auto")

    ax.set_xlabel("Key Position (patch)")
    ax.set_ylabel("Query Position (patch)")
    ax.set_title(f"Attention Weights — {subtitle}\n{title}")

    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_all_heads(
    attention_weights: np.ndarray,
    layer_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    Plot attention weights for all heads in a grid.

    Args:
        attention_weights: (n_heads, seq_len, seq_len)
        layer_idx: Layer index for title
        save_path: Save path
    """
    n_heads = attention_weights.shape[0]
    cols = min(4, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(attention_weights[h], cmap="viridis", aspect="auto")
        ax.set_title(f"Head {h}", fontsize=10)
        ax.set_xlabel("Key", fontsize=8)
        ax.set_ylabel("Query", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(f"Attention Weights — Layer {layer_idx}", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_cls_attention_over_patches(
    attention_weights_list: List[np.ndarray],
    save_path: Optional[str] = None,
):
    """
    Plot how the [CLS] token attends to each patch across layers.

    The CLS token (position 0) aggregates information for classification.
    This plot shows which temporal patches the model considers most important.
    In motor imagery, we'd expect peaks around the time of imagined movement.

    Args:
        attention_weights_list: List of (n_heads, seq_len, seq_len) per layer
        save_path: Save path
    """
    n_layers = len(attention_weights_list)
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 2.5 * n_layers), sharex=True)

    if n_layers == 1:
        axes = [axes]

    for layer_idx, attn in enumerate(attention_weights_list):
        # CLS attention = row 0 (what CLS attends to), averaged over heads
        cls_attn = attn.mean(axis=0)[0, 1:]  # skip self-attention to CLS

        axes[layer_idx].bar(range(len(cls_attn)), cls_attn, color="steelblue", alpha=0.8)
        axes[layer_idx].set_ylabel("Attention", fontsize=9)
        axes[layer_idx].set_title(f"Layer {layer_idx}", fontsize=10)
        axes[layer_idx].set_ylim(0, max(cls_attn) * 1.3)

    axes[-1].set_xlabel("Patch Index (temporal position)")
    fig.suptitle("[CLS] Token Attention Distribution Across Patches", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

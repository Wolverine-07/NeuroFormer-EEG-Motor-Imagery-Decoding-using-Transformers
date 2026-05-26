"""
EEG signal visualization.

Plots for exploring and understanding the raw EEG data:
  - Multi-channel time series
  - Power spectral density
  - Topographic maps (channel-level features)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Optional

matplotlib.use("Agg")


def plot_eeg_trial(
    data: np.ndarray,
    sfreq: float = 128.0,
    channel_names: List[str] = None,
    n_channels_to_show: int = 10,
    title: str = "EEG Trial",
    save_path: Optional[str] = None,
):
    """
    Plot a few channels of a single EEG trial.

    Args:
        data: (n_channels, n_times)
        sfreq: Sampling frequency
        channel_names: List of channel names
        n_channels_to_show: How many channels to display
        title: Plot title
        save_path: Save path
    """
    n_ch = min(n_channels_to_show, data.shape[0])
    times = np.arange(data.shape[1]) / sfreq

    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 1.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for i in range(n_ch):
        axes[i].plot(times, data[i], linewidth=0.5, color="navy")
        label = channel_names[i] if channel_names else f"Ch {i}"
        axes[i].set_ylabel(label, fontsize=8, rotation=0, labelpad=30)
        axes[i].tick_params(labelsize=7)
        axes[i].set_xlim(times[0], times[-1])

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_class_averaged_signals(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    channels: List[int] = None,
    channel_names: List[str] = None,
    sfreq: float = 128.0,
    save_path: Optional[str] = None,
):
    """
    Plot class-averaged EEG for selected channels.

    Averaging across trials of the same class reveals the event-related
    potential/desynchronization patterns that distinguish different
    motor imagery tasks. For left vs right imagery, you should see
    differences between C3 (left motor cortex) and C4 (right motor cortex).

    Args:
        X: (n_epochs, n_channels, n_times)
        y: (n_epochs,)
        class_names: Label names
        channels: Which channel indices to plot (default: first 4)
        channel_names: Channel name strings
        sfreq: Sampling frequency
        save_path: Save path
    """
    if channels is None:
        channels = list(range(min(4, X.shape[1])))

    times = np.arange(X.shape[-1]) / sfreq
    n_classes = len(class_names)
    n_ch = len(channels)

    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 3 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for ch_idx, ch in enumerate(channels):
        for cls_idx, cls_name in enumerate(class_names):
            mask = y == cls_idx
            if mask.sum() == 0:
                continue
            avg_signal = X[mask, ch, :].mean(axis=0)
            std_signal = X[mask, ch, :].std(axis=0)

            axes[ch_idx].plot(times, avg_signal, label=cls_name,
                              color=colors[cls_idx], linewidth=1.5)
            axes[ch_idx].fill_between(
                times,
                avg_signal - std_signal,
                avg_signal + std_signal,
                alpha=0.15, color=colors[cls_idx],
            )

        ch_label = channel_names[ch_idx] if channel_names else f"Channel {ch}"
        axes[ch_idx].set_ylabel(ch_label, fontsize=10)
        axes[ch_idx].legend(fontsize=8, loc="upper right")
        axes[ch_idx].grid(True, alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Class-Averaged EEG Signals", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_psd_comparison(
    freqs: np.ndarray,
    psd_by_class: dict,
    channel_idx: int = 0,
    channel_name: str = "",
    save_path: Optional[str] = None,
):
    """
    Compare power spectral density across classes for a specific channel.

    Should reveal mu (8-13 Hz) and beta (13-30 Hz) desynchronization
    differences between left and right imagery classes.

    Args:
        freqs: Frequency bins
        psd_by_class: {class_name: psd_array}
        channel_idx: Which channel
        channel_name: Channel name for title
        save_path: Save path
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, len(psd_by_class)))

    for (cls_name, psd), color in zip(psd_by_class.items(), colors):
        if psd.ndim > 1:
            psd_ch = psd[channel_idx]
        else:
            psd_ch = psd
        ax.semilogy(freqs, psd_ch, label=cls_name, color=color, linewidth=1.5)

    # Highlight frequency bands
    ax.axvspan(8, 13, alpha=0.1, color="blue", label="μ band (8-13 Hz)")
    ax.axvspan(13, 30, alpha=0.1, color="green", label="β band (13-30 Hz)")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (log scale)")
    ax.set_title(f"PSD Comparison — {channel_name or f'Channel {channel_idx}'}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

def plot_topographic_map(
    values: np.ndarray,
    channel_names: List[str],
    channel_positions: Optional[np.ndarray] = None,
    title: str = "Topographic Map",
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """
    Plot a topographic map of values (e.g. attention weights or activations) 
    over the scalp.

    If channel_positions are not provided, an approximate 10-20 system 
    layout will be generated for standard channel names.

    Args:
        values: (n_channels,) Array of values to plot
        channel_names: List of channel name strings
        channel_positions: (n_channels, 2) Array of 2D coordinates.
        title: Plot title
        cmap: Colormap to use
        save_path: Optional save path
    """
    # MNE is highly recommended for real topomaps, but here's a fallback
    # scatter plot implementation for our custom visualizations
    
    # Simple approx positions for some standard channels (x, y)
    standard_pos = {
        'Fp1': (-0.3, 0.4), 'Fp2': (0.3, 0.4),
        'F3': (-0.3, 0.2), 'F4': (0.3, 0.2),
        'C3': (-0.3, 0.0), 'C4': (0.3, 0.0),
        'P3': (-0.3, -0.2), 'P4': (0.3, -0.2),
        'O1': (-0.3, -0.4), 'O2': (0.3, -0.4),
        'Fz': (0.0, 0.2), 'Cz': (0.0, 0.0), 'Pz': (0.0, -0.2),
    }

    if channel_positions is None:
        # Try to use standard positions, otherwise random circle
        pos = []
        for i, ch in enumerate(channel_names):
            if ch in standard_pos:
                pos.append(standard_pos[ch])
            else:
                angle = i * (2 * np.pi / len(channel_names))
                pos.append((0.4 * np.cos(angle), 0.4 * np.sin(angle)))
        channel_positions = np.array(pos)

    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Draw "head" outline
    head = plt.Circle((0, 0), 0.5, color='black', fill=False, linewidth=2)
    ax.add_artist(head)
    
    # Draw "nose"
    nose_x = [ -0.05, 0.0, 0.05 ]
    nose_y = [ 0.48, 0.55, 0.48 ]
    ax.plot(nose_x, nose_y, color='black', linewidth=2)
    
    # Draw "ears"
    ax.add_artist(plt.Circle((-0.51, 0), 0.05, color='black', fill=False, linewidth=1.5))
    ax.add_artist(plt.Circle((0.51, 0), 0.05, color='black', fill=False, linewidth=1.5))

    # Scatter plot for values
    sc = ax.scatter(
        channel_positions[:, 0], 
        channel_positions[:, 1], 
        c=values, 
        cmap=cmap, 
        s=200, 
        edgecolor='black',
        zorder=3
    )

    # Add channel labels
    for i, txt in enumerate(channel_names):
        ax.annotate(
            txt, 
            (channel_positions[i, 0], channel_positions[i, 1]), 
            ha='center', va='center', 
            fontsize=8,
            color='white' if values[i] < np.mean(values) else 'black',
            zorder=4
        )

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_channel_importance(
    importance_scores: np.ndarray,
    channel_names: List[str],
    top_k: int = 20,
    title: str = "Channel Importance",
    save_path: Optional[str] = None,
):
    """
    Plot a bar chart of channel importance scores.

    Args:
        importance_scores: (n_channels,) Array of importance scores
        channel_names: List of channel name strings
        top_k: Number of top channels to display (default 20)
        title: Plot title
        save_path: Optional save path
    """
    # Sort indices by descending score
    sorted_idx = np.argsort(importance_scores)[::-1]
    
    # Get top k
    top_idx = sorted_idx[:top_k]
    top_scores = importance_scores[top_idx]
    top_names = [channel_names[i] for i in top_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bar chart (highest at top)
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_scores, align='center', color='steelblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

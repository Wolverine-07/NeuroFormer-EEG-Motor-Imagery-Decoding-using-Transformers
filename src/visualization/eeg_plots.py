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

        ch_label = channel_names[ch] if channel_names else f"Channel {ch}"
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

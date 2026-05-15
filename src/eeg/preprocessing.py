"""
EEG Preprocessing utilities.

Additional preprocessing steps beyond what's in dataset.py, including
feature extraction helpers and signal quality checks.

The main preprocessing (bandpass filtering, epoching, baseline correction)
is handled directly in dataset.load_subject_data(). This module provides
supplementary tools for signal analysis and quality control.
"""

import numpy as np
from typing import Tuple, Optional


def compute_power_spectral_density(
    data: np.ndarray,
    sfreq: float = 128.0,
    fmin: float = 1.0,
    fmax: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD using Welch's method.

    Useful for verifying that the bandpass filter worked correctly
    and for visualizing mu/beta rhythm power.

    Args:
        data: EEG data (n_channels, n_times) or (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        fmin: Minimum frequency of interest
        fmax: Maximum frequency of interest

    Returns:
        freqs: Frequency bins
        psd: Power spectral density values
    """
    from scipy import signal

    if data.ndim == 3:
        # Average PSD across epochs
        psds = []
        for epoch in data:
            f, p = signal.welch(epoch, fs=sfreq, nperseg=min(256, epoch.shape[-1]))
            psds.append(p)
        psd = np.mean(psds, axis=0)
        freqs = f
    else:
        freqs, psd = signal.welch(data, fs=sfreq, nperseg=min(256, data.shape[-1]))

    # Crop to frequency range of interest
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[freq_mask], psd[..., freq_mask]


def compute_band_power(
    data: np.ndarray,
    sfreq: float = 128.0,
    bands: dict = None,
) -> dict:
    """
    Compute average power in standard frequency bands.

    Default bands relevant to motor imagery:
      - mu (8-13 Hz): desynchronizes during motor imagery
      - beta (13-30 Hz): also desynchronizes, rebounds post-imagery
      - theta (4-8 Hz): sometimes elevated during cognitive effort
      - gamma (30-45 Hz): sometimes modulated by motor imagery

    Args:
        data: (n_channels, n_times) or (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        bands: Dict of {band_name: (fmin, fmax)}. None uses defaults.

    Returns:
        Dict of {band_name: power_array} where power_array is per-channel
    """
    if bands is None:
        bands = {
            "theta": (4, 8),
            "mu": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }

    freqs, psd = compute_power_spectral_density(data, sfreq, fmin=1, fmax=50)
    band_powers = {}

    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        if mask.any():
            band_powers[band_name] = psd[..., mask].mean(axis=-1)
        else:
            band_powers[band_name] = np.zeros(psd.shape[:-1])

    return band_powers


def detect_bad_channels(
    data: np.ndarray,
    threshold_std: float = 4.0,
) -> list:
    """
    Simple bad channel detection based on amplitude statistics.

    A channel is flagged as bad if its standard deviation is more than
    threshold_std times the median channel std. This catches channels
    with excessive noise or flat signals.

    Args:
        data: (n_channels, n_times)
        threshold_std: Number of median absolute deviations for threshold

    Returns:
        List of bad channel indices
    """
    channel_stds = data.std(axis=-1)
    median_std = np.median(channel_stds)
    mad = np.median(np.abs(channel_stds - median_std))

    if mad < 1e-10:
        return []

    bad_channels = []
    for ch_idx, std in enumerate(channel_stds):
        deviation = abs(std - median_std) / mad
        if deviation > threshold_std or std < 1e-10:
            bad_channels.append(ch_idx)

    return bad_channels


def segment_with_overlap(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlapping windows from EEG epochs for data augmentation.

    Sliding window approach is commonly used in EEG-BCI literature to
    increase the effective training set size. Each epoch is split into
    multiple overlapping segments.

    Args:
        X: (n_epochs, n_channels, n_times)
        y: (n_epochs,)
        window_size: Window length in samples
        step_size: Step between consecutive windows

    Returns:
        X_windowed: (n_windows, n_channels, window_size)
        y_windowed: (n_windows,) — same label for all windows from an epoch
    """
    windows = []
    labels = []

    for epoch_idx in range(len(X)):
        n_times = X.shape[-1]
        for start in range(0, n_times - window_size + 1, step_size):
            end = start + window_size
            windows.append(X[epoch_idx, :, start:end])
            labels.append(y[epoch_idx])

    return np.array(windows), np.array(labels)

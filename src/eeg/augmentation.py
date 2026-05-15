"""
EEG Data Augmentation

Augmentation strategies for EEG signals. These are critical because
EEG datasets are typically small, and augmentation helps prevent
overfitting — especially for transformer models which are data-hungry.

These augmentations are designed to be physiologically plausible:
they shouldn't create signals that would never occur in real EEG.
"""

import numpy as np
from typing import Optional


class EEGAugmenter:
    """
    Composes multiple EEG augmentation transforms.

    Each augmentation is applied independently with some probability.
    The augmentations modify the signal in ways that preserve the
    underlying neural patterns while adding realistic variability.

    Args:
        noise_std: Std dev of Gaussian noise (relative to signal std)
        noise_prob: Probability of applying noise
        time_shift_max: Maximum time shift in samples
        time_shift_prob: Probability of applying time shift
        channel_dropout_prob: Probability of dropping each channel
        channel_dropout_apply_prob: Probability of applying channel dropout at all
        scale_range: (min_scale, max_scale) for amplitude scaling
        scale_prob: Probability of applying scaling
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        noise_prob: float = 0.5,
        time_shift_max: int = 10,
        time_shift_prob: float = 0.3,
        channel_dropout_prob: float = 0.05,
        channel_dropout_apply_prob: float = 0.3,
        scale_range: tuple = (0.8, 1.2),
        scale_prob: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.time_shift_max = time_shift_max
        self.time_shift_prob = time_shift_prob
        self.channel_dropout_prob = channel_dropout_prob
        self.channel_dropout_apply_prob = channel_dropout_apply_prob
        self.scale_range = scale_range
        self.scale_prob = scale_prob
        self.rng = np.random.RandomState(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to a single EEG trial.

        Args:
            x: (n_channels, n_times)

        Returns:
            Augmented copy of x
        """
        x = x.copy()

        if self.rng.random() < self.noise_prob:
            x = self._add_noise(x)

        if self.rng.random() < self.time_shift_prob:
            x = self._time_shift(x)

        if self.rng.random() < self.channel_dropout_apply_prob:
            x = self._channel_dropout(x)

        if self.rng.random() < self.scale_prob:
            x = self._amplitude_scale(x)

        return x

    def _add_noise(self, x: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise proportional to signal standard deviation.

        Simulates the inherent noise variability in EEG recordings.
        The noise is scaled per-channel so channels with larger signals
        get proportionally more noise.
        """
        ch_std = x.std(axis=-1, keepdims=True)
        ch_std[ch_std < 1e-8] = 1.0
        noise = self.rng.randn(*x.shape) * ch_std * self.noise_std
        return x + noise

    def _time_shift(self, x: np.ndarray) -> np.ndarray:
        """
        Shift signal in time by a random amount.

        Small temporal shifts are realistic because the exact onset
        of motor imagery varies slightly between trials. This helps
        the model be robust to timing variations.
        """
        shift = self.rng.randint(-self.time_shift_max, self.time_shift_max + 1)
        if shift == 0:
            return x
        return np.roll(x, shift, axis=-1)

    def _channel_dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Randomly zero out entire channels.

        Simulates electrode contact issues or bad channels that weren't
        caught during preprocessing. Also acts as a regularizer, forcing
        the model to not rely on any single electrode.
        """
        n_channels = x.shape[0]
        mask = self.rng.random(n_channels) > self.channel_dropout_prob
        x = x * mask[:, np.newaxis]
        return x

    def _amplitude_scale(self, x: np.ndarray) -> np.ndarray:
        """
        Random amplitude scaling.

        EEG amplitude varies between sessions and subjects due to
        differences in electrode impedance, scalp thickness, etc.
        This augmentation helps with generalization.
        """
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale

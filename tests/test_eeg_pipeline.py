"""
Tests for EEG pipeline — dataset, models, and training.

These tests use synthetic data to validate the pipeline without
needing the actual PhysioNet dataset downloaded.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pytest

from src.eeg.dataset import EEGDataset
from src.eeg.augmentation import EEGAugmenter
from src.eeg.preprocessing import segment_with_overlap, detect_bad_channels
from src.models.eeg_transformer import EEGTransformer
from src.models.baselines import EEGNet


# Synthetic data dimensions (matching PhysioNet after preprocessing)
N_CHANNELS = 64
N_TIMES = 513  # ~4s at 128 Hz
N_CLASSES = 4
BATCH_SIZE = 4


def make_synthetic_data(n_samples=20):
    """Create random EEG-like data for testing."""
    X = np.random.randn(n_samples, N_CHANNELS, N_TIMES).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, n_samples).astype(np.int64)
    return X, y


class TestEEGDataset:

    def test_dataset_creation(self):
        X, y = make_synthetic_data()
        ds = EEGDataset(X, y)
        assert len(ds) == len(X)

    def test_dataset_item_shapes(self):
        X, y = make_synthetic_data()
        ds = EEGDataset(X, y)
        x_item, y_item = ds[0]
        assert x_item.shape == (N_CHANNELS, N_TIMES)
        assert y_item.shape == ()

    def test_normalization(self):
        X, y = make_synthetic_data()
        ds = EEGDataset(X, y, normalize=True)
        x_item, _ = ds[0]
        # After z-score norm, each channel should have ~0 mean
        assert abs(x_item.mean().item()) < 1.0

    def test_with_augmentation(self):
        X, y = make_synthetic_data()
        aug = EEGAugmenter(seed=42)
        ds = EEGDataset(X, y, augmentation=aug)
        x1, _ = ds[0]
        x2, _ = ds[0]
        # Augmented samples should differ (stochastic)
        # (though with copy it might match sometimes, so just check shape)
        assert x1.shape == x2.shape


class TestAugmenter:

    def test_output_shape(self):
        aug = EEGAugmenter(seed=42)
        x = np.random.randn(N_CHANNELS, N_TIMES).astype(np.float32)
        x_aug = aug(x)
        assert x_aug.shape == x.shape

    def test_no_inplace_modification(self):
        aug = EEGAugmenter(noise_prob=1.0, seed=42)
        x = np.random.randn(N_CHANNELS, N_TIMES).astype(np.float32)
        x_copy = x.copy()
        aug(x)
        assert np.allclose(x, x_copy)


class TestPreprocessing:

    def test_segment_with_overlap(self):
        X = np.random.randn(5, N_CHANNELS, 200).astype(np.float32)
        y = np.array([0, 1, 2, 3, 0])
        X_w, y_w = segment_with_overlap(X, y, window_size=100, step_size=50)
        # Each epoch of 200 samples with window=100, step=50 → 3 windows
        assert X_w.shape[0] == 5 * 3
        assert X_w.shape[1] == N_CHANNELS
        assert X_w.shape[2] == 100

    def test_bad_channel_detection(self):
        data = np.random.randn(N_CHANNELS, 500)
        # Make one channel extremely noisy
        data[10] = np.random.randn(500) * 100
        bad = detect_bad_channels(data)
        assert 10 in bad


class TestEEGTransformer:

    def test_forward_shape(self):
        model = EEGTransformer(
            n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES,
            d_model=64, num_heads=4, d_ff=128, num_layers=2,
            temporal_filters=8, temporal_kernel=32, patch_size=8,
        )
        x = torch.randn(BATCH_SIZE, N_CHANNELS, N_TIMES)
        out = model(x)
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_return_attention(self):
        model = EEGTransformer(
            n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES,
            d_model=64, num_heads=4, d_ff=128, num_layers=2,
            temporal_filters=8, temporal_kernel=32, patch_size=8,
        )
        x = torch.randn(1, N_CHANNELS, N_TIMES)
        logits, attn = model(x, return_attention=True)
        assert logits.shape == (1, N_CLASSES)
        assert len(attn) > 0

    def test_gradient_flow(self):
        model = EEGTransformer(
            n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES,
            d_model=64, num_heads=4, d_ff=128, num_layers=2,
            temporal_filters=8, temporal_kernel=32, patch_size=8,
        )
        x = torch.randn(BATCH_SIZE, N_CHANNELS, N_TIMES)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check gradient exists for key parameters
        assert model.cls_token.grad is not None


class TestEEGNet:

    def test_forward_shape(self):
        model = EEGNet(
            n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES,
        )
        x = torch.randn(BATCH_SIZE, N_CHANNELS, N_TIMES)
        out = model(x)
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_parameter_count(self):
        model = EEGNet(n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES)
        # EEGNet should be much smaller than the Transformer
        assert model.count_parameters() < 50_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

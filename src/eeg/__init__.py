"""EEG data loading, preprocessing, and augmentation."""

from src.eeg.dataset import (
    EEGDataset,
    download_physionet_data,
    load_subject_data,
    load_all_subjects,
    create_subject_dependent_splits,
    create_cross_subject_splits,
    CLASS_NAMES,
)
from src.eeg.augmentation import EEGAugmenter
from src.eeg.preprocessing import (
    compute_power_spectral_density,
    compute_band_power,
    detect_bad_channels,
    segment_with_overlap,
)

__all__ = [
    "EEGDataset",
    "EEGAugmenter",
    "download_physionet_data",
    "load_subject_data",
    "load_all_subjects",
    "create_subject_dependent_splits",
    "create_cross_subject_splits",
    "compute_power_spectral_density",
    "compute_band_power",
    "detect_bad_channels",
    "segment_with_overlap",
    "CLASS_NAMES",
]

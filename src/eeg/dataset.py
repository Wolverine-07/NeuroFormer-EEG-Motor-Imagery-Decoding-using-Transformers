"""
EEG Dataset — PhysioNet Motor Movement/Imagery Dataset

Handles downloading, loading, and serving the PhysioNet EEG Motor
Movement/Imagery dataset for training and evaluation.

Dataset details:
  - 109 subjects, 64 EEG channels, 160 Hz sampling rate
  - Subjects perform motor execution and motor imagery tasks
  - We focus on motor imagery runs (imagined movement):
      * Left fist
      * Right fist
      * Both fists
      * Both feet

Reference:
  Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet:
  Components of a New Research Resource for Complex Physiologic Signals"
  Circulation, 101(23), 2000.
  https://physionet.org/content/eegmmidb/1.0.0/
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# PhysioNet run mapping
# Runs 1-2: baselines (eyes open, eyes closed)
# Runs 3,7,11: Task 1 — open/close left or right fist (execution)
# Runs 4,8,12: Task 2 — imagine opening/closing left or right fist
# Runs 5,9,13: Task 3 — open/close both fists or both feet (execution)
# Runs 6,10,14: Task 4 — imagine opening/closing both fists or both feet

# Motor imagery runs (what we use for BCI classification)
MI_RUNS_LR = [4, 8, 12]    # Left vs right fist imagery
MI_RUNS_HF = [6, 10, 14]   # Both hands vs both feet imagery

# Event codes in the dataset
EVENT_MAPPING_LR = {
    "T1": 0,   # Left fist
    "T2": 1,   # Right fist
}
EVENT_MAPPING_HF = {
    "T1": 2,   # Both fists
    "T2": 3,   # Both feet
}

# Class labels for 4-class problem
CLASS_NAMES = ["left_fist", "right_fist", "both_fists", "both_feet"]

# Subjects known to have issues in the dataset (commonly excluded in literature)
BAD_SUBJECTS = [88, 92, 100, 104]


def download_physionet_data(
    data_dir: str,
    subjects: Optional[List[int]] = None,
) -> str:
    """
    Download PhysioNet EEG Motor Imagery dataset using MNE.

    Args:
        data_dir: Directory to store downloaded data
        subjects: List of subject IDs (1-109). None = all subjects.

    Returns:
        Path to the downloaded data directory
    """
    import mne

    if subjects is None:
        subjects = [s for s in range(1, 110) if s not in BAD_SUBJECTS]

    os.makedirs(data_dir, exist_ok=True)

    logger.info(f"Downloading PhysioNet data for {len(subjects)} subjects to {data_dir}")

    # MNE handles the download and caching internally
    # We just need to trigger a load for each subject to ensure data is cached
    runs = MI_RUNS_LR + MI_RUNS_HF  # All motor imagery runs

    for i, subj in enumerate(subjects):
        if (i + 1) % 10 == 0:
            logger.info(f"  Downloaded {i+1}/{len(subjects)} subjects...")
        try:
            mne.datasets.eegbci.load_data(subj, runs, path=data_dir, update_path=False)
        except Exception as e:
            logger.warning(f"  Failed to download subject {subj}: {e}")

    logger.info("Download complete.")
    return data_dir


def load_subject_data(
    subject_id: int,
    data_dir: str,
    runs: List[int] = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
    bandpass: Tuple[float, float] = (4.0, 40.0),
    resample_freq: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess EEG data for a single subject.

    Pipeline:
      1. Load raw EDF files via MNE
      2. Apply bandpass filter (4-40 Hz for mu/beta rhythms)
      3. Extract epochs around motor imagery events
      4. Baseline correction
      5. Return as numpy arrays

    The 4-40 Hz bandpass captures the mu rhythm (8-13 Hz) and beta rhythm
    (13-30 Hz) which are the primary neural signatures of motor imagery.
    These rhythms show event-related desynchronization (ERD) during
    imagined movement, which is what the model needs to learn.

    Args:
        subject_id: Subject number (1-109)
        data_dir: Path to PhysioNet data
        runs: Which runs to load. None = all MI runs.
        tmin: Epoch start time relative to event onset (seconds)
        tmax: Epoch end time relative to event onset (seconds)
        bandpass: (low_freq, high_freq) for bandpass filter
        resample_freq: Optional resampling frequency. None = keep original 160 Hz.

    Returns:
        X: EEG epochs (n_epochs, n_channels, n_times)
        y: Labels (n_epochs,) — 0: left, 1: right, 2: both_fists, 3: both_feet
    """
    import mne

    if runs is None:
        runs = MI_RUNS_LR + MI_RUNS_HF

    all_epochs = []
    all_labels = []

    for run in runs:
        try:
            # Load raw data
            raw_fnames = mne.datasets.eegbci.load_data(
                subject_id, [run], path=data_dir, update_path=False
            )
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)

            # Standardize channel names to 10-20 system
            mne.datasets.eegbci.standardize(raw)
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage, on_missing="warn")

            # Bandpass filter
            raw.filter(bandpass[0], bandpass[1], fir_design="firwin", verbose=False)

            # Optional resampling
            if resample_freq is not None:
                raw.resample(resample_freq, verbose=False)

            # Extract events
            events, event_id = mne.events_from_annotations(raw, verbose=False)

            # Determine label mapping based on run type
            if run in MI_RUNS_LR:
                label_map = {event_id.get("T1", -1): 0, event_id.get("T2", -1): 1}
            else:  # MI_RUNS_HF
                label_map = {event_id.get("T1", -1): 2, event_id.get("T2", -1): 3}

            # Filter to only T1 and T2 events (ignore T0 = rest)
            valid_events_ids = [event_id.get("T1", -1), event_id.get("T2", -1)]
            valid_events_ids = [e for e in valid_events_ids if e != -1]

            if not valid_events_ids:
                continue

            # Create epochs
            epochs = mne.Epochs(
                raw, events,
                event_id={k: v for k, v in event_id.items() if k in ["T1", "T2"]},
                tmin=tmin, tmax=tmax,
                baseline=(tmin, 0),  # Baseline correction using pre-stimulus period
                preload=True,
                verbose=False,
            )

            # Drop bad epochs (large amplitude artifacts)
            epochs.drop_bad(reject=dict(eeg=100e-6), verbose=False)

            if len(epochs) == 0:
                continue

            # Get data and labels
            epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
            epoch_events = epochs.events[:, 2]

            # Map event codes to our label scheme
            epoch_labels = np.array([label_map.get(e, -1) for e in epoch_events])

            # Filter out any unmapped events
            valid_mask = epoch_labels >= 0
            epoch_data = epoch_data[valid_mask]
            epoch_labels = epoch_labels[valid_mask]

            all_epochs.append(epoch_data)
            all_labels.append(epoch_labels)

        except Exception as e:
            logger.warning(f"Error loading subject {subject_id}, run {run}: {e}")
            continue

    if not all_epochs:
        return np.array([]), np.array([])

    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    return X, y


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG motor imagery data.

    Wraps preprocessed EEG epochs as a torch Dataset for use with DataLoader.
    Applies optional augmentation and z-score normalization.

    Args:
        X: EEG data (n_samples, n_channels, n_times)
        y: Labels (n_samples,)
        normalize: Whether to z-score normalize per channel per trial
        augmentation: Optional augmentation function
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = True,
        augmentation=None,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.normalize = normalize
        self.augmentation = augmentation

        if normalize and len(X) > 0:
            self._normalize()

    def _normalize(self):
        """Z-score normalization per channel per trial."""
        # For each trial, normalize each channel independently
        # This removes per-trial amplitude differences while preserving
        # relative channel patterns that encode spatial information
        mean = self.X.mean(axis=-1, keepdims=True)
        std = self.X.std(axis=-1, keepdims=True)
        std[std < 1e-8] = 1.0  # avoid div by zero for flat channels
        self.X = (self.X - mean) / std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].copy()
        y = self.y[idx]

        # Apply augmentation if provided
        if self.augmentation is not None:
            x = self.augmentation(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def load_all_subjects(
    data_dir: str,
    subjects: Optional[List[int]] = None,
    tmin: float = 0.0,
    tmax: float = 4.0,
    bandpass: Tuple[float, float] = (4.0, 40.0),
    resample_freq: Optional[float] = 128.0,
    verbose: bool = True,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load preprocessed EEG data for multiple subjects.

    Returns a dict mapping subject_id -> (X, y) so we can flexibly
    create subject-dependent or cross-subject splits later.

    Args:
        data_dir: Path to PhysioNet data
        subjects: List of subject IDs. None = all valid subjects.
        tmin, tmax: Epoch time window
        bandpass: Filter frequencies
        resample_freq: Target sampling rate (128 Hz is common in BCI literature;
                       reduces computation while keeping relevant frequencies)
        verbose: Print progress

    Returns:
        Dict mapping subject_id -> (X, y)
    """
    if subjects is None:
        subjects = [s for s in range(1, 110) if s not in BAD_SUBJECTS]

    subject_data = {}
    total_epochs = 0

    for i, subj in enumerate(subjects):
        X, y = load_subject_data(
            subj, data_dir,
            tmin=tmin, tmax=tmax,
            bandpass=bandpass,
            resample_freq=resample_freq,
        )

        if len(X) > 0:
            subject_data[subj] = (X, y)
            total_epochs += len(X)

            if verbose and (i + 1) % 10 == 0:
                print(f"  Loaded {i+1}/{len(subjects)} subjects ({total_epochs} total epochs)")

    if verbose:
        print(f"Loaded {len(subject_data)} subjects, {total_epochs} total epochs")
        # Print class distribution
        all_y = np.concatenate([v[1] for v in subject_data.values()])
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            count = (all_y == cls_idx).sum()
            print(f"  {cls_name}: {count} epochs ({100*count/len(all_y):.1f}%)")

    return subject_data


def create_subject_dependent_splits(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Create train/test splits within each subject.

    This is the easier evaluation protocol — training and testing on the
    same subject's data. Still useful because it shows the model can learn
    individual brain patterns.

    Important: We split by trial, not by time point, to avoid data leakage
    from temporal autocorrelation in EEG signals.

    Args:
        subject_data: Dict from load_all_subjects
        test_ratio: Fraction of trials for testing
        seed: Random seed for reproducibility

    Returns:
        Dict[subject_id, {"train": (X, y), "test": (X, y)}]
    """
    rng = np.random.RandomState(seed)
    splits = {}

    for subj_id, (X, y) in subject_data.items():
        n_samples = len(X)
        indices = rng.permutation(n_samples)
        split_point = int(n_samples * (1 - test_ratio))

        train_idx = indices[:split_point]
        test_idx = indices[split_point:]

        splits[subj_id] = {
            "train": (X[train_idx], y[train_idx]),
            "test": (X[test_idx], y[test_idx]),
        }

    return splits


def create_cross_subject_splits(
    subject_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Create cross-subject train/test splits using K-fold over subjects.

    This is the harder but more realistic evaluation protocol — the model
    must generalize to completely unseen subjects. This is critical for
    real BCI deployment where you can't collect training data from every user.

    Args:
        subject_data: Dict from load_all_subjects
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of fold dicts, each with {"train": (X, y), "test": (X, y),
                                       "train_subjects": [...], "test_subjects": [...]}
    """
    subject_ids = sorted(subject_data.keys())
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_subj_idx, test_subj_idx in kf.split(subject_ids):
        train_subjects = [subject_ids[i] for i in train_subj_idx]
        test_subjects = [subject_ids[i] for i in test_subj_idx]

        X_train = np.concatenate([subject_data[s][0] for s in train_subjects])
        y_train = np.concatenate([subject_data[s][1] for s in train_subjects])
        X_test = np.concatenate([subject_data[s][0] for s in test_subjects])
        y_test = np.concatenate([subject_data[s][1] for s in test_subjects])

        folds.append({
            "train": (X_train, y_train),
            "test": (X_test, y_test),
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
        })

    return folds

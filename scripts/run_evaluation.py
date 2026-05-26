#!/usr/bin/env python3
"""
Run the full evaluation pipeline using synthetic EEG data.

This script demonstrates the complete pipeline:
  1. Trains EEG-Transformer on synthetic motor imagery data
  2. Trains EEGNet baseline for comparison
  3. Generates all visualizations (confusion matrix, attention maps, etc.)
  4. Saves results to assets/

The synthetic data mimics real PhysioNet EEG:
  - 64 channels, 128 Hz, 4-second trials
  - Class-dependent mu/beta patterns in motor cortex channels
  - Realistic noise levels

For real data evaluation, use scripts/train_eeg.py instead.

Usage:
    python scripts/run_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eeg.dataset import EEGDataset, CLASS_NAMES
from src.eeg.augmentation import EEGAugmenter
from src.models.eeg_transformer import EEGTransformer
from src.models.baselines import EEGNet
from src.training.metrics import compute_classification_metrics
from src.visualization.training_curves import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_model_comparison,
)
from src.visualization.attention_maps import plot_all_heads, plot_cls_attention_over_patches
from src.visualization.eeg_plots import plot_class_averaged_signals


# ============================================================
# Synthetic data generation (mimics real motor imagery EEG)
# ============================================================

def generate_synthetic_eeg(
    n_subjects: int = 10,
    n_trials_per_class: int = 30,
    n_channels: int = 64,
    sfreq: float = 128.0,
    duration: float = 4.0,
    seed: int = 42,
):
    """
    Generate synthetic EEG data with class-dependent neural patterns.

    Simulates the key signatures of motor imagery:
      - Mu rhythm (10 Hz) desynchronization in contralateral motor cortex
      - Beta rhythm (20 Hz) modulation
      - Class-specific spatial patterns

    This gives the model something real to learn, rather than pure noise.
    """
    rng = np.random.RandomState(seed)
    n_times = int(sfreq * duration) + 1  # 513 samples at 128 Hz
    n_classes = 4
    t = np.linspace(0, duration, n_times)

    subject_data = {}

    for subj in range(1, n_subjects + 1):
        # Each subject has slightly different brain patterns
        subj_offset = rng.randn() * 0.1
        all_X = []
        all_y = []

        for cls in range(n_classes):
            for trial in range(n_trials_per_class):
                # Base EEG: pink noise (1/f spectrum)
                eeg = rng.randn(n_channels, n_times) * 0.3

                # Add alpha rhythm (background, ~10 Hz)
                alpha_phase = rng.uniform(0, 2 * np.pi)
                for ch in range(n_channels):
                    eeg[ch] += 0.15 * np.sin(2 * np.pi * 10 * t + alpha_phase + rng.randn() * 0.5)

                # Class-specific motor imagery patterns
                # C3 channels: ~20-25, C4 channels: ~30-35 (approximate for 64-ch)
                signal_strength = 0.4 + subj_offset + rng.randn() * 0.05

                if cls == 0:  # Left fist — ERD in right motor cortex (C4)
                    for ch in range(30, 36):
                        eeg[ch] += signal_strength * np.sin(2 * np.pi * 10 * t) * np.exp(-t / 2.5)
                        eeg[ch] += 0.2 * np.sin(2 * np.pi * 20 * t) * np.exp(-t / 2)
                elif cls == 1:  # Right fist — ERD in left motor cortex (C3)
                    for ch in range(20, 26):
                        eeg[ch] += signal_strength * np.sin(2 * np.pi * 10 * t) * np.exp(-t / 2.5)
                        eeg[ch] += 0.2 * np.sin(2 * np.pi * 20 * t) * np.exp(-t / 2)
                elif cls == 2:  # Both fists — bilateral
                    for ch in list(range(20, 26)) + list(range(30, 36)):
                        eeg[ch] += signal_strength * 0.7 * np.sin(2 * np.pi * 12 * t) * np.exp(-t / 2)
                else:  # Both feet — central (Cz ~ ch 0-5)
                    for ch in range(0, 8):
                        eeg[ch] += signal_strength * np.sin(2 * np.pi * 8 * t) * np.exp(-t / 3)

                all_X.append(eeg)
                all_y.append(cls)

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)

        # Shuffle
        perm = rng.permutation(len(X))
        subject_data[subj] = (X[perm], y[perm])

    return subject_data


def train_model(model, train_loader, test_loader, device, n_epochs=60):
    """Train a model, return metrics and training history."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    total_steps = n_epochs * len(train_loader)
    warmup = 5 * len(train_loader)

    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(total_steps - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * len(by)
            correct += (logits.argmax(1) == by).sum().item()
            total += len(by)

        train_losses.append(epoch_loss / total)
        train_accs.append(correct / total)

        # Eval
        model.eval()
        epoch_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                epoch_loss += loss.item() * len(by)
                preds = logits.argmax(1)
                correct += (preds == by).sum().item()
                total += len(by)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(by.cpu().numpy())

        val_losses.append(epoch_loss / total)
        val_accs.append(correct / total)

        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train: {train_accs[-1]:.4f} | Val: {val_accs[-1]:.4f}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    # Final metrics
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(by.cpu().numpy())

    metrics = compute_classification_metrics(np.array(all_labels), np.array(all_preds), CLASS_NAMES)

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }

    return metrics, history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    assets_dir = Path("assets/results")
    plots_dir = Path("assets/plots")
    assets_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    N_CHANNELS = 64
    N_TIMES = 513
    N_CLASSES = 4

    # ============================================================
    # Generate data
    # ============================================================
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC EEG DATA")
    print("=" * 60)

    subject_data = generate_synthetic_eeg(n_subjects=10, n_trials_per_class=30)
    print(f"Generated data for {len(subject_data)} subjects")
    print(f"Each subject: {30*4} trials, {N_CHANNELS} channels, {N_TIMES} time points")

    # Combine all subjects for a pooled evaluation
    all_X = np.concatenate([d[0] for d in subject_data.values()])
    all_y = np.concatenate([d[1] for d in subject_data.values()])
    print(f"Total: {len(all_X)} trials")

    # Split 80/20
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(all_X))
    split = int(0.8 * len(all_X))
    train_X, train_y = all_X[perm[:split]], all_y[perm[:split]]
    test_X, test_y = all_X[perm[split:]], all_y[perm[split:]]

    print(f"Train: {len(train_X)} | Test: {len(test_X)}")

    # Class distribution
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {(all_y == i).sum()} trials")

    # --- Plot class-averaged signals ---
    plot_class_averaged_signals(
        all_X, all_y, CLASS_NAMES,
        channels=[0, 22, 32, 50],
        channel_names=["Cz (central)", "C3 (left motor)", "C4 (right motor)", "Pz (parietal)"],
        sfreq=128.0,
        save_path=str(plots_dir / "class_averaged_signals.png"),
    )

    # Create datasets
    augmenter = EEGAugmenter(noise_std=0.1, noise_prob=0.5, seed=42)
    train_ds = EEGDataset(train_X, train_y, normalize=True, augmentation=augmenter)
    test_ds = EEGDataset(test_X, test_y, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # ============================================================
    # Train EEG-Transformer
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING EEG-TRANSFORMER")
    print("=" * 60)

    transformer_model = EEGTransformer(
        n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES,
        d_model=128, num_heads=4, d_ff=256, num_layers=4, dropout=0.3,
        temporal_filters=16, temporal_kernel=64, patch_size=16,
    ).to(device)
    print(f"Parameters: {transformer_model.count_parameters():,}")

    t0 = time.time()
    transformer_metrics, transformer_history = train_model(
        transformer_model, train_loader, test_loader, device, n_epochs=60
    )
    t_time = time.time() - t0

    print(f"\nEEG-Transformer Results ({t_time:.0f}s):")
    print(f"  Accuracy:     {transformer_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):   {transformer_metrics['f1_macro']:.4f}")
    print(f"  Cohen's κ:    {transformer_metrics['cohen_kappa']:.4f}")

    # ============================================================
    # Train EEGNet baseline
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING EEGNET BASELINE")
    print("=" * 60)

    eegnet_model = EEGNet(
        n_channels=N_CHANNELS, n_times=N_TIMES, n_classes=N_CLASSES,
    ).to(device)
    print(f"Parameters: {eegnet_model.count_parameters():,}")

    t0 = time.time()
    eegnet_metrics, eegnet_history = train_model(
        eegnet_model, train_loader, test_loader, device, n_epochs=60
    )
    e_time = time.time() - t0

    print(f"\nEEGNet Results ({e_time:.0f}s):")
    print(f"  Accuracy:     {eegnet_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):   {eegnet_metrics['f1_macro']:.4f}")
    print(f"  Cohen's κ:    {eegnet_metrics['cohen_kappa']:.4f}")

    # ============================================================
    # Generate plots
    # ============================================================
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Training curves
    plot_training_curves(
        transformer_history["train_losses"],
        transformer_history["val_losses"],
        transformer_history["train_accs"],
        transformer_history["val_accs"],
        title="EEG-Transformer Training",
        save_path=str(plots_dir / "transformer_training_curves.png"),
    )
    plot_training_curves(
        eegnet_history["train_losses"],
        eegnet_history["val_losses"],
        eegnet_history["train_accs"],
        eegnet_history["val_accs"],
        title="EEGNet Training",
        save_path=str(plots_dir / "eegnet_training_curves.png"),
    )

    # Confusion matrices
    plot_confusion_matrix(
        transformer_metrics["confusion_matrix"],
        CLASS_NAMES,
        title="EEG-Transformer — Confusion Matrix",
        save_path=str(plots_dir / "transformer_confusion_matrix.png"),
    )
    plot_confusion_matrix(
        eegnet_metrics["confusion_matrix"],
        CLASS_NAMES,
        title="EEGNet — Confusion Matrix",
        save_path=str(plots_dir / "eegnet_confusion_matrix.png"),
    )

    # Model comparison
    plot_model_comparison(
        {
            "EEG-Transformer": {
                "accuracy": transformer_metrics["accuracy"],
                "accuracy_std": 0,
                "f1_macro": transformer_metrics["f1_macro"],
                "f1_macro_std": 0,
            },
            "EEGNet": {
                "accuracy": eegnet_metrics["accuracy"],
                "accuracy_std": 0,
                "f1_macro": eegnet_metrics["f1_macro"],
                "f1_macro_std": 0,
            },
        },
        metric="accuracy",
        save_path=str(plots_dir / "model_comparison_accuracy.png"),
    )

    # Attention maps
    transformer_model.eval()
    with torch.no_grad():
        sample = torch.tensor(test_X[:1], dtype=torch.float32).to(device)
        # Normalize like the dataset does
        mean = sample.mean(dim=-1, keepdim=True)
        std = sample.std(dim=-1, keepdim=True)
        std[std < 1e-8] = 1.0
        sample = (sample - mean) / std

        _, attn_weights = transformer_model(sample, return_attention=True)

    if attn_weights:
        last_attn = attn_weights[-1].squeeze().cpu().numpy()
        plot_all_heads(last_attn, layer_idx=3,
                       save_path=str(plots_dir / "attention_heads_layer3.png"))

        attn_np = [w.squeeze().cpu().numpy() for w in attn_weights]
        plot_cls_attention_over_patches(attn_np,
                                        save_path=str(plots_dir / "cls_attention_over_patches.png"))

    # ============================================================
    # Save results
    # ============================================================
    results = {
        "eeg_transformer": {
            "accuracy": transformer_metrics["accuracy"],
            "f1_macro": transformer_metrics["f1_macro"],
            "cohen_kappa": transformer_metrics["cohen_kappa"],
            "parameters": transformer_model.count_parameters(),
            "training_time_s": round(t_time, 1),
        },
        "eegnet": {
            "accuracy": eegnet_metrics["accuracy"],
            "f1_macro": eegnet_metrics["f1_macro"],
            "cohen_kappa": eegnet_metrics["cohen_kappa"],
            "parameters": eegnet_model.count_parameters(),
            "training_time_s": round(e_time, 1),
        },
    }

    with open(assets_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model checkpoints
    ckpt_dir = Path("assets/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(transformer_model.state_dict(), ckpt_dir / "eeg_transformer.pt")
    torch.save(eegnet_model.state_dict(), ckpt_dir / "eegnet.pt")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1':>10} {'κ':>10} {'Params':>12}")
    print("-" * 62)
    print(f"{'EEG-Transformer':<20} {transformer_metrics['accuracy']:>10.4f} "
          f"{transformer_metrics['f1_macro']:>10.4f} "
          f"{transformer_metrics['cohen_kappa']:>10.4f} "
          f"{transformer_model.count_parameters():>12,}")
    print(f"{'EEGNet':<20} {eegnet_metrics['accuracy']:>10.4f} "
          f"{eegnet_metrics['f1_macro']:>10.4f} "
          f"{eegnet_metrics['cohen_kappa']:>10.4f} "
          f"{eegnet_model.count_parameters():>12,}")
    print(f"{'Chance level':<20} {'0.2500':>10}")
    print()
    print(f"Plots saved to: {plots_dir}/")
    print(f"Results saved to: {assets_dir}/evaluation_results.json")
    print(f"Checkpoints saved to: {ckpt_dir}/")


if __name__ == "__main__":
    main()

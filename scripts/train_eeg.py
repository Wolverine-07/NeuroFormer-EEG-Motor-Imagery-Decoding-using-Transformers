#!/usr/bin/env python3
"""
Train EEG-Transformer on PhysioNet Motor Imagery dataset.

Supports two evaluation protocols:
  1. Subject-dependent: train/test within each subject
  2. Cross-subject: K-fold over subjects (leave-subjects-out)

Usage:
    # Subject-dependent (default)
    python scripts/train_eeg.py --config configs/eeg_subject_dependent.yaml

    # Cross-subject
    python scripts/train_eeg.py --config configs/eeg_cross_subject.yaml --mode cross_subject

    # Use EEGNet baseline
    python scripts/train_eeg.py --config configs/eeg_subject_dependent.yaml --model eegnet

    # Quick test with fewer subjects
    python scripts/train_eeg.py --subjects 1 2 3 4 5
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path

# Optional wandb import
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from src.eeg.dataset import (
    download_physionet_data,
    load_all_subjects,
    create_subject_dependent_splits,
    create_cross_subject_splits,
    EEGDataset,
    CLASS_NAMES,
)
from src.eeg.augmentation import EEGAugmenter
from src.models.eeg_transformer import EEGTransformer
from src.models.baselines import EEGNet
from src.training.metrics import compute_classification_metrics


def build_model(config: dict, n_times: int, device: torch.device) -> nn.Module:
    """Create model based on config."""
    mc = config["model"]
    name = mc.get("name", "eeg_transformer")

    if name == "eeg_transformer":
        model = EEGTransformer(
            n_channels=mc["n_channels"],
            n_times=n_times,
            n_classes=mc["n_classes"],
            d_model=mc["d_model"],
            num_heads=mc["num_heads"],
            d_ff=mc["d_ff"],
            num_layers=mc["num_layers"],
            dropout=mc["dropout"],
            temporal_filters=mc.get("temporal_filters", 16),
            temporal_kernel=mc.get("temporal_kernel", 64),
            patch_size=mc.get("patch_size", 16),
        )
    elif name == "eegnet":
        model = EEGNet(
            n_channels=mc["n_channels"],
            n_times=n_times,
            n_classes=mc["n_classes"],
            dropout=mc["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    model = model.to(device)
    print(f"Model: {name} | Parameters: {model.count_parameters():,}")
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(batch_y)
        total_correct += (logits.argmax(1) == batch_y).sum().item()
        total_samples += len(batch_y)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model, return loss, accuracy, and predictions."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * len(batch_y)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / len(all_labels)

    metrics = compute_classification_metrics(all_labels, all_preds, CLASS_NAMES)
    metrics["loss"] = avg_loss

    return metrics


def train_and_evaluate(
    train_X, train_y, test_X, test_y,
    config, device, run_name="run",
    augmenter=None,
):
    """
    Full training loop for one split.

    Returns test metrics dict.
    """
    tc = config["training"]
    n_times = train_X.shape[-1]

    # Create datasets
    train_dataset = EEGDataset(train_X, train_y, normalize=True, augmentation=augmenter)
    test_dataset = EEGDataset(test_X, test_y, normalize=True, augmentation=None)

    train_loader = DataLoader(
        train_dataset, batch_size=tc["batch_size"], shuffle=True,
        num_workers=0, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=tc["batch_size"], shuffle=False,
        num_workers=0,
    )

    # Build model
    model = build_model(config, n_times, device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )

    # Cosine annealing scheduler with warmup
    total_steps = tc["num_epochs"] * len(train_loader)
    warmup_steps = tc.get("warmup_epochs", 5) * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_val_acc = 0
    patience_counter = 0
    patience = tc.get("patience", 15)
    best_model_state = None

    for epoch in range(1, tc["num_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device
        )

        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, test_loader, criterion, device)
            val_acc = val_metrics["accuracy"]

            print(
                f"  [{run_name}] Epoch {epoch:3d} | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"Val: loss={val_metrics['loss']:.4f} acc={val_acc:.4f} "
                f"kappa={val_metrics['cohen_kappa']:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 5  # we check every 5 epochs

            if patience_counter >= patience:
                print(f"  [{run_name}] Early stopping at epoch {epoch}")
                break

    # Load best model and compute final test metrics
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    final_metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"  [{run_name}] Final: acc={final_metrics['accuracy']:.4f} "
        f"f1={final_metrics['f1_macro']:.4f} "
        f"kappa={final_metrics['cohen_kappa']:.4f}"
    )

    return final_metrics, model


def run_subject_dependent(config, device, subjects=None):
    """Run subject-dependent evaluation."""
    dc = config["data"]

    print("=" * 70)
    print("SUBJECT-DEPENDENT EVALUATION")
    print("=" * 70)

    # Download and load data
    print("\nDownloading/loading PhysioNet data...")
    download_physionet_data(dc["data_dir"], subjects)
    subject_data = load_all_subjects(
        dc["data_dir"], subjects,
        tmin=dc["tmin"], tmax=dc["tmax"],
        bandpass=tuple(dc["bandpass"]),
        resample_freq=dc.get("resample_freq", 128.0),
    )

    # Create splits
    splits = create_subject_dependent_splits(
        subject_data,
        test_ratio=dc.get("test_ratio", 0.2),
        seed=config["training"]["seed"],
    )

    # Augmenter
    augmenter = None
    ac = config.get("augmentation", {})
    if ac.get("enabled", False):
        augmenter = EEGAugmenter(
            noise_std=ac.get("noise_std", 0.1),
            noise_prob=ac.get("noise_prob", 0.5),
            time_shift_max=ac.get("time_shift_max", 8),
            time_shift_prob=ac.get("time_shift_prob", 0.3),
            channel_dropout_prob=ac.get("channel_dropout_prob", 0.05),
            scale_range=tuple(ac.get("scale_range", [0.85, 1.15])),
        )

    # Train on each subject
    all_metrics = {}
    subject_ids = sorted(splits.keys())

    for subj_id in subject_ids:
        train_X, train_y = splits[subj_id]["train"]
        test_X, test_y = splits[subj_id]["test"]

        if len(train_X) < 10 or len(test_X) < 3:
            print(f"\nSkipping subject {subj_id} (too few trials)")
            continue

        print(f"\n--- Subject {subj_id} ({len(train_X)} train, {len(test_X)} test) ---")

        metrics, _ = train_and_evaluate(
            train_X, train_y, test_X, test_y,
            config, device,
            run_name=f"S{subj_id:03d}",
            augmenter=augmenter,
        )
        all_metrics[subj_id] = metrics

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Subject-Dependent)")
    print("=" * 70)

    accs = [m["accuracy"] for m in all_metrics.values()]
    f1s = [m["f1_macro"] for m in all_metrics.values()]
    kappas = [m["cohen_kappa"] for m in all_metrics.values()]

    print(f"Subjects evaluated: {len(all_metrics)}")
    print(f"Accuracy:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1 (macro):   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Cohen's κ:    {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")

    return all_metrics


def run_cross_subject(config, device, subjects=None):
    """Run cross-subject evaluation (K-fold over subjects)."""
    dc = config["data"]

    print("=" * 70)
    print("CROSS-SUBJECT EVALUATION")
    print("=" * 70)

    print("\nDownloading/loading PhysioNet data...")
    download_physionet_data(dc["data_dir"], subjects)
    subject_data = load_all_subjects(
        dc["data_dir"], subjects,
        tmin=dc["tmin"], tmax=dc["tmax"],
        bandpass=tuple(dc["bandpass"]),
        resample_freq=dc.get("resample_freq", 128.0),
    )

    folds = create_cross_subject_splits(
        subject_data,
        n_folds=dc.get("n_folds", 5),
        seed=config["training"]["seed"],
    )

    augmenter = None
    ac = config.get("augmentation", {})
    if ac.get("enabled", False):
        augmenter = EEGAugmenter(
            noise_std=ac.get("noise_std", 0.15),
            noise_prob=ac.get("noise_prob", 0.6),
            time_shift_max=ac.get("time_shift_max", 10),
            time_shift_prob=ac.get("time_shift_prob", 0.4),
            channel_dropout_prob=ac.get("channel_dropout_prob", 0.08),
            scale_range=tuple(ac.get("scale_range", [0.8, 1.2])),
        )

    fold_metrics = []
    for fold_idx, fold in enumerate(folds):
        train_X, train_y = fold["train"]
        test_X, test_y = fold["test"]
        test_subjs = fold["test_subjects"]

        print(f"\n--- Fold {fold_idx+1}/{len(folds)} "
              f"(test subjects: {test_subjs[:5]}{'...' if len(test_subjs) > 5 else ''}) ---")
        print(f"    Train: {len(train_X)} epochs | Test: {len(test_X)} epochs")

        metrics, _ = train_and_evaluate(
            train_X, train_y, test_X, test_y,
            config, device,
            run_name=f"Fold{fold_idx+1}",
            augmenter=augmenter,
        )
        fold_metrics.append(metrics)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Cross-Subject)")
    print("=" * 70)

    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1_macro"] for m in fold_metrics]
    kappas = [m["cohen_kappa"] for m in fold_metrics]

    print(f"Folds: {len(fold_metrics)}")
    print(f"Accuracy:     {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1 (macro):   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"Cohen's κ:    {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Train EEG-Transformer")
    parser.add_argument("--config", type=str, default="configs/eeg_subject_dependent.yaml")
    parser.add_argument("--mode", type=str, default="subject_dependent",
                        choices=["subject_dependent", "cross_subject"])
    parser.add_argument("--model", type=str, default=None,
                        help="Override model: 'eeg_transformer' or 'eegnet'")
    parser.add_argument("--subjects", nargs="+", type=int, default=None,
                        help="Specific subject IDs to use (for quick tests)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="neuroformer",
                        help="W&B project name (default: 'neuroformer')")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override model if specified
    if args.model:
        config["model"]["name"] = args.model

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set seed
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize W&B if enabled
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("Warning: --wandb flag set but wandb is not installed. Skipping.")
        else:
            _wandb.init(
                project=args.wandb_project,
                config=config,
                name=f"{config['model']['name']}_{args.mode}",
                tags=[args.mode, config["model"]["name"]],
            )
            print(f"W&B run initialized: project={args.wandb_project}")

    # Run
    start_time = time.time()

    if args.mode == "subject_dependent":
        results = run_subject_dependent(config, device, args.subjects)
    else:
        results = run_cross_subject(config, device, args.subjects)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Save results
    results_dir = Path("assets/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model"]["name"]
    results_file = results_dir / f"{model_name}_{args.mode}_results.json"

    # Convert to serializable format
    serializable = {}
    if isinstance(results, dict):
        for k, v in results.items():
            serializable[str(k)] = {
                "accuracy": v["accuracy"],
                "f1_macro": v["f1_macro"],
                "cohen_kappa": v["cohen_kappa"],
            }
    else:
        for i, v in enumerate(results):
            serializable[f"fold_{i}"] = {
                "accuracy": v["accuracy"],
                "f1_macro": v["f1_macro"],
                "cohen_kappa": v["cohen_kappa"],
            }

    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {results_file}")

    # Log final results to W&B and finish
    if args.wandb and _WANDB_AVAILABLE and _wandb.run is not None:
        # Compute aggregate metrics
        if isinstance(results, dict):
            accs = [v["accuracy"] for v in results.values()]
            f1s = [v["f1_macro"] for v in results.values()]
            kappas = [v["cohen_kappa"] for v in results.values()]
        else:
            accs = [v["accuracy"] for v in results]
            f1s = [v["f1_macro"] for v in results]
            kappas = [v["cohen_kappa"] for v in results]

        _wandb.log({
            "final/mean_accuracy": float(np.mean(accs)),
            "final/std_accuracy": float(np.std(accs)),
            "final/mean_f1_macro": float(np.mean(f1s)),
            "final/std_f1_macro": float(np.std(f1s)),
            "final/mean_cohen_kappa": float(np.mean(kappas)),
            "final/std_cohen_kappa": float(np.std(kappas)),
            "final/total_time_minutes": elapsed / 60.0,
        })

        # Save results file as W&B artifact
        artifact = _wandb.Artifact(name="results", type="results")
        artifact.add_file(str(results_file))
        _wandb.log_artifact(artifact)

        _wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()

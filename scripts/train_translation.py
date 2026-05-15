#!/usr/bin/env python3
"""
Transformer Validation — Copy Task

This script validates that the Transformer implementation is correct by training
it on a simple "copy task": the model must learn to copy the input sequence
to the output. If the architecture is implemented correctly, the model should
achieve near-perfect accuracy on this trivial task.

This is the standard validation approach used by:
  - Harvard's "Annotated Transformer"
  - Most Transformer from-scratch implementations

Why this works:
  The copy task is the simplest possible sequence-to-sequence problem.
  If the model can't learn to copy, something is fundamentally wrong with
  the attention mechanism, masking, or training loop. If it can copy
  perfectly, we can be confident the architecture is correctly assembled.

Usage:
    python scripts/train_translation.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.transformer.transformer import Transformer
from src.training.scheduler import NoamScheduler
from src.training.losses import LabelSmoothingLoss
from src.training.trainer import greedy_decode


def generate_copy_data(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    pad_idx: int = 0,
    sos_idx: int = 1,
    eos_idx: int = 2,
) -> list:
    """
    Generate data for the copy task.

    Each sample consists of:
      - Source: [SOS, random_tokens..., EOS, PAD, PAD, ...]
      - Target: [SOS, random_tokens..., EOS, PAD, PAD, ...]

    The target is identical to the source — the model must learn to copy.

    Args:
        num_samples: Number of samples to generate
        seq_len: Length of the random token sequence (excluding SOS/EOS)
        vocab_size: Total vocabulary size
        pad_idx: Padding token index
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index

    Returns:
        List of (src, tgt) tensor pairs
    """
    data = []
    for _ in range(num_samples):
        # Random tokens (avoiding special tokens 0, 1, 2)
        tokens = torch.randint(3, vocab_size, (seq_len,))

        # Build sequence: [SOS, tokens..., EOS]
        seq = torch.cat([
            torch.tensor([sos_idx]),
            tokens,
            torch.tensor([eos_idx]),
        ])

        # Source and target are the same for copy task
        data.append((seq.clone(), seq.clone()))

    return data


def create_batches(data, batch_size):
    """Create batches from data."""
    batches = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        src_batch = torch.stack([d[0] for d in batch_data])
        tgt_batch = torch.stack([d[1] for d in batch_data])
        batches.append((src_batch, tgt_batch))
    return batches


def main():
    # ============================================================
    # Hyperparameters
    # ============================================================
    # Using a small model for the copy task (no need for the full
    # base model with d_model=512)
    VOCAB_SIZE = 20       # Small vocabulary
    D_MODEL = 64          # Small model dimension
    NUM_HEADS = 4         # 4 heads × d_k=16 = 64
    D_FF = 128            # Small FFN
    NUM_LAYERS = 2        # Just 2 layers
    DROPOUT = 0.1
    SEQ_LEN = 10          # Copy sequences of length 10
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    NUM_TRAIN = 3000      # Training samples
    NUM_VAL = 500         # Validation samples
    WARMUP_STEPS = 400
    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ============================================================
    # Create Model
    # ============================================================
    print("\n" + "=" * 60)
    print("TRANSFORMER COPY TASK VALIDATION")
    print("=" * 60)
    print(f"\nModel config: d_model={D_MODEL}, heads={NUM_HEADS}, "
          f"layers={NUM_LAYERS}, d_ff={D_FF}")

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pad_idx=PAD_IDX,
    ).to(device)

    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")

    # ============================================================
    # Optimizer & Scheduler (Paper Section 5.2-5.3)
    # ============================================================
    # Adam with paper's β values
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0,  # LR controlled by scheduler
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    # Noam scheduler
    scheduler = NoamScheduler(optimizer, d_model=D_MODEL, warmup_steps=WARMUP_STEPS)

    # Label smoothing loss (Paper Section 5.4)
    criterion = LabelSmoothingLoss(
        smoothing=0.1,
        pad_idx=PAD_IDX,
        vocab_size=VOCAB_SIZE,
    )

    # ============================================================
    # Generate Data
    # ============================================================
    print(f"\nGenerating copy task data...")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Train samples: {NUM_TRAIN}")
    print(f"  Val samples: {NUM_VAL}")

    train_data = generate_copy_data(NUM_TRAIN, SEQ_LEN, VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX)
    val_data = generate_copy_data(NUM_VAL, SEQ_LEN, VOCAB_SIZE, PAD_IDX, SOS_IDX, EOS_IDX)

    train_batches = create_batches(train_data, BATCH_SIZE)
    val_batches = create_batches(val_data, BATCH_SIZE)

    # ============================================================
    # Training Loop
    # ============================================================
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    print("-" * 60)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for src, tgt in train_batches:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_input)
            loss = criterion(logits, tgt_output)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_batches)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for src, tgt in val_batches:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                logits = model(src, tgt_input)
                loss = criterion(logits, tgt_output)
                val_loss += loss.item()

                # Token-level accuracy (ignoring padding)
                preds = logits.argmax(dim=-1)
                mask = tgt_output != PAD_IDX
                correct += (preds == tgt_output)[mask].sum().item()
                total += mask.sum().item()

        val_loss /= len(val_batches)
        accuracy = correct / total if total > 0 else 0

        lr = scheduler.get_lr()
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.4f} | "
            f"LR: {lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # ============================================================
    # Test: Greedy Decoding
    # ============================================================
    print("\n" + "=" * 60)
    print("GREEDY DECODING TEST")
    print("=" * 60)

    model.eval()
    num_perfect = 0
    num_tests = 10

    for i in range(num_tests):
        # Generate a test sequence
        test_tokens = torch.randint(3, VOCAB_SIZE, (SEQ_LEN,))
        src_seq = torch.cat([
            torch.tensor([SOS_IDX]),
            test_tokens,
            torch.tensor([EOS_IDX]),
        ]).unsqueeze(0)  # Add batch dimension

        # Greedy decode
        decoded = greedy_decode(
            model, src_seq, max_len=SEQ_LEN + 3,
            start_token=SOS_IDX, end_token=EOS_IDX, device=device,
        )

        src_tokens = src_seq[0].tolist()
        dec_tokens = decoded[0].tolist()

        # Check if the copy is perfect
        is_perfect = src_tokens == dec_tokens[:len(src_tokens)]

        if is_perfect:
            num_perfect += 1

        status = "✓" if is_perfect else "✗"
        print(f"  {status} Source:  {src_tokens}")
        print(f"    Decoded: {dec_tokens}")
        print()

    print(f"Perfect copies: {num_perfect}/{num_tests}")

    # ============================================================
    # Final Verdict
    # ============================================================
    print("\n" + "=" * 60)
    if accuracy > 0.95 and num_perfect >= 8:
        print("✅ TRANSFORMER ARCHITECTURE VERIFIED — Copy task passed!")
        print("   The model can copy sequences with high accuracy.")
        print("   All components (attention, masking, encoding, decoding)")
        print("   are working correctly.")
    elif accuracy > 0.80:
        print("⚠️  PARTIAL SUCCESS — Model is learning but not perfect.")
        print("   Try training for more epochs or adjusting hyperparameters.")
    else:
        print("❌ VERIFICATION FAILED — Something may be wrong.")
        print("   Check masking, attention, or the training loop.")
    print("=" * 60)


if __name__ == "__main__":
    main()

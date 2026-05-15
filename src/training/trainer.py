"""
Training Loop for the Transformer.

Provides a clean, reusable training loop with:
  - Gradient accumulation
  - Learning rate scheduling (Noam scheduler)
  - Training/validation split
  - Logging and progress tracking
  - Greedy decoding for inference

Based on the training procedure described in Section 5 of the paper.
"""

import time
from typing import Optional, Callable

import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.scheduler import NoamScheduler
from src.training.metrics import compute_sequence_accuracy


class Trainer:
    """
    Transformer Trainer.

    Handles the training loop, validation, and basic logging.

    Paper training details (Section 5.2):
        - Optimizer: Adam with β₁=0.9, β₂=0.98, ε=10⁻⁹
        - LR Schedule: Noam with warmup_steps=4000
        - Regularization: Dropout P=0.1, Label smoothing ε=0.1

    Args:
        model: The Transformer model
        criterion: Loss function (typically LabelSmoothingLoss)
        optimizer: Optimizer (typically Adam with paper's β values)
        scheduler: LR scheduler (typically NoamScheduler)
        device: Device to train on (cpu/cuda)
        grad_accum_steps: Number of steps to accumulate gradients
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[NoamScheduler] = None,
        device: torch.device = None,
        grad_accum_steps: int = 1,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cpu")
        self.grad_accum_steps = grad_accum_steps

        self.model.to(self.device)

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self,
        data_iter,
        epoch: int = 0,
        log_interval: int = 50,
    ) -> float:
        """
        Train for one epoch.

        Args:
            data_iter: Iterator yielding (src, tgt) batches
            epoch: Current epoch number (for logging)
            log_interval: Print loss every N batches

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        start_time = time.time()

        self.optimizer.zero_grad()

        for i, (src, tgt) in enumerate(tqdm(data_iter, desc=f"Epoch {epoch}")):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Target input (everything except last token) and
            # target output (everything except first token — shifted right)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Forward pass
            logits = self.model(src, tgt_input)

            # Compute loss
            loss = self.criterion(logits, tgt_output)
            loss = loss / self.grad_accum_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (i + 1) % self.grad_accum_steps == 0:
                # Gradient clipping (helps with training stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item() * self.grad_accum_steps
            total_tokens += 1

            # Log
            if (i + 1) % log_interval == 0:
                avg_loss = total_loss / total_tokens
                elapsed = time.time() - start_time
                lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.param_groups[0]["lr"]
                print(
                    f"  Step {i+1} | Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.6f} | Time: {elapsed:.1f}s"
                )

        avg_loss = total_loss / max(total_tokens, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_iter) -> float:
        """
        Evaluate on validation data.

        Args:
            data_iter: Iterator yielding (src, tgt) batches

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for src, tgt in data_iter:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = self.model(src, tgt_input)
            loss = self.criterion(logits, tgt_output)

            total_loss += loss.item()
            total_tokens += 1

        avg_loss = total_loss / max(total_tokens, 1)
        self.val_losses.append(avg_loss)
        return avg_loss

    def fit(
        self,
        train_iter_fn: Callable,
        val_iter_fn: Callable = None,
        num_epochs: int = 10,
        log_interval: int = 50,
    ):
        """
        Full training loop.

        Args:
            train_iter_fn: Callable that returns a fresh training iterator
            val_iter_fn: Callable that returns a fresh validation iterator (optional)
            num_epochs: Number of training epochs
            log_interval: Print frequency
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("-" * 60)

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(
                train_iter_fn(), epoch=epoch, log_interval=log_interval
            )

            msg = f"Epoch {epoch} | Train Loss: {train_loss:.4f}"

            if val_iter_fn is not None:
                val_loss = self.evaluate(val_iter_fn())
                msg += f" | Val Loss: {val_loss:.4f}"

            print(msg)
            print("-" * 60)


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    max_len: int,
    start_token: int,
    end_token: int = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Greedy decoding (auto-regressive inference).

    At each step, the model generates the most probable next token
    and appends it to the output sequence. This continues until
    max_len is reached or the end token is generated.

    This is the simplest decoding strategy. The paper also uses
    beam search for better results, but greedy decode is sufficient
    for validation.

    Args:
        model: Trained Transformer model
        src: Source sequence (1, src_len) — single example
        max_len: Maximum output length
        start_token: Start-of-sequence token index
        end_token: End-of-sequence token index (stops generation)
        device: Device

    Returns:
        Generated token indices (1, generated_len)
    """
    model.eval()
    device = device or next(model.parameters()).device
    src = src.to(device)

    # Encode the source once
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    memory = model.encode(src, src_mask)

    # Start with the start token
    ys = torch.tensor([[start_token]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        # Create target mask (causal)
        tgt_len = ys.size(1)
        tgt_mask = torch.tril(
            torch.ones(tgt_len, tgt_len, device=device)
        ).unsqueeze(0).unsqueeze(0)

        # Decode
        out = model.decode(ys, memory, src_mask, tgt_mask)

        # Get the last token's logits and pick the most probable
        logits = model.output_projection(out[:, -1, :])
        next_token = logits.argmax(dim=-1, keepdim=True)

        # Append to output
        ys = torch.cat([ys, next_token], dim=1)

        # Stop if end token is generated
        if end_token is not None and next_token.item() == end_token:
            break

    return ys

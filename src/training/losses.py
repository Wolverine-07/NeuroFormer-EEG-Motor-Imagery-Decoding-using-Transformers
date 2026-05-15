"""
Loss Functions — "Attention Is All You Need" Section 5.4

Label Smoothing:
    From the paper:
        "During training, we employed label smoothing of value ε_ls = 0.1.
         This hurt perplexity, as the model learns to be more unsure, but
         improved accuracy and BLEU score."

    Label smoothing replaces the hard target distribution (one-hot) with a
    smoothed distribution that puts (1 - ε) probability on the correct class
    and distributes ε uniformly over all other classes.

    Without smoothing: target = [0, 0, 1, 0, 0]  (one-hot)
    With smoothing:    target = [0.025, 0.025, 0.9, 0.025, 0.025]  (smoothed, ε=0.1)

    This acts as a regularizer, preventing the model from becoming too
    confident in its predictions and improving generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss (Paper Section 5.4).

    Implements cross-entropy loss with label smoothing using KL divergence.

    The smoothed target distribution is:
        q(k) = (1 - ε) · δ(k, y) + ε / V

    where:
        - ε = smoothing value (paper uses 0.1)
        - y = true label
        - V = vocabulary size
        - δ(k, y) = 1 if k == y, else 0

    Args:
        smoothing: Label smoothing value ε (paper default: 0.1)
        pad_idx: Index of padding token (excluded from loss computation)
        vocab_size: Size of the vocabulary
    """

    def __init__(self, smoothing: float = 0.1, pad_idx: int = 0, vocab_size: int = 1):
        super().__init__()
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Model output logits (batch * seq_len, vocab_size) or (batch, seq_len, vocab_size)
            target: True token indices (batch * seq_len,) or (batch, seq_len)

        Returns:
            Scalar loss value
        """
        # Reshape if needed: (batch, seq_len, vocab) → (batch * seq_len, vocab)
        if logits.dim() == 3:
            logits = logits.contiguous().view(-1, logits.size(-1))
        if target.dim() == 2:
            target = target.contiguous().view(-1)

        vocab_size = logits.size(-1)

        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Create the smoothed target distribution
        # Start with uniform distribution: ε / V for all classes
        smooth_targets = torch.full_like(log_probs, self.smoothing / (vocab_size - 2))

        # Put (1 - ε) + ε/V on the correct class
        smooth_targets.scatter_(1, target.unsqueeze(1), self.confidence)

        # Zero out the padding index (don't distribute probability there)
        smooth_targets[:, self.pad_idx] = 0

        # Create mask for padding positions in the target
        # (positions where target is <pad> should not contribute to loss)
        pad_mask = target == self.pad_idx
        smooth_targets[pad_mask] = 0

        # KL divergence loss: sum(-q * log_p)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Average over non-padding tokens
        non_pad_count = (~pad_mask).sum()
        if non_pad_count > 0:
            loss = loss.sum() / non_pad_count
        else:
            loss = loss.sum()

        return loss

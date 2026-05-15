"""
Learning Rate Scheduler — "Attention Is All You Need" Section 5.3

From the paper:
    "We used the Adam optimizer with β₁ = 0.9, β₂ = 0.98 and ε = 10⁻⁹.
     We varied the learning rate over the course of training, according to
     the formula:

        lr = d_model^(-0.5) · min(step_num^(-0.5), step_num · warmup_steps^(-1.5))

     This corresponds to increasing the learning rate linearly for the first
     warmup_steps training steps, and decreasing it thereafter proportionally
     to the inverse square root of the step number. We used warmup_steps = 4000."

This is commonly known as the "Noam" scheduler (named after one of the authors).
"""


class NoamScheduler:
    """
    Noam Learning Rate Scheduler (Paper Section 5.3).

    Implements the learning rate schedule from the paper:
        lr = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))

    The schedule has two phases:
    1. Warm-up (steps 1 to warmup_steps): LR increases linearly
    2. Decay (steps > warmup_steps): LR decreases proportionally to 1/sqrt(step)

    Peak learning rate occurs at step = warmup_steps:
        lr_peak = d_model^(-0.5) · warmup_steps^(-0.5)

    For the paper's settings (d_model=512, warmup_steps=4000):
        lr_peak ≈ 0.00070

    Args:
        optimizer: The optimizer to schedule
        d_model: Model dimension (determines peak LR)
        warmup_steps: Number of warmup steps (paper default: 4000)
        factor: Multiplicative factor for the learning rate (default: 1.0)
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000, factor: float = 1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def step(self):
        """Update the learning rate for the current step."""
        self._step += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _compute_lr(self) -> float:
        """
        Compute learning rate using the Noam formula.

        lr = factor · d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
        """
        step = max(self._step, 1)  # Avoid division by zero
        return self.factor * (
            self.d_model ** (-0.5)
            * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )

    def get_lr(self) -> float:
        """Get the current learning rate."""
        return self._compute_lr()

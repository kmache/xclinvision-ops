"""Standalone loss functions used by the training stack.

These classes only depend on ``torch`` and ``torch.nn``. They live in their
own module so callers (notably tests and lightweight utilities) can import
them without pulling in ``pytorch_lightning`` and the rest of the trainer
module's heavy dependency tree.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in medical imaging (Multiclass)."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        probs = F.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt) ** self.gamma
        return (self.alpha * focal_term * ce_loss).mean()


class MultilabelFocalLoss(nn.Module):
    """Focal Loss for multi-label classification using sigmoid + BCE.

    IMPORTANT: pos_weight is applied as a multiplicative alpha factor
    OUTSIDE the BCE, not inside it.  Passing pos_weight inside
    ``binary_cross_entropy_with_logits`` causes the focal term
    ``(1-pt)^gamma`` to amplify the already-weighted BCE
    exponentially, creating a ~1000:1 gradient imbalance between
    positive and negative samples when the classifier is initialised
    with a negative bias (sigmoid ≈ 0.12).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 1.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none",
        )
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)

        if self.pos_weight is not None:
            alpha_t = targets * self.pos_weight + (1 - targets) * 1.0
        else:
            alpha_t = self.alpha

        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification.

    Ridnik et al., 2021 — "Asymmetric Loss For Multi-Label Classification".

    Key idea: use different focusing parameters for positive (gamma_pos)
    and negative (gamma_neg) samples.  Hard-threshold probability shifting
    on negatives further suppresses easy-negative gradients.

    This implementation is numerically stable under AMP fp16 by:
      - Using F.logsigmoid (log-sum-exp trick) instead of log(sigmoid(x))
      - Casting to fp32 for the loss body to avoid fp16 underflow
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        orig_dtype = inputs.dtype
        inputs = inputs.float()
        targets = targets.float()

        log_pos = F.logsigmoid(inputs)
        log_neg = F.logsigmoid(-inputs)

        probs = torch.sigmoid(inputs)

        if self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1.0)
            log_neg = torch.log1p(-probs_neg + 1e-8)
        else:
            probs_neg = probs

        loss_pos = -targets * log_pos
        loss_neg = -(1.0 - targets) * log_neg

        if self.gamma_pos > 0:
            loss_pos = loss_pos * ((1.0 - probs) ** self.gamma_pos)
        if self.gamma_neg > 0:
            loss_neg = loss_neg * (probs_neg ** self.gamma_neg)

        loss = loss_pos + loss_neg

        if self.pos_weight is not None:
            pw = targets * self.pos_weight + (1.0 - targets)
            loss = loss * pw

        return loss.mean().to(orig_dtype)


__all__ = ["FocalLoss", "MultilabelFocalLoss", "AsymmetricLoss"]

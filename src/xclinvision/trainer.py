"""Training module with PyTorch Lightning integration.

Defines the XClinVisionModel LightningModule and associated training logic,
including progressive unfreezing, focal loss, and comprehensive metrics tracking.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
from torchmetrics import MetricCollection, Accuracy, F1Score, AUROC
from sklearn.metrics import classification_report, confusion_matrix

from .modeling import freeze_backbone, unfreeze_layers, get_param_counts

# Determine PL major version once at import time so _apply_unfreeze_schedule
# can branch without repeated string parsing.
_PL_MAJOR = int(pl.__version__.split(".")[0])


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in medical imaging."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Register as buffer so it moves to the correct device automatically
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            n_classes = inputs.size(-1)
            targets_oh = F.one_hot(targets, n_classes).float()
            targets_oh = targets_oh * (1 - self.label_smoothing) + (self.label_smoothing / n_classes)
            # F.cross_entropy ignores the `weight` tensor when targets are float
            # (soft labels). Apply class weights manually via the hard label index.
            ce_loss = F.cross_entropy(inputs, targets_oh, reduction="none")
            if self.weight is not None:
                sample_weights = self.weight[targets]  # (B,)
                ce_loss = ce_loss * sample_weights
        else:
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")

        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class XClinVisionModel(pl.LightningModule):
    """
    PyTorch Lightning module for XClinVision training.

    Features:
    - Progressive unfreezing for transfer learning
    - Focal loss with class weighting
    - TorchMetrics for distributed-safe metrics tracking
    - Discriminative learning rates (backbone vs head)
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        loss_type: str = "focal",
        label_smoothing: float = 0.1,
        class_weights: Optional[List[float]] = None,
        progressive_unfreezing: bool = True,
        unfreeze_schedule: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.progressive_unfreezing = progressive_unfreezing
        self.unfreeze_schedule = unfreeze_schedule or [0, 5, 10, 20]
        self.current_phase = 0
        self.label_smoothing = label_smoothing
        self._oom_steps = 0  # L-2: track OOM-skipped batches per epoch

        # Loss function
        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        )
        self.criterion = self._setup_loss(loss_type, weight_tensor, label_smoothing)

        # TorchMetrics — distributed-safe, auto-synced across devices
        metrics = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "f1_macro": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            "auc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Log model info (note: unfreezing has not run yet at init)
        counts = get_param_counts(model)
        logger.info(
            f"Model loaded: {counts['total_m']:.2f}M params "
            f"(trainable% will update after first unfreeze phase)"
        )

    # ---- loss setup -------------------------------------------------------

    @staticmethod
    def _setup_loss(
        loss_type: str,
        weight: Optional[torch.Tensor],
        label_smoothing: float = 0.1,
    ) -> nn.Module:
        """Setup loss function with optional class weighting."""
        if loss_type == "focal":
            return FocalLoss(weight=weight, label_smoothing=label_smoothing)
        elif loss_type == "ce":
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        raise ValueError(f"Unknown loss type: {loss_type}")

    # ---- progressive unfreezing -------------------------------------------

    def _apply_unfreeze_schedule(self):
        """Apply progressive unfreezing based on current epoch."""
        if not self.progressive_unfreezing:
            return

        target_phase = sum(1 for e in self.unfreeze_schedule if self.current_epoch >= e)

        if target_phase != self.current_phase:
            self.current_phase = target_phase
            # Phase 1: head only  |  Phase 2: last 2 blocks  |
            # Phase 3: last 4 blocks  |  Phase 4+: full fine-tuning
            actions = {
                1: (lambda: freeze_backbone(self.model, unfreeze_head=True), "Training head only"),
                2: (lambda: unfreeze_layers(self.model, num_layers=2), "Unfrozen last 2 blocks"),
                3: (lambda: unfreeze_layers(self.model, num_layers=4), "Unfrozen last 4 blocks"),
            }
            if target_phase in actions:
                fn, msg = actions[target_phase]
                fn()
            else:
                unfreeze_layers(self.model, num_layers=0)
                msg = "Full fine-tuning"
                
            self.print(f"Epoch {self.current_epoch}: Phase {target_phase} - {msg}")

            # Rebuild / extend the optimizer so newly unfrozen parameters
            # are included in gradient updates.
            if self.trainer is not None:
                if _PL_MAJOR >= 2:
                    # PL 2.x: add each new param individually with the same
                    # backbone/head discriminative LR used in configure_optimizers.
                    opt = self.optimizers()
                    if isinstance(opt, list):
                        opt = opt[0]
                    existing_ids = {id(p) for group in opt.param_groups for p in group["params"]}
                    named_params = list(self.model.named_parameters())
                    head_cutoff = int(len(named_params) * 0.9)
                    new_backbone, new_head = [], []
                    for i, (name, param) in enumerate(named_params):
                        if not param.requires_grad or id(param) in existing_ids:
                            continue
                        is_head = i >= head_cutoff or any(
                            k in name for k in ("head", "fc", "classifier")
                        )
                        (new_head if is_head else new_backbone).append(param)
                    if new_backbone:
                        opt.add_param_group({
                            "params": new_backbone,
                            "lr": self.learning_rate * 0.1,
                            "name": "progressive_backbone",
                        })
                    if new_head:
                        opt.add_param_group({
                            "params": new_head,
                            "lr": self.learning_rate,
                            "name": "progressive_head",
                        })
                else:
                    # PL 1.x: fully rebuild the optimizer from scratch via strategy.
                    try:
                        self.trainer.strategy.setup_optimizers(self.trainer)
                    except Exception as exc:
                        logger.warning(
                            f"Could not rebuild optimizer after unfreeze (PL 1.x): {exc}"
                        )

    # ---- forward / steps --------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self):
        """Apply progressive unfreezing at epoch start."""
        self._apply_unfreeze_schedule()

    def training_step(self, batch, _batch_idx):
        x, y = batch
        try:
            logits = self(x)
            loss = self.criterion(logits, y)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._oom_steps += 1
            logger.warning(
                f"[OOM] training_step skipped (batch_size={x.shape[0]}, "
                f"img_size={x.shape[-1]}). Consider reducing --batch-size."
            )
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics.update(logits, y)
        return loss

    def on_train_epoch_end(self):
        try:
            self.log_dict(self.train_metrics.compute(), prog_bar=True)
        except ValueError:
            logger.warning("[OOM] Entire train epoch was skipped — no samples to compute metrics.")
        finally:
            if self._oom_steps > 0:
                self.log("train_oom_steps", float(self._oom_steps), prog_bar=False)
                logger.warning(
                    f"Epoch {self.current_epoch}: {self._oom_steps} training "
                    f"batch(es) were skipped due to OOM — metrics are computed "
                    f"over the remaining batches only."
                )
                self._oom_steps = 0
            self.train_metrics.reset()

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        try:
            logits = self(x)
            loss = self.criterion(logits, y)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning(
                f"[OOM] validation_step skipped (batch_size={x.shape[0]}, "
                f"img_size={x.shape[-1]}). Consider reducing --batch-size."
            )
            return None
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_metrics.update(logits, y)

        # Return dict so callbacks (e.g. MetricsCallback) can collect outputs
        # Detach val_loss so PL doesn't retain the full computation graph in memory
        return {"val_loss": loss.detach(), "preds": preds, "targets": y, "probs": probs}

    def on_validation_epoch_end(self):
        try:
            self.log_dict(self.val_metrics.compute(), prog_bar=True)
        except ValueError:
            logger.warning("[OOM] Entire validation epoch was skipped — no samples to compute metrics.")
        finally:
            self.val_metrics.reset()

    def test_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.test_metrics.update(logits, y)
        return {"test_loss": loss.detach(), "preds": preds, "targets": y, "probs": probs}

    def on_test_epoch_end(self):
        try:
            self.log_dict(self.test_metrics.compute(), prog_bar=True)
        except ValueError:
            logger.warning("[OOM] Entire test epoch was skipped — no samples to compute metrics.")
        finally:
            self.test_metrics.reset()

    # ---- optimizer --------------------------------------------------------

    def configure_optimizers(self):
        """Configure optimizer with discriminative learning rates."""
        backbone_params, head_params = [], []
        named_params = list(self.model.named_parameters())
        head_cutoff = int(len(named_params) * 0.9)

        for i, (name, param) in enumerate(named_params):
            # Only register params that currently require gradients to avoid
            # PyTorch UserWarning about requires_grad=False params in optimizer groups.
            # Progressive unfreezing calls configure_optimizers again via
            # trainer.strategy.setup_optimizers() so newly unfrozen params are picked up.
            if not param.requires_grad:
                continue
            if i >= head_cutoff or any(k in name for k in ("head", "fc", "classifier")):
                head_params.append(param)
            else:
                backbone_params.append(param)

        # Guard: AdamW raises ValueError on an empty param list.
        # Build groups dynamically so neither an empty backbone nor empty head crashes.
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.learning_rate * 0.1, "name": "backbone"})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.learning_rate, "name": "head"})

        # Fallback: if nothing is trainable yet (e.g. before first unfreeze), include all params
        if not param_groups:
            param_groups = [{"params": list(self.model.parameters()), "lr": self.learning_rate}]

        optimizer = AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=self.learning_rate * 1e-3
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class MetricsCallback(Callback):
    """Callback for detailed classification metrics and optional prediction saving."""

    def __init__(self, save_predictions: bool = False, output_dir: Optional[str] = None):
        super().__init__()
        self.save_predictions = save_predictions
        self.output_dir = output_dir
        self._reset()

    def _reset(self):
        self.val_preds: List[int] = []
        self.val_targets: List[int] = []
        self.val_probs: List[np.ndarray] = []  # each row is shape (num_classes,)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, **kwargs
    ):
        """Collect validation outputs. **kwargs handles PL 2.x dataloader_idx injections."""
        if outputs is None or not isinstance(outputs, dict):
            return
        
        # Detach and move to CPU immediately to prevent memory leaks
        self.val_preds.extend(outputs["preds"].detach().cpu().numpy())
        self.val_targets.extend(outputs["targets"].detach().cpu().numpy())
        self.val_probs.extend(outputs["probs"].detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log detailed classification metrics."""
        if len(self.val_preds) == 0:
            return

        preds = np.array(self.val_preds)
        targets = np.array(self.val_targets)

        # Classification report — pass labels explicitly so it always covers
        # all 3 classes even when a small batch (e.g. sanity check) is missing one
        target_names = ["Normal", "Pneumonia", "Tuberculosis"]
        report = classification_report(
            targets,
            preds,
            labels=list(range(len(target_names))),
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        # Log per-class metrics
        for cls_name, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        pl_module.log(f"val_{cls_name}_{metric_name}", float(value))

        # Confusion matrix — pass labels so the matrix is always (num_classes x
        # num_classes) even when a single class dominates a small batch.
        cm = confusion_matrix(targets, preds, labels=list(range(len(target_names))))
        logger.info(f"\nConfusion Matrix (epoch {trainer.current_epoch}):\n{cm}")

        # Save predictions if requested
        if self.save_predictions and self.output_dir:
            self._save_predictions(trainer.current_epoch)

        self._reset()

    def _save_predictions(self, epoch: int):
        """Persist predictions to disk."""
        import json
        import os
        from datetime import datetime

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_data = {
            "epoch": epoch,
            "timestamp": timestamp,
            "predictions": [int(p) for p in self.val_preds],
            "targets": [int(t) for t in self.val_targets],
            "probabilities": [p.tolist() for p in self.val_probs],
        }

        filepath = os.path.join(self.output_dir, f"val_predictions_{timestamp}.json")
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)


class XAIValidationCallback(Callback):
    """Periodic XAI validation callback — runs explainability analysis every N epochs."""

    def __init__(
        self,
        every_n_epochs: int = 5,
        max_samples: int = 100,
        output_dir: str = "outputs/xai",
        architecture: str = "unknown",
        image_size: int = 384,
    ):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.max_samples = max_samples
        self.output_dir = output_dir
        self.architecture = architecture
        self.image_size = image_size

    def on_validation_epoch_end(self, trainer, pl_module):
        """Run XAI validation periodically."""
        epoch = trainer.current_epoch

        if (epoch + 1) % self.every_n_epochs != 0:
            return

        try:
            from .xai import ValidationXAI

            output_path = f"{self.output_dir}/epoch_{epoch:03d}"
            xai = ValidationXAI(pl_module.model, self.architecture, output_path, img_size=self.image_size)

            # val_dataloaders is a list in PL ≥ 1.6; use the first dataloader.
            # In PL 2.x the attribute may be None if the datamodule hasn't been
            # set up yet — fall back to the datamodule if available.
            val_dataloaders = getattr(trainer, "val_dataloaders", None)
            if val_dataloaders is None and trainer.datamodule is not None:
                val_dataloaders = trainer.datamodule.val_dataloader()
            if val_dataloaders is None:
                logger.warning("XAIValidationCallback: could not resolve val_dataloaders — skipping.")
                return
            val_loader = (
                val_dataloaders[0]
                if isinstance(val_dataloaders, (list, tuple))
                else val_dataloaders
            )
            logger.info(f"\nRunning XAI validation for epoch {epoch}...")
            metrics = xai.process_dataset(val_loader, max_samples=self.max_samples)

            pl_module.log("xai_plausibility", metrics["clinical_plausibility"]["score"])
            pl_module.log("xai_qc_pass_rate", 1 - metrics["qc_failure_rate"])

            if not metrics["clinical_plausibility"]["is_acceptable"]:
                logger.warning(
                    f"Epoch {epoch} - Low clinical plausibility! "
                    f"Model may be learning shortcuts."
                )
        except Exception as e:
            import traceback
            logger.error(f"XAI validation failed: {e}")
            traceback.print_exc()
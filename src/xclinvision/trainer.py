"""Training module with PyTorch Lightning integration.

Defines the XClinVisionModel LightningModule and associated training logic,
including progressive unfreezing, focal loss, and comprehensive metrics tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
from torchmetrics import MetricCollection, Accuracy, F1Score, AUROC
from sklearn.metrics import classification_report, confusion_matrix

from .config import get_class_map, get_class_names
from .modeling import freeze_backbone, unfreeze_layers, get_param_counts

# Determine PL major version once at import time so _apply_unfreeze_schedule
_PL_MAJOR = int(pl.__version__.split(".")[0])


# ---------------------------------------------------------------------------
# Mixup / CutMix helpers
# ---------------------------------------------------------------------------

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup (Zhang et al., 2018) to a batch.

    Returns mixed inputs, pairs of targets, and the lambda coefficient.
    """
    if alpha > 0:
        # Use torch.distributions so the random state lives on the torch RNG,
        # which is seeded per-rank in DDP — avoiding mismatched lam values.
        lam = float(
            torch.distributions.Beta(
                torch.tensor(alpha), torch.tensor(alpha)
            ).sample().item()
        )
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for Mixup-augmented batch."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in medical imaging."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # alpha=1.0 is a uniform global scaling factor — it does NOT provide
        # per-class reweighting (that role is fulfilled by the `weight` buffer).
        # Reduce alpha below 1.0 to globally discount the focal modulation term.
        self.alpha = alpha
        self.gamma = gamma
        # Register as buffer so it moves to the correct device automatically
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Compute pt as the true-class probability via softmax + gather.
        # This is the correct multi-class generalisation of focal loss:
        # exp(-CE_unweighted) equals softmax[target] only in the binary case.
        pt = (
            F.softmax(inputs, dim=-1)
            .gather(1, targets.unsqueeze(1))
            .squeeze(1)
            .detach()  # stop gradients through pt; only modulate the loss scale
        )

        # 2. Compute focal term
        focal_term = (1 - pt) ** self.gamma

        # 3. Compute base cross entropy (with optional label smoothing)
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
            # Apply standard weighted cross entropy
            ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
            
        # 4. Modulate and return
        return (self.alpha * focal_term * ce_loss).mean()


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
        mixup_alpha: float = 0.2,
        mixup_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.progressive_unfreezing = progressive_unfreezing
        self.unfreeze_schedule = unfreeze_schedule or [1, 5, 10, 20]
        self.current_phase = 0
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob  # fraction of batches where Mixup is applied
        self._oom_steps = 0  # L-2: track OOM-skipped batches per epoch

        # Loss function
        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        )
        self.criterion = self._setup_loss(loss_type, weight_tensor, label_smoothing)

        metrics = MetricCollection({
            "acc": Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
            "f1_macro": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
            "auc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
        })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Freeze the backbone right at init so configure_optimizers only sees the head.
        # Start at phase 1 so _apply_unfreeze_schedule doesn't redundantly re-freeze
        # the backbone on the very first epoch call (fix #13).
        if self.progressive_unfreezing:
            freeze_backbone(self.model, unfreeze_head=True)
            self.current_phase = 1  # backbone already frozen — skip phase-1 re-freeze

        # Log model info
        counts = get_param_counts(model)
        logger.info(
            f"Model loaded: {counts['total_m']:.2f}M params "
            f"({counts['trainable_m']:.2f}M trainable)"
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

        # Fix #13: only advance to higher phases \u2014 never go backwards.
        # Using > instead of != means that if the backbone was already frozen at
        # init (current_phase=1), the epoch-0 target_phase=0 does NOT trigger a
        # spurious full-unfreeze, and the epoch-1 target_phase=1 is also skipped
        # since the backbone is already in the correct state.
        if target_phase > self.current_phase:
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

            # Fix P0: Rebuild optimizer so newly-unfrozen params actually receive
            # gradient updates.  Without this, params unfrozen after init are
            # absent from all optimizer param-groups and never get updated.
            self._rebuild_optimizers()

    def _rebuild_optimizers(self):
        """Rebuild optimizer + LR scheduler after unfreezing new parameters.

        Uses PL's public ``lr_schedulers`` / ``optimizers`` replacement pattern
        rather than the private ``_configure_schedulers`` API so this remains
        stable across PL minor versions.
        """
        if self.trainer is None:
            return
        opt_config = self.configure_optimizers()
        optimizer = opt_config["optimizer"]
        scheduler = opt_config["lr_scheduler"]["scheduler"]
        interval = opt_config["lr_scheduler"].get("interval", "epoch")

        # Replace the optimizer list in-place (public attribute, documented).
        self.trainer.optimizers = [optimizer]

        # Wrap scheduler in a LRSchedulerConfig (PL ≥ 2.0) or a plain dict (PL 1.x).
        # The spurious _AcceleratorConnector import guard has been removed — it was
        # a no-op that obscured the intent and broke on some PL builds.
        try:
            from pytorch_lightning.utilities.types import LRSchedulerConfig
            self.trainer.lr_scheduler_configs = [
                LRSchedulerConfig(scheduler=scheduler, interval=interval)
            ]
        except (ImportError, AttributeError):
            # PL < 2.0 fallback: lr_schedulers is a plain list of dicts.
            self.trainer.lr_schedulers = [{"scheduler": scheduler, "interval": interval}]

        trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        self.print(f"  → Optimizer rebuilt ({trainable} trainable params)")

    # ---- forward / steps --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self):
        """Apply progressive unfreezing at epoch start."""
        self._apply_unfreeze_schedule()

    def training_step(self, batch, _batch_idx):
        x, y = batch
        try:
            # Apply Mixup probabilistically so hard examples still appear
            # unblended ~50 % of the time (helps boundary learning late in training).
            use_mixup = (
                self.mixup_alpha > 0
                and self.training
                and torch.rand(1).item() < self.mixup_prob
            )
            if use_mixup:
                mixed_x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
                logits = self(mixed_x)
                loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
            else:
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
        # Fix #7: skip metric updates during Mixup steps.  Blended logits vs.
        # hard labels produce a systematically low accuracy/F1 signal that
        # pollutes the training metrics dashboard.  Only update when the batch
        # is un-mixed so the metrics reflect true model capability.
        if not use_mixup:
            self.train_metrics.update(logits.detach(), y)
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
        """Configure optimizer with discriminative learning rates and warmup."""
        backbone_params, head_params = [], []

        for name, param in self.model.named_parameters():
            # Fix #7: skip frozen parameters entirely — AdamW still allocates
            # moment tensors for params with requires_grad=False, wasting GPU/CPU
            # memory proportional to frozen backbone size.
            if not param.requires_grad:
                continue
            if any(k in name for k in ("head", "fc", "classifier", "last_linear")):
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups = []
        if backbone_params:
            # Fix #14: 5× discriminative ratio instead of 10× so the backbone
            # retains enough learning signal when it unfreezes mid-training.
            param_groups.append({"params": backbone_params, "lr": self.learning_rate * 0.2, "name": "backbone"})
        if head_params:
            param_groups.append({"params": head_params, "lr": self.learning_rate, "name": "head"})

        # Fallback: if somehow empty, include all params
        if not param_groups:
            param_groups = [{"params": list(self.model.parameters()), "lr": self.learning_rate}]

        optimizer = AdamW(
            param_groups,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        # Build the LR schedule.  On the very first call (epoch 0) we start the
        # standard warmup → cosine sequence.  On subsequent calls from
        # _rebuild_optimizers (unfreeze boundaries) warmup is already complete,
        # so we build a plain CosineAnnealingLR for the *remaining* epochs only.
        # This prevents the LR from jumping back to the full rate at every
        # unfreeze phase (issue #18).
        first_unfreeze = self.unfreeze_schedule[0] if self.unfreeze_schedule else 5
        warmup_epochs = max(1, min(first_unfreeze - 1, 5))
        current_epoch = getattr(self, "current_epoch", 0)

        if current_epoch < warmup_epochs:
            # Warmup not yet finished — full SequentialLR
            warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(self.trainer.max_epochs - warmup_epochs, 1),
                eta_min=1e-6,
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            # Warmup already done — cosine only for remaining epochs so the
            # scheduler clock doesn't reset on every optimizer rebuild.
            remaining = max(self.trainer.max_epochs - current_epoch, 1)
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)

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
        if outputs is None or not isinstance(outputs, dict) or "preds" not in outputs:
            return
        
        # Detach and move to CPU immediately to prevent memory leaks
        self.val_preds.extend(outputs["preds"].detach().cpu().numpy())
        self.val_targets.extend(outputs["targets"].detach().cpu().numpy())
        self.val_probs.extend(outputs["probs"].detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log detailed classification metrics.

        _reset() is called in a finally block so stale predictions never
        accumulate into the next epoch if an exception occurs mid-reporting.
        """
        if len(self.val_preds) == 0:
            return

        preds = np.array(self.val_preds)
        targets = np.array(self.val_targets)

        try:
            # Always use all classes to prevent dimension mismatch during sanity checks
            target_names = getattr(pl_module, 'class_names', get_class_names())
            all_labels = list(range(len(target_names)))

            report = classification_report(
                targets,
                preds,
                labels=all_labels,
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

            cm = confusion_matrix(targets, preds, labels=all_labels)
            logger.info(f"\nConfusion Matrix (epoch {trainer.current_epoch}):\n{cm}")

            # Save predictions if requested
            if self.save_predictions and self.output_dir:
                self._save_predictions(trainer.current_epoch)
        finally:
            # Always reset — even if logging or saving throws an exception —
            # so stale data never leaks into the next epoch.
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


class BestModelExportCallback(Callback):
    """
    Exports the best model weights as a portable ``xclinvision_{model_name}_{run_id}.pth``
    file at the end of training.  The file contains only the model ``state_dict`` plus
    lightweight metadata so it can be loaded for inference without PyTorch Lightning.

    Loading example::

        ckpt = torch.load("models/xclinvision_resnet50_v0c4f.pth", map_location="cpu")
        model = build_model(ckpt["model_name"], num_classes=ckpt["num_classes"], pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    """

    def __init__(
        self,
        model_name: str,
        export_dir: str,
        num_classes: int | None = None,
        class_names: list[str] | None = None,
        run_id: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.export_dir = export_dir
        self.class_names = class_names or get_class_names()
        self.CLASS_MAP = get_class_map()
        self.num_classes = num_classes if num_classes is not None else len(self.class_names)
        # Unique short ID per run: "v" + first 4 hex of sha256(model_name + timestamp).
        # Using a timestamp ensures different training runs of the same architecture
        # produce distinct filenames and don't overwrite each other.
        if run_id is None:
            from datetime import datetime as _dt
            seed = f"{model_name}_{_dt.now().isoformat()}"
            digest = hashlib.sha256(seed.encode()).hexdigest()[:4]
            self.run_id = f"v{digest}"
        else:
            self.run_id = run_id

    def _fit_temperature_scaler(self, pl_module, trainer) -> Optional[float]:
        """Fit temperature scaling on the validation set.

        The model must already hold the *best-checkpoint* weights when this
        method is called.  Returns the learned temperature scalar, or None on
        any failure (missing datamodule, empty loader, optimisation error).
        """
        from xclinvision.evaluator import TemperatureScaler

        if trainer.datamodule is None:
            logger.warning(
                "BestModelExportCallback: no datamodule — skipping temperature calibration."
            )
            return None

        try:
            val_loader = trainer.datamodule.val_dataloader()
        except Exception as exc:
            logger.warning("BestModelExportCallback: could not get val_dataloader: %s", exc)
            return None

        device = next(pl_module.model.parameters()).device
        all_logits: list = []
        all_labels: list = []
        pl_module.model.eval()

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(device)
                logits = pl_module.model(x)
                all_logits.append(logits.cpu().float().numpy())
                all_labels.append(
                    y.cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)
                )

        if not all_logits:
            logger.warning("BestModelExportCallback: empty val loader — skipping calibration.")
            return None

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        scaler = TemperatureScaler()
        try:
            temperature = scaler.fit(logits_np, labels_np)
        except Exception as exc:
            logger.warning("BestModelExportCallback: temperature fit failed: %s", exc)
            return None

        return temperature

    def on_train_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        """Load the best checkpoint and export a clean .pth weights file."""
        # Locate the ModelCheckpoint callback
        ckpt_callback = next(
            (cb for cb in trainer.callbacks if hasattr(cb, "best_model_path")),
            None,
        )
        if ckpt_callback is None:
            logger.warning("BestModelExportCallback: no ModelCheckpoint found — skipping .pth export.")
            return

        best_path = ckpt_callback.best_model_path
        if not best_path or not os.path.exists(best_path):
            logger.warning(
                f"BestModelExportCallback: best_model_path '{best_path}' not found — skipping .pth export."
            )
            return

        try:
            raw = torch.load(best_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.error(f"BestModelExportCallback: failed to load checkpoint '{best_path}': {exc}")
            return

        # Strip the "model." prefix that PL adds to every key in state_dict
        raw_sd = raw.get("state_dict", {})
        model_sd = {
            k[len("model."):]: v
            for k, v in raw_sd.items()
            if k.startswith("model.")
        }
        if not model_sd:
            logger.warning("BestModelExportCallback: no 'model.*' keys found in checkpoint — skipping .pth export.")
            return

        # Pull best metric / epoch from the checkpoint path name if possible
        best_val_auc = getattr(ckpt_callback, "best_model_score", None)
        best_val_auc = float(best_val_auc) if best_val_auc is not None else None

        # --- Temperature calibration on best-checkpoint weights --------
        # Temporarily swap in the best-checkpoint weights so calibration
        # runs on the exported model rather than the end-of-training state.
        temperature = None
        original_sd = None
        try:
            original_sd = {k: v.clone() for k, v in pl_module.model.state_dict().items()}
            load_result = pl_module.model.load_state_dict(model_sd, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                logger.warning(
                    "Temperature calibration: checkpoint mismatch "
                    "(missing=%s, unexpected=%s) — skipping.",
                    load_result.missing_keys[:3],
                    load_result.unexpected_keys[:3],
                )
            else:
                temperature = self._fit_temperature_scaler(pl_module, trainer)
        except Exception as exc:
            logger.warning("BestModelExportCallback: temperature calibration failed: %s", exc)
        finally:
            if original_sd is not None:
                pl_module.model.load_state_dict(original_sd, strict=False)
        # ---------------------------------------------------------------

        payload = {
            "model_state_dict": model_sd,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "run_id": self.run_id,
            "class_map": self.CLASS_MAP,
            "class_names": self.class_names,
            "best_val_auc": best_val_auc,
            "temperature": temperature,
            "source_ckpt": best_path,
        }

        os.makedirs(self.export_dir, exist_ok=True)
        out_name = f"xclinvision_{self.model_name}_{self.run_id}.pth"
        out_path = os.path.join(self.export_dir, out_name)
        torch.save(payload, out_path)

        # Write a companion metadata JSON for quick inspection without loading tensors
        meta = {k: v for k, v in payload.items() if k != "model_state_dict"}
        meta_path = os.path.join(self.export_dir, f"xclinvision_{self.model_name}_{self.run_id}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[BestModelExport] Saved → {out_path}  (val_auc={best_val_auc})")
        print(f"\n  Best model exported → {out_path}")


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
"""Training module with PyTorch Lightning integration.

Defines the XClinVisionModel LightningModule and associated training logic,
including progressive unfreezing, focal loss, and comprehensive metrics tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
from torchmetrics import MetricCollection, Accuracy, F1Score, AUROC
from sklearn.metrics import classification_report, confusion_matrix

from .config import get_class_map, get_class_names, is_multilabel, PipelineConfig
from .modeling import freeze_backbone, unfreeze_layers, get_param_counts, _init_classifier_bias

logger = logging.getLogger(__name__)

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
    """Compute loss for Mixup-augmented batch (Multiclass)."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
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
        # 1. Compute Base Cross Entropy (handles label smoothing natively)
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        
        # 2. Extract probability of the true class (pt)
        probs = F.softmax(inputs, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 3. Compute focal term & modulate
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
        # Unweighted BCE — pos_weight is applied separately below
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none",
        )
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)

        # Class-balanced alpha: upweight positives, keep negatives at 1
        if self.pos_weight is not None:
            alpha_t = targets * self.pos_weight + (1 - targets) * 1.0
        else:
            alpha_t = self.alpha

        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


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
    
    Args:
        **kwargs: Unused arguments captured here to allow passing arbitrary 
                  model-building configurations dynamically.
    """
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_type: str = "ce",
        label_smoothing: float = 0.1,
        class_weights: Optional[List[float]] = None,
        progressive_unfreezing: bool = True,
        unfreeze_schedule: Optional[List[int]] = None,
        mixup_alpha: float = 0.0,
        mixup_prob: float = 0.5,
        config: Optional[PipelineConfig] = None,
        backbone_lr_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "config"])

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.progressive_unfreezing = progressive_unfreezing
        self.unfreeze_schedule = unfreeze_schedule or [3]
        self.current_phase = 0
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self._oom_steps = 0
        self._consecutive_ooms = 0
        self._config = config
        self.backbone_lr_factor = backbone_lr_factor
        self.class_names = kwargs.pop("class_names", config.class_names if config else get_class_names())
        self.multilabel = config.multilabel if config else is_multilabel()
        self.decision_threshold = 0.5

        # Loss function
        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        )
        self.criterion = self._setup_loss(loss_type, weight_tensor, label_smoothing, self.multilabel)

        # Classifier bias depends on the loss function:
        #  - BCE: bias=-2.0 so sigmoid starts ~0.12 (standard for sparse multilabel)
        #  - Focal: bias=0.0 so sigmoid starts at 0.5 — with bias=-2.0 the focal
        #    term (1-pt)^gamma suppresses 96% of the negative gradient, creating a
        #    1000:1 gradient imbalance that prevents discrimination.
        if self.multilabel:
            if loss_type == "focal":
                _init_classifier_bias(self.model, bias_value=0.0)
                logger.info("Focal loss: classifier bias = 0.0 (balanced focal weighting)")
            else:
                _init_classifier_bias(self.model, bias_value=-2.0)
                logger.info("BCE loss: classifier bias = -2.0 (low initial sigmoid)")

        if self.multilabel:
            metrics = MetricCollection({
                "acc": Accuracy(task="multilabel", num_labels=num_classes, threshold=0.5),
                "f1_macro": F1Score(task="multilabel", num_labels=num_classes, average="macro", threshold=0.5),
                "auc": AUROC(task="multilabel", num_labels=num_classes, average="macro"),
            })
        else:
            metrics = MetricCollection({
                "acc": Accuracy(task="multiclass", num_classes=num_classes, average="macro"),
                "f1_macro": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
                "auc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
            })
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        if self.progressive_unfreezing:
            freeze_backbone(self.model, unfreeze_head=True)
            self.current_phase = 1 

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
        multilabel: bool = False,
    ) -> nn.Module:
        """Setup loss function with optional class weighting."""
        if multilabel:
            if loss_type == "focal":
                return MultilabelFocalLoss(pos_weight=weight)
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        
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

        # Add 1 because Phase 1 is the starting phase
        target_phase = 1 + sum(1 for e in self.unfreeze_schedule if self.current_epoch >= e)
        if target_phase > self.current_phase:
            self.current_phase = target_phase
            if target_phase == 1:
                freeze_backbone(self.model, unfreeze_head=True)
                msg = "Training head only"
            else:
                # Any phase beyond 1: full fine-tuning with all layers
                unfreeze_layers(self.model, num_layers=0)
                msg = "Full fine-tuning"
                
            self.print(f"Epoch {self.current_epoch}: Phase {target_phase} - {msg}")
            self._rebuild_optimizers()

    def _rebuild_optimizers(self):
        """SAFER optimizer rebuild - avoid trainer.state corruption."""
        if self.trainer is None:
            return
            
        opt_config = self.configure_optimizers()
        
        # Clear and append optimizers safely
        if hasattr(self.trainer, "optimizers") and isinstance(self.trainer.optimizers, list):
            self.trainer.optimizers.clear()
            self.trainer.optimizers.append(opt_config["optimizer"])
        
        # Handle LR schedulers safely
        if "lr_scheduler" in opt_config:
            scheduler_cfg = opt_config["lr_scheduler"]
            scheduler = scheduler_cfg["scheduler"]
            interval = scheduler_cfg.get("interval", "epoch")
            
            try:
                from pytorch_lightning.utilities.types import LRSchedulerConfig
                if hasattr(self.trainer, "lr_scheduler_configs") and isinstance(self.trainer.lr_scheduler_configs, list):
                    self.trainer.lr_scheduler_configs.clear()
                    self.trainer.lr_scheduler_configs.append(
                        LRSchedulerConfig(scheduler=scheduler, interval=interval)
                    )
            except (ImportError, AttributeError):
                if hasattr(self.trainer, "lr_schedulers") and isinstance(self.trainer.lr_schedulers, list):
                    self.trainer.lr_schedulers.clear()
                    self.trainer.lr_schedulers.append({"scheduler": scheduler, "interval": interval})

        trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
        self.print(f"  → Optimizer rebuilt ({trainable} trainable params)")

    # ---- forward / steps --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_epoch_start(self):
        self._apply_unfreeze_schedule()

    def training_step(self, batch, _batch_idx):
        x, y = batch

        try:
            # Fade out mixup probability over the course of training
            max_epochs = self.trainer.max_epochs if self.trainer else 100
            current_mixup_prob = self.mixup_prob * max(0.0, 1.0 - (self.current_epoch / max(1, max_epochs)))
            
            use_mixup = (
                self.mixup_alpha > 0
                and self.training
                and torch.rand(1).item() < current_mixup_prob
            )
            
            if use_mixup:
                mixed_x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
                logits = self(mixed_x)
                
                if self.multilabel:
                    # Linear mix of binary target vectors is robust for BCE
                    # Clamped to [0,1] to prevent BCE logits crashing
                    mixed_y = torch.clamp(lam * y_a.float() + (1 - lam) * y_b.float(), 0.0, 1.0)
                    loss = self.criterion(logits, mixed_y)
                else:
                    loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                    
                self.log("train_loss_mixup", loss, on_step=False, on_epoch=True)
                
                # Compute un-mixed logits solely for metric tracking
                with torch.no_grad():
                    clean_logits = self(x)
                
                # Torchmetrics ALWAYS needs long
                y_metric = y.long()
                clean_probs = torch.sigmoid(clean_logits.detach()) if self.multilabel else F.softmax(clean_logits.detach(), dim=1)
                self.train_metrics.update(clean_probs, y_metric)
            else:
                logits = self(x)
                if self.multilabel:
                    assert logits.shape == y.shape, (
                        f"Shape mismatch: logits {logits.shape} vs targets {y.shape}"
                    )

                # BCE expects floats, Multiclass CE expects long targets
                # NOTE: Label smoothing for multilabel BCE is handled by
                # the loss function or omitted entirely — manual target
                # smoothing corrupts binary labels and causes plateau.
                # if not self.multilabel and self.label_smoothing > 0:
                #     y_loss = y.long()  # CrossEntropy handles label_smoothing natively
                # else:
                y_loss = y.float() if self.multilabel else y.long()
                loss = self.criterion(logits, y_loss)
                
                # Torchmetrics ALWAYS use hard 0/1 labels
                y_metric = y.long()
                probs = torch.sigmoid(logits.detach()) if self.multilabel else F.softmax(logits.detach(), dim=1)
                self.train_metrics.update(probs, y_metric)
                
            self._consecutive_ooms = 0  # Reset on success
            
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._oom_steps += 1
            self._consecutive_ooms += 1
            logger.warning(
                f"[OOM] training_step skipped (batch_size={x.shape[0]}, "
                f"img_size={x.shape[-1]}). Consider reducing --batch-size."
            )
            if self._consecutive_ooms > 5:
                raise RuntimeError("Too many consecutive OOM errors. Please reduce batch size.")
            return None

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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
                    f"batch(es) skipped due to OOM — metrics computed over remainder."
                )
                self._oom_steps = 0
            self.train_metrics.reset()

    def validation_step(self, batch, _batch_idx):
        x, y = batch

        try:
            logits = self(x)
            if self.multilabel:
                assert logits.shape == y.shape, f"Shape mismatch: {logits.shape} vs {y.shape}"
            # BCE expects floats, Multiclass CE expects long targets
            y_loss = y.float() if self.multilabel else y.long()
            loss = self.criterion(logits, y_loss)
            self._consecutive_ooms = 0
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._consecutive_ooms += 1
            logger.warning(f"[OOM] validation_step skipped. Consider reducing batch size.")
            if self._consecutive_ooms > 5:
                raise RuntimeError("Too many consecutive OOM errors. Please reduce batch size.")
            return None

        if self.multilabel:
            probs = torch.sigmoid(logits)
            preds = (probs > self.decision_threshold).int()
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        # Torchmetrics ALWAYS needs long
        y_metric = y.long()
        self.val_metrics.update(probs, y_metric)

        return {"val_loss": loss.detach(), "preds": preds, "targets": y, "probs": probs}

    def on_validation_epoch_end(self):
        try:
            self.log_dict(self.val_metrics.compute(), prog_bar=True)
        except ValueError:
            logger.warning("[OOM] Entire validation epoch was skipped.")
        finally:
            self.val_metrics.reset()

    def test_step(self, batch, _batch_idx):
        x, y = batch
        logits = self(x)
        # BCE expects floats, Multiclass CE expects long targets
        y_loss = y.float() if self.multilabel else y.long()
        loss = self.criterion(logits, y_loss)

        if self.multilabel:
            probs = torch.sigmoid(logits)
            preds = (probs > self.decision_threshold).int()
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        
        # Torchmetrics ALWAYS needs long
        y_metric = y.long()
        self.test_metrics.update(probs, y_metric)
        return {"test_loss": loss.detach(), "preds": preds, "targets": y, "probs": probs}

    def on_test_epoch_end(self):
        try:
            self.log_dict(self.test_metrics.compute(), prog_bar=True)
        except ValueError:
            logger.warning("[OOM] Entire test epoch was skipped.")
        finally:
            self.test_metrics.reset()

    # ---- optimizer --------------------------------------------------------
    def configure_optimizers(self):
        backbone_params, head_params = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ("head", "fc", "classifier", "last_linear")):
                head_params.append(param)
            else:
                backbone_params.append(param)

        # Use self.learning_rate for head, backbone_lr for backbone
        head_lr = self.learning_rate
        backbone_lr = self.learning_rate * self.backbone_lr_factor

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})

        optimizer = AdamW(param_groups, weight_decay=self.weight_decay, eps=1e-8)

        max_epochs = self.trainer.max_epochs if self.trainer else 50
        remaining = max(1, max_epochs - self.current_epoch)

        # On initial setup (epoch 0), use a normal warmup.
        # On rebuild (progressive unfreezing), use at most 1 epoch warmup
        # to avoid wasting cycles — the differential LR already protects
        # newly unfrozen backbone layers.
        if self.current_epoch == 0:
            warmup_epochs = min(2, remaining // 4)
        else:
            warmup_epochs = min(1, remaining // 4)
        if warmup_epochs > 0:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=remaining - warmup_epochs, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
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
        self.multilabel = is_multilabel()
        self._reset()

    def _reset(self):
        self.val_preds: List[np.ndarray] = []
        self.val_targets: List[np.ndarray] = []
        self.val_probs: List[np.ndarray] = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, **kwargs
    ):
        if outputs is None or not isinstance(outputs, dict) or "preds" not in outputs:
            return
        
        # Detach and move to CPU immediately to prevent memory leaks
        self.val_preds.extend(outputs["preds"].detach().cpu().numpy())
        self.val_targets.extend(outputs["targets"].detach().cpu().numpy())
        self.val_probs.extend(outputs["probs"].detach().cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if len(self.val_preds) == 0:
            return

        preds = np.array(self.val_preds)
        targets = np.array(self.val_targets)

        try:
            target_names = getattr(pl_module, 'class_names', get_class_names())
            all_labels = list(range(len(target_names)))

            if self.multilabel:
                from sklearn.metrics import classification_report as _cr
                report = _cr(
                    targets,
                    preds,
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                )
            else:
                report = classification_report(
                    targets,
                    preds,
                    labels=all_labels,
                    target_names=target_names,
                    output_dict=True,
                    zero_division=0,
                )

            for cls_name, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            pl_module.log(f"val_{cls_name}_{metric_name}", float(value))

            cm = confusion_matrix(targets, preds, labels=all_labels) if not self.multilabel else None
            if cm is not None:
                logger.info(f"\nConfusion Matrix (epoch {trainer.current_epoch}):\n{cm}")

            # Per-class threshold optimization for multilabel
            if self.multilabel and len(self.val_probs) > 0:
                from sklearn.metrics import f1_score as _f1
                probs_arr = np.array(self.val_probs)
                opt_thresholds = []
                for c in range(targets.shape[1]):
                    best_f1, best_th = 0.0, 0.5
                    for th in np.arange(0.1, 0.9, 0.05):
                        f1_c = _f1(targets[:, c], (probs_arr[:, c] >= th).astype(int), zero_division=0)
                        if f1_c > best_f1:
                            best_f1, best_th = f1_c, th
                    opt_thresholds.append(best_th)
                opt_preds = np.column_stack([
                    (probs_arr[:, c] >= opt_thresholds[c]).astype(int) for c in range(targets.shape[1])
                ])
                opt_f1 = _f1(targets, opt_preds, average='macro', zero_division=0)
                pl_module.log("val_f1_macro_opt", opt_f1, prog_bar=True)
                logger.info(
                    "Optimized thresholds: %s -> F1_macro=%.4f",
                    {n: f"{t:.2f}" for n, t in zip(target_names, opt_thresholds)}, opt_f1,
                )

            if self.save_predictions and self.output_dir:
                self._save_predictions(trainer.current_epoch)
        finally:
            self._reset()

    def _save_predictions(self, epoch: int):
        import json
        import os
        from datetime import datetime

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.multilabel:
            save_data = {
                "epoch": epoch,
                "timestamp": timestamp,
                "predictions": [p.tolist() if hasattr(p, 'tolist') else list(p) for p in self.val_preds],
                "targets": [t.tolist() if hasattr(t, 'tolist') else list(t) for t in self.val_targets],
                "probabilities": [p.tolist() for p in self.val_probs],
            }
        else:
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
        if run_id is None:
            from datetime import datetime as _dt
            seed = f"{model_name}_{_dt.now().isoformat()}"
            digest = hashlib.sha256(seed.encode()).hexdigest()[:4]
            self.run_id = f"v{digest}"
        else:
            self.run_id = run_id

    def _fit_temperature_scaler(self, pl_module, trainer) -> Optional[float]:
        from xclinvision.evaluator import TemperatureScaler
        if is_multilabel():
            logger.info("BestModelExportCallback: skipping temperature calibration (multilabel).")
            return None

        if trainer.datamodule is None:
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

    def on_train_end(self, trainer, pl_module) -> None: 
        ckpt_callback = next(
            (cb for cb in trainer.callbacks if hasattr(cb, "best_model_path")),
            None,
        )
        if ckpt_callback is None:
            logger.warning("BestModelExportCallback: no ModelCheckpoint found — skipping export.")
            return

        best_path = ckpt_callback.best_model_path
        if not best_path or not os.path.exists(best_path):
            logger.warning(f"BestModelExportCallback: best_model_path '{best_path}' not found.")
            return

        try:
            raw = torch.load(best_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.error(f"BestModelExportCallback: failed to load checkpoint '{best_path}': {exc}")
            return

        raw_sd = raw.get("state_dict", {})
        model_sd = {
            k[len("model."):]: v
            for k, v in raw_sd.items()
            if k.startswith("model.")
        }
        if not model_sd:
            logger.warning("BestModelExportCallback: no 'model.*' keys found.")
            return

        best_val_auc = getattr(ckpt_callback, "best_model_score", None)
        best_val_auc = float(best_val_auc) if best_val_auc is not None else None

        temperature = None
        original_sd = None
        try:
            original_sd = {k: v.clone() for k, v in pl_module.model.state_dict().items()}
            load_result = pl_module.model.load_state_dict(model_sd, strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                logger.warning(
                    "Temperature calibration: checkpoint mismatch (missing=%s, unexpected=%s).",
                    load_result.missing_keys[:3], load_result.unexpected_keys[:3]
                )
            else:
                temperature = self._fit_temperature_scaler(pl_module, trainer)
        except Exception as exc:
            logger.warning("BestModelExportCallback: temperature calibration failed: %s", exc)
        finally:
            if original_sd is not None:
                pl_module.model.load_state_dict(original_sd, strict=False)

        thresholds = None
        if is_multilabel() and trainer.datamodule is not None:
            try:
                from xclinvision.evaluator import ThresholdOptimizer
                val_loader = trainer.datamodule.val_dataloader()
                device = next(pl_module.model.parameters()).device
                all_probs, all_labels = [], []
                pl_module.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        x, y = batch
                        x = x.to(device)
                        logits = pl_module.model(x)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_probs.append(probs)
                        all_labels.append(
                            y.cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)
                        )
                if all_probs:
                    probs_np = np.concatenate(all_probs, axis=0)
                    labels_np = np.concatenate(all_labels, axis=0)
                    thresh_opt = ThresholdOptimizer(class_names=self.class_names)
                    thresh_opt.fit(labels_np, probs_np)
                    thresholds = thresh_opt.thresholds
                    logger.info(f"[BestModelExport] Optimized thresholds: {thresholds}")
            except ImportError:
                logger.warning("BestModelExportCallback: ThresholdOptimizer not found in evaluator. Skipping threshold optimization.")
            except Exception as exc:
                logger.warning("BestModelExportCallback: threshold optimization failed: %s", exc)

        payload = {
            "model_state_dict": model_sd,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "run_id": self.run_id,
            "class_map": self.CLASS_MAP,
            "class_names": self.class_names,
            "best_val_auc": best_val_auc,
            "temperature": temperature,
            "thresholds": thresholds,
            "source_ckpt": best_path,
        }

        os.makedirs(self.export_dir, exist_ok=True)
        out_name = f"xclinvision_{self.model_name}_{self.run_id}.pth"
        out_path = os.path.join(self.export_dir, out_name)
        torch.save(payload, out_path)

        meta = {k: v for k, v in payload.items() if k != "model_state_dict"}
        meta_path = os.path.join(self.export_dir, f"xclinvision_{self.model_name}_{self.run_id}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"[BestModelExport] Saved -> {out_path}  (val_auc={best_val_auc})")
        print(f"\n  Best model exported -> {out_path}")


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
        epoch = trainer.current_epoch

        if (epoch + 1) % self.every_n_epochs != 0:
            return

        try:
            from .xai import ValidationXAI

            output_path = f"{self.output_dir}/epoch_{epoch:03d}"
            xai = ValidationXAI(pl_module.model, self.architecture, output_path, img_size=self.image_size)

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



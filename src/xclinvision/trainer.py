"""Training module with PyTorch Lightning integration."""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class XClinVisionModel(pl.LightningModule):
    """PyTorch Lightning module for XClinVision training."""
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        loss_type: str = "focal",
        class_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize loss function
        if loss_type == "focal":
            weight = torch.tensor(class_weights) if class_weights else None
            self.criterion = FocalLoss(weight=weight)
        elif loss_type == "ce":
            weight = torch.tensor(class_weights) if class_weights else None
            self.criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        # Metrics storage
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store for epoch-end metrics
        self.train_preds.extend(preds.cpu().numpy())
        self.train_targets.extend(y.cpu().numpy())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        # Store for epoch-end metrics
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(y.cpu().numpy())
        
        return {"val_loss": loss, "preds": preds, "targets": y, "probs": probs}
    
    def on_train_epoch_end(self):
        """Compute training metrics at epoch end."""
        if len(self.train_preds) > 0:
            train_f1 = f1_score(
                self.train_targets,
                self.train_preds,
                average="macro",
                zero_division=0,
            )
            self.log("train_f1", train_f1, prog_bar=True)
            
        # Clear storage
        self.train_preds = []
        self.train_targets = []
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end."""
        if len(self.val_preds) == 0:
            return
            
        # Compute metrics
        val_acc = accuracy_score(self.val_targets, self.val_preds)
        val_f1 = f1_score(
            self.val_targets,
            self.val_preds,
            average="macro",
            zero_division=0,
        )
        
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        
        # Clear storage
        self.val_preds = []
        self.val_targets = []
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class MetricsCallback(Callback):
    """Callback for computing detailed metrics."""
    
    def __init__(self):
        super().__init__()
        self.val_preds = []
        self.val_targets = []
        self.val_probs = []
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect validation outputs."""
        self.val_preds.extend(outputs["preds"].cpu().numpy())
        self.val_targets.extend(outputs["targets"].cpu().numpy())
        self.val_probs.extend(outputs["probs"].cpu().numpy())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute and log detailed metrics."""
        if len(self.val_preds) == 0:
            return
            
        # Compute AUC-ROC
        try:
            # One-hot encode targets for multi-class AUC
            y_true_onehot = np.eye(3)[self.val_targets]
            val_auc = roc_auc_score(
                y_true_onehot,
                np.array(self.val_probs),
                multi_class="ovr",
                average="macro",
            )
            pl_module.log("val_auc", val_auc, prog_bar=True)
        except ValueError:
            pass
            
        # Clear storage
        self.val_preds = []
        self.val_targets = []
        self.val_probs = []

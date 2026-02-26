"""Evaluation metrics and calibration analysis."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from scipy.special import softmax


class MetricsComputer:
    """Compute comprehensive evaluation metrics for medical AI."""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ["Normal", "Pneumonia", "Tuberculosis"]
        
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all relevant metrics for medical classification."""
        metrics = {}
        
        # Basic metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["macro_precision"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["macro_recall"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["macro_f1"] = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f"{class_name}_precision"] = precision_score(
                y_true, y_pred, labels=[i], average=None, zero_division=0
            )[0]
            metrics[f"{class_name}_recall"] = recall_score(
                y_true, y_pred, labels=[i], average=None, zero_division=0
            )[0]
            metrics[f"{class_name}_f1"] = f1_score(
                y_true, y_pred, labels=[i], average=None, zero_division=0
            )[0]
            
        # Sensitivity and Specificity
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        for i, class_name in enumerate(self.class_names):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            fn = np.sum(cm[i, :]) - cm[i, i]
            tp = cm[i, i]
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics[f"{class_name}_sensitivity"] = sensitivity
            metrics[f"{class_name}_specificity"] = specificity
            
        # AUC-ROC
        try:
            y_true_onehot = np.eye(len(self.class_names))[y_true]
            metrics["macro_auc"] = roc_auc_score(
                y_true_onehot, y_probs, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["macro_auc"] = 0.0
            
        return metrics
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Compute confusion matrix."""
        return confusion_matrix(
            y_true, y_pred, labels=range(len(self.class_names))
        )


class CalibrationAnalyzer:
    """Analyze model calibration using Expected Calibration Error."""
    
    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins
        
    def compute_ece(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> float:
        """Compute Expected Calibration Error."""
        # Get predicted class and confidence
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        
        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = np.logical_and(
                confidences > bin_lower,
                confidences <= bin_upper,
            )
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                ece += (bin_size / len(y_true)) * np.abs(avg_confidence - avg_accuracy)
                
        return ece
    
    def compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute calibration curve data."""
        y_pred = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = np.logical_and(
                confidences > bin_lower,
                confidences <= bin_upper,
            )
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(np.mean(accuracies[in_bin]))
                bin_counts.append(bin_size)
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0.0)
                bin_counts.append(0)
                
        return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)


class TemperatureScaler:
    """Temperature scaling for model calibration."""
    
    def __init__(self):
        self.temperature = 1.0
        
    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Learn optimal temperature on validation set."""
        import torch
        from torch.optim import LBFGS
        
        # Convert to tensors
        logits_tensor = torch.FloatTensor(logits)
        labels_tensor = torch.LongTensor(y_true)
        
        # Initialize temperature
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        
        # Optimize
        def eval_fn():
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss
            
        optimizer = LBFGS([temperature], lr=0.01, max_iter=50)
        optimizer.step(eval_fn)
        
        self.temperature = temperature.item()
        return self.temperature
    
    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities."""
        scaled_logits = self.scale(logits)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

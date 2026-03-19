"""Evaluation metrics and calibration analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    multilabel_confusion_matrix,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricsComputer:
    """Compute comprehensive evaluation metrics for medical AI classification.

    Covers accuracy, per-class precision/recall/F1, macro/weighted averages,
    sensitivity, specificity, AUC-ROC, and sklearn classification report.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        from xclinvision.config import get_class_names, is_multilabel
        self.class_names = class_names or get_class_names()
        self.multilabel = is_multilabel()

    # ------------------------------------------------------------------
    # Full metric suite
    # ------------------------------------------------------------------

    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics.

        Supports both multi-class (integer labels) and multi-label (binary
        matrix) depending on ``self.multilabel``.

        Args:
            y_true: Ground-truth labels — shape (N,) for multiclass or
                (N, num_classes) binary matrix for multilabel.
            y_pred: Predicted labels — same shape convention as y_true.
            y_probs: Predicted probabilities, shape (N, num_classes).

        Returns:
            Flat dict mapping metric names to float values.
        """
        if self.multilabel:
            return self._compute_multilabel_metrics(y_true, y_pred, y_probs)
        return self._compute_multiclass_metrics(y_true, y_pred, y_probs)

    # ------------------------------------------------------------------
    # Multi-class metrics (original)
    # ------------------------------------------------------------------

    def _compute_multiclass_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict[str, float]:
        n_classes = len(self.class_names)
        metrics: Dict[str, float] = {}

        # ---- aggregate -----------------------------------------------
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["macro_precision"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["macro_recall"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["weighted_precision"] = float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["weighted_recall"] = float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )

        # ---- per-class -----------------------------------------------
        per_precision = precision_score(
            y_true, y_pred, labels=list(range(n_classes)),
            average=None, zero_division=0
        )
        per_recall = recall_score(
            y_true, y_pred, labels=list(range(n_classes)),
            average=None, zero_division=0
        )
        per_f1 = f1_score(
            y_true, y_pred, labels=list(range(n_classes)),
            average=None, zero_division=0
        )

        for i, name in enumerate(self.class_names):
            metrics[f"{name}_precision"] = float(per_precision[i])
            metrics[f"{name}_recall"]    = float(per_recall[i])
            metrics[f"{name}_f1"]        = float(per_f1[i])

        # ---- sensitivity / specificity (OvR) --------------------------
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        total = np.sum(cm)
        for i, name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = total - tp - fp - fn

            metrics[f"{name}_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics[f"{name}_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics[f"{name}_ppv"]         = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            metrics[f"{name}_npv"]         = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

        # ---- AUC-ROC --------------------------------------------------
        try:
            if np.any(y_true < 0) or np.any(y_true >= n_classes):
                raise ValueError(
                    f"y_true contains labels outside [0, {n_classes}): "
                    f"min={int(np.min(y_true))}, max={int(np.max(y_true))}"
                )
            metrics["macro_auc"] = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            )
            metrics["weighted_auc"] = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
            )
            # Per-class OvR AUC — clinically more informative than aggregate only
            per_class_auc = roc_auc_score(
                y_true, y_probs, multi_class="ovr", average=None
            )
            for i, name in enumerate(self.class_names):
                metrics[f"{name}_auc"] = float(per_class_auc[i])
        except ValueError as exc:
            logger.warning(f"AUC-ROC computation failed: {exc}")
            metrics["macro_auc"] = 0.0
            metrics["weighted_auc"] = 0.0
            for name in self.class_names:
                metrics[f"{name}_auc"] = 0.0

        return metrics

    # ------------------------------------------------------------------
    # Multi-label metrics
    # ------------------------------------------------------------------

    def _compute_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict[str, float]:
        """Compute evaluation metrics for multi-label classification.

        Args:
            y_true: Binary ground-truth matrix, shape (N, num_classes).
            y_pred: Binary prediction matrix, shape (N, num_classes).
            y_probs: Predicted probabilities, shape (N, num_classes).
        """
        n_classes = len(self.class_names)
        metrics: Dict[str, float] = {}

        # ---- aggregate (sample-averaged) ---------------------------------
        metrics["subset_accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["macro_precision"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["macro_recall"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["macro_f1"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["weighted_f1"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["sample_precision"] = float(
            precision_score(y_true, y_pred, average="samples", zero_division=0)
        )
        metrics["sample_recall"] = float(
            recall_score(y_true, y_pred, average="samples", zero_division=0)
        )
        metrics["sample_f1"] = float(
            f1_score(y_true, y_pred, average="samples", zero_division=0)
        )

        # ---- per-class ---------------------------------------------------
        per_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, name in enumerate(self.class_names):
            metrics[f"{name}_precision"] = float(per_precision[i])
            metrics[f"{name}_recall"]    = float(per_recall[i])
            metrics[f"{name}_f1"]        = float(per_f1[i])

        # ---- per-class sensitivity / specificity via multilabel CM --------
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        for i, name in enumerate(self.class_names):
            tn, fp, fn, tp = mcm[i].ravel()
            metrics[f"{name}_sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics[f"{name}_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics[f"{name}_ppv"]         = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            metrics[f"{name}_npv"]         = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

        # ---- AUC-ROC (per-label binary) -----------------------------------
        try:
            metrics["macro_auc"] = float(
                roc_auc_score(y_true, y_probs, average="macro")
            )
            metrics["weighted_auc"] = float(
                roc_auc_score(y_true, y_probs, average="weighted")
            )
            per_auc = roc_auc_score(y_true, y_probs, average=None)
            for i, name in enumerate(self.class_names):
                metrics[f"{name}_auc"] = float(per_auc[i])
        except ValueError as exc:
            logger.warning(f"AUC-ROC computation failed (multilabel): {exc}")
            metrics["macro_auc"] = 0.0
            metrics["weighted_auc"] = 0.0
            for name in self.class_names:
                metrics[f"{name}_auc"] = 0.0

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Return confusion matrix (or multilabel confusion matrix)."""
        if self.multilabel:
            return multilabel_confusion_matrix(y_true, y_pred)
        return confusion_matrix(
            y_true, y_pred, labels=list(range(len(self.class_names)))
        )

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False,
    ):
        """Return sklearn's classification report as a string or dict.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            output_dict: If True return a dict; otherwise return a string.
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0,
            output_dict=output_dict,
        )

    def print_summary(
        self,
        metrics: Dict[str, float],
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
    ) -> None:
        """Pretty-print key metrics and (optionally) a confusion matrix."""
        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("EVALUATION SUMMARY")
        lines.append("=" * 60)
        lines.append(f"  Accuracy      : {metrics.get('accuracy', metrics.get('subset_accuracy', 0)):.4f}")
        lines.append(f"  Macro F1      : {metrics.get('macro_f1', 0):.4f}")
        lines.append(f"  Weighted F1   : {metrics.get('weighted_f1', 0):.4f}")
        lines.append(f"  Macro AUC     : {metrics.get('macro_auc', 0):.4f}")
        lines.append("")
        for name in self.class_names:
            sens = metrics.get(f"{name}_sensitivity", 0)
            spec = metrics.get(f"{name}_specificity", 0)
            f1   = metrics.get(f"{name}_f1", 0)
            lines.append(f"  {name:<14}: Sens={sens:.3f}  Spec={spec:.3f}  F1={f1:.3f}")
        if y_true is not None and y_pred is not None:
            lines.append("")
            if self.multilabel:
                # multilabel_confusion_matrix returns (C, 2, 2) — show per-label TN/FP/FN/TP
                mcm = self.compute_confusion_matrix(y_true, y_pred)
                lines.append("Per-label Confusion Matrices (TN, FP, FN, TP):")
                for i, name in enumerate(self.class_names):
                    tn, fp, fn, tp = mcm[i].ravel()
                    lines.append(f"  {name}: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
            else:
                lines.append("Confusion Matrix (rows=true, cols=pred):")
                cm = self.compute_confusion_matrix(y_true, y_pred)
                header = "  " + "  ".join(f"{n[:6]:>6}" for n in self.class_names)
                lines.append(header)
                for row, name in zip(cm, self.class_names):
                    lines.append(f"  {name[:6]:>6}  " + "  ".join(f"{v:>6}" for v in row))
        lines.append("=" * 60)
        lines.append("")
        logger.info("\n".join(lines))

    def save_results(
        self,
        metrics: Dict,
        output_path: str,
    ) -> None:
        """Serialise the metrics dict to a JSON file.

        Args:
            metrics: Dict produced by compute_all_metrics (may contain nested
                dicts or lists – each value must be JSON-serialisable).
            output_path: Destination file path (created if needed).
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Results saved to {path}")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class CalibrationAnalyzer:
    """Analyse model calibration using Expected Calibration Error (ECE)."""

    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins

    def compute_ece(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> float:
        """Compute Expected Calibration Error.

        Partitions samples into confidence bins; ECE is the weighted mean
        absolute difference between average confidence and average accuracy
        within each bin.

        Args:
            y_true: Ground-truth integer labels, shape (N,).
            y_probs: Predicted probabilities, shape (N, num_classes).

        Returns:
            ECE ∈ [0, 1] (lower is better-calibrated).
        """
        y_pred      = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies  = (y_pred == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin  = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
            bin_n   = int(np.sum(in_bin))
            if bin_n > 0:
                avg_conf = float(np.mean(confidences[in_bin]))
                avg_acc  = float(np.mean(accuracies[in_bin]))
                ece += (bin_n / len(y_true)) * abs(avg_conf - avg_acc)

        return ece

    def compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute reliability diagram data.

        Returns:
            bin_centers   (num_bins,) – mid-point of each confidence bin
            bin_accuracies(num_bins,) – mean accuracy within each bin
            bin_counts    (num_bins,) – number of samples in each bin
        """
        y_pred      = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies  = (y_pred == y_true).astype(float)

        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_centers, bin_accuracies, bin_counts = [], [], []

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin  = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
            bin_n   = int(np.sum(in_bin))
            midpoint = float((lo + hi) / 2)
            bin_centers.append(midpoint)
            bin_accuracies.append(float(np.mean(accuracies[in_bin])) if bin_n > 0 else 0.0)
            bin_counts.append(bin_n)

        return (
            np.array(bin_centers),
            np.array(bin_accuracies),
            np.array(bin_counts),
        )


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Post-hoc calibration via temperature scaling (Guo et al., 2017).

    Learns a single scalar T on a held-out validation set, then divides
    logits by T before softmax to improve calibration.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self._is_fitted: bool = False

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
        """Learn the optimal temperature on a validation set.

        Uses L-BFGS to minimise NLL w.r.t. temperature.

        Args:
            logits: Raw model logits, shape (N, num_classes).
            y_true: Ground-truth integer labels, shape (N,).

        Returns:
            The learned temperature scalar T.
        """
        from torch.optim import LBFGS

        logits_t = torch.FloatTensor(logits)
        labels_t = torch.LongTensor(y_true)
        # Initialize log(T=1.5) ≈ 0.405
        log_temperature = torch.nn.Parameter(torch.ones(1) * 0.405)

        def eval_fn():
            optimizer.zero_grad()
            # torch.exp guarantees strictly positive temperature
            loss = torch.nn.CrossEntropyLoss()(logits_t / torch.exp(log_temperature), labels_t)
            loss.backward()
            return loss

        optimizer = LBFGS([log_temperature], lr=0.01, max_iter=50)
        optimizer.step(eval_fn)

        self.temperature = max(float(torch.exp(log_temperature).item()), 0.01)
        self._is_fitted = True
        logger.info(
            f"Temperature scaling converged: T = {self.temperature:.4f}, "
            f"NLL = {torch.nn.CrossEntropyLoss()(logits_t / self.temperature, labels_t).item():.4f}"
        )
        return self.temperature

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Divide raw logits by the learned temperature.

        Args:
            logits: Raw model logits, shape (N, num_classes).

        Returns:
            Temperature-scaled logits (same shape).
        """
        if not self._is_fitted:
            logger.warning(
                "TemperatureScaler.scale() called before fit() – "
                "returning unscaled logits (T=1.0)."
            )
        return logits / max(self.temperature, 1e-4)

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Return calibrated softmax probabilities.

        Args:
            logits: Raw model logits, shape (N, num_classes).

        Returns:
            Calibrated probabilities, shape (N, num_classes).
        """
        scaled = self.scale(logits)
        exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def save(self, path: str) -> None:
        """Persist the learned temperature to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)
        logger.info(f"TemperatureScaler saved to {p}")

    def load(self, path: str) -> None:
        """Load a previously saved temperature from disk."""
        with open(path) as f:
            data = json.load(f)
        self.temperature = float(data["temperature"])
        self._is_fitted = True
        logger.info(f"TemperatureScaler loaded: T = {self.temperature:.4f}")

        

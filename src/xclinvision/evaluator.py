"""Evaluation metrics and calibration analysis."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        multilabel: Optional[bool] = None,
    ):
        from xclinvision.config import get_class_names, is_multilabel
        self.class_names = class_names or get_class_names()
        # `multilabel` is now a hint; the authoritative routing in
        # compute_all_metrics is the input-array shape. Keep the attribute
        # for back-compat with callers that mutate it directly.
        self.multilabel = is_multilabel() if multilabel is None else multilabel

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

        Routing is determined by the *shape* of ``y_true``:

        - 1-D integer labels → multiclass metrics
        - 2-D binary matrix (N, num_classes) → multilabel metrics

        ``self.multilabel`` is retained as a hint for back-compat but is
        ignored when it conflicts with the input shape. Mismatched arrays
        (e.g. 1-D y_true with 2-D y_pred) raise ValueError.

        Args:
            y_true: Ground-truth labels — shape (N,) for multiclass or
                (N, num_classes) binary matrix for multilabel.
            y_pred: Predicted labels — same shape convention as y_true.
            y_probs: Predicted probabilities, shape (N, num_classes).

        Returns:
            Flat dict mapping metric names to float values.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim != y_pred.ndim:
            raise ValueError(
                f"y_true.ndim ({y_true.ndim}) != y_pred.ndim ({y_pred.ndim}); "
                "use 1-D arrays for multiclass or 2-D for multilabel."
            )
        is_ml = y_true.ndim == 2 and y_true.shape[-1] > 1
        if is_ml:
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
        multilabel: bool = False,
    ) -> float:
        """Compute Expected Calibration Error.

        For multiclass: partitions samples into confidence bins based on
        argmax probability.
        For multilabel: computes per-label binary ECE and returns the mean.

        Args:
            y_true: Ground-truth labels. Shape (N,) for multiclass or (N, C) for multilabel.
            y_probs: Predicted probabilities, shape (N, num_classes).
            multilabel: If True, treat as multilabel binary calibration.

        Returns:
            ECE ∈ [0, 1] (lower is better-calibrated).
        """
        if multilabel:
            return self._compute_ece_multilabel(y_true, y_probs)

        y_pred      = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies  = (y_pred == y_true).astype(float)

        return self._bin_ece(confidences, accuracies)

    def _compute_ece_multilabel(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> float:
        """Compute mean per-label binary ECE for multilabel classification.

        For each label, bins samples by predicted P(positive) and compares
        against the actual positive rate in each bin.  This is the standard
        binary calibration measure: "when the model says P=0.7, is the
        condition actually present ~70% of the time?"
        """
        num_labels = y_probs.shape[1]
        eces = []
        for c in range(num_labels):
            probs = y_probs[:, c]                   # P(positive)
            labels = y_true[:, c].astype(float)      # actual positive rate
            eces.append(self._bin_ece(probs, labels))
        return float(np.mean(eces))

    def _bin_ece(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """Compute binned ECE from confidence and accuracy arrays."""
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        ece = 0.0
        n = len(confidences)
        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
            bin_n = int(np.sum(in_bin))
            if bin_n > 0:
                avg_conf = float(np.mean(confidences[in_bin]))
                avg_acc  = float(np.mean(accuracies[in_bin]))
                ece += (bin_n / n) * abs(avg_conf - avg_acc)
        return ece

    def compute_calibration_curve(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        multilabel: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute reliability diagram data.

        For multilabel, averages per-label binary calibration curves.

        Returns:
            bin_centers   (num_bins,) – mid-point of each confidence bin
            bin_accuracies(num_bins,) – mean accuracy within each bin
            bin_counts    (num_bins,) – number of samples in each bin
        """
        if multilabel:
            return self._calibration_curve_multilabel(y_true, y_probs)

        y_pred      = np.argmax(y_probs, axis=1)
        confidences = np.max(y_probs, axis=1)
        accuracies  = (y_pred == y_true).astype(float)

        return self._bin_calibration_curve(confidences, accuracies)

    def _calibration_curve_multilabel(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Average per-label binary calibration curves for multilabel."""
        num_labels = y_probs.shape[1]
        all_accs = np.zeros(self.num_bins)
        all_counts = np.zeros(self.num_bins, dtype=int)
        centers = None
        for c in range(num_labels):
            probs = y_probs[:, c]                   # P(positive)
            labels = y_true[:, c].astype(float)      # actual positive rate
            bc, ba, bn = self._bin_calibration_curve(probs, labels)
            centers = bc
            all_accs += ba * bn
            all_counts += bn
        safe_counts = np.maximum(all_counts, 1)
        return centers, all_accs / safe_counts, all_counts

    def _bin_calibration_curve(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute binned calibration curve from confidence and accuracy arrays."""
        bin_boundaries = np.linspace(0, 1, self.num_bins + 1)
        bin_centers, bin_accuracies, bin_counts = [], [], []

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
            bin_n = int(np.sum(in_bin))
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
        #: Per-class bias, populated only by :meth:`fit_per_class`. ``None``
        #: means pure temperature scaling (the b = 0 special case).
        self.bias: Optional[List[float]] = None
        self._is_fitted: bool = False

    @property
    def is_per_class(self) -> bool:
        """True when :attr:`temperature` holds one value per class."""
        return isinstance(self.temperature, (list, tuple, np.ndarray))

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        multilabel: bool = False,
    ) -> float:
        """Learn the optimal temperature on a validation set.

        Uses L-BFGS to minimise NLL w.r.t. temperature.

        Args:
            logits: Raw model logits, shape (N, num_classes).
            y_true: Ground-truth labels — shape (N,) for multiclass
                    or (N, C) for multilabel.
            multilabel: If True, use BCEWithLogitsLoss instead of
                        CrossEntropyLoss.

        Returns:
            The learned temperature scalar T.
        """
        from torch.optim import LBFGS

        logits_t = torch.FloatTensor(logits)
        # Initialize log(T=1.5) ≈ 0.405
        log_temperature = torch.nn.Parameter(torch.ones(1) * 0.405)

        if multilabel:
            labels_t = torch.FloatTensor(y_true)
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            labels_t = torch.LongTensor(y_true)
            loss_fn = torch.nn.CrossEntropyLoss()

        def eval_fn():
            optimizer.zero_grad()
            # torch.exp guarantees strictly positive temperature
            loss = loss_fn(logits_t / torch.exp(log_temperature), labels_t)
            loss.backward()
            return loss

        optimizer = LBFGS([log_temperature], lr=0.01, max_iter=50)
        optimizer.step(eval_fn)

        self.temperature = max(float(torch.exp(log_temperature).item()), 0.01)
        self._is_fitted = True
        final_loss = loss_fn(
            logits_t / self.temperature, labels_t
        ).item()
        logger.info(
            f"Temperature scaling converged: T = {self.temperature:.4f}, "
            f"loss = {final_loss:.4f}"
        )
        return self.temperature

    def fit_per_class(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        *,
        with_bias: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """Fit one affine calibrator per class: ``p = sigmoid(z / T + b)``.

        A single global temperature averages every class's miscalibration into
        one scalar, which on this task fits T ~ 1.0 and does nothing. Fitting
        per class recovers the per-label scale.

        ``with_bias`` is what makes this usable here. Temperature scaling can
        only pull logits toward or away from zero; it cannot *shift* them, so
        it cannot correct a systematic offset. These models carry exactly such
        an offset — training used pos_weight 5.1-15.2, which biases every head
        toward positive — and the fitted bias comes out strongly negative on
        every class. With b fixed at 0 the residual ECE stays around 0.17-0.28;
        with b free it drops to roughly 0.015. Pass ``with_bias=False`` for
        textbook temperature scaling (T only).

        Both parameters are fitted by L-BFGS on per-class binary NLL. Fit this
        on a validation split, never on test.

        Args:
            logits: Raw model logits, shape (N, C).
            y_true: Binary ground truth, shape (N, C).
            with_bias: Fit the offset. False gives temperature-only.

        Returns:
            ``(temperatures, biases)`` — one entry per class. Biases are all
            zero when ``with_bias`` is False.
        """
        logits = np.asarray(logits, dtype=np.float32)
        y_true = np.asarray(y_true, dtype=np.float32)
        if logits.shape != y_true.shape:
            raise ValueError(
                f"logits {logits.shape} and y_true {y_true.shape} must have the same shape"
            )

        temps: List[float] = []
        biases: List[float] = []

        for c in range(logits.shape[1]):
            z = torch.from_numpy(logits[:, c])
            y = torch.from_numpy(y_true[:, c])

            log_t = torch.zeros(1, requires_grad=True)
            b = torch.zeros(1, requires_grad=True)
            params = [log_t, b] if with_bias else [log_t]
            optimizer = torch.optim.LBFGS(params, lr=0.1, max_iter=200)

            def eval_fn():
                optimizer.zero_grad()
                scaled = z / torch.exp(log_t) + (b if with_bias else 0.0)
                loss = F.binary_cross_entropy_with_logits(scaled, y)
                loss.backward()
                return loss

            optimizer.step(eval_fn)
            temps.append(max(float(torch.exp(log_t).item()), 1e-4))
            biases.append(float(b.item()) if with_bias else 0.0)

        self.temperature = temps
        self.bias = biases
        self._is_fitted = True
        logger.info(
            "Per-class calibration fitted: T=%s bias=%s",
            [round(t, 4) for t in temps], [round(x, 4) for x in biases],
        )
        return temps, biases

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
        if self.is_per_class:
            t = np.asarray(self.temperature, dtype=np.float32)
            b = np.asarray(self.bias if self.bias is not None else 0.0, dtype=np.float32)
            return logits / np.maximum(t, 1e-4) + b
        return logits / max(self.temperature, 1e-4)

    def predict_proba(
        self, logits: np.ndarray, multilabel: bool = False,
    ) -> np.ndarray:
        """Return calibrated probabilities.

        Args:
            logits: Raw model logits, shape (N, num_classes).
            multilabel: If True, apply sigmoid; otherwise softmax.

        Returns:
            Calibrated probabilities, shape (N, num_classes).
        """
        scaled = self.scale(logits)
        if multilabel:
            return 1.0 / (1.0 + np.exp(-scaled))
        exp = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def save(self, path: str) -> None:
        """Persist the learned calibration to disk.

        Writes ``bias`` alongside ``temperature`` so a per-class affine fit
        round-trips; the key is absent for a scalar fit.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, object] = {"temperature": self.temperature}
        if self.bias is not None:
            payload["bias"] = self.bias
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"TemperatureScaler saved to {p}")

    def load(self, path: str) -> None:
        """Load a previously saved calibration from disk."""
        with open(path) as f:
            data = json.load(f)
        raw = data["temperature"]
        if isinstance(raw, (list, tuple)):
            self.temperature = [float(t) for t in raw]
            self.bias = [float(b) for b in data.get("bias", [0.0] * len(raw))]
            logger.info("TemperatureScaler loaded per-class: T = %s", self.temperature)
        else:
            self.temperature = float(raw)
            self.bias = None
            logger.info(f"TemperatureScaler loaded: T = {self.temperature:.4f}")
        self._is_fitted = True


# ---------------------------------------------------------------------------
# Per-class Threshold Optimization (multilabel)
# ---------------------------------------------------------------------------

class ThresholdOptimizer:
    """Find optimal per-class decision thresholds for multilabel classification.

    Searches a grid of thresholds per class to maximise per-class F1 on a
    validation set.  Results are saved as a JSON sidecar file alongside model
    weights.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        from xclinvision.config import get_class_names
        self.class_names = class_names or get_class_names()
        self.thresholds: Dict[str, float] = {n: 0.5 for n in self.class_names}
        self._is_fitted: bool = False

    def fit(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
    ) -> Dict[str, float]:
        """Find optimal per-class thresholds that maximise F1.

        Uses sklearn's ``precision_recall_curve`` to compute the exact
        threshold that maximises the F1 score for each class — faster and
        more precise than a brute-force grid search.

        Args:
            y_true: Binary ground-truth matrix, shape (N, num_classes).
            y_probs: Predicted probabilities, shape (N, num_classes).

        Returns:
            Dict mapping class name → optimal threshold.
        """
        from sklearn.metrics import precision_recall_curve

        best: Dict[str, float] = {}

        for i, name in enumerate(self.class_names):
            precisions, recalls, thresholds = precision_recall_curve(
                y_true[:, i], y_probs[:, i],
            )
            f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = int(np.argmax(f1_scores))
            best_t = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
            best_f1 = float(f1_scores[best_idx])
            best[name] = round(best_t, 4)
            logger.info(f"  {name}: threshold={best_t:.4f}  F1={best_f1:.4f}")

        self.thresholds = best
        self._is_fitted = True
        return best

    def apply(self, y_probs: np.ndarray) -> np.ndarray:
        """Apply optimised thresholds to probability matrix.

        Args:
            y_probs: shape (N, num_classes).

        Returns:
            Binary predictions (N, num_classes).
        """
        thresholds = np.array([self.thresholds[n] for n in self.class_names])
        return (y_probs >= thresholds).astype(int)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({"thresholds": self.thresholds}, f, indent=2)
        logger.info(f"ThresholdOptimizer saved to {p}")

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.thresholds = data["thresholds"]
        self._is_fitted = True
        logger.info(f"ThresholdOptimizer loaded: {self.thresholds}")

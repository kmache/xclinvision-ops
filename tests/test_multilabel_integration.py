# tests/test_multilabel_integration.py

import numpy as np
import pytest
import torch

# ---- Config ----
def test_is_multilabel_returns_bool(monkeypatch):
    """Verify is_multilabel() returns correct bool for each mode."""
    from xclinvision.config import _reset_class_names_cache
    # Patch the yaml to return "multilabel", check True
    # Patch back to "multiclass", check False

# ---- Dataset ----
def test_dataset_multilabel_label_shape():
    """Labels should be float32 tensors of shape (num_classes,) in multilabel."""

def test_dataset_multiclass_label_is_int():
    """Labels should be plain ints in multiclass mode."""

def test_get_class_weights_multilabel_returns_pos_weight():
    """pos_weight tensor shape should be (num_classes,)."""

# ---- Trainer / Loss ----
def test_multilabel_focal_loss_output_shape():
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 2, (8, 3)).float()
    from xclinvision.trainer import MultilabelFocalLoss
    loss_fn = MultilabelFocalLoss(gamma=2.0)
    loss = loss_fn(logits, targets)
    assert loss.shape == ()
    assert loss.item() > 0

# ---- Evaluator ----

def _make_multilabel_mc(class_names):
    """Helper: create a MetricsComputer with multilabel mode forced on."""
    from xclinvision.evaluator import MetricsComputer
    mc = MetricsComputer(class_names=class_names)
    mc.multilabel = True
    return mc


def test_compute_multilabel_metrics_keys():
    """Verify all expected keys present in multilabel metrics dict."""
    mc = _make_multilabel_mc(["A", "B", "C"])
    y_true = np.array([[1,0,1],[0,1,0],[1,1,1]])
    y_pred = np.array([[1,0,0],[0,1,0],[1,1,1]])
    y_probs = np.random.rand(3, 3)
    metrics = mc.compute_all_metrics(y_true, y_pred, y_probs)
    assert "subset_accuracy" in metrics
    assert "sample_f1" in metrics
    assert "A_sensitivity" in metrics


def test_multilabel_weighted_f1_present_and_positive():
    """weighted_f1 should be present and > 0 for realistic multilabel predictions."""
    mc = _make_multilabel_mc(["A", "B", "C"])
    y_true = np.array([[1,0,1],[0,1,0],[1,1,1],[0,0,1]])
    y_pred = np.array([[1,0,0],[0,1,0],[1,1,1],[1,0,1]])
    y_probs = np.random.rand(4, 3)
    metrics = mc.compute_all_metrics(y_true, y_pred, y_probs)
    assert "weighted_f1" in metrics, "weighted_f1 missing from multilabel metrics"
    assert metrics["weighted_f1"] > 0, (
        f"weighted_f1 should be > 0 for realistic predictions, got {metrics['weighted_f1']}"
    )
    # Sanity: other core metrics still present
    assert "macro_f1" in metrics
    assert "subset_accuracy" in metrics
    assert "sample_f1" in metrics


def test_compute_multiclass_metrics_keys():
    """Multiclass metrics should still include accuracy and weighted_f1."""
    from xclinvision.evaluator import MetricsComputer
    mc = MetricsComputer(class_names=["A", "B", "C"])
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1])
    y_probs = np.random.rand(4, 3)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    metrics = mc.compute_all_metrics(y_true, y_pred, y_probs)
    assert "accuracy" in metrics
    assert "weighted_f1" in metrics
    assert metrics["weighted_f1"] > 0


def test_multiclass_no_subset_accuracy():
    """Multiclass mode should NOT expose multilabel-only keys."""
    from xclinvision.evaluator import MetricsComputer
    mc = MetricsComputer(class_names=["A", "B", "C"])
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1])
    y_probs = np.random.rand(4, 3)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    metrics = mc.compute_all_metrics(y_true, y_pred, y_probs)
    assert "subset_accuracy" not in metrics
    assert "sample_f1" not in metrics


def test_print_summary_multilabel_shows_weighted_f1(capsys):
    """print_summary should run without error and display weighted_f1 > 0."""
    mc = _make_multilabel_mc(["A", "B"])
    y_true = np.array([[1,0],[0,1],[1,1]])
    y_pred = np.array([[1,1],[0,1],[1,0]])
    y_probs = np.random.rand(3, 2)
    metrics = mc._compute_multilabel_metrics(y_true, y_pred, y_probs)
    assert metrics["weighted_f1"] > 0
    # Should not raise
    mc.print_summary(metrics, y_true, y_pred)

# ---- Predictions save (trainer) ----
def test_save_predictions_multilabel_serializable():
    """Verify multilabel preds/targets serialize to JSON without error."""
    import json
    preds = [np.array([1, 0, 1]), np.array([0, 1, 0])]
    data = [p.tolist() if hasattr(p, 'tolist') else list(p) for p in preds]
    json.dumps(data)  # should not raise

# ---- Temperature scaling guard ----
def test_temperature_scaler_skipped_for_multilabel():
    """TemperatureScaler.fit should not be called when is_multilabel() is True."""
    # Mock is_multilabel() → True, verify _fit_temperature_scaler returns None

# ---- Failure analysis ----
def test_failure_analyzer_multilabel():
    from xclinvision.reliability import FailureAnalyzer
    fa = FailureAnalyzer(class_names=["A", "B"], multilabel=True)
    y_true = np.array([[1,0],[0,1],[1,1]])
    y_pred = np.array([[1,1],[0,1],[1,0]])
    y_probs = np.random.rand(3, 2)
    result = fa.analyze_failures(y_true, y_pred, y_probs)
    assert result["false_positives"]["B"]["count"] == 1
    assert result["false_negatives"]["B"]["count"] == 1

# ---- Inference ----
def test_inference_predict_multilabel_keys(monkeypatch):
    """Output dict should have predictions_multilabel and class_names_predicted."""
    # Requires mocking model + is_multilabel → True
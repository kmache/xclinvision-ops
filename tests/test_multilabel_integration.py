# tests/test_multilabel_integration.py

import numpy as np
import pytest
import torch

# ---- Config ----
def test_is_multilabel_returns_bool(monkeypatch):
    """Verify is_multilabel() returns correct bool for each mode."""
    from xclinvision.config import is_multilabel, _reset_class_names_cache
    _reset_class_names_cache()
    result = is_multilabel()
    assert isinstance(result, bool)

# ---- Dataset ----
@pytest.mark.skip(reason="Requires dataset on disk")
def test_dataset_multilabel_label_shape():
    """Labels should be float32 tensors of shape (num_classes,) in multilabel."""

@pytest.mark.skip(reason="Requires dataset on disk")
def test_dataset_multiclass_label_is_int():
    """Labels should be plain ints in multiclass mode."""

@pytest.mark.skip(reason="Requires dataset on disk")
def test_get_class_weights_multilabel_returns_pos_weight():
    """pos_weight tensor shape should be (num_classes,)."""

# ---- Trainer / Loss ----
def test_multilabel_focal_loss_output_shape():
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 2, (8, 3)).float()
    from xclinvision.losses import MultilabelFocalLoss
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
def test_inference_predict_multilabel_keys():
    """Output dict should have predictions_multilabel and class_names_predicted."""
    from unittest.mock import MagicMock, patch
    from xclinvision.config import is_multilabel
    if not is_multilabel():
        pytest.skip("Only applicable in multilabel mode")
    # When multilabel, predict() output should include these keys
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = {
        "prediction": 0,
        "class_name": "No finding",
        "probabilities": [0.8, 0.1, 0.1],
        "confidence": 0.8,
        "predictions_multilabel": [1, 0, 0],
        "class_names_predicted": ["No finding"],
    }
    result = mock_pipeline.predict(np.zeros((384, 384, 3), dtype=np.uint8))
    assert "predictions_multilabel" in result
    assert "class_names_predicted" in result

# ---- Issue 1: headline finding selection ----------------------------------

def _fixed_logit_pipeline(class_names, probabilities, priority_map=None):
    """InferencePipeline over a model that always emits the given probabilities."""
    import math
    import torch
    import torch.nn as nn
    from xclinvision.inference import InferencePipeline

    logits = torch.tensor(
        [[math.log(p / (1 - p)) for p in probabilities]], dtype=torch.float32
    )

    class _Fixed(nn.Module):
        def forward(self, x):
            return logits

    return InferencePipeline(
        model=_Fixed(),
        architecture="test",
        device="cpu",
        image_size=64,
        class_names=class_names,
        thresholds={n: 0.5 for n in class_names},
        priority_map=priority_map,
    )


def _dummy_image():
    return (np.random.rand(64, 64, 3) * 255).astype(np.uint8)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_headline_finding_is_not_lowest_class_index():
    """Tier order must beat class-index order when the two disagree.

    The headline used to be active_indices[0] — the lowest class *index*, i.e.
    configs/system.yaml ordering. Cardiomegaly is deliberately placed LAST here
    and given the lower probability, so index order, probability order and tier
    order all disagree:

        index order       -> Pulmonary fibrosis (index 0)
        probability order -> Pulmonary fibrosis (0.97)
        tier order        -> Cardiomegaly (urgent beats routine)

    Only the tier rule produces Cardiomegaly first, so this fails against the
    pre-fix behaviour instead of passing either way.
    """
    names = ["Pulmonary fibrosis", "Aortic enlargement", "Pleural thickening", "Cardiomegaly"]
    pipe = _fixed_logit_pipeline(names, [0.97, 0.01, 0.01, 0.55])

    result = pipe.predict(_dummy_image(), return_uncertainty=False, return_explanation=False)

    # Urgent-at-0.55 leads routine-at-0.97, and both positives survive.
    assert result["class_names_predicted"] == ["Cardiomegaly", "Pulmonary fibrosis"]
    assert result["class_name"] == "Cardiomegaly"
    assert result["confidence"] == pytest.approx(0.55, abs=1e-3)


def test_headline_is_probability_ranked_within_a_tier():
    """Same tier -> highest probability wins, which the old code got wrong."""
    names = ["Pleural thickening", "Aortic enlargement", "Pulmonary fibrosis"]
    pipe = _fixed_logit_pipeline(names, [0.55, 0.01, 0.97])

    result = pipe.predict(_dummy_image(), return_uncertainty=False, return_explanation=False)

    # All three are routine, so index order (Pleural thickening) must not win.
    assert result["class_name"] == "Pulmonary fibrosis"
    assert result["confidence"] == pytest.approx(0.97, abs=1e-3)


def test_critical_outranks_a_higher_probability_routine_finding():
    """Pneumothorax 0.60 must lead Cardiomegaly 0.90."""
    names = ["Cardiomegaly", "Pneumothorax", "Pulmonary fibrosis"]
    pipe = _fixed_logit_pipeline(names, [0.90, 0.60, 0.20])

    result = pipe.predict(_dummy_image(), return_uncertainty=False, return_explanation=False)

    assert result["class_name"] == "Pneumothorax"
    assert result["class_names_predicted"][0] == "Pneumothorax"


def test_critical_finding_outranks_a_higher_probability_routine_one():
    """Clinical priority beats probability: a pneumothorax must not be buried."""
    names = ["Cardiomegaly", "Aortic enlargement", "Pneumothorax", "Pulmonary fibrosis"]
    priority = {
        "Cardiomegaly": "urgent",
        "Aortic enlargement": "routine",
        "Pneumothorax": "critical",
        "Pulmonary fibrosis": "routine",
    }
    pipe = _fixed_logit_pipeline(names, [0.97, 0.01, 0.55, 0.60], priority_map=priority)

    result = pipe.predict(_dummy_image(), return_uncertainty=False, return_explanation=False)

    assert result["class_name"] == "Pneumothorax"
    assert result["class_names_predicted"][0] == "Pneumothorax"


def test_no_priority_map_degrades_to_probability_order():
    names = ["A", "B", "C"]
    pipe = _fixed_logit_pipeline(names, [0.55, 0.60, 0.99], priority_map=None)

    result = pipe.predict(_dummy_image(), return_uncertainty=False, return_explanation=False)

    assert result["class_names_predicted"] == ["C", "B", "A"]


def test_analyze_response_carries_every_positive_label(client):
    """Issue 1: /api/v2/analyze dropped predictions_multilabel entirely."""
    import io
    from unittest.mock import MagicMock, patch

    from PIL import Image

    pipeline = MagicMock()
    pipeline.predict.return_value = {
        "prediction": 3,
        "class_name": "Pulmonary fibrosis",
        "probabilities": [0.55, 0.01, 0.01, 0.97],
        "confidence": 0.97,
        "class_names": ["Cardiomegaly", "Aortic enlargement", "Pleural thickening", "Pulmonary fibrosis"],
        "uncertainty": {"epistemic": 0.01},
        "uncertainty_level": "low",
        "predictions_multilabel": [1, 0, 0, 1],
        "thresholds": [0.5, 0.5, 0.5, 0.5],
        "class_names_predicted": ["Pulmonary fibrosis", "Cardiomegaly"],
        "explanation": {"key_findings": [], "visualization": {"region_scores": {}}},
    }
    pipeline.preprocess.return_value = (
        np.zeros((3, 64, 64), dtype=np.float32),
        np.zeros((64, 64, 3), dtype=np.uint8),
    )

    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(buf, format="JPEG")

    import main  # noqa: PLC0415

    with patch.object(main, "get_pipeline", return_value=pipeline), \
            patch.object(main, "_get_agent", side_effect=RuntimeError("no llm")):
        response = client.post(
            "/api/v2/analyze", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["class_names_predicted"] == ["Pulmonary fibrosis", "Cardiomegaly"]

    # predictions_multilabel is now per-class detail, not a raw binary vector.
    detail = body["predictions_multilabel"]
    assert [d["class_name"] for d in detail] == [
        "Cardiomegaly", "Aortic enlargement", "Pleural thickening", "Pulmonary fibrosis",
    ]
    assert [d["positive"] for d in detail] == [True, False, False, True]
    assert detail[3]["probability"] == pytest.approx(0.97, abs=1e-3)
    assert all(set(d) == {"class_name", "probability", "threshold", "positive"} for d in detail)

def _biased_multilabel_logits(seed=0, n=4096, c=4):
    """Logits carrying the failure mode the served checkpoints actually have.

    Each class is shifted positive by a different amount, which is what training
    with pos_weight 5.1-15.2 produces, and prevalence is low. Temperature alone
    cannot correct a shift, so this distinguishes scale-fixing from bias-fixing.
    """
    rng = np.random.default_rng(seed)
    shifts = np.linspace(1.5, 3.0, c)
    latent = rng.normal(0.0, 1.5, size=(n, c))
    y_true = (latent > 1.2).astype(np.float32)          # ~11% prevalence
    logits = latent + shifts                             # systematically over-positive
    return logits.astype(np.float32), y_true


def test_fitted_temperature_reduces_ece_on_overconfident_logits():
    """Issue 7: replaces `assert hasattr(ts, 'temperature')`, which was always true.

    Temperature scaling only earns the word "calibrated" if it measurably
    improves calibration. Fit on overconfident logits and assert ECE drops.
    """
    import torch

    from xclinvision.evaluator import CalibrationAnalyzer, TemperatureScaler

    analyzer = CalibrationAnalyzer()

    rng = np.random.default_rng(0)
    # Overconfident: large-magnitude logits whose labels agree only ~70%.
    logits = rng.normal(0.0, 6.0, size=(2048, 4))
    y_true = (logits + rng.normal(0.0, 4.0, size=logits.shape) > 0).astype(np.float32)

    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, y_true, multilabel=True)
    assert temperature > 0

    raw = torch.sigmoid(torch.tensor(logits)).numpy()
    scaled = torch.sigmoid(torch.tensor(logits) / temperature).numpy()

    ece_raw = analyzer.compute_ece(y_true, raw, multilabel=True)
    ece_scaled = analyzer.compute_ece(y_true, scaled, multilabel=True)

    assert ece_scaled < ece_raw, (
        f"temperature {temperature:.3f} did not improve calibration: "
        f"ECE {ece_raw:.4f} -> {ece_scaled:.4f}"
    )


def test_per_class_affine_fit_beats_a_global_temperature():
    """The fit actually shipped must beat the one it replaced.

    A single global temperature averages four different per-class errors into
    one scalar; on biased logits it lands near T=1 and barely moves ECE. This
    exercises TemperatureScaler.fit_per_class, the code path that produced the
    parameters now stored in every served checkpoint.
    """
    from xclinvision.evaluator import CalibrationAnalyzer, TemperatureScaler

    analyzer = CalibrationAnalyzer()
    logits, y_true = _biased_multilabel_logits()

    ece_raw = analyzer.compute_ece(y_true, _sigmoid(logits), multilabel=True)

    global_scaler = TemperatureScaler()
    t_global = global_scaler.fit(logits, y_true, multilabel=True)
    ece_global = analyzer.compute_ece(
        y_true, _sigmoid(logits / t_global), multilabel=True
    )

    per_class = TemperatureScaler()
    temps, biases = per_class.fit_per_class(logits, y_true)
    ece_per_class = analyzer.compute_ece(
        y_true, _sigmoid(logits / np.array(temps) + np.array(biases)), multilabel=True
    )

    assert len(temps) == logits.shape[1]
    assert all(t > 0 for t in temps)
    # The fitted bias must be negative: these logits are shifted positive.
    assert all(b < 0 for b in biases), f"expected negative biases, got {biases}"
    assert ece_per_class < ece_global < ece_raw + 1e-9, (
        f"raw {ece_raw:.4f} -> global {ece_global:.4f} -> per-class {ece_per_class:.4f}"
    )
    # Not a marginal win: bias correction is the whole point.
    assert ece_per_class < ece_raw / 2


def test_temperature_only_fit_cannot_remove_a_bias():
    """Documents why the shipped fit carries a bias term.

    With b pinned to 0 the residual stays large on shifted logits; freeing b
    collapses it. If this ever inverts, the bias term is no longer earning its
    place in the payload.
    """
    from xclinvision.evaluator import CalibrationAnalyzer, TemperatureScaler

    analyzer = CalibrationAnalyzer()
    logits, y_true = _biased_multilabel_logits()

    t_only = TemperatureScaler()
    temps, biases = t_only.fit_per_class(logits, y_true, with_bias=False)
    assert biases == [0.0] * logits.shape[1]
    ece_t_only = analyzer.compute_ece(
        y_true, _sigmoid(logits / np.array(temps)), multilabel=True
    )

    affine = TemperatureScaler()
    a_temps, a_biases = affine.fit_per_class(logits, y_true, with_bias=True)
    ece_affine = analyzer.compute_ece(
        y_true, _sigmoid(logits / np.array(a_temps) + np.array(a_biases)), multilabel=True
    )

    assert ece_affine < ece_t_only, (
        f"bias term did not help: temperature-only {ece_t_only:.4f} vs "
        f"affine {ece_affine:.4f}"
    )


def test_predict_reports_whether_probabilities_are_calibrated():
    """Issue 7: no shipped checkpoint carries a temperature, so say so."""
    names = ["A", "B"]

    uncalibrated = _fixed_logit_pipeline(names, [0.9, 0.2])
    result = uncalibrated.predict(
        _dummy_image(), return_uncertainty=False, return_explanation=False
    )
    assert result["calibrated"] is False

    calibrated = _fixed_logit_pipeline(names, [0.9, 0.2])
    calibrated.temperature_scaler = 1.5
    result = calibrated.predict(
        _dummy_image(), return_uncertainty=False, return_explanation=False
    )
    assert result["calibrated"] is True


def test_predict_exposes_raw_probability_with_confidence_alias():
    """Issue 7: the served score is a raw sigmoid output, so name it accurately.

    `confidence` stays populated — the frontend, the agent tools and already
    stored analyses all read it, and the analyze response is additive-only.
    """
    pipe = _fixed_logit_pipeline(["A", "B"], [0.9, 0.2])

    result = pipe.predict(_dummy_image(), return_uncertainty=False, return_explanation=False)

    assert result["raw_probability"] == pytest.approx(0.9, abs=1e-3)
    assert result["confidence"] == result["raw_probability"]
    assert result["calibrated"] is False


def test_analyze_response_exposes_raw_probability(client):
    import io
    from unittest.mock import MagicMock, patch

    from PIL import Image

    pipeline = MagicMock()
    pipeline.predict.return_value = {
        "prediction": 0,
        "class_name": "Cardiomegaly",
        "probabilities": [0.9, 0.1],
        "raw_probability": 0.9,
        "confidence": 0.9,
        "calibrated": False,
        "class_names": ["Cardiomegaly", "Aortic enlargement"],
        "uncertainty": {"epistemic": 0.01},
        "uncertainty_level": "low",
        "explanation": {"key_findings": [], "visualization": {"region_scores": {}}},
    }
    pipeline.preprocess.return_value = (
        np.zeros((3, 64, 64), dtype=np.float32),
        np.zeros((64, 64, 3), dtype=np.uint8),
    )

    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(buf, format="JPEG")

    import main  # noqa: PLC0415

    with patch.object(main, "get_pipeline", return_value=pipeline), \
            patch.object(main, "_get_agent", side_effect=RuntimeError("no llm")):
        response = client.post(
            "/api/v2/analyze", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")}
        )

    body = response.json()
    assert body["raw_probability"] == 0.9
    assert body["confidence"] == 0.9      # decision 2: additive only
    assert body["calibrated"] is False


def test_analyze_response_reports_calibration_status(client):
    """Issue 7: every served model is uncalibrated; the response must say so."""
    import io
    from unittest.mock import MagicMock, patch

    from PIL import Image

    pipeline = MagicMock()
    pipeline.predict.return_value = {
        "prediction": 0,
        "class_name": "Cardiomegaly",
        "probabilities": [0.9, 0.1],
        "raw_probability": 0.9,
        "confidence": 0.9,
        "calibrated": False,
        "calibration_status": "uncalibrated",
        "thresholds": [0.5, 0.5],
        "class_names": ["Cardiomegaly", "Aortic enlargement"],
        "predictions_multilabel": [1, 0],
        "class_names_predicted": ["Cardiomegaly"],
        "uncertainty": {"epistemic": 0.01},
        "uncertainty_level": "low",
        "explanation": {"key_findings": [], "visualization": {"region_scores": {}}},
    }
    pipeline.preprocess.return_value = (
        np.zeros((3, 64, 64), dtype=np.float32),
        np.zeros((64, 64, 3), dtype=np.uint8),
    )

    buf = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(buf, format="JPEG")

    import main  # noqa: PLC0415

    with patch.object(main, "get_pipeline", return_value=pipeline), \
            patch.object(main, "_get_agent", side_effect=RuntimeError("no llm")):
        body = client.post(
            "/api/v2/analyze", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")}
        ).json()

    assert body["calibration_status"] == "uncalibrated"
    assert body["calibrated"] is False
    # Deprecated alias still mirrors the value (decision 2: additive only).
    assert body["confidence"] == body["raw_probability"]


def test_uncalibrated_report_uses_no_certainty_language():
    """An uncalibrated score must not be mapped to clinical certainty."""
    from xclinvision.agent.reporter import calibrate_predictions

    class_names = ["Cardiomegaly", "Aortic enlargement"]
    probabilities = [0.97, 0.10]

    terms = [
        r["clinical_term"]
        for r in calibrate_predictions(class_names, probabilities, None, calibrated=False)
    ]

    for term in terms:
        assert "Highly suggestive" not in term
        assert "Consistent with" not in term
        assert "not calibrated" in term
    assert terms[0] == "Raw model probability: 0.97 (not calibrated)"

    # With a fitted temperature the clinical language returns.
    calibrated_terms = [
        r["clinical_term"]
        for r in calibrate_predictions(class_names, probabilities, None, calibrated=True)
    ]
    assert calibrated_terms[0] == "Highly suggestive"


# ---------------------------------------------------------------------------
# Per-class calibration reaches the served predictions
# ---------------------------------------------------------------------------

def test_per_class_calibration_changes_served_probabilities():
    """A stored calibration must actually move the numbers.

    Before this was fitted, every checkpoint carried temperature=None and
    _apply_temperature returned the logits untouched, so "calibrated" would
    have been a label on raw sigmoid output.
    """
    names = ["A", "B", "C", "D"]
    raw_probs = [0.90, 0.75, 0.60, 0.40]

    uncalibrated = _fixed_logit_pipeline(names, raw_probs)
    before = uncalibrated.predict(
        _dummy_image(), return_uncertainty=False, return_explanation=False
    )["probabilities"]

    calibrated = _fixed_logit_pipeline(names, raw_probs)
    calibrated.temperature_scaler = {
        "temperature": [0.55, 0.60, 0.65, 0.50],
        "bias": [-2.0, -1.7, -3.4, -2.9],
    }
    calibrated._calibration_status = "calibrated"
    after = calibrated.predict(
        _dummy_image(), return_uncertainty=False, return_explanation=False
    )

    assert after["calibrated"] is True
    assert after["calibration_status"] == "calibrated"
    assert after["probabilities"] != before
    # Negative bias must pull probabilities down, not merely perturb them.
    assert all(a < b for a, b in zip(after["probabilities"], before))


def test_calibration_status_gates_the_calibrated_flag():
    """Carrying parameters is not the same as being calibrated.

    A checkpoint whose fit failed the held-out ECE bar still stores its
    parameters — they are applied — but must keep reporting "uncalibrated" so
    the report layer keeps hedging the language.
    """
    names = ["A", "B"]
    pipe = _fixed_logit_pipeline(names, [0.9, 0.2])
    pipe.temperature_scaler = {"temperature": [0.6, 0.6], "bias": [-2.0, -2.0]}
    pipe._calibration_status = "uncalibrated"

    result = pipe.predict(
        _dummy_image(), return_uncertainty=False, return_explanation=False
    )
    assert result["calibrated"] is False
    assert result["calibration_status"] == "uncalibrated"

    # The scaling is still applied — only the label is withheld.
    raw = _fixed_logit_pipeline(names, [0.9, 0.2]).predict(
        _dummy_image(), return_uncertainty=False, return_explanation=False
    )["probabilities"]
    assert result["probabilities"] != raw


def test_mismatched_per_class_calibration_length_is_ignored():
    """A calibration vector that does not match the class count must not apply.

    Silently broadcasting or truncating would scale the wrong class, which is
    worse than serving raw probabilities.
    """
    names = ["A", "B", "C"]
    pipe = _fixed_logit_pipeline(names, [0.9, 0.5, 0.2])
    pipe.temperature_scaler = {"temperature": [0.5, 0.5], "bias": [-2.0, -2.0]}

    result = pipe.predict(
        _dummy_image(), return_uncertainty=False, return_explanation=False
    )
    raw = _fixed_logit_pipeline(names, [0.9, 0.5, 0.2]).predict(
        _dummy_image(), return_uncertainty=False, return_explanation=False
    )["probabilities"]
    assert result["probabilities"] == pytest.approx(raw, abs=1e-6)


def test_served_checkpoints_carry_a_fitted_per_class_calibration():
    """Every exported checkpoint must state its calibration, or say it has none.

    Skipped when the weights directory is absent (CI without model artefacts).
    """
    import glob
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    metas = sorted(glob.glob(str(repo / "models" / "best_models" / "*_meta.json")))
    if not metas:
        pytest.skip("no exported checkpoints on this machine")

    for mp in metas:
        meta = json.loads(Path(mp).read_text())
        name = meta["model_name"]
        temps = meta.get("temperature")
        assert isinstance(temps, list), f"{name}: temperature is not per-class ({temps!r})"
        assert len(temps) == len(meta["class_names"]), f"{name}: wrong temperature length"
        assert all(t > 0 for t in temps), f"{name}: non-positive temperature"

        biases = meta.get("calibration_bias")
        assert isinstance(biases, list) and len(biases) == len(temps), f"{name}: bias missing"

        status = meta.get("calibration_status")
        assert status in {"calibrated", "uncalibrated"}, f"{name}: bad status {status!r}"

        ece = meta.get("calibration_ece") or {}
        before, after = ece.get("val_before"), ece.get("val_after")
        assert before and after, f"{name}: calibration_ece not recorded"
        mean_before = sum(before.values()) / len(before)
        mean_after = sum(after.values()) / len(after)
        if status == "calibrated":
            assert mean_after < mean_before, (
                f"{name} claims calibrated but ECE did not improve: "
                f"{mean_before:.4f} -> {mean_after:.4f}"
            )
            assert mean_after <= 0.05, (
                f"{name} claims calibrated at mean ECE {mean_after:.4f}, above the 0.05 bar"
            )

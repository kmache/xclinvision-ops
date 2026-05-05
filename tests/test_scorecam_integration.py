"""Tests for Score-CAM integration across the full stack.

Validates that the ``scorecam`` XAI method is properly threaded through
the inference pipeline, backend API, and frontend schema.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "app" / "backend"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    from main import app as _app
    return _app


@pytest.fixture(scope="session")
def client(app):
    from starlette.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def dummy_image_bytes() -> bytes:
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _fake_pipeline(method: str = "gradcam++"):
    """Return a mock InferencePipeline with realistic predict() output."""
    pipeline = MagicMock()

    def _predict(
        image,
        return_uncertainty=True,
        return_explanation=True,
        xai_method="gradcam++",
        target_class=None,
    ):
        return {
            "prediction": 0,
            "class_name": "No finding",
            "probabilities": [0.85, 0.05, 0.04, 0.03, 0.03],
            "confidence": 0.85,
            "uncertainty": {"epistemic": 0.02, "aleatoric": 0.01},
            "uncertainty_level": "low",
            "explanation": {
                "key_findings": ["Normal cardiac silhouette"],
                "clinical_plausibility": 0.9,
                "visualization": {
                    "grayscale_cam": np.random.rand(64, 64).astype(np.float32),
                    "region_scores": {"left_lung": 0.3, "right_lung": 0.25},
                    "method": xai_method,
                },
            },
            "predictions_multilabel": [1, 0, 0, 0, 0],
            "class_names_predicted": ["No finding"],
        }

    pipeline.predict.side_effect = _predict
    pipeline.preprocess.return_value = (
        np.random.rand(3, 384, 384).astype(np.float32),
        np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8),
    )
    return pipeline


# ===========================================================================
# Schema validation
# ===========================================================================

class TestSchemaScoreCAM:
    def test_explanation_params_accepts_scorecam(self):
        from schemas import ExplanationParams  # type: ignore[import-not-found]
        params = ExplanationParams(analysis_id="XCL-TEST", method="scorecam")
        assert params.method == "scorecam"

    def test_explanation_params_rejects_invalid(self):
        from schemas import ExplanationParams  # type: ignore[import-not-found]
        with pytest.raises(Exception):
            ExplanationParams(analysis_id="XCL-TEST", method="invalid_method")


# ===========================================================================
# Pipeline method threading
# ===========================================================================

class TestPipelineMethodThreading:
    def test_generate_explanation_accepts_method(self):
        """generate_explanation() should accept and forward a method kwarg."""
        from xclinvision.xai import generate_explanation

        import inspect
        sig = inspect.signature(generate_explanation)
        assert "method" in sig.parameters
        assert sig.parameters["method"].default == "gradcam++"

    def test_inference_predict_accepts_xai_method(self):
        """InferencePipeline.predict() should accept xai_method kwarg."""
        from xclinvision.inference import InferencePipeline

        import inspect
        sig = inspect.signature(InferencePipeline.predict)
        assert "xai_method" in sig.parameters
        assert sig.parameters["xai_method"].default == "gradcam++"

    def test_stateless_predict_accepts_xai_method(self):
        """Top-level predict() should accept xai_method kwarg."""
        from xclinvision.inference import predict

        import inspect
        sig = inspect.signature(predict)
        assert "xai_method" in sig.parameters


# ===========================================================================
# API endpoint tests
# ===========================================================================

class TestAnalyzeWithScoreCAM:
    @patch("main._get_agent")
    @patch("main.get_pipeline")
    def test_analyze_with_scorecam_method(self, mock_get_pipe, mock_get_agent, client, dummy_image_bytes):
        pipeline = _fake_pipeline()
        mock_get_pipe.return_value = pipeline
        mock_agent = MagicMock()
        mock_agent.generate_report.return_value = {
            "findings": "Normal exam.",
            "impression": "No acute findings.",
        }
        mock_get_agent.return_value = mock_agent

        r = client.post(
            "/api/v2/analyze",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
            data={
                "patient_id": "TEST-SCORECAM",
                "model_name": "convnext_small",
                "xai_method": "scorecam",
            },
        )
        assert r.status_code == 200

        # Verify the pipeline was called with xai_method="scorecam"
        pipeline.predict.assert_called_once()
        call_kwargs = pipeline.predict.call_args
        assert call_kwargs.kwargs.get("xai_method") == "scorecam" or \
               (len(call_kwargs.args) > 0 and "scorecam" in str(call_kwargs))


class TestExplainV2WithScoreCAM:
    @patch("main.get_pipeline")
    def test_explain_with_scorecam(self, mock_get_pipe, client, dummy_image_bytes):
        pipeline = _fake_pipeline()
        mock_get_pipe.return_value = pipeline

        # First, create an analysis in the store
        from main import _analysis_store, _image_store_put  # type: ignore[import-not-found]
        analysis_id = "XCL-SCORECAM-TEST"
        _analysis_store[analysis_id] = {
            "analysis_id": analysis_id,
            "prediction": "No finding",
            "confidence": 0.85,
            "model_version": "convnext_small",
            "heatmap_gradcam": None,
            "heatmap_overlay": None,
            "region_scores": {},
        }
        _image_store_put(analysis_id, dummy_image_bytes)

        r = client.get(
            f"/api/v2/explain/{analysis_id}",
            params={"method": "scorecam", "threshold": 0.5, "opacity": 0.6},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["method"] == "scorecam"

        # Verify predict was called with xai_method="scorecam"
        pipeline.predict.assert_called_once()
        _, kwargs = pipeline.predict.call_args
        assert kwargs.get("xai_method") == "scorecam"

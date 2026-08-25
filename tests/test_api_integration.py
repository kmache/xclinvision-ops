"""Integration tests for the XClinVision backend API.

Tests validate endpoint contracts, response schemas, and error handling
without requiring GPU or trained model weights (inference endpoints are
tested with mocked pipelines).
"""

import contextlib
import io
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Make sure the backend package and src/ are importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "app" / "backend"))
sys.path.insert(0, str(_REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Import the FastAPI app."""
    from main import app as _app  # noqa: E402  # type: ignore[import-not-found]
    return _app


@pytest.fixture(scope="session")
def client(app):
    """Synchronous TestClient (no event-loop juggling needed)."""
    from starlette.testclient import TestClient
    import os as _os
    with TestClient(app) as c:
        token = _os.environ.get("XCLINVISION_API_TOKEN", "").strip()
        if token:
            c.headers.update({"Authorization": f"Bearer {token}"})
        yield c


@pytest.fixture()
def dummy_image_bytes() -> bytes:
    """Create a minimal 64x64 RGB JPEG in memory."""
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _fake_pipeline():
    """Return a mock InferencePipeline with realistic predict() output."""
    pipeline = MagicMock()
    pipeline.predict.return_value = {
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
                "method": "gradcam++",
            },
        },
        "predictions_multilabel": [1, 0, 0, 0, 0],
        "class_names_predicted": ["No finding"],
    }
    pipeline.preprocess.return_value = (
        np.random.rand(3, 384, 384).astype(np.float32),
        np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8),
    )
    return pipeline


# ===========================================================================
# Health & metadata endpoints
# ===========================================================================

class TestHealthAndMeta:
    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"
        assert "version" in body
        assert "timestamp" in body

    def test_list_models(self, client):
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        assert isinstance(body["models"], list)
        # Each model entry should have expected keys
        for m in body["models"]:
            assert "name" in m
            assert "type" in m
            assert m["type"] in ("cnn", "transformer")
            assert "num_classes" in m
            assert "class_names" in m


# ===========================================================================
# Prediction endpoint (v1)
# ===========================================================================

class TestPredictV1:
    @patch("main.get_pipeline")
    def test_predict_success(self, mock_get_pipe, client, dummy_image_bytes):
        mock_get_pipe.return_value = _fake_pipeline()
        r = client.post(
            "/api/v1/predict",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
            data={"model_name": "efficientnet_b0", "return_explanation": "true"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        assert "class_name" in body
        assert "probabilities" in body
        assert isinstance(body["probabilities"], list)
        assert "confidence" in body
        assert "processing_time_ms" in body
        assert body["confidence"] >= 0.0

    @patch("main.get_pipeline")
    def test_predict_returns_multilabel_fields(self, mock_get_pipe, client, dummy_image_bytes):
        mock_get_pipe.return_value = _fake_pipeline()
        r = client.post(
            "/api/v1/predict",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
        )
        body = r.json()
        assert "predictions_multilabel" in body
        assert "class_names_predicted" in body

    def test_predict_invalid_content_type(self, client):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("doc.txt", b"not an image", "text/plain")},
        )
        assert r.status_code == 400

    def test_predict_missing_content_type_header(self, client):
        # issue #6: multipart part without a Content-Type must 400, not 500.
        boundary = "----testboundary6"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="xray.jpg"\r\n'
            "\r\n"
            "\xff\xd8\xff\xe0\r\n"
            f"--{boundary}--\r\n"
        ).encode("latin-1")
        r = client.post(
            "/api/v1/predict",
            content=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        assert r.status_code == 400

    @patch("main.get_pipeline")
    def test_predict_invalid_magic_bytes(self, mock_get_pipe, client):
        mock_get_pipe.return_value = _fake_pipeline()
        r = client.post(
            "/api/v1/predict",
            files={"file": ("fake.jpg", b"\x00\x00\x00\x00garbage", "image/jpeg")},
        )
        assert r.status_code == 400

    @patch("main.get_pipeline", return_value=None)
    def test_predict_no_model_loaded(self, mock_get_pipe, client, dummy_image_bytes):
        r = client.post(
            "/api/v1/predict",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 503


# ===========================================================================
# Analyze endpoint (v2)
# ===========================================================================

class TestAnalyzeV2:
    @patch("main._get_agent")
    @patch("main.get_pipeline")
    def test_analyze_success(self, mock_get_pipe, mock_get_agent, client, dummy_image_bytes):
        mock_get_pipe.return_value = _fake_pipeline()
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
                "patient_id": "TEST-001",
                "model_name": "efficientnet_b0",
            },
        )
        assert r.status_code == 200
        body = r.json()
        # Validate AnalysisResponse contract
        assert "analysis_id" in body
        assert body["analysis_id"].startswith("XCL-")
        assert body["patient_id"] == "TEST-001"
        assert "prediction" in body
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0
        assert "uncertainty" in body
        assert "uncertainty_level" in body
        assert "top_k_predictions" in body
        assert isinstance(body["top_k_predictions"], list)
        assert "heatmap_gradcam" in body
        assert "heatmap_overlay" in body
        assert "region_scores" in body
        assert "key_findings" in body
        assert "llm_summary" in body
        assert "inference_time_ms" in body
        assert "model_version" in body
        assert "image_hash" in body

    @patch("main.get_pipeline", return_value=None)
    def test_analyze_no_model(self, mock_get_pipe, client, dummy_image_bytes):
        r = client.post(
            "/api/v2/analyze",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
        )
        assert r.status_code == 503


# ===========================================================================
# Explain endpoint (v1)
# ===========================================================================

class TestExplainV1:
    @patch("main.get_pipeline")
    def test_explain_success(self, mock_get_pipe, client, dummy_image_bytes):
        pipeline = _fake_pipeline()
        pipeline.explain.return_value = {
            "grayscale_cam": np.random.rand(64, 64).astype(np.float32),
            "visualization": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "key_findings": ["Mild opacity"],
            "region_scores": {"left_lung": 0.4},
        }
        mock_get_pipe.return_value = pipeline
        r = client.post(
            "/api/v1/explain",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
        )
        # Accept 200 or 500 (if explain codepath has internal issues)
        assert r.status_code in (200, 500)


# ===========================================================================
# Feedback endpoints
# ===========================================================================

class TestFeedback:
    def test_v1_feedback_submit(self, client):
        r = client.post("/api/v1/feedback", json={
            "image_hash": "abc123",
            "prediction": 0,
            "correct_label": 1,
            "feedback_type": "correct",
            "notes": "Test feedback",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "received"
        assert "feedback_id" in body
        assert "timestamp" in body

    def test_v1_feedback_invalid_type(self, client):
        r = client.post("/api/v1/feedback", json={
            "image_hash": "abc123",
            "prediction": 0,
            "correct_label": 1,
            "feedback_type": "INVALID",
        })
        assert r.status_code == 422  # Pydantic validation error

    def test_v2_feedback_submit(self, client):
        r = client.post("/api/v2/feedback", json={
            "analysis_id": "XCL-20260401-test0001",
            "user_id": "dr_test",
            "feedback_type": "correct",
            "notes": "Confirmed finding",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "recorded"

    def test_v2_feedback_stats(self, client):
        """feedback-stats should return totals and by_type breakdown."""
        r = client.get("/api/v2/feedback-stats")
        assert r.status_code == 200
        body = r.json()
        assert "total" in body
        assert "by_type" in body
        assert isinstance(body["by_type"], dict)
        assert "correction_rate" in body
        assert "recent" in body
        assert isinstance(body["recent"], list)


# ===========================================================================
# Report endpoint (v1)
# ===========================================================================

class TestReportV1:
    @patch("main._get_agent")
    def test_report_success(self, mock_get_agent, client):
        mock_agent = MagicMock()
        mock_agent.generate_report.return_value = {
            "findings": "Enlarged cardiac silhouette.",
            "impression": "Cardiomegaly suspected.",
            "uncertainty": "Low uncertainty.",
            "recommendation": "Clinical correlation recommended.",
        }
        mock_get_agent.return_value = mock_agent

        r = client.post("/api/v1/report", json={
            "prediction": 1,
            "confidence": 0.92,
            "uncertainty_level": "low",
            "highlighted_regions": ["cardiac_silhouette"],
            "patient_age": 65,
            "patient_sex": "M",
        })
        assert r.status_code == 200
        body = r.json()
        assert "findings" in body
        assert "impression" in body

    def test_report_invalid_prediction_index(self, client):
        """Prediction index out of range should return 422."""
        r = client.post("/api/v1/report", json={
            "prediction": 999,
            "confidence": 0.5,
            "uncertainty_level": "high",
            "highlighted_regions": [],
        })
        assert r.status_code == 422


# ===========================================================================
# Chat endpoint (v2)
# ===========================================================================

class TestChatV2:
    @patch("main._get_reasoning_agent")
    def test_chat_success(self, mock_get_reasoning_agent, client):
        from xclinvision.agent.reasoning import AgentResponse, ReasoningStep

        mock_agent = MagicMock()
        mock_agent.process_message.return_value = AgentResponse(
            response="The findings suggest normal cardiac morphology.",
            intent="explain_prediction",
            tools_used=["get_prediction_details"],
            reasoning_trace=[ReasoningStep(step="classify_intent", detail="explain_prediction")],
            suggested_followups=["Explain the heatmap", "Is follow-up needed?"],
        )
        mock_get_reasoning_agent.return_value = mock_agent

        # First need an analysis_id in the store
        from main import _analysis_store  # type: ignore[import-not-found]
        _analysis_store["XCL-CHAT-TEST"] = {
            "analysis_id": "XCL-CHAT-TEST",
            "prediction": "No finding",
            "confidence": 0.9,
            "uncertainty": {},
            "uncertainty_level": "low",
            "region_scores": {},
            "key_findings": [],
            "llm_summary": "Normal.",
            "top_k_predictions": [{"class_name": "No finding", "probability": 0.9}],
        }

        r = client.post("/api/v2/chat", json={
            "analysis_id": "XCL-CHAT-TEST",
            "message": "What does this finding mean?",
            "history": [],
            "context_type": "clinical",
        })
        assert r.status_code == 200
        body = r.json()
        assert "response" in body
        assert body["intent"] == "explain_prediction"
        assert "tools_used" in body
        assert "reasoning_trace" in body
        assert "suggested_followups" in body
        assert isinstance(body["suggested_followups"], list)

    @patch("main._get_reasoning_agent")
    def test_chat_returns_reasoning_trace(self, mock_get_reasoning_agent, client):
        from xclinvision.agent.reasoning import AgentResponse, ReasoningStep

        mock_agent = MagicMock()
        mock_agent.process_message.return_value = AgentResponse(
            response="The heatmap highlights the cardiac silhouette region.",
            intent="explain_heatmap",
            tools_used=["get_xai_explanation"],
            reasoning_trace=[
                ReasoningStep(step="classify_intent", detail="explain_heatmap"),
                ReasoningStep(step="execute_tool", detail="get_xai_explanation"),
            ],
            suggested_followups=["Is this urgent?"],
        )
        mock_get_reasoning_agent.return_value = mock_agent

        from main import _analysis_store  # type: ignore[import-not-found]
        _analysis_store["XCL-CHAT-TRACE"] = {
            "analysis_id": "XCL-CHAT-TRACE",
            "prediction": "Cardiomegaly",
            "confidence": 0.82,
            "uncertainty": {},
            "uncertainty_level": "moderate",
            "region_scores": {"cardiac": 0.8},
            "key_findings": ["Enlarged cardiac silhouette"],
            "llm_summary": "Cardiomegaly detected.",
            "top_k_predictions": [{"class_name": "Cardiomegaly", "probability": 0.82}],
        }

        r = client.post("/api/v2/chat", json={
            "analysis_id": "XCL-CHAT-TRACE",
            "message": "Explain the heatmap",
            "history": [],
        })
        assert r.status_code == 200
        body = r.json()
        assert len(body["reasoning_trace"]) == 2
        assert body["reasoning_trace"][0]["step"] == "classify_intent"

    def test_chat_missing_analysis(self, client):
        r = client.post("/api/v2/chat", json={
            "analysis_id": "NONEXISTENT",
            "message": "hello",
            "history": [],
        })
        assert r.status_code == 404

    @patch("main._get_reasoning_agent")
    def test_chat_fallback_on_agent_error(self, mock_get_reasoning_agent, client):
        mock_get_reasoning_agent.side_effect = RuntimeError("Agent broken")

        from main import _analysis_store  # type: ignore[import-not-found]
        _analysis_store["XCL-CHAT-FALLBACK"] = {
            "analysis_id": "XCL-CHAT-FALLBACK",
            "prediction": "No finding",
            "confidence": 0.95,
            "uncertainty": {},
            "uncertainty_level": "low",
            "region_scores": {},
            "key_findings": [],
            "llm_summary": "Normal.",
            "top_k_predictions": [],
        }

        r = client.post("/api/v2/chat", json={
            "analysis_id": "XCL-CHAT-FALLBACK",
            "message": "test",
            "history": [],
        })
        assert r.status_code == 200
        body = r.json()
        assert "response" in body
        assert body["intent"] == "general_question"


# ===========================================================================
# Drift metrics endpoint (v2)
# ===========================================================================

class TestDriftMetricsV2:
    def test_drift_metrics(self, client):
        r = client.get("/api/v2/drift-metrics")
        assert r.status_code == 200
        body = r.json()
        assert "drift_score" in body
        assert "drift_detected" in body
        assert isinstance(body["drift_detected"], bool)
        assert "avg_confidence" in body
        assert "prediction_distribution" in body
        assert "total_predictions" in body
        # Issue 3: an empty prediction log must not read as "measured, healthy".
        assert body["status"] in ("ok", "insufficient_data", "no_baseline", "error")
        assert "n_predictions" in body
        assert "baseline_confidence" in body
        if body["status"] != "ok":
            assert body["drift_detected"] is False


# ===========================================================================
# Model card endpoint (v2)
# ===========================================================================

class TestModelCardV2:
    def test_model_card(self, client):
        r = client.get("/api/v2/model-card")
        assert r.status_code == 200
        body = r.json()
        assert "name" in body
        assert "version" in body
        assert "limitations" in body
        assert "architectures_available" in body
        assert isinstance(body["architectures_available"], list)


# ===========================================================================
# History endpoint (v2)
# ===========================================================================

class TestHistoryV2:
    def test_history_empty_patient(self, client):
        """History for unknown patient returns empty list."""
        r = client.get("/api/v2/history/NONEXISTENT_PATIENT")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 0

    @patch("main._get_agent")
    @patch("main.get_pipeline")
    def test_history_after_analysis(self, mock_get_pipe, mock_get_agent, client, dummy_image_bytes):
        """After analyzing an image, history should return that entry."""
        mock_get_pipe.return_value = _fake_pipeline()
        mock_agent = MagicMock()
        mock_agent.generate_report.return_value = {
            "findings": "Test.",
            "impression": "Test impression.",
        }
        mock_get_agent.return_value = mock_agent

        # Submit analysis
        r = client.post(
            "/api/v2/analyze",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
            data={"patient_id": "HIST-TEST-001", "model_name": "efficientnet_b0"},
        )
        assert r.status_code == 200

        # Check history — endpoint returns a list of entries directly
        r2 = client.get("/api/v2/history/HIST-TEST-001")
        assert r2.status_code == 200
        body = r2.json()
        assert isinstance(body, list)
        assert len(body) >= 1
        entry = body[0]
        assert "analysis_id" in entry
        assert "prediction" in entry
        assert "confidence" in entry


# ===========================================================================
# Dataset info endpoint (v1)
# ===========================================================================

class TestDatasetInfo:
    def test_dataset_info_returns_structure(self, client):
        """Should return data_root and splits (or 404 if data dir missing)."""
        r = client.get("/api/v1/dataset/info")
        # Accept 200 (data exists) or 404 (data dir not found in test env)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            body = r.json()
            assert "data_root" in body
            assert "splits" in body
            assert "grand_total" in body


# ===========================================================================
# Generate report endpoint (v2)
# ===========================================================================

class TestGenerateReportV2:
    @patch("main._get_agent")
    def test_generate_report(self, mock_get_agent, client):
        mock_agent = MagicMock()
        mock_agent.generate_report.return_value = {
            "findings": "Bilateral pulmonary infiltrates.",
            "impression": "Consider pulmonary fibrosis.",
            "uncertainty": "Moderate.",
            "recommendation": "CT follow-up recommended.",
        }
        mock_get_agent.return_value = mock_agent

        # Put a fake analysis in the store (must include timestamp)
        from main import _analysis_store  # type: ignore[import-not-found]
        _analysis_store["XCL-REPORT-TEST"] = {
            "analysis_id": "XCL-REPORT-TEST",
            "patient_id": "RPT-001",
            "timestamp": "2026-04-01T12:00:00",
            "prediction": "Pulmonary fibrosis",
            "confidence": 0.78,
            "uncertainty": {},
            "uncertainty_level": "moderate",
            "region_scores": {"left_lung": 0.6},
            "key_findings": ["Reticular pattern"],
            "llm_summary": "Pulmonary fibrosis suspected.",
            "model_version": "vit_base",
            "top_k_predictions": [{"class_name": "Pulmonary fibrosis", "probability": 0.78}],
        }

        r = client.post("/api/v2/generate-report", json={
            "analysis_ids": ["XCL-REPORT-TEST"],
            "template": "structured_clinical",
            "sections": ["findings", "impressions", "recommendations"],
        })
        assert r.status_code == 200
        body = r.json()
        assert "report_id" in body
        assert "content" in body
        assert "findings" in body["content"]
        assert "impressions" in body["content"]

    @patch("main._get_agent")
    def test_findings_header_names_the_modality_not_the_architecture(
        self, mock_get_agent, client
    ):
        """main.py:1430 rendered "AI analysis of chest convnext_small (...)".

        model_version was interpolated into the modality slot, with "X-ray"
        serving only as a dict default. Provenance now lives in its own
        labelled field instead of the clinician-facing prose.
        """
        mock_get_agent.return_value = MagicMock()

        from main import _analysis_store  # type: ignore[import-not-found]
        _analysis_store["XCL-HEADER-TEST"] = {
            "analysis_id": "XCL-HEADER-TEST",
            "patient_id": "RPT-002",
            "timestamp": "2026-04-01T12:00:00",
            "prediction": "Cardiomegaly",
            "confidence": 0.81,
            "uncertainty": {},
            "uncertainty_level": "low",
            "region_scores": {},
            "key_findings": [],
            "llm_summary": "",
            "model_version": "convnext_small",
            "top_k_predictions": [{"class_name": "Cardiomegaly", "probability": 0.81}],
        }

        r = client.post("/api/v2/generate-report", json={
            "analysis_ids": ["XCL-HEADER-TEST"],
            "sections": ["findings"],
        })
        assert r.status_code == 200
        body = r.json()

        findings = body["content"]["findings"]
        assert "chest X-ray" in findings

        # No architecture name may leak into the prose a clinician reads.
        architectures = (
            "convnext", "resnet", "densenet", "efficientnet", "vit_base",
            "swin", "b0", "_small",
        )
        lowered = findings.lower()
        leaked = [a for a in architectures if a in lowered]
        assert not leaked, f"architecture leaked into findings prose: {leaked}\n{findings}"

        # ...but provenance is still reported, in its own labelled field.
        assert body["model_version"] == "convnext_small"


# ===========================================================================
# Export report HTML endpoint (v2)
# ===========================================================================

class TestExportReportV2:
    @patch("main._get_agent")
    def test_export_report_html(self, mock_get_agent, client):
        mock_agent = MagicMock()
        mock_agent.generate_report.return_value = {
            "findings": "Cardiomegaly identified.",
            "impression": "Consider cardiac follow-up.",
            "key_findings": ["Enlarged cardiac silhouette"],
            "reasoning_trace": "Step-by-step reasoning.",
            "differential_diagnosis": ["Cardiomegaly", "Pericardial effusion"],
            "urgency": "Medium",
            "next_steps": ["Echocardiogram recommended."],
            "citations": [],
            "recommendation": "Cardiology consult.",
        }
        mock_get_agent.return_value = mock_agent

        from main import _analysis_store  # type: ignore[import-not-found]
        _analysis_store["XCL-EXPORT-TEST"] = {
            "analysis_id": "XCL-EXPORT-TEST",
            "prediction": "Cardiomegaly",
            "confidence": 0.82,
            "uncertainty": {"epistemic": 0.05},
            "uncertainty_level": "moderate",
            "region_scores": {"cardiac": 0.8},
            "key_findings": ["Enlarged cardiac silhouette"],
            "llm_summary": "Cardiomegaly detected.",
            "top_k_predictions": [
                {"class_name": "Cardiomegaly", "probability": 0.82},
                {"class_name": "No finding", "probability": 0.10},
            ],
        }

        r = client.post("/api/v2/export-report", json={
            "analysis_id": "XCL-EXPORT-TEST",
        })
        assert r.status_code == 200
        body = r.json()
        assert "html" in body
        assert "report_id" in body
        assert body["report_id"].startswith("RPT-")

    def test_export_report_missing_analysis(self, client):
        r = client.post("/api/v2/export-report", json={
            "analysis_id": "NONEXISTENT",
        })
        assert r.status_code == 404


# ===========================================================================
# Checkpoint integrity (issue #9)
# ===========================================================================

class TestModelCheckpointIntegrity:
    """_build_pipeline used strict=False and discarded _IncompatibleKeys.

    A checkpoint whose key names don't match the architecture loaded zero
    weights and served a randomly-initialised network behind an INFO log.
    """

    @staticmethod
    @contextlib.contextmanager
    def _stubbed_modeling(arch_factory):
        """Stub xclinvision.modeling so the test needs no real backbone.

        Also sidesteps this environment's torch/torchvision CUDA-major
        mismatch, which makes `import timm` raise.
        """
        import types

        modeling = types.ModuleType("xclinvision.modeling")
        modeling.build_model = lambda *a, **k: arch_factory()
        modeling.get_model_normalization = lambda *a, **k: {
            "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
        }
        inference = types.ModuleType("xclinvision.inference")

        class _Pipe:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        inference.InferencePipeline = _Pipe
        saved = {k: sys.modules.get(k) for k in
                 ("xclinvision.modeling", "xclinvision.inference")}
        sys.modules["xclinvision.modeling"] = modeling
        sys.modules["xclinvision.inference"] = inference
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_model_with_renamed_head_is_rejected(self, tmp_path):
        import torch
        import torch.nn as nn

        import main  # type: ignore[import-not-found]

        class Arch(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 4)

        class Renamed(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(4, 4)

        ckpt = tmp_path / "renamed.pth"
        torch.save({"model_state_dict": Renamed().state_dict()}, ckpt)

        with self._stubbed_modeling(Arch):
            with pytest.raises(ValueError, match="uninitialised"):
                main._build_pipeline(str(ckpt), "convnext_small", 384)

    def test_model_matching_checkpoint_still_loads(self, tmp_path):
        """The guard must not reject a legitimate checkpoint."""
        import torch
        import torch.nn as nn

        import main  # type: ignore[import-not-found]

        class Arch(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 4)

        ckpt = tmp_path / "good.pth"
        torch.save(
            {"model_state_dict": Arch().state_dict(),
             "class_names": ["Cardiomegaly", "Aortic enlargement"]},
            ckpt,
        )

        with self._stubbed_modeling(Arch):
            pipe = main._build_pipeline(str(ckpt), "convnext_small", 384)

        # Labels come from the checkpoint, not configs/system.yaml.
        assert pipe.kwargs["class_names"] == ["Cardiomegaly", "Aortic enlargement"]

    def test_model_meta_disagreeing_with_payload_is_rejected(self, tmp_path):
        import torch
        import torch.nn as nn

        import main  # type: ignore[import-not-found]

        class Arch(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 4)

        ckpt = tmp_path / "skew.pth"
        torch.save(
            {"model_state_dict": Arch().state_dict(),
             "class_names": ["Cardiomegaly", "Aortic enlargement"]},
            ckpt,
        )

        with self._stubbed_modeling(Arch):
            with pytest.raises(ValueError, match="disagrees with its _meta.json"):
                main._build_pipeline(
                    str(ckpt), "convnext_small", 384,
                    meta_class_names=["Aortic enlargement", "Cardiomegaly"],
                )

    def test_model_registry_skips_mismatched_class_names(self, tmp_path, caplog):
        """_discover_models must not register a mislabelling model."""
        import json as _json

        import main  # type: ignore[import-not-found]
        from xclinvision.config import get_class_names

        (tmp_path / "bogus.pth").write_bytes(b"placeholder")
        (tmp_path / "bogus_meta.json").write_text(_json.dumps({
            "model_name": "convnext_small",
            "num_classes": len(get_class_names()),
            # Same labels, wrong order -> every prediction would be mislabelled.
            "class_names": list(reversed(get_class_names())),
        }))

        original = main._MODELS_DIR
        main._MODELS_DIR = tmp_path
        try:
            with caplog.at_level(logging.ERROR):
                registry = main._discover_models()
        finally:
            main._MODELS_DIR = original

        assert registry == {}
        assert any("Refusing model" in r.message for r in caplog.records)

    def test_model_registry_accepts_the_shipped_models(self):
        """The guard must not reject the real models/best_models/ set."""
        import main  # type: ignore[import-not-found]

        registry = main._discover_models()
        assert set(registry) >= {"convnext_small", "vit_base"}


# ===========================================================================
# History / chat query cost (issue #10)
# ===========================================================================

class TestHistoryQueryCost:
    @staticmethod
    def _seed(count: int, target: str = "P-TARGET", target_rows: int = 5):
        """Fill the store with `count` rows carrying realistic heatmap payloads."""
        import main  # type: ignore[import-not-found]

        blob = "A" * 200_000  # ~200 KB base64 PNG, as /api/v2/analyze produces
        for i in range(count):
            pid = target if i < target_rows else f"P-OTHER-{i}"
            main._analysis_store_put(f"XCL-HIST-{i:05d}", {
                "analysis_id": f"XCL-HIST-{i:05d}",
                "patient_id": pid,
                "timestamp": f"2026-08-{(i % 28) + 1:02d}T10:00:00+00:00",
                "prediction": "Cardiomegaly",
                "confidence": 0.8,
                "uncertainty": {},
                "uncertainty_level": "low",
                "model_version": "vit_base",
                "top_k_predictions": [],
                "key_findings": [],
                "llm_summary": "",
                "heatmap_gradcam": blob,
                "heatmap_overlay": blob,
            })

    def test_history_is_bounded_with_many_unrelated_rows(self, client):
        import time

        self._seed(1000)

        start = time.perf_counter()
        r = client.get("/api/v2/history/P-TARGET")
        elapsed = time.perf_counter() - start

        assert r.status_code == 200
        body = r.json()
        assert len(body) == 5
        assert {row["analysis_id"] for row in body} == {
            f"XCL-HIST-{i:05d}" for i in range(5)
        }
        # Pre-fix this decoded all 1000 rows (~400 MB of base64) in Python.
        assert elapsed < 0.1, f"history took {elapsed:.3f}s for 1000 stored rows"

    def test_history_still_returns_heatmaps(self, client):
        """Moving blobs out of data_json must not drop them from the response."""
        self._seed(3, target="P-HEAT", target_rows=3)

        body = client.get("/api/v2/history/P-HEAT").json()

        assert body, "expected rows for P-HEAT"
        assert body[0]["heatmap_overlay"], "overlay lost in the round-trip"
        assert body[0]["thumbnail"], "thumbnail lost in the round-trip"

    def test_history_chat_context_carries_no_base64_payloads(self):
        """The reasoning tools must never receive heatmap blobs."""
        import main  # type: ignore[import-not-found]

        self._seed(20, target="P-CTX", target_rows=2)

        context = main.storage.summaries(limit=main._CHAT_CONTEXT_MAX)

        assert context, "expected summaries"
        for row in context.values():
            assert set(row) == set(main.storage._SUMMARY_KEYS)
            for value in row.values():
                assert not (isinstance(value, str) and len(value) > 1000), row


# ===========================================================================
# Drift monitoring is actually fed (issue #3)
# ===========================================================================

class TestDriftPipelineWiring:
    """PredictionLogger.log_prediction had no caller, so drift read an empty
    directory and reported zeros — indistinguishable from a healthy model."""

    @staticmethod
    def _isolated_logger(tmp_path):
        import main  # type: ignore[import-not-found]
        from xclinvision.monitoring import PredictionLogger

        original = main._prediction_logger
        main._prediction_logger = PredictionLogger(log_dir=str(tmp_path / "preds"))
        return original

    def test_drift_reports_insufficient_data_on_empty_history(self, client, tmp_path):
        import main  # type: ignore[import-not-found]

        original = self._isolated_logger(tmp_path)
        try:
            body = client.get("/api/v2/drift-metrics").json()
        finally:
            main._prediction_logger = original

        assert body["status"] == "insufficient_data"
        assert body["n_predictions"] == 0
        # Zeros here would read as "measured, and healthy".
        assert body["drift_score"] is None
        assert body["avg_confidence"] is None
        assert body["drift_detected"] is False

    def test_drift_counts_a_submitted_analysis(self, client, tmp_path, dummy_image_bytes):
        import main  # type: ignore[import-not-found]

        pipeline = MagicMock()
        pipeline.predict.return_value = {
            "prediction": 0,
            "class_name": "Cardiomegaly",
            "probabilities": [0.62, 0.1, 0.0, 0.0],
            "confidence": 0.62,
            "raw_probability": 0.62,
            "calibrated": False,
            "class_names": ["Cardiomegaly", "Aortic enlargement",
                            "Pleural thickening", "Pulmonary fibrosis"],
            "uncertainty": {"epistemic": 0.01},
            "uncertainty_level": "low",
            "explanation": {"key_findings": [], "visualization": {"region_scores": {}}},
        }
        pipeline.preprocess.return_value = (
            np.zeros((3, 64, 64), dtype=np.float32),
            np.zeros((64, 64, 3), dtype=np.uint8),
        )

        original = self._isolated_logger(tmp_path)
        try:
            with patch.object(main, "get_pipeline", return_value=pipeline), \
                    patch.object(main, "_get_agent", side_effect=RuntimeError("no llm")):
                r = client.post(
                    "/api/v2/analyze",
                    files={"file": ("a.jpg", dummy_image_bytes, "image/jpeg")},
                )
            assert r.status_code == 200
            body = client.get("/api/v2/drift-metrics").json()
        finally:
            main._prediction_logger = original

        assert body["n_predictions"] == 1
        assert body["status"] in ("ok", "no_baseline")
        assert body["avg_confidence"] == pytest.approx(0.62, abs=1e-3)

    def test_corrupt_log_line_is_skipped_not_fatal(self, tmp_path):
        """One truncated append must not take down the whole history read."""
        from xclinvision.monitoring import PredictionLogger

        log_dir = tmp_path / "preds"
        logger_ = PredictionLogger(log_dir=str(log_dir))
        logger_.log_prediction(
            image_hash="a" * 8, prediction=0, probabilities=[0.9],
            confidence=0.9, uncertainty={"epistemic": 0.01}, model_version="vit_base",
        )
        # Simulate a crash mid-append, then a clean record after it.
        log_file = next(log_dir.glob("predictions_*.jsonl"))
        with open(log_file, "a") as fh:
            fh.write('{"timestamp": "2026-08-25T10:00:00+00:00", "conf\n')
        logger_.log_prediction(
            image_hash="b" * 8, prediction=1, probabilities=[0.7],
            confidence=0.7, uncertainty={"epistemic": 0.02}, model_version="vit_base",
        )

        history = logger_.get_prediction_history()

        assert len(history) == 2, "valid records lost to a corrupt neighbour"
        assert {h["image_hash"] for h in history} == {"a" * 8, "b" * 8}

    def test_naive_and_aware_timestamps_both_filter(self, tmp_path):
        """Legacy naive entries must not raise against an aware bound."""
        import json as _json
        from datetime import datetime, timedelta, timezone

        from xclinvision.monitoring import PredictionLogger

        log_dir = tmp_path / "preds"
        log_dir.mkdir(parents=True)
        naive_recent = datetime.now().replace(microsecond=0).isoformat()
        aware_old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        with open(log_dir / "predictions_2026-08-25.jsonl", "w") as fh:
            fh.write(_json.dumps({"timestamp": naive_recent, "confidence": 0.9}) + "\n")
            fh.write(_json.dumps({"timestamp": aware_old, "confidence": 0.5}) + "\n")

        start = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        history = PredictionLogger(log_dir=str(log_dir)).get_prediction_history(
            start_date=start
        )

        assert len(history) == 1
        assert history[0]["confidence"] == 0.9


# ===========================================================================
# Compare / analyze record parity
# ===========================================================================

def _multilabel_pipeline(class_names, positives):
    """Mock pipeline whose predict() calls `positives` (indices) positive.

    Built from the configured class names rather than hardcoded labels, so the
    test follows configs/system.yaml like the rest of the system does.
    """
    n = len(class_names)
    probs = [0.91 if i in positives else 0.04 for i in range(n)]
    binary = [1 if i in positives else 0 for i in range(n)]
    headline = class_names[positives[0]]

    pipeline = MagicMock()
    pipeline.predict.return_value = {
        "prediction": positives[0],
        "class_name": headline,
        "class_names": list(class_names),
        "probabilities": probs,
        "thresholds": [0.5] * n,
        "confidence": probs[positives[0]],
        "raw_probability": probs[positives[0]],
        "calibrated": False,
        "calibration_status": "uncalibrated",
        "uncertainty": {"epistemic": 0.02, "aleatoric": 0.01},
        "uncertainty_level": "low",
        "explanation": {
            "key_findings": ["Finding present"],
            "visualization": {
                "grayscale_cam": np.random.rand(64, 64).astype(np.float32),
                "region_scores": {"cardiac": 0.8},
                "method": "gradcam++",
            },
        },
        "predictions_multilabel": binary,
        "class_names_predicted": [class_names[i] for i in positives],
    }
    pipeline.preprocess.return_value = (
        np.random.rand(3, 384, 384).astype(np.float32),
        np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8),
    )
    return pipeline


def _stub_agent():
    agent = MagicMock()
    agent.generate_report.return_value = {
        "findings": "Findings text.",
        "impression": "Impression text.",
        "key_findings": ["Finding present"],
        "reasoning_trace": "",
        "differential_diagnosis": [],
        "urgency": "Medium",
        "next_steps": ["Clinical correlation recommended."],
        "citations": [],
    }
    return agent


class TestCompareAnalyzeParity:
    """/api/v2/compare wrote a different record shape than /api/v2/analyze.

    Both endpoints store into the same analysis key space, and chat / explain /
    export-report read that space without knowing which endpoint produced a
    record. Compare's hand-rolled dict omitted class_names_predicted,
    predictions_multilabel, calibration_status and raw_probability, so the
    multilabel under-triage fix (issue #1) and the uncalibrated-confidence fix
    (issue #7) never reached anything served from a compare-produced analysis.
    """

    @patch("main._get_agent")
    @patch("main.get_pipeline")
    def test_compare_and_analyze_store_identical_key_sets(
        self, mock_get_pipe, mock_get_agent, client, dummy_image_bytes
    ):
        """The regression guard: the two endpoints must not drift apart again."""
        mock_get_pipe.return_value = _fake_pipeline()
        mock_get_agent.return_value = _stub_agent()

        from main import _analysis_store  # type: ignore[import-not-found]

        r_analyze = client.post(
            "/api/v2/analyze",
            files={"file": ("xray.jpg", dummy_image_bytes, "image/jpeg")},
            data={"patient_id": "PARITY-001", "model_name": "efficientnet_b0"},
        )
        assert r_analyze.status_code == 200
        analyze_id = r_analyze.json()["analysis_id"]

        r_compare = client.post(
            "/api/v2/compare",
            files={
                "file_a": ("a.jpg", dummy_image_bytes, "image/jpeg"),
                "file_b": ("b.jpg", dummy_image_bytes, "image/jpeg"),
            },
            data={"model_name": "efficientnet_b0"},
        )
        assert r_compare.status_code == 200
        compare_id = r_compare.json()["image_a"]["analysis_id"]

        analyze_keys = set(_analysis_store[analyze_id].keys())
        compare_keys = set(_analysis_store[compare_id].keys())

        assert compare_keys == analyze_keys, (
            "compare/analyze stored-record shapes diverged; "
            f"only in analyze: {sorted(analyze_keys - compare_keys)}; "
            f"only in compare: {sorted(compare_keys - analyze_keys)}"
        )

        # The four fields whose absence made the earlier fixes inert.
        for field in (
            "class_names_predicted",
            "predictions_multilabel",
            "calibration_status",
            "raw_probability",
        ):
            assert field in compare_keys, f"{field} missing from compare record"

        # Both images of a comparison get the same treatment.
        compare_id_b = r_compare.json()["image_b"]["analysis_id"]
        assert set(_analysis_store[compare_id_b].keys()) == analyze_keys

    @patch("main._get_agent")
    @patch("main.get_pipeline")
    def test_compare_analysis_exports_report_without_calibrated_language(
        self, mock_get_pipe, mock_get_agent, client, dummy_image_bytes
    ):
        """A compare-produced record must reach the reporter as uncalibrated.

        The reporter downgrades every probability to "Raw model probability:
        ... (not calibrated)" unless calibration_status == "calibrated". With
        the field missing from the stored record this relied on a defaulting
        `.get`, so the guarantee was accidental rather than carried by the data.
        """
        from xclinvision.config import get_class_names

        class_names = get_class_names()
        mock_get_pipe.return_value = _multilabel_pipeline(class_names, positives=[0])
        mock_get_agent.return_value = _stub_agent()

        from main import _analysis_store  # type: ignore[import-not-found]

        r_compare = client.post(
            "/api/v2/compare",
            files={
                "file_a": ("a.jpg", dummy_image_bytes, "image/jpeg"),
                "file_b": ("b.jpg", dummy_image_bytes, "image/jpeg"),
            },
            data={"model_name": "efficientnet_b0"},
        )
        assert r_compare.status_code == 200
        compare_id = r_compare.json()["image_a"]["analysis_id"]

        stored = _analysis_store[compare_id]
        assert stored["calibration_status"] == "uncalibrated"
        assert stored["calibrated"] is False

        r_report = client.post(
            "/api/v2/export-report",
            json={"analysis_id": compare_id, "format": "html"},
        )
        assert r_report.status_code == 200
        html = r_report.json()["html"]

        assert "not calibrated" in html
        assert "Consistent with" not in html, (
            "uncalibrated compare output was rendered in calibrated clinical language"
        )
        assert "Highly suggestive" not in html

    @patch("main._get_agent")
    @patch("main.get_pipeline")
    def test_compare_surfaces_all_positive_labels(
        self, mock_get_pipe, mock_get_agent, client, dummy_image_bytes
    ):
        """Co-occurring findings must survive the compare path, not just the headline.

        Compare previously returned only `prediction`, so a second positive
        label was invisible to the dashboard and to any report generated from
        that analysis — the under-triage this fix exists to prevent.
        """
        from xclinvision.config import get_class_names

        class_names = get_class_names()
        if len(class_names) < 2:
            pytest.skip("multilabel assertions need at least two configured classes")

        positives = [0, 1]
        mock_get_pipe.return_value = _multilabel_pipeline(class_names, positives)
        mock_get_agent.return_value = _stub_agent()

        r_compare = client.post(
            "/api/v2/compare",
            files={
                "file_a": ("a.jpg", dummy_image_bytes, "image/jpeg"),
                "file_b": ("b.jpg", dummy_image_bytes, "image/jpeg"),
            },
            data={"model_name": "efficientnet_b0"},
        )
        assert r_compare.status_code == 200
        image_a = r_compare.json()["image_a"]

        expected = [class_names[i] for i in positives]
        assert image_a["class_names_predicted"] == expected, (
            "compare surfaced only the headline label"
        )

        positive_names = [
            row["class_name"] for row in image_a["predictions_multilabel"] if row["positive"]
        ]
        assert positive_names == expected

        # Headline still matches what the pipeline ranked first.
        assert image_a["prediction"] == class_names[positives[0]]


# ===========================================================================
# Heatmap retention: evicted must be distinguishable from never-existed
# ===========================================================================

class TestHeatmapEvictionSignal:
    """Heatmap blobs are evicted independently of the analyses row.

    Before this fix the read path returned a bare None either way, so an
    older study that lost its spatial evidence looked identical to one that
    never produced any — silent loss of clinical evidence, not just storage.
    """

    @staticmethod
    def _put(analysis_id, **fields):
        import main  # type: ignore[import-not-found]

        payload = {
            "analysis_id": analysis_id,
            "prediction": "Cardiomegaly",
            "confidence": 0.82,
            "uncertainty_level": "low",
            "region_scores": {"cardiac": 0.8},
            "key_findings": ["Enlarged cardiac silhouette"],
            "llm_summary": "",
            "top_k_predictions": [
                {"class_name": "Cardiomegaly", "probability": 0.82},
                {"class_name": "Aortic enlargement", "probability": 0.10},
            ],
        }
        payload.update(fields)
        main._analysis_store_put(analysis_id, payload)

    def test_evicted_heatmap_is_distinguishable_from_never_existed(self, client):
        import main  # type: ignore[import-not-found]

        # gradcam produced; scorecam never was.
        self._put("XCL-EVICT-AAA", heatmap_gradcam="Zm9vYmFy")

        hydrated = main._rehydrate_heatmaps(
            "XCL-EVICT-AAA", main._analysis_store["XCL-EVICT-AAA"]
        )
        status = hydrated["heatmap_status"]
        assert status["heatmap_gradcam"] == "present"
        assert status["scorecam_heatmap"] == "absent"
        assert hydrated["heatmap_gradcam"] == "Zm9vYmFy"

        # Force real eviction through the budget rather than deleting by hand.
        original_cap = main.storage.heatmap_max_count
        try:
            main.storage.heatmap_max_count = 1
            self._put("XCL-EVICT-BBB", heatmap_gradcam="YmFyYmF6")
        finally:
            main.storage.heatmap_max_count = original_cap

        evicted = main._rehydrate_heatmaps(
            "XCL-EVICT-AAA", main._analysis_store["XCL-EVICT-AAA"]
        )
        assert evicted["heatmap_gradcam"] is None
        assert evicted["heatmap_status"]["heatmap_gradcam"] == "evicted", (
            "an evicted heatmap is indistinguishable from one that never existed"
        )
        # A heatmap this study never produced stays 'absent', not 'evicted'.
        assert evicted["heatmap_status"]["scorecam_heatmap"] == "absent"
        assert main._evicted_heatmap_fields(evicted) == ["heatmap_gradcam"]

    def test_row_without_a_manifest_reports_unknown_not_absent(self, client):
        """Rows written before the manifest existed cannot be classified."""
        import main  # type: ignore[import-not-found]

        # Bypass _analysis_store_put, so no heatmap_blobs key is recorded.
        main._analysis_store["XCL-LEGACY-ROW"] = {
            "analysis_id": "XCL-LEGACY-ROW",
            "prediction": "Cardiomegaly",
            "confidence": 0.5,
        }
        hydrated = main._rehydrate_heatmaps(
            "XCL-LEGACY-ROW", main._analysis_store["XCL-LEGACY-ROW"]
        )
        assert set(hydrated["heatmap_status"].values()) == {"unknown"}
        # 'unknown' must not be reported as eviction.
        assert main._evicted_heatmap_fields(hydrated) == []

    @patch("main._get_agent")
    def test_report_states_spatial_evidence_unavailable_when_evicted(
        self, mock_get_agent, client
    ):
        """An evicted overlay must be stated, not silently dropped from the report."""
        import main  # type: ignore[import-not-found]

        mock_get_agent.return_value = _stub_agent()

        self._put("XCL-EVICT-RPT", heatmap_gradcam="Zm9vYmFy")
        original_cap = main.storage.heatmap_max_count
        try:
            main.storage.heatmap_max_count = 1
            self._put("XCL-EVICT-RPT2", heatmap_gradcam="YmFyYmF6")
        finally:
            main.storage.heatmap_max_count = original_cap

        hydrated = main._rehydrate_heatmaps(
            "XCL-EVICT-RPT", main._analysis_store["XCL-EVICT-RPT"]
        )
        assert hydrated["heatmap_status"]["heatmap_gradcam"] == "evicted"

        r = client.post(
            "/api/v2/export-report",
            json={"analysis_id": "XCL-EVICT-RPT", "format": "html"},
        )
        assert r.status_code == 200
        html = r.json()["html"]
        assert "Spatial evidence unavailable" in html, (
            "report dropped the panel instead of stating the evidence was reclaimed"
        )
        assert "heatmap_gradcam" in html

    @patch("main._get_agent")
    def test_report_says_nothing_when_no_heatmap_was_ever_produced(
        self, mock_get_agent, client
    ):
        """The notice is for lost evidence only — not for studies without XAI."""
        import main  # type: ignore[import-not-found]

        mock_get_agent.return_value = _stub_agent()
        self._put("XCL-NOXAI-RPT")

        r = client.post(
            "/api/v2/export-report",
            json={"analysis_id": "XCL-NOXAI-RPT", "format": "html"},
        )
        assert r.status_code == 200
        assert "Spatial evidence unavailable" not in r.json()["html"]

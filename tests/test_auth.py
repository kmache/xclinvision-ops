"""Auth smoke tests for protected endpoints.

These verify that protected endpoints enforce the bearer token. v1
endpoints are now also bearer-protected (issue #1). Open endpoints
(health, model-card, llm/providers, llm/health) must keep working
without auth.
"""
from __future__ import annotations


# ---------------------------------------------------------------- protected
def test_v2_analyze_requires_auth(unauth_client, dummy_image_bytes):
    r = unauth_client.post(
        "/api/v2/analyze",
        files={"file": ("x.jpg", dummy_image_bytes, "image/jpeg")},
        data={"patient_id": "P1"},
    )
    assert r.status_code == 401


def test_v2_history_requires_auth(unauth_client):
    r = unauth_client.get("/api/v2/history/P1")
    assert r.status_code == 401


def test_v2_feedback_requires_auth(unauth_client):
    r = unauth_client.post("/api/v2/feedback", json={"analysis_id": "x", "feedback_type": "correct"})
    assert r.status_code == 401


def test_v2_chat_requires_auth(unauth_client):
    r = unauth_client.post("/api/v2/chat", json={"analysis_id": "x", "message": "hi"})
    assert r.status_code == 401


def test_v2_drift_metrics_requires_auth(unauth_client):
    r = unauth_client.get("/api/v2/drift-metrics")
    assert r.status_code == 401


def test_v2_llm_switch_requires_auth(unauth_client):
    r = unauth_client.post("/api/v2/llm/switch", json={"provider": "openai", "model": "x"})
    assert r.status_code == 401


def test_v1_predict_requires_auth(unauth_client, dummy_image_bytes):
    r = unauth_client.post(
        "/api/v1/predict",
        files={"file": ("x.jpg", dummy_image_bytes, "image/jpeg")},
    )
    assert r.status_code == 401


def test_v1_feedback_requires_auth(unauth_client):
    r = unauth_client.post(
        "/api/v1/feedback",
        json={"prediction": 0, "correct_label": 0, "feedback_type": "correct"},
    )
    assert r.status_code == 401


def test_v1_dataset_info_requires_auth(unauth_client):
    r = unauth_client.get("/api/v1/dataset/info")
    assert r.status_code == 401


def test_invalid_token_is_rejected(unauth_client):
    r = unauth_client.get(
        "/api/v2/history/P1", headers={"Authorization": "Bearer wrong-token"}
    )
    assert r.status_code == 401


# --------------------------------------------------------------------- open
def test_health_open(unauth_client):
    r = unauth_client.get("/health")
    assert r.status_code == 200


def test_model_card_open(unauth_client):
    r = unauth_client.get("/api/v2/model-card")
    assert r.status_code == 200


def test_llm_providers_requires_auth(unauth_client):
    """/api/v2/llm/providers discloses backend LLM configuration."""
    r = unauth_client.get("/api/v2/llm/providers")
    assert r.status_code == 401


def test_llm_health_requires_auth(unauth_client):
    """Each provider probe is a real billed completion — never leave it open."""
    r = unauth_client.get("/api/v2/llm/health")
    assert r.status_code == 401

"""Auth smoke tests for the v2 PII endpoints.

These verify that protected endpoints enforce the bearer token. Open
endpoints (health, v1, model-card, llm/providers, llm/health) must
keep working without auth.
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


def test_llm_providers_open(unauth_client):
    r = unauth_client.get("/api/v2/llm/providers")
    assert r.status_code == 200

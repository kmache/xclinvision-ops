"""Regression tests for ReasoningAgent._format_urgency_assessment.

Issue #4: pneumothorax / consolidation must escalate to High urgency
in the rule-based path, even at moderate confidence — under-triage
of these critical findings is a patient-safety regression.
"""
from __future__ import annotations

import pytest

from xclinvision.agent.guardrails import CRITICAL_CONDITIONS
from xclinvision.agent.reasoning import ReasoningAgent
from xclinvision.agent.tools import ToolResult


def _pred_result(prediction: str, confidence: float, uncertainty: str) -> ToolResult:
    return ToolResult(
        success=True,
        data={
            "prediction": prediction,
            "confidence": confidence,
            "uncertainty_level": uncertainty,
        },
        tool_name="get_prediction_details",
    )


@pytest.mark.parametrize("prediction", sorted(CRITICAL_CONDITIONS))
def test_critical_conditions_escalate_to_high(prediction):
    """Pneumothorax / Consolidation -> High regardless of confidence."""
    results = {
        "get_prediction_details": _pred_result(prediction, confidence=0.55, uncertainty="medium"),
    }
    out = ReasoningAgent._format_urgency_assessment(results)
    assert "Urgency Assessment: High" in out
    assert prediction in out


def test_no_finding_remains_low():
    results = {
        "get_prediction_details": _pred_result("No finding", confidence=0.95, uncertainty="low"),
    }
    out = ReasoningAgent._format_urgency_assessment(results)
    assert "Urgency Assessment: Low" in out


def test_aortic_enlargement_high_confidence_remains_high():
    results = {
        "get_prediction_details": _pred_result("Aortic enlargement", confidence=0.85, uncertainty="low"),
    }
    out = ReasoningAgent._format_urgency_assessment(results)
    assert "Urgency Assessment: High" in out


def test_non_critical_high_confidence_returns_medium():
    """Non-critical findings at high confidence stay Medium (existing rule)."""
    results = {
        "get_prediction_details": _pred_result("Cardiomegaly", confidence=0.85, uncertainty="low"),
    }
    out = ReasoningAgent._format_urgency_assessment(results)
    assert "Urgency Assessment: Medium" in out


def test_missing_prediction_returns_unable_message():
    out = ReasoningAgent._format_urgency_assessment({})
    assert "Unable to assess urgency" in out

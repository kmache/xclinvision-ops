"""Agent Tool Registry — callable tools for the reasoning agent.

Each tool is a function that the agent can invoke during its reasoning loop
to interact with backend services (inference, evaluation, monitoring, reports).

Tools follow a standard interface:
    - Accept a ``context: dict`` with tool-specific parameters.
    - Return a ``ToolResult`` with structured output or error info.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool result / descriptor
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Standard return value from any agent tool."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    tool_name: str = ""

    def to_prompt_text(self) -> str:
        """Serialize the result for injection into an LLM prompt."""
        if not self.success:
            return f"[TOOL ERROR: {self.tool_name}] {self.error}"
        if isinstance(self.data, dict):
            return json.dumps(self.data, indent=2, default=str)
        return str(self.data)


@dataclass
class ToolDescriptor:
    """Metadata describing a tool the agent can call."""

    name: str
    description: str
    parameters: Dict[str, str] = field(default_factory=dict)
    func: Optional[Callable[..., ToolResult]] = None


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Central registry of tools available to the reasoning agent.

    Tools are registered at init time and looked up by name during the
    agent's plan-execute loop.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDescriptor] = {}

    def register(self, descriptor: ToolDescriptor) -> None:
        self._tools[descriptor.name] = descriptor

    def get(self, name: str) -> Optional[ToolDescriptor]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDescriptor]:
        return list(self._tools.values())

    def list_tool_descriptions(self) -> str:
        """Format tool list for inclusion in an LLM system prompt."""
        lines = []
        for t in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items())
            lines.append(f"- **{t.name}**({params}): {t.description}")
        return "\n".join(lines)

    def execute(self, name: str, context: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name. Returns ToolResult with error on failure."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool '{name}'",
                tool_name=name,
            )
        if tool.func is None:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' has no implementation",
                tool_name=name,
            )
        try:
            result = tool.func(context)
            result.tool_name = name
            return result
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", name)
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                tool_name=name,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in tool implementations
# ═══════════════════════════════════════════════════════════════════════════════


def _tool_get_prediction_details(context: Dict[str, Any]) -> ToolResult:
    """Extract and summarize prediction data from the current analysis."""
    analysis = context.get("analysis")
    if not analysis:
        return ToolResult(success=False, error="No analysis data available.")

    top_k = analysis.get("top_k_predictions", [])
    data = {
        "prediction": analysis.get("prediction", "Unknown"),
        "confidence": analysis.get("confidence", 0.0),
        "uncertainty_level": analysis.get("uncertainty_level", "unknown"),
        "top_predictions": [
            {"class": p["class_name"], "probability": round(p["probability"], 4)}
            for p in top_k
        ],
        "model_version": analysis.get("model_version", "unknown"),
    }
    return ToolResult(success=True, data=data)


def _tool_get_xai_explanation(context: Dict[str, Any]) -> ToolResult:
    """Retrieve and interpret XAI spatial evidence from the analysis."""
    analysis = context.get("analysis")
    if not analysis:
        return ToolResult(success=False, error="No analysis data available.")

    region_scores = analysis.get("region_scores", {})
    key_findings = analysis.get("key_findings", [])

    # Import the XAI interpreter
    try:
        from xclinvision.agent.xclinvisionagent import interpret_xai_regions
        spatial = interpret_xai_regions(region_scores)
    except Exception:
        spatial = {k: f"score={v:.2f}" for k, v in region_scores.items()}

    data = {
        "region_scores": region_scores,
        "spatial_interpretation": spatial,
        "key_findings": key_findings,
        "has_heatmap": analysis.get("heatmap_gradcam") is not None,
    }
    return ToolResult(success=True, data=data)


def _tool_get_evaluation_metrics(context: Dict[str, Any]) -> ToolResult:
    """Load evaluation metrics for the current model from saved reports."""
    import os
    from pathlib import Path

    model_name = context.get("model_name") or "vit_base"

    outputs_dir = Path(os.getenv(
        "XCLINVISION_OUTPUTS_DIR",
        str(Path(__file__).resolve().parent.parent.parent.parent / "outputs"),
    ))

    eval_dir = outputs_dir / f"evaluation_384_{model_name}"
    report_path = eval_dir / f"{model_name}_test_evaluation_report.json"

    if not report_path.exists():
        # Fallback: search for any evaluation report
        candidates = sorted(outputs_dir.glob(f"*{model_name}*/*evaluation_report.json"))
        if candidates:
            report_path = candidates[-1]
        else:
            return ToolResult(
                success=False,
                error=f"No evaluation report found for model '{model_name}'.",
            )

    try:
        with open(report_path) as f:
            metrics = json.load(f)
        # Extract key metrics
        summary = {
            "model": model_name,
            "source": str(report_path),
            "macro_auc": metrics.get("macro_auc"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "subset_accuracy": metrics.get("subset_accuracy"),
        }
        # Add per-class AUC
        per_class = {}
        for key, val in metrics.items():
            if key.endswith("_auc") and key not in ("macro_auc", "weighted_auc"):
                class_name = key.replace("_auc", "")
                per_class[class_name] = {
                    "auc": val,
                    "sensitivity": metrics.get(f"{class_name}_sensitivity"),
                    "specificity": metrics.get(f"{class_name}_specificity"),
                    "f1": metrics.get(f"{class_name}_f1"),
                }
            if key == "calibration" and isinstance(val, dict):
                summary["ece"] = val.get("expected_calibration_error")
        summary["per_class"] = per_class
        return ToolResult(success=True, data=summary)
    except Exception as exc:
        return ToolResult(success=False, error=f"Failed to load metrics: {exc}")


def _tool_get_monitoring_status(context: Dict[str, Any]) -> ToolResult:
    """Get drift monitoring and feedback statistics."""
    # Pull from in-memory stores if available (injected via context).
    # Backend passes these under "analysis_store"/"feedback_store"; the
    # earlier "_analysis_store"/"_feedback_store" lookups silently always
    # returned empty defaults.
    feedback_store = context.get("feedback_store", [])
    analysis_store = context.get("analysis_store", {})

    total_predictions = len(analysis_store)
    total_feedback = len(feedback_store)

    by_type: Dict[str, int] = {}
    for fb in feedback_store:
        ft = fb.get("feedback_type", "unknown")
        by_type[ft] = by_type.get(ft, 0) + 1

    # Simple drift indicator from confidence distribution
    confidences = [
        a.get("confidence", 0.0) for a in analysis_store.values()
    ]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    data = {
        "total_predictions": total_predictions,
        "total_feedback": total_feedback,
        "feedback_by_type": by_type,
        "avg_confidence": round(avg_conf, 4),
        "correction_rate": round(
            by_type.get("incorrect", 0) / max(total_feedback, 1) * 100, 1
        ),
    }
    return ToolResult(success=True, data=data)


def _tool_generate_report(context: Dict[str, Any]) -> ToolResult:
    """Generate a structured clinical report from the current analysis."""
    analysis = context.get("analysis")
    if not analysis:
        return ToolResult(success=False, error="No analysis data available.")

    try:
        from xclinvision.agent import ClinicalContext, create_agent

        probs = [p["probability"] for p in analysis.get("top_k_predictions", [])]
        class_names = [p["class_name"] for p in analysis.get("top_k_predictions", [])]

        ctx = ClinicalContext(
            prediction=analysis.get("prediction", ""),
            probabilities=probs,
            confidence=analysis.get("confidence", 0.0),
            uncertainty_level=analysis.get("uncertainty_level", "unknown"),
            highlighted_regions=list(analysis.get("region_scores", {}).keys())[:5],
            class_names=class_names,
        )

        agent = create_agent()
        report = agent.generate_report(ctx)
        return ToolResult(success=True, data=report)
    except Exception as exc:
        return ToolResult(success=False, error=f"Report generation failed: {exc}")


def _tool_compare_with_history(context: Dict[str, Any]) -> ToolResult:
    """Compare current analysis with prior studies for the same patient."""
    analysis = context.get("analysis")
    analysis_store = context.get("analysis_store", {})

    if not analysis:
        return ToolResult(success=False, error="No current analysis available.")

    patient_id = analysis.get("patient_id", "")
    if not patient_id or patient_id == "UNKNOWN":
        return ToolResult(
            success=False,
            error="No patient ID — cannot retrieve history.",
        )

    # Find prior analyses for this patient
    history = [
        v for k, v in analysis_store.items()
        if v.get("patient_id") == patient_id
        and k != analysis.get("analysis_id")
    ]
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    if not history:
        return ToolResult(success=True, data={
            "has_prior": False,
            "message": "No prior studies found for this patient.",
        })

    prior = history[0]
    data = {
        "has_prior": True,
        "current": {
            "prediction": analysis.get("prediction"),
            "confidence": analysis.get("confidence"),
            "timestamp": analysis.get("timestamp"),
        },
        "prior": {
            "prediction": prior.get("prediction"),
            "confidence": prior.get("confidence"),
            "timestamp": prior.get("timestamp"),
            "analysis_id": prior.get("analysis_id"),
        },
        "total_prior_studies": len(history),
    }
    return ToolResult(success=True, data=data)


def _tool_suggest_next_steps(context: Dict[str, Any]) -> ToolResult:
    """Suggest clinical next steps based on the current analysis."""
    analysis = context.get("analysis")
    if not analysis:
        return ToolResult(success=False, error="No analysis data available.")

    prediction = analysis.get("prediction", "Unknown")
    confidence = analysis.get("confidence", 0.0)
    uncertainty = analysis.get("uncertainty_level", "unknown")

    steps = []

    # Always recommend clinical correlation
    steps.append("Correlate AI findings with clinical presentation and patient history.")

    # Confidence-based
    if confidence < 0.5:
        steps.append(
            f"Low confidence ({confidence:.0%}) — consider additional imaging "
            "(lateral view, CT) or specialist consultation."
        )
    elif confidence < 0.7:
        steps.append(
            f"Moderate confidence ({confidence:.0%}) — recommend radiologist "
            "review to confirm findings."
        )

    # Uncertainty-based
    if uncertainty in ("high", "very_high"):
        steps.append(
            "High model uncertainty detected — findings should be interpreted "
            "with caution. Consider repeat imaging."
        )

    # Prediction-specific
    _RECOMMENDATION_MAP = {
        "Cardiomegaly": "Obtain echocardiogram for cardiac function assessment.",
        "Aortic enlargement": "Consider CT angiography to evaluate aortic dimensions.",
        "Pleural thickening": "Review occupational/exposure history. Consider CT for detailed assessment.",
        "Pulmonary fibrosis": "Refer to pulmonology. Consider high-resolution CT and pulmonary function tests.",
        "No finding": "No acute findings detected. Continue routine follow-up as clinically indicated.",
    }
    if prediction in _RECOMMENDATION_MAP:
        steps.append(_RECOMMENDATION_MAP[prediction])

    steps.append("Document findings in patient record and communicate with referring clinician.")

    return ToolResult(success=True, data={"next_steps": steps, "prediction": prediction})


# ═══════════════════════════════════════════════════════════════════════════════
# Factory — build a fully-configured ToolRegistry
# ═══════════════════════════════════════════════════════════════════════════════


def build_default_tool_registry() -> ToolRegistry:
    """Create a ToolRegistry with all built-in clinical tools."""
    registry = ToolRegistry()

    registry.register(ToolDescriptor(
        name="get_prediction_details",
        description="Get detailed prediction data including class probabilities, confidence, and uncertainty.",
        parameters={"analysis": "dict — current analysis data (auto-provided)"},
        func=_tool_get_prediction_details,
    ))

    registry.register(ToolDescriptor(
        name="get_xai_explanation",
        description="Get spatial XAI evidence: region activation scores, key findings, and heatmap availability.",
        parameters={"analysis": "dict — current analysis data (auto-provided)"},
        func=_tool_get_xai_explanation,
    ))

    registry.register(ToolDescriptor(
        name="get_evaluation_metrics",
        description="Load model evaluation metrics (AUC, F1, sensitivity, specificity, calibration) from saved reports.",
        parameters={"model_name": "str — architecture name (e.g. 'vit_base')"},
        func=_tool_get_evaluation_metrics,
    ))

    registry.register(ToolDescriptor(
        name="get_monitoring_status",
        description="Get drift monitoring stats: prediction count, feedback breakdown, correction rate, avg confidence.",
        parameters={},
        func=_tool_get_monitoring_status,
    ))

    registry.register(ToolDescriptor(
        name="generate_report",
        description="Generate a structured clinical report (findings, impression, recommendations) from the analysis.",
        parameters={"analysis": "dict — current analysis data (auto-provided)"},
        func=_tool_generate_report,
    ))

    registry.register(ToolDescriptor(
        name="compare_with_history",
        description="Compare current analysis with prior studies for the same patient (temporal comparison).",
        parameters={"analysis": "dict — current analysis data", "analysis_store": "dict — all stored analyses"},
        func=_tool_compare_with_history,
    ))

    registry.register(ToolDescriptor(
        name="suggest_next_steps",
        description="Suggest clinical next steps based on the prediction, confidence, and uncertainty levels.",
        parameters={"analysis": "dict — current analysis data (auto-provided)"},
        func=_tool_suggest_next_steps,
    ))

    return registry

"""Clinical Reasoning Agent – knowledge base, retrieval, reasoning, and reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lightweight modules (no heavy deps) — always available
from xclinvision.agent.tools import (  # noqa: E402
    ToolRegistry,
    ToolResult,
    ToolDescriptor,
    build_default_tool_registry,
)
from xclinvision.agent.reasoning import (  # noqa: E402
    ReasoningAgent,
    AgentResponse,
    ReasoningStep,
    classify_intent,
)
from xclinvision.agent.llm_provider import (  # noqa: E402
    LLMProvider,
    OpenAIProvider,
    LocalProvider,
)
from xclinvision.agent.llm_manager import (  # noqa: E402
    LLMManager,
    get_llm_manager,
    reset_llm_manager,
)

# Heavy agent sub-modules depend on optional packages (chromadb, sentence-
# transformers, etc.).  Import them lazily so that lightweight consumers
# (e.g. the backend importing only ClinicalContext) work without those
# packages installed.
try:
    from xclinvision.agent.audit import AuditTrail
    from xclinvision.agent.dialogue import ClinicalDialogueManager, SessionMemory
    from xclinvision.agent.guardrails import (
        GuardrailValidator,
        ThresholdProfile,
        build_clinical_threshold_profile,
    )
    from xclinvision.agent.ingest_knowledge import HybridRetriever
    from xclinvision.agent.reporter import ClinicalReporter
    from xclinvision.agent.xclinvisionagent import (
        ClinicalReasoningAgent,
        ClinicalReport,
        interpret_xai_regions,
    )
    _AGENT_DEPS_AVAILABLE = True
except Exception as _exc:
    _AGENT_DEPS_AVAILABLE = False
    logger.exception("Agent dependency import failed; rule-based fallback active.")


# ---------------------------------------------------------------------------
# ClinicalContext — lightweight data container used by the backend API
# ---------------------------------------------------------------------------

@dataclass
class ClinicalContext:
    """Container for clinical context passed from the backend to the agent."""

    prediction: str = ""
    probabilities: List[float] = field(default_factory=list)
    confidence: float = 0.0
    uncertainty_level: str = "unknown"
    highlighted_regions: List[str] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


# ---------------------------------------------------------------------------
# Lightweight agent wrapper returned by create_agent()
# ---------------------------------------------------------------------------

class _SimpleClinicalAgent:
    """Lightweight agent wrapper that bridges ClinicalContext to ClinicalReasoningAgent.

    When the full LLM agent fails to initialise (no API key, missing
    knowledge base, etc.) this wrapper produces rule-based reports so
    the platform remains functional without an LLM backend.
    """

    def __init__(self, full_agent: Optional[ClinicalReasoningAgent] = None):
        self._agent = full_agent

    def generate_report(self, context: ClinicalContext) -> Dict[str, Any]:
        """Generate a clinical report dict from a ClinicalContext.

        Returns a dict with keys: findings, impression, recommendation,
        uncertainty, urgency.
        """
        # Try the full LLM agent first
        if self._agent is not None:
            try:
                vision_data: Dict[str, Any] = {
                    "class_names": context.class_names or [context.prediction],
                    "class_names_predicted": [context.prediction] if context.prediction else [],
                    "probabilities": context.probabilities,
                    "confidence": context.confidence,
                    "explanation": {
                        "region_scores": {r: 0.5 for r in context.highlighted_regions},
                    },
                }
                patient_meta: Dict[str, Any] = {}
                if context.patient_age is not None:
                    patient_meta["age"] = context.patient_age
                if context.patient_sex is not None:
                    patient_meta["sex"] = context.patient_sex

                report = self._agent.generate_clinical_report(vision_data, patient_meta)
                return {
                    "findings": "; ".join(report.findings) if report.findings else "",
                    "impression": report.impression,
                    "recommendation": "; ".join(report.next_steps) if report.next_steps else "Clinical correlation recommended.",
                    "uncertainty": f"Uncertainty level: {context.uncertainty_level}",
                    "urgency": report.urgency,
                }
            except Exception as exc:
                logger.warning("Full LLM agent failed, falling back to rule-based report: %s", exc)

        # Rule-based fallback
        findings = f"AI analysis detected {context.prediction}" if context.prediction else "No significant findings detected"
        if context.confidence > 0:
            findings += f" with {context.confidence:.1%} confidence"
        findings += "."

        if context.highlighted_regions:
            regions_str = ", ".join(context.highlighted_regions[:3])
            findings += f" Key regions: {regions_str}."

        impression = findings
        recommendation = "Clinical correlation with patient history and physical examination recommended."
        if context.confidence < 0.5:
            recommendation += " Low confidence — consider additional imaging or specialist consultation."

        return {
            "findings": findings,
            "impression": impression,
            "recommendation": recommendation,
            "uncertainty": f"Uncertainty level: {context.uncertainty_level}",
            "urgency": "Medium" if context.confidence < 0.7 else "Low",
        }


def create_agent() -> _SimpleClinicalAgent:
    """Factory function to create a clinical agent.

    Tries to initialise the full LLM-powered ClinicalReasoningAgent with
    HybridRetriever and GuardrailValidator.  Falls back to a rule-based
    agent if dependencies (API key, vector DB, etc.) are unavailable.

    All LLM calls are routed through :func:`get_llm_manager`.
    """
    full_agent = None
    if _AGENT_DEPS_AVAILABLE:
        try:
            manager = get_llm_manager()

            # HybridRetriever needs an ingested vector store.
            # Default location mirrors ingest_knowledge.py conventions.
            import os
            from pathlib import Path
            vectorstore_dir = os.environ.get(
                "XCLINVISION_VECTORSTORE_DIR",
                str(Path(__file__).resolve().parents[3] / "data" / "vector_db"),
            )
            retriever = HybridRetriever(vectorstore_dir=vectorstore_dir)
            guardrails = GuardrailValidator()
            audit = AuditTrail()
            full_agent = ClinicalReasoningAgent(
                retriever=retriever,
                call_llm=manager.call_llm,
                guardrail_validator=guardrails,
                audit_trail=audit,
            )
            logger.info(
                "Full ClinicalReasoningAgent initialised (provider=%s).",
                manager.active_name,
            )
        except Exception as exc:
            logger.warning(
                "Could not initialise full LLM agent (%s). "
                "Using rule-based fallback — reports will be template-based.",
                exc,
            )
    else:
        logger.info("Agent dependencies not installed — using rule-based fallback.")

    return _SimpleClinicalAgent(full_agent=full_agent)


def create_reasoning_agent() -> ReasoningAgent:
    """Factory function to create a reasoning agent with tool-calling capabilities.

    When an LLM provider is available (via :class:`LLMManager`), the
    reasoning agent uses it for synthesis.  Otherwise, it falls back to
    deterministic rule-based logic.  All tools work regardless of LLM
    availability.
    """
    tool_registry = build_default_tool_registry()
    call_llm = None

    try:
        manager = get_llm_manager()
        if manager.active_provider is not None:
            call_llm = manager.call_llm
            logger.info(
                "ReasoningAgent initialised with LLM synthesis (provider=%s).",
                manager.active_name,
            )
        else:
            logger.info("No LLM providers available — ReasoningAgent using rule-based synthesis.")
    except Exception as exc:
        logger.warning("Could not initialise LLMManager for ReasoningAgent: %s", exc)

    return ReasoningAgent(tool_registry=tool_registry, call_llm=call_llm)


__all__ = [
    "ClinicalContext",
    "create_agent",
    "create_reasoning_agent",
    "ReasoningAgent",
    "AgentResponse",
    "ReasoningStep",
    "ToolRegistry",
    "ToolResult",
    "ToolDescriptor",
    "build_default_tool_registry",
    "classify_intent",
    "LLMProvider",
    "OpenAIProvider",
    "LocalProvider",
    "LLMManager",
    "get_llm_manager",
    "reset_llm_manager",
]

if _AGENT_DEPS_AVAILABLE:
    __all__ += [
        "AuditTrail",
        "ClinicalDialogueManager",
        "ClinicalReasoningAgent",
        "ClinicalReport",
        "ClinicalReporter",
        "GuardrailValidator",
        "HybridRetriever",
        "SessionMemory",
        "ThresholdProfile",
        "build_clinical_threshold_profile",
        "interpret_xai_regions",
    ]

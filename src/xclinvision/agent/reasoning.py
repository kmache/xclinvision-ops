"""Reasoning Agent — multi-step planner that interprets user intent and calls tools.

Architecture: **Plan → Execute → Synthesize**

1. **Intent Classification** — determine what the user is asking for.
2. **Action Planning** — select which tools to call and in what order.
3. **Execution** — run tools, collect results.
4. **Synthesis** — combine tool outputs into a coherent, grounded response.

The reasoning loop supports both LLM-powered and rule-based modes.
When an LLM (OpenAI-compatible) is available, it drives the planning and
synthesis steps.  Without an LLM, a deterministic rule-based planner
handles common intent patterns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from xclinvision.agent.tools import ToolRegistry, ToolResult, build_default_tool_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent types the agent can recognise
# ---------------------------------------------------------------------------

INTENT_TYPES = Literal[
    "explain_prediction",
    "explain_heatmap",
    "suggest_next_steps",
    "assess_urgency",
    "compare_history",
    "get_metrics",
    "get_monitoring",
    "generate_report",
    "general_question",
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning trace."""

    step: str           # "intent", "plan", "tool_call", "synthesize"
    detail: str         # What happened
    data: Any = None    # Optional payload


@dataclass
class AgentResponse:
    """Structured response from the reasoning agent."""

    response: str
    intent: str = "general_question"
    tools_used: List[str] = field(default_factory=list)
    reasoning_trace: List[ReasoningStep] = field(default_factory=list)
    suggested_followups: List[str] = field(default_factory=list)
    report_data: Optional[Dict[str, Any]] = None

    def to_api_dict(self) -> Dict[str, Any]:
        """Serialize for the REST API response."""
        return {
            "response": self.response,
            "intent": self.intent,
            "tools_used": self.tools_used,
            "reasoning_trace": [
                {"step": s.step, "detail": s.detail} for s in self.reasoning_trace
            ],
            "suggested_followups": self.suggested_followups,
            "report_data": self.report_data,
            "references": [],  # backwards-compat with v2 chat contract
        }


# ---------------------------------------------------------------------------
# Intent detection patterns (rule-based fallback)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: List[tuple] = [
    ("explain_heatmap", [
        r"\bheatmap\b", r"\bgrad.?cam\b", r"\bactivation\b", r"\bregion\b",
        r"\bspatial\b", r"\bwhere\b.*\b(show|highlight|activat)",
        r"\bexplain.*(?:image|xai|overlay)\b",
    ]),
    ("explain_prediction", [
        r"\bwhat\b.*\b(mean|finding|detect|diagnos)\b",
        r"\bexplain\b.*\b(predict|result|finding|output)\b",
        r"\bwhy\b.*\b(predict|detect|flag)\b",
        r"\btell me\b.*\babout\b.*\b(finding|result)\b",
        r"\bwhat\b.*\bcould\b.*\bmean\b",
    ]),
    ("suggest_next_steps", [
        r"\bnext\s*step\b", r"\brecommend\b", r"\bwhat\s+should\b",
        r"\bfollow.?up\b", r"\bwhat\s+to\s+do\b", r"\baction\b",
        r"\bpropose\b.*\bstep\b",
    ]),
    ("assess_urgency", [
        r"\burgent\b", r"\burgency\b", r"\bcritical\b", r"\bemergenc\b",
        r"\bis\s+this\s+urgent\b", r"\bhow\s+serious\b", r"\brisk\b.*\blevel\b",
        r"\bsever\b",
    ]),
    ("compare_history", [
        r"\bhistory\b", r"\bprior\b", r"\bprevious\b", r"\bchange\b",
        r"\bprogress\b", r"\bcompar\b", r"\bover\s+time\b", r"\btemporal\b",
        r"\btrend\b",
    ]),
    ("get_metrics", [
        r"\bmetric\b", r"\bauc\b", r"\bf1\b", r"\baccuracy\b",
        r"\bperformance\b", r"\bsensitivity\b", r"\bspecificity\b",
        r"\bcalibration\b", r"\beval\b",
    ]),
    ("get_monitoring", [
        r"\bdrift\b", r"\bmonitor\b", r"\bfeedback\b.*\bstat\b",
        r"\bcorrection\s+rate\b", r"\bhow\s+many\b.*\bpredict\b",
    ]),
    ("generate_report", [
        r"\breport\b", r"\bgenerat\b.*\breport\b", r"\bsummary\b.*\breport\b",
        r"\bclinical\s+report\b", r"\bwrite\b.*\breport\b",
    ]),
]


def classify_intent(message: str) -> str:
    """Rule-based intent classification from user message."""
    msg_lower = message.lower()
    for intent, patterns in _INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, msg_lower):
                return intent
    return "general_question"


# ---------------------------------------------------------------------------
# Action plans per intent (rule-based)
# ---------------------------------------------------------------------------

_INTENT_TOOL_MAP: Dict[str, List[str]] = {
    "explain_prediction": ["get_prediction_details", "get_xai_explanation"],
    "explain_heatmap": ["get_xai_explanation", "get_prediction_details"],
    "suggest_next_steps": ["suggest_next_steps", "get_prediction_details"],
    "assess_urgency": ["get_prediction_details", "suggest_next_steps"],
    "compare_history": ["compare_with_history", "get_prediction_details"],
    "get_metrics": ["get_evaluation_metrics"],
    "get_monitoring": ["get_monitoring_status"],
    "generate_report": ["generate_report"],
    "general_question": ["get_prediction_details"],
}

_INTENT_FOLLOWUPS: Dict[str, List[str]] = {
    "explain_prediction": [
        "Explain the heatmap regions",
        "Is this urgent?",
        "What are the recommended next steps?",
    ],
    "explain_heatmap": [
        "What does this prediction mean?",
        "How reliable is this model?",
        "Suggest next steps",
    ],
    "suggest_next_steps": [
        "Is this urgent?",
        "Show model performance metrics",
        "Generate a clinical report",
    ],
    "assess_urgency": [
        "What are the next steps?",
        "Explain the prediction in detail",
        "Compare with prior studies",
    ],
    "compare_history": [
        "What changed since last time?",
        "Is this progressing?",
        "Suggest follow-up actions",
    ],
    "get_metrics": [
        "How does this affect my current result?",
        "Is the model well-calibrated?",
        "Explain the prediction",
    ],
    "get_monitoring": [
        "Show model metrics",
        "Are there any concerns?",
        "Generate a report",
    ],
    "generate_report": [
        "Export as PDF",
        "Explain the findings",
        "What follow-up is needed?",
    ],
    "general_question": [
        "Explain the heatmap",
        "Is this urgent?",
        "What could this mean?",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Reasoning Agent
# ═══════════════════════════════════════════════════════════════════════════════


class ReasoningAgent:
    """Multi-step reasoning agent with tool-calling capabilities.

    Parameters
    ----------
    tool_registry:
        The registry of available tools.  Defaults to the built-in set.
    call_llm:
        Optional LLM caller ``(system, user, *, temperature) -> str``.
        When provided, the agent uses LLM for planning and synthesis.
        Without it, deterministic rule-based logic is used.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        call_llm: Optional[Callable[..., str]] = None,
    ) -> None:
        self.tools = tool_registry or build_default_tool_registry()
        self.call_llm = call_llm

    # ── Public API ─────────────────────────────────────────────────────

    def process_message(
        self,
        message: str,
        analysis: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Process a user message through the full reasoning loop.

        Steps:
        1. Classify intent
        2. Plan tool calls
        3. Execute tools
        4. Synthesize response

        Parameters
        ----------
        message:
            The user's question or command.
        analysis:
            The current analysis dict (from ``_analysis_store``).
        history:
            Conversation history ``[{"role": ..., "content": ...}, ...]``.
        extra_context:
            Additional context (e.g., ``_feedback_store``, ``_analysis_store``).

        Returns
        -------
        AgentResponse
            Structured response with text, trace, and follow-ups.
        """
        trace: List[ReasoningStep] = []
        extra_context = extra_context or {}

        # ── Step 1: Intent classification ─────────────────────────────
        intent = classify_intent(message)
        trace.append(ReasoningStep(
            step="intent",
            detail=f"Classified user intent as '{intent}'",
        ))

        # ── Step 2: Plan tool calls ───────────────────────────────────
        tool_names = _INTENT_TOOL_MAP.get(intent, ["get_prediction_details"])
        trace.append(ReasoningStep(
            step="plan",
            detail=f"Planned tools: {', '.join(tool_names)}",
        ))

        # ── Step 3: Execute tools ─────────────────────────────────────
        tool_context = {**(extra_context or {})}
        if analysis:
            tool_context["analysis"] = analysis
            tool_context["model_name"] = analysis.get("model_version", "vit_base")

        tool_results: Dict[str, ToolResult] = {}
        for name in tool_names:
            result = self.tools.execute(name, tool_context)
            tool_results[name] = result
            trace.append(ReasoningStep(
                step="tool_call",
                detail=f"Executed '{name}' → {'OK' if result.success else 'FAILED'}",
                data=result.data if result.success else result.error,
            ))

        # ── Step 4: Synthesize response ───────────────────────────────
        if self.call_llm:
            response_text = self._llm_synthesize(
                message, intent, tool_results, analysis, history, trace,
            )
        else:
            response_text = self._rule_synthesize(
                message, intent, tool_results, analysis,
            )

        trace.append(ReasoningStep(
            step="synthesize",
            detail="Generated final response",
        ))

        # Extract report data if a report was generated
        report_data = None
        if "generate_report" in tool_results:
            rr = tool_results["generate_report"]
            if rr.success:
                report_data = rr.data

        followups = _INTENT_FOLLOWUPS.get(intent, _INTENT_FOLLOWUPS["general_question"])

        return AgentResponse(
            response=response_text,
            intent=intent,
            tools_used=list(tool_results.keys()),
            reasoning_trace=trace,
            suggested_followups=followups,
            report_data=report_data,
        )

    # ── LLM-powered synthesis ─────────────────────────────────────────

    _SYNTH_SYSTEM = """\
You are XClinVision's clinical reasoning assistant. You have just executed \
tools to gather evidence about a chest X-ray analysis. Your task is to \
synthesize the tool outputs into a clear, grounded, professional response.

RULES:
- You are an ASSISTIVE tool. You do NOT replace a radiologist.
- Ground every claim in the tool outputs provided. Do NOT invent findings.
- When uncertain, say so explicitly.
- Use evidence-based clinical language.
- Be concise but thorough.
- If tool data is missing or errored, acknowledge it gracefully.
"""

    def _llm_synthesize(
        self,
        message: str,
        intent: str,
        tool_results: Dict[str, ToolResult],
        analysis: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, str]]],
        trace: List[ReasoningStep],
    ) -> str:
        """Use an LLM to synthesize tool outputs into a response."""
        # Build tool output block
        tool_block = ""
        for name, result in tool_results.items():
            tool_block += f"\n### Tool: {name}\n{result.to_prompt_text()}\n"

        # Build history block
        history_block = ""
        if history:
            recent = history[-6:]
            history_block = "\n".join(
                f"{'User' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')}"
                for m in recent
            )

        user_prompt = f"""\
### User's Question
{message}

### Detected Intent
{intent}

### Tool Outputs
{tool_block}

### Conversation History
{history_block or 'No prior conversation.'}

---
Provide a focused, professional response to the user's question. \
Reference the tool outputs as evidence. Keep it concise."""

        try:
            return self.call_llm(self._SYNTH_SYSTEM, user_prompt, temperature=0.25)
        except Exception as exc:
            logger.warning("LLM synthesis failed: %s — falling back to rules", exc)
            return self._rule_synthesize(message, intent, tool_results, analysis)

    # ── Rule-based synthesis (no LLM needed) ──────────────────────────

    def _rule_synthesize(
        self,
        message: str,
        intent: str,
        tool_results: Dict[str, ToolResult],
        analysis: Optional[Dict[str, Any]],
    ) -> str:
        """Deterministic synthesis from tool results — works without LLM."""
        parts: List[str] = []

        if intent == "explain_prediction":
            parts.append(self._format_prediction_explanation(tool_results))
        elif intent == "explain_heatmap":
            parts.append(self._format_heatmap_explanation(tool_results))
        elif intent == "suggest_next_steps":
            parts.append(self._format_next_steps(tool_results))
        elif intent == "assess_urgency":
            parts.append(self._format_urgency_assessment(tool_results))
        elif intent == "compare_history":
            parts.append(self._format_history_comparison(tool_results))
        elif intent == "get_metrics":
            parts.append(self._format_metrics(tool_results))
        elif intent == "get_monitoring":
            parts.append(self._format_monitoring(tool_results))
        elif intent == "generate_report":
            parts.append(self._format_report_result(tool_results))
        else:
            # General: show prediction context
            parts.append(self._format_prediction_explanation(tool_results))

        return "\n\n".join(p for p in parts if p)

    # ── Formatting helpers ────────────────────────────────────────────

    @staticmethod
    def _format_prediction_explanation(results: Dict[str, ToolResult]) -> str:
        pred = results.get("get_prediction_details")
        if not pred or not pred.success:
            return "Unable to retrieve prediction details at this time."

        d = pred.data
        lines = [
            f"**AI Prediction: {d['prediction']}**",
            f"- Confidence: {d['confidence']:.1%}",
            f"- Uncertainty: {d['uncertainty_level']}",
            f"- Model: {d['model_version']}",
        ]
        if d.get("top_predictions"):
            lines.append("\nProbability breakdown:")
            for p in d["top_predictions"]:
                bar = "█" * int(p["probability"] * 20)
                lines.append(f"  {p['class']}: {p['probability']:.2%} {bar}")

        xai = results.get("get_xai_explanation")
        if xai and xai.success:
            spatial = xai.data.get("spatial_interpretation", {})
            if spatial:
                lines.append("\n**Spatial evidence (XAI):**")
                for region, desc in spatial.items():
                    lines.append(f"  • {region}: {desc}")
            findings = xai.data.get("key_findings", [])
            if findings:
                lines.append("\n**Key findings:**")
                for f in findings:
                    lines.append(f"  • {f}")

        lines.append(
            "\n*This AI analysis is assistive only. "
            "Clinical correlation and radiologist review are essential.*"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_heatmap_explanation(results: Dict[str, ToolResult]) -> str:
        xai = results.get("get_xai_explanation")
        if not xai or not xai.success:
            return "Unable to retrieve XAI explanation at this time."

        d = xai.data
        lines = ["**Grad-CAM++ Spatial Analysis:**"]

        spatial = d.get("spatial_interpretation", {})
        if spatial:
            for region, desc in spatial.items():
                lines.append(f"  • **{region}**: {desc}")
        else:
            lines.append("  No significant regional activations detected.")

        region_scores = d.get("region_scores", {})
        if region_scores:
            top_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("\nTop activation regions (raw scores):")
            for name, score in top_regions:
                bar = "█" * int(score * 20)
                lines.append(f"  {name}: {score:.3f} {bar}")

        if d.get("has_heatmap"):
            lines.append("\nA visual heatmap overlay is available in the analysis view.")

        findings = d.get("key_findings", [])
        if findings:
            lines.append("\n**Key findings from XAI:**")
            for f in findings:
                lines.append(f"  • {f}")

        return "\n".join(lines)

    @staticmethod
    def _format_next_steps(results: Dict[str, ToolResult]) -> str:
        ns = results.get("suggest_next_steps")
        if not ns or not ns.success:
            return "Unable to generate recommendations at this time. Please consult a radiologist."

        d = ns.data
        lines = [f"**Recommended Next Steps** (for {d.get('prediction', 'current finding')}):"]
        for i, step in enumerate(d.get("next_steps", []), 1):
            lines.append(f"  {i}. {step}")
        return "\n".join(lines)

    @staticmethod
    def _format_urgency_assessment(results: Dict[str, ToolResult]) -> str:
        pred = results.get("get_prediction_details")
        ns = results.get("suggest_next_steps")

        if not pred or not pred.success:
            return "Unable to assess urgency without prediction data."

        d = pred.data
        confidence = d["confidence"]
        uncertainty = d["uncertainty_level"]
        prediction = d["prediction"]

        # Rule-based urgency assessment
        if prediction == "No finding":
            urgency = "Low"
            assessment = "No acute findings detected. Routine follow-up as indicated."
        elif confidence >= 0.8 and uncertainty in ("low", "very_low"):
            urgency = "High" if prediction in ("Aortic enlargement",) else "Medium"
            assessment = (
                f"High-confidence detection of {prediction}. "
                "Timely clinical review recommended."
            )
        elif confidence >= 0.5:
            urgency = "Medium"
            assessment = (
                f"Moderate-confidence finding ({prediction}). "
                "Radiologist confirmation advised."
            )
        else:
            urgency = "Low"
            assessment = (
                f"Low-confidence finding ({confidence:.0%}). "
                "Consider additional imaging or alternative diagnosis."
            )

        lines = [
            f"**Urgency Assessment: {urgency}**",
            "",
            assessment,
            f"- Prediction: {prediction}",
            f"- Confidence: {confidence:.1%}",
            f"- Uncertainty: {uncertainty}",
        ]

        if ns and ns.success:
            steps = ns.data.get("next_steps", [])[:3]
            if steps:
                lines.append("\n**Immediate actions:**")
                for s in steps:
                    lines.append(f"  • {s}")

        return "\n".join(lines)

    @staticmethod
    def _format_history_comparison(results: Dict[str, ToolResult]) -> str:
        hist = results.get("compare_with_history")
        if not hist or not hist.success:
            return hist.error if hist else "Unable to retrieve patient history."

        d = hist.data
        if not d.get("has_prior"):
            return d.get("message", "No prior studies available for comparison.")

        cur = d["current"]
        prev = d["prior"]
        lines = [
            "**Temporal Comparison:**",
            "",
            f"**Current study** ({cur.get('timestamp', 'now')[:10]}):",
            f"  Prediction: {cur['prediction']} | Confidence: {cur['confidence']:.1%}",
            "",
            f"**Prior study** ({prev.get('timestamp', 'unknown')[:10]}):",
            f"  Prediction: {prev['prediction']} | Confidence: {prev['confidence']:.1%}",
        ]

        if cur["prediction"] != prev["prediction"]:
            lines.append(f"\n⚠ Finding changed from {prev['prediction']} to {cur['prediction']}.")
        elif cur["confidence"] > prev.get("confidence", 0) + 0.1:
            lines.append("\n↑ Confidence has increased — finding may be more definitive.")
        elif cur["confidence"] < prev.get("confidence", 0) - 0.1:
            lines.append("\n↓ Confidence has decreased — consider reassessment.")
        else:
            lines.append("\nFindings appear relatively stable.")

        lines.append(f"\nTotal prior studies: {d.get('total_prior_studies', 0)}")
        return "\n".join(lines)

    @staticmethod
    def _format_metrics(results: Dict[str, ToolResult]) -> str:
        met = results.get("get_evaluation_metrics")
        if not met or not met.success:
            return f"Unable to load metrics: {met.error if met else 'unknown error'}"

        d = met.data
        lines = [
            f"**Model Performance: {d.get('model', 'unknown')}**",
            "",
            f"- Macro AUC: {d.get('macro_auc', 'N/A'):.4f}" if d.get('macro_auc') else "- Macro AUC: N/A",
            f"- Macro F1: {d.get('macro_f1', 'N/A'):.4f}" if d.get('macro_f1') else "- Macro F1: N/A",
            f"- Subset Accuracy: {d.get('subset_accuracy', 'N/A'):.4f}" if d.get('subset_accuracy') else "- Subset Accuracy: N/A",
        ]

        if d.get("ece") is not None:
            lines.append(f"- ECE (calibration): {d['ece']:.4f}")

        per_class = d.get("per_class", {})
        if per_class:
            lines.append("\n**Per-class performance:**")
            for cls, vals in per_class.items():
                auc = vals.get("auc")
                sens = vals.get("sensitivity")
                spec = vals.get("specificity")
                cls_line = f"  {cls}: AUC={auc:.3f}" if auc else f"  {cls}:"
                if sens is not None:
                    cls_line += f" | Sens={sens:.3f}"
                if spec is not None:
                    cls_line += f" | Spec={spec:.3f}"
                lines.append(cls_line)

        return "\n".join(lines)

    @staticmethod
    def _format_monitoring(results: Dict[str, ToolResult]) -> str:
        mon = results.get("get_monitoring_status")
        if not mon or not mon.success:
            return "Unable to retrieve monitoring data."

        d = mon.data
        lines = [
            "**System Monitoring Status:**",
            "",
            f"- Total predictions: {d.get('total_predictions', 0)}",
            f"- Total feedback: {d.get('total_feedback', 0)}",
            f"- Correction rate: {d.get('correction_rate', 0):.1f}%",
            f"- Avg confidence: {d.get('avg_confidence', 0):.2%}",
        ]

        by_type = d.get("feedback_by_type", {})
        if by_type:
            lines.append("\nFeedback breakdown:")
            for ft, count in by_type.items():
                lines.append(f"  • {ft}: {count}")

        return "\n".join(lines)

    @staticmethod
    def _format_report_result(results: Dict[str, ToolResult]) -> str:
        rpt = results.get("generate_report")
        if not rpt or not rpt.success:
            return f"Report generation failed: {rpt.error if rpt else 'unknown error'}"

        d = rpt.data
        lines = ["**Clinical Report Generated:**", ""]

        findings = d.get("findings", "")
        if findings:
            lines.append(f"**Findings:** {findings}")

        impression = d.get("impression", "")
        if impression:
            lines.append(f"\n**Impression:** {impression}")

        recommendation = d.get("recommendation", "")
        if recommendation:
            lines.append(f"\n**Recommendation:** {recommendation}")

        uncertainty = d.get("uncertainty", "")
        if uncertainty:
            lines.append(f"\n**Uncertainty:** {uncertainty}")

        lines.append(
            "\n*You can export this report as PDF or JSON from the Report page.*"
        )
        return "\n".join(lines)

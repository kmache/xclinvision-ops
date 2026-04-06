"""Clinical Dialogue Manager – interactive reasoning with session memory.

Enables follow-up questions with context-aware, grounded responses.

Mechanism
---------
- **Session Memory** – stores the last report, reasoning trace, and
  conversation history.
- **Follow-up Handling** – when a user asks a question:
  1. Retrieve fresh knowledge (hybrid search) based on query + current context.
  2. Re-use previous reasoning from session memory.
  3. Generate a focused explanation referencing spatial evidence, retrieved
     sources, and original findings.
  4. If the query involves temporal comparison, incorporate ``previous_study_data``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from xclinvision.agent.ingest_knowledge import HybridRetriever

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# 1.  Session Memory
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class DialogueTurn:
    """A single turn in the clinical conversation."""

    role: str  # "user" or "assistant"
    content: str

@dataclass
class SessionMemory:
    """Stores clinical context for a single analysis session.

    Mirrors what would live in ``st.session_state`` in the Streamlit app,
    but framework-agnostic so it can also be used from the REST API.
    """

    report_json: Optional[Dict[str, Any]] = None
    reasoning_trace: Optional[str] = None
    spatial_evidence: Optional[Dict[str, str]] = None
    vision_data: Optional[Dict[str, Any]] = None
    patient_meta: Optional[Dict[str, Any]] = None
    previous_study_data: Optional[Dict[str, Any]] = None
    conversation_history: List[DialogueTurn] = field(default_factory=list)

    # ── helpers ───────────────────────────────────────────────────────

    def store_report(self, report_dict: Dict[str, Any]) -> None:
        """Cache the latest ClinicalReport as a dict."""
        self.report_json = report_dict
        self.reasoning_trace = report_dict.get("reasoning_trace")
        self.spatial_evidence = report_dict.get("spatial_evidence")

    def add_turn(self, role: str, content: str) -> None:
        self.conversation_history.append(DialogueTurn(role=role, content=content))

    def get_history_text(self, max_turns: int = 10) -> str:
        """Format recent conversation history for prompt injection."""
        recent = self.conversation_history[-max_turns:]
        if not recent:
            return "No prior conversation."
        return "\n".join(
            f"{'User' if t.role == 'user' else 'AI'}: {t.content}" for t in recent
        )

    def get_context_summary(self) -> str:
        """Compact summary of the current clinical context."""
        parts: List[str] = []
        if self.report_json:
            findings = self.report_json.get("findings", [])
            if findings:
                parts.append(f"Current findings: {', '.join(findings)}")
            impression = self.report_json.get("impression", "")
            if impression:
                parts.append(f"Impression: {impression}")
            urgency = self.report_json.get("urgency", "")
            if urgency:
                parts.append(f"Urgency: {urgency}")
        if self.spatial_evidence:
            spatial_parts = [f"{k}: {v}" for k, v in self.spatial_evidence.items()]
            parts.append(f"Spatial evidence: {'; '.join(spatial_parts)}")
        if self.patient_meta:
            meta_parts = [f"{k}={v}" for k, v in self.patient_meta.items() if v]
            if meta_parts:
                parts.append(f"Patient: {', '.join(meta_parts)}")
        return "\n".join(parts) if parts else "No clinical context available."

    def clear(self) -> None:
        """Reset session memory."""
        self.report_json = None
        self.reasoning_trace = None
        self.spatial_evidence = None
        self.vision_data = None
        self.patient_meta = None
        self.previous_study_data = None
        self.conversation_history.clear()


# ════════════════════════════════════════════════════════════════════════════════
# 2.  Prompt templates for dialogue
# ════════════════════════════════════════════════════════════════════════════════

_FOLLOW_UP_SYSTEM_PROMPT = """\
You are an expert chest-X-ray clinical reasoning assistant within the \
XClinVision decision-support platform.  You are responding to a follow-up \
question from a clinician about a previously generated AI report.

IMPORTANT SAFETY RULES:
- You are an ASSISTIVE tool only.  You do NOT replace a radiologist.
- Every claim MUST be grounded in the provided evidence (vision scores, XAI \
  regions, retrieved context, or original report).  Do NOT invent findings.
- When uncertain, say so explicitly and recommend human review.
- Use evidence-based language at all times.
"""

_FOLLOW_UP_USER_TEMPLATE = """\
### Current Clinical Context
{context_summary}

### Previous Reasoning
{reasoning_trace}

### Conversation History
{conversation_history}

### Retrieved Knowledge (fresh hybrid search)
{rag_block}

{temporal_block}

### User's Follow-up Question
{user_question}

---
Provide a focused, grounded answer that:
1. References specific spatial evidence and regions from the original analysis.
2. Cites retrieved sources by title/ID where applicable.
3. Acknowledges uncertainty when evidence is weak or conflicting.
4. If the question involves temporal comparison and prior data is available, \
   discuss changes over time.
5. If the question asks about risk level or urgency, explain the evidence \
   that supports the assigned urgency.

Respond in clear, professional clinical language.
"""

_TEMPORAL_QUERY_KEYWORDS = {
    "change", "changed", "progression", "progressed", "interval",
    "compared", "comparison", "previous", "prior", "worse", "better",
    "improved", "stable", "new", "resolved", "serial", "longitudinal",
    "follow-up", "follow up", "over time", "trend",
}


# ════════════════════════════════════════════════════════════════════════════════
# 3.  ClinicalDialogueManager
# ════════════════════════════════════════════════════════════════════════════════
class ClinicalDialogueManager:
    """Handles interactive follow-up questions with context-aware, grounded responses.

    Parameters
    ----------
    retriever:
        A :class:`HybridRetriever` for querying the clinical knowledge base.
    call_llm:
        LLM calling function ``(system, user, *, temperature) -> str``.
    rag_top_k:
        Number of RAG chunks to retrieve per follow-up query.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        call_llm: Optional[Callable[..., str]] = None,
        rag_top_k: int = 5,
        audit_trail: Optional[Any] = None,
    ) -> None:
        self.retriever = retriever
        self.rag_top_k = rag_top_k
        self.audit_trail = audit_trail

        if call_llm is not None:
            self.call_llm = call_llm
        else:
            # Lazy import to avoid hard dependency
            from xclinvision.agent.xclinvisionagent import _default_call_llm
            self.call_llm = _default_call_llm

    def handle_follow_up(
        self,
        question: str,
        session: SessionMemory,
    ) -> str:
        """Process a follow-up question and return a grounded response.

        Steps:
        1. Build an augmented query from the question + current context.
        2. Run hybrid search for fresh knowledge.
        3. Detect if temporal comparison is relevant.
        4. Combine prior reasoning + new knowledge into a focused answer.
        5. Store the exchange in session memory.

        Parameters
        ----------
        question:
            The user's follow-up question.
        session:
            The current :class:`SessionMemory` with report + history.

        Returns
        -------
        str
            The assistant's response.
        """
        # 1. Build augmented query
        augmented_query = self._build_augmented_query(question, session)

        # 2. Retrieve fresh knowledge
        rag_results = self.retriever.query(augmented_query, top_k=self.rag_top_k)
        rag_block = self._format_rag_results(rag_results)

        # 3. Check for temporal query
        is_temporal = self._is_temporal_query(question)
        temporal_block = ""
        if is_temporal and session.previous_study_data:
            temporal_block = self._format_temporal_context(session.previous_study_data)
        elif is_temporal:
            temporal_block = (
                "### Temporal Context\n"
                "No prior study data is available for temporal comparison. "
                "Recommend obtaining prior imaging for longitudinal assessment."
            )

        # 4. Build and send the prompt
        context_summary = session.get_context_summary()
        reasoning_trace = session.reasoning_trace or "No prior reasoning available."
        conversation_history = session.get_history_text()

        user_prompt = _FOLLOW_UP_USER_TEMPLATE.format(
            context_summary=context_summary,
            reasoning_trace=reasoning_trace,
            conversation_history=conversation_history,
            rag_block=rag_block,
            temporal_block=temporal_block,
            user_question=question,
        )

        response = self.call_llm(
            _FOLLOW_UP_SYSTEM_PROMPT, user_prompt, temperature=0.25
        )

        # 5. Store the exchange
        session.add_turn("user", question)
        session.add_turn("assistant", response)

        # 6. Audit trail
        if self.audit_trail is not None:
            try:
                rag_sources = [
                    r.get("metadata", {}).get("source", "")
                    for r in rag_results
                ]
                self.audit_trail.log_interaction(
                    question=question,
                    response=response,
                    context_summary=session.get_context_summary(),
                    rag_sources=rag_sources,
                )
            except Exception:
                logger.warning("Failed to log follow-up interaction to audit trail.", exc_info=True)

        logger.info(
            "Follow-up answered | temporal=%s | rag_hits=%d | history_len=%d",
            is_temporal,
            len(rag_results),
            len(session.conversation_history),
        )
        return response

    # ── internal helpers ──────────────────────────────────────────────
    def _build_augmented_query(self, question: str, session: SessionMemory) -> str:
        """Combine the user's question with current findings for better retrieval."""
        parts = [question]
        if session.report_json:
            findings = session.report_json.get("findings", [])
            parts.extend(findings[:3])  # Top findings for context
        return " ".join(parts)

    @staticmethod
    def _is_temporal_query(question: str) -> bool:
        """Detect if the question involves temporal / longitudinal comparison."""
        q_lower = question.lower()
        return any(kw in q_lower for kw in _TEMPORAL_QUERY_KEYWORDS)

    @staticmethod
    def _format_rag_results(results: List[Dict[str, Any]]) -> str:
        """Format RAG results into a readable block."""
        if not results:
            return "No relevant clinical context retrieved from knowledge base."
        lines: List[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "unknown")
            condition = r.get("metadata", {}).get("condition", "")
            doc_id = r.get("metadata", {}).get("doc_id", r.get("chunk_id", ""))
            text = r.get("text", "")[:400]
            cite = f"[{i}] {source}"
            if condition:
                cite += f" ({condition})"
            if doc_id:
                cite += f" — {doc_id}"
            lines.append(f"{cite}:\n{text}")
        return "\n\n".join(lines)

    @staticmethod
    def _format_temporal_context(prev_study: Dict[str, Any]) -> str:
        """Format prior study data for temporal comparison."""
        parts = ["### Temporal Context (Prior Study)"]
        if prev_study.get("date"):
            parts.append(f"**Prior study date:** {prev_study['date']}")
        if prev_study.get("findings"):
            parts.append(f"**Prior findings:** {prev_study['findings']}")
        if prev_study.get("impression"):
            parts.append(f"**Prior impression:** {prev_study['impression']}")
        return "\n".join(parts)

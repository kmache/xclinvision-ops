"""Clinical Reasoning Agent – LLM-powered synthesis of vision, XAI, and RAG evidence.

This module implements the **Agent Brain** for XClinVision.  It orchestrates a
multi-step Chain-of-Thought reasoning loop that:

1. Analyses vision-model predictions against patient metadata.
2. Translates spatial heatmap activations into clinical language.
3. Retrieves guidelines / case summaries via :class:`HybridRetriever`.
4. Synthesises a differential diagnosis with cross-verification.
5. Enforces structured output via the :class:`ClinicalReport` Pydantic model.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from xclinvision.agent.audit import AuditTrail
from xclinvision.agent.guardrails import GuardrailValidator
from xclinvision.agent.ingest_knowledge import HybridRetriever

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# 1.  Pydantic schema
# ════════════════════════════════════════════════════════════════════════════════


class ClinicalReport(BaseModel):
    """Structured clinical report produced by the reasoning agent."""

    findings: List[str] = Field(description="Detected abnormalities, e.g. 'Cardiomegaly, Aortic enlargement'")
    spatial_evidence: Dict[str, str] = Field(description="Anatomical region → activation description")
    reasoning_trace: str = Field(description="Chain-of-Thought explanation")
    differential_diagnosis: List[str] = Field(description="Ranked differential diagnoses")
    impression: str = Field(description="Summary impression for the radiologist")
    urgency: Literal["Low", "Medium", "High"] = Field(description="Clinical urgency level")
    next_steps: List[str] = Field(description="Recommended follow-up actions")
    citations: List[str] = Field(description="Sources from RAG context")
    requires_human_review: bool = Field(default=False, description="Flagged for radiologist review")
    temporal_changes: Optional[str] = Field(default=None, description="Comparison with prior study, if available")


# ════════════════════════════════════════════════════════════════════════════════
# 2.  XAI spatial interpreter
# ════════════════════════════════════════════════════════════════════════════════

# Human-readable names for the region keys used by xai.score_lung_regions()
_REGION_LABELS: Dict[str, str] = {
    "right_upper": "right upper lobe",
    "right_middle": "right middle lobe",
    "right_lower": "right lower lobe",
    "left_upper": "left upper lobe",
    "left_middle": "left middle lobe",
    "left_lower": "left lower lobe",
    "hilar": "hilar region",
    "cardiac": "cardiac silhouette",
    "apical": "bilateral apices",
}

_HIGH_ACTIVATION_THRESHOLD = 0.30
_MODERATE_ACTIVATION_THRESHOLD = 0.15


def interpret_xai_regions(
    region_scores: Dict[str, float],
    *,
    high_thresh: float = _HIGH_ACTIVATION_THRESHOLD,
    moderate_thresh: float = _MODERATE_ACTIVATION_THRESHOLD,
) -> Dict[str, str]:
    """Convert numeric region activation scores into clinical spatial language.

    Returns a mapping of *human-readable region name* → description string,
    e.g. ``{"right lower lobe": "high activation (0.72)"}``.
    """
    evidence: Dict[str, str] = {}
    if not region_scores:
        return evidence

    for key, score in sorted(region_scores.items(), key=lambda kv: kv[1], reverse=True):
        label = _REGION_LABELS.get(key, key.replace("_", " "))
        if score >= high_thresh:
            evidence[label] = f"high activation ({score:.2f})"
        elif score >= moderate_thresh:
            evidence[label] = f"moderate activation ({score:.2f})"
    return evidence


def _summarise_spatial_evidence(spatial: Dict[str, str]) -> str:
    """One-liner summary used inside prompts."""
    if not spatial:
        return "No significant regional activation detected."
    parts = [f"{region}: {desc}" for region, desc in spatial.items()]
    return "; ".join(parts)


# ════════════════════════════════════════════════════════════════════════════════
# 3.  Prompt templates
# ════════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are an expert chest-X-ray clinical reasoning assistant integrated into \
the XClinVision decision-support platform.  Your task is to synthesise \
findings from an AI vision model, spatial explainability evidence, \
patient metadata, and retrieved medical literature into a structured \
clinical report.

IMPORTANT SAFETY RULES:
- You are an ASSISTIVE tool only. You do NOT replace a radiologist.
- Every claim MUST be grounded in the provided evidence (vision scores, XAI \
  regions, or retrieved context). Do NOT invent findings.
- If evidence is conflicting or weak, flag it honestly.
- Output MUST conform to the JSON schema provided.
"""

_COT_PROMPT_TEMPLATE = """\
### STEP 1 — Observation Analysis
**Vision-model predictions** (class → probability):
{predictions_block}

**Patient metadata**: {patient_meta_block}

### STEP 2 — Spatial Evidence (XAI)
{spatial_block}

### STEP 3 — Retrieved Clinical Context (RAG)
{rag_block}

### STEP 4 — Temporal / Prior Study Data
{temporal_block}

---
Using a step-by-step Chain-of-Thought approach:
1. **Observation Analysis** – State which AI predictions are consistent \
   with the patient's age, sex, and reported symptoms.  Note any \
   discrepancies.
2. **XAI Interpretation** – Describe the spatial distribution of \
   activations and whether it is anatomically plausible for each finding.
3. **Contextual Synthesis** – Integrate the retrieved literature to \
   support or refute each finding.  Cite sources by title/ID.
4. **Differential Diagnosis** – Rank probable diagnoses from most to \
   least likely.
5. **Impression** – Provide a concise summary suitable for a radiologist.
6. **Urgency** – Classify as Low, Medium, or High.
7. **Next Steps** – Recommend follow-up imaging, labs, or consultations.

Respond with a JSON object that strictly follows this schema:
```json
{{
  "findings": ["..."],
  "spatial_evidence": {{"region": "description", ...}},
  "reasoning_trace": "...",
  "differential_diagnosis": ["..."],
  "impression": "...",
  "urgency": "Low | Medium | High",
  "next_steps": ["..."],
  "citations": ["..."],
  "requires_human_review": true/false,
  "temporal_changes": "..." or null
}}
```
"""

_CROSS_VERIFICATION_PROMPT_TEMPLATE = """\
You are a medical-AI safety verifier.  Below is a DRAFT clinical report \
and the EVIDENCE used to generate it.

### Draft Report
{draft_json}

### Evidence
**Vision scores**: {predictions_block}
**Spatial XAI**: {spatial_block}
**RAG context snippets**: {rag_block}

For EACH claim in the draft report's "findings", "differential_diagnosis", \
and "impression":
- Check if it is directly supported by the vision scores (confidence ≥ the \
  threshold) OR by the retrieved context (explicitly cited).
- If a claim is UNSUPPORTED, list it.

Respond with a JSON object:
```json
{{
  "unsupported_claims": ["..."],
  "revision_suggestions": ["..."],
  "force_human_review": true/false
}}
```
If all claims are supported, return empty lists and false.
"""


# ════════════════════════════════════════════════════════════════════════════════
# 4.  Default LLM caller (OpenAI-compatible)
# ════════════════════════════════════════════════════════════════════════════════


def _default_call_llm(system: str, user: str, *, temperature: float = 0.2) -> str:
    """Call an OpenAI-compatible chat model.

    Uses ``OPENAI_API_KEY`` and optionally ``OPENAI_MODEL`` / ``OPENAI_API_BASE``
    from environment variables.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install the 'openai' package: pip install openai") from exc

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_API_BASE"),
    )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


# ════════════════════════════════════════════════════════════════════════════════
# 5.  ClinicalReasoningAgent
# ════════════════════════════════════════════════════════════════════════════════


class ClinicalReasoningAgent:
    """Multi-step clinical reasoning agent that fuses vision, XAI, and RAG evidence.

    Parameters
    ----------
    retriever:
        A :class:`HybridRetriever` instance for querying the clinical knowledge base.
    call_llm:
        A callable ``(system_prompt, user_prompt, *, temperature) -> str``.
        Defaults to an OpenAI-compatible caller.
    confidence_threshold:
        Minimum top-1 confidence to avoid forcing human review.
    rag_top_k:
        Number of RAG chunks to retrieve per query.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        call_llm: Optional[Callable[..., str]] = None,
        confidence_threshold: float = 0.30,
        rag_top_k: int = 6,
        guardrail_validator: Optional[GuardrailValidator] = None,
        audit_trail: Optional[AuditTrail] = None,
    ) -> None:
        self.retriever = retriever
        self.call_llm = call_llm or _default_call_llm
        self.confidence_threshold = confidence_threshold
        self.rag_top_k = rag_top_k
        self.guardrail_validator = guardrail_validator or GuardrailValidator()
        self.audit_trail = audit_trail

    # ── public API ────────────────────────────────────────────────────────

    def generate_clinical_report(
        self,
        vision_data: Dict[str, Any],
        patient_meta: Dict[str, Any],
        prev_study_data: Optional[Dict[str, Any]] = None,
    ) -> ClinicalReport:
        """Run the full reasoning loop and return a validated :class:`ClinicalReport`.

        Parameters
        ----------
        vision_data:
            Output of ``InferencePipeline.predict()``.  Expected keys:

            - ``class_names`` / ``class_names_predicted``: list of str
            - ``probabilities``: list of float  (len == num_classes)
            - ``confidence``: float  (top-1 probability)
            - ``explanation`` (optional): dict with ``region_scores``, ``key_findings``
        patient_meta:
            Dict with optional keys: ``age``, ``sex``, ``symptoms``,
            ``clinical_history``, ``patient_id``, ``study_date``.
        prev_study_data:
            Optional dict describing a prior study for longitudinal comparison.
            Keys may include: ``date``, ``findings``, ``impression``.
        """
        start_time = time.perf_counter()

        # ── Step 1: Observation analysis & equivocal check ────────────
        predictions_block, top_confidence, predicted_labels = self._format_predictions(vision_data)
        equivocal = top_confidence < self.confidence_threshold

        # ── Step 2: XAI interpretation ────────────────────────────────
        region_scores: Dict[str, float] = (
            vision_data.get("explanation", {}).get("region_scores", {})
        )
        spatial = interpret_xai_regions(region_scores)
        spatial_block = _summarise_spatial_evidence(spatial)

        # ── Step 3: Contextual retrieval ──────────────────────────────
        augmented_query = self._build_rag_query(predicted_labels, spatial)
        rag_results = self.retriever.query(augmented_query, top_k=self.rag_top_k)
        rag_block, citations = self._format_rag_results(rag_results)

        # ── Step 4: Temporal context ──────────────────────────────────
        temporal_block = self._format_temporal(prev_study_data)

        # ── Step 5: Chain-of-Thought synthesis ────────────────────────
        patient_meta_block = self._format_patient_meta(patient_meta)

        cot_prompt = _COT_PROMPT_TEMPLATE.format(
            predictions_block=predictions_block,
            patient_meta_block=patient_meta_block,
            spatial_block=spatial_block,
            rag_block=rag_block,
            temporal_block=temporal_block,
        )

        raw_response = self.call_llm(_SYSTEM_PROMPT, cot_prompt, temperature=0.2)
        report = self._parse_report(raw_response, spatial, citations, equivocal)

        # ── Step 6: Cross-verification ────────────────────────────────
        report = self._cross_verify(report, predictions_block, spatial_block, rag_block)

        # ── Step 7: Guardrail validation ──────────────────────────────
        guardrail_result = self.guardrail_validator.validate_and_sanitise(report)

        # ── Step 8: Final equivocal guard ─────────────────────────────
        if equivocal:
            report.requires_human_review = True
            if "low overall confidence" not in report.reasoning_trace.lower():
                report.reasoning_trace += (
                    f"\n\n[SYSTEM NOTE] Top-1 confidence ({top_confidence:.2f}) is below "
                    f"threshold ({self.confidence_threshold}). Human review required."
                )

        # ── Step 9: Audit trail ───────────────────────────────────────
        duration_ms = (time.perf_counter() - start_time) * 1000
        if self.audit_trail is not None:
            try:
                self.audit_trail.log_report(
                    report_dict=report.model_dump(),
                    vision_data=vision_data,
                    patient_meta=patient_meta,
                    guardrail_result={
                        "passed": guardrail_result.passed,
                        "term_violations": guardrail_result.term_violations,
                        "structural_warnings": guardrail_result.structural_warnings,
                        "rewrites_applied": guardrail_result.rewrites_applied,
                        "force_human_review": guardrail_result.force_human_review,
                    },
                    duration_ms=duration_ms,
                )
            except Exception:
                logger.warning("Audit trail logging failed; continuing.")

        logger.info(
            "Clinical report generated | equivocal=%s | urgency=%s | review=%s | guardrail_passed=%s | duration_ms=%.1f",
            equivocal, report.urgency, report.requires_human_review, guardrail_result.passed, duration_ms,
        )
        return report

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _format_predictions(vision_data: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """Format vision model output into a prompt-ready block.

        Returns ``(text_block, top_confidence, predicted_label_list)``.
        """
        class_names: List[str] = vision_data.get("class_names", [])
        probs: List[float] = vision_data.get("probabilities", [])
        confidence: float = vision_data.get("confidence", 0.0)

        # Build per-class lines sorted by descending probability
        pairs = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
        lines = [f"- {name}: {prob:.3f}" for name, prob in pairs]
        block = "\n".join(lines) if lines else "No predictions available."

        # Identify labels the model considers positive
        predicted: List[str] = vision_data.get("class_names_predicted", [])
        if not predicted and pairs:
            # Fallback: take labels above 0.5 for multilabel, or top-1
            predicted = [n for n, p in pairs if p >= 0.5] or [pairs[0][0]]

        return block, confidence, predicted

    @staticmethod
    def _format_patient_meta(meta: Dict[str, Any]) -> str:
        if not meta:
            return "No patient metadata provided."
        parts: List[str] = []
        if meta.get("age"):
            parts.append(f"Age: {meta['age']}")
        if meta.get("sex"):
            parts.append(f"Sex: {meta['sex']}")
        if meta.get("symptoms"):
            parts.append(f"Symptoms: {meta['symptoms']}")
        if meta.get("clinical_history"):
            parts.append(f"History: {meta['clinical_history']}")
        return "; ".join(parts) if parts else "No patient metadata provided."

    @staticmethod
    def _format_temporal(prev_study: Optional[Dict[str, Any]]) -> str:
        if not prev_study:
            return "No prior study available for comparison."
        parts: List[str] = []
        if prev_study.get("date"):
            parts.append(f"Prior study date: {prev_study['date']}")
        if prev_study.get("findings"):
            parts.append(f"Prior findings: {prev_study['findings']}")
        if prev_study.get("impression"):
            parts.append(f"Prior impression: {prev_study['impression']}")
        return "\n".join(parts) if parts else "No prior study available for comparison."

    def _build_rag_query(self, labels: List[str], spatial: Dict[str, str]) -> str:
        """Combine predicted labels with spatial hints for a richer retrieval query."""
        parts = list(labels)
        # Add dominant anatomical regions
        for region, desc in spatial.items():
            if "high" in desc:
                parts.append(region)
        return " ".join(parts)

    @staticmethod
    def _format_rag_results(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Format RAG results into a prompt block and a citations list."""
        if not results:
            return "No relevant clinical context retrieved.", []

        lines: List[str] = []
        citations: List[str] = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "unknown")
            condition = r.get("metadata", {}).get("condition", "")
            doc_id = r.get("metadata", {}).get("doc_id", r.get("chunk_id", ""))
            text = r.get("text", "")[:500]
            citation = f"[{i}] {source}"
            if condition:
                citation += f" ({condition})"
            if doc_id:
                citation += f" — {doc_id}"
            lines.append(f"{citation}:\n{text}\n")
            citations.append(citation)

        return "\n".join(lines), citations

    def _parse_report(
        self,
        raw: str,
        spatial_fallback: Dict[str, str],
        citations_fallback: List[str],
        equivocal: bool,
    ) -> ClinicalReport:
        """Parse the LLM JSON response into a ClinicalReport.

        Falls back to sensible defaults if parsing fails so the pipeline
        never crashes.
        """
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            report = ClinicalReport.model_validate_json(cleaned)
        except Exception:
            logger.warning("Failed to parse LLM response as ClinicalReport; using fallback.")
            report = ClinicalReport(
                findings=["Unable to parse AI-generated findings — raw output retained for review."],
                spatial_evidence=spatial_fallback,
                reasoning_trace=raw[:2000],
                differential_diagnosis=[],
                impression="Automated interpretation could not be completed. Please review manually.",
                urgency="Medium",
                next_steps=["Radiologist review recommended."],
                citations=citations_fallback,
                requires_human_review=True,
                temporal_changes=None,
            )

        # Merge citations from RAG that the LLM may have missed
        existing_cites = set(report.citations)
        for c in citations_fallback:
            if c not in existing_cites:
                report.citations.append(c)

        # Merge spatial evidence
        for k, v in spatial_fallback.items():
            report.spatial_evidence.setdefault(k, v)

        if equivocal:
            report.requires_human_review = True

        return report

    def _cross_verify(
        self,
        report: ClinicalReport,
        predictions_block: str,
        spatial_block: str,
        rag_block: str,
    ) -> ClinicalReport:
        """Cross-verification step: check that every claim is evidence-backed."""
        verification_prompt = _CROSS_VERIFICATION_PROMPT_TEMPLATE.format(
            draft_json=report.model_dump_json(indent=2),
            predictions_block=predictions_block,
            spatial_block=spatial_block,
            rag_block=rag_block,
        )

        try:
            raw = self.call_llm(_SYSTEM_PROMPT, verification_prompt, temperature=0.1)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except Exception:
            logger.warning("Cross-verification call failed; skipping.")
            return report

        unsupported: List[str] = result.get("unsupported_claims", [])
        revisions: List[str] = result.get("revision_suggestions", [])
        force_review: bool = result.get("force_human_review", False)

        if unsupported:
            report.reasoning_trace += (
                "\n\n[CROSS-VERIFICATION] Unsupported claims detected: "
                + "; ".join(unsupported)
            )
            if revisions:
                report.reasoning_trace += (
                    "\nSuggested revisions: " + "; ".join(revisions)
                )
 
        if force_review or unsupported:
            report.requires_human_review = True

        return report

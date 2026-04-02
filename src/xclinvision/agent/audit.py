"""Audit Trail – persistent JSON logging of reports, citations, and interactions.

Every report generation, follow-up interaction, and user feedback is saved as
a structured JSON record in ``data/reports/``.  This provides a complete,
immutable audit trail for regulatory compliance and quality assurance.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AuditTrail:
    """Persist clinical reports, interactions, and feedback as JSON.

    Directory structure::

        data/reports/
        ├── reports/
        │   └── report_20260328-143022_abc12345.json
        ├── interactions/
        │   └── interaction_20260328-143100_abc12345.json
        └── feedback/
            └── feedback_20260328-143200_abc12345.json

    Parameters
    ----------
    base_dir:
        Root directory for audit data (default: ``data/reports``).
    """

    def __init__(self, base_dir: str = "data/reports") -> None:
        self.base_dir = Path(base_dir)
        self._reports_dir = self.base_dir / "reports"
        self._interactions_dir = self.base_dir / "interactions"
        self._feedback_dir = self.base_dir / "feedback"

        # Create directories
        for d in (self._reports_dir, self._interactions_dir, self._feedback_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Report logging ────────────────────────────────────────────────

    def log_report(
        self,
        report_dict: Dict[str, Any],
        vision_data: Optional[Dict[str, Any]] = None,
        patient_meta: Optional[Dict[str, Any]] = None,
        guardrail_result: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> Path:
        """Save a complete clinical report with all supporting evidence.

        Parameters
        ----------
        report_dict:
            The serialised :class:`ClinicalReport`.
        vision_data:
            Raw vision model output (probabilities, confidence, etc.).
        patient_meta:
            Patient metadata dict.
        guardrail_result:
            Result of the guardrail validation pass.
        session_id:
            Optional session identifier for grouping interactions.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        timestamp = datetime.now(tz=timezone.utc)
        record = {
            "type": "clinical_report",
            "timestamp": timestamp.isoformat(),
            "session_id": session_id,
            "report": report_dict,
            "vision_data": _sanitise_for_json(vision_data) if vision_data else None,
            "patient_meta": patient_meta,
            "guardrail_result": guardrail_result,
            "telemetry": {"duration_ms": duration_ms} if duration_ms is not None else None,
        }
        filename = f"report_{timestamp.strftime('%Y%m%d-%H%M%S')}_{_short_hash(record)}.json"
        path = self._reports_dir / filename
        self._write_json(path, record)
        logger.info("Audit: report saved → %s", path)
        return path

    # ── Interaction logging ───────────────────────────────────────────

    def log_interaction(
        self,
        question: str,
        response: str,
        *,
        session_id: Optional[str] = None,
        context_summary: Optional[str] = None,
        rag_sources: Optional[List[str]] = None,
    ) -> Path:
        """Save a follow-up Q&A interaction.

        Parameters
        ----------
        question:
            User's follow-up question.
        response:
            AI assistant's response.
        session_id:
            Session identifier.
        context_summary:
            Summary of the clinical context at the time.
        rag_sources:
            Sources retrieved for answering.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        timestamp = datetime.now(tz=timezone.utc)
        record = {
            "type": "interaction",
            "timestamp": timestamp.isoformat(),
            "session_id": session_id,
            "question": question,
            "response": response,
            "context_summary": context_summary,
            "rag_sources": rag_sources,
        }
        filename = f"interaction_{timestamp.strftime('%Y%m%d-%H%M%S')}_{_short_hash(record)}.json"
        path = self._interactions_dir / filename
        self._write_json(path, record)
        logger.info("Audit: interaction saved → %s", path)
        return path

    # ── Feedback logging ──────────────────────────────────────────────

    def log_feedback(
        self,
        report_id: str,
        feedback_type: str,
        feedback_content: Dict[str, Any],
        *,
        session_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> Path:
        """Save radiologist/clinician feedback on a report.

        Parameters
        ----------
        report_id:
            Identifier of the report being reviewed.
        feedback_type:
            Category: "agreement", "correction", "rejection", "comment".
        feedback_content:
            Structured feedback payload.
        session_id:
            Session identifier.
        reviewer_id:
            Anonymised reviewer identifier.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        timestamp = datetime.now(tz=timezone.utc)
        record = {
            "type": "feedback",
            "timestamp": timestamp.isoformat(),
            "session_id": session_id,
            "report_id": report_id,
            "reviewer_id": reviewer_id,
            "feedback_type": feedback_type,
            "feedback_content": feedback_content,
        }
        filename = f"feedback_{timestamp.strftime('%Y%m%d-%H%M%S')}_{_short_hash(record)}.json"
        path = self._feedback_dir / filename
        self._write_json(path, record)
        logger.info("Audit: feedback saved → %s", path)
        return path

    # ── Query helpers ─────────────────────────────────────────────────

    def list_reports(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        """Load recent report audit records (newest first)."""
        return self._load_records(self._reports_dir, limit=limit)

    def list_interactions(
        self, *, session_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Load recent interaction records, optionally filtered by session."""
        records = self._load_records(self._interactions_dir, limit=limit)
        if session_id:
            records = [r for r in records if r.get("session_id") == session_id]
        return records

    # ── internal ──────────────────────────────────────────────────────

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)

    @staticmethod
    def _load_records(directory: Path, *, limit: int = 50) -> List[Dict[str, Any]]:
        files = sorted(directory.glob("*.json"), reverse=True)[:limit]
        records: List[Dict[str, Any]] = []
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as f:
                    records.append(json.load(f))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping corrupt audit file %s: %s", fp, exc)
        return records


# ── Utilities ──────────────────────────────────────────────────────────────────


def _short_hash(data: Any) -> str:
    """Generate a short hash for deduplication / file naming."""
    import hashlib

    raw = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:8]


def _sanitise_for_json(data: Any) -> Any:
    """Recursively convert non-serialisable types to native Python.

    Handles: numpy, torch, sets/frozensets, Decimal, pandas Series/DataFrame.
    """
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: _sanitise_for_json(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_sanitise_for_json(v) for v in data]
    if isinstance(data, (set, frozenset)):
        return [_sanitise_for_json(v) for v in sorted(data, key=str)]

    # Decimal
    import decimal
    if isinstance(data, decimal.Decimal):
        return int(data) if data == int(data) else float(data)

    # pandas
    try:
        import pandas as pd

        if isinstance(data, pd.Series):
            return data.tolist()
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")
    except ImportError:
        pass

    # numpy
    try:
        import numpy as np

        if isinstance(data, (np.integer, np.floating)):
            return data.item()
        if isinstance(data, np.ndarray):
            return data.tolist()
    except ImportError:
        pass

    # torch
    try:
        import torch

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().tolist()
    except ImportError:
        pass
    return data

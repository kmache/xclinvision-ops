"""Centralized configuration for the XClinVision Streamlit frontend.

Single source of truth for: connection settings, timeouts, class labels,
UI metadata, and all API endpoint paths.
"""

import os
from typing import Final, List
from dataclasses import dataclass
from pathlib import Path

import yaml


def _load_class_names() -> List[str]:
    """Read class_names from system.yaml, falling back to a safe default."""
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "system.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
            names = cfg.get("model", {}).get("class_names")
            if isinstance(names, list) and len(names) >= 2:
                return names
    except Exception:
        pass
    return ["No finding", "Cardiomegaly", "Aortic enlargement", "Pleural thickening", "Pulmonary fibrosis"]


# ==============================================================================
# 1. SYSTEM & CONNECTION
# ==============================================================================
ENVIRONMENT: Final = os.getenv("ENVIRONMENT", "development").lower()
API_BASE_URL: Final = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Timeouts (seconds)
HEALTH_CHECK_TIMEOUT: Final = 2.0
DEFAULT_TIMEOUT: Final = float(os.getenv("REQUEST_TIMEOUT", "15"))
INFERENCE_TIMEOUT: Final = 120.0       
EXPLAIN_TIMEOUT: Final = 60.0         
CHAT_TIMEOUT: Final = 60.0            
FEEDBACK_TIMEOUT: Final = 10.0
REPORT_TIMEOUT: Final = 30.0
HISTORY_TIMEOUT: Final = 30.0
DRIFT_TIMEOUT: Final = 15.0
MODEL_CARD_TIMEOUT: Final = 10.0
FEEDBACK_STATS_TIMEOUT: Final = 10.0
EXPORT_REPORT_TIMEOUT: Final = 30.0
COMPARE_TIMEOUT: Final = 240.0       

# ==============================================================================
# 2. CLASS LABELS & COLOURS
# ==============================================================================
CLASS_NAMES: Final = _load_class_names()


# ==============================================================================
# 3. UI METADATA
# ==============================================================================
@dataclass(frozen=True)
class UIConfig:
    APP_TITLE: str = "XClinVision | Medical Imaging AI"
    APP_ICON: str = "🩺"
    APP_VERSION: str = "2.3.0"
    SIDEBAR_TITLE: str = "🩺 XClinVision"
    SIDEBAR_SUBTITLE: str = "Medical Imaging AI Platform"

    # Upload constraints
    MAX_UPLOAD_DIM: int = 1024
    ALLOWED_EXTENSIONS: tuple = ("png", "jpg", "jpeg", "dcm", "dicom")

    # Image overlay defaults
    DEFAULT_THRESHOLD: float = 0.5
    DEFAULT_OPACITY: float = 0.6
    DEFAULT_EXPLAIN_METHOD: str = "gradcam++"

    DISCLAIMER: str = (
        "⚠️ **Disclaimer**: This system is for research and decision support "
        "only. It does not provide medical diagnoses and must always be used "
        "under clinician supervision."
    )


UI = UIConfig()


# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================
class Endpoints:
    """All backend routes declared once — format placeholders where needed."""

    # Health
    HEALTH = "/health"

    # ── v1 ────────────────────────────────────────────────────────────
    DATASET_INFO = "/api/v1/dataset/info"
    MODELS = "/api/v1/models"
    PREDICT = "/api/v1/predict"
    EXPLAIN_V1 = "/api/v1/explain"
    REPORT_V1 = "/api/v1/report"
    FEEDBACK_V1 = "/api/v1/feedback"
    METRICS_V1 = "/api/v1/metrics"

    # ── v2 (dashboard) ────────────────────────────────────────────────
    ANALYZE = "/api/v2/analyze"
    EXPLAIN = "/api/v2/explain/{analysis_id}"
    HISTORY = "/api/v2/history/{patient_id}"
    FEEDBACK = "/api/v2/feedback"
    CHAT = "/api/v2/chat"
    CHAT_STREAM = "/api/v2/chat/stream"
    LLM_PROVIDERS = "/api/v2/llm/providers"
    LLM_SWITCH = "/api/v2/llm/switch"
    LLM_HEALTH = "/api/v2/llm/health"
    GENERATE_REPORT = "/api/v2/generate-report"
    DRIFT_METRICS = "/api/v2/drift-metrics"
    MODEL_CARD = "/api/v2/model-card"
    FEEDBACK_STATS = "/api/v2/feedback-stats"
    EXPORT_REPORT = "/api/v2/export-report"
    COMPARE = "/api/v2/compare"

    @classmethod
    def url(cls, endpoint: str, **params) -> str:
        """Build a full URL from an endpoint template.

        >>> Endpoints.url(Endpoints.EXPLAIN, analysis_id="abc123")
        'http://localhost:8000/api/v2/explain/abc123'
        """
        if params:
            endpoint = endpoint.format(**params)
        return f"{API_BASE_URL}{endpoint}"

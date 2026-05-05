"""Pydantic schemas for API requests/responses."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Tuple
from datetime import datetime


# ---------------------------------------------------------------------------
# Core / Existing Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


class ModelInfo(BaseModel):
    name: str
    type: str
    params: str
    input_size: tuple
    strengths: List[str]
    weaknesses: List[str]
    expected_recall: float
    expected_ece: float


class PredictRequest(BaseModel):
    model_name: str = "convnext_small"
    return_explanation: bool = True


class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: List[float]
    confidence: float
    uncertainty: Optional[Dict] = None
    uncertainty_level: Optional[str] = None
    explanation: Optional[Dict] = None
    processing_time_ms: float
    predictions_multilabel: Optional[List[int]] = None
    class_names_predicted: Optional[List[str]] = None

class ExplainRequest(BaseModel):
    model_name: str = "convnext_small"
    target_class: Optional[int] = None


class ExplainResponse(BaseModel):
    heatmap_url: Optional[str]
    region_scores: Dict[str, float]
    method: str
    key_findings: List[str] = []


class ReportRequest(BaseModel):
    prediction: int
    confidence: float
    uncertainty_level: str
    highlighted_regions: List[str]
    probabilities: Optional[List[float]] = None
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


class ReportResponse(BaseModel):
    findings: str
    impression: str
    uncertainty: str
    recommendation: str


class FeedbackRequest(BaseModel):
    image_hash: str
    prediction: int
    correct_label: int
    # Fix #20: use Literal to enforce valid values instead of unvalidated str.
    feedback_type: Literal["verify", "correct", "error"]
    notes: Optional[str] = None
    clinician_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    timestamp: str


class MetricsResponse(BaseModel):
    model_version: str
    total_predictions: int
    average_confidence: float
    accuracy: float
    ece: float
    macro_auc: Optional[float] = None
    sensitivity: Optional[Dict[str, float]] = None
    specificity: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Dashboard V2 Schemas
# ---------------------------------------------------------------------------


class AnalysisResponse(BaseModel):
    """Full analysis response including prediction, uncertainty, XAI, and LLM summary."""
    analysis_id: str
    patient_id: str
    timestamp: datetime
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: Dict = Field(default_factory=dict)
    uncertainty_level: str = "unknown"
    top_k_predictions: List[Dict[str, float]] = Field(default_factory=list)
    heatmap_gradcam: Optional[str] = None  # Base64 encoded PNG
    heatmap_overlay: Optional[str] = None  # Base64 overlay
    region_scores: Dict[str, float] = Field(default_factory=dict)
    key_findings: List[str] = Field(default_factory=list)
    llm_summary: str = ""
    inference_time_ms: float = 0.0
    model_version: str = "unknown"
    image_hash: str = ""


class ExplanationParams(BaseModel):
    """Parameters for regenerating XAI heatmaps with adjustable settings."""
    analysis_id: str
    method: Literal["gradcam++", "scorecam", "lime", "integrated_gradients", "attention_rollout"] = "gradcam++"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    opacity: float = Field(default=0.6, ge=0.0, le=1.0)
    colormap: str = "jet"


class DashboardFeedbackRequest(BaseModel):
    """Clinician feedback from the dashboard UI."""
    analysis_id: str
    user_id: str = "anonymous"
    feedback_type: Literal["correct", "incorrect", "uncertain"] = "correct"
    notes: Optional[str] = None
    corrected_diagnosis: Optional[str] = None
    severity_score: Optional[int] = Field(default=None, ge=1, le=5)


class ChatMessage(BaseModel):
    """Single chat message."""
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat request with conversation history."""
    analysis_id: str
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    context_type: Literal["clinical", "technical", "patient_friendly"] = "clinical"


class DashboardReportRequest(BaseModel):
    """Report generation request for the dashboard."""
    analysis_ids: List[str]
    template: Literal["structured_clinical", "comprehensive", "brief_summary", "research"] = "structured_clinical"
    sections: List[str] = Field(default_factory=lambda: ["findings", "impressions", "recommendations"])
    language: str = "en"
    include_uncertainty: bool = True
    include_comparison: bool = False


class DriftMetrics(BaseModel):
    """Drift monitoring metrics."""
    drift_score: float = 0.0
    drift_detected: bool = False
    avg_confidence: float = 0.0
    avg_uncertainty: float = 0.0
    prediction_distribution: Dict[str, int] = Field(default_factory=dict)
    feedback_counts: Dict[str, int] = Field(default_factory=dict)
    total_predictions: int = 0


class PatientHistoryEntry(BaseModel):
    """Single entry in a patient's analysis history."""
    analysis_id: str
    timestamp: str
    prediction: str
    confidence: float
    uncertainty: Dict = Field(default_factory=dict)
    uncertainty_level: str = "unknown"
    llm_summary: str = ""
    model_version: str = "unknown"
    thumbnail: Optional[str] = None


class ExportReportRequest(BaseModel):
    """Request to export a clinical report in HTML, PDF, or JSON format."""
    analysis_id: str
    format: Literal["html", "pdf", "json"] = "html"
    include_xai: bool = True
    include_uncertainty: bool = True
    indication: str = ""
    comments: str = ""
    conversation_log: Optional[List[Dict[str, str]]] = None

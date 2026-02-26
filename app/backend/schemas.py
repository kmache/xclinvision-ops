"""Pydantic schemas for API requests/responses."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


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
    model_name: str = "efficientnet_b2"
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


class ExplainRequest(BaseModel):
    model_name: str = "efficientnet_b2"
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
    feedback_type: str  # verify, correct, error
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

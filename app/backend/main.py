"""FastAPI backend for XClinVision inference and feedback."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
from PIL import Image
import io
import hashlib
from datetime import datetime

app = FastAPI(
    title="XClinVision API",
    description="Explainable Medical Imaging AI Platform API",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: List[float]
    confidence: float
    uncertainty: Optional[Dict] = None
    uncertainty_level: Optional[str] = None
    explanation: Optional[Dict] = None
    processing_time_ms: float


class FeedbackRequest(BaseModel):
    image_hash: str
    prediction: int
    correct_label: int
    feedback_type: str  # verify, correct, error
    notes: Optional[str] = None
    clinician_id: Optional[str] = None


class ReportRequest(BaseModel):
    prediction: int
    confidence: float
    uncertainty_level: str
    highlighted_regions: List[str]
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/models")
async def list_models():
    """List available models."""
    from xclinvision.architecture import MODEL_REGISTRY, get_model_info
    
    models = []
    for name in MODEL_REGISTRY.keys():
        info = get_model_info(name)
        models.append({
            "name": name,
            **info,
        })
        
    return {"models": models}


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model_name: str = "efficientnet_b2",
    return_explanation: bool = True,
):
    """Predict class for uploaded chest X-ray image."""
    import time
    start_time = time.time()
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type. Please upload an image.")
        
    # Read image
    contents = await file.read()
    image_hash = hashlib.md5(contents).hexdigest()
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {str(e)}")
        
    # Convert to numpy
    image_np = np.array(image)
    
    # Run inference (placeholder - integrate with actual model)
    # result = pipeline.predict(image_np)
    
    processing_time = (time.time() - start_time) * 1000
    
    # Placeholder response
    return PredictionResponse(
        prediction=0,
        class_name="Normal",
        probabilities=[0.8, 0.15, 0.05],
        confidence=0.8,
        uncertainty={"epistemic": 0.02, "predictive_entropy": 0.5},
        uncertainty_level="low",
        explanation=None,
        processing_time_ms=processing_time,
    )


@app.post("/api/v1/explain")
async def explain(
    file: UploadFile = File(...),
    model_name: str = "efficientnet_b2",
    target_class: Optional[int] = None,
):
    """Generate Grad-CAM++ explanation for image."""
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type")
        
    contents = await file.read()
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {str(e)}")
        
    # Placeholder - integrate with actual explanation generation
    return {
        "heatmap_url": None,
        "region_scores": {
            "left_upper": 0.3,
            "right_upper": 0.4,
            "left_lower": 0.2,
            "right_lower": 0.1,
            "center": 0.5,
        },
        "method": "gradcam++",
    }


@app.post("/api/v1/report")
async def generate_report(request: ReportRequest):
    """Generate clinical report using LLM agent."""
    from xclinvision.agent import ClinicalContext, create_agent
    
    context = ClinicalContext(
        prediction=["Normal", "Pneumonia", "Tuberculosis"][request.prediction],
        probabilities=[0.0, 0.0, 0.0],  # Placeholder
        confidence=request.confidence,
        uncertainty_level=request.uncertainty_level,
        highlighted_regions=request.highlighted_regions,
        patient_age=request.patient_age,
        patient_sex=request.patient_sex,
    )
    
    # Generate report
    agent = create_agent()
    report = agent.generate_report(context)
    
    return report


@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit clinician feedback for model prediction."""
    # Log feedback
    from xclinvision.monitoring import PredictionLogger
    
    logger = PredictionLogger()
    # Store feedback
    
    return {
        "status": "received",
        "feedback_id": f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get model performance metrics."""
    return {
        "model_version": "0.1.0",
        "total_predictions": 0,
        "average_confidence": 0.85,
        "accuracy": 0.92,
        "ece": 0.05,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

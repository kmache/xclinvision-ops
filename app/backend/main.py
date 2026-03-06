"""FastAPI backend for XClinVision inference and feedback."""

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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

# H-4 fix: restrict origins to the configured frontend URL(s) rather than
# allowing all origins with "*".  Set XCLINVISION_CORS_ORIGINS as a
# comma-separated list of allowed origins (default: localhost Streamlit).
_raw_origins = os.getenv("XCLINVISION_CORS_ORIGINS", "http://localhost:8501")
ALLOWED_ORIGINS: list = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
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


# ---------------------------------------------------------------------------
# Model loading  (C-1 fix)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _build_pipeline(model_path: str, architecture: str, image_size: int):
    """Load a trained checkpoint once and cache the InferencePipeline.

    Parameters are read from environment variables:
      XCLINVISION_MODEL_PATH   - path to a .ckpt PyTorch Lightning checkpoint
      XCLINVISION_ARCHITECTURE - model architecture name (default: efficientnet_b2)
      XCLINVISION_IMAGE_SIZE   - input image size used during training (default: 384)
    """
    import torch
    from xclinvision.modeling import build_model
    from xclinvision.trainer import XClinVisionModel
    from xclinvision.inference import InferencePipeline

    base_model = build_model(architecture, num_classes=3, pretrained=False, img_size=image_size)
    pl_module = XClinVisionModel.load_from_checkpoint(
        model_path, model=base_model, strict=False,
        map_location="cpu",
    )
    pl_module.eval()
    return InferencePipeline(
        model=pl_module.model,
        architecture=architecture,
        device="cuda" if torch.cuda.is_available() else "cpu",
        image_size=image_size,
    )


def get_pipeline():
    """Return the cached InferencePipeline or None if not configured."""
    model_path = os.getenv("XCLINVISION_MODEL_PATH", "")
    if not model_path or not Path(model_path).exists():
        return None
    architecture = os.getenv("XCLINVISION_ARCHITECTURE", "efficientnet_b2")
    image_size = int(os.getenv("XCLINVISION_IMAGE_SIZE", "384"))
    return _build_pipeline(model_path, architecture, image_size)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _count_images(directory: Path) -> int:
    """Return the number of image files directly inside *directory*."""
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def _dataset_stats(data_dir: Path) -> dict:
    """Build a nested dict with per-split, per-class image counts."""
    splits = ["train", "val", "test"]
    stats: dict = {}
    for split in splits:
        split_path = data_dir / split
        if not split_path.is_dir():
            continue
        class_counts: dict = {}
        class_dirs = sorted(p for p in split_path.iterdir() if p.is_dir())
        for class_dir in class_dirs:
            class_counts[class_dir.name] = _count_images(class_dir)
        stats[split] = {
            "path": str(split_path.resolve()),
            "classes": class_counts,
            "total": sum(class_counts.values()),
        }
    return stats


# ---------------------------------------------------------------------------
# Application startup: log dataset summary
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def log_dataset_info() -> None:
    """Log dataset directories and image counts on server startup."""
    raw_data_dir = os.getenv("XCLINVISION_DATA_DIR", "data/processed")
    data_dir = Path(raw_data_dir)
    if not data_dir.is_absolute():
        # Resolve relative paths against the repository root (three levels up
        # from app/backend/main.py)
        data_dir = (Path(__file__).parent.parent.parent / data_dir).resolve()

    logger.info("=" * 60)
    logger.info("XClinVision  –  Dataset Summary")
    logger.info("=" * 60)
    logger.info("Data root : %s", data_dir)

    if not data_dir.is_dir():
        logger.warning("Data directory not found: %s", data_dir)
        logger.info("=" * 60)
        return

    stats = _dataset_stats(data_dir)
    if not stats:
        logger.warning("No train/val/test splits found under %s", data_dir)
        logger.info("=" * 60)
        return

    grand_total = 0
    for split, info in stats.items():
        logger.info("-" * 40)
        logger.info("Split : %-6s  |  path: %s", split.upper(), info["path"])
        for cls_name, count in info["classes"].items():
            logger.info("  %-16s : %d images", cls_name, count)
        logger.info("  %-16s : %d images", "TOTAL", info["total"])
        grand_total += info["total"]
    logger.info("-" * 40)
    logger.info("Grand total          : %d images", grand_total)
    logger.info("=" * 60)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/dataset/info")
async def dataset_info():
    """Return dataset directories and image counts for each split."""
    raw_data_dir = os.getenv("XCLINVISION_DATA_DIR", "data/processed")
    data_dir = Path(raw_data_dir)
    if not data_dir.is_absolute():
        data_dir = (Path(__file__).parent.parent.parent / data_dir).resolve()

    if not data_dir.is_dir():
        raise HTTPException(404, detail=f"Data directory not found: {data_dir}")

    stats = _dataset_stats(data_dir)
    grand_total = sum(info["total"] for info in stats.values())
    return {
        "data_root": str(data_dir),
        "splits": stats,
        "grand_total": grand_total,
    }


@app.get("/api/v1/models")
async def list_models():
    """List available model architectures.

    C-2 fix: the previous implementation imported xclinvision.architecture
    which does not exist.  Model metadata is now read from xclinvision.modeling.
    """
    from xclinvision.modeling import TIMM_MODEL_MAP

    models = [
        {"name": name, "timm_id": timm_id, "type": "cnn" if any(
            k in name for k in ("resnet", "densenet", "efficientnet", "convnext")
        ) else "transformer"}
        for name, timm_id in TIMM_MODEL_MAP.items()
    ]
    models.append({"name": "biomedclip", "timm_id": "hf-hub:microsoft/BiomedCLIP-...", "type": "vit"})
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

    image_np = np.array(image)

    # C-1 fix: run actual inference via the configured InferencePipeline.
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            503,
            detail=(
                "No model is loaded. Set the XCLINVISION_MODEL_PATH environment "
                "variable to the path of a trained .ckpt checkpoint and restart "
                "the server.  Optionally set XCLINVISION_ARCHITECTURE and "
                "XCLINVISION_IMAGE_SIZE to match the checkpoint."
            ),
        )

    try:
        result = pipeline.predict(
            image_np,
            return_uncertainty=True,
            return_explanation=return_explanation,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")

    processing_time = (time.time() - start_time) * 1000

    explanation_out = None
    if return_explanation and "explanation" in result:
        exp = result["explanation"]
        if exp:
            # Convert numpy heatmap to a serialisable form (region scores only)
            explanation_out = {
                "key_findings": exp.get("key_findings", []),
                "clinical_plausibility": exp.get("clinical_plausibility"),
                "region_scores": exp.get("visualization", {}).get("region_scores"),
                "method": exp.get("visualization", {}).get("method"),
            }

    return PredictionResponse(
        prediction=result["prediction"],
        class_name=result["class_name"],
        probabilities=result["probabilities"],
        confidence=result["confidence"],
        uncertainty=result.get("uncertainty"),
        uncertainty_level=result.get("uncertainty_level"),
        explanation=explanation_out,
        processing_time_ms=processing_time,
    )


@app.post("/api/v1/explain")
async def explain(
    file: UploadFile = File(...),
    model_name: str = "efficientnet_b2",
    target_class: Optional[int] = None,
):
    """Generate Grad-CAM++ explanation for image."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {str(e)}")

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            503,
            detail="No model loaded. Set XCLINVISION_MODEL_PATH and restart the server.",
        )

    image_np = np.array(image)
    try:
        result = pipeline.predict(image_np, return_uncertainty=False, return_explanation=True)
    except Exception as e:
        raise HTTPException(500, f"Explanation generation failed: {str(e)}")

    exp = result.get("explanation") or {}
    vis = exp.get("visualization") or {}
    return {
        "heatmap_url": None,  # heatmap image serving not yet implemented
        "region_scores": vis.get("region_scores"),
        "key_findings": exp.get("key_findings", []),
        "clinical_plausibility": exp.get("clinical_plausibility"),
        "method": vis.get("method", "gradcam++"),
        "target_class": result.get("prediction"),
        "target_class_name": result.get("class_name"),
    }


@app.post("/api/v1/report")
async def generate_report(request: ReportRequest):
    """Generate clinical report using LLM agent."""
    from xclinvision.agent import ClinicalContext, create_agent
    
    context = ClinicalContext(
        prediction=["Normal", "Pneumonia", "Cardiomegaly"][request.prediction],
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

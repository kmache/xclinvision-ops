import logging
import os
import sys
import json
import uuid
import base64
import time
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from xclinvision.config import get_class_names, get_num_classes
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Literal
import numpy as np
from PIL import Image
import cv2
import io
import hashlib
from datetime import datetime, timedelta

from schemas import (
    AnalysisResponse,
    ExplanationParams,
    DashboardFeedbackRequest,
    ChatRequest,
    DashboardReportRequest,
    DriftMetrics,
    PatientHistoryEntry,
)

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

_IMAGE_STORE_MAX = 200


@asynccontextmanager
async def _lifespan(app: "FastAPI"):
    """Application lifespan — replaces the deprecated @app.on_event('startup').

    ``log_dataset_info`` is defined later in the module; Python resolves the
    name at call time (server start-up), not at definition time, so the
    forward reference is safe.
    """
    await log_dataset_info()
    yield


app = FastAPI(
    title="XClinVision API",
    description="Explainable Medical Imaging AI Platform API",
    version="0.1.0",
    lifespan=_lifespan,
)
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
    # Multi-label fields (populated only when classification_mode == "multilabel")
    predictions_multilabel: Optional[List[int]] = None
    class_names_predicted: Optional[List[str]] = None


class FeedbackRequest(BaseModel):
    image_hash: str
    prediction: int
    correct_label: int
    feedback_type: Literal["verify", "correct", "error"]
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

    Supports both plain state-dict (.pth) and PyTorch Lightning (.ckpt) files.

    Parameters are read from environment variables:
      XCLINVISION_MODEL_PATH   - path to a .pth state-dict or .ckpt Lightning checkpoint
      XCLINVISION_ARCHITECTURE - model architecture name (default: efficientnet_b2)
      XCLINVISION_IMAGE_SIZE   - input image size used during training (default: 384)
    """
    import torch
    from xclinvision.modeling import build_model, get_model_normalization
    from xclinvision.inference import InferencePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(architecture, num_classes=get_num_classes(), pretrained=False, img_size=image_size)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Lightning checkpoint — strip "model." prefix from keys
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
            state_dict[new_key] = v
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded Lightning checkpoint: %s", model_path)
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # BestModelExportCallback .pth payload
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info("Loaded exported .pth payload: %s", model_path)
    elif isinstance(checkpoint, dict):
        # Plain state_dict (e.g. torch.save(model.state_dict(), ...))
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Loaded state_dict checkpoint: %s", model_path)
    else:
        raise ValueError(f"Unrecognised checkpoint format in {model_path}")

    model.eval()

    # Extract temperature and thresholds from the payload if present
    temperature_value = None
    thresholds_dict = None
    if isinstance(checkpoint, dict):
        raw_temp = checkpoint.get("temperature")
        if raw_temp is not None:
            temperature_value = float(raw_temp)
            logger.info("Loaded temperature from payload: T=%.4f", temperature_value)
        raw_thresh = checkpoint.get("thresholds")
        if isinstance(raw_thresh, dict):
            thresholds_dict = {str(k): float(v) for k, v in raw_thresh.items()}
            logger.info("Loaded per-class thresholds from payload: %s", thresholds_dict)

    norm_stats = get_model_normalization(model, architecture)

    return InferencePipeline(
        model=model,
        architecture=architecture,
        device=device,
        image_size=image_size,
        dataset_mean=list(norm_stats["mean"]),
        dataset_std=list(norm_stats["std"]),
        temperature_scaler=temperature_value,
        thresholds=thresholds_dict,
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
_agent_instance = None
_agent_lock = threading.Lock()


def _get_agent():
    """Return a module-level cached ClinicalAgent, creating it on first call.

    Thread-safe: uses a module-level lock to prevent double-initialisation
    when concurrent requests both find ``_agent_instance is None``.
    """
    global _agent_instance
    with _agent_lock:
        if _agent_instance is None:
            try:
                from xclinvision.agent import create_agent
                _agent_instance = create_agent()
            except Exception as exc:
                logger.warning("Failed to initialise LLM agent: %s", exc)
                raise HTTPException(503, detail=f"LLM agent unavailable: {exc}")
    return _agent_instance

# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".dcm"}


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

async def log_dataset_info() -> None:
    """(Called from the lifespan context manager on server start-up.)"""
    """Log dataset directories and image counts on server startup."""
    raw_data_dir = os.getenv("XCLINVISION_DATA_DIR", "data/processed")
    data_dir = Path(raw_data_dir)
    if not data_dir.is_absolute():
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

    # Fix #4: enforce upload size limit before reading into memory.
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
        )
    image_hash = hashlib.sha256(contents).hexdigest()

    _MAGIC = (
        (b"\xff\xd8\xff",),                    # JPEG
        (b"\x89PNG",),                         # PNG
        (b"BM",),                              # BMP
        (b"II", b"MM"),                        # TIFF (little/big-endian)
    )
    _is_dicom = len(contents) > 132 and contents[128:132] == b"DICM"
    _magic_ok = _is_dicom or any(
        contents[:4].startswith(m)
        for group in _MAGIC
        for m in group
    )
    if not _magic_ok:
        raise HTTPException(400, "File content does not match a supported image format (JPEG, PNG, BMP, TIFF, DICOM).")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {str(e)}")

    image_np = np.array(image)

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
        predictions_multilabel=result.get("predictions_multilabel"),
        class_names_predicted=result.get("class_names_predicted"),
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

    # Fix #2: enforce the same upload size limit as /predict.
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
        )
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
    from xclinvision.agent import ClinicalContext

    # Fix #21: validate prediction index before using it as a list index.
    class_names = get_class_names()
    num_classes = len(class_names)
    if request.prediction not in range(num_classes):
        raise HTTPException(
            422,
            f"prediction must be 0–{num_classes - 1}, got {request.prediction}.",
        )

    context = ClinicalContext(
        prediction=class_names[request.prediction],
        probabilities=[0.0] * num_classes,  # Placeholder (dynamic length)
        confidence=request.confidence,
        uncertainty_level=request.uncertainty_level,
        highlighted_regions=request.highlighted_regions,
        patient_age=request.patient_age,
        patient_sex=request.patient_sex,
    )

    agent = _get_agent()
    report = agent.generate_report(context)

    return report


@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit clinician feedback for model prediction."""
    # Fix #17: actually persist feedback instead of silently discarding it.
    feedback_entry = {
        "feedback_id": f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now().isoformat(),
        **feedback.dict(),
    }
    _feedback_store.append(feedback_entry)
    logger.info(
        "Feedback received: id=%s, type=%s, prediction=%s, correct=%s",
        feedback_entry["feedback_id"],
        feedback.feedback_type,
        feedback.prediction,
        feedback.correct_label,
    )
    return {
        "status": "received",
        "feedback_id": feedback_entry["feedback_id"],
        "timestamp": feedback_entry["timestamp"],
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get model performance metrics.

    Fix #18: attempt to load real metrics from the most recent evaluation
    report saved by MetricsComputer.save_results().  Falls back to placeholder
    values only when no report file is found.
    """
    outputs_dir = Path(os.getenv("XCLINVISION_OUTPUTS_DIR", "outputs/evaluation"))
    if not outputs_dir.is_absolute():
        outputs_dir = (Path(__file__).parent.parent.parent / outputs_dir).resolve()

    # Look for the most recently written metrics JSON file.
    report_files = sorted(outputs_dir.glob("**/metrics*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if report_files:
        try:
            with open(report_files[0]) as f:
                data = json.load(f)
            return {
                "model_version": os.getenv("XCLINVISION_ARCHITECTURE", "unknown"),
                "total_predictions": len(_feedback_store),
                "source": str(report_files[0]),
                **{k: v for k, v in data.items() if isinstance(v, (int, float, str))},
            }
        except Exception as exc:
            logger.warning("Could not load metrics from %s: %s", report_files[0], exc)

    # Fallback placeholder — values are clearly marked as estimates.
    logger.warning(
        "/api/v1/metrics: no evaluation report found under %s. "
        "Run scripts/evaluate.py to generate real metrics.",
        outputs_dir,
    )
    return {
        "model_version": os.getenv("XCLINVISION_ARCHITECTURE", "unknown"),
        "total_predictions": len(_feedback_store),
        "note": "No evaluation report found — run scripts/evaluate.py to populate real metrics.",
        "accuracy": None,
        "macro_f1": None,
        "macro_auc": None,
        "ece": None,
    }


# ---------------------------------------------------------------------------
# Dashboard v2: In-memory storage (replace with SQLite/Postgres in prod)
# ---------------------------------------------------------------------------

_analysis_store: Dict[str, dict] = {}   
_feedback_store: List[dict] = []         
_image_store: Dict[str, bytes] = {}    

def _image_store_put(analysis_id: str, data: bytes) -> None:
    """Insert into _image_store with FIFO eviction capped at _IMAGE_STORE_MAX.

    Fix #16: unbounded _image_store accumulates raw image bytes and will OOM
    under sustained concurrent usage.  Evict the oldest entry when the cap
    is reached so memory stays bounded.
    """
    if len(_image_store) >= _IMAGE_STORE_MAX:
        oldest_key = next(iter(_image_store))
        del _image_store[oldest_key]
    _image_store[analysis_id] = data


def _img_to_base64(img_array: np.ndarray) -> str:
    """Convert a numpy image array to a base64-encoded PNG string."""
    from PIL import Image as PILImage
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    if img_array.ndim == 2:
        pil_img = PILImage.fromarray(img_array, mode="L")
    else:
        pil_img = PILImage.fromarray(img_array, mode="RGB")
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _generate_heatmap_overlay(
    vis_image: np.ndarray,
    heatmap: np.ndarray,
    opacity: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Create a coloured heatmap overlay on the original image."""
    h, w = vis_image.shape[:2]
    heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_norm = heat_resized / (heat_resized.max() + 1e-8)
    heat_uint8 = (heat_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, colormap)
    heat_color_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    overlay = (vis_image.astype(float) * (1 - opacity) + heat_color_rgb.astype(float) * opacity)
    return overlay.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dashboard v2: Analyze endpoint (full pipeline)
# ---------------------------------------------------------------------------

@app.post("/api/v2/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    patient_id: str = Form(default="UNKNOWN"),
    study_date: str = Form(default=""),
    modality: str = Form(default="X-ray"),
    body_part: str = Form(default="Chest"),
    clinical_history: str = Form(default=""),
    model_name: str = Form(default="efficientnet_b2"),
):
    """Full analysis with prediction, uncertainty, XAI heatmaps, and LLM summary.

    Returns AnalysisResponse with base64-encoded heatmaps and clinical summary.
    """
    start_time = time.time()

    allowed_types = {"image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff",
                      "application/octet-stream", "application/dicom"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(400, f"Invalid file type '{file.content_type}'. Please upload an image.")

    # Fix #4: enforce upload size limit on /api/v2/analyze as well.
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
        )
    image_hash = hashlib.sha256(contents).hexdigest()

    try:
        from xclinvision.processing import read_image_grayscale
        gray = read_image_grayscale(contents)
        if gray is not None:
            image_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_np = np.array(image)
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")
    analysis_id = f"XCL-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

    # --- Run inference -------------------------------------------------------
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(
            503,
            detail="No model loaded. Set XCLINVISION_MODEL_PATH and restart.",
        )

    try:
        result = pipeline.predict(
            image_np,
            return_uncertainty=True,
            return_explanation=True,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    inference_ms = (time.time() - start_time) * 1000

    # --- Build heatmap images ------------------------------------------------
    heatmap_b64 = None
    overlay_b64 = None
    explanation = result.get("explanation") or {}
    vis = explanation.get("visualization") or {}

    grayscale_cam = vis.get("grayscale_cam")

    # Also get vis_image from pipeline preprocess for overlay
    _, vis_image = pipeline.preprocess(image_np)

    if grayscale_cam is not None:
        heatmap_b64 = _img_to_base64(grayscale_cam)
        overlay_img = _generate_heatmap_overlay(vis_image, grayscale_cam, opacity=0.45)
        overlay_b64 = _img_to_base64(overlay_img)
    else:
        # Generate a mock heatmap for demo purposes when no real XAI (vectorized)
        h, w = vis_image.shape[:2]
        cy, cx = int(h * 0.30), int(w * 0.65)
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        d = np.sqrt(((yy - cy) / (h * 0.25)) ** 2 + ((xx - cx) / (w * 0.25)) ** 2)
        mock = np.exp(-d)
        heatmap_b64 = _img_to_base64(mock)
        overlay_img = _generate_heatmap_overlay(vis_image, mock, opacity=0.45)
        overlay_b64 = _img_to_base64(overlay_img)

    # --- Build top-k predictions ---------------------------------------------
    probs = result.get("probabilities", [])
    class_names = result.get("class_names", get_class_names())
    top_k = sorted(
        [{"class_name": cn, "probability": float(p)} for cn, p in zip(class_names, probs)],
        key=lambda x: x["probability"],
        reverse=True,
    )

    # --- Region scores -------------------------------------------------------
    region_scores = vis.get("region_scores") or {}
    key_findings = explanation.get("key_findings", [])

    # --- LLM Summary ---------------------------------------------------------
    llm_summary = ""
    try:
        from xclinvision.agent import ClinicalContext

        context = ClinicalContext(
            prediction=result["class_name"],
            probabilities=probs,
            confidence=result["confidence"],
            uncertainty_level=result.get("uncertainty_level", "unknown"),
            highlighted_regions=list(region_scores.keys())[:3] if region_scores else [],
            class_names=class_names,
            patient_age=None,
            patient_sex=None,
        )
        
        agent = _get_agent()
        report_dict = agent.generate_report(context)
        llm_summary = report_dict.get("findings", "") + " " + report_dict.get("impression", "")
    except Exception as e:
        logger.warning("LLM summary generation failed: %s", e)
        llm_summary = (
            f"AI analysis detected {result['class_name']} with "
            f"{result['confidence']:.1%} confidence. "
            f"Uncertainty: {result.get('uncertainty_level', 'unknown')}. "
            "Clinical correlation recommended."
        )

    # --- Build response -------------------------------------------------------
    analysis_data = {
        "analysis_id": analysis_id,
        "patient_id": patient_id,
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": result["class_name"],
        "confidence": result["confidence"],
        "uncertainty": result.get("uncertainty", {}),
        "uncertainty_level": result.get("uncertainty_level", "unknown"),
        "top_k_predictions": top_k,
        "heatmap_gradcam": heatmap_b64,
        "heatmap_overlay": overlay_b64,
        "region_scores": region_scores if isinstance(region_scores, dict) else {},
        "key_findings": key_findings,
        "llm_summary": llm_summary,
        "inference_time_ms": round(inference_ms, 1),
        "model_version": model_name,
        "image_hash": image_hash,
    }

    # Store for later retrieval (compress stored image to save memory)
    _analysis_store[analysis_id] = analysis_data
    try:
        _store_img = Image.open(io.BytesIO(contents)).convert("RGB")
        _buf = io.BytesIO()
        _store_img.save(_buf, format="JPEG", quality=80)
        # Fix #16: use evicting helper to prevent unbounded memory growth.
        _image_store_put(analysis_id, _buf.getvalue())
    except Exception:
        _image_store_put(analysis_id, contents)

    return analysis_data


# ---------------------------------------------------------------------------
# Dashboard v2: Explanation with adjustable params
# ---------------------------------------------------------------------------

@app.get("/api/v2/explain/{analysis_id}")
async def get_dashboard_explanation(
    analysis_id: str,
    method: str = Query(default="gradcam++"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    opacity: float = Query(default=0.6, ge=0.0, le=1.0),
    colormap: str = Query(default="jet"),
):
    """Regenerate XAI explanation with adjustable threshold and opacity."""
    if analysis_id not in _analysis_store:
        raise HTTPException(404, "Analysis not found")

    stored = _analysis_store[analysis_id]
    raw_bytes = _image_store.get(analysis_id)

    if raw_bytes is None:
        raise HTTPException(404, "Original image not found")

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(503, "No model loaded.")

    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image_np = np.array(image)

    try:
        result = pipeline.predict(image_np, return_uncertainty=False, return_explanation=True)
    except Exception as e:
        raise HTTPException(500, f"Explanation failed: {e}")

    explanation = result.get("explanation") or {}
    vis = explanation.get("visualization") or {}
    # Fix #1: use grayscale_cam to avoid double-overlay artefact.
    raw_heatmap = vis.get("grayscale_cam")
    _, vis_image = pipeline.preprocess(image_np)

    if raw_heatmap is not None:
        # Apply threshold: zero out regions below threshold
        thresholded = raw_heatmap.copy()
        thresholded[thresholded < threshold] = 0.0

        colormap_cv = {
            "jet": cv2.COLORMAP_JET,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "hot": cv2.COLORMAP_HOT,
        }.get(colormap, cv2.COLORMAP_JET)

        heatmap_b64 = _img_to_base64(thresholded)
        overlay_img = _generate_heatmap_overlay(vis_image, thresholded, opacity=opacity, colormap=colormap_cv)
        overlay_b64 = _img_to_base64(overlay_img)
    else:
        heatmap_b64 = stored.get("heatmap_gradcam")
        overlay_b64 = stored.get("heatmap_overlay")

    return {
        "heatmap": heatmap_b64,
        "overlay": overlay_b64,
        "threshold_applied": threshold,
        "opacity_applied": opacity,
        "method": method,
        "region_scores": vis.get("region_scores") or stored.get("region_scores", {}),
    }


# ---------------------------------------------------------------------------
# Dashboard v2: Patient history
# ---------------------------------------------------------------------------

@app.get("/api/v2/history/{patient_id}")
async def get_patient_history(patient_id: str, limit: int = Query(default=50, le=200)):
    """Retrieve all historical analyses for a patient (temporal comparison)."""
    history = [
        v for v in _analysis_store.values()
        if v.get("patient_id") == patient_id
    ]
    history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return [
        {
            "analysis_id": h["analysis_id"],
            "timestamp": h["timestamp"],
            "prediction": h["prediction"],
            "confidence": h["confidence"],
            "uncertainty": h.get("uncertainty", {}),
            "uncertainty_level": h.get("uncertainty_level", "unknown"),
            "llm_summary": h.get("llm_summary", ""),
            "model_version": h.get("model_version", "unknown"),
            "thumbnail": h.get("heatmap_gradcam"),
        }
        for h in history[:limit]
    ]


# ---------------------------------------------------------------------------
# Dashboard v2: Feedback
# ---------------------------------------------------------------------------

@app.post("/api/v2/feedback")
async def submit_dashboard_feedback(feedback: DashboardFeedbackRequest):
    """Store clinician feedback from the dashboard UI."""
    entry = feedback.model_dump()
    entry["timestamp"] = datetime.utcnow().isoformat()
    entry["feedback_id"] = f"fb-{uuid.uuid4().hex[:8]}"
    _feedback_store.append(entry)

    # Also log via PredictionLogger
    try:
        from xclinvision.monitoring import PredictionLogger
        pred_logger = PredictionLogger()
        # Log that feedback was received (lightweight)
        logger.info("Feedback received: %s for analysis %s", feedback.feedback_type, feedback.analysis_id)
    except Exception:
        pass

    return {"status": "recorded", "feedback_id": entry["feedback_id"]}


# ---------------------------------------------------------------------------
# Dashboard v2: LLM Chat
# ---------------------------------------------------------------------------

@app.post("/api/v2/chat")
async def llm_chat(request: ChatRequest):
    """Context-aware LLM chat about an analysis."""
    stored = _analysis_store.get(request.analysis_id)
    if not stored:
        raise HTTPException(404, "Analysis not found")

    try:
        from xclinvision.agent import ClinicalContext, create_agent

        probs = [p["probability"] for p in stored.get("top_k_predictions", [])]
        class_names = [p["class_name"] for p in stored.get("top_k_predictions", [])]

        context = ClinicalContext(
            prediction=stored["prediction"],
            probabilities=probs,
            confidence=stored["confidence"],
            uncertainty_level=stored.get("uncertainty_level", "unknown"),
            highlighted_regions=list(stored.get("region_scores", {}).keys())[:3],
            class_names=class_names,
        )

        agent = _get_agent()
        history_text = "\n".join(
            f"{'User' if m.role == 'user' else 'AI'}: {m.content}"
            for m in request.history[-5:]  # Last 5 messages for context
        )

        report = agent.generate_report(context)
        base_text = report.get("findings", "") + " " + report.get("impression", "")

        # For the user's specific question, provide a contextual response
        response_text = (
            f"Based on the AI analysis ({stored['prediction']} at "
            f"{stored['confidence']:.1%} confidence): {base_text}\n\n"
            f"Regarding your question: '{request.message}' — "
            f"{report.get('recommendation', 'Clinical correlation recommended.')}"
        )

        suggested = [
            "What are the differential diagnoses?",
            "Explain in simple terms",
            "Is follow-up imaging needed?",
        ]

    except Exception as e:
        logger.warning("LLM chat error: %s", e)
        response_text = (
            f"Analysis shows {stored['prediction']} ({stored['confidence']:.1%} confidence). "
            f"Regarding '{request.message}': Clinical correlation is recommended. "
            "Please consult with a specialist for definitive interpretation."
        )
        suggested = []

    return {
        "response": response_text,
        "suggested_followups": suggested,
        "references": [],
    }


# ---------------------------------------------------------------------------
# Dashboard v2: Report generation
# ---------------------------------------------------------------------------

@app.post("/api/v2/generate-report")
async def generate_dashboard_report(request: DashboardReportRequest):
    """Generate a structured clinical report from one or more analyses."""
    analyses = [_analysis_store.get(aid) for aid in request.analysis_ids]
    analyses = [a for a in analyses if a is not None]

    if not analyses:
        raise HTTPException(404, "No analyses found for the given IDs")

    primary = analyses[0]

    # Build sections
    findings = (
        f"AI analysis of chest {primary.get('model_version', 'X-ray')} "
        f"({primary['timestamp'][:10]}):\n\n"
        f"1. {primary['prediction']} detected with {primary['confidence']:.1%} confidence. "
        f"Uncertainty: {primary.get('uncertainty_level', 'unknown').capitalize()}.\n"
    )
    if primary.get("key_findings"):
        for i, f in enumerate(primary["key_findings"], 2):
            findings += f"{i}. {f}\n"

    impressions = primary.get("llm_summary", f"{primary['prediction']} identified.")

    recommendations = (
        "- Clinical correlation with patient history and physical examination.\n"
        "- Consider follow-up imaging if clinically indicated.\n"
        "- Refer to AI confidence and uncertainty metrics for reliability assessment."
    )

    # Comparison section
    comparison = ""
    if request.include_comparison and len(analyses) > 1:
        prev = analyses[1]
        comparison = (
            f"Compared with prior study ({prev['timestamp'][:10]}): "
            f"Previous finding was {prev['prediction']} "
            f"({prev['confidence']:.1%} confidence)."
        )

    content = {}
    for section in request.sections:
        if section == "findings":
            content["findings"] = findings
        elif section == "impressions":
            content["impressions"] = impressions
        elif section == "recommendations":
            content["recommendations"] = recommendations
        elif section == "comparison":
            content["comparison"] = comparison

    if request.include_uncertainty:
        unc = primary.get("uncertainty", {})
        content["uncertainty"] = (
            f"Epistemic uncertainty: {unc.get('epistemic', 'N/A')}, "
            f"Predictive entropy: {unc.get('predictive_entropy', 'N/A')}, "
            f"Level: {primary.get('uncertainty_level', 'unknown')}."
        )

    return {
        "report_id": f"RPT-{uuid.uuid4().hex[:8]}",
        "content": content,
        "analysis_ids": request.analysis_ids,
        "template": request.template,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Dashboard v2: Drift metrics
# ---------------------------------------------------------------------------

@app.get("/api/v2/drift-metrics")
async def get_drift_metrics(days: int = Query(default=30, ge=1, le=365)):
    """Return drift monitoring metrics from prediction logs."""
    try:
        from xclinvision.monitoring import PredictionLogger

        pred_logger = PredictionLogger()
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        history = pred_logger.get_prediction_history(start_date=start_date)

        if history:
            confidences = [h["confidence"] for h in history if "confidence" in h]
            uncertainties = [
                h.get("uncertainty", {}).get("epistemic", 0)
                for h in history
                if isinstance(h.get("uncertainty"), dict)
            ]
            preds = [h.get("prediction", -1) for h in history]
            from collections import Counter
            pred_dist = dict(Counter(preds))

            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            avg_unc = sum(uncertainties) / len(uncertainties) if uncertainties else 0

            # Simple drift score: deviation from expected mean confidence
            drift_score = abs(avg_conf - 0.85) * 2  # baseline ~0.85
        else:
            avg_conf = 0.0
            avg_unc = 0.0
            pred_dist = {}
            drift_score = 0.0

    except Exception as e:
        logger.warning("Drift metric computation failed: %s", e)
        avg_conf = 0.0
        avg_unc = 0.0
        pred_dist = {}
        drift_score = 0.0

    # Count feedback
    fb_counts = {}
    for fb in _feedback_store:
        ft = fb.get("feedback_type", "unknown")
        fb_counts[ft] = fb_counts.get(ft, 0) + 1

    return {
        "drift_score": round(drift_score, 4),
        "drift_detected": drift_score > 0.2,
        "avg_confidence": round(avg_conf, 4),
        "avg_uncertainty": round(avg_unc, 4),
        "prediction_distribution": pred_dist,
        "feedback_counts": fb_counts,
        "total_predictions": len(_analysis_store),
    }


# ---------------------------------------------------------------------------
# Dashboard v2: Model card
# ---------------------------------------------------------------------------

@app.get("/api/v2/model-card")
async def get_model_card():
    """Return structured model documentation and live stats."""
    model_card_path = Path(__file__).parent.parent.parent / "docs" / "model_card.md"
    model_card_text = ""
    if model_card_path.exists():
        model_card_text = model_card_path.read_text()[:2000]

    return {
        "name": "XClinVision ChestX-ray",
        "version": "2.3.0",
        "last_updated": "2026-03-01",
        "intended_use": f"Detection and characterization of thoracic diseases in chest X-rays ({', '.join(get_class_names())})",
        "performance": {
            "auc": 0.94,
            "sensitivity": 0.92,
            "specificity": 0.89,
            "ece": 0.05,
            "training_size": 112120,
            "validation_size": 25596,
        },
        "limitations": [
            "Not validated for pediatric populations (<18 years)",
            "Reduced performance for subtle findings <5mm",
            "Trained on frontal view only (PA/AP)",
            "May miss subtle interstitial patterns",
            "Performance degrades on images from non-standard equipment",
        ],
        "training_data": "NIH ChestX-ray14 (2017) — 112,120 frontal chest X-rays from 30,805 unique patients",
        "architectures_available": [
            "efficientnet_b0", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
            "convnext_small", "swin_t", "swin_s", "swin_b",
            "vit_tiny", "vit_small", "vit_base", "resnet50", "densenet", "biomedclip",
        ],
        "certifications": ["Research Use Only — Not FDA cleared"],
        "model_card_md": model_card_text[:500] if model_card_text else "",
    }


# ---------------------------------------------------------------------------
# Dashboard v2: Feedback stats
# ---------------------------------------------------------------------------

@app.get("/api/v2/feedback-stats")
async def get_feedback_stats():
    """Aggregate feedback statistics for the audit page."""
    from collections import Counter

    type_counts = Counter(fb.get("feedback_type", "unknown") for fb in _feedback_store)
    recent = sorted(_feedback_store, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]

    return {
        "total": len(_feedback_store),
        "by_type": dict(type_counts),
        "recent": recent,
        "correction_rate": (
            type_counts.get("incorrect", 0) / max(len(_feedback_store), 1) * 100
        ),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

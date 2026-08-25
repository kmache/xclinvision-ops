import logging
import os
import sys
import json
import uuid
import base64
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything reads environment variables.
# In Docker the env vars come from docker-compose; .env acts as local fallback.
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path, override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from xclinvision.config import get_class_names, get_num_classes
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

try:
    from .auth import require_auth  # type: ignore[import-not-found]
except ImportError:
    from auth import require_auth  # type: ignore[import-not-found,no-redef]
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, List, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from xclinvision.inference import InferencePipeline
import numpy as np
from PIL import Image
import cv2
import io
import hashlib
from datetime import datetime, timedelta, timezone

from schemas import (
    AnalysisResponse,
    ExplanationParams,
    DashboardFeedbackRequest,
    ChatRequest,
    ChatMessage,
    DashboardReportRequest,
    DriftMetrics,
    PatientHistoryEntry,
    PredictResponse as PredictionResponse,
    FeedbackRequest,
    ReportRequest,
    ExportReportRequest,
)

MAX_UPLOAD_MB = 20
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# PIL decompression-bomb guard: cap pixel count for any decoded image.
# A 20 MB compressed PNG/TIFF can decode to multi-GB pixel buffers, so
# bounding compressed bytes is not enough.
MAX_IMAGE_PIXELS = 50_000_000  # 50 megapixels, well above any clinical CXR
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def _safe_open_rgb(raw: bytes, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    """Decode bytes into an RGB PIL image, refusing decompression bombs.

    Validates declared pixel count via ``Image.size`` BEFORE decoding the
    full pixel buffer (Pillow lazy-loads on first access). Raises
    HTTPException(413) on oversize input.
    """
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    if w * h > max_pixels:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {w}x{h} exceeds {max_pixels} pixel cap",
        )
    return img.convert("RGB")


# Per-store caps now live in app.backend.storage; constants kept here only for
# any external tooling that still imports them.
_IMAGE_STORE_MAX = int(os.environ.get("IMAGE_STORE_MAX", 200))
_IMAGE_STORE_MAX_BYTES = int(os.environ.get("IMAGE_STORE_MAX_BYTES", 256 * 1024 * 1024))


def _warm_start() -> None:
    """Pay the agent import and first model load at boot, not in request #1.

    ``import xclinvision.agent`` pulls chromadb + sentence-transformers and
    costs ~9 s cold; it used to run inside the first /api/v2/analyze call, on
    the event loop, stalling every concurrent request. Set
    ``XCLINVISION_WARM_START=0`` to skip (the test suite does).
    """
    if os.getenv("XCLINVISION_WARM_START", "1").strip().lower() in ("0", "false", "no"):
        logger.info("Warm start disabled via XCLINVISION_WARM_START")
        return
    start = time.time()
    try:
        import xclinvision.agent  # noqa: F401
        logger.info("Warm start: agent module imported (%.1fs)", time.time() - start)
    except Exception as exc:
        logger.warning("Warm start: agent import failed (%s); reports degrade to rule-based", exc)
    try:
        if get_pipeline() is not None:
            logger.info("Warm start: default model loaded (%.1fs total)", time.time() - start)
        else:
            logger.warning("Warm start: no model available to preload")
    except Exception as exc:
        logger.warning("Warm start: model preload failed (%s)", exc)


@asynccontextmanager
async def _lifespan(app: "FastAPI"):
    """Application lifespan — replaces the deprecated @app.on_event('startup').

    ``log_dataset_info`` and ``_warm_start`` are defined later in the module;
    Python resolves the names at call time (server start-up), not at
    definition time, so the forward references are safe.
    """
    await log_dataset_info()
    _warm_start()
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

# ---------------------------------------------------------------------------
# Routers (Issue #8: incremental extraction of the god-module)
# ---------------------------------------------------------------------------
try:
    from .routers.llm import router as _llm_router  # type: ignore[import-not-found]
except ImportError:
    from routers.llm import router as _llm_router  # type: ignore[import-not-found,no-redef]

app.include_router(_llm_router)

# All Pydantic schemas are consolidated in schemas.py (imported at top).


# ---------------------------------------------------------------------------
# Model loading — dynamic model registry with per-architecture caching
# ---------------------------------------------------------------------------

_pipeline_cache: Dict[str, "InferencePipeline"] = {}
_pipeline_lock = threading.Lock()
_MAX_PIPELINE_CACHE = int(os.getenv("XCLINVISION_MAX_CACHED_MODELS", "4"))

# Directory containing exported best model .pth files + _meta.json sidecars
_MODELS_DIR = Path(os.getenv(
    "XCLINVISION_MODELS_DIR",
    str(Path(__file__).parent.parent.parent / "models" / "best_models"),
))


def _discover_models() -> Dict[str, dict]:
    """Scan _MODELS_DIR for *_meta.json files and return {arch_name: metadata}."""
    registry: Dict[str, dict] = {}
    if not _MODELS_DIR.is_dir():
        return registry
    for meta_file in sorted(_MODELS_DIR.glob("*_meta.json")):
        try:
            meta = json.loads(meta_file.read_text())
            arch = meta.get("model_name", "")
            if not arch:
                continue
            pth_path = meta_file.with_name(meta_file.name.replace("_meta.json", ".pth"))
            if not pth_path.exists():
                logger.warning("Model checkpoint not found for %s: %s", arch, pth_path)
                continue
            # Serving labels come from configs/system.yaml, but the checkpoint
            # carries its own class_names. If they disagree, every prediction
            # this model produces would be mislabelled with no error anywhere,
            # so refuse to register it rather than serve a wrong diagnosis.
            meta_classes = meta.get("class_names")
            if meta_classes is not None and list(meta_classes) != get_class_names():
                logger.error(
                    "Refusing model '%s': class_names %s do not match "
                    "configs/system.yaml %s. Predictions would be mislabelled.",
                    arch, list(meta_classes), get_class_names(),
                )
                continue
            meta["_pth_path"] = str(pth_path)
            meta["_meta_path"] = str(meta_file)
            registry[arch] = meta
        except Exception as exc:
            logger.warning("Failed to read model metadata %s: %s", meta_file, exc)
    return registry


# Discover models once at import time; refresh on demand
_model_registry: Dict[str, dict] = _discover_models()


def _build_pipeline(model_path: str, architecture: str, image_size: int):
    """Load a trained checkpoint and return an InferencePipeline.

    Supports both plain state-dict (.pth) and PyTorch Lightning (.ckpt) files.
    """
    import torch
    from xclinvision.modeling import build_model, get_model_normalization
    from xclinvision.inference import InferencePipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(architecture, num_classes=get_num_classes(), pretrained=False, img_size=image_size)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    def _load_checked(state_dict: dict, kind: str) -> None:
        """load_state_dict(strict=False) + refuse a partial load.

        strict=False is needed to tolerate benign extras (aux heads, buffers),
        but it also silently accepts a checkpoint whose keys do not match the
        architecture at all — leaving a randomly-initialised network to serve
        clinical predictions behind an INFO log. Missing keys are therefore
        treated as fatal.
        """
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise ValueError(
                f"{kind} {model_path} left {len(incompatible.missing_keys)} parameter(s) "
                f"uninitialised (first: {incompatible.missing_keys[:3]}); "
                f"architecture '{architecture}' does not match this checkpoint."
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "%s %s carried %d unexpected key(s), ignored (first: %s)",
                kind, model_path, len(incompatible.unexpected_keys),
                incompatible.unexpected_keys[:3],
            )
        logger.info("Loaded %s: %s", kind, model_path)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            new_key = k.replace("model.", "", 1) if k.startswith("model.") else k
            state_dict[new_key] = v
        _load_checked(state_dict, "Lightning checkpoint")
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        _load_checked(checkpoint["model_state_dict"], "exported .pth payload")
    elif isinstance(checkpoint, dict):
        _load_checked(checkpoint, "state_dict checkpoint")
    else:
        raise ValueError(f"Unrecognised checkpoint format in {model_path}")

    model.eval()

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


def get_pipeline(model_name: Optional[str] = None):
    """Return a cached InferencePipeline for the requested model architecture.

    Looks up models in the auto-discovered registry (models/best_models/).
    Falls back to XCLINVISION_MODEL_PATH env var if no registry match.
    Caches pipelines so each architecture is loaded only once.
    """
    # 1. Try to resolve from the auto-discovered registry
    # NOTE: the cache lookup is intentionally done only under the lock. A prior
    # unlocked fast-path read could race with eviction (see _MAX_PIPELINE_CACHE
    # branch below) and raise KeyError between the ``in`` check and the ``[]``.
    if model_name and model_name in _model_registry:
        meta = _model_registry[model_name]
        with _pipeline_lock:
            cached = _pipeline_cache.get(model_name)
            if cached is not None:
                return cached
            try:
                pipeline = _build_pipeline(
                    model_path=meta["_pth_path"],
                    architecture=model_name,
                    image_size=384,  # All best_models are trained at 384
                )
                # Evict oldest entry if cache is full
                if len(_pipeline_cache) >= _MAX_PIPELINE_CACHE:
                    oldest = next(iter(_pipeline_cache))
                    del _pipeline_cache[oldest]
                    logger.info("Evicted model '%s' from pipeline cache", oldest)
                _pipeline_cache[model_name] = pipeline
                logger.info("Loaded model '%s' from registry (%s)", model_name, meta["_pth_path"])
                return pipeline
            except Exception as exc:
                logger.error("Failed to load model '%s': %s", model_name, exc)

    # 2. Fallback to env-var based loading
    env_path = os.getenv("XCLINVISION_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        env_arch = os.getenv("XCLINVISION_ARCHITECTURE", "convnext_small")
        cache_key = f"_env_{env_arch}"
        image_size = int(os.getenv("XCLINVISION_IMAGE_SIZE", "384"))
        with _pipeline_lock:
            cached = _pipeline_cache.get(cache_key)
            if cached is not None:
                return cached
            pipeline = _build_pipeline(env_path, env_arch, image_size)
            _pipeline_cache[cache_key] = pipeline
            return pipeline

    # 3. Auto-load the first available model from the registry
    if _model_registry and not _pipeline_cache:
        first_arch = next(iter(_model_registry))
        return get_pipeline(first_arch)

    return None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
_agent_instance = None
_agent_lock = threading.Lock()

_reasoning_agent_instance = None
_reasoning_agent_lock = threading.Lock()

_prediction_logger = None
_prediction_logger_lock = threading.Lock()


def _get_prediction_logger():
    """Module-level PredictionLogger, created once.

    Nothing wrote prediction logs before, so /api/v2/drift-metrics read an
    empty directory and reported zeros regardless of what the model was doing.
    """
    global _prediction_logger
    with _prediction_logger_lock:
        if _prediction_logger is None:
            from xclinvision.monitoring import PredictionLogger
            _prediction_logger = PredictionLogger()
    return _prediction_logger




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


def _get_reasoning_agent():
    """Return a module-level cached ReasoningAgent (tool-calling reasoning loop)."""
    global _reasoning_agent_instance
    with _reasoning_agent_lock:
        if _reasoning_agent_instance is None:
            try:
                from xclinvision.agent import create_reasoning_agent
                _reasoning_agent_instance = create_reasoning_agent()
            except Exception as exc:
                logger.warning("Failed to initialise reasoning agent: %s", exc)
                raise HTTPException(503, detail=f"Reasoning agent unavailable: {exc}")
    return _reasoning_agent_instance

# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".dcm"}


def _count_images(directory: Path) -> int:
    """Return the number of image files directly inside *directory*."""
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def _dataset_stats(data_dir: Path) -> dict:
    """Build a nested dict with per-split image counts.

    Supports two layouts:
    1. Folder-based: ``data_dir/train/ClassName/*.png``
    2. Manifest-based: ``data_dir/splits/{train,val,test}.txt`` + ``data_dir/images/``
    """
    splits = ["train", "val", "test"]
    stats: dict = {}

    # Try manifest-based layout first (data/processed_384/)
    splits_dir = data_dir / "splits"
    images_dir = data_dir / "images"
    if splits_dir.is_dir() and images_dir.is_dir():
        for split in splits:
            split_file = splits_dir / f"{split}.txt"
            if not split_file.exists():
                continue
            try:
                lines = split_file.read_text().strip().splitlines()
                stats[split] = {
                    "path": str(split_file),
                    "classes": {},
                    "total": len(lines),
                }
            except Exception:
                continue
        # Augment with manifest class counts if available
        manifest = data_dir / "manifest.csv"
        if manifest.exists():
            try:
                import csv
                with open(manifest, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sp = row.get("split", "")
                        label = row.get("label", row.get("class_name", ""))
                        if sp in stats and label:
                            stats[sp]["classes"][label] = stats[sp]["classes"].get(label, 0) + 1
            except Exception:
                pass
        if stats:
            return stats

    # Fallback: folder-based layout (data_dir/train/ClassName/images)
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
    raw_data_dir = os.getenv("XCLINVISION_DATA_DIR", "data/processed_384")
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


@app.get("/api/v1/dataset/info", dependencies=[Depends(require_auth)])
async def dataset_info():
    """Return dataset directories and image counts for each split."""
    raw_data_dir = os.getenv("XCLINVISION_DATA_DIR", "data/processed_384")
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


@app.get("/api/v1/models", dependencies=[Depends(require_auth)])
async def list_models():
    """List available trained models from the auto-discovered registry."""
    global _model_registry
    _model_registry = _discover_models()  # Refresh on each call

    models = []
    for arch, meta in _model_registry.items():
        model_type = "cnn" if any(
            k in arch for k in ("resnet", "densenet", "efficientnet", "convnext")
        ) else "transformer"
        models.append({
            "name": arch,
            "type": model_type,
            "num_classes": meta.get("num_classes", get_num_classes()),
            "class_names": meta.get("class_names", get_class_names()),
            "thresholds": meta.get("thresholds"),
            "best_val_f1": meta.get("best_val_auc"),  # Actually stores F1 macro
            "loaded": arch in _pipeline_cache,
        })
    return {"models": models}


@app.post("/api/v1/predict", response_model=PredictionResponse, dependencies=[Depends(require_auth)])
def predict(
    file: UploadFile = File(...),
    model_name: str = "convnext_small",
    return_explanation: bool = True,
):
    """Predict class for uploaded chest X-ray image."""
    import time
    start_time = time.time()

    # Validate file (issue #6: guard against None content_type)
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type. Please upload an image.")

    # Fix #4: enforce upload size limit before reading into memory.
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
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
        image = _safe_open_rgb(contents)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {str(e)}")

    image_np = np.array(image)

    pipeline = get_pipeline(model_name=model_name)
    if pipeline is None:
        raise HTTPException(
            503,
            detail=(
                "No model is loaded. Ensure trained model checkpoints exist in "
                "models/best_models/ or set the XCLINVISION_MODEL_PATH environment "
                "variable and restart the server."
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


@app.post("/api/v1/explain", dependencies=[Depends(require_auth)])
def explain(
    file: UploadFile = File(...),
    model_name: str = "convnext_small",
    target_class: Optional[int] = None,
):
    """Generate Grad-CAM++ explanation for image."""
    # issue #6: guard against None content_type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type")

    # Fix #2: enforce the same upload size limit as /predict.
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
        )
    try:
        image = _safe_open_rgb(contents)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {str(e)}")

    pipeline = get_pipeline(model_name=model_name)
    if pipeline is None:
        raise HTTPException(
            503,
            detail="No model loaded. Ensure models exist in models/best_models/ and restart the server.",
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


@app.post("/api/v1/report", dependencies=[Depends(require_auth)])
def generate_report(request: ReportRequest):
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
        probabilities=request.probabilities if request.probabilities and len(request.probabilities) == num_classes else [0.0] * num_classes,
        confidence=request.confidence,
        uncertainty_level=request.uncertainty_level,
        highlighted_regions=request.highlighted_regions,
        patient_age=request.patient_age,
        patient_sex=request.patient_sex,
    )

    agent = _get_agent()
    report = agent.generate_report(context)

    return report


@app.post("/api/v1/feedback", dependencies=[Depends(require_auth)])
async def submit_feedback(feedback: FeedbackRequest):
    """Submit clinician feedback for model prediction."""
    # Fix #17: actually persist feedback instead of silently discarding it.
    feedback_entry = {
        "feedback_id": f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now().isoformat(),
        **feedback.model_dump(),
    }
    _feedback_store_append(feedback_entry)
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


@app.get("/api/v1/metrics", dependencies=[Depends(require_auth)])
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
# Dashboard v2: persistent storage (SQLite + filesystem)
# ---------------------------------------------------------------------------
# Backed by app/backend/storage.py. The legacy module-level names
# ``_analysis_store`` / ``_feedback_store`` / ``_image_store`` are preserved
# as thin dict/list-like proxies so existing call sites and tests that poke
# them directly continue to work unchanged.

try:
    from .storage import build_storage_from_env  # type: ignore[import-not-found]
except ImportError:
    # Tests load this file as a top-level ``main`` module (sys.path injected
    # to ``app/backend``); fall back to absolute import in that case.
    from storage import build_storage_from_env  # type: ignore[import-not-found,no-redef]

storage = build_storage_from_env()

_ANALYSIS_STORE_MAX = storage.analysis_max
_FEEDBACK_STORE_MAX = storage.feedback_max


class _AnalysisStoreProxy:
    """Dict-like proxy over storage.analyses (id -> data)."""

    def __getitem__(self, key: str) -> dict:
        v = storage.get_analysis(key)
        if v is None:
            raise KeyError(key)
        return v

    def __setitem__(self, key: str, value: dict) -> None:
        storage.put_analysis(key, value)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and storage.exists_analysis(key)

    def __len__(self) -> int:
        return storage.count_analyses()

    def __iter__(self):
        return iter(storage.all_analyses())

    def get(self, key: str, default=None):
        v = storage.get_analysis(key)
        return v if v is not None else default

    def values(self):
        return list(storage.all_analyses().values())

    def items(self):
        return list(storage.all_analyses().items())

    def keys(self):
        return list(storage.all_analyses().keys())


class _FeedbackStoreProxy:
    """List-like proxy over storage.feedback."""

    def __iter__(self):
        return iter(storage.all_feedback())

    def __reversed__(self):
        return reversed(storage.all_feedback())

    def __len__(self) -> int:
        return storage.count_feedback()

    def __getitem__(self, idx):
        return storage.all_feedback()[idx]

    def append(self, entry: dict) -> None:
        storage.append_feedback(entry)


class _ImageStoreProxy:
    """Dict-like proxy over storage images."""

    def __getitem__(self, key: str) -> bytes:
        v = storage.get_image(key)
        if v is None:
            raise KeyError(key)
        return v

    def __setitem__(self, key: str, value: bytes) -> None:
        storage.put_image(key, value)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and storage.get_image(key) is not None

    def get(self, key: str, default=None):
        v = storage.get_image(key)
        return v if v is not None else default


_analysis_store = _AnalysisStoreProxy()
_feedback_store = _FeedbackStoreProxy()
_image_store = _ImageStoreProxy()


def _analysis_store_put(analysis_id: str, data: dict) -> None:
    """Persist an analysis through the Storage layer."""
    storage.put_analysis(analysis_id, data)


def _feedback_store_append(entry: dict) -> None:
    """Persist a feedback entry through the Storage layer."""
    storage.append_feedback(entry)


def _image_store_put(analysis_id: str, data: bytes) -> None:
    """Persist a raw image through the Storage layer."""
    storage.put_image(analysis_id, data)


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

@app.post("/api/v2/analyze", dependencies=[Depends(require_auth)])
def analyze_image(
    file: UploadFile = File(...),
    patient_id: str = Form(default="UNKNOWN"),
    study_date: str = Form(default=""),
    modality: str = Form(default="X-ray"),
    body_part: str = Form(default="Chest"),
    clinical_history: str = Form(default=""),
    model_name: str = Form(default="convnext_small"),
    xai_method: str = Form(default="gradcam++"),
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
    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
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
            h, w = gray.shape[:2]
            if w * h > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    413,
                    f"Image too large: {w}x{h} exceeds {MAX_IMAGE_PIXELS} pixel cap",
                )
            image_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            image = _safe_open_rgb(contents)
            image_np = np.array(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")
    analysis_id = f"XCL-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

    # --- Run inference -------------------------------------------------------
    pipeline = get_pipeline(model_name=model_name)
    if pipeline is None:
        raise HTTPException(
            503,
            detail="No model loaded. Ensure models exist in models/best_models/ and restart.",
        )

    try:
        result = pipeline.predict(
            image_np,
            return_uncertainty=True,
            return_explanation=True,
            xai_method=xai_method,
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    inference_ms = (time.time() - start_time) * 1000

    # --- Build heatmap images ------------------------------------------------
    heatmap_b64 = None
    overlay_b64 = None
    scorecam_heatmap_b64 = None
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
        # No real XAI available — return null heatmaps instead of fake data
        logger.warning("XAI heatmap unavailable for this analysis — returning null heatmap fields")
        heatmap_b64 = None
        overlay_b64 = None

    # Score-CAM fallback (generated when confidence / plausibility is low)
    scorecam_vis = explanation.get("scorecam_visualization") or {}
    scorecam_cam = scorecam_vis.get("grayscale_cam")
    scorecam_overlay_b64 = None
    if scorecam_cam is not None:
        scorecam_heatmap_b64 = _img_to_base64(scorecam_cam)
        sc_overlay_img = _generate_heatmap_overlay(vis_image, scorecam_cam, opacity=0.45)
        scorecam_overlay_b64 = _img_to_base64(sc_overlay_img)

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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": result["class_name"],
        "confidence": result["confidence"],
        "uncertainty": result.get("uncertainty", {}),
        "uncertainty_level": result.get("uncertainty_level", "unknown"),
        "top_k_predictions": top_k,
        "heatmap_gradcam": heatmap_b64,
        "heatmap_overlay": overlay_b64,
        "scorecam_heatmap": scorecam_heatmap_b64,
        "scorecam_overlay": scorecam_overlay_b64,
        "xai_method": xai_method,
        "region_scores": region_scores if isinstance(region_scores, dict) else {},
        "key_findings": key_findings,
        "llm_summary": llm_summary,
        "inference_time_ms": round(inference_ms, 1),
        "model_version": model_name,
        "image_hash": image_hash,
        "patient_meta": {
            "Patient ID": patient_id or "N/A",
            "Modality": modality or "N/A",
            "Body Part": body_part or "N/A",
            "Clinical History": clinical_history or "N/A",
        },
    }

    # Feed the drift monitor. Best-effort: a logging failure must not fail
    # an analysis the clinician is waiting on.
    try:
        _get_prediction_logger().log_prediction(
            image_hash=image_hash,
            prediction=result["prediction"],
            probabilities=result.get("probabilities", []),
            confidence=result["confidence"],
            uncertainty=result.get("uncertainty"),
            model_version=model_name,
        )
    except Exception as exc:
        logger.warning("Prediction logging failed for %s: %s", analysis_id, exc)

    # Store for later retrieval (compress stored image to save memory)
    _analysis_store_put(analysis_id, analysis_data)
    try:
        _store_img = _safe_open_rgb(contents)
        _buf = io.BytesIO()
        _store_img.save(_buf, format="JPEG", quality=80)
        # Fix #16: use evicting helper to prevent unbounded memory growth.
        _image_store_put(analysis_id, _buf.getvalue())
    except HTTPException:
        raise
    except Exception:
        _image_store_put(analysis_id, contents)

    return analysis_data


# ---------------------------------------------------------------------------
# Dashboard v2: Compare two uploaded images
# ---------------------------------------------------------------------------

def _run_single_analysis(
    contents: bytes, model_name: str, xai_method: str,
) -> dict:
    """Run inference + XAI on raw image bytes and return an analysis dict.

    Shared helper used by the /api/v2/compare endpoint so that we
    don't duplicate the full analysis pipeline.
    """
    image_hash = hashlib.sha256(contents).hexdigest()
    try:
        from xclinvision.processing import read_image_grayscale
        gray = read_image_grayscale(contents)
        if gray is not None:
            h, w = gray.shape[:2]
            if w * h > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    413,
                    f"Image too large: {w}x{h} exceeds {MAX_IMAGE_PIXELS} pixel cap",
                )
            image_np = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            image_np = np.array(_safe_open_rgb(contents))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not process image: {e}")

    pipeline = get_pipeline(model_name=model_name)
    if pipeline is None:
        raise HTTPException(503, "No model loaded.")

    start = time.time()
    result = pipeline.predict(
        image_np, return_uncertainty=True, return_explanation=True,
        xai_method=xai_method,
    )
    inference_ms = (time.time() - start) * 1000

    explanation = result.get("explanation") or {}
    vis = explanation.get("visualization") or {}
    grayscale_cam = vis.get("grayscale_cam")
    _, vis_image = pipeline.preprocess(image_np)

    heatmap_b64 = None
    overlay_b64 = None
    if grayscale_cam is not None:
        heatmap_b64 = _img_to_base64(grayscale_cam)
        overlay_img = _generate_heatmap_overlay(vis_image, grayscale_cam, opacity=0.45)
        overlay_b64 = _img_to_base64(overlay_img)

    # Encode the original uploaded image as a thumbnail
    thumbnail_b64 = _img_to_base64(image_np)

    probs = result.get("probabilities", [])
    class_names = result.get("class_names", get_class_names())
    top_k = sorted(
        [{"class_name": cn, "probability": float(p)} for cn, p in zip(class_names, probs)],
        key=lambda x: x["probability"], reverse=True,
    )

    analysis_id = f"XCL-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    analysis_data = {
        "analysis_id": analysis_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": result["class_name"],
        "confidence": result["confidence"],
        "uncertainty_level": result.get("uncertainty_level", "unknown"),
        "top_k_predictions": top_k,
        "heatmap_gradcam": heatmap_b64,
        "heatmap_overlay": overlay_b64,
        "thumbnail": thumbnail_b64,
        "region_scores": (vis.get("region_scores") or {}),
        "key_findings": explanation.get("key_findings", []),
        "inference_time_ms": round(inference_ms, 1),
        "model_version": model_name,
        "image_hash": image_hash,
    }

    # Store so the analysis can be retrieved later (e.g. for XAI, chat, report)
    _analysis_store_put(analysis_id, analysis_data)
    try:
        _image_store_put(analysis_id, contents)
    except Exception:
        logger.debug("Could not cache image for compare analysis %s", analysis_id)

    return analysis_data


@app.post("/api/v2/compare", dependencies=[Depends(require_auth)])
def compare_images(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    model_name: str = Form(default="convnext_small"),
    xai_method: str = Form(default="gradcam++"),
):
    """Analyze two uploaded images and return side-by-side comparison data."""
    allowed_types = {
        "image/jpeg", "image/png", "image/gif", "image/bmp",
        "image/tiff", "application/octet-stream", "application/dicom",
    }
    for label, f in [("Image A", file_a), ("Image B", file_b)]:
        if f.content_type and f.content_type not in allowed_types:
            raise HTTPException(400, f"{label}: invalid file type '{f.content_type}'.")

    contents_a = file_a.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents_a) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image A too large. Max {MAX_UPLOAD_MB} MB.")
    contents_b = file_b.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents_b) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image B too large. Max {MAX_UPLOAD_MB} MB.")

    analysis_a = _run_single_analysis(contents_a, model_name, xai_method)
    analysis_b = _run_single_analysis(contents_b, model_name, xai_method)

    return {"image_a": analysis_a, "image_b": analysis_b}


# ---------------------------------------------------------------------------
# Dashboard v2: Explanation with adjustable params
# ---------------------------------------------------------------------------

@app.get("/api/v2/explain/{analysis_id}", dependencies=[Depends(require_auth)])
def get_dashboard_explanation(
    analysis_id: str,
    method: str = Query(default="gradcam++"),
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    opacity: float = Query(default=0.6, ge=0.0, le=1.0),
    colormap: str = Query(default="jet"),
    finding: Optional[str] = Query(default=None),
):
    """Regenerate XAI explanation with adjustable threshold and opacity.

    The optional ``finding`` parameter selects which class to generate
    the Grad-CAM for.  When omitted, the top predicted class is used.
    """
    if analysis_id not in _analysis_store:
        raise HTTPException(404, "Analysis not found")

    stored = _analysis_store[analysis_id]
    raw_bytes = _image_store.get(analysis_id)

    if raw_bytes is None:
        raise HTTPException(404, "Original image not found")

    pipeline = get_pipeline(model_name=stored.get("model_version"))
    if pipeline is None:
        raise HTTPException(503, "No model loaded.")

    image = _safe_open_rgb(raw_bytes)
    image_np = np.array(image)

    # Resolve target class index from the finding name
    target_class_idx: Optional[int] = None
    if finding:
        class_names = get_class_names()
        for i, cn in enumerate(class_names):
            if cn.lower() == finding.lower():
                target_class_idx = i
                break

    try:
        result = pipeline.predict(
            image_np, return_uncertainty=False, return_explanation=True,
            xai_method=method, target_class=target_class_idx,
        )
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

@app.get("/api/v2/history/{patient_id}", dependencies=[Depends(require_auth)])
async def get_patient_history(patient_id: str, limit: int = Query(default=50, le=200)):
    """Retrieve all historical analyses for a patient (temporal comparison)."""
    # Indexed lookup (idx_analyses_patient) instead of decoding every stored
    # analysis — rows embed base64 heatmaps and the table caps at 5000.
    history = storage.by_patient(patient_id, limit=limit)

    return [
        {
            "analysis_id": h["analysis_id"],
            "timestamp": h["timestamp"],
            "prediction": h["prediction"],
            "confidence": h["confidence"],
            "uncertainty": h.get("uncertainty", {}),
            "uncertainty_level": h.get("uncertainty_level", "unknown"),
            "top_k_predictions": h.get("top_k_predictions", []),
            "heatmap_overlay": h.get("heatmap_overlay"),
            "llm_summary": h.get("llm_summary", ""),
            "model_version": h.get("model_version", "unknown"),
            "thumbnail": h.get("heatmap_gradcam"),
        }
        for h in history[:limit]
    ]


# ---------------------------------------------------------------------------
# Dashboard v2: Feedback
# ---------------------------------------------------------------------------

@app.post("/api/v2/feedback", dependencies=[Depends(require_auth)])
async def submit_dashboard_feedback(feedback: DashboardFeedbackRequest):
    """Store clinician feedback from the dashboard UI."""
    entry = feedback.model_dump()
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    entry["feedback_id"] = f"fb-{uuid.uuid4().hex[:8]}"
    _feedback_store_append(entry)

    logger.info(
        "Feedback received: %s for analysis %s",
        feedback.feedback_type, feedback.analysis_id,
    )

    return {"status": "recorded", "feedback_id": entry["feedback_id"]}


# ---------------------------------------------------------------------------
# Dashboard v2: LLM Chat
# ---------------------------------------------------------------------------

@app.post("/api/v2/chat", dependencies=[Depends(require_auth)])
def llm_chat(request: ChatRequest):
    """Context-aware LLM chat powered by the reasoning agent.

    The reasoning agent classifies user intent, selects & executes tools,
    and synthesises a contextual response — far richer than the old
    "re-generate full report per message" approach.
    """
    stored = _analysis_store.get(request.analysis_id)
    if not stored:
        raise HTTPException(404, "Analysis not found")

    try:
        reasoning_agent = _get_reasoning_agent()

        # Build conversation history for context
        history = [
            {"role": m.role, "content": m.content}
            for m in (request.history or [])[-8:]
        ]

        # Extra context the tools may need
        extra_context = {
            "analysis_store": storage.summaries(),
            "feedback_store": storage.all_feedback(),
        }

        agent_response = reasoning_agent.process_message(
            message=request.message,
            analysis=stored,
            history=history,
            extra_context=extra_context,
        )

        return agent_response.to_api_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("LLM chat error: %s", e)
        # Graceful fallback — still useful even if reasoning agent fails
        response_text = (
            f"Analysis shows {stored['prediction']} ({stored['confidence']:.1%} confidence). "
            f"Regarding '{request.message}': Clinical correlation is recommended. "
            "Please consult with a specialist for definitive interpretation."
        )
        return {
            "response": response_text,
            "suggested_followups": [],
            "references": [],
            "reasoning_trace": [],
            "intent": "general_question",
            "tools_used": [],
        }


# ---------------------------------------------------------------------------
# Dashboard v2: Report generation
# ---------------------------------------------------------------------------

@app.post("/api/v2/generate-report", dependencies=[Depends(require_auth)])
async def generate_dashboard_report(request: DashboardReportRequest):
    """Generate a structured clinical report from one or more analyses."""
    analyses = [_analysis_store.get(aid) for aid in request.analysis_ids]
    analyses = [a for a in analyses if a is not None]

    if not analyses:
        raise HTTPException(404, "No analyses found for the given IDs")

    primary = analyses[0]

    # Build sections
    findings = (
        f"AI analysis of chest X-ray ({primary['timestamp'][:10]}, "
        f"model {primary.get('model_version', 'unknown')}):\n\n"
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Dashboard v2: HTML report export (ClinicalReporter)
# ---------------------------------------------------------------------------


@app.post("/api/v2/export-report", dependencies=[Depends(require_auth)])
def export_report_html(request: ExportReportRequest):
    """Generate a clinical report in HTML, PDF, or JSON format."""
    stored = _analysis_store.get(request.analysis_id)
    if not stored:
        raise HTTPException(404, "Analysis not found")

    report_id = f"RPT-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── JSON shortcut: return structured data directly ────────────────
    if request.format == "json":
        return {
            "json": {
                "report_id": report_id,
                "analysis_id": request.analysis_id,
                "patient_id": stored.get("patient_id", ""),
                "patient_meta": stored.get("patient_meta", {}),
                "prediction": stored["prediction"],
                "confidence": stored["confidence"],
                "uncertainty": stored.get("uncertainty", {}),
                "uncertainty_level": stored.get("uncertainty_level", "unknown"),
                "top_k_predictions": stored.get("top_k_predictions", []),
                "key_findings": stored.get("key_findings", []),
                "llm_summary": stored.get("llm_summary", ""),
                "model_version": stored.get("model_version", ""),
                "region_scores": stored.get("region_scores", {}),
                "indication": request.indication or "",
                "comments": request.comments or "",
                "timestamp": timestamp,
            },
            "report_id": report_id,
            "format": "json",
            "timestamp": timestamp,
        }

    # ── HTML / PDF: full ClinicalReporter pipeline ────────────────────
    try:
        from xclinvision.agent import ClinicalContext, create_agent
        from xclinvision.agent.reporter import ClinicalReporter
        from xclinvision.agent.xclinvisionagent import ClinicalReport
        import numpy as np

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
        report_data = agent.generate_report(context)

        raw_regions = stored.get("region_scores", {})
        spatial_evidence = {
            k: f"{float(v):.2f}" if isinstance(v, (int, float)) else str(v)
            for k, v in raw_regions.items()
        }
        clinical_report = ClinicalReport(
            findings=report_data.get("key_findings", [stored["prediction"]]),
            spatial_evidence=spatial_evidence,
            reasoning_trace=report_data.get("reasoning_trace", ""),
            differential_diagnosis=report_data.get("differential_diagnosis", []),
            impression=report_data.get("impression", report_data.get("findings", "")),
            urgency=report_data.get("urgency", "Medium"),
            next_steps=report_data.get("next_steps", [report_data.get("recommendation", "Clinical correlation recommended.")]),
            citations=report_data.get("citations", []),
        )

        vision_data = {
            "class_names": class_names,
            "probabilities": probs,
        }
        # Embed XAI heatmaps into the report if available
        if request.include_xai:
            explanation_data = {}
            used_method = stored.get("xai_method", "gradcam++")
            # The stored heatmaps are base64 PNGs; decode them to numpy arrays
            # for the reporter's overlay generation.
            if stored.get("heatmap_gradcam"):
                try:
                    import base64 as _b64
                    hm_bytes = _b64.b64decode(stored["heatmap_gradcam"])
                    hm_img = Image.open(io.BytesIO(hm_bytes)).convert("L")
                    hm_np = np.array(hm_img).astype(np.float32) / 255.0
                    # Map to the correct key so the report template shows the right label
                    if used_method == "attention_rollout":
                        explanation_data["attention_map"] = hm_np
                    elif used_method == "scorecam":
                        explanation_data["scorecam"] = hm_np
                    else:
                        explanation_data["heatmap"] = hm_np
                except Exception:
                    pass
            # Score-CAM fallback heatmap (generated for low-confidence predictions)
            if stored.get("scorecam_heatmap"):
                try:
                    import base64 as _b64
                    sc_bytes = _b64.b64decode(stored["scorecam_heatmap"])
                    sc_img = Image.open(io.BytesIO(sc_bytes)).convert("L")
                    explanation_data["scorecam"] = np.array(sc_img).astype(np.float32) / 255.0
                except Exception:
                    pass
            if stored.get("explanation"):
                explanation_data.update(stored["explanation"])
            if explanation_data:
                vision_data["explanation"] = explanation_data

        # Use the real uploaded image when available
        raw_bytes = _image_store.get(request.analysis_id)
        if raw_bytes:
            _pil = _safe_open_rgb(raw_bytes)
            image_data = np.array(_pil)
        else:
            image_data = np.full((384, 384, 3), 128, dtype=np.uint8)

        reporter = ClinicalReporter()
        html = reporter.generate_html(
            report=clinical_report,
            vision_data=vision_data,
            image_data=image_data,
            patient_meta=stored.get("patient_meta"),
            indication=request.indication,
            conversation_log=request.conversation_log,
            comments=request.comments,
        )

        # ── PDF conversion ────────────────────────────────────────────
        if request.format == "pdf":
            try:
                from weasyprint import HTML as WeasyprintHTML  # type: ignore
                pdf_bytes = WeasyprintHTML(string=html).write_pdf()
                import base64 as b64mod
                pdf_b64 = b64mod.b64encode(pdf_bytes).decode("ascii")
                return {
                    "pdf_base64": pdf_b64,
                    "report_id": report_id,
                    "format": "pdf",
                    "timestamp": timestamp,
                }
            except ImportError:
                raise HTTPException(
                    500,
                    "PDF generation requires weasyprint. "
                    "Install it with: pip install weasyprint",
                )

        # ── Default: HTML ─────────────────────────────────────────────
        return {
            "html": html,
            "report_id": report_id,
            "format": "html",
            "timestamp": timestamp,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Report export error: %s", e)
        raise HTTPException(500, detail=f"Failed to generate report: {e}")


# ---------------------------------------------------------------------------
# Dashboard v2: Drift metrics
# ---------------------------------------------------------------------------

@app.get("/api/v2/drift-metrics", dependencies=[Depends(require_auth)])
async def get_drift_metrics(days: int = Query(default=30, ge=1, le=365)):
    """Return drift monitoring metrics from prediction logs."""
    # Reference mean confidence for a healthy model. This is a configured
    # value, not a measured one — the evaluation reports carry no mean-
    # confidence field. Calibrate it per model before trusting drift_detected.
    baseline_conf = float(os.getenv("XCLINVISION_DRIFT_BASELINE_CONFIDENCE", "0.85"))
    status = "ok"
    try:
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        history = _get_prediction_logger().get_prediction_history(start_date=start_date)

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

            # Simple drift score: deviation from the configured mean confidence
            drift_score = abs(avg_conf - baseline_conf) * 2
        else:
            # No logged predictions in the window. Say so — zeros here read as
            # "measured, and healthy", which is the opposite of the truth.
            status = "insufficient_data"
            avg_conf = 0.0
            avg_unc = 0.0
            pred_dist = {}
            drift_score = 0.0

    except Exception as e:
        logger.warning("Drift metric computation failed: %s", e)
        status = "error"
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
        "status": status,
        "baseline_confidence": baseline_conf,
        "drift_score": round(drift_score, 4),
        "drift_detected": status == "ok" and drift_score > 0.2,
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

    # Load real performance metrics from evaluation reports if available
    eval_base = Path(__file__).parent.parent.parent / "outputs"
    performance = {}
    for model_key, dir_name in [
        ("vit_base", "evaluation_384_vit_base"),
        ("convnext_small", "evaluation_384_convnext_small"),
        ("efficientnet_b0", "evaluation_384_efficientnet_b0"),
        ("densenet", "evaluation_384_densenet"),
    ]:
        report_dir = eval_base / dir_name
        if not report_dir.is_dir():
            continue
        candidates = sorted(report_dir.glob("*_test_evaluation_report.json"))
        if not candidates:
            continue
        try:
            import json as _json
            rpt = _json.loads(candidates[0].read_text())
            performance[model_key] = {
                "macro_auc": rpt.get("macro_auc", 0),
                "macro_f1": rpt.get("macro_f1", 0),
                "subset_accuracy": rpt.get("subset_accuracy", 0),
                "ece": rpt.get("calibration", {}).get("expected_calibration_error", 0),
            }
        except Exception:
            pass

    return {
        "name": "XClinVision ChestX-ray",
        "version": "1.0.0",
        "last_updated": "2026-04-01",
        "intended_use": f"Detection and characterization of thoracic diseases in chest X-rays ({', '.join(get_class_names())})",
        "performance": performance if performance else {
            "note": "No evaluation reports found. Run scripts/evaluate.py",
        },
        "limitations": [
            "Not validated for pediatric populations (<18 years)",
            "Reduced performance for subtle findings <5mm",
            "Trained on frontal view only (PA/AP)",
            "May miss subtle interstitial patterns",
            "Performance degrades on images from non-standard equipment",
        ],
        "training_data": "VinBigData Chest X-ray (2021) — 14,304 frontal radiographs, 5-class multilabel (train 10,020 / val 2,133 / test 2,151)",
        "architectures_available": [
            "vit_base", "convnext_small", "efficientnet_b0", "densenet",
        ],
        "certifications": ["Research Use Only — Not FDA cleared"],
        "model_card_md": model_card_text[:500] if model_card_text else "",
    }


# ---------------------------------------------------------------------------
# Dashboard v2: Feedback statistics
# ---------------------------------------------------------------------------

@app.get("/api/v2/feedback-stats", dependencies=[Depends(require_auth)])
async def get_feedback_stats():
    """Return aggregated feedback statistics for the audit dashboard."""
    total = len(_feedback_store)
    by_type: Dict[str, int] = {}
    recent: List[dict] = []

    for fb in _feedback_store:
        ft = fb.get("feedback_type", "unknown")
        by_type[ft] = by_type.get(ft, 0) + 1

    # Most recent 50 entries, newest first
    for fb in reversed(_feedback_store[-50:]):
        recent.append({
            "feedback_id": fb.get("feedback_id", ""),
            "analysis_id": fb.get("analysis_id", ""),
            "feedback_type": fb.get("feedback_type", ""),
            "user_id": fb.get("user_id", "anonymous"),
            "notes": fb.get("notes"),
            "timestamp": fb.get("timestamp", ""),
        })

    correct = by_type.get("correct", 0)
    incorrect = by_type.get("incorrect", 0)
    correction_rate = (incorrect / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "by_type": by_type,
        "correction_rate": round(correction_rate, 1),
        "recent": recent,
    }


# ---------------------------------------------------------------------------
# LLM Provider Management — moved to app/backend/routers/llm.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Streaming Chat (Server-Sent Events)
# ---------------------------------------------------------------------------


def _build_sse_event(data: str, event: str = "message") -> str:
    """Format a single SSE frame.

    SSE uses newlines as frame delimiters.  Any literal newlines inside
    *data* must be sent as separate ``data:`` lines so the client can
    reassemble them.
    """
    lines = data.split("\n")
    data_part = "\n".join(f"data: {line}" for line in lines)
    return f"event: {event}\n{data_part}\n\n"


@app.post("/api/v2/chat/stream", dependencies=[Depends(require_auth)])
async def llm_chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events.

    Sends real-time token-by-token responses to the frontend for
    a typing-indicator UX.  Falls back to a single non-streamed
    message if streaming is unavailable.
    """
    stored = _analysis_store.get(request.analysis_id)
    if not stored:
        raise HTTPException(404, "Analysis not found")

    def event_generator():
        try:
            from xclinvision.agent.llm_manager import get_llm_manager

            manager = get_llm_manager()

            # Build the prompt from the reasoning agent's synthesis logic
            reasoning_agent = _get_reasoning_agent()

            # Classify intent & plan tools
            from xclinvision.agent.reasoning import classify_intent, _INTENT_TOOL_MAP
            intent = classify_intent(request.message)

            # Build context
            history = [
                {"role": m.role, "content": m.content}
                for m in (request.history or [])[-8:]
            ]

            extra_context = {
                "analysis_store": storage.summaries(),
                "feedback_store": storage.all_feedback(),
            }

            # Execute tools first (non-streamed)
            tool_names = _INTENT_TOOL_MAP.get(intent, ["get_prediction_details"])

            tool_context = {**extra_context}
            tool_context["analysis"] = stored
            tool_context["model_name"] = stored.get("model_version", "vit_base")

            tool_results = {}
            for name in tool_names:
                result = reasoning_agent.tools.execute(name, tool_context)
                tool_results[name] = result

            # Build the synthesis prompt
            tool_block = ""
            for name, result in tool_results.items():
                tool_block += f"\n### Tool: {name}\n{result.to_prompt_text()}\n"

            history_block = ""
            if history:
                recent = history[-6:]
                history_block = "\n".join(
                    f"{'User' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')}"
                    for m in recent
                )

            system_prompt = (
                "You are XClinVision's clinical reasoning assistant — a helpful, "
                "natural, conversational AI that answers questions about chest X-ray "
                "analyses. You have just executed tools to gather evidence.\n\n"
                "RULES:\n"
                "- Respond in clear, concise natural language — NOT raw JSON.\n"
                "- You are an ASSISTIVE tool. You do NOT replace a radiologist.\n"
                "- Ground every claim in the tool outputs provided. Do NOT invent findings.\n"
                "- When uncertain, say so explicitly.\n"
                "- Use evidence-based clinical language but keep it accessible.\n"
                "- Be concise but thorough. Avoid repeating what you said before.\n"
                "- If tool data is missing or errored, acknowledge it gracefully.\n"
                "- Never return raw JSON, debug output, or code — only human-readable text."
            )

            user_prompt = (
                f"### User's Question\n{request.message}\n\n"
                f"### Detected Intent\n{intent}\n\n"
                f"### Tool Outputs\n{tool_block}\n\n"
                f"### Conversation History\n{history_block or 'No prior conversation.'}\n\n"
                "---\n"
                "Provide a focused, professional response to the user's question. "
                "Reference the tool outputs as evidence. Keep it concise."
            )

            # Send metadata event
            meta = json.dumps({
                "intent": intent,
                "tools_used": list(tool_results.keys()),
                "provider": manager.active_name,
            })
            yield _build_sse_event(meta, event="metadata")

            # Stream LLM response
            for chunk in manager.stream_llm(system_prompt, user_prompt, temperature=0.25):
                yield _build_sse_event(json.dumps({"token": chunk}), event="token")

            # Done event
            yield _build_sse_event(json.dumps({"status": "done"}), event="done")

        except Exception as e:
            logger.warning("Streaming chat error: %s", e, exc_info=True)
            # Send a fallback non-streamed response as SSE
            fallback = (
                f"I found that the analysis shows **{stored['prediction']}** "
                f"with {stored['confidence']:.0%} confidence. "
                f"Regarding your question about '{request.message}': "
                "I'm currently unable to provide a detailed AI-powered response "
                "(the language model may be temporarily unavailable). "
                "Clinical correlation is recommended — please consult with a specialist."
            )
            yield _build_sse_event(
                json.dumps({"token": fallback}), event="token",
            )
            yield _build_sse_event(
                json.dumps({"status": "done", "fallback": True}), event="done",
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""Centralized pytest fixtures for the XClinVision test suite.

Provides shared fixtures for the FastAPI test client, dummy images,
and mocked ML pipelines so individual test modules don't need to
duplicate boilerplate.
"""

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure backend and src packages are importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "app" / "backend"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Point persistent storage at a per-run temp directory before main is imported.
# ---------------------------------------------------------------------------
_TMP_STORAGE_DIR = Path(tempfile.mkdtemp(prefix="xclinvision_test_storage_"))
os.environ.setdefault("XCLINVISION_DB_PATH", str(_TMP_STORAGE_DIR / "xclinvision.db"))
os.environ.setdefault("XCLINVISION_IMAGE_DIR", str(_TMP_STORAGE_DIR / "images"))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def app():
    """Session-scoped FastAPI application instance."""
    from main import app as _app  # type: ignore[import-not-found]
    return _app


@pytest.fixture(scope="session")
def client(app):
    """Session-scoped synchronous TestClient."""
    from starlette.testclient import TestClient
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def dummy_image_bytes() -> bytes:
    """Create a minimal 64×64 RGB JPEG in memory."""
    img = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def fake_pipeline() -> MagicMock:
    """Return a mock InferencePipeline with realistic predict() output."""
    pipeline = MagicMock()
    pipeline.predict.return_value = {
        "prediction": 0,
        "class_name": "No finding",
        "probabilities": [0.85, 0.05, 0.04, 0.03, 0.03],
        "confidence": 0.85,
        "uncertainty": {"epistemic": 0.02, "aleatoric": 0.01},
        "uncertainty_level": "low",
        "explanation": {
            "key_findings": ["Normal cardiac silhouette"],
            "clinical_plausibility": 0.9,
            "visualization": {
                "grayscale_cam": np.random.rand(64, 64).astype(np.float32),
                "region_scores": {"left_lung": 0.3, "right_lung": 0.25},
                "method": "gradcam++",
            },
        },
        "predictions_multilabel": [1, 0, 0, 0, 0],
        "class_names_predicted": ["No finding"],
    }
    pipeline.preprocess.return_value = (
        np.random.rand(3, 384, 384).astype(np.float32),
        np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8),
    )
    return pipeline

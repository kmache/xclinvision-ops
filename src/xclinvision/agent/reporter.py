"""Report Synthesis Engine – professional clinical document generation.

Takes a :class:`ClinicalReport` from the reasoning agent, augments it with
visual diagnostics (Grad-CAM++ overlay, radar chart), and renders a
self-contained HTML document suitable for clinical review or PDF export.

Components
----------
1. **Calibration Layer** – maps raw probabilities to clinical language.
2. **Visual Dashboard** – Grad-CAM overlay + radar chart (matplotlib).
3. **Templating** – Jinja2-driven HTML with embedded base64 assets.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from jinja2 import Environment, FileSystemLoader

import pathlib

from xclinvision.agent.xclinvisionagent import ClinicalReport
from xclinvision.config import get_class_names

if TYPE_CHECKING:
    from xclinvision.agent.guardrails import ThresholdProfile

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# 1.  Calibration Layer
# ════════════════════════════════════════════════════════════════════════════════


#: Width of the band just below threshold where a finding is worth a second
#: look rather than being called absent. Outside it, a sub-threshold score is
#: reported as negative — the pipeline classified it that way.
EQUIVOCAL_MARGIN = 0.10


def calibrate_probability(prob: float, threshold: float = 0.50) -> str:
    """Map a raw probability to clinical language.

    ``threshold`` must be the same per-class threshold the pipeline used to
    decide positive/negative, otherwise the report contradicts the
    classification. A score below it is NEVER rendered in positive language.

    Bands
    -----
    >= threshold + 0.35                  → "Highly suggestive"
    >= threshold                         → "Consistent with"
    >= threshold - EQUIVOCAL_MARGIN      → "Equivocal; consider clinical correlation"
    otherwise                            → "No significant evidence"
    """
    if prob >= threshold + 0.35:
        return "Highly suggestive"
    if prob >= threshold:
        return "Consistent with"
    if prob >= threshold - EQUIVOCAL_MARGIN:
        return "Equivocal; consider clinical correlation"
    return "No significant evidence"


def calibrate_predictions(
    class_names: List[str],
    probabilities: List[float],
    threshold_profile: Optional[ThresholdProfile] = None,
) -> List[Dict[str, Any]]:
    """Return a list of ``{class_name, probability, clinical_term}`` dicts,
    sorted by descending probability."""
    if len(class_names) != len(probabilities):
        raise ValueError(
            f"class_names ({len(class_names)}) and probabilities "
            f"({len(probabilities)}) must have the same length."
        )
    pairs = sorted(
        zip(class_names, probabilities, strict=True), key=lambda x: x[1], reverse=True
    )
    return [
        {
            "class_name": name,
            "probability": prob,
            "pct": f"{prob:.1%}",
            "clinical_term": calibrate_probability(
                prob,
                threshold_profile.get_threshold(name) if threshold_profile else 0.50,
            ),
        }
        for name, prob in pairs
    ]


# ════════════════════════════════════════════════════════════════════════════════
# 2.  Visual Dashboard helpers
# ════════════════════════════════════════════════════════════════════════════════


def _fig_to_base64(fig: Figure, *, dpi: int = 150) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", transparent=False)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_radar_chart(
    class_names: List[str],
    probabilities: List[float],
    *,
    baseline: float = 0.10,
    title: str = "Prediction Profile vs Normal Baseline",
) -> str:
    """Create a spider / radar chart comparing predictions to a baseline.

    Returns a base64-encoded PNG string.
    """
    n = len(class_names)
    if n == 0:
        return ""

    # Compute angles (one per class, closing the loop)
    angles = [i / n * 2 * np.pi for i in range(n)]
    angles += angles[:1]

    values = list(probabilities) + [probabilities[0]]
    baseline_vals = [baseline] * n + [baseline]

    fig = Figure(figsize=(7, 7))
    FigureCanvasAgg(fig)  # attach renderer (thread-safe, no global state)
    ax = fig.add_subplot(111, polar=True)

    # Baseline
    ax.plot(angles, baseline_vals, linewidth=1.5, linestyle="--", color="#999999", label="Normal baseline")
    ax.fill(angles, baseline_vals, alpha=0.08, color="#999999")

    # Predictions
    ax.plot(angles, values, linewidth=2, color="#1f77b4", label="AI Prediction")
    ax.fill(angles, values, alpha=0.20, color="#1f77b4")

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, size=8, wrap=True)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=7, color="#666")
    ax.set_title(title, size=12, weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=8)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.50,
) -> str:
    """Blend a Grad-CAM heatmap onto the original X-ray and return base64 PNG.

    Parameters
    ----------
    image:
        Original image as a numpy array (H×W or H×W×3, uint8 or float [0,1]).
    heatmap:
        Grad-CAM activation map (H×W, float [0,1]).
    alpha:
        Blending weight for the heatmap overlay.
    """
    # Normalise image to uint8 for cv2 (handles uint16, float32/64, etc.)
    if image.dtype != np.uint8:
        img_min, img_max = float(image.min()), float(image.max())
        if img_max > img_min:
            img = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(image, dtype=np.uint8)
    else:
        img = image.copy()

    # Ensure 3-channel
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize heatmap to match image
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    coloured = cv2.applyColorMap(np.uint8(255 * np.clip(heatmap, 0, 1)), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 1.0 - alpha, coloured, alpha, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    # Encode to PNG → base64
    _, buf = cv2.imencode(".png", cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ════════════════════════════════════════════════════════════════════════════════
# 3.  ClinicalReporter
# ════════════════════════════════════════════════════════════════════════════════


def _image_to_base64_png(image: np.ndarray) -> str:
    """Encode a numpy image (H×W or H×W×3) to a base64 PNG string."""
    if image.dtype != np.uint8:
        img_min, img_max = float(image.min()), float(image.max())
        if img_max > img_min:
            image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf.tobytes()).decode("ascii")


class ClinicalReporter:
    """Renders a :class:`ClinicalReport` into a self-contained clinical HTML document.

    Parameters
    ----------
    class_names:
        Ordered disease class labels (defaults to ``system.yaml``).
    gradcam_alpha:
        Blending weight for the Grad-CAM overlay.
    radar_baseline:
        Baseline probability shown on the radar chart (represents "normal").
    """

    def __init__(
        self,
        *,
        class_names: Optional[List[str]] = None,
        gradcam_alpha: float = 0.50,
        radar_baseline: float = 0.10,
        threshold_profile: Optional[ThresholdProfile] = None,
    ) -> None:
        self.class_names = class_names or get_class_names()
        self.gradcam_alpha = gradcam_alpha
        self.radar_baseline = radar_baseline
        self.threshold_profile = threshold_profile
        template_dir = pathlib.Path(__file__).parent / "templates"
        self._env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=True)
        self._template = self._env.get_template("clinical_report.html")

    # ── Public API ────────────────────────────────────────────────────────

    def generate_html(
        self,
        report: ClinicalReport,
        vision_data: Dict[str, Any],
        image_data: np.ndarray,
        *,
        patient_meta: Optional[Dict[str, Any]] = None,
        indication: str = "",
        conversation_log: Optional[List[Dict[str, str]]] = None,
        comments: str = "",
    ) -> str:
        """Produce a self-contained HTML string with embedded images/charts.

        Parameters
        ----------
        report:
            The :class:`ClinicalReport` returned by the reasoning agent.
        vision_data:
            Raw output of ``InferencePipeline.predict()`` – must contain
            ``class_names``, ``probabilities``, and optionally
            ``explanation.heatmap``, ``explanation.scorecam``,
            ``explanation.attention_map``.
        image_data:
            Original X-ray image as a numpy array (H×W or H×W×3).
        patient_meta:
            Optional patient metadata dict (``age``, ``sex``, etc.).
        indication:
            Clinical indication / reason for exam.
        conversation_log:
            Optional list of ``{"role": "...", "content": "..."}`` dicts.
        comments:
            Additional free-text comments for the report.
        """
        class_names: List[str] = vision_data.get("class_names", self.class_names)
        probabilities: List[float] = vision_data.get("probabilities", [])

        # 1. Calibrated findings table
        calibrated = calibrate_predictions(class_names, probabilities, self.threshold_profile)

        # 2. Original input image
        original_b64 = ""
        if image_data is not None and image_data.size > 0:
            original_b64 = _image_to_base64_png(image_data)

        # 3. Grad-CAM overlay
        explanation = vision_data.get("explanation")
        heatmap = explanation.get("heatmap") if isinstance(explanation, dict) else None
        gradcam_b64 = ""
        if heatmap is not None:
            gradcam_b64 = generate_gradcam_overlay(
                image_data, np.asarray(heatmap), alpha=self.gradcam_alpha
            )

        # 4. Score-CAM overlay (if available)
        scorecam_b64 = ""
        scorecam_map = explanation.get("scorecam") if isinstance(explanation, dict) else None
        if scorecam_map is not None:
            scorecam_b64 = generate_gradcam_overlay(
                image_data, np.asarray(scorecam_map), alpha=self.gradcam_alpha
            )

        # 5. Attention map overlay (if available)
        attention_b64 = ""
        attention_map = explanation.get("attention_map") if isinstance(explanation, dict) else None
        if attention_map is not None:
            attention_b64 = generate_gradcam_overlay(
                image_data, np.asarray(attention_map), alpha=self.gradcam_alpha
            )

        # 6. Radar chart
        radar_b64 = ""
        if probabilities:
            radar_b64 = generate_radar_chart(
                class_names, probabilities, baseline=self.radar_baseline
            )

        # 7. Render template
        report_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = self._template.render(
            report=report,
            calibrated=calibrated,
            original_b64=original_b64,
            gradcam_b64=gradcam_b64,
            scorecam_b64=scorecam_b64,
            attention_b64=attention_b64,
            radar_b64=radar_b64,
            patient_meta=patient_meta or {},
            indication=indication or "",
            conversation_log=conversation_log or [],
            comments=comments or "",
            report_id=report_id,
            generated_at=generated_at,
        )
        return html

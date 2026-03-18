"""Page: Historical Comparison — Temporal patient comparison with heatmaps and trends.

Compare a patient's current chest X-ray exam with a historical exam.
Visualises Grad-CAM overlays, a pixel-level difference map, and a
probability-over-time trend chart, plus an AI-generated clinical summary.
"""

# ==============================================================================
# Imports
# ==============================================================================
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles import COLORS
from config import CLASS_NAMES

# ==============================================================================
# Constants / mock data
# ==============================================================================
FINDINGS = CLASS_NAMES
THRESHOLD = 0.50

HISTORICAL_STUDIES = ["2023-08-15, PA", "2023-09-20, PA", "2023-10-10, PA"]
CURRENT_STUDIES = ["2023-10-26, PA"]

# Mock prediction tables keyed by study label — generated dynamically from CLASS_NAMES
def _make_mock_predictions() -> dict[str, list[dict]]:
    """Build mock prediction tables using the configured class names."""
    import random
    random.seed(42)
    studies = HISTORICAL_STUDIES + CURRENT_STUDIES
    base_probs = [round(random.uniform(0.10, 0.90), 2) for _ in CLASS_NAMES]
    result: dict[str, list[dict]] = {}
    for i, study in enumerate(studies):
        result[study] = [
            {"Finding": name, "Probability": round(min(base_probs[j] + i * 0.05, 0.99), 2), "Threshold": 0.50}
            for j, name in enumerate(CLASS_NAMES)
        ]
    return result

_MOCK_PREDICTIONS = _make_mock_predictions()

# Probability-over-time mock data (per finding)
def _make_temporal_data() -> dict:
    import random
    random.seed(7)
    dates = ["2023-08-15", "2023-09-20", "2023-10-10", "2023-10-26"]
    result = {}
    for name in CLASS_NAMES:
        base = round(random.uniform(0.10, 0.85), 2)
        result[name] = {
            "dates": dates,
            "probs": [round(min(base + i * 0.04, 0.99), 2) for i in range(len(dates))],
        }
    return result

_TEMPORAL_DATA = _make_temporal_data()

_CLINICAL_SUMMARIES = {
    "Improved": (
        "Findings suggest regression of infiltrate in the right middle lobe compared "
        "with the previous study. Patient's condition has improved, with reduced opacity "
        "and better-defined lung margins."
    ),
    "Worsened": (
        "Findings suggest progression of infiltrate in the right middle lobe since the "
        "August 15th study. Patient's condition appears to have worsened; clinical "
        "correlation and follow-up imaging are recommended."
    ),
    "No significant change": (
        "No significant interval change is identified compared with the prior study. "
        "Lung fields appear stable. Continued monitoring is advised."
    ),
}

# ==============================================================================
# Helper functions
# ==============================================================================

def load_exam_image(size: tuple = (224, 224)) -> np.ndarray:
    """Return a synthetic grayscale chest-X-ray placeholder (numpy uint8 RGB array)."""
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)
    img[:] = 18  # dark background

    # Lung fields — two bright ellipses
    cv2.ellipse(img, (w // 2 - 45, h // 2 + 10), (55, 80), 0, 0, 360, 160, -1)
    cv2.ellipse(img, (w // 2 + 45, h // 2 + 10), (55, 80), 0, 0, 360, 160, -1)

    # Rib-like arcs
    for rib_y in range(h // 4, int(h * 0.75), 22):
        cv2.ellipse(img, (w // 2, rib_y),
                    (int(w * 0.44), int(h * 0.12)), 0, 180, 360, 80, 1)

    # Spine / sternum
    cv2.line(img, (w // 2, h // 5), (w // 2, int(h * 0.85)), 120, 3)

    # Clavicles
    cv2.line(img, (w // 2 - 60, h // 4 - 20), (w // 2, h // 4 + 10), 130, 2)
    cv2.line(img, (w // 2 + 60, h // 4 - 20), (w // 2, h // 4 + 10), 130, 2)

    img = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def generate_mock_predictions(study_label: str) -> pd.DataFrame:
    """Return a DataFrame of mock predictions for the given study label."""
    rows = _MOCK_PREDICTIONS.get(study_label, _MOCK_PREDICTIONS["2023-10-26, PA"])
    df = pd.DataFrame(rows)
    df["Binary Label"] = df.apply(
        lambda r: "Positive" if r["Probability"] >= r["Threshold"] else "Negative",
        axis=1,
    )
    return df


def generate_gradcam_heatmap(
    base_img: np.ndarray,
    finding: str,
    study_label: str,
    intensity: float = 0.55,
) -> np.ndarray:
    """Overlay a synthetic Grad-CAM heatmap on *base_img* and return an RGB array."""
    h, w = base_img.shape[:2]

    _lobe_centers = {
        "Pneumonia":    (int(w * 0.63), int(h * 0.55)),
        "Effusion":     (int(w * 0.37), int(h * 0.65)),
        "Cardiomegaly": (int(w * 0.50), int(h * 0.55)),
    }
    cx, cy = _lobe_centers.get(finding, (w // 2, h // 2))

    # Gaussian activation blob
    xs = np.arange(w)
    ys = np.arange(h)
    xx, yy = np.meshgrid(xs, ys)
    activation = np.exp(
        -((xx - cx) ** 2 / (2 * (w * 0.15) ** 2) +
          (yy - cy) ** 2 / (2 * (h * 0.18) ** 2))
    ).astype(np.float32)

    activation = cv2.GaussianBlur(activation, (31, 31), 0)
    activation = (activation / (activation.max() + 1e-8) * 255).astype(np.uint8)

    heatmap_coloured = cv2.applyColorMap(activation, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_coloured, cv2.COLOR_BGR2RGB)

    base_rgb = base_img.copy() if base_img.ndim == 3 else cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(base_rgb, 1.0, heatmap_rgb, intensity, 0)
    return overlay


def compute_difference_map(
    hist_img: np.ndarray,
    curr_img: np.ndarray,
    threshold: float = 0.1,
    opacity: float = 0.6,
) -> np.ndarray:
    """Signed difference map overlay.  Red = increase, Blue = decrease."""
    h_gray = cv2.cvtColor(hist_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    diff = c_gray - h_gray
    max_abs = np.abs(diff).max() + 1e-8
    diff_norm = diff / max_abs  # range [-1, 1]

    base = cv2.addWeighted(hist_img, 0.5, curr_img, 0.5, 0).astype(np.float32)
    overlay = np.zeros_like(base)

    # Red → increase
    increase_mask = diff_norm > threshold
    overlay[increase_mask] = [255, 0, 0]

    # Blue → decrease
    decrease_mask = diff_norm < -threshold
    overlay[decrease_mask] = [0, 0, 255]

    result = cv2.addWeighted(base, 1.0 - opacity, overlay, opacity, 0)
    return result.astype(np.uint8)


def plot_probability_over_time(finding: str) -> go.Figure:
    """Return a Plotly line chart of probability over time for *finding*."""
    data = _TEMPORAL_DATA.get(finding, _TEMPORAL_DATA["Pneumonia"])
    dates = data["dates"]
    probs = [p * 100 for p in data["probs"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=probs,
            mode="lines+markers",
            line=dict(color=COLORS["safe"], width=3),
            marker=dict(size=8, color=COLORS["safe"]),
            hovertemplate="%{x}<br>Probability: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12),
        xaxis=dict(showgrid=False, linecolor=COLORS["border"], tickcolor=COLORS["neutral"]),
        yaxis=dict(showgrid=True, gridcolor=COLORS["border"], linecolor=COLORS["border"]),
        hovermode="x unified",
    )
    return fig


def generate_clinical_summary(
    finding: str,
    historical_study: str,
    current_study: str,
    trend: str,
) -> str:
    """Return an AI-style clinical summary string."""
    return _CLINICAL_SUMMARIES.get(trend, _CLINICAL_SUMMARIES["No significant change"])


# ==============================================================================
# Styling helpers
# ==============================================================================

def _card_start(title: str, title_color: str = COLORS["text"]) -> None:
    st.markdown(
        f"""
        <div style="
            background:{COLORS['card_bg']};
            border:1px solid {COLORS['border']};
            border-radius:10px;
            padding:16px 18px 14px 18px;
            margin-bottom:14px;
        ">
        <div style="font-weight:700;font-size:15px;color:{title_color};margin-bottom:10px;">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _badge(is_positive: bool) -> str:
    bg = COLORS["danger"] if is_positive else COLORS["safe"]
    label = "Positive" if is_positive else "Negative"
    return (
        f'<span style="background:{bg};color:#fff;border-radius:4px;'
        f'padding:2px 8px;font-size:12px;font-weight:700;">{label}</span>'
    )


def _render_prediction_table(df: pd.DataFrame) -> None:
    """Render a styled HTML prediction table."""
    rows_html = ""
    for _, row in df.iterrows():
        is_pos = row["Binary Label"] == "Positive"
        badge = _badge(is_pos)
        rows_html += (
            f"<tr>"
            f"<td style='padding:5px 8px;color:{COLORS['text']};font-size:13px;'>{row['Finding']}</td>"
            f"<td style='padding:5px 8px;color:{COLORS['text']};font-size:13px;'>{row['Probability']:.2f}</td>"
            f"<td style='padding:5px 8px;color:{COLORS['neutral']};font-size:13px;'>{row['Threshold']:.2f}</td>"
            f"<td style='padding:5px 8px;'>{badge}</td>"
            f"</tr>"
        )
    hdr = (
        f"padding:5px 8px;font-size:12px;font-weight:600;"
        f"color:{COLORS['neutral']};text-transform:uppercase;letter-spacing:0.5px;"
    )
    html = (
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<thead><tr>"
        f"<th style='{hdr}'>Finding</th>"
        f"<th style='{hdr}'>Prob.</th>"
        f"<th style='{hdr}'>Threshold</th>"
        f"<th style='{hdr}'>Label</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================================
# Main render function (called by the app router)
# ==============================================================================

def render() -> None:
    # ------------------------------------------------------------------
    # PAGE HEADER
    # ------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:28px;">
            <h1 style="
                color:{COLORS['highlight']};
                font-size:2.2rem;
                font-weight:800;
                margin-bottom:4px;
                text-shadow:0 0 10px rgba(0,204,150,0.25);
            ">Historical Comparison</h1>
            <p style="color:{COLORS['neutral']};font-size:15px;margin:0;">
                Compare current exam with previous exams for the same patient.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # SECTION 1 — Study Selection Bar
    # ------------------------------------------------------------------
    sel_c1, sel_c2, sel_c3, sel_c4, sel_c5 = st.columns([2, 2, 2, 2, 2])

    with sel_c1:
        patient_id = st.text_input("Patient ID", value="2023-10-26, PA")
    with sel_c2:
        historical_study = st.selectbox("Historical Study", HISTORICAL_STUDIES, index=0)
    with sel_c3:
        current_study = st.selectbox("Current Study", CURRENT_STUDIES, index=0)
    with sel_c4:
        finding_to_compare = st.selectbox("Finding to Compare", FINDINGS, index=0)
    with sel_c5:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        compute_btn = st.button(
            "Compute Difference Map", type="primary", width='stretch'
        )

    # Persist compute state across reruns
    if compute_btn:
        st.session_state["hist_computed"] = True
        st.session_state["hist_historical_study"] = historical_study
        st.session_state["hist_current_study"] = current_study
        st.session_state["hist_finding"] = finding_to_compare

    active_hist    = st.session_state.get("hist_historical_study", historical_study)
    active_curr    = st.session_state.get("hist_current_study", current_study)
    active_finding = st.session_state.get("hist_finding", finding_to_compare)

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:12px 0 16px 0;'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<span style='color:{COLORS['neutral']};font-size:13px;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:1px;'>Selection</span>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # SECTION 2 — Exam Comparison (3 columns)
    # ------------------------------------------------------------------
    col_hist, col_curr, col_diff = st.columns(3)

    # Pre-generate images and tables
    base_img     = load_exam_image()
    hist_heatmap = generate_gradcam_heatmap(base_img, active_finding, active_hist,  intensity=0.45)
    curr_heatmap = generate_gradcam_heatmap(base_img, active_finding, active_curr,  intensity=0.65)
    hist_df      = generate_mock_predictions(active_hist)
    curr_df      = generate_mock_predictions(active_curr)

    # ── Historical Exam ──────────────────────────────────────────────
    with col_hist:
        _card_start("Historical Exam")
        st.image(hist_heatmap, width='stretch')
        _render_prediction_table(hist_df)
        _card_end()

    # ── Current Exam ─────────────────────────────────────────────────
    with col_curr:
        _card_start("Current Exam")
        st.image(curr_heatmap, width='stretch')
        _render_prediction_table(curr_df)
        _card_end()

    # ── Difference Map ───────────────────────────────────────────────
    with col_diff:
        _card_start("Difference Map")

        _sl_left, _sl_right = st.columns(2)
        with _sl_left:
            diff_thresh = st.slider(
                "Difference Threshold", min_value=0.0, max_value=1.0,
                value=0.1, step=0.05, key="diff_threshold",
            )
        with _sl_right:
            diff_opacity = st.slider(
                "Difference Opacity", min_value=0.0, max_value=1.0,
                value=0.1, step=0.05, key="diff_opacity",
            )

        diff_map = compute_difference_map(
            hist_heatmap, curr_heatmap,
            threshold=diff_thresh,
            opacity=diff_opacity,
        )
        st.image(diff_map, width='stretch')

        # Colour legend
        st.markdown(
            f"""
            <div style="display:flex;gap:18px;margin-top:8px;font-size:13px;
                        color:{COLORS['neutral']};">
                <span><span style="color:{COLORS['danger']};font-weight:700;">■ Red</span>
                      &nbsp;= Increase</span>
                <span><span style="color:#4a90d9;font-weight:700;">■ Blue</span>
                      &nbsp;= Decrease</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _card_end()

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:6px 0 18px 0;'></div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # SECTION 3 & 4 — Temporal Trend | Clinical Summary (side-by-side)
    # ------------------------------------------------------------------
    col_trend, col_summary = st.columns(2)

    # ── Probability Over Time ────────────────────────────────────────
    with col_trend:
        _card_start("Probability over time")
        fig = plot_probability_over_time(active_finding)
        st.plotly_chart(fig, width='stretch', key="temporal_chart")
        _card_end()

    # ── Overall Change & Clinical Summary ────────────────────────────
    with col_summary:
        _card_start("Overall Change")

        trend_choice = st.radio(
            "Assessment",
            options=["Improved", "Worsened", "No significant change"],
            index=1,  # default: Worsened
            label_visibility="collapsed",
            key="overall_trend",
        )

        summary_text = generate_clinical_summary(
            finding=active_finding,
            historical_study=active_hist,
            current_study=active_curr,
            trend=trend_choice,
        )

        accent = {
            "Improved":             COLORS["safe"],
            "Worsened":             COLORS["danger"],
            "No significant change": COLORS["neutral"],
        }.get(trend_choice, COLORS["neutral"])

        st.markdown(
            f"""
            <div style="
                background:rgba(0,0,0,0.2);
                border-left:3px solid {accent};
                border-radius:4px;
                padding:10px 14px;
                margin-top:8px;
                font-size:14px;
                color:{COLORS['text']};
                line-height:1.65;
            ">{summary_text}</div>
            """,
            unsafe_allow_html=True,
        )
        _card_end()

    # ------------------------------------------------------------------
    # Footer disclaimer
    # ------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="
            margin-top:20px;
            padding:10px 16px;
            border-top:1px solid {COLORS['border']};
            font-size:11px;
            color:{COLORS['neutral']};
            text-align:center;
        ">
            These AI-generated findings are intended to assist — not replace — clinical judgement.
            Always correlate with clinical presentation and consult a qualified radiologist.
        </div>
        """,
        unsafe_allow_html=True,
    )

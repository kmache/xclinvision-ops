"""Page 2: Historical Comparison — Temporal patient comparison with heatmaps and trends.

Compare a patient's current and historical analyses retrieved from the
backend API.  Visualises XAI overlays, a pixel-level difference map,
probability-over-time trend chart, and AI-generated clinical summary.
"""

import base64
import io

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from config import CLASS_NAMES
from styles import COLORS

FINDINGS = [c for c in CLASS_NAMES if c != "No finding"]
THRESHOLD = 0.50

# Available models — same as page_inference
_MODELS = {
    "ViT-Base (best — macro F1 0.63, AUC 0.93)": "vit_base",
    "ConvNeXt-Small (macro F1 0.56, AUC 0.90)": "convnext_small",
    "EfficientNet-B0 (macro F1 0.54, AUC 0.90)": "efficientnet_b0",
    "DenseNet-121 (macro F1 0.53, AUC 0.89)": "densenet",
}

_XAI_METHODS = ["gradcam++", "scorecam", "attention_rollout"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _b64_to_image(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded PNG into an RGB numpy array."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def _resize_match(img: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Resize *img* to match *target_hw* (h, w)."""
    h, w = target_hw
    if img.shape[:2] != (h, w):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def _predictions_from_analysis(analysis: dict) -> pd.DataFrame:
    """Build a DataFrame from a stored analysis entry."""
    top_k = analysis.get("top_k_predictions", [])
    rows = []
    for entry in top_k:
        name = entry.get("class_name", "")
        if name == "No finding":
            continue
        prob = entry.get("probability", 0.0)
        rows.append({
            "Finding": name,
            "Probability": round(prob, 2),
            "Threshold": THRESHOLD,
            "Binary Label": "Positive" if prob >= THRESHOLD else "Negative",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Finding", "Probability", "Threshold", "Binary Label"],
    )


def compute_difference_map(
    hist_img: np.ndarray, curr_img: np.ndarray,
    threshold: float = 0.1, opacity: float = 0.6,
) -> np.ndarray:
    """Signed difference map overlay.  Red = increase, Blue = decrease."""
    curr_img = _resize_match(curr_img, hist_img.shape[:2])
    h_gray = cv2.cvtColor(hist_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = c_gray - h_gray
    max_abs = np.abs(diff).max() + 1e-8
    diff_norm = diff / max_abs

    base = cv2.addWeighted(hist_img, 0.5, curr_img, 0.5, 0).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[diff_norm > threshold] = [255, 0, 0]
    overlay[diff_norm < -threshold] = [0, 0, 255]
    return cv2.addWeighted(base, 1.0 - opacity, overlay, opacity, 0).astype(np.uint8)


def _build_temporal_chart(history: list, finding: str) -> go.Figure:
    """Line chart of probability over time for *finding*."""
    dates, probs = [], []
    for entry in sorted(history, key=lambda x: x.get("timestamp", "")):
        ts = entry.get("timestamp", "")[:10]
        for pk in entry.get("top_k_predictions", []):
            if pk.get("class_name") == finding:
                dates.append(ts)
                probs.append(pk["probability"] * 100)
                break

    fig = go.Figure()
    if dates:
        fig.add_trace(go.Scatter(
            x=dates, y=probs, mode="lines+markers",
            line=dict(color=COLORS["safe"], width=3),
            marker=dict(size=8, color=COLORS["safe"]),
            hovertemplate="%{x}<br>Probability: %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12),
        xaxis=dict(showgrid=False, linecolor=COLORS["border"]),
        yaxis=dict(showgrid=True, gridcolor=COLORS["border"], linecolor=COLORS["border"],
                   title="Probability (%)"),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

def _card_start(title: str, title_color: str = COLORS["text"]) -> None:
    st.markdown(
        f'<div style="background:{COLORS["card_bg"]};border:1px solid {COLORS["border"]};'
        f'border-radius:10px;padding:16px 18px 14px 18px;margin-bottom:14px;">'
        f'<div style="font-weight:700;font-size:15px;color:{title_color};margin-bottom:10px;">'
        f'{title}</div>',
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
    if df.empty:
        st.info("No predictions available.")
        return
    rows_html = ""
    for _, row in df.iterrows():
        badge = _badge(row["Binary Label"] == "Positive")
        rows_html += (
            f"<tr><td style='padding:5px 8px;color:{COLORS['text']};font-size:13px;'>{row['Finding']}</td>"
            f"<td style='padding:5px 8px;color:{COLORS['text']};font-size:13px;'>{row['Probability']:.2f}</td>"
            f"<td style='padding:5px 8px;color:{COLORS['neutral']};font-size:13px;'>{row['Threshold']:.2f}</td>"
            f"<td style='padding:5px 8px;'>{badge}</td></tr>"
        )
    hdr = (
        f"padding:5px 8px;font-size:12px;font-weight:600;"
        f"color:{COLORS['neutral']};text-transform:uppercase;letter-spacing:0.5px;"
    )
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;'><thead><tr>"
        f"<th style='{hdr}'>Finding</th><th style='{hdr}'>Prob.</th>"
        f"<th style='{hdr}'>Threshold</th><th style='{hdr}'>Label</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    client = st.session_state.get("api_client")

    st.markdown(
        f'<div style="text-align:center;margin-bottom:28px;">'
        f'<h1 style="color:{COLORS["highlight"]};font-size:2.2rem;font-weight:800;'
        f'margin-bottom:4px;text-shadow:0 0 10px rgba(0,204,150,0.25);">'
        f'Historical Comparison</h1>'
        f'<p style="color:{COLORS["neutral"]};font-size:15px;margin:0;">'
        f'Compare current exam with previous exams for the same patient.</p></div>',
        unsafe_allow_html=True,
    )

    tab_history, tab_manual = st.tabs(["📊 Patient History", "📤 Manual Comparison"])

    with tab_history:
        _render_patient_history(client)

    with tab_manual:
        _render_manual_comparison(client)


# ---------------------------------------------------------------------------
# Tab 1: Patient History (existing functionality)
# ---------------------------------------------------------------------------

def _render_patient_history(client) -> None:

    # --- Study selection bar --------------------------------------------------
    sel1, sel2 = st.columns([2, 1])
    with sel1:
        patient_id = st.text_input("Patient ID", value=st.session_state.get("current_patient_id", ""),
                                    placeholder="Enter a patient ID")
    with sel2:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        fetch_btn = st.button("📥 Fetch History", type="primary", width='stretch', disabled=(not patient_id))

    # Fetch history from backend
    if fetch_btn and patient_id and client:
        with st.spinner("Fetching patient history..."):
            history = client.get_patient_history(patient_id)
            if history:
                st.session_state["patient_history"] = history
            else:
                st.warning("No history found for this patient. Run analyses on the Inference page first.")
                st.session_state["patient_history"] = []

    history = st.session_state.get("patient_history", [])

    if not history:
        st.info(
            "Enter a patient ID and click **Fetch History** to load previous analyses. "
            "The patient must have been analyzed on the **Inference** page at least once."
        )

        # Footer
        st.markdown(
            f'<div style="margin-top:20px;padding:10px 16px;border-top:1px solid {COLORS["border"]};'
            f'font-size:11px;color:{COLORS["neutral"]};text-align:center;">'
            f'These AI-generated findings are intended to assist — not replace — clinical judgement.</div>',
            unsafe_allow_html=True,
        )
        return

    # --- Build selection options from real history ----------------------------
    study_labels = [
        f"{h.get('timestamp', '?')[:10]} — {h.get('prediction', '?')} ({h.get('confidence', 0):.0%})"
        for h in history
    ]

    sel_c1, sel_c2, sel_c3 = st.columns([3, 3, 3])
    with sel_c1:
        hist_idx = st.selectbox("Historical Study", range(len(study_labels)),
                                format_func=lambda i: study_labels[i],
                                index=min(1, len(study_labels) - 1) if len(study_labels) > 1 else 0)
    with sel_c2:
        curr_idx = st.selectbox("Current Study", range(len(study_labels)),
                                format_func=lambda i: study_labels[i], index=0)
    with sel_c3:
        finding_to_compare = st.selectbox("Finding to Compare", FINDINGS, index=0)

    hist_entry = history[hist_idx]
    curr_entry = history[curr_idx]

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:12px 0 16px 0;'></div>",
        unsafe_allow_html=True,
    )

    # --- Exam comparison (3 columns) ------------------------------------------
    col_hist, col_curr, col_diff = st.columns(3)

    hist_df = _predictions_from_analysis(hist_entry)
    curr_df = _predictions_from_analysis(curr_entry)

    # Images from heatmap thumbnails
    hist_has_img = hist_entry.get("thumbnail") or hist_entry.get("heatmap_overlay")
    curr_has_img = curr_entry.get("thumbnail") or curr_entry.get("heatmap_overlay")

    with col_hist:
        _card_start("Historical Exam")
        if hist_has_img:
            img_b64 = hist_entry.get("heatmap_overlay") or hist_entry.get("thumbnail")
            hist_img = _b64_to_image(img_b64)
            st.image(hist_img, width='stretch')
        else:
            st.info("No heatmap available")
            hist_img = None
        _render_prediction_table(hist_df)
        _card_end()

    with col_curr:
        _card_start("Current Exam")
        if curr_has_img:
            img_b64 = curr_entry.get("heatmap_overlay") or curr_entry.get("thumbnail")
            curr_img = _b64_to_image(img_b64)
            st.image(curr_img, width='stretch')
        else:
            st.info("No heatmap available")
            curr_img = None
        _render_prediction_table(curr_df)
        _card_end()

    with col_diff:
        _card_start("Difference Map")
        if hist_img is not None and curr_img is not None:
            sl1, sl2 = st.columns(2)
            with sl1:
                diff_thresh = st.slider("Diff Threshold", 0.0, 1.0, 0.1, 0.05, key="diff_threshold")
            with sl2:
                diff_opacity = st.slider("Diff Opacity", 0.0, 1.0, 0.6, 0.05, key="diff_opacity")
            diff_map = compute_difference_map(hist_img, curr_img, diff_thresh, diff_opacity)
            st.image(diff_map, width='stretch')
            st.markdown(
                f'<div style="display:flex;gap:18px;margin-top:8px;font-size:13px;color:{COLORS["neutral"]};">'
                f'<span><span style="color:{COLORS["danger"]};font-weight:700;">■ Red</span> = Increase</span>'
                f'<span><span style="color:#4a90d9;font-weight:700;">■ Blue</span> = Decrease</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Need two analyses with heatmaps to compute difference.")
        _card_end()

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:6px 0 18px 0;'></div>",
        unsafe_allow_html=True,
    )

    # --- Temporal trend & summary ---------------------------------------------
    col_trend, col_summary = st.columns(2)

    with col_trend:
        _card_start("Probability over time")
        fig = _build_temporal_chart(history, finding_to_compare)
        st.plotly_chart(fig, width='stretch', key="temporal_chart")
        _card_end()

    with col_summary:
        _card_start("Overall Change")

        # Compute real trend from probabilities
        hist_prob = 0.0
        curr_prob = 0.0
        for pk in hist_entry.get("top_k_predictions", []):
            if pk.get("class_name") == finding_to_compare:
                hist_prob = pk["probability"]
        for pk in curr_entry.get("top_k_predictions", []):
            if pk.get("class_name") == finding_to_compare:
                curr_prob = pk["probability"]

        diff_val = curr_prob - hist_prob
        if diff_val > 0.05:
            auto_trend = "Worsened"
        elif diff_val < -0.05:
            auto_trend = "Improved"
        else:
            auto_trend = "No significant change"

        trend_choice = st.radio(
            "Assessment", ["Improved", "Worsened", "No significant change"],
            index=["Improved", "Worsened", "No significant change"].index(auto_trend),
            label_visibility="collapsed", key="overall_trend",
        )

        # Build dynamic summary from real data
        summary_text = (
            f"Comparing {finding_to_compare}: historical probability {hist_prob:.1%} → "
            f"current {curr_prob:.1%} (Δ {diff_val:+.1%}). "
        )
        curr_summary = curr_entry.get("llm_summary", "")
        if curr_summary:
            summary_text += curr_summary
        else:
            summary_text += "Clinical correlation recommended."

        accent = {
            "Improved": COLORS["safe"],
            "Worsened": COLORS["danger"],
            "No significant change": COLORS["neutral"],
        }.get(trend_choice, COLORS["neutral"])

        st.markdown(
            f'<div style="background:rgba(0,0,0,0.2);border-left:3px solid {accent};'
            f'border-radius:4px;padding:10px 14px;margin-top:8px;font-size:14px;'
            f'color:{COLORS["text"]};line-height:1.65;">{summary_text}</div>',
            unsafe_allow_html=True,
        )
        _card_end()

    # Footer
    st.markdown(
        f'<div style="margin-top:20px;padding:10px 16px;border-top:1px solid {COLORS["border"]};'
        f'font-size:11px;color:{COLORS["neutral"]};text-align:center;">'
        f'These AI-generated findings are intended to assist — not replace — clinical judgement. '
        f'Always correlate with clinical presentation and consult a qualified radiologist.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 2: Manual Comparison — upload two images directly
# ---------------------------------------------------------------------------

def _render_manual_comparison(client) -> None:
    st.markdown(
        f'<p style="color:{COLORS["neutral"]};font-size:14px;margin-bottom:16px;">'
        f'Upload two chest X-ray images to run a side-by-side AI comparison with '
        f'XAI heatmaps and a pixel-level difference map.</p>',
        unsafe_allow_html=True,
    )

    # --- Upload row -----------------------------------------------------------
    up_a, up_b = st.columns(2)
    with up_a:
        _card_start("Image A")
        file_a = st.file_uploader(
            "Upload Image A", type=["jpg", "jpeg", "png", "bmp", "tiff", "dcm", "dicom"],
            key="manual_cmp_file_a", label_visibility="collapsed",
        )
        if file_a:
            st.image(file_a, caption=file_a.name, width='stretch')
        _card_end()

    with up_b:
        _card_start("Image B")
        file_b = st.file_uploader(
            "Upload Image B", type=["jpg", "jpeg", "png", "bmp", "tiff", "dcm", "dicom"],
            key="manual_cmp_file_b", label_visibility="collapsed",
        )
        if file_b:
            st.image(file_b, caption=file_b.name, width='stretch')
        _card_end()

    # --- Options row ----------------------------------------------------------
    opt1, opt2, opt3 = st.columns([3, 3, 2])
    with opt1:
        model_label = st.selectbox("Model", list(_MODELS.keys()), index=0,
                                   key="manual_cmp_model")
        model_name = _MODELS[model_label]
    with opt2:
        xai_method = st.selectbox("XAI Method", _XAI_METHODS, index=0,
                                  key="manual_cmp_xai")
    with opt3:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        both_uploaded = file_a is not None and file_b is not None
        compare_btn = st.button(
            "🔬 Compare", type="primary", width='stretch',
            disabled=(not both_uploaded or client is None),
            key="manual_cmp_btn",
        )

    if not both_uploaded:
        st.info("Upload both images above, then click **Compare** to run the analysis.")

    # --- Run comparison -------------------------------------------------------
    if compare_btn and both_uploaded and client:
        file_a.seek(0)
        file_b.seek(0)
        bytes_a = file_a.read()
        bytes_b = file_b.read()

        with st.spinner("Running AI analysis on both images — this may take a moment..."):
            result = client.compare_images(
                file_a_bytes=bytes_a, filename_a=file_a.name,
                file_b_bytes=bytes_b, filename_b=file_b.name,
                model_name=model_name, xai_method=xai_method,
            )

        if not result:
            st.error("Comparison failed. Check backend connectivity and try again.")
            return

        st.session_state["manual_cmp_result"] = result

    # --- Display results ------------------------------------------------------
    result = st.session_state.get("manual_cmp_result")
    if result is None:
        return

    analysis_a = result["image_a"]
    analysis_b = result["image_b"]

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:16px 0;'></div>",
        unsafe_allow_html=True,
    )

    df_a = _predictions_from_analysis(analysis_a)
    df_b = _predictions_from_analysis(analysis_b)

    img_a_b64 = analysis_a.get("heatmap_overlay") or analysis_a.get("thumbnail")
    img_b_b64 = analysis_b.get("heatmap_overlay") or analysis_b.get("thumbnail")
    img_a = _b64_to_image(img_a_b64) if img_a_b64 else None
    img_b = _b64_to_image(img_b_b64) if img_b_b64 else None

    col_a, col_b, col_d = st.columns(3)

    with col_a:
        _card_start("Image A — Analysis")
        if img_a is not None:
            st.image(img_a, width='stretch')
        else:
            st.info("No heatmap available")
        st.markdown(
            f'<div style="font-size:13px;color:{COLORS["text"]};margin:6px 0 4px 0;">'
            f'<b>Prediction:</b> {analysis_a["prediction"]} '
            f'({analysis_a["confidence"]:.1%})</div>',
            unsafe_allow_html=True,
        )
        _render_prediction_table(df_a)
        _card_end()

    with col_b:
        _card_start("Image B — Analysis")
        if img_b is not None:
            st.image(img_b, width='stretch')
        else:
            st.info("No heatmap available")
        st.markdown(
            f'<div style="font-size:13px;color:{COLORS["text"]};margin:6px 0 4px 0;">'
            f'<b>Prediction:</b> {analysis_b["prediction"]} '
            f'({analysis_b["confidence"]:.1%})</div>',
            unsafe_allow_html=True,
        )
        _render_prediction_table(df_b)
        _card_end()

    with col_d:
        _card_start("Difference Map")
        if img_a is not None and img_b is not None:
            sl1, sl2 = st.columns(2)
            with sl1:
                dt = st.slider("Diff Threshold", 0.0, 1.0, 0.1, 0.05,
                               key="manual_diff_threshold")
            with sl2:
                do = st.slider("Diff Opacity", 0.0, 1.0, 0.6, 0.05,
                               key="manual_diff_opacity")
            diff_map = compute_difference_map(img_a, img_b, dt, do)
            st.image(diff_map, width='stretch')
            st.markdown(
                f'<div style="display:flex;gap:18px;margin-top:8px;font-size:13px;'
                f'color:{COLORS["neutral"]};">'
                f'<span><span style="color:{COLORS["danger"]};font-weight:700;">■ Red</span>'
                f' = Increase</span>'
                f'<span><span style="color:#4a90d9;font-weight:700;">■ Blue</span>'
                f' = Decrease</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Heatmaps unavailable for difference map.")
        _card_end()

    # --- Timing info ----------------------------------------------------------
    st.markdown(
        f'<div style="margin-top:8px;font-size:12px;color:{COLORS["neutral"]};text-align:center;">'
        f'Image A: {analysis_a.get("inference_time_ms", 0):.0f} ms &nbsp;|&nbsp; '
        f'Image B: {analysis_b.get("inference_time_ms", 0):.0f} ms</div>',
        unsafe_allow_html=True,
    )

    # Footer
    st.markdown(
        f'<div style="margin-top:20px;padding:10px 16px;border-top:1px solid {COLORS["border"]};'
        f'font-size:11px;color:{COLORS["neutral"]};text-align:center;">'
        f'These AI-generated findings are intended to assist — not replace — clinical judgement. '
        f'Always correlate with clinical presentation and consult a qualified radiologist.</div>',
        unsafe_allow_html=True,
    )

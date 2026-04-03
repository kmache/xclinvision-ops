"""Page 4: Audit & Transparency — Model card, drift monitoring, and audit log.

Fetches real metrics from evaluation reports and the backend API
instead of using hardcoded placeholder data.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from config import CLASS_NAMES
from styles import COLORS

# --- Paths to real evaluation artefacts ------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_EVAL_DIR = _PROJECT_ROOT / "outputs"
_DOCS_DIR = _PROJECT_ROOT / "docs"

# Model name → eval directory stem
_EVAL_MODELS = {
    "vit_base":         "evaluation_384_vit_base",
    "convnext_small":   "evaluation_384_convnext_small",
    "efficientnet_b0":  "evaluation_384_efficientnet_b0",
    "densenet":         "evaluation_384_densenet",
}


# ---------------------------------------------------------------------------
# Load real evaluation data
# ---------------------------------------------------------------------------

def _load_evaluation_report(model_key: str) -> dict:
    """Load the evaluation report JSON for a given model."""
    stem = _EVAL_MODELS.get(model_key, "")
    report_dir = _EVAL_DIR / stem
    if not report_dir.is_dir():
        return {}
    candidates = sorted(report_dir.glob("*_test_evaluation_report.json"))
    if not candidates:
        return {}
    try:
        with open(candidates[0]) as f:
            return json.load(f)
    except Exception:
        return {}


def _load_all_model_reports() -> dict:
    """Return {model_key: report_dict} for all trained models."""
    return {k: _load_evaluation_report(k) for k in _EVAL_MODELS}


def _build_performance_table() -> pd.DataFrame:
    """Build a performance DataFrame from real evaluation reports."""
    reports = _load_all_model_reports()
    rows = []
    for model_key, rpt in reports.items():
        if not rpt:
            continue
        row = {"Model": model_key}
        row["Macro F1"] = f"{rpt.get('macro_f1', 0):.4f}"
        row["Macro AUC"] = f"{rpt.get('macro_auc', 0):.4f}"
        row["Accuracy"] = f"{rpt.get('subset_accuracy', 0):.4f}"
        cal = rpt.get("calibration", {})
        row["ECE"] = f"{cal.get('expected_calibration_error', 0):.4f}"
        rows.append(row)
    if not rows:
        return pd.DataFrame({"Info": ["No evaluation reports found. Run scripts/evaluate.py"]})
    return pd.DataFrame(rows)


def _build_per_class_table(model_key: str = "vit_base") -> pd.DataFrame:
    """Build per-class metrics table for the selected model."""
    rpt = _load_evaluation_report(model_key)
    if not rpt:
        return pd.DataFrame()
    pathology_names = [c for c in CLASS_NAMES if c != "No finding"]
    rows = []
    for name in pathology_names:
        rows.append({
            "Finding": name,
            "AUC": f"{rpt.get(f'{name}_auc', 0):.3f}",
            "Sensitivity": f"{rpt.get(f'{name}_sensitivity', 0):.3f}",
            "Specificity": f"{rpt.get(f'{name}_specificity', 0):.3f}",
            "F1 Score": f"{rpt.get(f'{name}_f1', 0):.3f}",
            "PPV": f"{rpt.get(f'{name}_ppv', 0):.3f}",
        })
    return pd.DataFrame(rows)


def _load_model_card_text() -> str:
    """Load model_card.md if it exists."""
    path = _DOCS_DIR / "model_card.md"
    if path.exists():
        return path.read_text()[:3000]
    return ""


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    st.markdown(
        f'<div style="text-align:center;margin-bottom:28px;">'
        f'<h1 style="color:{COLORS["highlight"]};font-size:2.2rem;font-weight:800;'
        f'margin-bottom:4px;text-shadow:0 0 10px rgba(0,204,150,0.25);">'
        f'Audit & Transparency</h1>'
        f'<p style="color:{COLORS["neutral"]};font-size:15px;margin:0;">'
        f'Model performance, drift monitoring, and audit logs.</p></div>',
        unsafe_allow_html=True,
    )

    tab_model, tab_drift, tab_log = st.tabs(
        ["Model Card", "Drift & Monitoring", "Audit Log"],
    )

    with tab_model:
        _render_model_card_tab()
    with tab_drift:
        _render_drift_tab()
    with tab_log:
        _render_audit_log_tab()


# ---------------------------------------------------------------------------
# Tab: Model Card
# ---------------------------------------------------------------------------

def _render_model_card_tab():
    left, right = st.columns([1, 1])

    with left:
        with st.container(border=True):
            st.markdown("#### Model Overview")
            st.markdown("**Name:** XClinVision ChestX-ray")
            st.markdown("**Organisation:** NounCode AI")
            st.markdown("**Architecture:** Multi-model (ViT-Base, ConvNeXt-Small, EfficientNet-B0, DenseNet-121)")
            st.markdown("**Training data:** VinBigData Chest X-ray — 14,304 frontal radiographs, 5-class multilabel")
            st.markdown("**Input size:** 384 × 384 px")
            st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")
            st.markdown("**Intended use:** Research-only decision support for thoracic disease detection")
            st.markdown("**Certification:** Research Use Only — Not FDA cleared")

        # Model card markdown
        md_text = _load_model_card_text()
        if md_text:
            with st.expander("📄 Full Model Card (docs/model_card.md)"):
                st.markdown(md_text)

    with right:
        with st.container(border=True):
            st.markdown("#### Cross-Model Performance")
            perf_df = _build_performance_table()
            st.dataframe(perf_df, hide_index=True, width='stretch')
            st.caption("Metrics from test set evaluation (n=2,151)")

        model_for_detail = st.selectbox(
            "Per-class detail for:", list(_EVAL_MODELS.keys()), index=0,
            key="audit_detail_model",
        )
        with st.container(border=True):
            st.markdown(f"#### Per-Class Metrics — {model_for_detail}")
            cls_df = _build_per_class_table(model_for_detail)
            if not cls_df.empty:
                st.dataframe(cls_df, hide_index=True, width='stretch')
            else:
                st.info("No evaluation report found for this model.")

        with st.container(border=True):
            st.markdown("#### Limitations & Risks")
            st.markdown(
                "- Trained on **frontal views only** (PA / AP)\n"
                "- **Not validated** for pediatric populations (<18 years)\n"
                "- Performance may degrade on images from non-standard equipment\n"
                "- Reduced sensitivity for subtle findings <5 mm\n"
                "- Research use only — always correlate with clinical judgement"
            )


# ---------------------------------------------------------------------------
# Tab: Drift & Monitoring
# ---------------------------------------------------------------------------

def _render_drift_tab():
    client = st.session_state.get("api_client")

    left_col, right_col = st.columns([1, 1], gap="medium")

    with left_col:
        with st.container(border=True):
            st.markdown("#### Live System Status")
            # Fetch drift metrics from backend
            drift_data = None
            if client:
                drift_data = client.get_drift_metrics(days=30)

            if drift_data:
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Analyses", drift_data.get("total_predictions", 0))
                m2.metric("Avg Confidence", f"{drift_data.get('avg_confidence', 0):.1%}")
                m3.metric("Drift Score", f"{drift_data.get('drift_score', 0):.4f}")

                if drift_data.get("drift_detected"):
                    st.error("⚠️ **Drift detected** — model performance may have degraded.")
                else:
                    st.success("✅ No significant drift detected.")

                # Prediction distribution
                pred_dist = drift_data.get("prediction_distribution", {})
                if pred_dist:
                    st.markdown("**Prediction Distribution**")
                    dist_df = pd.DataFrame(
                        {"Class": list(pred_dist.keys()), "Count": list(pred_dist.values())},
                    )
                    st.bar_chart(dist_df.set_index("Class"))
            else:
                st.info("Start the backend and run analyses to see live metrics.")

        # Filters
        f1, f2 = st.columns(2)
        with f1:
            st.date_input(
                "Date range",
                value=(date.today() - timedelta(days=30), date.today()),
                key="audit_date_range",
            )
        with f2:
            st.selectbox("Model", list(_EVAL_MODELS.keys()), key="audit_model_filter")

    with right_col:
        with st.container(border=True):
            st.markdown("#### Performance")
            perf_df = _build_performance_table()
            st.dataframe(perf_df, hide_index=True, width='stretch')
            st.caption("Test set evaluation (n=2,151)")

        with st.container(border=True):
            st.markdown("#### Feedback Statistics")
            fb_data = None
            if client:
                fb_data = client.get_feedback_stats()
            if fb_data and fb_data.get("total", 0) > 0:
                fb1, fb2, fb3 = st.columns(3)
                fb1.metric("Total Feedback", fb_data["total"])
                by_type = fb_data.get("by_type", {})
                fb2.metric("Correct", by_type.get("correct", 0))
                fb3.metric("Incorrect", by_type.get("incorrect", 0))
                correction_rate = fb_data.get("correction_rate", 0)
                st.progress(min(correction_rate / 100, 1.0),
                            text=f"Correction rate: {correction_rate:.1f}%")
            else:
                st.info("No feedback submitted yet. Use the Inference page to provide feedback.")

        with st.container(border=True):
            st.markdown("#### Limitations & Risks")
            st.markdown(
                "- Trained on frontal views only (PA / AP)\n"
                "- Not validated for pediatric populations\n"
                "- Performance degrades on non-standard equipment\n"
                "- Research use only"
            )

    # Bottom: audit tools
    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Audit Tools")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.text_input("Patient ID", placeholder="P-XXXX-XXXXXX", key="aud_patient_id")
        with b2:
            st.text_input("Study ID", placeholder="STU-XXXXXXXX", key="aud_study_id")
        with b3:
            st.selectbox("Model", list(_EVAL_MODELS.keys()), key="aud_model_ver")
        with b4:
            st.slider("Threshold", 0.0, 1.0, 0.5, 0.01, key="aud_threshold")

        b5, b6 = st.columns(2)
        with b5:
            st.text_input("User", placeholder="radiologist_001", key="aud_user")
        with b6:
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            if st.button("Export Performance CSV", width='stretch', key="aud_export_csv"):
                csv_bytes = _build_performance_table().to_csv(index=False).encode()
                st.download_button(
                    "Download CSV", data=csv_bytes,
                    file_name="xclinvision_performance.csv", mime="text/csv",
                )


# ---------------------------------------------------------------------------
# Tab: Audit Log
# ---------------------------------------------------------------------------

def _render_audit_log_tab():
    client = st.session_state.get("api_client")

    st.markdown("#### Audit Log")

    fc1, fc2 = st.columns([1, 1])
    with fc1:
        st.date_input(
            "Date range",
            value=(date.today() - timedelta(days=7), date.today()),
            key="log_date_range",
        )
    with fc2:
        st.selectbox(
            "Type filter",
            ["All", "correct", "incorrect", "uncertain"],
            key="log_type_filter",
        )

    # Fetch real feedback from backend
    fb_data = None
    if client:
        fb_data = client.get_feedback_stats()

    if fb_data and fb_data.get("recent"):
        recent = fb_data["recent"]
        log_rows = []
        for entry in recent:
            log_rows.append({
                "Timestamp": entry.get("timestamp", "—")[:19],
                "Analysis ID": entry.get("analysis_id", "—"),
                "Type": entry.get("feedback_type", "—"),
                "User": entry.get("user_id", "anonymous"),
                "Notes": entry.get("notes", "")[:80] if entry.get("notes") else "—",
            })
        log_df = pd.DataFrame(log_rows)
        st.dataframe(log_df, hide_index=True, width='stretch')

        csv_data = log_df.to_csv(index=False).encode()
        st.download_button(
            "Download audit log (CSV)", data=csv_data,
            file_name="audit_log.csv", mime="text/csv",
        )
    else:
        st.info(
            "No audit log entries yet. Feedback submitted on the Inference page "
            "will appear here."
        )

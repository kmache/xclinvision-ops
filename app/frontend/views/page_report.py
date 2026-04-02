"""Page: Report Generation — Structured clinical AI reporting interface.

Review AI output, compose structured findings, and export finalized
radiology reports to PDF, JSON, or HIS.  Populates from real analysis
results when available via ``st.session_state["current_analysis"]``.
"""

import json
from datetime import datetime

import streamlit as st

from config import CLASS_NAMES
from styles import COLORS

# ==============================================================================
# Helpers
# ==============================================================================

_SEVERITY_COLORS = {
    "Moderate": COLORS["warning"],
    "Severe":   COLORS["danger"],
    "Mild":     COLORS["safe"],
}

_LOCATIONS = [
    "Right lower lobe", "Left lower lobe",
    "Right upper lobe", "Left upper lobe",
    "Right middle lobe", "Bilateral",
]

_PATHOLOGY_CLASSES = [c for c in CLASS_NAMES if c != "No finding"]


def _severity_from_prob(prob: float) -> str:
    if prob >= 0.8:
        return "Severe"
    if prob >= 0.5:
        return "Moderate"
    return "Mild"


def _severity_badge(label: str) -> str:
    color = _SEVERITY_COLORS.get(label, COLORS["neutral"])
    return (
        f'<span style="background:{color};color:#fff;border-radius:4px;'
        f'padding:2px 9px;font-size:12px;font-weight:700;">{label}</span>'
    )


def _score_badge(score: float) -> str:
    return (
        f'<span style="background:{COLORS["card_bg"]};color:{COLORS["warning"]};'
        f'border:1px solid {COLORS["warning"]};border-radius:4px;'
        f'padding:2px 8px;font-size:12px;font-weight:700;">{score:.2f}</span>'
    )


# ==============================================================================
# Build dynamic text from current analysis
# ==============================================================================

def _get_analysis() -> dict:
    """Return the most recent analysis result from session state, or {}."""
    return st.session_state.get("current_analysis") or {}


def _build_indication() -> str:
    """Build indication text from the current analysis."""
    analysis = _get_analysis()
    if not analysis:
        return ""
    patient_id = analysis.get("patient_id", "Unknown")
    return (
        f"AI-assisted chest X-ray analysis for patient {patient_id}. "
        "Automated pathology screening with multilabel classification."
    )


def _build_impression() -> str:
    """Build impression text from real predictions."""
    analysis = _get_analysis()
    predictions = analysis.get("top_k_predictions", [])
    if not predictions:
        return ""

    positive = [p for p in predictions if p.get("probability", 0) >= 0.5 and p.get("class_name") != "No finding"]
    if not positive:
        return (
            "No significant pathological findings detected above the decision threshold. "
            "All classes below 0.5 confidence. Clinical correlation recommended."
        )

    parts = []
    for p in sorted(positive, key=lambda x: x.get("probability", 0), reverse=True):
        name = p.get("class_name", "Unknown")
        prob = p.get("probability", 0)
        parts.append(f"{name} (confidence {prob:.1%})")

    findings_str = "; ".join(parts)
    return (
        f"AI model detected: {findings_str}. "
        "Findings should be correlated with clinical presentation. "
        "Follow-up imaging may be warranted for confirmed findings."
    )


def _build_predictions_list() -> list[dict]:
    """Return list of {class_name, probability, severity} from current analysis."""
    analysis = _get_analysis()
    predictions = analysis.get("top_k_predictions", [])
    if predictions:
        return [
            {
                "class_name": p.get("class_name", ""),
                "probability": p.get("probability", 0),
                "severity": _severity_from_prob(p.get("probability", 0)),
            }
            for p in predictions
            if p.get("class_name", "") != "No finding"
        ]
    # Fallback: show all pathology classes with empty values
    return [
        {"class_name": c, "probability": 0.0, "severity": "Mild"}
        for c in _PATHOLOGY_CLASSES
    ]


# ==============================================================================
# Section renderers
# ==============================================================================

def render_header() -> tuple[str, str]:
    """Render the page title and info bar. Returns (patient_id, study)."""
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:28px;">
            <h1 style="
                color:{COLORS['highlight']};
                font-size:2.2rem;
                font-weight:800;
                margin-bottom:4px;
                text-shadow:0 0 10px rgba(0,204,150,0.25);
            ">Report Generation</h1>
            <p style="color:{COLORS['neutral']};font-size:15px;margin:0;">
                Review AI output, finalize findings, and export patient report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    analysis = _get_analysis()
    default_pid = analysis.get("patient_id", "")
    default_study = f"{datetime.now().strftime('%Y-%m-%d')}, PA" if analysis else ""
    model_name = analysis.get("model_version", "—")

    bar1, bar2, bar3, bar4 = st.columns([2, 2, 1.5, 1.5])
    with bar1:
        patient_id = st.text_input("Patient ID", value=default_pid, placeholder="P-XXXX-XXXXXX")
    with bar2:
        study = st.text_input("Current study", value=default_study, placeholder="YYYY-MM-DD, PA")
    with bar3:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='padding:8px 0;font-size:13px;color:{COLORS['neutral']};'>"
            f"Model: <b style='color:{COLORS['text']};'>{model_name}</b></div>",
            unsafe_allow_html=True,
        )
    with bar4:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        if st.button("← Back to inference", type="primary", width='stretch'):
            st.session_state["_page_idx"] = 0
            st.session_state["_nav_btn_triggered"] = True
            st.query_params["page"] = "0"
            st.rerun()

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:10px 0 22px 0;'></div>",
        unsafe_allow_html=True,
    )
    return patient_id, study


def render_structured_editor() -> str:
    """Render the left-column structured editor from real predictions. Returns indication text."""
    st.markdown(
        f"<span style='font-size:16px;font-weight:700;color:{COLORS['text']};'>"
        f"Structured editor</span>",
        unsafe_allow_html=True,
    )

    preds = _build_predictions_list()

    with st.container(border=True):
        st.markdown(
            f"<span style='font-size:14px;font-weight:600;color:{COLORS['neutral']};'>"
            f"Structured findings</span>",
            unsafe_allow_html=True,
        )

        # Table header
        hdr_c0, hdr_c1, hdr_c2, hdr_c3 = st.columns([0.6, 2.5, 1.8, 2.2])
        for col, txt in [(hdr_c0, ""), (hdr_c1, "Finding"), (hdr_c2, "Confidence"), (hdr_c3, "Severity")]:
            col.markdown(
                f"<span style='font-size:11px;font-weight:600;color:{COLORS['neutral']};'>{txt}</span>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='border-top:1px solid {COLORS['border']};margin:4px 0 6px 0;'></div>",
            unsafe_allow_html=True,
        )

        # Dynamic rows from predictions
        for i, pred in enumerate(preds):
            name = pred["class_name"]
            prob = pred["probability"]
            sev = pred["severity"]
            is_positive = prob >= 0.5

            rc0, rc1, rc2, rc3 = st.columns([0.6, 2.5, 1.2, 2.6])
            with rc0:
                st.checkbox(name, value=is_positive, key=f"chk_{i}", label_visibility="collapsed")
            with rc1:
                color = COLORS["text"] if is_positive else COLORS["neutral"]
                st.markdown(
                    f"<div style='padding-top:6px;color:{color};font-size:13px;'>{name}</div>",
                    unsafe_allow_html=True,
                )
            with rc2:
                st.markdown(
                    f"<div style='padding-top:4px;'>{_score_badge(prob)}</div>",
                    unsafe_allow_html=True,
                )
            with rc3:
                st.markdown(
                    f"<div style='padding-top:4px;'>{_severity_badge(sev)}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:14px 0 10px 0;'></div>",
        unsafe_allow_html=True,
    )

    # Indication
    st.markdown(
        f"<span style='font-size:14px;font-weight:600;color:{COLORS['neutral']};'>"
        f"Indication</span>",
        unsafe_allow_html=True,
    )
    indication = st.text_area(
        "Indication text",
        value=st.session_state.get("report_indication", _build_indication()),
        height=110,
        key="report_indication",
        label_visibility="collapsed",
    )
    return indication


def render_impression_section() -> tuple[str, str]:
    """Render the right-column impression / conclusion panel. Returns (impression, comments)."""
    st.markdown(
        f"<span style='font-size:16px;font-weight:700;color:{COLORS['text']};'>"
        f"Impression / Conclusion</span>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            f"<span style='font-size:13px;font-weight:600;color:{COLORS['neutral']};'>"
            f"Current exam</span>",
            unsafe_allow_html=True,
        )
        impression = st.text_area(
            "Impression text",
            value=st.session_state.get("report_impression", _build_impression()),
            height=260,
            key="report_impression",
            label_visibility="collapsed",
        )

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:14px 0 10px 0;'></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<span style='font-size:14px;font-weight:600;color:{COLORS['neutral']};'>"
        f"Additional comments</span>",
        unsafe_allow_html=True,
    )
    comments = st.text_area(
        "Additional comments",
        value=st.session_state.get("report_comments", ""),
        height=80,
        placeholder="Additional comments",
        key="report_comments",
        label_visibility="collapsed",
    )
    return impression, comments


def render_report_preview(
    patient_id: str, study: str, indication: str, impression: str, comments: str,
) -> None:
    """Render the full-width report preview and action buttons."""
    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:18px 0 16px 0;'></div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            f"<span style='font-size:15px;font-weight:700;color:{COLORS['text']};'>"
            f"Report preview</span>",
            unsafe_allow_html=True,
        )

        # Build dynamic preview from real fields
        preview_parts = []
        if indication.strip():
            preview_parts.append(f"**Indication:** {indication.strip()}")
        if impression.strip():
            preview_parts.append(f"**Impression:** {impression.strip()}")
        if comments.strip():
            preview_parts.append(f"**Additional notes:** {comments.strip()}")

        analysis = _get_analysis()
        if analysis.get("analysis_id"):
            preview_parts.append(f"**Analysis ID:** {analysis['analysis_id']}")

        preview_text = "<br><br>".join(preview_parts) if preview_parts else (
            "<em>Run an analysis on the Inference page to populate this report.</em>"
        )

        st.markdown(
            f"""
            <div style="
                background:{COLORS['background']};
                border-radius:6px;
                padding:16px 18px;
                margin-top:10px;
                font-size:13.5px;
                color:{COLORS['text']};
                line-height:1.7;
            ">{preview_text}</div>
            """,
            unsafe_allow_html=True,
        )

    # Action buttons
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _gap, btn1, btn2, btn3, _gap2 = st.columns([1, 2, 2.5, 2.5, 1])

    with btn1:
        if st.button("Save draft", width='stretch'):
            st.session_state["saved_draft"] = {
                "patient_id": patient_id,
                "study": study,
                "indication": indication,
                "impression": impression,
                "comments": comments,
                "saved_at": datetime.now().isoformat(),
            }
            st.toast("Draft saved.", icon="💾")
    with btn2:
        # Try to generate report via backend
        analysis = _get_analysis()
        analysis_id = analysis.get("analysis_id")
        if st.button("Generate AI Report", width='stretch',
                      disabled=not analysis_id):
            client = st.session_state.get("api_client")
            if client and analysis_id:
                with st.spinner("Generating report…"):
                    result = client.generate_report(analysis_ids=[analysis_id])
                if result and result.get("content"):
                    content = result["content"]
                    st.session_state["_pending_report_impression"] = content.get(
                        "impressions", impression
                    )
                    if content.get("findings"):
                        st.session_state["_pending_report_indication"] = content["findings"]
                    st.toast("Report generated from AI analysis.", icon="✅")
                    st.rerun()
                else:
                    st.toast("Report generation failed — using manual text.", icon="⚠️")
            else:
                st.toast("No API connection. Using manual text.", icon="⚠️")
    with btn3:
        export_col1, export_col2, export_col3 = st.columns(3)
        with export_col1:
            if st.button("Export HTML", width='stretch'):
                client = st.session_state.get("api_client")
                if client and analysis_id:
                    with st.spinner("Generating HTML report…"):
                        result = client.export_report_html(analysis_id=analysis_id)
                    if result and result.get("html"):
                        st.download_button(
                            "⬇ Download HTML",
                            data=result["html"],
                            file_name=f"clinical_report_{analysis_id[:8]}.html",
                            mime="text/html",
                            key="download_html_report",
                        )
                        st.toast("HTML report generated.", icon="✅")
                    else:
                        st.toast("HTML export failed.", icon="⚠️")
                else:
                    st.toast("No API connection.", icon="⚠️")
        with export_col2:
            if st.button("Export PDF", width='stretch'):
                st.toast("PDF export triggered. Use your browser's print dialog.", icon="📄")
        with export_col3:
            payload = json.dumps(
                {
                    "patient_id": patient_id,
                    "study": study,
                    "analysis_id": analysis.get("analysis_id", ""),
                    "model": analysis.get("model_version", ""),
                    "indication": indication,
                    "impression": impression,
                    "comments": comments,
                    "predictions": analysis.get("top_k_predictions", []),
                    "generated_at": datetime.now().isoformat(),
                },
                indent=2,
            )
            st.download_button(
                "Export JSON",
                data=payload,
                file_name="report_export.json",
                mime="application/json",
                width='stretch',
            )

    # Footer disclaimer
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


# ==============================================================================
# Main entry point
# ==============================================================================

def render() -> None:
    # ── Apply any pending AI-generated content BEFORE widgets render ──
    # Streamlit does not allow setting a widget-bound key after the widget
    # has been instantiated.  We stage values in _pending_* keys and apply
    # them here, before the widgets are created on this rerun.
    for field in ("report_impression", "report_indication"):
        pending_key = f"_pending_{field}"
        if pending_key in st.session_state:
            st.session_state[field] = st.session_state.pop(pending_key)

    patient_id, study = render_header()
    st.session_state["report_patient_id"] = patient_id
    st.session_state["report_study"] = study

    # Main two-column layout
    col_left, col_right = st.columns([45, 55])

    with col_left:
        indication = render_structured_editor()
    with col_right:
        impression, comments = render_impression_section()

    render_report_preview(patient_id, study, indication, impression, comments)

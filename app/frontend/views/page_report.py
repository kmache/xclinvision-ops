"""Page: Report Generation — Structured clinical AI reporting interface.

Review AI output, compose structured findings, and export finalized
radiology reports to PDF, JSON, or HIS.
"""

import json
from datetime import datetime

import streamlit as st

from styles import COLORS

# ==============================================================================
# Mock / default content
# ==============================================================================

_DEFAULT_INDICATION = (
    "Patient presents with persistent cough and fever for 5 days. "
    "Clinical suspicion for lower respiratory tract infection. "
    "AI-assisted imaging requested for diagnostic support."
)

_DEFAULT_IMPRESSION = (
    "Findings suggest progressive consolidation in the right middle lobe consistent "
    "with community-acquired pneumonia. Subtle blunting of the right costophrenic angle "
    "may indicate a small pleural effusion. The cardiac silhouette appears at the upper "
    "limit of normal size. No pneumothorax identified. Osseous structures appear intact. "
    "Soft tissues are unremarkable. AI model confidence exceeds threshold for all flagged "
    "findings. Clinical correlation with laboratory results and patient history is advised. "
    "Follow-up imaging in 4–6 weeks recommended to confirm resolution."
)

_DEFAULT_PREVIEW = (
    "This report was generated with AI assistance and reviewed by the attending radiologist. "
    "Findings include consolidation in the right middle lobe, minor pleural effusion, and "
    "mild cardiomegaly. Antibiotic therapy is suggested pending culture results. Follow-up "
    "imaging in 4–6 weeks is recommended to confirm resolution of the identified pathology. "
    "The patient's overall cardiopulmonary status warrants close monitoring."
)

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

# ==============================================================================
# Section renderers
# ==============================================================================

def render_header() -> tuple[str, str, str]:
    """Render the page title, subtitle, and info bar. Returns (patient_id, study, version)."""
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
                Review AI output, finalize previous exams, and patient report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Info bar
    bar1, bar2, bar3, bar4 = st.columns([2, 2, 1.5, 1.5])
    with bar1:
        patient_id = st.text_input("Patient ID", value="P-1224-333845", label_visibility="visible")
    with bar2:
        study = st.text_input("Current study", value="2023-15-26, PA", label_visibility="visible")
    with bar3:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='padding:8px 0;font-size:13px;color:{COLORS['neutral']};'>"
            f"Finding version: <b style='color:{COLORS['text']};'>v0.33</b></div>",
            unsafe_allow_html=True,
        )
    with bar4:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        if st.button("View prediction →", type="primary", use_container_width=True):
            st.session_state["_page_idx"] = 0
            st.session_state["_nav_btn_triggered"] = True
            st.query_params["page"] = "0"
            st.rerun()

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:10px 0 22px 0;'></div>",
        unsafe_allow_html=True,
    )
    return patient_id, study, "v0.33"


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


def render_structured_editor() -> str:
    """Render the left-column structured editor. Returns indication text."""
    st.markdown(
        f"<span style='font-size:16px;font-weight:700;color:{COLORS['text']};'>"
        f"Structured editor</span>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            f"<span style='font-size:14px;font-weight:600;color:{COLORS['neutral']};'>"
            f"Structured findings</span>",
            unsafe_allow_html=True,
        )

        # Table header
        hdr_c0, hdr_c1, hdr_c2, hdr_c3 = st.columns([0.6, 2.5, 1.8, 2.2])
        for col, txt in [(hdr_c0, ""), (hdr_c1, "Structured findings"), (hdr_c2, "Severity"), (hdr_c3, "")]:
            col.markdown(
                f"<span style='font-size:11px;font-weight:600;color:{COLORS['neutral']};'>{txt}</span>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<div style='border-top:1px solid {COLORS['border']};margin:4px 0 6px 0;'></div>",
            unsafe_allow_html=True,
        )

        # ── Row 1: Pneumonia with numeric score ─────────────────────
        r1c1, r1c2, r1c3, r1c4 = st.columns([0.6, 2.5, 1.2, 2.6])
        with r1c1:
            pneu_checked = st.checkbox("Pneumonia", value=True, key="chk_pneu", label_visibility="collapsed")
        with r1c2:
            st.markdown(
                f"<div style='padding-top:6px;color:{COLORS['text']};font-size:13px;'>Pneumonia</div>",
                unsafe_allow_html=True,
            )
        with r1c3:
            st.markdown(
                f"<div style='padding-top:4px;'>{_score_badge(8.85)}</div>",
                unsafe_allow_html=True,
            )
        with r1c4:
            st.markdown(
                f"<div style='padding-top:4px;'>{_severity_badge('Moderate')}</div>",
                unsafe_allow_html=True,
            )

        # ── Row 2: Effusion — text severity ─────────────────────────
        r2c1, r2c2, r2c3, r2c4 = st.columns([0.6, 2.5, 1.2, 2.6])
        with r2c1:
            st.checkbox("Effusion", value=True, key="chk_eff1", label_visibility="collapsed")
        with r2c2:
            st.markdown(
                f"<div style='padding-top:6px;color:{COLORS['text']};font-size:13px;'>Effusion</div>",
                unsafe_allow_html=True,
            )
        with r2c3:
            st.markdown(
                f"<div style='padding-top:6px;color:{COLORS['neutral']};font-size:13px;'>Moderate</div>",
                unsafe_allow_html=True,
            )
        with r2c4:
            st.markdown("", unsafe_allow_html=True)

        # ── Row 3: Effusion — location dropdown ─────────────────────
        r3c1, r3c2, r3c3 = st.columns([0.6, 2.5, 4.0])
        with r3c1:
            st.checkbox("Effusion location", value=True, key="chk_eff2", label_visibility="collapsed")
        with r3c2:
            st.markdown(
                f"<div style='padding-top:6px;color:{COLORS['text']};font-size:13px;'>Effusion</div>",
                unsafe_allow_html=True,
            )
        with r3c3:
            st.selectbox(
                "Location", _LOCATIONS, index=0,
                key="loc_eff2", label_visibility="collapsed",
            )

        # ── Row 4: Cardiomegaly — unchecked ─────────────────────────
        r4c1, r4c2, r4c3 = st.columns([0.6, 2.5, 4.0])
        with r4c1:
            st.checkbox("Cardiomegaly", value=False, key="chk_card1", label_visibility="collapsed")
        with r4c2:
            st.markdown(
                f"<div style='padding-top:6px;color:{COLORS['neutral']};font-size:13px;'>Cardiomegaly</div>",
                unsafe_allow_html=True,
            )
        with r4c3:
            st.markdown(
                f"<div style='padding-top:6px;color:{COLORS['neutral']};font-size:13px;'>"
                f"Right lower lobe</div>",
                unsafe_allow_html=True,
            )

        # ── Row 5: Cardiomegaly — numeric 0 ─────────────────────────
        r5c1, r5c2, r5c3, r5c4 = st.columns([0.6, 2.5, 1.2, 2.6])
        with r5c1:
            st.checkbox("Cardiomegaly severity", value=False, key="chk_card2", label_visibility="collapsed")
        with r5c2:
            st.markdown(
                f"<div style='padding-top:6px;color:{COLORS['neutral']};font-size:13px;'>Cardiomegaly</div>",
                unsafe_allow_html=True,
            )
        with r5c3:
            st.selectbox(
                "Severity", ["0", "1", "2", "3"], index=0,
                key="sev_card2", label_visibility="collapsed",
            )

    st.markdown(
        f"<div style='border-top:1px solid {COLORS['border']};margin:14px 0 10px 0;'></div>",
        unsafe_allow_html=True,
    )

    # ── Indication ───────────────────────────────────────────────────
    st.markdown(
        f"<span style='font-size:14px;font-weight:600;color:{COLORS['neutral']};'>"
        f"Indication</span>",
        unsafe_allow_html=True,
    )
    indication = st.text_area(
        "Indication text",
        value=st.session_state.get("report_indication", _DEFAULT_INDICATION),
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
            value=st.session_state.get("report_impression", _DEFAULT_IMPRESSION),
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


def render_report_preview(indication: str, impression: str, comments: str) -> None:
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

        # Assemble combined preview text
        preview_parts = [_DEFAULT_PREVIEW]
        if comments.strip():
            preview_parts.append(f"Additional notes: {comments.strip()}")
        preview_text = "  ".join(preview_parts)

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

    # ── Action buttons ───────────────────────────────────────────────
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    _gap, btn1, btn2, btn3, _gap2 = st.columns([1, 2, 2.5, 2.5, 1])

    with btn1:
        if st.button("Feedback", use_container_width=True):
            st.toast("Feedback submitted.", icon="✅")
    with btn2:
        if st.button("Save draft", use_container_width=True):
            st.session_state["saved_draft"] = {
                "indication": indication,
                "impression": impression,
                "comments": comments,
                "saved_at": datetime.now().isoformat(),
            }
            st.toast("Draft saved.", icon="💾")
    with btn3:
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            if st.button("Finalize and export PDF", use_container_width=True):
                st.toast("PDF export triggered. Use your browser's print dialog.", icon="📄")
        with export_col2:
            payload = json.dumps(
                {
                    "patient_id": st.session_state.get("report_patient_id", "P-1224-333845"),
                    "study": st.session_state.get("report_study", "2023-15-26, PA"),
                    "indication": indication,
                    "impression": impression,
                    "comments": comments,
                    "generated_at": datetime.now().isoformat(),
                },
                indent=2,
            )
            st.download_button(
                "Export JSON / send to HIS",
                data=payload,
                file_name="report_export.json",
                mime="application/json",
                use_container_width=True,
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
# Main entry point (called by the app router)
# ==============================================================================

def render() -> None:
    # Header + info bar
    patient_id, study, version = render_header()
    st.session_state["report_patient_id"] = patient_id
    st.session_state["report_study"] = study

    # Main two-column layout  (45 / 55)
    col_left, col_right = st.columns([45, 55])

    with col_left:
        indication = render_structured_editor()

    with col_right:
        impression, comments = render_impression_section()

    # Full-width preview + actions
    render_report_preview(indication, impression, comments)

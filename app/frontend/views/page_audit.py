"""Page 4: Audit & Transparency — Model card, drift monitoring, and audit log."""

from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as _components

from styles import COLORS
from config import CLASS_NAMES


# ---------------------------------------------------------------------------
# Dummy / static data
# ---------------------------------------------------------------------------

_PERF_DATA = pd.DataFrame(
    {
        "Finding":     CLASS_NAMES,
        "Probability": ["0.92",      "0.59",     "0.89"],
        "Sensitivity": ["0.88",      "—",        "—"],
        "Specificity": ["89",        "76",        "70"],
        "F1 Score":    ["0.81",      "0.75",      "—"],
    }
)

_SITES = ["All sites", "General Hospital A", "City Clinic B", "Regional Center C"]
_USERS = ["All users", "radiologist_001", "radiologist_002", "resident_003"]
_MODEL_VERSIONS = ["v1.2.0 (current)", "v1.1.3", "v1.0.8", "v0.9.5"]


def _make_error_rate_series(n: int = 30) -> pd.DataFrame:
    """Generate a fake increasing error-rate time series."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D")
    base = np.linspace(2, 18, n)
    noise = rng.normal(0, 1.5, n)
    values = np.clip(base + noise, 0, None)
    return pd.DataFrame({"Disagreement / error rate": values}, index=dates)


def _make_drift_series(n: int = 30) -> pd.DataFrame:
    """Generate a fake drift score time series (rising at the end)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D")
    base = np.linspace(0, 40, n)
    noise = rng.normal(0, 3, n)
    values = np.clip(base + noise, 0, None)
    return pd.DataFrame({"Drift score": values}, index=dates)


def _make_audit_log(n: int = 15) -> pd.DataFrame:
    now = datetime.now()
    actions = ["Analysis", "Feedback", "Report Export", "Threshold change", "Login"]
    users = ["radiologist_001", "radiologist_002", "resident_003", "admin"]
    rows = [
        {
            "Timestamp": (now - timedelta(hours=i * 3)).strftime("%Y-%m-%d %H:%M"),
            "User": users[i % len(users)],
            "Action": actions[i % len(actions)],
            "Patient / Case": f"P-{1200 - i:04d}",
            "Model version": "v1.2.0",
            "Status": "✅ OK" if i % 5 != 0 else "⚠️ Review",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render():
    # ── Auto-activate "Drift & Monitoring" tab on first load ─────────────────
    # Uses a session-state guard so the JS only fires once per page visit
    # (i.e. on load / refresh) and not on every widget interaction.
    if "_audit_tab_init" not in st.session_state:
        st.session_state["_audit_tab_init"] = True
        _components.html(
            """
            <script>
            setTimeout(function() {
                var tabs = window.parent.document.querySelectorAll(
                    'button[data-baseweb="tab"]'
                );
                if (tabs && tabs.length > 1) { tabs[1].click(); }
            }, 350);
            </script>
            """,
            height=0,
        )

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:28px;">
            <h1 style="
                color:{COLORS['highlight']};
                font-size:2.2rem;
                font-weight:800;
                margin-bottom:4px;
                text-shadow:0 0 10px rgba(0,204,150,0.25);
            ">Audit & Transparency</h1>
            <p style="color:{COLORS['neutral']};font-size:15px;margin:0;">
                Upload current image, previous exams, and manage assessments.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_model, tab_drift, tab_log = st.tabs(
        ["Model Card", "Drift & Monitoring", "Audit Log"]
    )

    # =========================================================================
    # TAB 1 — Model Card
    # =========================================================================
    with tab_model:
        _render_model_card_tab()

    # =========================================================================
    # TAB 2 — Drift & Monitoring  (active in the screenshot)
    # =========================================================================
    with tab_drift:
        _render_drift_tab()

    # =========================================================================
    # TAB 3 — Audit Log
    # =========================================================================
    with tab_log:
        _render_audit_log_tab()


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _render_model_card_tab():
    """Full model-card information."""
    left, right = st.columns([1, 1])

    with left:
        with st.container(border=True):
            st.markdown("#### Model overview")
            st.markdown("**Name:** CXR-Sense")
            st.markdown("**Version:** AI Health Labs")
            st.markdown("**Model type:** Convolutional Neural Network")
            st.markdown("**Training data summary:** diverse patient population")
            st.markdown("**Intended use:** Frontline pneumonia & effusion detection")
            st.markdown("**Population:** pediatric patients, portable AP exams")

    with right:
        with st.container(border=True):
            st.markdown("#### Performance")
            st.dataframe(_PERF_DATA, hide_index=True, width='stretch')
            st.caption("Datasets: Internal Test (n=10k), External (n=5k)")

        with st.container(border=True):
            st.markdown("#### Limitations & risks")
            st.success("**Supported** ✅")
            st.markdown(
                "- Non-supported groups: Pregnant individuals\n"
                "- Pediatric edge cases\n"
                "- Suboptimal imaging devices"
            )


def _render_drift_tab():
    """Drift & Monitoring — matches the screenshot layout."""
    left_col, right_col = st.columns([1, 1], gap="medium")

    # ── LEFT: Model overview + filters + error-rate chart ─────────────────
    with left_col:
        with st.container(border=True):
            st.markdown("#### Model overview")
            st.markdown("**Name:** CXR-Sense")
            st.markdown("**Version:** AI Health Labs")
            st.markdown("**Model type:** Convolutional Neural Network")
            st.markdown("**Training data summary:** diverse patient population")
            st.markdown("**Intended use:** Frontline pneumonia & effusion detection")
            st.markdown("**Population:** pediatric patients, portable AP exams")

        # Filters
        f1, f2 = st.columns(2)
        with f1:
            st.date_input(
                "Date range",
                value=(date.today() - timedelta(days=30), date.today()),
                key="audit_date_range",
            )
        with f2:
            st.selectbox("Site / Hospital", _SITES, key="audit_site")

        f3, f4 = st.columns(2)
        with f3:
            st.selectbox(
                "Volume view",
                ["Patients over time (cases/day)", "Studies per week"],
                key="audit_volume_view",
            )
        with f4:
            st.selectbox(
                "Exam context",
                ["Current exam", "Previous exams", "All exams"],
                key="audit_exam_ctx",
            )

        # Error-rate chart
        with st.container(border=True):
            st.markdown("**Disagreement / error rate**")
            st.line_chart(_make_error_rate_series(), width='stretch', height=200)

        # Download button
        csv_bytes = _make_error_rate_series().to_csv().encode()
        st.download_button(
            "Download CSV (filtered)",
            data=csv_bytes,
            file_name="error_rate_filtered.csv",
            mime="text/csv",
            width='stretch',
        )

    # ── RIGHT: Performance + Limitations + Drift chart ────────────────────
    with right_col:
        with st.container(border=True):
            st.markdown("#### Performance")
            st.dataframe(_PERF_DATA, hide_index=True, width='stretch')
            st.caption("Datasets: Internal Test (n=10k), External (n=5k)")

        with st.container(border=True):
            st.markdown("#### Limitations & risks")
            st.success("**Supported** ✅")
            st.markdown(
                "- Non-supported groups: Pregnant individuals\n"
                "- Pediatric edge cases\n"
                "- Suboptimal imaging devices"
            )
            st.line_chart(_make_drift_series(), width='stretch', height=180)

    # ── BOTTOM: Audit / Discard tools ─────────────────────────────────────
    st.markdown("---")
    with st.container(border=True):
        st.markdown("#### Discard / Audit tools")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.text_input("Patient ID", placeholder="P-XXXX-XXXXXX", key="aud_patient_id")
        with b2:
            st.text_input("Study ID", placeholder="STU-XXXXXXXX", key="aud_study_id")
        with b3:
            st.selectbox("Model version", _MODEL_VERSIONS, key="aud_model_ver")
        with b4:
            st.slider("Threshold", 0.0, 1.0, 0.5, 0.01, key="aud_threshold")

        b5, b6 = st.columns(2)
        with b5:
            st.selectbox("User", _USERS, key="aud_user")
        with b6:
            st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
            report_col, csv_col = st.columns(2)
            with report_col:
                if st.button("Generate Report", width='stretch', key="aud_gen_report"):
                    st.toast("Report generated ✅")
            with csv_col:
                audit_csv = _make_audit_log().to_csv(index=False).encode()
                st.download_button(
                    "Download CSV (filtered)",
                    data=audit_csv,
                    file_name="audit_log_filtered.csv",
                    mime="text/csv",
                    width='stretch',
                    key="aud_dl_csv",
                )


def _render_audit_log_tab():
    """Audit log with filters and downloadable table."""
    st.markdown("#### Audit Log")

    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        st.date_input(
            "Date range",
            value=(date.today() - timedelta(days=7), date.today()),
            key="log_date_range",
        )
    with fc2:
        st.selectbox("User filter", _USERS, key="log_user_filter")
    with fc3:
        st.selectbox(
            "Action filter",
            ["All actions", "Analysis", "Feedback", "Report Export", "Threshold change"],
            key="log_action_filter",
        )

    log_df = _make_audit_log()
    st.dataframe(log_df, hide_index=True, width='stretch')

    st.download_button(
        "Download full audit log (CSV)",
        data=log_df.to_csv(index=False).encode(),
        file_name="audit_log_full.csv",
        mime="text/csv",
    )

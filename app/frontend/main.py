"""Streamlit frontend for XClinVision Clinician Dashboard v2.

Four-page medical imaging AI dashboard:
  1. Inference & Explanation   – Upload, analyse, XAI, LLM chat, feedback
  2. Historical Comparison     – Temporal patient analysis
  3. Report Generation         – Structured clinical reports
  4. Audit & Transparency      – Governance, drift, bias monitoring
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st

from styles import apply_medical_styles, COLORS
from config import UI
from api_client import XClinVisionClient

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=UI.APP_TITLE,
    page_icon=UI.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": f"# XClinVision v{UI.APP_VERSION}\nAI-powered chest X-ray analysis with explainability.",
    },
)

# Apply custom styles
apply_medical_styles()

# ---------------------------------------------------------------------------
# Page list — single source of truth for routing + nav buttons
# ---------------------------------------------------------------------------
PAGES = [
    "🔍 Inference & Explanation",
    "📊 Historical Comparison",
    "📝 Report Generation",
    "⚖️ Audit & Transparency",
]

# Inject sidebar navigation & global nav-button styles
st.markdown(
    f"""
    <style>
    /* Navigation radio labels — default color, slightly smaller than heading */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {{
        font-size: 0.92rem !important;
        font-weight: 500;
    }}
    /* Nav buttons at page bottom */
    .nav-btn-next > button {{
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
    }}
    .nav-btn-prev > button {{
        background-color: #64748b !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
SESSION_DEFAULTS = {
    "current_analysis": None,
    "current_patient_id": None,
    "current_image_bytes": None,
    "chat_history": [],
    "threshold": 0.5,
    "opacity": 0.6,
    "explain_method": "gradcam++",
    "patient_history": [],
    "comparison_baseline": None,
    "comparison_current": None,
    "report_draft": None,
    "feedback_submitted": False,
    "api_connected": False,
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialise the API client once per session
if "api_client" not in st.session_state:
    st.session_state.api_client = XClinVisionClient()

# ---------------------------------------------------------------------------
# Page index — single source of truth (non-widget key).
# On a fresh browser load, seed from the ?page=N query param so the user
# lands on the same page they were on before refreshing.
# Nav buttons write ONLY to _page_idx, never to nav_radio (widget-bound).
# The nav_radio value is always synced FROM _page_idx BEFORE the widget
# renders, which is the only legal moment to set a widget-bound key.
# ---------------------------------------------------------------------------
if "_page_idx" not in st.session_state:
    try:
        idx = int(st.query_params.get("page", "0"))
        idx = max(0, min(idx, len(PAGES) - 1))
    except (ValueError, KeyError):
        idx = 0
    st.session_state["_page_idx"] = idx

# Sync nav_radio from _page_idx when a nav button triggered the rerun.
# This is the only safe moment to write to a widget-bound key.
# On sidebar radio clicks _nav_btn_triggered is False, so we leave nav_radio
# alone and let Streamlit honor the user's selection.
if st.session_state.get("_nav_btn_triggered", False):
    st.session_state["nav_radio"] = PAGES[st.session_state["_page_idx"]]
    st.session_state["_nav_btn_triggered"] = False
elif "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = PAGES[st.session_state["_page_idx"]]


# ---------------------------------------------------------------------------
# Navigation helper
# ---------------------------------------------------------------------------

def _render_nav_buttons(current_idx: int) -> None:
    """Render Previous (bottom-left) and Next (bottom-right) page buttons."""
    st.markdown("---")
    left, _spacer, right = st.columns([2, 5, 2])
    with left:
        if current_idx > 0:
            label = "← " + PAGES[current_idx - 1].split(" ", 1)[1]
            st.markdown('<div class="nav-btn-prev">', unsafe_allow_html=True)
            if st.button(label, key="nav_btn_prev", use_container_width=True):
                st.session_state["_page_idx"] = current_idx - 1
                st.session_state["_nav_btn_triggered"] = True
                st.query_params["page"] = str(current_idx - 1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    with right:
        if current_idx < len(PAGES) - 1:
            label = "→ " + PAGES[current_idx + 1].split(" ", 1)[1]
            st.markdown('<div class="nav-btn-next">', unsafe_allow_html=True)
            if st.button(label, key="nav_btn_next", use_container_width=True):
                st.session_state["_page_idx"] = current_idx + 1
                st.session_state["_nav_btn_triggered"] = True
                st.query_params["page"] = str(current_idx + 1)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 10px 0 5px 0;">
            <h1 style="color:{COLORS['highlight']}; margin-bottom:0; font-size:2rem; font-weight:800; text-shadow:0 0 12px rgba(0,204,150,0.35);">{UI.SIDEBAR_TITLE}</h1>
            <p style="color:{COLORS['neutral']}; font-size:0.9rem; margin-top:4px;">
                {UI.SIDEBAR_SUBTITLE}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        f"<p style='color:{COLORS['highlight']};font-size:1.05rem;font-weight:700;margin-bottom:4px;'>Navigation</p>",
        unsafe_allow_html=True,
    )
    selected = st.radio(
        "Navigation",
        PAGES,
        key="nav_radio",
        label_visibility="collapsed",
    )
    # Sidebar click: keep _page_idx and query-params in sync.
    new_idx = PAGES.index(selected)
    if new_idx != st.session_state["_page_idx"]:
        st.session_state["_page_idx"] = new_idx
    st.query_params["page"] = str(new_idx)

    st.markdown("---")

    # API health check
    try:
        is_healthy = st.session_state.api_client.check_health()
        if is_healthy:
            st.success("🟢 Backend Online")
            st.session_state.api_connected = True
        else:
            st.warning("🟡 Backend Degraded")
            st.session_state.api_connected = False
    except Exception:
        st.error("🔴 Backend Offline")
        st.session_state.api_connected = False

    st.markdown("---")

    # Disclaimer
    st.warning(UI.DISCLAIMER)

    st.caption(f"XClinVision v{UI.APP_VERSION} — © 2026 NounCode AI")


# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
current_page_idx = PAGES.index(selected)

if selected == PAGES[0]:
    from views.page_inference import render
    render()
elif selected == PAGES[1]:
    from views.page_history import render
    render()
elif selected == PAGES[2]:
    from views.page_report import render
    render()
elif selected == PAGES[3]:
    from views.page_audit import render
    render()

_render_nav_buttons(current_page_idx)

import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# 1. COLOR PALETTE
# ==============================================================================
# Color Palette
COLORS = {
    "background": "#0E1117",      # Main App Background
    "card_bg": "#181b21",         # Card Background
    "text": "#FAFAFA",
    "safe": "#00CC96",            # Green
    "danger": "#EF553B",          # Red
    "warning": "#FFA15A",         # Amber
    "neutral": "#8b92a1",         # Subtext Gray
    "border": "#2b3b4f",          # Card Border
    "highlight": "#00CC96",       # Title Color
    "primary": "#2563eb",         # Dashboard blue
    "primary_light": "#dbeafe",   # Light blue
    "secondary": "#64748b",       # Gray
}

# ==============================================================================
# 2. PAGE SETUP & CSS
# ==============================================================================
def setup_page(title="Sentinel Dashboard", layout="wide"):
    st.set_page_config(
        page_title=title,
        page_icon="🩺",
        layout=layout,
        initial_sidebar_state="expanded" 
    )
    
    st.markdown(f"""
    <style>
        /* Global App Background */
        .stApp {{
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }}

        /* RESET & PADDING */
        .stDeployButton {{ display: none; }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        
        .block-container {{
            padding-top: 3.5rem; 
            padding-bottom: 1rem;
        }}

        /* CENTRALIZED GLOBAL TITLE */
        .global-title {{
            text-align: center;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 5px;
            color: {COLORS['highlight']};
            line-height: 1.2;
            text-shadow: 0px 0px 10px rgba(0, 204, 150, 0.3);
        }}
        
        .global-summary {{
            text-align: center;
            color: {COLORS['neutral']};
            font-size: 1rem;
            margin-bottom: 25px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }}

        /* LEFT ALIGNED PAGE HEADERS */
        .page-header {{
            text-align: left;
            margin-top: 0px; 
            margin-bottom: 20px;
            border-bottom: 1px solid {COLORS['border']};
            padding-bottom: 10px;
        }}
        .page-header h2 {{
            font-size: 24px;
            font-weight: 700;
            color: {COLORS['text']};
            margin: 0;
        }}
        .page-header p {{
            font-size: 14px;
            color: {COLORS['neutral']};
            margin: 0;
        }}
        
        /* KPI CARD STYLING */
        .kpi-card {{
            background-color: {COLORS['card_bg']};
            border: 1px solid {COLORS['border']};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            margin-bottom: 10px;
            height: 100%; 
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 140px; 
            transition: transform 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-3px);
            border-color: {COLORS['highlight']};
        }}
        
        .kpi-title {{
            font-size: 14px;
            font-weight: 600;
            color: {COLORS['neutral']};
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .kpi-value {{
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 5px;
        }}
        
        .kpi-subtext {{
            font-size: 12px;
            color: {COLORS['neutral']};
            font-style: italic;
        }}

        /* STATUS INDICATORS (Useful for Sidebar/System health) */
        .status-indicator {{
            height: 10px;
            width: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }}
        .status-green {{ background-color: {COLORS['safe']}; box-shadow: 0 0 8px {COLORS['safe']}; }}
        .status-orange {{ background-color: {COLORS['warning']}; box-shadow: 0 0 8px {COLORS['warning']}; }}
        .status-red {{ background-color: {COLORS['danger']}; box-shadow: 0 0 8px {COLORS['danger']}; }}

        /* BUTTON STYLING */
        div.stButton > button {{
            width: 100%;
            background-color: {COLORS['card_bg']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 5px;
            height: 45px;
            font-weight: 600;
        }}
        div.stButton > button:hover {{
            border-color: {COLORS['safe']};
            color: {COLORS['safe']};
        }}
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# 3. DASHBOARD V2 STYLING
# ==============================================================================

def apply_medical_styles():
    """Apply XClinVision medical dashboard styling (v2 overlay)."""
    st.markdown(f"""
    <style>
    /* Medical dashboard enhancements */
    .main > div {{
        padding-top: 1rem;
        max-width: 1400px;
    }}

    /* Confidence indicators */
    .confidence-high {{ color: {COLORS['safe']}; font-weight: 700; }}
    .confidence-medium {{ color: {COLORS['warning']}; font-weight: 700; }}
    .confidence-low {{ color: {COLORS['danger']}; font-weight: 700; }}

    /* Chat messages */
    .stChatMessage {{
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
    }}

    /* Alert banners */
    .stAlert {{
        border-radius: 8px;
        border-left-width: 4px;
    }}

    /* Feedback buttons row */
    div[data-testid="stHorizontalBlock"] button {{
        border-radius: 8px;
        transition: all 0.2s;
    }}
    div[data-testid="stHorizontalBlock"] button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}

    /* Dataframes */
    .stDataFrame {{
        border-radius: 8px;
        border: 1px solid {COLORS['border']};
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }}

    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS['background']};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS['secondary']};
        border-radius: 4px;
    }}
    </style>
    """, unsafe_allow_html=True)


def get_confidence_color(confidence: float) -> str:
    """Return CSS class name based on confidence level."""
    if confidence >= 0.9:
        return "confidence-high"
    elif confidence >= 0.7:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_confidence_hex(confidence: float) -> str:
    """Return hex color based on confidence level."""
    if confidence >= 0.9:
        return COLORS["safe"]
    elif confidence >= 0.7:
        return COLORS["warning"]
    else:
        return COLORS["danger"]
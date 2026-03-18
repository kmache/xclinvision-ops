import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import cv2
import datetime
from styles import COLORS
from config import CLASS_NAMES

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def load_xray_image(uploaded_file):
    """Loads uploaded image or generates a mock grayscale X-ray for demo."""
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)
    else:
        # Mock simple grayscale image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img += 100
        # draw a simple shape to represent lungs
        cv2.circle(img, (150, 150), 60, (120, 120, 120), -1)
        cv2.circle(img, (250, 150), 60, (120, 120, 120), -1)
        return img

def generate_mock_predictions():
    """Returns mock prediction data."""
    probs = [0.85, 0.20, 0.40][:len(CLASS_NAMES)]
    # Pad if more classes than mock values
    while len(probs) < len(CLASS_NAMES):
        probs.append(0.10)
    def risk(p):
        if p >= 0.65: return "High"
        if p >= 0.35: return "Medium"
        return "Low"
    data = {
        "Finding": CLASS_NAMES,
        "Probability": probs,
        "Risk Level": [risk(p) for p in probs],
        "Binary label": ["Positive" if p >= 0.50 else "Negative" for p in probs],
    }
    return pd.DataFrame(data)

def render_confidence_gauge(confidence: float):
    """Renders a Plotly gauge for model confidence."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number={"suffix": "%", "font": {"size": 22, "color": "white"}},
        title={"text": "Model Confidence", "font": {"size": 11, "color": "#AAAAAA"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": "#00CC96" if confidence >= 0.65 else ("#F59E0B" if confidence >= 0.35 else "#FF4B4B")},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 35],   "color": "rgba(255,75,75,0.15)"},
                {"range": [35, 65],  "color": "rgba(245,158,11,0.15)"},
                {"range": [65, 100], "color": "rgba(0,204,150,0.15)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": confidence * 100},
        },
    ))
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig

def render_prediction_chart(df):
    """Renders a Plotly horizontal bar chart for predictions to match panel height."""
    colors = ['#FF4B4B' if val == 'Positive' else '#00CC96' for val in df["Binary label"]]
    fig = go.Figure(data=[
        go.Bar(
            x=df['Probability'],
            y=df['Finding'],
            orientation='h',
            marker_color=colors,
            text=df['Probability'],
            textposition='inside',
            insidetextanchor='middle',
        )
    ])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Probability",
        yaxis_title="",
        xaxis_range=[0, 1],
        height=160,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def generate_gradcam_overlay(image, opacity=0.5):
    """Generates a mock Grad-CAM overlay on the image."""
    # Mock heatmap (Gaussian blob on the bottom right lung)
    heatmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    center = (int(image.shape[1]*0.65), int(image.shape[0]*0.65))
    cv2.circle(heatmap, center, int(image.shape[0]*0.25), 1.0, -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    # Convert to RGB colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - opacity, heatmap_colored, opacity, 0)
    return overlay, heatmap_colored

def render_llm_agent():
    """Renders the LLM Clinical Agent panel with structured findings, chat input and quick chips."""

    # Initialize session state
    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []

    MOCK_RESPONSES = {
        "Explain the heatmap": "The highlighted region in the lower right lobe indicates where the model focused its attention most. Red areas signify activation peaks consistent with pneumonia opacity.",
        "Proposed next steps": "Recommend clinical correlation with patient history and follow-up with a confirmatory CT scan to further characterise the right lower lobe finding.",
        "Is this urgent?": "Given the high model confidence (85%) and localised opacity pattern, this finding should be reviewed by a radiologist within 24 hours.",
        "What could this mean?": "The opacity pattern is most consistent with bacterial pneumonia. Pulmonary abscess is a less likely differential. Effusion is not a primary concern at this stage.",
    }

    # Structured findings block
    st.markdown("""
        <div style="background:#1e2130; border-radius:8px; padding:14px 16px; margin-bottom:12px; font-size:13px; color:#e0e0e0; line-height:1.7;">
            <div style="margin-bottom:6px;"><span style="color:#00CC96; font-weight:700;">&#10003; Findings:</span><br>
            &nbsp;&nbsp;Localized opacity in right lower lobe with high significance.</div>
            <div style="margin-bottom:6px;"><span style="color:#00CC96; font-weight:700;">&#10003; Likely diagnosis:</span><br>
            &nbsp;&nbsp;&bull; Pneumonia, Pulmonary abscess (less likely)</div>
            <div style="margin-bottom:6px;"><span style="color:#00CC96; font-weight:700;">&#10003; Supporting evidence:</span><br>
            &nbsp;&nbsp;&bull; Opacity is in right lower lobe, showing high attention from the model.</div>
            <div><span style="color:#00CC96; font-weight:700;">&#10003; Recommended follow-up:</span><br>
            &nbsp;&nbsp;&bull; Recommend confirmatory CT to rule out abscess.</div>
        </div>
    """, unsafe_allow_html=True)

    # Previous chat messages
    for msg in st.session_state.llm_messages:
        role_color = "#4a90d9" if msg["role"] == "user" else "#2a2d3e"
        align = "right" if msg["role"] == "user" else "left"
        st.markdown(f"""
            <div style="text-align:{align}; margin:4px 0;">
              <span style="background:{role_color}; color:white; padding:6px 12px;
                border-radius:12px; font-size:12px; display:inline-block; max-width:90%;">
                {msg['content']}
              </span>
            </div>""", unsafe_allow_html=True)

    # Chat input — submits on Enter or click Send
    # Send button styling and focus ring override
    st.markdown("""
        <style>
        /* Remove red background from send button */
        button[data-testid="stChatInputSubmitButton"] {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        button[data-testid="stChatInputSubmitButton"]:hover {
            background-color: rgba(255,255,255,0.05) !important;
        }
        /* Default: gray send icon */
        button[data-testid="stChatInputSubmitButton"] svg {
            fill: #888888 !important;
            transition: fill 0.2s ease;
        }
        /* Green send icon when textarea has content */
        div:has(textarea[data-testid="stChatInputTextArea"]:not(:placeholder-shown))
            button[data-testid="stChatInputSubmitButton"] svg {
            fill: #00CC96 !important;
        }
        /* Remove red focus ring on chat input */
        div[data-testid="stChatInput"] > div {
            border-color: #555555 !important;
            box-shadow: none !important;
        }
        div[data-testid="stChatInput"] > div:focus-within {
            border-color: #555555 !important;
            box-shadow: 0 0 0 1px #555555 !important;
        }
        div[data-testid="stChatInput"] textarea:focus {
            outline: none !important;
            box-shadow: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    user_input = st.chat_input("Ask a follow-up question...", key="llm_chat_input")

    if user_input and user_input.strip():
        reply = MOCK_RESPONSES.get(user_input.strip(),
            "Based on the heatmaps, the opacity is clearly visible in the lower right lobe. "
            "I recommend correlating with the patient's clinical history.")
        st.session_state.llm_messages.append({"role": "user",    "content": user_input.strip()})
        st.session_state.llm_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # Quick-action chips
    chip_style = (
        "display:inline-block; padding:5px 10px; margin:3px 3px 0 0;"
        "border-radius:14px; font-size:11px; font-weight:600; cursor:pointer;"
        "color:white; border:none;"
    )
    chips = [
        ("Explain the heatmap",  "#3a6bc4"),
        ("Proposed next steps",  "#2e7d5e"),
        ("Is this urgent?",       "#b05020"),
        ("What could this mean?", "#5a3d8a"),
    ]
    chip_html = "".join(
        f"<button style='{chip_style} background:{color};' "
        f"onclick=\"navigator.clipboard.writeText('{label}')\">{label}</button>"
        for label, color in chips
    )
    st.markdown(f"<div style='margin-top:6px;'>{chip_html}</div>", unsafe_allow_html=True)
    st.caption("Click a chip to copy it, then paste into the input above.")

# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------

def render():
    # PAGE HEADER
    st.markdown(
        f"""
        <div style="text-align:center;margin-bottom:28px;">
            <h1 style="
                color:{COLORS['highlight']};
                font-size:2.2rem;
                font-weight:800;
                margin-bottom:4px;
                text-shadow:0 0 10px rgba(0,204,150,0.25);
            ">Chest X-ray Inference & Explanation</h1>
            <p style="color:{COLORS['neutral']};font-size:15px;margin:0;">
                Upload an image, get predictions, inspect heatmaps.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Layout: 2 columns
    col1, col2 = st.columns(2)

    # STATE
    df_preds = generate_mock_predictions()
    
    with col1:
        # SECTION 1: INPUT PANEL
        st.subheader("Input")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload DICOM/PNG/JPG", type=["png", "jpg", "jpeg", "dcm"])
            
            # Patient metadata grid
            c1, c2 = st.columns(2)
            pt_id = c1.text_input("Patient ID", value="P-1224-3345")
            age = c2.number_input("Age", value=55, min_value=0, max_value=120)
            
            c3, c4 = st.columns(2)
            sex = c3.selectbox("Sex", ["M", "F", "Other"])
            study_date = c4.date_input("Study date", value=datetime.date(2023, 10, 26))
            
            c5, c6 = st.columns(2)
            view = c5.selectbox("View", ["PA", "AP", "Lateral"])
            model_ver = c6.slider("Model version", 0.1, 1.0, 0.35, 0.05)
            
            image_array = load_xray_image(uploaded_file)
            
    with col2:
        # SECTION 2: PREDICTION SUMMARY
        st.subheader("Prediction summary")
        with st.container(border=True):
            # Build styled HTML table
            risk_colors = {"Low": "#00CC96", "Medium": "#F59E0B", "High": "#FF4B4B"}
            label_colors = {"Positive": "#FF4B4B", "Negative": "#00CC96"}
            rows_html = ""
            for _, row in df_preds.iterrows():
                rc = risk_colors.get(row["Risk Level"], "#AAAAAA")
                lc = label_colors.get(row["Binary label"], "#AAAAAA")
                rows_html += f"""
                <tr>
                  <td style='padding:6px 10px;'>{row['Finding']}</td>
                  <td style='padding:6px 10px; text-align:center;'>{row['Probability']:.2f}</td>
                  <td style='padding:6px 10px; text-align:center;'>
                    <span style='background:{rc};color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;'>
                      {row['Risk Level']}
                    </span>
                  </td>
                  <td style='padding:6px 10px; text-align:center;'>
                    <span style='background:{lc};color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;'>
                      {row['Binary label']}
                    </span>
                  </td>
                </tr>"""
            st.markdown(f"""
            <table style='width:100%;border-collapse:collapse;color:white;font-size:13px;'>
              <thead>
                <tr style='border-bottom:1px solid #444;color:#AAAAAA;'>
                  <th style='padding:6px 10px;text-align:left;'>Finding</th>
                  <th style='padding:6px 10px;text-align:center;'>Probability</th>
                  <th style='padding:6px 10px;text-align:center;'>Risk Level</th>
                  <th style='padding:6px 10px;text-align:center;'>Binary label</th>
                </tr>
              </thead>
              <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)

            # Bar chart + Gauge side by side
            chart_col, gauge_col = st.columns([3, 2])
            with chart_col:
                fig = render_prediction_chart(df_preds)
                st.plotly_chart(fig, width='stretch')
            with gauge_col:
                top_confidence = df_preds["Probability"].max()
                gauge_fig = render_confidence_gauge(top_confidence)
                st.plotly_chart(gauge_fig, width='stretch')

    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        # SECTION 3: FINDING TO EXPLAIN
        st.subheader("Finding to explain")
        with st.container(border=True):
            overlay_img, heatmap_raw = generate_gradcam_overlay(image_array, opacity=0.6)

            # Always show overlay image at full size
            st.image(overlay_img, width='stretch')

            # Horizontal Grad-CAM colorbar legend
            show_legend = st.checkbox("Legend", value=True)
            if show_legend:
                st.markdown("""
                    <div style="position:relative; width:100%; height:22px; margin-top:4px; border-radius:4px; overflow:hidden;">
                        <div style="
                            width:100%; height:100%;
                            background: linear-gradient(to right,
                                #0000ff, #00ffff, #00ff00, #ffff00, #ff7700, #ff0000);
                            border-radius:4px;">
                        </div>
                        <div style="
                            position:absolute; top:3px; left:6px;
                            color:white; font-size:11px; font-weight:600;
                            text-shadow: 0 0 3px #000;">Low</div>
                        <div style="
                            position:absolute; top:3px; right:6px;
                            color:white; font-size:11px; font-weight:600;
                            text-shadow: 0 0 3px #000;">High</div>
                    </div>
                """, unsafe_allow_html=True)

        # SECTION 5: FEEDBACK PANEL
        st.subheader("Prediction Feedback")
        with st.container(border=True):
            st.markdown("**How accurate is this prediction?**")
            st.markdown("""
                <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                    <button style="
                        flex: 1;
                        padding: 10px 0;
                        background-color: #00CC96;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: 14px;
                        font-weight: 600;
                        cursor: pointer;">
                        Correct
                    </button>
                    <button style="
                        flex: 1;
                        padding: 10px 0;
                        background-color: #FF4B4B;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: 14px;
                        font-weight: 600;
                        cursor: pointer;">
                        False positive
                    </button>
                    <button style="
                        flex: 1;
                        padding: 10px 0;
                        background-color: #F59E0B;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: 14px;
                        font-weight: 600;
                        cursor: pointer;">
                        Missed finding
                    </button>
                </div>
            """, unsafe_allow_html=True)

            st.text_input("Optional comment", placeholder="Enter feedback here...", label_visibility="collapsed")

    with col4:
        # SECTION 4: EXPLANATION CONTROLS
        st.subheader("Explanation Controls")
        with st.container(border=True):
            ec1, ec2 = st.columns(2)
            with ec1:
                finding_to_explain = st.selectbox("Finding to explain", CLASS_NAMES)
                ec1a, ec1b = st.columns(2)
                with ec1a:
                    original_opacity = st.slider("Original", 0.0, 1.0, 0.7, 0.1)
                with ec1b:
                    overlay_opacity = st.slider("Overlay", 0.0, 1.0, 0.7, 0.1)

            with ec2:
                ec2a, ec2b = st.columns(2)
                with ec2a:
                    heatmap_opacity = st.slider("Heatmap opacity", 0.0, 1.0, 0.7, 0.1)
                with ec2b:
                    threshold = st.slider("Threshold", 0.0, 1.0, 0.7, 0.1)
                show_heatmap_only = st.checkbox("Heatmap only", value=False)
            
            # Display thumbnails
            t1, t2 = st.columns([1, 1])
            _, heatmap_only = generate_gradcam_overlay(image_array, opacity=1.0)
            t1.image(image_array, caption="Original", width='stretch')
            t2.image(heatmap_only, caption="Heatmap overlay", width='stretch')

        # SECTION 6: LLM CLINICAL AGENT
        st.subheader("LLM Clinical Agent")
        with st.container(border=True):
            render_llm_agent()

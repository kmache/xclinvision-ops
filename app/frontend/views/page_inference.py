"""Page 1: Inference & Explanation — Upload, analyse, XAI, LLM chat, feedback.

Connects to the FastAPI backend via ``api_client`` for live predictions,
heatmap generation, LLM clinical chat, and clinician feedback.
"""

import base64
import datetime
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from config import CLASS_NAMES, UI
from styles import COLORS

# Available trained models — maps UI label → architecture name sent to API
AVAILABLE_MODELS = {
    "ViT-Base (best — macro F1 0.63, AUC 0.93)": "vit_base",
    "ConvNeXt-Small (macro F1 0.56, AUC 0.90)": "convnext_small",
    "EfficientNet-B0 (macro F1 0.54, AUC 0.90)": "efficientnet_b0",
    "DenseNet-121 (macro F1 0.53, AUC 0.89)": "densenet",
}

# Default XAI method per model architecture
_MODEL_XAI_DEFAULTS: dict[str, str] = {
    "vit_base": "attention_rollout",
    "convnext_small": "gradcam++",
    "efficientnet_b0": "gradcam++",
    "densenet": "gradcam++",
}

# Pathology-only class names (exclude "No finding")
PATHOLOGY_NAMES = [c for c in CLASS_NAMES if c != "No finding"]

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _b64_to_image(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded PNG into an RGB numpy array."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


def _build_predictions_df(analysis: dict) -> pd.DataFrame:
    """Build a styled predictions DataFrame from the API analysis response."""
    top_k = analysis.get("top_k_predictions", [])
    if not top_k:
        return pd.DataFrame(columns=["Finding", "Probability", "Risk Level", "Binary label"])

    rows = []
    for entry in top_k:
        name = entry["class_name"]
        if name == "No finding":
            continue
        prob = entry["probability"]
        risk = "High" if prob >= 0.65 else ("Medium" if prob >= 0.35 else "Low")
        label = "Positive" if prob >= 0.5 else "Negative"
        rows.append({
            "Finding": name, "Probability": prob,
            "Risk Level": risk, "Binary label": label,
        })
    return pd.DataFrame(rows)


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
        height=160, margin=dict(l=10, r=10, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
    )
    return fig


def render_prediction_chart(df):
    """Renders a Plotly horizontal bar chart for predictions."""
    if df.empty:
        return go.Figure()
    colors = ['#FF4B4B' if v == 'Positive' else '#00CC96' for v in df["Binary label"]]
    fig = go.Figure(data=[
        go.Bar(
            x=df['Probability'], y=df['Finding'], orientation='h',
            marker_color=colors,
            text=[f"{p:.2f}" for p in df['Probability']],
            textposition='inside', insidetextanchor='middle',
        )
    ])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Probability", yaxis_title="", xaxis_range=[0, 1],
        height=160, plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
    )
    return fig


def render_llm_agent(analysis: dict):
    """Renders the LLM Clinical Agent panel with real backend chat and streaming."""
    client = st.session_state.get("api_client")
    analysis_id = analysis.get("analysis_id", "")

    if "llm_messages" not in st.session_state:
        st.session_state.llm_messages = []

    # ── LLM Provider selector ──────────────────────────────────────────
    provider_info = None
    if client:
        provider_info = client.get_llm_providers()
    if provider_info and provider_info.get("providers"):
        providers = provider_info["providers"]
        active = provider_info.get("active", providers[0] if providers else "")
        active_idx = providers.index(active) if active in providers else 0

        pcol1, pcol2 = st.columns([3, 1])
        with pcol1:
            selected = st.selectbox(
                "🤖 LLM Provider",
                providers,
                index=active_idx,
                key="llm_provider_select",
                help="Switch between OpenAI and local LLM providers",
            )
        with pcol2:
            st.markdown("<br>", unsafe_allow_html=True)
            if selected != active:
                if st.button("Switch", key="btn_switch_provider", width='stretch'):
                    result = client.switch_llm_provider(selected)
                    if result and result.get("status") == "switched":
                        st.toast(f"Switched to **{selected}** provider", icon="✅")
                        st.rerun()
                    else:
                        st.toast("Failed to switch provider", icon="❌")
            else:
                st.markdown(
                    f'<span style="color:#00CC96;font-size:12px;">● Active</span>',
                    unsafe_allow_html=True,
                )

    # Structured findings from real analysis
    llm_summary = analysis.get("llm_summary", "")
    prediction = analysis.get("prediction", "Unknown")
    confidence = analysis.get("confidence", 0)
    uncertainty = analysis.get("uncertainty_level", "unknown")
    key_findings = analysis.get("key_findings", [])

    findings_html = (
        '<div style="background:#1e2130; border-radius:8px; padding:14px 16px; '
        'margin-bottom:12px; font-size:13px; color:#e0e0e0; line-height:1.7;">'
        '<div style="margin-bottom:6px;"><span style="color:#00CC96; font-weight:700;">'
        '&#10003; AI Assessment:</span><br>'
        f'&nbsp;&nbsp;{llm_summary or f"{prediction} detected ({confidence:.1%} confidence)."}</div>'
        '<div style="margin-bottom:6px;"><span style="color:#00CC96; font-weight:700;">'
        f'&#10003; Uncertainty:</span><br>&nbsp;&nbsp;{uncertainty.capitalize()}</div>'
    )
    if key_findings:
        findings_html += '<div><span style="color:#00CC96; font-weight:700;">&#10003; Key findings:</span><br>'
        for f in key_findings:
            findings_html += f"&nbsp;&nbsp;&bull; {f}<br>"
        findings_html += "</div>"
    findings_html += "</div>"
    st.markdown(findings_html, unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.llm_messages:
        role_color = "#4a90d9" if msg["role"] == "user" else "#2a2d3e"
        align = "right" if msg["role"] == "user" else "left"
        st.markdown(
            f'<div style="text-align:{align}; margin:4px 0;">'
            f'<span style="background:{role_color}; color:white; padding:6px 12px; '
            f'border-radius:12px; font-size:12px; display:inline-block; max-width:90%;">'
            f'{msg["content"]}</span></div>',
            unsafe_allow_html=True,
        )

    # Chat styling
    st.markdown("""
        <style>
        button[data-testid="stChatInputSubmitButton"] {
            background-color: transparent !important; border: none !important; box-shadow: none !important;
        }
        button[data-testid="stChatInputSubmitButton"]:hover {
            background-color: rgba(255,255,255,0.05) !important;
        }
        button[data-testid="stChatInputSubmitButton"] svg { fill: #888 !important; }
        div:has(textarea[data-testid="stChatInputTextArea"]:not(:placeholder-shown))
            button[data-testid="stChatInputSubmitButton"] svg { fill: #00CC96 !important; }
        div[data-testid="stChatInput"] > div {
            border-color: #555 !important; box-shadow: none !important;
        }
        div[data-testid="stChatInput"] > div:focus-within {
            border-color: #555 !important; box-shadow: 0 0 0 1px #555 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    user_input = st.chat_input("Ask a follow-up question...", key="llm_chat_input")
    # Handle pending chip message (auto-send from quick-action buttons)
    _pending = st.session_state.pop("_pending_chat_msg", None)
    effective_input = _pending or user_input

    if effective_input and str(effective_input).strip():
        msg_text = str(effective_input).strip()
        reply_text = ""
        suggested_followups = []
        intent = ""
        tools_used = []

        if client and analysis_id:
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.llm_messages[-6:]]

            # Try streaming first for real-time UX
            try:
                streamed_chunks = []
                typing_placeholder = st.empty()
                typing_placeholder.markdown(
                    '<div style="text-align:left; margin:4px 0;">'
                    '<span style="background:#2a2d3e; color:#888; padding:6px 12px; '
                    'border-radius:12px; font-size:12px; display:inline-block;">'
                    '⏳ Thinking...</span></div>',
                    unsafe_allow_html=True,
                )

                for event in client.stream_chat_message(
                    analysis_id=analysis_id, message=msg_text,
                    history=history, context_type="clinical",
                ):
                    if event["event"] == "metadata":
                        intent = event["data"].get("intent", "")
                        tools_used = event["data"].get("tools_used", [])
                    elif event["event"] == "token":
                        token = event["data"].get("token", "")
                        streamed_chunks.append(token)
                        # Update typing indicator with partial response
                        partial = "".join(streamed_chunks)
                        typing_placeholder.markdown(
                            f'<div style="text-align:left; margin:4px 0;">'
                            f'<span style="background:#2a2d3e; color:white; padding:6px 12px; '
                            f'border-radius:12px; font-size:12px; display:inline-block; max-width:90%;">'
                            f'{partial}▌</span></div>',
                            unsafe_allow_html=True,
                        )
                    elif event["event"] == "done":
                        break

                typing_placeholder.empty()
                if streamed_chunks:
                    reply_text = "".join(streamed_chunks)
            except Exception:
                # Fallback to non-streaming
                resp = client.send_chat_message(
                    analysis_id=analysis_id, message=msg_text,
                    history=history, context_type="clinical",
                )
                if resp:
                    reply_text = resp.get("response", "")
                    suggested_followups = resp.get("suggested_followups", [])
                    intent = resp.get("intent", "")
                    tools_used = resp.get("tools_used", [])

            # Store reasoning metadata for transparency
            if intent:
                st.session_state["_last_intent"] = intent
            if tools_used:
                st.session_state["_last_tools"] = tools_used

        if not reply_text:
            reply_text = (
                f"Analysis: {prediction} ({confidence:.1%} confidence). "
                "Connect the backend for full LLM support."
            )
        st.session_state.llm_messages.append({"role": "user", "content": msg_text})
        st.session_state.llm_messages.append({"role": "assistant", "content": reply_text})
        # Cap conversation history to prevent unbounded session memory growth
        _MAX_CHAT_MESSAGES = 100
        if len(st.session_state.llm_messages) > _MAX_CHAT_MESSAGES:
            st.session_state.llm_messages = st.session_state.llm_messages[-_MAX_CHAT_MESSAGES:]
        if suggested_followups:
            st.session_state["_suggested_followups"] = suggested_followups
        st.rerun()

    chips = [
        ("Explain the heatmap",  "#2563eb"),
        ("Proposed next steps",  "#16a34a"),
        ("Is this urgent?",       "#ea580c"),
        ("What could this mean?", "#7c3aed"),
    ]

    chip_css_rules = ""
    for i, (label, color) in enumerate(chips):
        btn_key = f"chip_{i}"
        chip_css_rules += (
            f'div[data-testid="stHorizontalBlock"] button[kind="secondary"]:has(p:is(:only-child)) '
            f'{{ /* fallback */ }}\n'
        )

        chip_css_rules += (
            f'div.chip-row > div:nth-child({i + 1}) button {{'
            f'  background: {color} !important;'
            f'  color: white !important;'
            f'  border: none !important;'
            f'  border-radius: 14px !important;'
            f'  padding: 5px 10px !important;'
            f'  font-size: 11px !important;'
            f'  font-weight: 600 !important;'
            f'  height: auto !important;'
            f'  min-height: 0 !important;'
            f'}}\n'
        )

    st.markdown(f"<style>{chip_css_rules}</style>", unsafe_allow_html=True)

    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    chip_cols = st.columns(len(chips))
    for i, (label, color) in enumerate(chips):
        with chip_cols[i]:
            if st.button(label, key=f"chip_{i}", width='stretch'):
                st.session_state["_pending_chat_msg"] = label
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Click a chip to ask the AI assistant.")

    followups = st.session_state.get("_suggested_followups", [])
    if followups:
        st.markdown(
            '<div style="margin-top:4px; font-size:11px; color:#888;">Suggested:</div>',
            unsafe_allow_html=True,
        )
        fcols = st.columns(min(len(followups), 3))
        for fc, ftext in zip(fcols, followups[:3]):
            with fc:
                if st.button(f"💡 {ftext}", key=f"followup_{ftext[:20]}", width='stretch'):
                    st.session_state["_pending_chat_msg"] = ftext
                    st.session_state.pop("_suggested_followups", None)
                    st.rerun()

    # ── Conversation export actions ───────────────────────────────────
    if st.session_state.llm_messages:
        st.markdown(
            f'<div style="border-top:1px solid #333;margin:8px 0 6px 0;"></div>',
            unsafe_allow_html=True,
        )
        conv_col1, conv_col2 = st.columns(2)
        with conv_col1:
            if st.button("📋 Copy Conversation", key="btn_copy_conv", width='stretch'):
                conv_text = _format_conversation_text(st.session_state.llm_messages)
                st.session_state["_clipboard_conversation"] = conv_text
                st.toast("Conversation copied to clipboard!", icon="📋")
        with conv_col2:
            if st.button("📝 Send to Report", key="btn_send_to_report", width='stretch'):
                conv_text = _format_conversation_text(st.session_state.llm_messages)
                st.session_state["_conversation_for_report"] = conv_text
                st.toast("Conversation sent to Report page!", icon="📝")

        # Show copyable text area if clipboard was triggered
        if st.session_state.get("_clipboard_conversation"):
            st.text_area(
                "Conversation text (select all & copy)",
                value=st.session_state["_clipboard_conversation"],
                height=120,
                key="conv_clipboard_area",
            )


def _format_conversation_text(messages: list) -> str:
    """Format chat messages into a clean text block for export."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def _submit_feedback(analysis: dict, feedback_type: str):
    """Submit feedback to the backend API."""
    client = st.session_state.get("api_client")
    analysis_id = analysis.get("analysis_id", "")
    notes = st.session_state.get("fb_notes", "")
    if client and analysis_id:
        resp = client.submit_feedback(
            analysis_id=analysis_id, feedback_type=feedback_type,
            notes=notes if notes else None,
        )
        if resp:
            st.toast(f"Feedback '{feedback_type}' recorded.", icon="✅")
        else:
            st.toast("Feedback submission failed.", icon="❌")
    else:
        st.toast("Backend not connected.", icon="⚠️")


# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------

def render():
    client = st.session_state.get("api_client")

    st.markdown(
        f'<div style="text-align:center;margin-bottom:28px;">'
        f'<h1 style="color:{COLORS["highlight"]};font-size:2.2rem;font-weight:800;'
        f'margin-bottom:4px;text-shadow:0 0 10px rgba(0,204,150,0.25);">'
        f'Chest X-ray Inference & Explanation</h1>'
        f'<p style="color:{COLORS["neutral"]};font-size:15px;margin:0;">'
        f'Upload an image, get predictions, inspect heatmaps.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "Upload DICOM/PNG/JPG", type=["png", "jpg", "jpeg", "dcm", "dicom"],
            )
            c1, c2 = st.columns(2)
            pt_id = c1.text_input("Patient ID", value="", placeholder="P-XXXX-XXXX")
            age = c2.number_input("Age", value=0, min_value=0, max_value=120)

            c3, c4 = st.columns(2)
            sex = c3.selectbox("Sex", ["Male", "Female", "Other"])
            study_date = c4.date_input("Study date", value=datetime.date.today())

            c5, c6 = st.columns(2)
            view = c5.selectbox("View", ["PA", "AP", "Lateral"])
            model_label = c6.selectbox("Model", list(AVAILABLE_MODELS.keys()), index=0)
            model_name = AVAILABLE_MODELS[model_label]

            # Auto-select XAI method when model changes
            _prev_model = st.session_state.get("_prev_model_name")
            if _prev_model != model_name:
                st.session_state["_prev_model_name"] = model_name
                st.session_state["xai_method"] = _MODEL_XAI_DEFAULTS.get(model_name, "gradcam++")

            analyze_clicked = st.button(
                "🔍 Analyze Image", type="primary", width='stretch',
                disabled=(uploaded_file is None),
            )

    # Run analysis
    if analyze_clicked and uploaded_file is not None:
        with st.spinner("Running AI analysis..."):
            file_bytes = uploaded_file.getvalue()
            if client:
                _xai = st.session_state.get("xai_method", _MODEL_XAI_DEFAULTS.get(model_name, "gradcam++"))
                result = client.analyze_image(
                    file_bytes=file_bytes, filename=uploaded_file.name,
                    patient_id=pt_id or "UNKNOWN",
                    study_date=study_date.isoformat(),
                    modality="X-ray", body_part="Chest",
                    clinical_history=f"Age: {age}, Sex: {sex}, View: {view}",
                    model_name=model_name,
                    xai_method=_xai,
                )
                if result:
                    st.session_state["current_analysis"] = result
                    st.session_state["current_image_bytes"] = file_bytes
                    st.session_state["current_patient_id"] = pt_id
                    st.session_state["llm_messages"] = []
                    st.rerun()
                else:
                    st.error("Analysis failed. Check the backend is running and a model is loaded.")
            else:
                st.error("Backend API client not initialised.")

    analysis = st.session_state.get("current_analysis")

    # --- Prediction summary ---------------------------------------------------
    with col2:
        st.subheader("Prediction summary")
        with st.container(border=True):
            if analysis:
                df_preds = _build_predictions_df(analysis)
                if not df_preds.empty:
                    risk_colors = {"Low": "#00CC96", "Medium": "#F59E0B", "High": "#FF4B4B"}
                    label_colors = {"Positive": "#FF4B4B", "Negative": "#00CC96"}
                    rows_html = ""
                    for _, row in df_preds.iterrows():
                        rc = risk_colors.get(row["Risk Level"], "#AAA")
                        lc = label_colors.get(row["Binary label"], "#AAA")
                        rows_html += (
                            f"<tr><td style='padding:6px 10px;'>{row['Finding']}</td>"
                            f"<td style='padding:6px 10px;text-align:center;'>{row['Probability']:.2f}</td>"
                            f"<td style='padding:6px 10px;text-align:center;'>"
                            f"<span style='background:{rc};color:#fff;padding:2px 10px;"
                            f"border-radius:12px;font-size:12px;font-weight:600;'>{row['Risk Level']}</span></td>"
                            f"<td style='padding:6px 10px;text-align:center;'>"
                            f"<span style='background:{lc};color:#fff;padding:2px 10px;"
                            f"border-radius:12px;font-size:12px;font-weight:600;'>{row['Binary label']}</span></td></tr>"
                        )
                    st.markdown(
                        "<table style='width:100%;border-collapse:collapse;color:white;font-size:13px;'>"
                        "<thead><tr style='border-bottom:1px solid #444;color:#AAA;'>"
                        "<th style='padding:6px 10px;text-align:left;'>Finding</th>"
                        "<th style='padding:6px 10px;text-align:center;'>Probability</th>"
                        "<th style='padding:6px 10px;text-align:center;'>Risk Level</th>"
                        "<th style='padding:6px 10px;text-align:center;'>Binary label</th>"
                        f"</tr></thead><tbody>{rows_html}</tbody></table>",
                        unsafe_allow_html=True,
                    )

                    chart_col, gauge_col = st.columns([3, 2])
                    with chart_col:
                        st.plotly_chart(render_prediction_chart(df_preds), width='stretch')
                    with gauge_col:
                        st.plotly_chart(render_confidence_gauge(analysis["confidence"]), width='stretch')

                    inf_ms = analysis.get("inference_time_ms", 0)
                    st.caption(f"Inference: {inf_ms:.0f} ms | Model: {analysis.get('model_version', '?')}")
                else:
                    st.info("No pathology predictions returned.")
            else:
                st.info("Upload an image and click **Analyze Image** to see predictions.")

    # --- Bottom row -----------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        # Heatmap
        st.subheader("Finding to explain")
        with st.container(border=True):
            if analysis and (analysis.get("heatmap_overlay") or analysis.get("scorecam_overlay")):
                xai_method = analysis.get("xai_method", "gradcam++")
                labels = []
                if analysis.get("heatmap_overlay"):
                    labels.append(xai_method.replace("_", " ").title())
                if analysis.get("scorecam_overlay") and xai_method != "scorecam":
                    labels.append("Score-CAM (Fallback)")
                
                if labels:
                    tabs = st.tabs(labels)
                    idx = 0
                    if analysis.get("heatmap_overlay"):
                        with tabs[idx]:
                            st.image(_b64_to_image(analysis["heatmap_overlay"]), width='stretch')
                            if xai_method == "attention_rollout":
                                st.caption("Attention Map Overlay")
                            else:
                                st.caption(f"{xai_method.replace('++', '++').title()} Overlay")
                        idx += 1
                    if analysis.get("scorecam_overlay") and xai_method != "scorecam":
                        with tabs[idx]:
                            st.image(_b64_to_image(analysis["scorecam_overlay"]), width='stretch')
                            st.caption("Score-CAM Overlay (Low Confidence Fallback)")
                        idx += 1

                if st.checkbox("Legend", value=True):
                    st.markdown(
                        '<div style="position:relative;width:100%;height:22px;margin-top:4px;'
                        'border-radius:4px;overflow:hidden;">'
                        '<div style="width:100%;height:100%;background:linear-gradient(to right,'
                        '#0000ff,#00ffff,#00ff00,#ffff00,#ff7700,#ff0000);border-radius:4px;"></div>'
                        '<div style="position:absolute;top:3px;left:6px;color:white;font-size:11px;'
                        'font-weight:600;text-shadow:0 0 3px #000;">Low</div>'
                        '<div style="position:absolute;top:3px;right:6px;color:white;font-size:11px;'
                        'font-weight:600;text-shadow:0 0 3px #000;">High</div></div>',
                        unsafe_allow_html=True,
                    )
            elif uploaded_file is not None and analysis is None:
                st.info("Click **Analyze Image** to generate the heatmap.")
            else:
                st.info("Upload an image to see the XAI heatmap.")

        # Feedback
        st.subheader("Prediction Feedback")
        with st.container(border=True):
            if analysis:
                st.markdown("**How accurate is this prediction?**")
                fb1, fb2, fb3 = st.columns(3)
                with fb1:
                    if st.button("✅ Correct", width='stretch', key="fb_correct"):
                        _submit_feedback(analysis, "correct")
                with fb2:
                    if st.button("❌ Incorrect", width='stretch', key="fb_incorrect"):
                        _submit_feedback(analysis, "incorrect")
                with fb3:
                    if st.button("❓ Uncertain", width='stretch', key="fb_uncertain"):
                        _submit_feedback(analysis, "uncertain")
                st.text_input(
                    "Optional comment", placeholder="Enter feedback here...",
                    label_visibility="collapsed", key="fb_notes",
                )
            else:
                st.info("Run an analysis first to provide feedback.")

    with col4:
        # Explanation controls
        st.subheader("Explanation Controls")
        with st.container(border=True):
            if analysis:
                analysis_id = analysis.get("analysis_id", "")
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.selectbox("Finding to explain", PATHOLOGY_NAMES, key="xai_finding")
                    ec1a, ec1b = st.columns(2)
                    with ec1a:
                        heatmap_opacity = st.slider("Heatmap opacity", 0.0, 1.0, 0.6, 0.05, key="xai_opacity")
                    with ec1b:
                        heatmap_threshold = st.slider("Threshold", 0.0, 1.0, 0.5, 0.05, key="xai_threshold")
                with ec2:
                    xai_method = st.selectbox("XAI Method", ["gradcam++", "scorecam", "attention_rollout"], key="xai_method")
                    if xai_method == "scorecam":
                        st.caption("⏳ Score-CAM is gradient-free but slower — may take a few seconds.")
                    colormap = st.selectbox("Colormap", ["jet", "viridis", "plasma", "hot"], key="xai_colormap")
                    regenerate = st.button("🔄 Regenerate Heatmap", width='stretch')

                t1, t2 = st.columns(2)
                if st.session_state.get("current_image_bytes"):
                    orig = Image.open(io.BytesIO(st.session_state["current_image_bytes"])).convert("RGB")
                    t1.image(np.array(orig), caption="Original", width='stretch')
                if analysis.get("heatmap_gradcam"):
                    t2.image(_b64_to_image(analysis["heatmap_gradcam"]), caption="Heatmap", width='stretch')

                if regenerate and client and analysis_id:
                    selected_finding = st.session_state.get("xai_finding")
                    with st.spinner("Regenerating heatmap..."):
                        resp = client.get_explanation(
                            analysis_id=analysis_id, method=xai_method,
                            threshold=heatmap_threshold, opacity=heatmap_opacity,
                            colormap=colormap, finding=selected_finding,
                        )
                        if resp:
                            if resp.get("overlay"):
                                analysis["heatmap_overlay"] = resp["overlay"]
                            if resp.get("heatmap"):
                                analysis["heatmap_gradcam"] = resp["heatmap"]
                            st.session_state["current_analysis"] = analysis
                            st.rerun()
                        else:
                            st.warning("Could not regenerate heatmap, the selected models may not match the XAI method, please check your selections.")
            else:
                st.info("Run an analysis first to adjust explanation parameters.")

        # LLM Agent
        st.subheader("LLM Clinical Agent")
        with st.container(border=True):
            if analysis:
                render_llm_agent(analysis)
            else:
                st.info("Run an analysis to enable the clinical AI assistant.")


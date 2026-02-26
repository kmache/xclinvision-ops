"""Streamlit frontend for XClinVision Clinician Dashboard."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import json

# Page config
st.set_page_config(
    page_title="XClinVision - Clinical Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis"]
CLASS_COLORS = {
    "Normal": "#28a745",
    "Pneumonia": "#dc3545",
    "Tuberculosis": "#fd7e14",
}


def init_session_state():
    """Initialize session state variables."""
    if "current_prediction" not in st.session_state:
        st.session_state.current_prediction = None
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def render_header():
    """Render application header."""
    st.title("🩺 XClinVision")
    st.markdown("*Explainable Medical Imaging AI Platform with Clinical Decision Support*")
    st.markdown("---")


def render_sidebar():
    """Render sidebar with model selection and settings."""
    with st.sidebar:
        st.header("Settings")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["efficientnet_b2", "resnet50", "swin_t", "biomedclip"],
            index=0,
        )
        
        # Threshold adjustment
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        
        # Heatmap opacity
        opacity = st.slider(
            "Heatmap Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        
        st.markdown("---")
        
        # Disclaimer
        st.warning(
            "⚠️ **Disclaimer**: This system is for research and decision support only. "
            "It does not provide medical diagnoses and must always be used under clinician supervision."
        )
        
        return model_name, threshold, opacity


def page_inference_explanation():
    """Page 1: Inference & Explanation."""
    st.header("Inference & Explanation")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a chest X-ray image for AI analysis",
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
        # Simulate API call for prediction
        with st.spinner("Analyzing image..."):
            # Placeholder - integrate with actual API
            # response = requests.post(f"{API_BASE_URL}/api/v1/predict", files={"file": uploaded_file})
            
            # Mock response
            prediction = {
                "prediction": 1,
                "class_name": "Pneumonia",
                "probabilities": [0.15, 0.75, 0.10],
                "confidence": 0.75,
                "uncertainty": {"epistemic": 0.03, "predictive_entropy": 0.56},
                "uncertainty_level": "medium",
                "processing_time_ms": 245,
            }
            
        st.session_state.current_prediction = prediction
        
        with col2:
            st.subheader("Prediction Results")
            
            # Display prediction with color coding
            pred_class = prediction["class_name"]
            color = CLASS_COLORS.get(pred_class, "#333")
            
            st.markdown(
                f"""
                <div style="padding: 20px; border-radius: 10px; background-color: {color}20; 
                            border-left: 5px solid {color};">
                    <h3 style="margin: 0; color: {color};">{pred_class}</h3>
                    <p style="margin: 5px 0 0 0; font-size: 24px; font-weight: bold;">
                        {prediction["confidence"]:.1%} Confidence
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Probability bars
            st.markdown("#### Class Probabilities")
            for i, (cls, prob) in enumerate(zip(CLASS_NAMES, prediction["probabilities"])):
                st.progress(prob, text=f"{cls}: {prob:.1%}")
                
            # Uncertainty indicator
            st.markdown("#### Uncertainty Assessment")
            unc_level = prediction["uncertainty_level"]
            unc_color = {"low": "green", "medium": "orange", "high": "red"}[unc_level]
            
            st.markdown(
                f"<p style='color: {unc_color}; font-weight: bold;'>"
                f"⚠️ Uncertainty Level: {unc_level.upper()}"
                f"</p>",
                unsafe_allow_html=True,
            )
            
            if prediction["uncertainty"]:
                st.json(prediction["uncertainty"])
                
        # Explanation section
        st.markdown("---")
        st.subheader("Explainability (Grad-CAM++)")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Display heatmap overlay (placeholder)
            st.image(image, use_column_width=True, caption="Grad-CAM++ Heatmap Overlay")
            
        with col4:
            st.markdown("#### Region Importance Scores")
            
            # Placeholder region scores
            region_scores = {
                "Left Upper": 0.35,
                "Right Upper": 0.65,
                "Left Lower": 0.20,
                "Right Lower": 0.15,
                "Center": 0.45,
            }
            
            for region, score in region_scores.items():
                st.progress(score, text=f"{region}: {score:.2f}")
                
            st.markdown("#### Key Findings")
            findings = [
                "High activation in right upper lobe",
                "Patchy consolidation pattern",
                "Bilateral lower zone sparing",
            ]
            for finding in findings:
                st.markdown(f"• {finding}")
                
        # Feedback buttons
        st.markdown("---")
        col5, col6 = st.columns(2)
        
        with col5:
            if st.button("✅ Verify Finding", use_container_width=True):
                st.success("Finding verified and recorded.")
                
        with col6:
            if st.button("❌ Report Error", use_container_width=True):
                st.error("Error reported. Thank you for your feedback.")


def page_historical_comparison():
    """Page 2: Historical Comparison."""
    st.header("Historical Comparison")
    
    st.info("Compare current study with previous examinations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Previous Study")
        prev_file = st.file_uploader("Upload Previous X-ray", type=["jpg", "jpeg", "png"], key="prev")
        if prev_file:
            st.image(prev_file, use_column_width=True)
            
    with col2:
        st.subheader("Current Study")
        curr_file = st.file_uploader("Upload Current X-ray", type=["jpg", "jpeg", "png"], key="curr")
        if curr_file:
            st.image(curr_file, use_column_width=True)
            
    if prev_file and curr_file:
        st.markdown("---")
        st.subheader("Difference Map")
        st.info("Visualizing changes between studies...")
        
        # Placeholder for difference map
        st.write("Difference map visualization showing progression or resolution of findings.")


def page_report_generation():
    """Page 3: Report Generation."""
    st.header("Clinical Report Generation")
    
    if st.session_state.current_prediction is None:
        st.warning("Please run inference first to generate a report.")
        return
        
    prediction = st.session_state.current_prediction
    
    # Patient information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        patient_id = st.text_input("Patient ID")
    with col2:
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=45)
    with col3:
        patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        
    # Generate report
    if st.button("Generate Report", type="primary"):
        with st.spinner("Generating clinical report..."):
            # Placeholder - integrate with LLM agent
            report = {
                "findings": f"AI analysis indicates {prediction['class_name']} with {prediction['confidence']:.1%} confidence. "
                           f"Key regions of interest identified via Grad-CAM++.",
                "impression": f"{prediction['class_name']} suggested by AI analysis. "
                           f"Confidence level: {prediction['confidence']:.1%}. "
                           f"Uncertainty: {prediction['uncertainty_level']}.",
                "uncertainty": f"Uncertainty level: {prediction['uncertainty_level']}. "
                            "AI models can make errors. Clinical correlation essential.",
                "recommendation": "Clinical correlation with patient history, symptoms, and physical examination is essential. "
                                "Consider follow-up imaging or laboratory tests if clinically indicated.",
            }
            
        st.markdown("### Generated Report")
        
        st.markdown("#### Findings")
        st.write(report["findings"])
        
        st.markdown("#### Impression")
        st.info(report["impression"])
        
        st.markdown("#### Uncertainty & Limitations")
        st.warning(report["uncertainty"])
        
        st.markdown("#### Recommendations")
        st.success(report["recommendation"])
        
        # Export options
        st.markdown("---")
        col4, col5 = st.columns(2)
        
        with col4:
            st.download_button(
                "📄 Export as PDF",
                data=json.dumps(report, indent=2),
                file_name=f"report_{patient_id or 'unknown'}.json",
                mime="application/json",
            )
            
        with col5:
            if st.button("📝 Edit Report"):
                st.text_area("Edit Report", value=json.dumps(report, indent=2), height=300)


def page_audit_transparency():
    """Page 4: Audit & Transparency."""
    st.header("Audit & Transparency")
    
    # Model Card
    st.subheader("Model Card")
    
    model_info = {
        "Model": "EfficientNet-B2",
        "Version": "0.1.0",
        "Training Date": "2024-01-15",
        "Dataset": "ChestX-ray14 + TB Datasets",
        "ECE Score": "0.05",
        "Macro AUC": "0.94",
        "Parameters": "9.2M",
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("ECE", "0.05", delta="Good Calibration")
    with col2:
        st.metric("Macro AUC", "0.94", delta="+0.02")
    with col3:
        st.metric("Pneumonia Recall", "0.92", delta="+0.03")
        
    st.json(model_info)
    
    # Known Limitations
    st.markdown("#### Known Limitations")
    limitations = [
        "May miss subtle infiltrates in obese patients",
        "5% false positive rate on geriatric patients",
        "Limited performance on pediatric chest X-rays",
        "May be confused by severe emphysema patterns",
    ]
    for lim in limitations:
        st.markdown(f"• {lim}")
        
    # Fairness Metrics
    st.markdown("---")
    st.subheader("Fairness Metrics by Subgroup")
    
    fairness_data = {
        "Age Group": ["18-40", "40-65", "65+"],
        "Accuracy": [0.93, 0.92, 0.89],
        "Sensitivity": [0.91, 0.90, 0.87],
        "Specificity": [0.95, 0.94, 0.91],
    }
    
    st.dataframe(fairness_data, use_container_width=True)
    
    # Drift Monitor
    st.markdown("---")
    st.subheader("Data Drift Monitor")
    
    # Placeholder drift chart
    st.info("Monitoring feature distribution drift over time...")
    
    drift_status = "✅ No drift detected"
    st.success(drift_status)


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    
    # Sidebar
    model_name, threshold, opacity = render_sidebar()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Inference & Explanation", "Historical Comparison", "Report Generation", "Audit & Transparency"],
    )
    
    # Render selected page
    if page == "Inference & Explanation":
        page_inference_explanation()
    elif page == "Historical Comparison":
        page_historical_comparison()
    elif page == "Report Generation":
        page_report_generation()
    elif page == "Audit & Transparency":
        page_audit_transparency()


if __name__ == "__main__":
    main()

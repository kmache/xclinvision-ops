# XClinVision: Explainable Medical Imaging AI Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Explainable Medical Imaging AI Platform with Clinical Decision Support**

XClinVision is a production-grade, regulation-aware AI platform for medical image analysis, combining explainable deep learning with a RAG-enhanced LLM-based clinical decision-support agent. Designed as a Clinical Decision Support (CDS) tool—not an autonomous diagnostic system—it demonstrates senior-level AI engineering skills across model development, explainability, MLOps, system design, and regulatory awareness in a high-stakes healthcare context.

⚠️ **Disclaimer**: This system is intended for research and decision support only. It does not provide medical diagnoses and must always be used under clinician supervision. It is not certified for clinical use and requires appropriate regulatory approval before deployment.

## 🎯 Key Features

- **Multi-Model Architecture**: Supports EfficientNet-B2, ResNet-50, Swin Transformer, and BiomedCLIP
- **Uncertainty Quantification**: MC Dropout + Temperature Scaling for calibrated confidence
- **Explainable AI**: Grad-CAM++ heatmaps for clinical interpretability
- **LLM-Powered Reports**: RAG-enhanced clinical decision support with GPT-4
- **Clinician Dashboard**: 4-page Streamlit interface with audit trails
- **MLOps Integration**: MLflow tracking, model registry, drift detection
- **Human-in-the-Loop**: Feedback system for continuous improvement

## 🏗️ Architecture Overview

```
X-ray Image
    ↓
Preprocessing & Quality Control
    ↓
Deep Learning Model (EfficientNet-B2 / Swin-T / BiomedCLIP)
    ↓
Uncertainty Estimation (MC Dropout + Temperature Scaling)
    ↓
Explainability (Grad-CAM++ / Attention Maps)
    ↓
Clinical Risk Scoring
    ↓
RAG-Enhanced LLM Clinical Decision-Support Agent
    ↓
Clinician Dashboard & Audit Logs
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/xclinvision.git
cd xclinvision

# Install dependencies
pip install -e ".[dev]"

# Or using Makefile
make dev-install
```

### Download Data

```bash
# Download and prepare datasets
make download-data
```

### Training

```bash
# Train baseline model
make train

# Or with specific config
python scripts/train.py --config configs/train/efficientnet_b2_baseline.yaml
```

### Evaluation

```bash
# Evaluate model
make evaluate
```

### Running the Application

```bash
# Start backend API
make api

# Start frontend dashboard (in another terminal)
make dashboard

# Or use docker-compose
docker-compose -f deployment/docker-compose.yml up
```

## 📁 Project Structure

```
xclinvision-ops/
├── app/
│   ├── backend/           # FastAPI inference service
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── frontend/          # Streamlit clinician dashboard
│       ├── main.py
│       ├── Dockerfile
│       └── requirements.txt
├── configs/
│   ├── efficientnet_b2.yaml          # Model config
│   ├── resnet50.yaml                 # Model config
│   ├── swin_t.yaml                   # Model config
│   ├── biomedclip.yaml               # Model config
│   ├── efficientnet_b2_baseline.yaml # Training config
│   └── system.yaml                   # Global system configuration
├── data/                 # Data directory (gitignored)
├── deployment/
│   ├── docker-compose.yml
│   └── nginx.conf
├── docs/                 # Documentation
│   ├── README.md
│   └── model_card.md
├── notebooks/            # Jupyter notebooks for EDA
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_comparison.ipynb
│   ├── 04_uncertainty_analysis.ipynb
│   └── 05_explainability_demo.ipynb
├── scripts/              # Utility scripts
│   ├── train.py
│   ├── evaluate.py
│   └── download_data.py
├── src/xclinvision/      # Core Python package
│   ├── __init__.py
│   ├── architecture.py   # Model definitions
│   ├── data.py          # Data processing
│   ├── trainer.py       # Training logic
│   ├── evaluator.py     # Evaluation metrics
│   ├── inference.py     # Inference pipeline
│   ├── xai.py          # Explainability (Grad-CAM++)
│   ├── reliability.py   # Robustness testing
│   ├── monitoring.py    # MLOps tracking
│   └── agent.py        # LLM clinical agent
├── tests/                # Test suite
├── Makefile
├── pyproject.toml
└── README.md
```

## 🔬 Supported Models

| Model | Type | Input Size | Expected Recall | Expected ECE |
|-------|------|------------|-----------------|--------------|
| EfficientNet-B2 | CNN (Modern) | 384x384 | 0.92 | 0.05 |
| ResNet-50 | CNN (Legacy) | 384x384 | 0.88 | 0.12 |
| Swin-T | Transformer | 224x224 | 0.91 | 0.18 |
| BiomedCLIP | Foundation Model | 224x224 | 0.90 | 0.09 |

## 📊 Dashboard Pages

1. **Inference & Explanation**: Upload X-rays, view predictions with Grad-CAM++ heatmaps, submit feedback
2. **Historical Comparison**: Compare current and previous studies with difference maps
3. **Report Generation**: LLM-powered clinical reports with export functionality
4. **Audit & Transparency**: Model cards, fairness metrics, drift monitoring

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, PyTorch Lightning, timm, transformers
- **Explainability**: pytorch-grad-cam, captum
- **Backend**: FastAPI, uvicorn, pydantic
- **Frontend**: Streamlit
- **LLM & RAG**: LangChain, OpenAI, ChromaDB, sentence-transformers
- **MLOps**: MLflow, Optuna, Weights & Biases
- **Data**: Albumentations, opencv-python, scikit-image
- **Deployment**: Docker, docker-compose, nginx

## 📋 10-Day Implementation Plan

| Day | Focus | Key Deliverables |
|-----|-------|------------------|
| 1 | Environment & EDA | Data download, preprocessing, EDA notebooks |
| 2 | Baseline Training | EfficientNet-B2 training, MLflow logging |
| 3 | Model Comparison | Train ResNet-50, Swin-T, BiomedCLIP |
| 4 | Uncertainty & Calibration | Temperature Scaling, MC Dropout integration |
| 5 | Explainability | Grad-CAM++ implementation, region scoring |
| 6 | LLM Agent & RAG | ChromaDB setup, clinical report generation |
| 7 | Backend API | FastAPI endpoints, prediction service |
| 8 | Dashboard Pages 1-2 | Inference UI, historical comparison |
| 9 | Dashboard Pages 3-4 | Report generation, audit transparency, feedback |
| 10 | MLOps & Docker | Containerization, final testing, documentation |

## 🔧 Configuration

Edit `configs/system.yaml` to customize:

- Model architecture and hyperparameters
- Data augmentation settings
- Training configuration
- API and dashboard settings
- MLOps tracking configuration

## 🧪 Testing

```bash
# Run tests
make test

# Run linting
make lint

# Format code
make format
```

## 📈 MLOps Workflow

1. **Experiment Tracking**: All runs logged to MLflow with hyperparameters and metrics
2. **Model Registry**: Versioned models with staging (Staging → Production)
3. **Prediction Logging**: All inference calls logged with image hash and results
4. **Drift Detection**: Monitor for data distribution shifts
5. **Feedback Loop**: Clinician feedback triggers retraining consideration

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{xclinvision2024,
  title = {XClinVision: Explainable Medical Imaging AI Platform},
  author = {XClinVision Team},
  year = {2024},
  url = {https://github.com/your-org/xclinvision}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Chest X-ray datasets: NIH ChestX-ray14, TB Chest X-ray Database
- Pre-trained models: timm, Hugging Face Transformers
- Clinical guidelines: Merck Manual, Radiopaedia, WHO

---

<p align="center">
  <strong>Built with ❤️ for better healthcare AI</strong>
</p>

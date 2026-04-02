# XClinVision-Ops

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg)](https://fastapi.tiangolo.com/)[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Explainable Medical Imaging AI Platform with Clinical Decision Support**

XClinVision-Ops is a production-grade platform for chest X-ray analysis that combines multi-model deep learning inference, explainability (Grad-CAM++), uncertainty quantification, and a RAG-enhanced reasoning agent for clinical decision support. It ships as a FastAPI backend + Streamlit dashboard, containerised with Docker.

> **Disclaimer:** This system is for research and clinical decision support only.It does not provide autonomous medical diagnoses and must be used under cliniciansupervision. Not certified for clinical use.

---

## Table of Contents

-   [Key Features](#key-features)
-   [Architecture](#architecture)
-   [Project Structure](#project-structure)
-   [Installation](#installation)
-   [Running the Project](#running-the-project)
-   [Usage Guide](#usage-guide)
-   [Agent & RAG](#agent--rag)
-   [Configuration](#configuration)
-   [Outputs](#outputs)
-   [Testing](#testing)
-   [Development Guidelines](#development-guidelines)
-   [Troubleshooting](#troubleshooting)
-   [Future Improvements](#future-improvements)

---

## Key Features

Area

Capabilities

**Model Inference**

4 production architectures (ConvNeXt-Small, DenseNet-121, EfficientNet-B0, ViT-Base), auto-discovered model registry, multilabel classification across 5 chest pathology classes

**Explainability**

Grad-CAM++ heatmap overlays, per-region clinical scoring, adjustable threshold & opacity

**Uncertainty**

MC Dropout + Temperature Scaling for calibrated confidence; uncertainty-aware clinical guidance

**Evaluation & Monitoring**

Per-class metrics (AUC, F1, sensitivity, specificity), threshold optimisation, calibration analysis (ECE), drift detection

**Reasoning Agent**

LLM-powered intent → plan → execute → synthesise loop with 7 callable tools and ChromaDB RAG over radiology literature

**Interactive Dashboard**

4-page Streamlit UI: inference & XAI, historical comparison, report generation, audit & transparency

**Report Generation**

Structured clinical reports with findings, impressions, and recommendations; HTML export

**MLOps**

MLflow experiment tracking, prediction logging, clinician feedback loop

**Target Classes:** No Finding · Cardiomegaly · Aortic Enlargement · Pleural Thickening · Pulmonary Fibrosis

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐│                         Streamlit Dashboard                             ││   Inference & XAI │ History │ Report Generation │ Audit & Transparency  │└────────┬───────────────────────────────────────────────────┬────────────┘         │  HTTP (REST)                                      │┌────────▼───────────────────────────────────────────────────▼────────────┐│                          FastAPI Backend                                 ││                                                                         ││  /api/v1/*  (predict, explain, report, feedback, metrics)               ││  /api/v2/*  (analyze, chat, generate-report, export-report,             ││              drift-metrics, model-card, feedback-stats)                  │└──┬──────────────┬──────────────┬──────────────┬────────────────────┬────┘   │              │              │              │                    │   ▼              ▼              ▼              ▼                    ▼ Inference     XAI Engine    Evaluator    ReasoningAgent       Monitoring Pipeline      (Grad-CAM++)  (Metrics,    (Intent→Plan→       (Drift, (PyTorch,     (heatmaps,    Calibration,  Execute→Synthesise)  Feedback  TIMM,        region        Thresholds)   7 Tools + LLM/RAG)  Stats)  Uncertainty)  scoring)
```

### Module Breakdown

Module

Location

Responsibility

**Backend**

`app/backend/`

FastAPI server, API endpoints (v1 + v2), model registry, pipeline caching

**Frontend**

`app/frontend/`

Streamlit dashboard, 4 pages (inference, history, reports, audit)

**Core Library**

`src/xclinvision/`

All ML logic: model building, data processing, training, evaluation, XAI, inference, uncertainty

**Agent**

`src/xclinvision/agent/`

ReasoningAgent, 7 tools, clinical report generation, dialogue management, guardrails, RAG ingestion

**Configs**

`configs/`

System-wide settings (`system.yaml`), per-model training configs, guardrail vocabulary

**Scripts**

`scripts/`

CLI entry points for training, evaluation, XAI generation, data preparation

**Deployment**

`deployment/`

Production Docker Compose + nginx reverse proxy

**Tests**

`tests/`

API integration tests + model pipeline tests

---

## Project Structure

```
xclinvision-ops/├── app/│   ├── backend/                    # FastAPI inference service│   │   ├── main.py                 #   API endpoints (v1 + v2)│   │   ├── schemas.py              #   Pydantic request/response models│   │   ├── Dockerfile              #   Backend container image│   │   └── requirements.txt        #   Backend-specific dependencies│   └── frontend/                   # Streamlit clinician dashboard│       ├── main.py                 #   App entrypoint & page navigation│       ├── api_client.py           #   Backend HTTP client│       ├── config.py               #   Frontend configuration│       ├── styles.py               #   Medical-themed CSS│       ├── Dockerfile              #   Frontend container image│       ├── requirements.txt        #   Frontend-specific dependencies│       └── views/                  #   Dashboard pages│           ├── page_inference.py   #     Inference & Explanation│           ├── page_history.py     #     Historical Comparison│           ├── page_report.py      #     Report Generation│           └── page_audit.py       #     Audit & Transparency├── src/xclinvision/                # Core Python package│   ├── modeling.py                 #   Model architectures (GeM pooling, build_model)│   ├── inference.py                #   Inference pipeline│   ├── processing.py               #   Data processing pipeline│   ├── dataset.py                  #   PyTorch Dataset & DataModule│   ├── trainer.py                  #   Lightning training module│   ├── evaluator.py                #   Metrics, calibration, threshold optimisation│   ├── xai.py                      #   Grad-CAM++ explainability│   ├── config.py                   #   Configuration loading│   ├── monitoring.py               #   Drift detection│   ├── reliability.py              #   Uncertainty & failure analysis│   └── agent/                      #   Clinical decision-support agent│       ├── reasoning.py            #     ReasoningAgent (plan → execute → synthesise)│       ├── tools.py                #     7 callable tools + ToolRegistry│       ├── xclinvisionagent.py     #     Full LLM + RAG agent│       ├── reporter.py             #     Clinical report generation│       ├── dialogue.py             #     Dialogue management│       ├── guardrails.py           #     Safety guardrails│       ├── audit.py                #     Agent audit logging│       ├── ingest_knowledge.py     #     RAG knowledge ingestion│       └── templates/              #     Report templates├── configs/                        # Configuration files│   ├── system.yaml                 #   Global: classes, paths, clinical rules, thresholds│   ├── guardrail_terms.yaml        #   Agent safety vocabulary│   ├── convnext_small.yaml         #   ConvNeXt-Small config (recommended)│   ├── densenet.yaml               #   DenseNet-121 config│   ├── efficientnet_b0.yaml        #   EfficientNet-B0 config│   └── vit_base.yaml               #   ViT-Base config├── scripts/                        # CLI entry points│   ├── train.py                    #   Training (auto-processes data on first run)│   ├── evaluate.py                 #   Evaluation with calibration & threshold tuning│   ├── generate_xai.py             #   Grad-CAM++ / Score-CAM heatmap generation│   ├── download_data.py            #   Dataset download│   ├── organize_data.py            #   Data preparation & splitting│   ├── clean_system.sh             #   Cache & temp file cleanup│   └── run_training/               #   Shell launchers for batch training│       ├── train_all.sh            #     Train all 4 models sequentially│       ├── train_convnext_small.sh│       ├── train_densenet.sh│       ├── train_efficientnet_b0.sh│       └── train_vit_base.sh├── data/                           # Data directory (gitignored)│   ├── raw/                        #   Original unprocessed images│   ├── processed_384/              #   Processed at 384px│   ├── processed_1024_1024/        #   Processed at 1024px│   └── vector_db/                  #   ChromaDB embeddings for RAG├── models/                         # Trained models (gitignored)│   └── best_models/                #   Production-ready checkpoints (.pth + _meta.json)├── NLMCXR_reports/                 # Radiology reports for RAG ingestion (gitignored)├── outputs/                        # Evaluation & XAI outputs (gitignored)├── logs/                           # Runtime logs (gitignored)├── notebooks/                      # Analysis notebooks│   ├── 01_data_exploration.ipynb│   ├── 03_post_evaluation.ipynb│   ├── 04_uncertainty_analysis.ipynb│   ├── 05_explainability_demo.ipynb│   └── 06_data_audit.ipynb├── deployment/                     # Production deployment│   ├── docker-compose.yml          #   Production compose (backend + frontend + nginx)│   └── nginx.conf                  #   Reverse proxy with rate limiting├── tests/                          # Test suite│   ├── test_api_integration.py     #   Backend API integration tests│   └── test_multilabel_integration.py  # Model pipeline tests├── archive/                        # Archived experiments & unused files├── docs/                           # Extended documentation│   ├── README.md                   #   Detailed workflow docs│   ├── model_card.md               #   Model card│   └── SickKids.pdf                #   Reference material├── docker-compose.yml              # Development compose (backend + frontend)├── Makefile                        # Common commands├── pyproject.toml                  # Package metadata & dependencies├── requirements.txt                # Full dependency list└── README.md
```

---

## Installation

### Prerequisites

-   Python 3.10+
-   [Conda](https://docs.conda.io/) (recommended) or virtualenv
-   Docker & Docker Compose (for containerised deployment)
-   NVIDIA GPU + CUDA (optional, for training; CPU inference supported)

### 1. Clone the repository

```bash
git clone https://github.com/your-org/xclinvision-ops.gitcd xclinvision-ops
```

### 2. Create an environment

```bash
# Using condaconda create -n xclinvision_env python=3.12 -yconda activate xclinvision_env# Or using venvpython -m venv .venv && source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Production installpip install -e .# Development install (includes linting, testing tools)pip install -e ".[dev]"# Or via Makefilemake install          # productionmake dev-install      # development
```

### 4. Environment variables

Create a `.env` file or export variables directly:

```bash
# Required for LLM-powered agent features (optional without)export OPENAI_API_KEY="sk-..."# Backend configuration (defaults shown)export XCLINVISION_MODELS_DIR="models/best_models"export XCLINVISION_DATA_DIR="data/processed_384"export XCLINVISION_OUTPUTS_DIR="outputs/evaluation"export XCLINVISION_ARCHITECTURE="convnext_small"export XCLINVISION_IMAGE_SIZE="384"# Frontendexport API_URL="http://localhost:8000"
```

All variables have sensible defaults — the system runs without a `.env` filefor local development if model weights are placed in `models/best_models/`.

### 5. Download data (optional — for training)

```bash
make download-data    # Downloads VinBigData from Kagglemake preprocess       # Organises and splits the dataset
```

---

## Running the Project

### Local Development

Start backend and frontend in separate terminals:

```bash
# Terminal 1 — Backend API (port 8000)make api# Terminal 2 — Frontend Dashboard (port 8501)make dashboard
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Docker (Development)

```bash
docker-compose up --build
```

This starts:

-   **Backend** on port `8000`
-   **Frontend** on port `8501`

Model weights are mounted read-only from `models/best_models/`.

### Docker (Production)

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

This adds an **nginx** reverse proxy on port `80` with:

-   Rate limiting (10 req/s per IP on `/api`)
-   20 MB upload limit
-   WebSocket support for Streamlit

```bash
# Build, start, and stop via Makefilemake docker-buildmake docker-upmake docker-down
```

---

## Usage Guide

### 1. Select a Model

The backend auto-discovers models from `XCLINVISION_MODELS_DIR`. Each modelneeds a `.pth` weights file and a `_meta.json` descriptor. The dashboardmodel selector shows all discovered models.

Available architectures:

Model

Config

Best For

ConvNeXt-Small

`convnext_small.yaml`

Best overall (recommended default)

DenseNet-121

`densenet.yaml`

Lightweight, fast inference

EfficientNet-B0

`efficientnet_b0.yaml`

Efficient compute/accuracy balance

ViT-Base

`vit_base.yaml`

Attention-based, global context

### 2. Run Inference

**Dashboard:** Navigate to **Inference & Explanation**, upload a chest X-ray,select a model, and click **Run Analysis**. You'll see:

-   Class predictions with probability bars
-   Confidence gauge with uncertainty level
-   Grad-CAM++ heatmap overlay (adjustable threshold & opacity)

**API:**

```bash
# V1 — Simple predictioncurl -X POST http://localhost:8000/api/v1/predict   -F "file=@chest_xray.jpg"   -F "model_name=convnext_small"# V2 — Full analysis (prediction + uncertainty + XAI + LLM summary)curl -X POST http://localhost:8000/api/v2/analyze   -F "file=@chest_xray.jpg"   -F "patient_id=P001"   -F "model_name=convnext_small"
```

### 3. View Metrics

**Dashboard:** Go to **Audit & Transparency** → **Model Card** tab forcross-model performance tables (AUC, F1, sensitivity, specificity) loadedfrom saved evaluation reports.

**API:**

```bash
# Evaluation metricscurl http://localhost:8000/api/v1/metrics# Drift monitoringcurl http://localhost:8000/api/v2/drift-metrics# Model cardcurl http://localhost:8000/api/v2/model-card
```

### 4. Chat with the Agent

On the **Inference & Explanation** page, use the chat panel or clickquick-action chips:

-   *"Explain this prediction"*
-   *"What should be done next?"*
-   *"Assess urgency"*
-   *"Generate report"*

The agent classifies intent, executes relevant tools, and synthesises agrounded response.

**API:**

```bash
curl -X POST http://localhost:8000/api/v2/chat   -H "Content-Type: application/json"   -d '{"message": "Explain the prediction in detail", "analysis_id": "XCL-..."}'
```

### 5. Generate & Export Reports

**Dashboard:** Go to **Report Generation**, select analyses, and generatestructured clinical reports. Edit findings/impressions, then export as HTML.

**API:**

```bash
# Generate reportcurl -X POST http://localhost:8000/api/v2/generate-report   -H "Content-Type: application/json"   -d '{"analysis_id": "XCL-...", "template": "standard"}'# Export as self-contained HTMLcurl -X POST http://localhost:8000/api/v2/export-report   -H "Content-Type: application/json"   -d '{"analysis_id": "XCL-..."}'
```

---

## Agent & RAG

The **ReasoningAgent** implements a structured reasoning loop:

```
User Query    ↓Intent Classification (9 intent types, regex-based)    ↓Action Planning (intent → ordered tool list)    ↓Tool Execution (7 built-in tools, collected results)    ↓Synthesis (LLM-powered if OPENAI_API_KEY set, else rule-based)    ↓Grounded Response + Follow-up Suggestions
```

### Intent Types

`explain_prediction` · `explain_heatmap` · `suggest_next_steps` · `assess_urgency` ·`compare_history` · `get_metrics` · `get_monitoring` · `generate_report` · `general_question`

### Tools

Tool

Description

`get_prediction_details`

Prediction data: confidence, uncertainty, top-k classes

`get_xai_explanation`

XAI spatial evidence: region scores, key findings

`get_evaluation_metrics`

Model metrics: AUC, F1, sensitivity, specificity, ECE

`get_monitoring_status`

Drift score, prediction counts, feedback stats

`generate_report`

Structured clinical report (findings, impressions, recommendations)

`compare_with_history`

Temporal comparison with prior studies

`suggest_next_steps`

Confidence and uncertainty-aware clinical recommendations

### RAG Pipeline

The agent can ingest radiology literature (NLMCXR reports) into a ChromaDBvector store via `ingest_knowledge.py`. During inference, the`ClinicalReasoningAgent` retrieves relevant passages to ground LLM responsesin clinical evidence.

```bash
# Ingest reports into vector DBpython -m xclinvision.agent.ingest_knowledge
```

Set `OPENAI_API_KEY` to enable LLM synthesis. Without it, the agent fallsback to rule-based response generation.

---

## Configuration

### Two-Layer Config System

1.  **`configs/system.yaml`** — Global settings shared by all models:
    
    -   Class names and classification mode (`multilabel`)
    -   Data paths (raw, processed, quarantine)
    -   Clinical rules (expected regions per class)
    -   Dashboard settings (title, theme, layout)
    -   MLOps settings (experiment tracking, drift thresholds)
    -   Logging configuration
2.  **`configs/{model}.yaml`** — Per-model training parameters:
    
    -   Architecture and backbone
    -   Input size, dropout rate
    -   Loss function (`focal` / `ce` / `asl`)
    -   Pooling type (`gem` / `avg`)
    -   Differential learning rates (backbone vs. head)
    -   Explainability target layer

Model configs are self-contained — each defines everything needed to train.The system merges model config over `system.yaml` at runtime.

### Example: ConvNeXt-Small Config

```yaml
name: convnext_smallarchitecture:  backbone: convnext_small  pretrained: true  dropout_rate: 0.3input:  size: [384, 384]training:  loss: focal  pooling: gem  process_size: 1024transfer_learning:  differential_lr:    backbone_lr: 5.0e-5    head_lr: 5.0e-4
```

Override any parameter via CLI:

```bash
python scripts/train.py --config configs/convnext_small.yaml --lr 1e-4 --epochs 50
```

---

## Outputs

Directory

Contents

`outputs/evaluation/`

Evaluation reports (JSON), classification reports (TXT), per-sample predictions (CSV), optimised thresholds, temperature parameters

`outputs/xai/`

Grad-CAM++ / Score-CAM heatmap images

`logs/`

Runtime logs (`xclinvision.log`), rotating file handler (10 MB, 5 backups)

`logs/predictions/`

Prediction audit logs (per-inference records)

`models/{model}_{timestamp}/`

Training run outputs (checkpoints, metrics, config snapshot)

`models/best_models/`

Promoted production checkpoints (`.pth` + `_meta.json`)

`mlruns/`

MLflow experiment tracking data

All output directories are gitignored. `.gitkeep` files preserve empty directories.

---

## Testing

```bash
# Run all testsmake test# Run with verbose outputpytest tests/ -v# Run only API integration testspytest tests/test_api_integration.py -v# Run only model pipeline testspytest tests/test_multilabel_integration.py -v
```

### Test Coverage

File

Tests

Description

`test_api_integration.py`

28

Health check, model listing, v1 predict/explain/report/feedback, v2 analyze/chat/report/export/drift/model-card/feedback-stats, error handling

`test_multilabel_integration.py`

14

Config, dataset shapes, focal loss, metrics computation, prediction serialisation, temperature scaling, failure analysis

All API tests use mocked inference pipelines — no GPU or model weightsrequired.

```bash
# Linting & formattingmake lint      # black --check, isort --check, flake8, mypymake format    # auto-format with black + isort
```

---

## Development Guidelines

### Code Organisation

```
src/xclinvision/     ← All ML and business logic lives hereapp/backend/         ← Thin API layer: imports from src/, no ML logicapp/frontend/        ← UI layer: calls backend API, no direct ML imports
```

-   **All model, evaluation, agent, and inference logic** belongs in `src/xclinvision/`.
-   **Backend** (`app/backend/main.py`) is a thin API layer that imports from thecore package. Pydantic schemas live in `schemas.py`.
-   **Frontend** (`app/frontend/`) communicates exclusively through the REST APIvia `api_client.py` — it never imports `src/` directly.

### Adding a New Model

1.  Create a config file: `configs/{model_name}.yaml` (use an existing one as template).
2.  Register the architecture in `src/xclinvision/modeling.py` → `build_model()`.
3.  Train: `python scripts/train.py --config configs/{model_name}.yaml`
4.  Evaluate: `python scripts/evaluate.py --checkpoint-path models/.../ --model-name {model_name}`
5.  Promote to production: copy `.pth` + `_meta.json` to `models/best_models/`.

The backend auto-discovers models in `best_models/` — no code changes neededto serve a new model.

### Adding a New Agent Tool

1.  Define the tool function in `src/xclinvision/agent/tools.py`:
    
    ```python
    def my_tool(analysis: dict, context: dict) -> ToolResult:    ...    return ToolResult(tool="my_tool", success=True, data={...})
    ```
    
2.  Register it in `build_default_tool_registry()`.
3.  Map it to intents in `reasoning.py` → `_INTENT_TOOL_MAP`.

### Adding a New API Endpoint

1.  Define request/response models in `app/backend/schemas.py`.
2.  Add the route in `app/backend/main.py`.
3.  Add a client method in `app/frontend/api_client.py`.
4.  Add tests in `tests/test_api_integration.py`.

---

## Troubleshooting

### Docker

Issue

Fix

Backend health check fails

Ensure `models/best_models/` contains at least one model (`.pth` + `_meta.json`). Check logs: `docker-compose logs backend`

Frontend can't connect to backend

Verify `API_URL=http://backend:8000` in the frontend service. Ensure backend is healthy before frontend starts

Upload size limit exceeded

Production nginx limits to 20 MB. Adjust `client_max_body_size` in `deployment/nginx.conf`

GPU not available in container

Add `deploy.resources.reservations.devices` to backend service in `docker-compose.yml`. Ensure NVIDIA Container Toolkit is installed

### Dependencies

Issue

Fix

`torch` installation fails

Install PyTorch separately first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

`opencv-python` conflicts

Backend/frontend use `opencv-python-headless` to avoid GUI dependencies. Don't mix both

`chromadb` build fails

Ensure `gcc` and `python3-dev` are installed: `sudo apt install build-essential python3-dev`

### Paths & Data

Issue

Fix

"No models found" on startup

Set `XCLINVISION_MODELS_DIR` to the directory containing your `.pth` files, or place them in `models/best_models/`

Processed data not found

Run `python scripts/organize_data.py` first, or let `train.py` auto-process on first run

Agent responses are generic

Set `OPENAI_API_KEY` for LLM-powered synthesis. Without it, the agent uses rule-based fallback

---

## Future Improvements

-   **Multi-GPU and distributed training** for faster experimentation
-   **DICOM viewer integration** with direct PACS connectivity
-   **Additional architectures** (e.g., Swin Transformers, BiomedCLIP) — archived configs available in `archive/configs/`
-   **Ensemble inference** combining predictions from multiple models
-   **PostgreSQL persistence** for predictions, feedback, and audit trails (currently in-memory)
-   **User authentication** and role-based access control
-   **CI/CD pipeline** with automated testing, linting, and container builds
-   **FHIR integration** for export to electronic health records
-   **Multi-language report generation**

---

## License

MIT — see [LICENSE](LICENSE) for details.
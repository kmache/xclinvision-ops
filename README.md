# XClinVision-Ops

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-powered Medical Imaging Platform with Clinical Decision Support**

XClinVision-Ops is a production-grade chest X-ray analysis platform combining multi-model deep learning inference, visual explainability, uncertainty quantification, an LLM-powered reasoning agent with RAG, and full report generation. It ships as a FastAPI backend + Streamlit dashboard, containerised with Docker.

> **Disclaimer:** This system is intended for research and clinical decision support only. It does not provide autonomous medical diagnoses and must be used under qualified clinician supervision. Not certified for clinical use.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Environment Variables](#environment-variables)
- [LLM Integration](#llm-integration)
- [RAG Pipeline](#rag-pipeline)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Report Generation](#report-generation)
- [Docker & Deployment](#docker--deployment)
- [Configuration](#configuration)
- [Testing](#testing)
- [Development Notes](#development-notes)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Key Features

| Area | Capabilities |
|---|---|
| **Model Inference** | 4 production architectures (ConvNeXt-Small, DenseNet-121, EfficientNet-B0, ViT-Base); auto-discovered model registry; multilabel classification across 5 chest pathology classes |
| **Explainability** | Grad-CAM++ and Score-CAM heatmap overlays; attention maps; per-region clinical scoring; adjustable threshold & opacity |
| **Uncertainty** | MC Dropout + Temperature Scaling for calibrated confidence; epistemic vs. aleatoric uncertainty breakdown |
| **LLM Agent** | Intent classification → tool planning → execution → LLM synthesis loop; 7 callable tools; streaming chat with quick-action chips |
| **RAG** | ChromaDB + BM25 hybrid retrieval (RRF fusion) over 3 900+ IU CXR radiology reports; retrieval grounded in clinical evidence |
| **Report Generation** | Full clinical reports via Jinja2 template; Grad-CAM++ overlay + radar chart; export as HTML, PDF (WeasyPrint), or JSON |
| **History & Comparison** | Per-patient analysis history; side-by-side temporal comparison; manual multi-image upload comparison |
| **Evaluation & Monitoring** | Per-class AUC, F1, sensitivity, specificity; ECE calibration; threshold optimisation; drift detection |
| **MLOps** | MLflow experiment tracking; prediction logging; clinician feedback loop; audit trails |

**Target pathologies:** No Finding · Cardiomegaly · Aortic Enlargement · Pleural Thickening · Pulmonary Fibrosis

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Streamlit Dashboard                            │
│  Inference & XAI │ History & Comparison │ Reports │ Audit          │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP / SSE (streaming)
┌────────────────────────────▼────────────────────────────────────────┐
│                        FastAPI Backend                              │
│  /api/v1/*  predict · explain · report · feedback · metrics        │
│  /api/v2/*  analyze · chat · chat/stream · compare ·               │
│             generate-report · export-report ·                       │
│             llm/providers · drift-metrics · model-card             │
└──┬──────────────┬──────────────┬──────────────┬──────────────┬─────┘
   │              │              │              │              │
   ▼              ▼              ▼              ▼              ▼
Inference    XAI Engine    Evaluator    ReasoningAgent    Monitoring
Pipeline    (Grad-CAM++,  (Metrics,    (Intent → Plan →  (Drift,
(PyTorch,    Score-CAM,   Calibration, Execute →         Feedback,
 TIMM,       Heatmaps,    Thresholds)  Synthesise)       Audit)
 GeM Pool)   Regions)                  │
                                       ├─ LLM Provider
                                       │   (OpenAI / local)
                                       └─ RAG Retriever
                                           (ChromaDB + BM25)
```

---

## Project Structure

```
xclinvision-ops/
├── app/
│   ├── backend/                        # FastAPI inference service
│   │   ├── main.py                     #   All API endpoints (v1 + v2)
│   │   ├── schemas.py                  #   Pydantic request/response models
│   │   ├── Dockerfile                  #   Backend container image
│   │   └── requirements.txt            #   Backend-specific dependencies
│   └── frontend/                       # Streamlit clinician dashboard
│       ├── main.py                     #   App entrypoint & page navigation
│       ├── api_client.py               #   Backend HTTP client
│       ├── config.py                   #   Frontend configuration & endpoints
│       ├── styles.py                   #   Medical-themed CSS
│       ├── Dockerfile                  #   Frontend container image
│       ├── requirements.txt
│       └── views/                      #   Dashboard pages
│           ├── page_inference.py       #     Inference, XAI & chat
│           ├── page_history.py         #     Historical comparison
│           ├── page_report.py          #     Report generation & export
│           └── page_audit.py           #     Audit & transparency
├── src/xclinvision/                    # Core Python package
│   ├── modeling.py                     #   Model architectures (GeM pooling, build_model)
│   ├── inference.py                    #   Inference pipeline with uncertainty
│   ├── processing.py                   #   DICOM/image processing pipeline
│   ├── dataset.py                      #   PyTorch Dataset & Lightning DataModule
│   ├── trainer.py                      #   Lightning training & transfer learning
│   ├── evaluator.py                    #   Metrics, ECE calibration, threshold optimisation
│   ├── xai.py                          #   Grad-CAM++ & Score-CAM explainability
│   ├── config.py                       #   Configuration loading & class registry
│   ├── monitoring.py                   #   Drift detection & prediction logging
│   ├── reliability.py                  #   Uncertainty & failure analysis
│   └── agent/                          #   Clinical decision-support agent
│       ├── xclinvisionagent.py         #     Full LLM + RAG agent
│       ├── reasoning.py                #     Intent → plan → execute → synthesise
│       ├── tools.py                    #     7 callable tools + ToolRegistry
│       ├── reporter.py                 #     Clinical HTML/PDF report generation
│       ├── ingest_knowledge.py         #     RAG knowledge ingestion pipeline
│       ├── dialogue.py                 #     Streaming dialogue management
│       ├── guardrails.py               #     Safety guardrails
│       ├── llm_provider.py             #     LLM provider abstraction (OpenAI, local)
│       ├── audit.py                    #     Agent audit logging
│       └── templates/
│           └── clinical_report.html    #     Jinja2 clinical report template
├── configs/                            # YAML configuration files
│   ├── system.yaml                     #   Global: classes, paths, clinical rules
│   ├── guardrail_terms.yaml            #   Agent safety vocabulary
│   ├── convnext_small.yaml             #   ConvNeXt-Small (recommended)
│   ├── densenet.yaml                   #   DenseNet-121
│   ├── efficientnet_b0.yaml            #   EfficientNet-B0
│   └── vit_base.yaml                   #   ViT-Base
├── scripts/                            # CLI entry points
│   ├── train.py                        #   Training
│   ├── evaluate.py                     #   Evaluation with calibration & threshold tuning
│   ├── generate_xai.py                 #   Grad-CAM++ / Score-CAM heatmap generation
│   ├── download_data.py                #   Dataset download
│   ├── organize_data.py                #   Data preparation & splitting
│   ├── clean_system.sh                 #   Cache & temp cleanup
│   └── run_training/                   #   Shell launchers (train all 4 models)
├── data/                               # Data (gitignored)
│   ├── raw/                            #   Original images
│   ├── processed_384/                  #   Processed 384px
│   ├── processed_1024_1024/            #   Processed 1024px
│   └── vector_db/                      #   ChromaDB + BM25 for RAG
├── models/
│   └── best_models/                    # Production checkpoints (.pth + _meta.json)
├── NLMCXR_reports/                     # IU CXR radiology reports (RAG source, gitignored)
├── outputs/                            # Evaluation & XAI outputs (gitignored)
├── logs/                               # Runtime logs (gitignored)
├── notebooks/                          # Analysis notebooks
├── deployment/                         # Production deployment
│   ├── docker-compose.yml              #   Prod compose (backend + frontend + nginx)
│   └── nginx.conf                      #   Reverse proxy with rate limiting
├── tests/                              # Test suite
├── docs/                               # Extended documentation & model card
├── docker-compose.yml                  # Development compose
├── Makefile                            # Common commands
└── pyproject.toml                      # Package metadata
```

---

## Installation

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for containerised deployment)
- NVIDIA GPU + CUDA (optional — CPU inference is supported)

### 1. Clone

```bash
git clone https://github.com/your-org/xclinvision-ops.git
cd xclinvision-ops
```

### 2. Create environment

```bash
# Conda (recommended)
conda create -n xclinvision python=3.12 -y
conda activate xclinvision

# Or virtualenv
python -m venv .venv && source .venv/bin/activate
```

### 3. Install

```bash
pip install -e .          # production
pip install -e ".[dev]"   # with linting & test tools
```

> **Note:** PyTorch is not included automatically due to CUDA variant selection.
> Install separately before the above:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Create `.env`

```bash
cp .env.example .env
# Then fill in OPENAI_API_KEY (optional but required for LLM features)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for LLM-powered agent and report generation |
| `OPENAI_MODEL` | `gpt-5.4-nano` | Primary chat/reasoning model |
| `OPENAI_API_BASE` | OpenAI default | Custom base URL (Azure OpenAI, DeepSeek, etc.) |
| `XCLINVISION_MODELS_DIR` | `models/best_models` | Directory scanned for `.pth` model checkpoints |
| `XCLINVISION_DATA_DIR` | `data/processed_384` | Processed image data root |
| `XCLINVISION_OUTPUTS_DIR` | `outputs/evaluation` | Evaluation output directory |
| `XCLINVISION_ARCHITECTURE` | `convnext_small` | Default model architecture |
| `XCLINVISION_IMAGE_SIZE` | `384` | Input image size |
| `API_URL` | `http://localhost:8000` | Backend URL (used by the frontend) |

All variables have defaults. The system will start without a `.env` file,but LLM features will fall back to rule-based responses.

---

## LLM Integration

### Supported Providers

The `LLMProvider` abstraction in `src/xclinvision/agent/llm_provider.py`supports the following backends:

| Provider | Class | How to Enable |
|---|---|---|
| **OpenAI** | `OpenAIProvider` | Set `OPENAI_API_KEY` |
| **DeepSeek** | `OpenAIProvider` (compatible) | Set `OPENAI_API_KEY` + `OPENAI_API_BASE=https://api.deepseek.com/v1` |
| **Gemini** | `OpenAIProvider` (compatible) | Set `OPENAI_API_KEY` + `OPENAI_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai` |
| **Local (Ollama/vLLM)** | `LocalProvider` | Set `OPENAI_API_BASE=http://localhost:11434/v1` (no key needed) |

### Model selection & fallback

```
Primary model:   OPENAI_MODEL (default: gpt-5.4-nano)
Fallback model:  gpt-4o-mini (automatic if primary fails)
```

The provider automatically:
- Uses `max_completion_tokens` for gpt-5/o-series models and `max_tokens` for others
- Falls back to `gpt-4o-mini` if the primary model returns a model-not-found error
- Tests TCP reachability for local providers before attempting inference

### Streaming

The backend exposes Server-Sent Events (SSE) at `POST /api/v2/chat/stream`.The frontend chat panel consumes the stream via `requests` chunked transfer.Streaming falls back to non-streaming if the provider doesn't support it.

### Switch providers at runtime

```bash
# List available providers
curl http://localhost:8000/api/v2/llm/providers

# Switch active provider
curl -X POST http://localhost:8000/api/v2/llm/switch \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai"}'

# Health check
curl http://localhost:8000/api/v2/llm/health
```

---

## RAG Pipeline

The **HybridRetriever** combines ChromaDB semantic search with BM25 keyword search, fused via Reciprocal Rank Fusion (RRF).

### Knowledge ingestion

```bash
python -m xclinvision.agent.ingest_knowledge \
  --reports-dir NLMCXR_reports/ecgen-radiology \
  --output-dir data/vector_db \
  --embedding-model BAAI/bge-m3 \
  --validate
```

This ingests:
- **Indiana University CXR reports** (XML) — findings + impression sections
- **Clinical guidelines** (PDF, HTML)
- **Tabular annotations** (CSV, Excel)
- **Synthetic clinical signals** (built-in)

Output:
- `data/vector_db/chroma/` — ChromaDB persistent store
- `data/vector_db/bm25_index.pkl` — Serialised BM25 index

### Retrieval during inference

Each agent request retrieves the top-k relevant passages. These are injectedas context into the LLM system prompt, grounding responses in clinical evidence rather than hallucination.

When `OPENAI_API_KEY` is not set, the agent uses rule-based synthesis withoutRAG retrieval.

---

## Usage Guide

### 1. Run an analysis

Upload a chest X-ray via the **Inference & Explanation** dashboard page orsend it directly to the API:

```bash
curl -X POST http://localhost:8000/api/v2/analyze \
  -F "file=@chest_xray.jpg" \
  -F "patient_id=P-0001" \
  -F "model_name=convnext_small"
```

You'll receive predictions, confidence scores, uncertainty level, an LLM summary, and XAI region scores.

### 2. View explainability

On the dashboard, the Grad-CAM++ overlay appears automatically after inference. To fetch it separately:

```bash
curl http://localhost:8000/api/v2/explain/{analysis_id}
```

You can switch between **Grad-CAM++**, **Score-CAM**, and **Attention** methods via the XAI method selector on the inference page.

### 3. Chat with the agent

Type free-text questions in the chat panel, or click quick-action chips:
- *"Explain this prediction"*
- *"What should be done next?"*
- *"Assess urgency"*
- *"Generate report"*

Alternatively, call the streaming API:

```bash
curl -X POST http://localhost:8000/api/v2/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the prediction", "analysis_id": "XCL-...", "context_type": "clinical"}'
```

### 4. Generate and export reports

Navigate to **Report Generation**, edit findings/impressions/comments, then click:

| Button | Action |
|---|---|
| **Save Draft** | Stores current text to session state |
| **Generate AI Report** | Calls `/api/v2/generate-report` to auto-fill findings from the agent |
| **Export HTML** | Calls `/api/v2/export-report` → downloads self-contained HTML with Grad-CAM overlays |
| **Export PDF** | Calls `/api/v2/export-report` (WeasyPrint) → downloads PDF |
| **Export JSON** | Downloads structured JSON report from local session data |

### 5. Compare analyses

On the **History** page, select a patient ID to view all past analyses. Use the **Manual Comparison** tab to upload two images side-by-side for direct comparison.

---

## API Reference

### Core

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/models` | List available models |
| `GET` | `/api/v1/dataset/info` | Dataset statistics |
| `GET` | `/api/v1/metrics` | Evaluation metrics |

### Inference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/predict` | Simple prediction (image file) |
| `POST` | `/api/v1/explain` | Grad-CAM++ explanation (image + model) |
| `POST` | `/api/v2/analyze` | Full analysis: prediction + uncertainty + XAI + LLM summary |
| `POST` | `/api/v2/compare` | Side-by-side comparison of two uploaded images |
| `GET` | `/api/v2/explain/{analysis_id}` | Fetch explanation for stored analysis |
| `GET` | `/api/v2/history/{patient_id}` | Patient analysis history |

### Chat & Agent

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v2/chat` | Non-streaming chat |
| `POST` | `/api/v2/chat/stream` | Streaming chat (SSE) |
| `GET` | `/api/v2/llm/providers` | List LLM providers |
| `POST` | `/api/v2/llm/switch` | Switch active LLM provider |
| `GET` | `/api/v2/llm/health` | LLM provider health check |

### Reports

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v2/generate-report` | Generate structured report sections |
| `POST` | `/api/v2/export-report` | Export as HTML, PDF, or JSON |

**Export-report request:**
```json
{
  "analysis_id": "XCL-abc12345",
  "format": "html",
  "include_xai": true,
  "include_uncertainty": true
}
```

**Export-report response (HTML):**
```json
{
  "html": "<html>...</html>",
  "report_id": "RPT-a1b2c3d4",
  "format": "html",
  "timestamp": "2026-04-03T12:00:00"
}
```

**Export-report response (PDF):**
```json
{
  "pdf_base64": "JVBERi0x...",
  "report_id": "RPT-a1b2c3d4",
  "format": "pdf",
  "timestamp": "2026-04-03T12:00:00"
}
```

### Monitoring

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/feedback` | Submit clinician feedback |
| `POST` | `/api/v2/feedback` | Submit v2 feedback |
| `GET` | `/api/v2/feedback-stats` | Aggregated feedback statistics |
| `GET` | `/api/v2/drift-metrics` | Dataset drift monitoring metrics |
| `GET` | `/api/v2/model-card` | Model documentation + live stats |

---

## Report Generation

Clinical reports are generated via the `ClinicalReporter` class using a Jinja2 template (`src/xclinvision/agent/templates/clinical_report.html`). The template produces a fully self-contained HTML document with:

- AI findings table with calibrated confidence language
- Grad-CAM++ overlay embedded as base64 PNG
- Radar/spider chart (prediction profile vs. normal baseline)
- Reasoning trace and differential diagnosis
- Impression, urgency, and next steps
- Citations and clinical disclaimer
- `@media print` CSS for browser/WeasyPrint PDF fidelity

**PDF generation** requires WeasyPrint, installed automatically via `requirements.txt`. System libraries (`libpango`, `libcairo2`, etc.) are installed in the Docker image.

---

## Docker & Deployment

### Development (docker-compose.yml)

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

Services:
- `backend` → port `8000`
- `frontend` → port `8501`

Volumes: `models/best_models/` is mounted read-only into the backend.

### Production (deployment/docker-compose.yml)

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

Adds an **nginx** reverse proxy on port `80`:
- Rate limiting: 10 req/s per IP on `/api/*`
- Upload limit: 20 MB
- WebSocket passthrough for Streamlit

### Environment variables in Docker

Pass variables via `.env` file or inline:

```yaml
# docker-compose.yml
services:
  backend:
    env_file: .env
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
```

### Required system libraries (backend image)

The backend Dockerfile installs these for OpenCV and WeasyPrint:

```
libgl1  libglib2.0-0  libsm6  libxext6  libxrender-dev
libpango-1.0-0  libpangocairo-1.0-0  libgdk-pixbuf-2.0-0
libcairo2  libffi-dev
```

---

## Configuration

### 1. `configs/system.yaml` — Global settings

```yaml
classification:
  mode: multilabel
  classes: [No finding, Cardiomegaly, Aortic enlargement, Pleural thickening, Pulmonary fibrosis]
  n_classes: 5
paths:
  models_dir: models/best_models
  data_dir: data/processed_384
training:
  batch_size: 16
  image_size: 384
```

### 2. `configs/{model}.yaml` — Per-model training config

```yaml
name: convnext_small
architecture:
  backbone: convnext_small
  pretrained: true
  dropout_rate: 0.3
input:
  size: [384, 384]
training:
  loss: focal
  pooling: gem
transfer_learning:
  differential_lr:
    backbone_lr: 5.0e-5
    head_lr: 5.0e-4
```

Model configs override system.yaml at runtime. Override any field via CLI:

```bash
python scripts/train.py --config configs/convnext_small.yaml \
  --lr 1e-4 --epochs 50 --batch-size 32
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Skip tests requiring GPU/torch
pytest tests/test_api_integration.py -v

# Run with coverage
pytest tests/ --cov=src/xclinvision --cov-report=term-missing
```

All API integration tests use mocked inference pipelines — no GPU or weightsrequired.

---

## Development Notes

### Extending the system

**Adding a new model:**
1. Create `configs/{name}.yaml`
2. Register the backbone in `src/xclinvision/modeling.py` → `build_model()`
3. Train: `python scripts/train.py --config configs/{name}.yaml`
4. Evaluate: `python scripts/evaluate.py ...`
5. Copy `.pth` + `_meta.json` to `models/best_models/` — the backend auto-discovers it

**Adding a new agent tool:**
1. Define a function in `src/xclinvision/agent/tools.py` returning a `ToolResult`
2. Register it in `build_default_tool_registry()`
3. Map it to intents in `reasoning.py` → `_INTENT_TOOL_MAP`

**Adding a new explainability method:**
1. Implement in `src/xclinvision/xai.py`
2. Register as a selectable method in the backend `explain` endpoint
3. Expose via the XAI method selector in `page_inference.py`

**Adding a new report template:**
1. Create a Jinja2 HTML file in `src/xclinvision/agent/templates/`
2. Load it by name in `ClinicalReporter.__init__()` or pass `template_name` to `generate_html()`

### Request flow

```
Frontend button click
  → api_client.py (requests.Session)
  → FastAPI endpoint (main.py)
  → xclinvision.inference.InferencePipeline  (model prediction)
  → xclinvision.xai                          (heatmap generation)
  → xclinvision.agent.XClinVisionAgent       (LLM + RAG synthesis)
  → xclinvision.agent.reporter.ClinicalReporter  (HTML/PDF report)
  → HTTP response → Streamlit widget
```

---

## Troubleshooting

### Docker build

| Error | Fix |
|---|---|
| `Package 'libgdk-pixbuf2.0-0' has no installation candidate` | Debian Trixie renamed this package. Use `libgdk-pixbuf-2.0-0` (with dash before 2) in the Dockerfile |
| `exit code: 100` during apt-get | A listed package doesn't exist in the base image's apt repo — check package names against the current Debian release |

### Runtime errors

| Error | Fix |
|---|---|
| `No module named 'tiktoken'` | Add `tiktoken>=0.5.0` to `app/backend/requirements.txt` and rebuild the Docker image |
| `No module named 'dotenv'` | Add `python-dotenv>=1.0.0` to requirements |
| `max_tokens is not supported` | OpenAI o-series/gpt-5 models require `max_completion_tokens`. The provider handles this automatically via `_completion_tokens_kwarg()` |
| `500 Internal Server Error` on `/api/v2/export-report` | Check backend logs (`docker-compose logs backend`). Common causes: missing `tiktoken`, missing model weights, or agent import failure |
| `Connection refused` to backend | Ensure the backend is healthy before the frontend starts. Use `depends_on: condition: service_healthy` in docker-compose |
| Streaming response not appearing | Ensure the client is consuming SSE chunks incrementally (not buffering). The backend uses `StreamingResponse` with `text/event-stream` |

### Configuration

| Issue | Fix |
|---|---|
| "No models found" at startup | Set `XCLINVISION_MODELS_DIR` to the folder containing `.pth` files with matching `_meta.json` descriptors |
| Agent responses are generic/non-medical | Set `OPENAI_API_KEY`. Without it the system falls back to rule-based response generation. Run the RAG ingestion pipeline to improve retrieval quality |
| GPU not detected | Install NVIDIA Container Toolkit and add `deploy.resources.reservations.devices` to the backend service in docker-compose |

---

## Contributing

1. Fork and clone the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make changes, add or update tests
5. Run the test suite: `pytest tests/ -v`
6. Format code: `black src/ app/ tests/ && isort src/ app/ tests/`
7. Submit a pull request with a clear description of the change

Please follow the existing code organisation:
- ML logic → `src/xclinvision/`
- API layer → `app/backend/main.py`
- UI layer → `app/frontend/views/`

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

> These AI-generated findings are intended to assist — not replace — clinical judgement.
> Always correlate with clinical presentation and consult a qualified radiologist.

# XClinVision-Ops

An end-to-end chest X-ray analysis system: multilabel classification across four
findings, Grad-CAM++/Score-CAM explainability, MC-Dropout uncertainty, per-class
calibration, an LLM reasoning agent with RAG retrieval over radiology reports, and
HTML/PDF report generation — served as a FastAPI backend with a Streamlit dashboard.
Built as a portfolio and research artifact.

> **Not for clinical use.** This is not a medical device and has no regulatory clearance.
> It was never validated prospectively or on any population outside its training dataset,
> and two of its four findings are below usable precision (see
> [Known limitations](#known-limitations)). Do not use it for any decision affecting a
> patient, and do not send it real patient data — analyses and images are stored
> unencrypted with no de-identification.

[Demo video](https://youtu.be/4_3VytZUHDE?si=rAhWslSZANGs_5Tl) ·
[Model card](docs/MODEL_CARD.md) ·
[Full evaluation](docs/model_evaluation.md) ·
[Review log](docs/reviews/)

---

## Measured performance

Validation split, n = 2,133. Best checkpoint (`vit_base`). Average precision must be read
against prevalence, not against 0.5.

| Class | Prevalence | AUC | AP | P @ 90% recall | Usable alone? |
|---|---:|---:|---:|---:|---|
| Cardiomegaly | 12.80% | 0.957 | 0.757 | 0.568 | yes |
| Aortic enlargement | 16.50% | 0.959 | 0.809 | 0.605 | yes |
| Pleural thickening | 6.19% | 0.904 | 0.429 | 0.179 | **no** |
| Pulmonary fibrosis | 7.17% | 0.898 | 0.523 | 0.183 | **no** |

| Checkpoint | macro AUC | macro F1 | mean test ECE |
|---|---:|---:|---:|
| `vit_base` | **0.930** | **0.626** | 0.0151 |
| `convnext_small` | 0.900 | 0.563 | 0.0147 |
| `efficientnet_b0` | 0.896 | 0.539 | 0.0148 |
| `densenet` | 0.889 | 0.530 | 0.0161 |

Per-class figures for all four checkpoints, ECE before/after, fitted parameters and
thresholds: [`docs/model_evaluation.md`](docs/model_evaluation.md).

### Calibration

**Calibrated.** All four checkpoints carry a fitted per-class affine map,
`p = sigmoid(z / T + b)`, fitted on the validation split and stored in the checkpoint
payload. Mean ECE fell from **0.199–0.291 to 0.0147–0.0161** on the held-out test split.

Served responses expose `raw_probability` alongside `calibration_status`. A checkpoint
only reports `calibrated` when its held-out ECE both improves and lands at or below 0.05;
one that fails the bar keeps its parameters, still applies them, and continues reporting
`uncalibrated` so the report layer keeps hedging its language.

Caveat: calibration was fitted on the same split used for model selection. There is no
separate calibration split.

---

## Subsystems

| Subsystem | Lines | Files | Test coverage |
|---|---:|---:|---:|
| `src/xclinvision/` core ML — modeling, training, inference, XAI, evaluation | 12,225 | 24 | **24%** |
| `src/xclinvision/agent/` — LLM providers, RAG, reasoning, reporting | (of the above) | 11 | **38%** |
| `app/backend/` — FastAPI, storage, auth | 3,247 | 6 | **74%** |
| `app/frontend/` — Streamlit dashboard | 3,306 | 9 | **0%** |
| `tests/` | 4,052 | 13 | — |

Overall statement coverage across measured packages: **38%** (6,425 statements, 3,974
uncovered). The distribution is uneven and worth stating plainly: the API surface is well
covered (`auth.py` 94%, `storage.py` 90%, `main.py` 71%), while `trainer.py`,
`modeling.py` and `dataset.py` sit at **0%** and `xai.py` at **14%** — the training and
explainability paths are exercised by hand, not by tests. The Streamlit frontend has no
tests at all.

```
Streamlit view ──HTTP/SSE──▶ FastAPI ──▶ InferencePipeline (PyTorch + MC-Dropout)
                                    ├──▶ xai (Grad-CAM++ / Score-CAM)
                                    ├──▶ XClinVisionAgent ──▶ LLM provider
                                    │                    └──▶ HybridRetriever (Chroma + BM25, RRF)
                                    └──▶ ClinicalReporter (Jinja2 → HTML/PDF)
```

---

## Quickstart

Verified from a clean clone into an empty virtualenv. **Read the ceiling first:** this
gets you a running stack and a passing test suite, not working inference — model weights
are not in the repository.

```bash
git clone <repo-url> xclinvision-ops && cd xclinvision-ops
python -m venv .venv && source .venv/bin/activate

# PyTorch is installed separately: the CUDA build is not in pyproject.toml.
# The pair is not interchangeable — see the note in Installation below.
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

pip install -e ".[dev]"

python -m pytest tests/ -q          # 190 passed, 5 skipped
```

Five skips rather than three: three tests need the dataset on disk, and two need trained
checkpoints. With both present the suite reports **192 passed, 3 skipped**.

Then start the services:

```bash
cp .env.example .env                # set XCLINVISION_API_TOKEN or protected routes 503
make api                            # backend on :8000
make dashboard                      # dashboard on :8501, separate shell
curl localhost:8000/health          # {"status":"healthy","version":"0.1.0",...}
```

Every command above was run against a fresh clone in an empty virtualenv before being
written here. `/health` returns healthy; `/api/v2/analyze` returns the 503 described
below.

### What a clean clone cannot do

`models/` is gitignored and ships only a `.gitkeep`, so **there are no checkpoints**.
The services start and `/health` is green, but `/api/v1/predict`, `/api/v2/analyze` and
`/api/v2/compare` return **503 "No model loaded"** until a `<name>.pth` plus matching
`<name>_meta.json` exists in `models/best_models/`. Verified:

```
$ curl -X POST localhost:8000/api/v2/analyze -H "Authorization: Bearer $TOKEN" -F file=@chest.png
{"detail":"No model loaded. Ensure models exist in models/best_models/ and restart."}
```

To get there you must train:

```bash
# 1. obtain VinBigData Chest X-ray (Kaggle) and organise it
python scripts/organize_data.py --help
# 2. train — GPU hours
python scripts/train.py --config configs/convnext_small.yaml
# 3. fit calibration and thresholds into the exported checkpoint
python scripts/calibrate_checkpoints.py
```

`data/test_sample/` ships ~35 chest X-ray PNGs for exercising the UI once weights exist.

The dev stack also runs under Docker, which has the same weights requirement:

```bash
docker-compose up --build           # backend + dashboard, volume-mounts models/ and data/
```

---

## Known limitations

**Two of the four classes are not usable on their own.** Pleural thickening and Pulmonary
fibrosis reach 0.18 precision at 90% recall — roughly five false positives per true
finding — against 0.57–0.61 for the two mediastinal classes. Their ranking is real
(AUC 0.898–0.904); their precision at any clinically meaningful recall is not. This is a
two-finding system, and calibration does not change that: the limit is separability.

**Calibration is fitted, not independent.** ECE is now 0.015 on held-out test, but the fit
used the same validation split as model selection and early stopping. No separate
calibration split exists.

**`horizontal_flip: true` contradicts the XAI layer's laterality claims.**
`configs/system.yaml:9` flips images during training, teaching partial left/right
invariance, while `src/xclinvision/xai.py:49-58` maps regions by anatomical side
("Patient LEFT lung is on RIGHT of image") and reports render those labels as spatial
evidence. Measured: flipping at inference costs at most 0.026 AUC and usually under
0.005, but shifts per-image probabilities by 0.03–0.09. The model is largely insensitive
to the laterality the XAI layer asserts. Current resolution: the flip stays enabled, the
labels stay side-specific, and left/right attribution in reports should be treated as
unverified.

**XAI faithfulness is unvalidated.** Grad-CAM++, Score-CAM and attention rollout are
displayed and embedded into exported reports, and **no faithfulness metric exists in the
codebase** — no deletion/insertion curves, no sanity checks, no agreement against the
bounding boxes the dataset ships. A convincing heatmap over a wrong prediction is the
expected failure mode and nothing would catch it.

**Patient-level leakage cannot be excluded.** No image appears in two splits, and a
perceptual-hash sweep found 0 real duplicates across train/val among 96 candidates. But
VinBigData ships no patient identifier, so the split is image-level; multiple studies of
one patient could straddle splits undetected.

**Authentication is one shared token.** `app/backend/auth.py` enforces a single bearer
token and fails closed when unset, but there is no per-patient scoping, no RBAC, no
per-user identity and no audit trail. Any holder of the token reads every stored analysis.

**Also:** ImageNet pretraining rather than medical-domain; one institution, one country,
no external validation; no subgroup analysis by age or sex (the dataset ships neither);
MC-Dropout uncertainty reported but its calibration unvalidated.

Full detail and ethical considerations: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Installation

Python 3.10+ (developed on 3.12). Optional NVIDIA GPU — CPU inference works.

**The torch/torchvision pair is not interchangeable.** `torchvision 0.26.0` pins
`torch==2.11.0`, and both wheels must carry the same CUDA build. torchvision checks this
at *import* time, so a mismatch breaks every model load before a checkpoint is read:

```
RuntimeError: Detected that PyTorch and torchvision were compiled with different
CUDA major versions. PyTorch has CUDA Version=13.0 and torchvision has CUDA Version=12.8.
```

`pyproject.toml` pins the bare versions because a `+cu130` local tag is unsatisfiable
from plain PyPI; the build comes from the `--index-url`. For CPU-only, swap `cu130` for
`cpu` on **both** packages.

*Gotcha:* a `torch` installed in `~/.local` (user site) takes precedence over your venv
and will shadow it, producing the error above even when the environment is internally
consistent. Check what actually loads:

```bash
python -c "import torch, torchvision; print(torch.__version__, torch.__file__); print(torchvision.__version__)"
```

WeasyPrint (PDF export) needs system pango/cairo. On Debian/Ubuntu:
`libpango-1.0-0 libpangoft2-1.0-0 libgdk-pixbuf-2.0-0 libffi-dev`.

### Environment

`.env` is loaded by the backend and by docker-compose (compose env wins in Docker). See
`.env.example` for the full list — none are strictly required except the API token:

| Variable | Purpose |
|---|---|
| `XCLINVISION_API_TOKEN` | **Required.** Unset → every protected v2 endpoint returns 503 |
| `OPENAI_API_KEY` | Optional. Without it, LLM features fall back to rule-based responses |
| `OPENAI_API_BASE` | Points the OpenAI-compatible client elsewhere (DeepSeek, Gemini, Ollama, vLLM) |
| `XCLINVISION_MODELS_DIR` | Checkpoint directory (default `models/best_models/`) |
| `HEATMAP_STORE_MAX` | Heatmap blob retention (default 20,000 ≈ 4 GB) |

---

## API reference

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
| `POST` | `/api/v1/predict` | Simple prediction |
| `POST` | `/api/v1/explain` | Grad-CAM++ explanation |
| `POST` | `/api/v2/analyze` | Prediction + uncertainty + XAI + LLM summary |
| `POST` | `/api/v2/compare` | Side-by-side comparison of two images |
| `GET` | `/api/v2/explain/{analysis_id}` | Explanation for a stored analysis |
| `GET` | `/api/v2/history/{patient_id}` | Patient analysis history |

### Chat, reports, monitoring

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v2/chat` · `/api/v2/chat/stream` | Chat, non-streaming and SSE |
| `GET` | `/api/v2/llm/providers` · `/api/v2/llm/health` | Provider list and health |
| `POST` | `/api/v2/llm/switch` | Switch active provider at runtime |
| `POST` | `/api/v2/generate-report` | Structured report sections |
| `POST` | `/api/v2/export-report` | Export HTML, PDF or JSON |
| `POST` | `/api/v1/feedback` · `/api/v2/feedback` | Clinician feedback |
| `GET` | `/api/v2/feedback-stats` · `/api/v2/drift-metrics` | Aggregates and drift |
| `GET` | `/api/v2/model-card` | Model documentation + live stats |

Both API versions are live: `/api/v1/*` is the simple predict/explain surface, `/api/v2/*`
adds analyze, chat, streaming, compare, report export and drift.

**Export-report** takes `{analysis_id, format: html|pdf|json, include_xai,
include_uncertainty}` and returns `{html|pdf_base64|json, report_id, format, timestamp}`.

---

## Development

```bash
python -m pytest tests/ -v                                   # full suite
python -m pytest tests/test_api_integration.py -v            # API only, no GPU or weights
python -m pytest tests/ --cov=xclinvision --cov-report=term-missing
make lint      # black --check + isort --check + flake8 + mypy
make format    # black + isort
```

Most API tests run without torch weights or a GPU — `tests/conftest.py` patches the
pipeline with a mock. Preserve that when adding tests.

### Conventions worth knowing

- **Class names are config-driven.** Always go through `xclinvision.config.get_class_names()`
  / `get_class_map()` / `is_multilabel()`; the source of truth is
  `configs/system.yaml → model.class_names`. In multilabel mode `"No finding"` is filtered
  out automatically, which is why four classes are served from a five-name list.
- **Lazy imports in `src/xclinvision/__init__.py`.** Heavy ML deps load only when their
  symbol is accessed.
- **Models are auto-discovered** from `XCLINVISION_MODELS_DIR`. Drop `<name>.pth` plus
  `<name>_meta.json` and the registry picks it up.
- **Per-model YAMLs override `system.yaml`** at runtime; CLI flags override both.

### Extending

| Goal | Steps |
|---|---|
| New architecture | `configs/<name>.yaml` → register in `modeling.py::build_model` → train → drop into `models/best_models/` |
| New agent tool | Add to `agent/tools.py` returning `ToolResult` → register in `build_default_tool_registry()` → map intent in `agent/reasoning.py::_INTENT_TOOL_MAP` |
| New XAI method | Implement in `xai.py` → expose in the backend `explain` endpoint → add to the selector in `page_inference.py` |
| New report template | Drop a Jinja2 file in `agent/templates/` → pass `template_name` to `ClinicalReporter.generate_html()` |

### Deployment

`deployment/docker-compose.yml` is the production stack (adds nginx with rate limiting
and a 20 MB upload cap); the root `docker-compose.yml` is dev-only. Neither lists
`OPENAI_*` in its `environment:` block — an explicit entry there would override
`env_file` and mask your `.env`.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | Intended and out-of-scope use, training data, metrics, limitations, ethics |
| [`docs/model_evaluation.md`](docs/model_evaluation.md) | Every measured number: per-class AUC/AP/ECE, calibration parameters, thresholds |
| [`docs/reviews/`](docs/reviews/) | Review log — each issue mapped to the commit that fixed it |

## License

MIT for the code. The dataset carries its own terms — see
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md#training-data).

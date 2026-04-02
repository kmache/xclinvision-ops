# XClinVision — Detailed Documentation

## Workflow Overview

```
1. Data Processing    →  data/processed_384/ or data/processed_1024_1024/
2. Training           →  models/{model}_{timestamp}/
3. Evaluation         →  outputs/evaluation/
4. Explainability     →  outputs/xai/
5. RAG Ingestion      →  data/vector_db/
```

## 1. Data Processing

Raw images are processed once and reused by all training runs that share the
same `process_size`.

**Directory naming**: `data/processed_{target_size}` (e.g. `processed_1024`).
No version suffixes, no hashes — one directory per resolution.

Processing is triggered **automatically** by `train.py` on the first run.
To manually re-process:

```bash
python scripts/train.py --config configs/convnext_small.yaml --force-reprocess
```

## 2. Training

### Config-Driven Workflow

Each architecture has a YAML config in `configs/`. The 4 best-performing
models have active configs; archived configs for other architectures are
in `archive/configs/`. The config defines **everything** needed to train:

- `input.size`: model input resolution (images resized from process_size)
- `training.loss`: loss function (`focal` recommended)
- `training.pooling`: pooling type (`gem` recommended)
- `training.process_size`: resolution for processed images on disk
- `transfer_learning`: differential learning rates
- `architecture.dropout_rate`: dropout rate

**To train**, just specify the config:

```bash
python scripts/train.py --config configs/convnext_small.yaml
```

### Focal + GeM v2 (Default)

All configs ship with `loss: focal` + `pooling: gem` — the combination that
empirically produces the best results:

- **Focal Loss**: Handles severe class imbalance in medical imaging by
  down-weighting easy negatives and focusing on hard samples.
- **GeM Pooling**: Learnable pooling that amplifies strong local activations,
  critical for small pathologies (fibrosis, pleural thickening) that occupy
  <5% of the feature map.

To compare with the baseline (CE + AvgPool):
```bash
python scripts/train.py --config configs/convnext_small.yaml --loss ce --pooling avg
```

### CLI Overrides

Any config value can be overridden:

| Flag | Description |
|------|-------------|
| `--loss focal\|ce\|asl` | Loss function |
| `--pooling gem\|avg` | Pooling type |
| `--lr 1e-4` | Learning rate |
| `--epochs 100` | Max epochs |
| `--batch-size 32` | Batch size |
| `--process-size 1024` | Processing resolution |
| `--image-size 384` | Model input resolution |
| `--force-reprocess` | Force data re-processing |
| `--no-progressive-unfreeze` | Disable progressive unfreezing |
| `--accumulate-grad-batches 2` | Gradient accumulation |
| `--patience 10` | Early stopping patience |

## 3. Evaluation

### Basic Usage

```bash
python scripts/evaluate.py \
    --checkpoint-path models/convnext_small_20260401_003723/best.ckpt \
    --model-name convnext_small \
    --image-size 384 \
    --pooling gem \
    --output-dir outputs/evaluation
```

### Full Evaluation Pipeline

```bash
python scripts/evaluate.py \
    --checkpoint-path models/convnext_small_20260401_003723/best.ckpt \
    --model-name convnext_small \
    --image-size 384 \
    --pooling gem \
    --calibrate \
    --optimize-thresholds \
    --tta \
    --output-dir outputs/evaluation
```

### Evaluation Flags

| Flag | Description |
|------|-------------|
| `--checkpoint-path` | **(Required)** Path to `.ckpt` file |
| `--model-name` | Architecture name (must match checkpoint) |
| `--image-size 384` | Must match training resolution |
| `--pooling gem` | Must match training pooling |
| `--calibrate` | Fit temperature scaling on val set |
| `--optimize-thresholds` | Tune per-class thresholds (multilabel) |
| `--tta` | Test-time augmentation |
| `--split test\|val` | Which split to evaluate |
| `--batch-size 32` | Inference batch size |
| `--output-dir` | Where to save results |

### Output Files

| File | Contents |
|------|----------|
| `{model}_test_evaluation_report.json` | Macro F1, macro AUC, per-class P/R/F1, ECE, failure analysis |
| `{model}_test_classification_report.txt` | Text classification report |
| `{model}_test_predictions.csv` | Per-sample: filepath, true labels, predictions, confidence, probabilities |
| `{model}_thresholds.json` | Per-class optimised thresholds |
| `{model}_temperature.json` | Temperature scaling parameter |

### What to Expect

Typical metrics for ConvNeXt-Small (Focal+GeM v2, VinBigData 5-class multilabel):

| Metric | Range | Notes |
|--------|-------|-------|
| Macro F1 | 0.55–0.65 | Across 5 classes including "No Finding" |
| Macro AUC | 0.80–0.88 | |
| Subset Accuracy | 0.45–0.55 | Exact-match (strict for multilabel) |
| ECE | 0.03–0.08 | Expected calibration error |

Per-class performance varies — "No Finding" and "Cardiomegaly" typically score
highest; "Pleural Thickening" and "Pulmonary Fibrosis" are harder due to
subtle, small-area features.

### Threshold Optimization

When `--optimize-thresholds` is used, the script:
1. Collects predictions on the **validation** set
2. Searches for per-class thresholds that maximise F1
3. Re-applies those thresholds to the **test** set predictions
4. Prints a comparison table (default 0.5 vs optimised)

This typically improves macro F1 by 2–5 points.

## 4. Explainability

```bash
python scripts/generate_xai.py \
    --checkpoint-path models/convnext_small_.../best.ckpt \
    --model-name convnext_small \
    --image-size 384 \
    --pooling gem \
    --output-dir outputs/xai \
    --num-samples 5
```

Generates side-by-side images: Original | Grad-CAM++ | Score-CAM for
stratified random test samples.

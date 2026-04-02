#!/bin/bash
# Train ConvNeXt-Small — Focal+GeM v2 (best empirical config)
# CNN — 50M params | img_size 384 | batch_size 32
# Uses existing processed images from data/processed_384

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/convnext_small.yaml \
  --model   convnext_small \
  --epochs  50 \
  --batch-size 32 \
  --image-size  384 \
  --process-size 384 \
  --lr      5e-4 \
  --loss    focal \
  --pooling gem \
  --weight-decay 1e-4 \
  --accumulate-grad-batches 2 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

#!/bin/bash
# Train DenseNet-121
# CNN — 8M params | img_size 384 | batch_size 24

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/densenet.yaml \
  --model   densenet \
  --epochs  50 \
  --batch-size 32 \
  --image-size  384 \
  --process-size 384 \
  --lr      5e-4 \
  --loss    focal \
  --pooling gem \
  --weight-decay 1e-4 \
  --accumulate-grad-batches 3 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

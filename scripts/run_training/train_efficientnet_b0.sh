#!/bin/bash
# Train EfficientNet-B0
# CNN — 5.3M params | img_size 384 | batch_size 32

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/efficientnet_b0.yaml \
  --model   efficientnet_b0 \
  --epochs  50 \
  --batch-size 32 \
  --image-size  384 \
  --process-size 1024 \
  --lr      5e-4 \
  --loss    ce \
  --weight-decay 1e-4 \
  --accumulate-grad-batches 2 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

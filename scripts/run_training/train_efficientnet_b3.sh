#!/bin/bash
# Train EfficientNet-B3
# CNN — 12M params | img_size 384 | batch_size 24

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/efficientnet_b3.yaml \
  --model   efficientnet_b3 \
  --epochs  50 \
  --batch-size 32 \
  --image-size  384 \
  --lr      1e-4 \
  --loss    focal \
  --weight-decay 1e-4 \
  --label-smoothing 0.15 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

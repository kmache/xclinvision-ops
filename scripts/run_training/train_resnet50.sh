#!/bin/bash
# Train ResNet-50
# CNN — 25M params | img_size 384 | batch_size 64

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/resnet50.yaml \
  --model   resnet50 \
  --epochs  50 \
  --batch-size 64 \
  --lr      1e-4 \
  --loss    focal \
  --weight-decay 1e-4 \
  --label-smoothing 0.1 \
  --output-dir models \
  --num-workers 4 \
  --seed    42

#!/bin/bash
# Train Swin-Base (patch4_window7_224)
# Transformer — 88M params | img_size 224 | batch_size 16 | lower lr

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/swin_b.yaml \
  --model   swin_b \
  --epochs  50 \
  --batch-size 16 \
  --lr      5e-5 \
  --loss    focal \
  --weight-decay 1e-4 \
  --label-smoothing 0.1 \
  --output-dir models \
  --num-workers 4 \
  --seed    42

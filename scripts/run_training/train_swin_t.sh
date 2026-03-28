#!/bin/bash
# Train Swin-Tiny (patch4_window7_224, IN-22k)
# Transformer — 28M params | img_size 224 | batch_size 24

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/swin_t.yaml \
  --model   swin_t \
  --epochs  50 \
  --batch-size 24 \
  --image-size  224 \
  --process-size 1024 \
  --lr      5e-4 \
  --loss    ce \
  --weight-decay 1e-4 \
  --accumulate-grad-batches 3 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

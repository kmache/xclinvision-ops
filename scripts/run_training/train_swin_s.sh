#!/bin/bash
# Train Swin-Small (patch4_window7_224, IN-22k)
# Transformer — 50M params | img_size 224 | batch_size 16

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/swin_s.yaml \
  --model   swin_s \
  --epochs  50 \
  --batch-size 16 \
  --image-size  224 \
  --process-size 1024 \
  --lr      5e-4 \
  --loss    ce \
  --weight-decay 1e-4 \
  --accumulate-grad-batches 4 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

#!/bin/bash
# Train ViT-Base (patch16_384, IN-21k)
# Transformer — 86M params | img_size 384 | batch_size 6

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/vit_base.yaml \
  --model   vit_base \
  --epochs  50 \
  --batch-size 6 \
  --image-size  384 \
  --process-size 1024 \
  --lr      2e-4 \
  --loss    ce \
  --weight-decay 1e-4 \
  --accumulate-grad-batches 12 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

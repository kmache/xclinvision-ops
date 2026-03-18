#!/bin/bash
# Train ViT-Base (patch16_224)
# Transformer — 86M params | img_size 224 | batch_size 16 | lower lr

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
  --batch-size 16 \
  --image-size  512 \
  --lr      5e-5 \
  --loss    focal \
  --weight-decay 1e-4 \
  --label-smoothing 0.15 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

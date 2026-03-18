#!/bin/bash
# Train ViT-Small (patch16_224)
# Transformer — 22M params | img_size 224 | batch_size 24

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Free GPU VRAM before training
bash "$PROJECT_ROOT/scripts/clean_system.sh"

python3 scripts/train.py \
  --config  configs/vit_small.yaml \
  --model   vit_small \
  --epochs  50 \
  --batch-size 32 \
  --image-size  512 \
  --lr      1e-4 \
  --loss    focal \
  --weight-decay 1e-4 \
  --label-smoothing 0.15 \
  --output-dir models \
  --num-workers 4 \
  --seed    42 \
  "$@"

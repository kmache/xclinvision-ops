#!/bin/bash
# Run full training suite — calls each per-model script in sequence.
# To train a single model, run its script directly:
#   bash scripts/run_train/train_resnet50.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
  echo ""
  echo "================================================================"
  echo "  Training: $1"
  echo "================================================================"
  bash "$SCRIPT_DIR/run_train/$1"
}

echo "Starting model training suite..."

# --- CNNs ---
run train_densenet.sh
run train_resnet50.sh
run train_efficientnet_b0.sh
run train_efficientnet_b2.sh
run train_efficientnet_b3.sh
run train_efficientnet_b4.sh
run train_convnext_tiny.sh
run train_convnext_small.sh

# --- Transformers ---
run train_vit_tiny.sh
run train_vit_small.sh
run train_vit_base.sh
run train_swin_t.sh
run train_swin_s.sh
run train_swin_b.sh

echo ""
echo "All models trained successfully."


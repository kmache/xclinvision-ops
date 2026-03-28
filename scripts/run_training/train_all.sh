#!/bin/bash
# Train ALL models sequentially
# Processes data at 1024x1024 on first run, then trains each model.
# Results saved to models/<model_name>_<timestamp>/
# Logs each model's exit status to outputs/train_all_results.txt

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_FILE="$PROJECT_ROOT/outputs/train_all_results.txt"
mkdir -p "$PROJECT_ROOT/outputs"
echo "=== Training All Models — $(date) ===" > "$RESULTS_FILE"

# Order: small/fast models first, large/slow last
MODELS=(
  train_densenet
  train_resnet50
  train_efficientnet_b0
  train_efficientnet_b2
  train_efficientnet_b3
  train_efficientnet_b4
  train_convnext_tiny
  train_convnext_small
  train_vit_tiny
  train_vit_small
  train_vit_base
  train_swin_t
  train_swin_s
  train_swin_b
)

TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  IDX=$((i + 1))
  echo ""
  echo "================================================================"
  echo "[$IDX/$TOTAL] Training: $MODEL"
  echo "================================================================"

  START_TIME=$(date +%s)

  if bash "$PROJECT_ROOT/scripts/run_training/${MODEL}.sh"; then
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "[$IDX/$TOTAL] $MODEL — SUCCESS (${ELAPSED}s)" | tee -a "$RESULTS_FILE"
    PASSED=$((PASSED + 1))
  else
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "[$IDX/$TOTAL] $MODEL — FAILED (${ELAPSED}s)" | tee -a "$RESULTS_FILE"
    FAILED=$((FAILED + 1))
  fi
done

echo "" | tee -a "$RESULTS_FILE"
echo "=== Summary: $PASSED passed, $FAILED failed out of $TOTAL ===" | tee -a "$RESULTS_FILE"
echo "Results logged to: $RESULTS_FILE"

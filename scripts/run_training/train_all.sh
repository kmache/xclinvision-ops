#!/bin/bash
# Train ALL models sequentially, then evaluate each with threshold optimization.
# Processes data at 1024x1024 on first run, trains each model at 384x384,
# then runs full evaluation with per-class threshold optimization.
# Results saved to models/<model_name>_<timestamp>/ and outputs/evaluation_384/
# Logs each model's exit status to outputs/train_all_results.txt

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_FILE="$PROJECT_ROOT/outputs/train_all_results.txt"
EVAL_DIR="$PROJECT_ROOT/outputs/evaluation_384"
mkdir -p "$PROJECT_ROOT/outputs" "$EVAL_DIR"
echo "=== Training All Models — $(date) ===" > "$RESULTS_FILE"

# Activate conda/mamba environment
eval "$(mamba shell hook --shell bash)"
mamba activate xclinvision_env

# Manifest for the 1024 processed cache (images resize to 384 on-the-fly)
# Auto-detect: use the most recently modified processed_1024 directory
find_manifest() {
  local latest_dir
  latest_dir=$(ls -dt "$PROJECT_ROOT"/data/processed_1024_*/ 2>/dev/null | head -1)
  if [ -n "$latest_dir" ] && [ -f "${latest_dir}manifest.csv" ]; then
    echo "${latest_dir}manifest.csv"
  else
    echo ""
  fi
}

# Model script name → model name mapping (script extracts model name from filename)
MODELS=(
  train_densenet
  train_efficientnet_b0
  train_convnext_small
  train_vit_base
)

# Derive model name from script name: train_<model_name> → <model_name>
get_model_name() {
  echo "${1#train_}"
}

TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  MODEL_NAME=$(get_model_name "$MODEL")
  IDX=$((i + 1))
  echo ""
  echo "================================================================"
  echo "[$IDX/$TOTAL] Training: $MODEL_NAME"
  echo "================================================================"

  START_TIME=$(date +%s)

  if bash "$PROJECT_ROOT/scripts/run_training/${MODEL}.sh"; then
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "[$IDX/$TOTAL] $MODEL_NAME — TRAIN SUCCESS (${ELAPSED}s)" | tee -a "$RESULTS_FILE"
    PASSED=$((PASSED + 1))

    # --- Evaluate the just-trained model ---
    # Find the newest model directory for this architecture
    LATEST_DIR=$(ls -dt "$PROJECT_ROOT/models/${MODEL_NAME}_"*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_DIR" ]; then
      echo "[$IDX/$TOTAL] $MODEL_NAME — EVAL SKIPPED (no model dir found)" | tee -a "$RESULTS_FILE"
      continue
    fi

    # Find best checkpoint (non-last)
    BEST_CKPT=$(ls "$LATEST_DIR"*.ckpt 2>/dev/null | grep -v last.ckpt | head -1)
    if [ -z "$BEST_CKPT" ]; then
      BEST_CKPT="$LATEST_DIR/last.ckpt"
    fi

    echo "[$IDX/$TOTAL] Evaluating $MODEL_NAME from: $BEST_CKPT"
    MANIFEST=$(find_manifest)
    if [ -z "$MANIFEST" ]; then
      echo "[$IDX/$TOTAL] $MODEL_NAME — EVAL SKIPPED (no manifest found)" | tee -a "$RESULTS_FILE"
      continue
    fi
    EVAL_START=$(date +%s)

    if python3 "$PROJECT_ROOT/scripts/evaluate.py" \
      --checkpoint-path "$BEST_CKPT" \
      --model-name "$MODEL_NAME" \
      --image-size 384 \
      --pooling gem \
      --batch-size 16 \
      --manifest "$MANIFEST" \
      --output-dir "$EVAL_DIR" \
      --optimize-thresholds; then
      EVAL_END=$(date +%s)
      EVAL_ELAPSED=$(( EVAL_END - EVAL_START ))
      echo "[$IDX/$TOTAL] $MODEL_NAME — EVAL SUCCESS (${EVAL_ELAPSED}s)" | tee -a "$RESULTS_FILE"
    else
      EVAL_END=$(date +%s)
      EVAL_ELAPSED=$(( EVAL_END - EVAL_START ))
      echo "[$IDX/$TOTAL] $MODEL_NAME — EVAL FAILED (${EVAL_ELAPSED}s)" | tee -a "$RESULTS_FILE"
    fi
  else
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "[$IDX/$TOTAL] $MODEL_NAME — TRAIN FAILED (${ELAPSED}s)" | tee -a "$RESULTS_FILE"
    FAILED=$((FAILED + 1))
  fi
done

echo "" | tee -a "$RESULTS_FILE"
echo "=== Summary: $PASSED passed, $FAILED failed out of $TOTAL ===" | tee -a "$RESULTS_FILE"
echo "=== Evaluation reports: $EVAL_DIR ===" | tee -a "$RESULTS_FILE"
echo "Results logged to: $RESULTS_FILE"

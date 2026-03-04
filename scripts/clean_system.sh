#!/bin/bash
# clean_system.sh — Free GPU VRAM before training by killing zombie processes.
# Safe to run at any time. Only kills processes matching scripts/train.py.
#
# Usage:
#   bash scripts/clean_system.sh
#   bash scripts/clean_system.sh --quiet   (suppress output)

QUIET=false
[[ "${1}" == "--quiet" ]] && QUIET=true

log() { [[ "$QUIET" == false ]] && echo "$@"; }

log ""
log "=== XClinVision System Cleanup ==="

# ── 1. Kill any lingering train.py processes ───────────────────────────────
PIDS=$(pgrep -f "scripts/train.py" 2>/dev/null)

if [[ -n "$PIDS" ]]; then
    log "Found zombie training processes: $PIDS"
    pkill -9 -f "scripts/train.py"
    sleep 2
    log "Killed."
else
    log "No lingering train.py processes found."
fi

# ── 2. Report VRAM status ──────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    log "GPU VRAM — used: ${VRAM_USED} MiB | free: ${VRAM_FREE} MiB | total: ${VRAM_TOTAL} MiB"

    # Warn if more than 2 GB is still occupied (likely a non-training process)
    if (( VRAM_USED > 2048 )); then
        echo "[WARN] More than 2 GB still in use. Check nvidia-smi for non-training processes."
    fi
else
    log "nvidia-smi not found — skipping VRAM check."
fi

log "=== Cleanup done. Ready to train. ==="
log ""

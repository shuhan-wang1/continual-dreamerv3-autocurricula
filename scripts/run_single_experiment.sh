#!/bin/bash
# Run a single Craftax experiment (e.g. overnight).
# Usage: bash scripts/run_single_experiment.sh [baseline|soft|hard]
# Default: soft mask, 500k steps

MODE="${1:-soft}"
STEPS="${STEPS:-500000}"
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs/action_mask_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${MODE}.log"
echo "Mode=$MODE steps=$STEPS log=$LOG_FILE"

if [ "$MODE" = "baseline" ]; then
  python train.py --env_type craftax --steps "$STEPS" --tag baseline_ --logdir "$LOG_DIR" 2>&1 | tee "$LOG_FILE"
else
  python train.py --env_type craftax --action_mask_enabled --action_mask_mode "$MODE" \
    --steps "$STEPS" --tag "${MODE}_" --logdir "$LOG_DIR" 2>&1 | tee "$LOG_FILE"
fi

echo "Done. Log: $LOG_FILE"
ls -la "$LOG_DIR"

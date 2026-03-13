#!/bin/bash
# Run Craftax action mask experiments (baseline, soft, hard).
# Usage: bash scripts/run_action_mask_experiments.sh
# Or in screen: screen -S craftax; bash scripts/run_action_mask_experiments.sh

set -e
cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-logs/action_mask_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-500000}"
mkdir -p "$LOG_DIR"
echo "Logs: $LOG_DIR"

# 1. Baseline (no mask)
echo "=== 1/3 Baseline (no mask) ==="
python train.py --env_type craftax --steps "$STEPS" --tag baseline_ \
  --logdir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/baseline.log"

# 2. Soft mask
echo "=== 2/3 Soft mask ==="
python train.py --env_type craftax --action_mask_enabled --action_mask_mode soft \
  --steps "$STEPS" --tag soft_ --logdir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/soft.log"

# 3. Hard mask
echo "=== 3/3 Hard mask ==="
python train.py --env_type craftax --action_mask_enabled --action_mask_mode hard \
  --steps "$STEPS" --tag hard_ --logdir "$LOG_DIR" 2>&1 | tee "$LOG_DIR/hard.log"

echo "Done. Results in $LOG_DIR"
python scripts/summarize_action_mask_results.py "$LOG_DIR" 2>/dev/null || true

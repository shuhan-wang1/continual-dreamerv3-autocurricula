#!/bin/bash
# =============================================================
# G2: A0 baseline (50:50 reservoir+recency) @ 10M steps, 3 seeds
# Control experiment: same as A0 but 10x longer training.
# Target: AutoDL consumer GPU (RTX 3090/4090, 24GB VRAM)
# Estimated: ~24h/seed on RTX 4090, ~72h total sequential
# =============================================================

set -euo pipefail

# --- Activate conda ---
# Adjust this to match your AutoDL conda setup
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: Cannot find conda. Set your conda path."; exit 1; }
conda activate dreamer

PYTHON=$(which python)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH:-}"

mkdir -p logs experiment_results/10m

echo "============================================"
echo "  G2: A0 baseline @ 10M steps"
echo "  50:50 reservoir+recency, no P2E, no intrinsic"
echo "  (AutoDL)"
echo "============================================"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown') | Start: $(date)"
nvidia-smi
echo "---"

# --- Hyperparameters (same as Myriad for reproducibility) ---
# Default recent_frac=0.5 gives 50:50 sampling.
COMMON="--steps 10000000 --batch_size 48 --batch_length 64 --envs 64 --model_size 25m --wandb_proj_name craftax-10m --wandb_mode online --resume"
NO_P2E="--no_plan2explore"

FLAGS="$COMMON $NO_P2E"

FAILED=0
COMPLETED=0
TOTAL=3
START_TIME=$SECONDS

run_experiment() {
    local NAME=$1
    shift
    echo ""
    echo "======================================================================"
    echo "  [$((COMPLETED + FAILED + 1))/$TOTAL] $NAME"
    echo "  Start: $(date)"
    echo "======================================================================"

    $PYTHON train_craftax.py "$@"
    local RC=$?

    if [ $RC -eq 0 ]; then
        COMPLETED=$((COMPLETED + 1))
        echo "  [OK] $NAME completed (exit $RC)"
    else
        FAILED=$((FAILED + 1))
        echo "  [FAIL] $NAME failed (exit $RC)"
    fi
    return $RC
}

run_experiment "G2_baseline_10m_seed1" \
    --seed 1 --tag G2_baseline_10m --logdir experiment_results/10m/G2_baseline_10m_seed1 \
    $FLAGS --wandb_group G2_baseline_10m

run_experiment "G2_baseline_10m_seed4" \
    --seed 4 --tag G2_baseline_10m --logdir experiment_results/10m/G2_baseline_10m_seed4 \
    $FLAGS --wandb_group G2_baseline_10m

run_experiment "G2_baseline_10m_seed42" \
    --seed 42 --tag G2_baseline_10m --logdir experiment_results/10m/G2_baseline_10m_seed42 \
    $FLAGS --wandb_group G2_baseline_10m

# --- Summary ---
ELAPSED=$((SECONDS - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "======================================================================"
echo "  G2 COMPLETE (AutoDL)"
echo "======================================================================"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed:    $FAILED / $TOTAL"
echo "  Duration:  ${HOURS}h ${MINS}m"
echo "  End:       $(date)"
echo "======================================================================"

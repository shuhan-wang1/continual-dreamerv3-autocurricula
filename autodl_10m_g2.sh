#!/bin/bash
# =============================================================
# G2: A0 baseline (50:50 reservoir+recency) @ 10M steps, 3 seeds
# Control experiment: same as A0 but 10x longer training.
# Target: AutoDL consumer GPU (RTX 5090, 32GB VRAM)
#
# Resume-safe: re-run this script after interruption and it will
#   - skip seeds that already finished (DONE marker)
#   - resume in-progress seeds from checkpoint (--resume)
#   - retry failed seeds up to MAX_RETRIES times
# =============================================================

set -euo pipefail

# --- Activate conda ---
# Adjust this to match your AutoDL conda setup
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: Cannot find conda. Set your conda path."; exit 1; }
conda activate dreamer_cuda13

PYTHON=$(which python)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_force_compilation_parallelism=1"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH:-}"

mkdir -p logs experiment_results/10m

MAX_RETRIES=3

echo "============================================"
echo "  G2: A0 baseline @ 10M steps"
echo "  50:50 reservoir+recency, no P2E, no intrinsic"
echo "  Resume-safe (max $MAX_RETRIES retries/seed)"
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

SKIPPED=0
FAILED=0
COMPLETED=0
TOTAL=3
START_TIME=$SECONDS

run_experiment() {
    local NAME=$1
    local LOGDIR=$2
    shift 2

    echo ""
    echo "======================================================================"
    echo "  [$((COMPLETED + SKIPPED + FAILED + 1))/$TOTAL] $NAME"
    echo "======================================================================"

    # Skip if already completed
    if [ -f "$LOGDIR/DONE" ]; then
        echo "  [SKIP] $NAME already completed (DONE marker found)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    # Retry loop
    local ATTEMPT=0
    while [ $ATTEMPT -lt $MAX_RETRIES ]; do
        ATTEMPT=$((ATTEMPT + 1))
        echo "  Attempt $ATTEMPT/$MAX_RETRIES | Start: $(date)"

        $PYTHON train_craftax.py "$@"
        local RC=$?

        if [ $RC -eq 0 ]; then
            touch "$LOGDIR/DONE"
            COMPLETED=$((COMPLETED + 1))
            echo "  [OK] $NAME completed (exit $RC)"
            return 0
        fi

        echo "  [WARN] $NAME attempt $ATTEMPT failed (exit $RC)"
        if [ $ATTEMPT -lt $MAX_RETRIES ]; then
            echo "  Retrying in 10s..."
            sleep 10
        fi
    done

    FAILED=$((FAILED + 1))
    echo "  [FAIL] $NAME failed after $MAX_RETRIES attempts"
    return 1
}

run_experiment "G2_baseline_10m_seed1" "experiment_results/10m/G2_baseline_10m_seed1" \
    --seed 1 --tag G2_baseline_10m --logdir experiment_results/10m/G2_baseline_10m_seed1 \
    $FLAGS --wandb_group G2_baseline_10m

run_experiment "G2_baseline_10m_seed4" "experiment_results/10m/G2_baseline_10m_seed4" \
    --seed 4 --tag G2_baseline_10m --logdir experiment_results/10m/G2_baseline_10m_seed4 \
    $FLAGS --wandb_group G2_baseline_10m

run_experiment "G2_baseline_10m_seed42" "experiment_results/10m/G2_baseline_10m_seed42" \
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
echo "  Skipped:   $SKIPPED / $TOTAL"
echo "  Failed:    $FAILED / $TOTAL"
echo "  Duration:  ${HOURS}h ${MINS}m"
echo "  End:       $(date)"
echo "======================================================================"

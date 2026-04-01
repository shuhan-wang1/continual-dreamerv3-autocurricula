#!/bin/bash
# =============================================================
# G4: soft_mask only (no Intrinsic, no NLU)  @  10M steps, 3 seeds
# Ablation: isolate the contribution of soft action masking alone,
# without intrinsic rewards or NLU sampling.
# Target: AutoDL consumer GPU (RTX 5090, 32GB VRAM)
#
# Resume-safe: re-run this script after interruption and it will
#   - skip seeds that already finished (DONE marker)
#   - resume in-progress seeds from checkpoint (--resume)
#   - retry failed seeds up to MAX_RETRIES times
# =============================================================

set -euo pipefail

# --- Activate conda ---
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: Cannot find conda. Set your conda path."; exit 1; }
conda activate dreamer_cuda13

PYTHON=$(which python)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Use BFC allocator with preallocation — 'platform' allocator causes
# CUDA_ERROR_ILLEGAL_ADDRESS from memory fragmentation on long runs.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

# RTX 5090 Blackwell workaround: disable command buffers (CUDA graphs)
# and compilation parallelism (jax-ml/jax#33910, jax-ml/jax#34696).
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_force_compilation_parallelism=1"

export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH:-}"

mkdir -p logs experiment_results/10m

MAX_RETRIES=3

echo "============================================"
echo "  G4: soft_mask only (no Intrinsic, no NLU)"
echo "  10M steps, 3 seeds (AutoDL)"
echo "  Resume-safe (max $MAX_RETRIES retries/seed)"
echo "============================================"
echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown') | Start: $(date)"
nvidia-smi
echo "---"

# --- Hyperparameters ---
COMMON="--steps 10000000 --batch_size 48 --batch_length 64 --envs 64 --model_size 25m --wandb_proj_name craftax-10m --wandb_mode online --resume"
# Soft mask only
MASK="--action_mask_enabled --action_mask_mode soft --action_mask_lambda_penalty 5.0"
# No intrinsic rewards
# No NLU sampling
# No P2E (harmful)
NO_P2E="--no_plan2explore"

FLAGS="$COMMON $MASK $NO_P2E"

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

run_experiment "G4_mask_only_seed1" "experiment_results/10m/G4_mask_only_seed1" \
    --seed 1 --tag G4_mask_only --logdir experiment_results/10m/G4_mask_only_seed1 \
    $FLAGS --wandb_group G4_mask_only

run_experiment "G4_mask_only_seed4" "experiment_results/10m/G4_mask_only_seed4" \
    --seed 4 --tag G4_mask_only --logdir experiment_results/10m/G4_mask_only_seed4 \
    $FLAGS --wandb_group G4_mask_only

run_experiment "G4_mask_only_seed42" "experiment_results/10m/G4_mask_only_seed42" \
    --seed 42 --tag G4_mask_only --logdir experiment_results/10m/G4_mask_only_seed42 \
    $FLAGS --wandb_group G4_mask_only

# --- Summary ---
ELAPSED=$((SECONDS - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "======================================================================"
echo "  G4 COMPLETE (AutoDL)"
echo "======================================================================"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Skipped:   $SKIPPED / $TOTAL"
echo "  Failed:    $FAILED / $TOTAL"
echo "  Duration:  ${HOURS}h ${MINS}m"
echo "  End:       $(date)"
echo "======================================================================"

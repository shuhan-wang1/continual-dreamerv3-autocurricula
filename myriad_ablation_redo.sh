#!/bin/bash -l
# =============================================================
# Ablation REDO: 7 runs broken by deleted W&B runs (409 conflict)
#
# Affected: A2_p2e (3 seeds), D1_nlr (3 seeds), D2_nlu seed1
#
# FIX: Omit --resume so wandb.init() gets id=None (fresh random
#      ID) instead of the poisoned "{group}_{tag}" ID that W&B
#      refuses to recreate after deletion.
#      No checkpoints exist for these runs, so nothing to resume.
#
# Estimated time: 7 runs x ~1.5h = ~10.5h
# =============================================================

#$ -N abl-redo
#$ -l h_rt=15:59:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l gpu=1
#$ -ac allow=LUV
#$ -o /home/ucab327/Scratch/projects/continual-dreamerv3-autocurricula/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/ucab327/Scratch/projects/continual-dreamerv3-autocurricula/logs/$JOB_NAME.$JOB_ID.err

# --- Load modules ---
module unload compilers mpi gcc-libs 2>/dev/null || true
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load python/miniconda3/4.10.3
module load cudnn/9.2.0.82/cuda-12
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate dreamer311

CONDA_ENV_PREFIX="$HOME/.conda/envs/dreamer311"
if [ "$(which python 2>/dev/null)" != "${CONDA_ENV_PREFIX}/bin/python" ]; then
    echo "WARNING: conda activate did not work, using absolute path"
    export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
fi
PYTHON="${CONDA_ENV_PREFIX}/bin/python"

PROJECT_DIR="$HOME/Scratch/projects/continual-dreamerv3-autocurricula"
cd $PROJECT_DIR

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH}"
export WANDB_API_KEY="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"

mkdir -p logs experiment_results/ablation

echo "============================================"
echo "  Ablation REDO: 7 broken W&B runs"
echo "  A2_p2e (3), D1_nlr (3), D2_nlu_seed1 (1)"
echo "============================================"
echo "Job ID: $JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi
echo "---"

# Common args (no --resume: fresh W&B IDs, no checkpoints to load anyway)
COMMON="--steps 1000000 --batch_size 48 --batch_length 64 --envs 64 --model_size 25m --wandb_proj_name craftax-ablation --wandb_mode online"

FAILED=0
COMPLETED=0
START_TIME=$SECONDS

run_experiment() {
    local NAME=$1
    shift
    echo ""
    echo "======================================================================"
    echo "  [$((COMPLETED + FAILED + 1))/7] $NAME"
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

# --- A2_p2e: Plan2Explore only (no extrinsic reward) ---
run_experiment "A2_p2e_seed1" \
    --seed 1 --tag A2_p2e --logdir experiment_results/ablation/A2_p2e_seed1 \
    $COMMON --wandb_group A2_p2e --plan2explore

run_experiment "A2_p2e_seed4" \
    --seed 4 --tag A2_p2e --logdir experiment_results/ablation/A2_p2e_seed4 \
    $COMMON --wandb_group A2_p2e --plan2explore

run_experiment "A2_p2e_seed42" \
    --seed 42 --tag A2_p2e --logdir experiment_results/ablation/A2_p2e_seed42 \
    $COMMON --wandb_group A2_p2e --plan2explore

# --- D1_nlr: NLR non-privileged ---
run_experiment "D1_nlr_seed1" \
    --seed 1 --tag D1_nlr --logdir experiment_results/ablation/D1_nlr_seed1 \
    $COMMON --wandb_group D1_nlr --no_plan2explore --nlr_sampling

run_experiment "D1_nlr_seed4" \
    --seed 4 --tag D1_nlr --logdir experiment_results/ablation/D1_nlr_seed4 \
    $COMMON --wandb_group D1_nlr --no_plan2explore --nlr_sampling

run_experiment "D1_nlr_seed42" \
    --seed 42 --tag D1_nlr --logdir experiment_results/ablation/D1_nlr_seed42 \
    $COMMON --wandb_group D1_nlr --no_plan2explore --nlr_sampling

# --- D2_nlu seed1: NLU non-privileged (seeds 4,42 already done) ---
run_experiment "D2_nlu_seed1" \
    --seed 1 --tag D2_nlu --logdir experiment_results/ablation/D2_nlu_seed1 \
    $COMMON --wandb_group D2_nlu --no_plan2explore --nlu_sampling

# --- Summary ---
ELAPSED=$((SECONDS - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "======================================================================"
echo "  REDO COMPLETE"
echo "======================================================================"
echo "  Completed: $COMPLETED / 7"
echo "  Failed:    $FAILED / 7"
echo "  Duration:  ${HOURS}h ${MINS}m"
echo "  End:       $(date)"
echo "======================================================================"

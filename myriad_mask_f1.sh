#!/bin/bash -l
# =============================================================
# F1_mask_soft: 3 seeds (seed1 resumes from ~764k checkpoint)
# With GPU-batched mask context fix — ~2h/seed, ~6h total
# =============================================================

#$ -N mask-f1
#$ -l h_rt=7:59:00
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
echo "  F1_mask_soft: 3 seeds"
echo "============================================"
echo "Job ID: $JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi
echo "---"

COMMON="--steps 1000000 --batch_size 48 --batch_length 64 --envs 64 --model_size 25m --wandb_proj_name craftax-ablation --wandb_mode online --resume"
MASK_SOFT="--no_plan2explore --no_intrinsic_spatial --action_mask_enabled --action_mask_mode soft --action_mask_lambda_penalty 5.0"

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

run_experiment "F1_mask_soft_seed1" \
    --seed 1 --tag F1_mask_soft --logdir experiment_results/ablation/F1_mask_soft_seed1 \
    $COMMON --wandb_group F1_mask_soft $MASK_SOFT

run_experiment "F1_mask_soft_seed4" \
    --seed 4 --tag F1_mask_soft --logdir experiment_results/ablation/F1_mask_soft_seed4 \
    $COMMON --wandb_group F1_mask_soft $MASK_SOFT

run_experiment "F1_mask_soft_seed42" \
    --seed 42 --tag F1_mask_soft --logdir experiment_results/ablation/F1_mask_soft_seed42 \
    $COMMON --wandb_group F1_mask_soft $MASK_SOFT

# --- Summary ---
ELAPSED=$((SECONDS - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "======================================================================"
echo "  F1_mask_soft COMPLETE"
echo "======================================================================"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed:    $FAILED / $TOTAL"
echo "  Duration:  ${HOURS}h ${MINS}m"
echo "  End:       $(date)"
echo "======================================================================"

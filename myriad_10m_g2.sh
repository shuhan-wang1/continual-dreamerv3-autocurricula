#!/bin/bash -l
# =============================================================
# G2: A0 baseline (50:50 reservoir+recency) @ 10M steps, 3 seeds
# Control experiment: same as A0 but 10x longer training.
# Estimated: ~15h/seed, ~45h total sequential
# =============================================================

#$ -N g2-baseline-10m
#$ -l h_rt=47:59:00
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

mkdir -p logs experiment_results/10m

echo "============================================"
echo "  G2: A0 baseline @ 10M steps"
echo "  50:50 reservoir+recency, no P2E, no intrinsic"
echo "============================================"
echo "Job ID: $JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi
echo "---"

# Same as A0 but 10M steps. Default recent_frac=0.5 gives 50:50 sampling.
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
echo "  G2 COMPLETE"
echo "======================================================================"
echo "  Completed: $COMPLETED / $TOTAL"
echo "  Failed:    $FAILED / $TOTAL"
echo "  Duration:  ${HOURS}h ${MINS}m"
echo "  End:       $(date)"
echo "======================================================================"

#!/bin/bash -l
# =============================================================
# UCL Myriad Ablation Job Script (A100, 1B steps per experiment)
# Usage:   qsub -v "EXP_ID=A0_5050_baseline,SEED=1" myriad_ablation_job.sh
#          (or use submit_myriad_ablation.py to submit all jobs)
#
# This script runs ONE experiment with ONE seed.
# It auto-resubmits when walltime expires so training resumes
# from the last checkpoint until 1B steps are complete.
# =============================================================

# --- Job configuration (A100-40G optimised) ---
#$ -l h_rt=5:59:00            # 6h walltime per job submission
#$ -l mem=16G                  # RAM per core (16G x 4 cores = 64G total)
#$ -pe smp 4                   # CPU cores
#$ -l gpu=1                    # Number of GPUs
#$ -ac allow=L                 # GPU type: L=A100-40G
#$ -o /home/ucab327/Scratch/projects/continual-dreamerv3-autocurricula/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/ucab327/Scratch/projects/continual-dreamerv3-autocurricula/logs/$JOB_NAME.$JOB_ID.err

# --- Validate parameters ---
if [ -z "$EXP_ID" ] || [ -z "$SEED" ]; then
    echo "ERROR: Must set EXP_ID and SEED via qsub -v"
    echo "Usage: qsub -v \"EXP_ID=A0_5050_baseline,SEED=1\" myriad_ablation_job.sh"
    exit 1
fi

# --- Load modules ---
module unload compilers mpi gcc-libs 2>/dev/null || true
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load python/miniconda3/4.10.3
module load cudnn/9.2.0.82/cuda-12
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate dreamer311

# --- Resolve conda env python (fallback to absolute path if activate fails) ---
CONDA_ENV_PREFIX="$HOME/.conda/envs/dreamer311"
if [ "$(which python 2>/dev/null)" != "${CONDA_ENV_PREFIX}/bin/python" ]; then
    echo "WARNING: conda activate did not work, using absolute path"
    export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
fi
PYTHON="${CONDA_ENV_PREFIX}/bin/python"

# --- Set working directory ---
PROJECT_DIR="$HOME/Scratch/projects/continual-dreamerv3-autocurricula"
cd $PROJECT_DIR

# --- Environment variables (A100 optimised) ---
# Use on-demand allocation so JIT compilation has full VRAM headroom.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH}"
export WANDB_API_KEY="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"

# --- Ensure directories exist ---
mkdir -p logs
mkdir -p experiment_results/ablation

# --- Print job info ---
echo "============================================"
echo "  Ablation Job: ${EXP_ID} seed=${SEED}"
echo "============================================"
echo "Job ID:       $JOB_ID"
echo "Job Name:     $JOB_NAME"
echo "Node:         $(hostname)"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Start Time:   $(date)"
echo "Working Dir:  $(pwd)"
echo "Python:       $PYTHON ($($PYTHON --version 2>&1))"
nvidia-smi
echo "---"

# --- Build training command from experiment definitions ---
# Uses run_ablation.py's experiment registry to avoid duplicating configs.
TRAIN_ARGS=$($PYTHON -c "
import sys
sys.path.insert(0, '.')
from run_ablation import EXPERIMENTS, build_command
exp_id = '${EXP_ID}'
seed = ${SEED}
if exp_id not in EXPERIMENTS:
    print(f'ERROR: Unknown experiment ID: {exp_id}', file=sys.stderr)
    print(f'Available: {list(EXPERIMENTS.keys())}', file=sys.stderr)
    sys.exit(1)
cmd, logdir = build_command(exp_id, EXPERIMENTS[exp_id], seed, 'experiment_results/ablation', 'online')
# Print only the training arguments (skip 'python' and 'train_craftax.py')
print(' '.join(cmd[2:]))
")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build training command for EXP_ID=${EXP_ID}, SEED=${SEED}"
    exit 1
fi

echo "Training command:"
echo "  $PYTHON train_craftax.py ${TRAIN_ARGS}"
echo "---"

# --- Run training ---
$PYTHON train_craftax.py ${TRAIN_ARGS}

EXIT_CODE=$?
echo "---"
echo "End Time:   $(date)"
echo "Exit Code:  $EXIT_CODE"

# --- Auto-resubmit if training is not finished ---
# Exit code 0 = completed all 1B steps. Non-zero = interrupted by walltime.
# Only resubmit if a checkpoint exists (training made progress).
if [ $EXIT_CODE -ne 0 ]; then
    LOGDIR="experiment_results/ablation/${EXP_ID}_seed${SEED}"
    # Find checkpoint in nested DreamerV3 logdir
    CKPT_EXISTS=false
    for d in ${LOGDIR}/*/ckpt ${LOGDIR}/ckpt; do
        if [ -d "$d" ]; then
            CKPT_EXISTS=true
            break
        fi
    done

    if [ "$CKPT_EXISTS" = true ]; then
        echo "Checkpoint found. Auto-resubmitting ${EXP_ID} seed=${SEED}..."
        qsub -N "abl-${EXP_ID}-s${SEED}" -v "EXP_ID=${EXP_ID},SEED=${SEED}" $0
        echo "Resubmitted: $0"
    else
        echo "WARNING: No checkpoint found in ${LOGDIR}. NOT resubmitting."
        echo "Training may have failed before producing a checkpoint."
    fi
fi

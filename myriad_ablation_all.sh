#!/bin/bash -l
# =============================================================
# UCL Myriad: Run ALL ablation experiments on 1 A100 (1M steps)
#
# Usage:   qsub myriad_ablation_all.sh
# Monitor: qstat
#          tail -f logs/ablation-all.$JOB_ID.out
#
# 14 experiments × 3 seeds = 42 runs @ ~17 min each ≈ ~12h
# Completed runs are auto-skipped; incomplete runs resume from checkpoint.
# =============================================================

#$ -N ablation-all
#$ -l h_rt=47:59:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l gpu=1
#$ -ac allow=L
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
# Preallocate + fixed fraction starves the XLA compiler of temp memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH}"
export WANDB_API_KEY="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"

# --- Ensure directories exist ---
mkdir -p logs
mkdir -p experiment_results/ablation

# --- Print job info ---
echo "============================================"
echo "  Ablation Suite: ALL experiments (15 configs × 3 seeds = 45 runs, 1M steps)"
echo "============================================"
echo "Job ID:       $JOB_ID"
echo "Node:         $(hostname)"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Start Time:   $(date)"
echo "Python:       $PYTHON ($($PYTHON --version 2>&1))"
nvidia-smi
echo "---"

# --- Run all experiments sequentially ---
$PYTHON run_ablation.py --wandb_mode online

echo "---"
echo "End Time:   $(date)"
echo "Exit Code:  $?"

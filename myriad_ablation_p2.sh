#!/bin/bash -l
# =============================================================
# Ablation Job 2/2: A1, A3, B1, B2, D3, D4, F2
# 7 configs × 3 seeds = 21 runs, ~35h
# =============================================================

#$ -N abl-p2
#$ -l h_rt=35:59:00
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
echo "  Ablation P2: A1,A3,B1,B2,D3,D4,F2 (21 runs)"
echo "============================================"
echo "Job ID: $JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi
echo "---"

$PYTHON run_ablation.py \
    --only A1_uniform_baseline,A3_intrinsic,B1_spatial_only,B2_craft_only,D3_nlr_priv,D4_nlu_priv,F2_mask_hard \
    --wandb_mode online

echo "--- End: $(date) | Exit: $?"

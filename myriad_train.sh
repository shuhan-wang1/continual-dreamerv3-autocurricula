#!/bin/bash -l
# =============================================================
# UCL Myriad GPU Job Submission Script
# Usage:   qsub myriad_train.sh
# Monitor: qstat
# Delete:  qdel <job_id>
# =============================================================

# --- Job configuration ---
#$ -l h_rt=47:59:00           # Max walltime (just under 48h limit)
#$ -l mem=32G                  # RAM per core
#$ -pe smp 4                   # CPU cores
#$ -l gpu=1                    # Number of GPUs
#$ -ac allow=L                 # GPU type: L=A100-40G, EF=V100, UV=A100-80G
#$ -N dreamerv3                # Job name
#$ -o $HOME/logs/$JOB_NAME.$JOB_ID.out
#$ -e $HOME/logs/$JOB_NAME.$JOB_ID.err

# --- Create log directory ---
mkdir -p $HOME/logs

# --- Load modules ---
module unload compilers mpi gcc-libs 2>/dev/null || true
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate dreamer

# --- Set working directory ---
PROJECT_DIR="$HOME/continual-dreamerv3-autocurricula"
cd $PROJECT_DIR

# --- Environment variables ---
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH}"

# --- Print job info ---
echo "Job ID:       $JOB_ID"
echo "Job Name:     $JOB_NAME"
echo "Node:         $(hostname)"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Start Time:   $(date)"
echo "Working Dir:  $(pwd)"
nvidia-smi
echo "---"

# --- Run training ---
# Modify arguments as needed
python train_craftax.py \
    --seed 1 \
    --wandb_mode online

echo "End Time: $(date)"

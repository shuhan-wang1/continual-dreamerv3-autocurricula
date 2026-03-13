#!/bin/bash -l
# =============================================================
# UCL Myriad GPU Job Submission Script (A100, 1B steps)
# Usage:   qsub myriad_train.sh
# Monitor: qstat
# Delete:  qdel <job_id>
#
# 1B steps cannot finish in one 48h job. This script auto-resubmits
# itself when time is running out. DreamerV3 checkpoints enable
# seamless resumption.
# =============================================================

# --- Job configuration (A100-40G optimised) ---
#$ -l h_rt=47:59:00           # Max walltime (just under 48h limit)
#$ -l mem=16G                  # RAM per core (16G x 4 cores = 64G total)
#$ -pe smp 4                   # CPU cores
#$ -l gpu=1                    # Number of GPUs
#$ -ac allow=L                 # GPU type: L=A100-40G (safe for 25m model)
#$ -N dreamerv3                # Job name
#$ -o /home/ucab327/Scratch/projects/continual-dreamerv3-autocurricula/logs/$JOB_NAME.$JOB_ID.out
#$ -e /home/ucab327/Scratch/projects/continual-dreamerv3-autocurricula/logs/$JOB_NAME.$JOB_ID.err

# --- Load modules ---
module unload compilers mpi gcc-libs 2>/dev/null || true
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load python/miniconda3/4.10.3
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate dreamer311

# --- Set working directory ---
PROJECT_DIR="$HOME/continual-dreamerv3-autocurricula"
cd $PROJECT_DIR

# --- Environment variables (A100 optimised, prevent OOM) ---
# Pre-allocate 80% of GPU memory upfront to avoid fragmentation OOM.
# train_craftax.py sets XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 internally,
# so this is a fallback in case that code path is skipped.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH}"

# --- Print job info ---
echo "Job ID:       $JOB_ID"
echo "Job Name:     $JOB_NAME"
echo "Node:         $(hostname)"
echo "GPUs:         $CUDA_VISIBLE_DEVICES"
echo "Start Time:   $(date)"
echo "Working Dir:  $(pwd)"
echo "Python:       $(which python) ($(python --version 2>&1))"
nvidia-smi
echo "---"

# --- Run training (1B steps, checkpoint resume enabled) ---
# --resume: continue from last checkpoint (critical for multi-job 1B runs)
# --envs 32: A100 can handle 32 parallel envs with 25m model
# --model_size 25m: ~4GB VRAM, safe on A100-40G with room to spare
python train_craftax.py \
    --seed 1 \
    --steps 1000000000 \
    --envs 32 \
    --batch_size 16 \
    --batch_length 64 \
    --model_size 25m \
    --resume \
    --wandb_mode online

EXIT_CODE=$?
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"

# --- Auto-resubmit if training is not finished ---
# Check if we hit the walltime limit (training should save checkpoint before exit).
# If exit code is 0, training completed all 1B steps — no resubmit needed.
if [ $EXIT_CODE -ne 0 ]; then
    echo "Training did not complete (exit code $EXIT_CODE). Auto-resubmitting..."
    qsub $0
    echo "Resubmitted: $0"
fi

#!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt=1:00:00
#$ -l mem=8G
#$ -pe smp 1
#$ -N plot_neurips_figs
#$ -wd /home/ucab327/Scratch/projects/continual-dreamerv3-autocurricula
#$ -o logs/plot_figures.$JOB_ID.out
#$ -e logs/plot_figures.$JOB_ID.err

# CPU-only job for generating NeurIPS figures from ablation experiment logs.
# Uses dreamer311 conda env (has matplotlib).
#
# Submit:  qsub myriad_plot_figures.sh

set -euo pipefail

PROJECT_DIR="/home/ucab327/Scratch/projects/continual-dreamerv3-autocurricula"
CONDA_ENV_PREFIX="$HOME/.conda/envs/dreamer311"
PYTHON="${CONDA_ENV_PREFIX}/bin/python"

# Load modules
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load python/miniconda3/4.10.3

echo "=== Plot NeurIPS Figures ==="
echo "Date:    $(date)"
echo "Host:    $(hostname)"
echo "Job ID:  ${JOB_ID:-local}"
echo "Python:  ${PYTHON}"
echo ""

# Verify matplotlib is available
${PYTHON} -c "import matplotlib; print('matplotlib', matplotlib.__version__)"

# Generate all figures
cd "${PROJECT_DIR}"
${PYTHON} experiment_results/plot_neurips_figures.py \
    --results_dir experiment_results/ablation \
    --output_dir experiment_results/figures

echo ""
echo "=== Completed at $(date) ==="
ls -lh experiment_results/figures/

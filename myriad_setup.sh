#!/bin/bash -l
# =============================================================
# UCL Myriad Environment Setup Script
# Project: continual-dreamerv3-autocurricula
# Usage:   bash myriad_setup.sh
#
# IMPORTANT: Run this as "bash myriad_setup.sh", NOT "sh myriad_setup.sh"
# If conda activate fails, try: source myriad_setup.sh
# =============================================================

set -e

ENV_NAME="dreamer"
PYTHON_VERSION="3.10"

echo "============================================"
echo "  Myriad Environment Setup: ${ENV_NAME}"
echo "============================================"

# ----------------------------------------------------------
# Step 1: Unload default modules that may conflict
# ----------------------------------------------------------
echo "[1/7] Unloading default modules..."
module unload compilers mpi gcc-libs 2>/dev/null || true

# ----------------------------------------------------------
# Step 2: Load required modules
# ----------------------------------------------------------
echo "[2/7] Loading modules..."
module load gcc-libs/10.2.0
module load python/miniconda3/4.10.3

# Activate conda — try both common paths
if [ -n "$UCL_CONDA_PATH" ]; then
    source $UCL_CONDA_PATH/etc/profile.d/conda.sh
elif [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: Cannot find conda.sh. Check 'module avail python' for available versions."
    exit 1
fi

# ----------------------------------------------------------
# Step 3: Create conda environment with Python 3.10
# ----------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[3/7] Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[3/7] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENV_NAME}

# ----------------------------------------------------------
# Step 3.5: Verify we are using the CONDA python, not system
# ----------------------------------------------------------
echo "[3.5/7] Verifying conda environment..."
ACTUAL_PYTHON=$(which python)
ACTUAL_PIP=$(which pip)
ACTUAL_VERSION=$(python --version 2>&1)
echo "  Python binary:  ${ACTUAL_PYTHON}"
echo "  Pip binary:     ${ACTUAL_PIP}"
echo "  Python version: ${ACTUAL_VERSION}"

# Abort if pip/python is NOT from conda env
if [[ "$ACTUAL_PIP" != *"envs/${ENV_NAME}"* ]]; then
    echo ""
    echo "ERROR: pip is NOT from the conda env!"
    echo "  Expected path containing: envs/${ENV_NAME}"
    echo "  Got: ${ACTUAL_PIP}"
    echo ""
    echo "FIX: Run this script with 'source' instead of 'bash':"
    echo "  source myriad_setup.sh"
    echo ""
    echo "Or manually activate first, then run:"
    echo "  conda activate ${ENV_NAME}"
    echo "  bash myriad_setup.sh"
    exit 1
fi

# ----------------------------------------------------------
# Step 4: Install JAX with CUDA 12 (pip-bundled CUDA libs)
# ----------------------------------------------------------
echo "[4/7] Installing JAX with CUDA 12..."
pip install --upgrade pip
# JAX 0.6.2 requires nvidia-cudnn-cu12>=9.8 which is unavailable on Myriad.
# Use 0.4.33 (matches dreamerv3/requirements.txt) which works with available CUDA libs.
pip install "jax[cuda12]==0.4.33"

# ----------------------------------------------------------
# Step 5: Install all project dependencies
# ----------------------------------------------------------
echo "[5/7] Installing project dependencies..."

# Core DreamerV3 dependencies
pip install \
    chex \
    einops \
    elements>=3.19.1 \
    ninjax>=3.5.1 \
    optax \
    numpy==1.26.4 \
    jaxtyping \
    flax \
    distrax \
    dm-env \
    dm-tree \
    rlax \
    tensorflow-probability

# Environment dependencies
pip install \
    craftax==1.5.0 \
    navix==0.7.4 \
    gymnax==0.0.9 \
    gymnasium==1.2.3 \
    pygame==2.6.1

# Utilities
pip install \
    wandb==0.24.1 \
    ruamel.yaml==0.19.1 \
    opencv-python-headless \
    tqdm \
    rich==14.3.2 \
    tyro==1.0.5 \
    pillow==12.1.0 \
    matplotlib==3.10.8 \
    seaborn==0.13.2 \
    pandas==2.3.3 \
    scipy==1.15.3 \
    imageio==2.37.2

# Portal, scope, granular (DreamerV3 infra)
pip install \
    portal==3.7.4 \
    google-resumable-media==2.8.0 \
    google-cloud-storage==3.9.0

# Dev tools (optional, comment out if not needed)
pip install \
    ipdb \
    colored_traceback \
    pytest==9.0.2

# ----------------------------------------------------------
# Step 6: Verify installation
# ----------------------------------------------------------
echo "[6/7] Verifying installation..."
echo "----------------------------------------"
python -c "
import jax
print(f'JAX version:     {jax.__version__}')
print(f'JAX devices:     {jax.devices()}')
print(f'GPU available:   {len(jax.devices(\"gpu\")) > 0}')
"
python -c "import craftax; print(f'Craftax OK')"
python -c "import dreamerv3; print(f'DreamerV3 OK')" 2>/dev/null || echo "  (dreamerv3 needs to be in PYTHONPATH)"
echo "----------------------------------------"

echo ""
echo "[7/7] Summary"
echo "============================================"
echo "  Setup complete!"
echo "  Python: $(python --version)"
echo "  pip:    $(which pip)"
echo ""
echo "  To activate in future sessions:"
echo "    module load gcc-libs/10.2.0"
echo "    module load python/miniconda3/4.10.3"
echo "    source \$UCL_CONDA_PATH/etc/profile.d/conda.sh"
echo "    conda activate ${ENV_NAME}"
echo "============================================"

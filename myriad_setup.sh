#!/bin/bash -l
# =============================================================
# UCL Myriad Environment Setup Script
# Project: continual-dreamerv3-autocurricula
# Usage:   bash myriad_setup.sh
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
echo "[1/6] Unloading default modules..."
module unload compilers mpi gcc-libs 2>/dev/null || true

# ----------------------------------------------------------
# Step 2: Load required modules
# ----------------------------------------------------------
echo "[2/6] Loading modules..."
module load gcc-libs/10.2.0
module load python/miniconda3/4.10.3

# Activate conda base
source $UCL_CONDA_PATH/etc/profile.d/conda.sh

# ----------------------------------------------------------
# Step 3: Create conda environment
# ----------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[3/6] Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[3/6] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

conda activate ${ENV_NAME}
echo "  Python: $(python --version)"

# ----------------------------------------------------------
# Step 4: Install JAX with CUDA 12 (pip-bundled CUDA libs)
# ----------------------------------------------------------
echo "[4/6] Installing JAX with CUDA 12..."
pip install --upgrade pip
pip install jax[cuda12]==0.6.2

# ----------------------------------------------------------
# Step 5: Install all project dependencies
# ----------------------------------------------------------
echo "[5/6] Installing project dependencies..."

# Core DreamerV3 dependencies
pip install \
    chex==0.1.90 \
    einops==0.8.2 \
    elements==3.21.0 \
    ninjax==3.6.2 \
    optax==0.2.6 \
    numpy==1.26.4 \
    jaxtyping \
    flax==0.10.7 \
    distrax==0.1.5 \
    dm-env==1.6 \
    dm-tree==0.1.9 \
    rlax==0.1.7 \
    tensorflow-probability==0.25.0

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
echo "[6/6] Verifying installation..."
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
echo "============================================"
echo "  Setup complete!"
echo "  To activate:  conda activate ${ENV_NAME}"
echo "============================================"

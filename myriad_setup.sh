#!/bin/bash -l
# =============================================================
# UCL Myriad Environment Setup Script
# Project: continual-dreamerv3-autocurricula
# Usage:   bash myriad_setup.sh
#
# Prerequisites:
#   conda create -n dreamer311 python=3.11 -y
# =============================================================

set -e

ENV_NAME="dreamer311"

echo "============================================"
echo "  Myriad Environment Setup: ${ENV_NAME}"
echo "============================================"

# ----------------------------------------------------------
# Step 1: Unload default modules that may conflict
# ----------------------------------------------------------
echo "[1/6] Unloading default modules..."
module unload compilers mpi gcc-libs 2>/dev/null || true

# ----------------------------------------------------------
# Step 2: Load required modules (GCC compiler + conda)
# ----------------------------------------------------------
echo "[2/6] Loading modules..."
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load python/miniconda3/4.10.3

# ----------------------------------------------------------
# Step 3: Resolve conda env paths (bypass conda activate)
# ----------------------------------------------------------
echo "[3/6] Locating conda env '${ENV_NAME}'..."

# Find conda env prefix
CONDA_PREFIX=$(conda env list | grep "^${ENV_NAME} " | awk '{print $NF}')
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: conda env '${ENV_NAME}' not found. Create it first:"
    echo "  conda create -n ${ENV_NAME} python=3.11 -y"
    exit 1
fi

# Use the env's own pip and python directly by absolute path.
# This bypasses 'conda activate' which fails in non-interactive bash scripts.
PIP="${CONDA_PREFIX}/bin/pip"
PYTHON="${CONDA_PREFIX}/bin/python"

echo "  Env prefix:     ${CONDA_PREFIX}"
echo "  Python binary:  ${PYTHON}"
echo "  Python version: $(${PYTHON} --version 2>&1)"
echo "  Pip binary:     ${PIP}"

# ----------------------------------------------------------
# Step 4: Install JAX with CUDA 12 (pip-bundled CUDA libs)
# ----------------------------------------------------------
echo "[4/6] Installing JAX with CUDA 12..."
${PIP} install --upgrade pip
# Python 3.11 supports JAX 0.4.33+. Use 0.4.33 for stability.
${PIP} install "jax[cuda12]==0.4.33"

# ----------------------------------------------------------
# Step 5: Install all project dependencies
# ----------------------------------------------------------
echo "[5/6] Installing project dependencies..."

# Core DreamerV3 dependencies
${PIP} install \
    chex \
    einops \
    "elements>=3.19.1" \
    "ninjax>=3.5.1" \
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
${PIP} install \
    craftax==1.5.0 \
    navix==0.7.4 \
    gymnax==0.0.9 \
    gymnasium==1.2.3 \
    pygame==2.6.1

# Utilities
${PIP} install \
    wandb==0.17.0 \
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

# Portal (DreamerV3 infra)
${PIP} install \
    portal==3.7.4 \
    google-resumable-media==2.8.0 \
    google-cloud-storage==3.9.0

# Dev tools
${PIP} install \
    ipdb \
    colored_traceback \
    pytest==9.0.2

# ----------------------------------------------------------
# Step 6: Verify installation
# ----------------------------------------------------------
echo "[6/6] Verifying installation..."
echo "----------------------------------------"
${PYTHON} -c "
import jax
print(f'JAX version:     {jax.__version__}')
print(f'JAX devices:     {jax.devices()}')
print(f'GPU available:   {len(jax.devices(\"gpu\")) > 0}')
"
${PYTHON} -c "import craftax; print('Craftax OK')"
${PYTHON} -c "import tensorstore; print(f'tensorstore {tensorstore.__version__} OK')"
echo "----------------------------------------"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Python: $(${PYTHON} --version 2>&1)"
echo ""
echo "  To activate in future sessions:"
echo "    module load gcc-libs/10.2.0"
echo "    module load compilers/gnu/10.2.0"
echo "    module load python/miniconda3/4.10.3"
echo "    source \$UCL_CONDA_PATH/etc/profile.d/conda.sh"
echo "    conda activate ${ENV_NAME}"
echo "============================================"

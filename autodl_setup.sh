#!/usr/bin/env bash
# ===========================================================
# AutoDL RTX 5090 (Blackwell / CUDA 13) Environment Setup
# Creates conda env: dreamer_cuda13
# ===========================================================
set -euo pipefail

ENV_NAME="dreamer_cuda13"
PYTHON_VERSION="3.11"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------
# Mirrors (AutoDL / China)
# -----------------------------------------------------------
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_TRUSTED="pypi.tuna.tsinghua.edu.cn"
CONDA_MIRROR="https://mirrors.tuna.tsinghua.edu.cn/anaconda"

# -----------------------------------------------------------
# 0. Pre-flight checks
# -----------------------------------------------------------
echo "============================================"
echo "  AutoDL RTX 5090 Setup - dreamer_cuda13"
echo "============================================"

# Check NVIDIA driver
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "[INFO] GPU info:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
    if [ "$DRIVER_VER" -lt 570 ]; then
        echo "[WARN] Driver ${DRIVER_VER}.x detected. CUDA 13 requires driver >= 570."
        echo "       Please upgrade the NVIDIA driver before continuing."
        exit 1
    fi
    echo "[OK] Driver >= 570"
else
    echo "[WARN] nvidia-smi not found. Skipping driver check."
fi

# Check conda
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. Please install Miniconda/Anaconda first."
    echo "  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh"
    echo "  bash /tmp/miniconda.sh -b -p \$HOME/miniconda3"
    echo "  eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\""
    exit 1
fi

# -----------------------------------------------------------
# 1. Create conda environment
# -----------------------------------------------------------
echo ""
echo "[1/5] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"

if conda env list | grep -qw "$ENV_NAME"; then
    echo "  Environment '${ENV_NAME}' already exists. Removing..."
    conda env remove -n "$ENV_NAME" -y
fi

# Configure conda to use TUNA mirror
conda config --add channels "${CONDA_MIRROR}/pkgs/main"
conda config --add channels "${CONDA_MIRROR}/pkgs/free"
conda config --set show_channel_urls yes

conda create -n "$ENV_NAME" python="${PYTHON_VERSION}" -y

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Configure pip to use Aliyun mirror (AutoDL internal network, fastest)
pip config set global.index-url "$PIP_MIRROR"
pip config set global.trusted-host "$PIP_TRUSTED"

echo "  Python: $(python --version)"
echo "  pip:    $(pip --version)"
echo "  pip mirror: ${PIP_MIRROR}"

# -----------------------------------------------------------
# 2. Install JAX + CUDA 13 (Blackwell sm_120a support)
# -----------------------------------------------------------
echo ""
echo "[2/5] Installing JAX 0.9.2 with CUDA 13 (jaxlib + jax-cuda13-plugin)"

pip install --upgrade pip setuptools wheel \
    -i "$PIP_MIRROR" --trusted-host "$PIP_TRUSTED"

# JAX with bundled CUDA 13 libraries (no system CUDA toolkit needed)
pip install "jax[cuda13]==0.9.2" \
    -i "$PIP_MIRROR" --trusted-host "$PIP_TRUSTED"

# -----------------------------------------------------------
# 3. Install JAX ecosystem + core numerical
# -----------------------------------------------------------
echo ""
echo "[3/5] Installing JAX ecosystem & numerical packages"

pip install \
    "flax==0.12.6" \
    "optax==0.2.8" \
    "chex==0.1.91" \
    "jaxtyping>=0.2.34" \
    "numpy>=2.0,<2.2" \
    "scipy>=1.13" \
    "einops>=0.8.0" \
    -i "$PIP_MIRROR" --trusted-host "$PIP_TRUSTED"

# -----------------------------------------------------------
# 4. Install DreamerV3 framework + RL envs + everything else
# -----------------------------------------------------------
echo ""
echo "[4/5] Installing DreamerV3 framework, RL environments & utilities"

pip install \
    "ninjax==3.6.2" \
    "elements==3.22.0" \
    "portal==3.8.1" \
    cloudpickle \
    "craftax==1.5.0" \
    "navix==0.7.4" \
    "gymnax==0.0.9" \
    "gymnasium>=1.2.0" \
    "pygame>=2.6.1" \
    dm-env \
    "wandb>=0.16.5" \
    "matplotlib>=3.9" \
    "seaborn>=0.13.2" \
    "pandas>=2.2" \
    "imageio>=2.37" \
    "pillow>=10.0" \
    "ruamel.yaml>=0.18" \
    opencv-python-headless \
    tqdm \
    "rich>=13.0" \
    "tyro>=0.8" \
    "setuptools>=59.5.0" \
    ipdb \
    colored_traceback \
    "pytest>=8.0" \
    -i "$PIP_MIRROR" --trusted-host "$PIP_TRUSTED"

# -----------------------------------------------------------
# 5. Verify installation
# -----------------------------------------------------------
echo ""
echo "[5/5] Verifying installation..."
echo ""

python - <<'PYEOF'
import sys
print(f"Python:  {sys.version}")

import jax
print(f"JAX:     {jax.__version__}")

import jaxlib
print(f"jaxlib:  {jaxlib.__version__}")

import flax
print(f"Flax:    {flax.__version__}")

import optax
print(f"Optax:   {optax.__version__}")

import numpy as np
print(f"NumPy:   {np.__version__}")

# GPU check
devices = jax.devices()
gpu_devices = [d for d in devices if d.platform == "gpu"]
print(f"\nDevices: {devices}")
print(f"GPUs:    {len(gpu_devices)}")

if gpu_devices:
    # Quick matrix multiply test
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2048, 2048))
    y = jnp.dot(x, x).block_until_ready()
    print(f"GPU matmul test: OK  (shape={y.shape})")
else:
    print("[WARN] No GPU detected! JAX will run on CPU only.")
    print("  Check: nvidia-smi, CUDA_VISIBLE_DEVICES, driver version")
PYEOF

# -----------------------------------------------------------
# Done
# -----------------------------------------------------------
echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  Activate:  conda activate ${ENV_NAME}"
echo ""
echo "  RTX 5090 workaround (if XLA crashes):"
echo "    export XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1"
echo ""
echo "  Train:"
echo "    python train_craftax.py"
echo ""

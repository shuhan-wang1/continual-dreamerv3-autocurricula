#!/bin/bash
# =============================================================
# Execute demo_notebook.ipynb with the dreamer_cuda13 conda env
# and produce an executed notebook with all outputs.
#
# Usage:
#   bash run_demo_notebook.sh
#
# Output:
#   demo_notebook_executed.ipynb  (notebook with cell outputs)
# =============================================================
set -euo pipefail

# ── Activate conda ──
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: Cannot find conda."; exit 1; }
conda activate dreamer_cuda13

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# JAX env vars (same as training scripts)
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_command_buffer= --xla_gpu_force_compilation_parallelism=1"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH:-}"

# ── Ensure jupyter/nbconvert are available ──
python -c "import nbconvert" 2>/dev/null || pip install -q jupyter nbconvert ipykernel

# ── Register this conda env as a Jupyter kernel (one-time) ──
python -m ipykernel install --user --name dreamer_cuda13 --display-name "Python (dreamer_cuda13)" 2>/dev/null || true

echo "============================================"
echo "  Executing demo_notebook.ipynb"
echo "  Conda env: dreamer_cuda13"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================"

# ── Execute notebook ──
# --ExecutePreprocessor.timeout=-1  -> no cell timeout
# --ExecutePreprocessor.kernel_name -> use our conda kernel
jupyter nbconvert \
    --to notebook \
    --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=dreamer_cuda13 \
    --output demo_notebook_executed.ipynb \
    demo_notebook.ipynb

echo ""
echo "============================================"
echo "  Done! Output: demo_notebook_executed.ipynb"
echo "============================================"

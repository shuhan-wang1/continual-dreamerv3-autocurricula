#!/bin/bash -l
# =============================================================
# UCL Myriad GPU Test Script (1 hour, A100)
# Usage:   qsub myriad_test_gpu.sh
# Monitor: qstat
#
# Tests: CUDA visibility, JAX GPU detection, small Craftax run
# =============================================================

# --- Job configuration ---
#$ -l h_rt=1:00:00            # 1 hour
#$ -l mem=16G                  # RAM
#$ -l gpu=1                    # 1 GPU
#$ -ac allow=L                 # L=A100-40G (easier to get)
#$ -N gpu_test                 # Job name
#$ -o $HOME/logs/gpu_test.$JOB_ID.out
#$ -e $HOME/logs/gpu_test.$JOB_ID.err

mkdir -p $HOME/logs

# --- Load modules ---
module unload compilers mpi gcc-libs 2>/dev/null || true
module load gcc-libs/10.2.0
module load compilers/gnu/10.2.0
module load python/miniconda3/4.10.3

# --- Resolve conda env ---
ENV_NAME="dreamer311"
CONDA_PREFIX=$(conda env list | grep "^${ENV_NAME} " | awk '{print $NF}')
PIP="${CONDA_PREFIX}/bin/pip"
PYTHON="${CONDA_PREFIX}/bin/python"

# --- Project dir ---
PROJECT_DIR="$HOME/continual-dreamerv3-autocurricula"
cd $PROJECT_DIR

export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/dreamerv3:${PYTHONPATH}"

echo "============================================"
echo "  GPU Test — $(date)"
echo "============================================"
echo ""

# --- Test 1: System info ---
echo ">>> Test 1: System Info"
echo "  Job ID:    $JOB_ID"
echo "  Node:      $(hostname)"
echo "  Python:    $(${PYTHON} --version 2>&1)"
echo "  CUDA_VIS:  $CUDA_VISIBLE_DEVICES"
echo ""

# --- Test 2: nvidia-smi ---
echo ">>> Test 2: nvidia-smi"
nvidia-smi
echo ""

# --- Test 3: JAX GPU detection ---
echo ">>> Test 3: JAX GPU Detection"
${PYTHON} << 'PYEOF'
import jax
import jaxlib
print(f"  JAX:    {jax.__version__}")
print(f"  jaxlib: {jaxlib.__version__}")
print(f"  Devices: {jax.devices()}")
gpu_devices = [d for d in jax.devices() if d.platform == "gpu"]
print(f"  GPU count: {len(gpu_devices)}")
for d in gpu_devices:
    print(f"    - {d}")
assert len(gpu_devices) > 0, "NO GPU DETECTED!"
print("  [PASS] GPU detected")
PYEOF
echo ""

# --- Test 4: JAX computation on GPU ---
echo ">>> Test 4: JAX Computation on GPU"
${PYTHON} << 'PYEOF'
import jax
import jax.numpy as jnp
import time

# Matrix multiply benchmark
size = 4096
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (size, size))
b = jax.random.normal(key, (size, size))

# Warmup (JIT compile)
c = jnp.dot(a, b).block_until_ready()

# Benchmark
start = time.time()
for _ in range(10):
    c = jnp.dot(a, b).block_until_ready()
elapsed = time.time() - start

tflops = 10 * 2 * size**3 / elapsed / 1e12
print(f"  Matrix {size}x{size} matmul x10: {elapsed:.2f}s ({tflops:.2f} TFLOPS)")
print(f"  Device: {c.devices()}")
print("  [PASS] GPU computation OK")
PYEOF
echo ""

# --- Test 5: JAX memory usage ---
echo ">>> Test 5: JAX Memory"
${PYTHON} << 'PYEOF'
import jax
backend = jax.lib.xla_bridge.get_backend()
for d in backend.devices():
    stats = d.memory_stats()
    if stats:
        total = stats.get("bytes_limit", 0) / 1e9
        used  = stats.get("bytes_in_use", 0) / 1e9
        print(f"  {d}: {used:.1f}GB / {total:.1f}GB")
    else:
        print(f"  {d}: memory stats not available")
print("  [PASS] Memory check OK")
PYEOF
echo ""

# --- Test 6: Import all key packages ---
echo ">>> Test 6: Import Smoke Test"
${PYTHON} << 'PYEOF'
imports = [
    "jax", "jaxlib", "flax", "optax", "chex", "einops",
    "elements", "ninjax", "distrax", "rlax",
    "craftax", "navix", "gymnax", "gymnasium",
    "wandb", "numpy", "scipy",
]
ok = 0
fail = 0
for name in imports:
    try:
        __import__(name)
        ok += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        fail += 1
print(f"  {ok}/{ok+fail} packages imported OK")
if fail == 0:
    print("  [PASS] All imports OK")
else:
    print(f"  [FAIL] {fail} packages failed to import")
PYEOF
echo ""

# --- Test 7: Short Craftax training run (500 steps) ---
echo ">>> Test 7: Short Craftax Training (500 steps)"
${PYTHON} train_craftax.py \
    --seed 42 \
    --steps 500 \
    --envs 4 \
    --batch_size 4 \
    --batch_length 16 \
    --model_size 25m \
    --wandb_mode disabled \
    --logdir /tmp/gpu_test_craftax_$JOB_ID \
    --no_plan2explore
TEST_EXIT=$?

if [ $TEST_EXIT -eq 0 ]; then
    echo "  [PASS] Craftax training completed"
else
    echo "  [FAIL] Craftax training failed (exit code $TEST_EXIT)"
fi

# Cleanup test logdir
rm -rf /tmp/gpu_test_craftax_$JOB_ID

echo ""
echo "============================================"
echo "  GPU Test Complete — $(date)"
echo "============================================"

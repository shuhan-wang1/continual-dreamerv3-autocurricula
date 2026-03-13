#!/bin/bash -l
# =============================================================
# Myriad Environment Check Script
# Usage: bash myriad_check_env.sh
# =============================================================

ENV_NAME="dreamer311"

# --- Load modules & resolve conda env path ---
module load gcc-libs/10.2.0 2>/dev/null || true
module load compilers/gnu/10.2.0 2>/dev/null || true
module load python/miniconda3/4.10.3 2>/dev/null || true

CONDA_PREFIX=$(conda env list | grep "^${ENV_NAME} " | awk '{print $NF}')
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: conda env '${ENV_NAME}' not found."
    exit 1
fi

PYTHON="${CONDA_PREFIX}/bin/python"
PIP="${CONDA_PREFIX}/bin/pip"

echo "============================================"
echo "  Environment Check: ${ENV_NAME}"
echo "============================================"
echo ""

# --- System info ---
echo ">>> System"
echo "  Python:        $(${PYTHON} --version 2>&1)"
echo "  Python binary: ${PYTHON}"
echo "  Pip binary:    ${PIP}"
echo "  Platform:      $(uname -s -r -m)"
echo "  Node:          $(hostname)"
echo ""

# --- GPU info ---
echo ">>> GPU"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed (not on GPU node?)"
else
    echo "  nvidia-smi not found (login node has no GPU — submit a GPU job to test)"
fi
echo ""

# --- Required packages and expected versions ---
echo ">>> Package Versions"
echo ""

${PYTHON} << 'PYEOF'
import importlib
import sys

packages = [
    ("jax",                "jax",                "0.4.33"),
    ("jaxlib",             "jaxlib",             "0.4.33"),
    ("chex",               "chex",               None),
    ("einops",             "einops",             None),
    ("elements",           "elements",           ">=3.19.1"),
    ("ninjax",             "ninjax",             ">=3.5.1"),
    ("optax",              "optax",              None),
    ("numpy",              "numpy",              "1.26.4"),
    ("flax",               "flax",               None),
    ("distrax",            "distrax",            None),
    ("dm-env",             "dm_env",             None),
    ("dm-tree",            "tree",               None),
    ("rlax",               "rlax",               None),
    ("jaxtyping",          "jaxtyping",          None),
    ("tensorflow-probability", "tensorflow_probability", None),
    ("portal",             "portal",             None),
    ("craftax",            "craftax",            None),
    ("navix",              "navix",              None),
    ("gymnax",             "gymnax",             None),
    ("gymnasium",          "gymnasium",          None),
    ("pygame",             "pygame",             None),
    ("wandb",              "wandb",              None),
    ("ruamel.yaml",        "ruamel.yaml",        None),
    ("opencv-python-headless", "cv2",            None),
    ("tqdm",               "tqdm",               None),
    ("rich",               "rich",               None),
    ("tyro",               "tyro",               None),
    ("pillow",             "PIL",                None),
    ("matplotlib",         "matplotlib",         None),
    ("seaborn",            "seaborn",            None),
    ("pandas",             "pandas",             None),
    ("scipy",              "scipy",              None),
    ("imageio",            "imageio",            None),
    ("google-cloud-storage", "google.cloud.storage", None),
    ("google-resumable-media", "google.resumable_media", None),
    ("ipdb",               "ipdb",               None),
    ("colored_traceback",  "colored_traceback",  None),
    ("pytest",             "pytest",             None),
]

ok = 0
fail = 0
warn = 0

for pkg_name, import_name, expected in packages:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        status = "OK"
        if expected and not expected.startswith(">="):
            if ver != expected and ver != "?":
                status = "MISMATCH"
                warn += 1
            else:
                ok += 1
        else:
            ok += 1
        print(f"  {status:10s}  {pkg_name:30s}  {ver}")
    except ImportError as e:
        fail += 1
        print(f"  {'MISSING':10s}  {pkg_name:30s}  ({e})")

print("")
print(">>> JAX Device Check")
try:
    import jax
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    print(f"  All devices: {devices}")
    if gpu_devices:
        print(f"  GPU count:   {len(gpu_devices)}")
        for d in gpu_devices:
            print(f"    - {d}")
    else:
        print("  WARNING: No GPU detected (expected on login node, test via qsub)")
except Exception as e:
    print(f"  ERROR: {e}")

print("")
print("============================================")
print(f"  OK: {ok}   MISMATCH: {warn}   MISSING: {fail}")
if fail > 0:
    print(f"  {fail} packages need to be installed!")
    sys.exit(1)
else:
    print("  All dependencies satisfied.")
print("============================================")
PYEOF

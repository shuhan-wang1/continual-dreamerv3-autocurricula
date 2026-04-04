#!/bin/bash
# =============================================================
# Package G-group model checkpoints for local download/backup.
#
# Collects the ckpt/ directory from every experiment under
# experiment_results/10m/ — excludes replay/ buffers (huge).
#
# Output:  g_group_ckpts_<timestamp>.zip
#
# Zip layout:
#   g_ckpts/<exp_name>/ckpt/<checkpoint_files>
#
# Local unzip workflow:
#   scp user@autodl-host:<project_dir>/g_group_ckpts_<ts>.zip .
#   unzip g_group_ckpts_<ts>.zip          # extracts into ./g_ckpts/
# =============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="experiment_results/10m"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${PROJECT_DIR}/g_group_ckpts_${TIMESTAMP}.zip"

echo "========================================"
echo "  Package G-group Checkpoints"
echo "  Source:  $RESULTS_DIR"
echo "  Output:  $OUTPUT"
echo "  Note:    replay/ buffers are excluded"
echo "  $(date)"
echo "========================================"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

COUNT=0
MISSING=0

for exp_dir in "$RESULTS_DIR"/*/; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")
    ckpt_dir="${exp_dir}ckpt"

    if [ ! -d "$ckpt_dir" ]; then
        echo "  [!] $exp_name — no ckpt/ directory (not yet checkpointed?)"
        MISSING=$((MISSING + 1))
        continue
    fi

    dest="$STAGING/g_ckpts/$exp_name"
    mkdir -p "$dest"
    cp -r "$ckpt_dir" "$dest/"

    # Human-readable checkpoint size
    ckpt_size=$(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)
    echo "  [+] $exp_name/ckpt/  ($ckpt_size)"
    COUNT=$((COUNT + 1))
done

if [ "$COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No ckpt/ directories found under $RESULTS_DIR"
    exit 1
fi

echo ""
echo "Creating zip archive (this may take a while for large checkpoints)..."
cd "$STAGING"
zip -r "$OUTPUT" g_ckpts/
cd "$PROJECT_DIR"

SIZE=$(du -sh "$OUTPUT" | cut -f1)

echo ""
echo "========================================"
echo "  Done."
echo "  Checkpoints packaged : $COUNT"
echo "  Missing              : $MISSING"
echo "  Archive              : $OUTPUT"
echo "  Size                 : $SIZE"
echo ""
echo "  Download & unzip locally:"
echo "    scp user@<autodl-host>:$OUTPUT ."
echo "    unzip $(basename "$OUTPUT")"
echo "  → files land in ./g_ckpts/<exp_name>/ckpt/"
echo "========================================"

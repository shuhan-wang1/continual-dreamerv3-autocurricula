#!/bin/bash
# =============================================================
# Package G-group online_metrics.jsonl files for local download.
#
# Output:  g_group_jsonl_<timestamp>.zip
#
# Zip layout:
#   all_results/<exp_name>/craftax_<tag>/online_metrics.jsonl
#
# Local unzip workflow:
#   scp user@autodl-host:<project_dir>/g_group_jsonl_<ts>.zip .
#   unzip g_group_jsonl_<ts>.zip          # extracts into ./all_results/
# =============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="experiment_results/10m"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${PROJECT_DIR}/g_group_jsonl_${TIMESTAMP}.zip"

echo "========================================"
echo "  Package G-group JSONL files"
echo "  Source:  $RESULTS_DIR"
echo "  Output:  $OUTPUT"
echo "  $(date)"
echo "========================================"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Build the zip with all_results/ prefix so unzipping locally
# produces the correct directory layout.
STAGING=$(mktemp -d)
trap 'rm -rf "$STAGING"' EXIT

COUNT=0
MISSING=0

for exp_dir in "$RESULTS_DIR"/*/; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")

    # Collect every online_metrics.jsonl under the experiment dir
    # (typically one file at craftax_<tag>/online_metrics.jsonl)
    while IFS= read -r -d '' jsonl_file; do
        # Relative path inside the experiment dir, e.g. craftax_G1v2.../online_metrics.jsonl
        rel="${jsonl_file#${exp_dir}}"
        dest_dir="$STAGING/all_results/$exp_name/$(dirname "$rel")"
        mkdir -p "$dest_dir"
        cp "$jsonl_file" "$dest_dir/"
        echo "  [+] $exp_name/$rel"
        COUNT=$((COUNT + 1))
    done < <(find "$exp_dir" -name "online_metrics.jsonl" -print0 2>/dev/null)

    # Warn if an experiment directory has no metrics yet
    if ! find "$exp_dir" -name "online_metrics.jsonl" -quit 2>/dev/null | grep -q .; then
        echo "  [!] $exp_name — no online_metrics.jsonl found (training not started?)"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No online_metrics.jsonl files found under $RESULTS_DIR"
    exit 1
fi

echo ""
echo "Creating zip archive..."
cd "$STAGING"
zip -r "$OUTPUT" all_results/
cd "$PROJECT_DIR"

SIZE=$(du -sh "$OUTPUT" | cut -f1)

echo ""
echo "========================================"
echo "  Done."
echo "  Files packaged : $COUNT"
echo "  Missing        : $MISSING"
echo "  Archive        : $OUTPUT"
echo "  Size           : $SIZE"
echo ""
echo "  Download & unzip locally:"
echo "    scp user@<autodl-host>:$OUTPUT ."
echo "    unzip $(basename "$OUTPUT")"
echo "  → files land in ./all_results/"
echo "========================================"

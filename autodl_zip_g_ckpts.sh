#!/bin/bash
# =============================================================
# Package G-group model checkpoints for local download/backup.
#
# Actual path on AutoDL (DreamerV3 appends craftax_{tag} to logdir):
#   experiment_results/10m/<exp_name>/craftax_<tag>/ckpt/<timestamp>/
#     agent.pkl.filepart  ← model weights
#     step.pkl            ← training step counter
#     done                ← save-complete marker
#     replay.pkl          ← replay buffer (EXCLUDED — can be many GB)
#
# Output:  g_group_ckpts_<timestamp>.zip
#
# Zip layout:
#   g_ckpts/<exp_name>/craftax_<tag>/ckpt/<timestamp>/{agent,step,done}
#   g_ckpts/<exp_name>/craftax_<tag>/ckpt/latest
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
echo "  Note:    replay.pkl excluded (model weights only)"
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

    # DreamerV3 saves into <logdir>/craftax_<tag>/ckpt/
    found=0
    for craftax_dir in "$exp_dir"craftax_*/; do
        [ -d "$craftax_dir" ] || continue
        ckpt_dir="${craftax_dir}ckpt"
        [ -d "$ckpt_dir" ] || continue

        craftax_name=$(basename "$craftax_dir")
        dest_ckpt="$STAGING/g_ckpts/$exp_name/$craftax_name/ckpt"
        mkdir -p "$dest_ckpt"

        # Copy latest marker
        [ -f "$ckpt_dir/latest" ] && cp "$ckpt_dir/latest" "$dest_ckpt/"

        # Copy each timestamped checkpoint subdir, excluding replay.pkl
        for ts_dir in "$ckpt_dir"/*/; do
            [ -d "$ts_dir" ] || continue
            ts_name=$(basename "$ts_dir")
            dest_ts="$dest_ckpt/$ts_name"
            mkdir -p "$dest_ts"
            for f in "$ts_dir"*; do
                [ -f "$f" ] || continue
                fname=$(basename "$f")
                [ "$fname" = "replay.pkl" ] && continue   # skip — huge
                cp "$f" "$dest_ts/"
            done
        done

        ckpt_size=$(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)
        echo "  [+] $exp_name/$craftax_name/ckpt/  ($ckpt_size on disk)"
        COUNT=$((COUNT + 1))
        found=1
    done

    if [ "$found" -eq 0 ]; then
        echo "  [!] $exp_name — no craftax_*/ckpt/ found (not yet checkpointed?)"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No checkpoints found under $RESULTS_DIR"
    exit 1
fi

echo ""
echo "Creating zip archive (this may take a while)..."
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
echo "  → files land in ./g_ckpts/<exp_name>/craftax_<tag>/ckpt/"
echo "========================================"

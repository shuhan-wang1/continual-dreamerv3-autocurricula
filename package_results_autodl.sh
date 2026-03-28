#!/bin/bash
# =============================================================
# Package experiment results for Google Drive (AutoDL version)
#
# Collects:
#   - All generated figures (PDF + PNG)
#   - A structured per-experiment JSON summary
#   - Raw online_metrics.jsonl for each run
#   - Configs and paper draft
#
# Output:  experiment_results/packaged/craftax_results_autodl_<date>.tar.gz
#
# Usage:   bash package_results_autodl.sh [--results_dir DIR]
#
# Default results_dir: experiment_results/10m
# =============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Parse optional --results_dir
RESULTS_DIR="experiment_results/10m"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results_dir) RESULTS_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

FIGURES_DIR="experiment_results/figures"
PACK_DIR="experiment_results/packaged"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STAGING="${PACK_DIR}/staging_${TIMESTAMP}"
ARCHIVE="${PACK_DIR}/craftax_results_autodl_${TIMESTAMP}.tar.gz"

# AutoDL conda env
PYTHON=""
for candidate in \
    "${HOME}/.conda/envs/dreamer_cuda13/bin/python" \
    "${HOME}/miniconda3/envs/dreamer_cuda13/bin/python" \
    "$(which python3 2>/dev/null)" \
    "$(which python 2>/dev/null)"; do
    if [ -x "$candidate" 2>/dev/null ]; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: No Python found. Activate your conda env first."
    exit 1
fi

mkdir -p "$PACK_DIR" "$STAGING"

echo "========================================"
echo "  Package AutoDL Experiment Results"
echo "  Results dir: $RESULTS_DIR"
echo "  Python:      $PYTHON"
echo "  $(date)"
echo "========================================"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    echo "  If your experiments are elsewhere, use: bash $0 --results_dir /path/to/results"
    exit 1
fi

# ── 1. Copy figures ───────────────────────────────────────────
if [ -d "$FIGURES_DIR" ]; then
    echo "[1/4] Copying figures..."
    cp -r "$FIGURES_DIR" "$STAGING/figures"
    echo "  $(ls "$STAGING/figures/" | wc -l) files copied"
else
    echo "[1/4] No figures directory found — skipping"
    mkdir -p "$STAGING/figures"
fi

# ── 2. Build structured JSON summaries ────────────────────────
echo "[2/4] Building structured experiment summaries..."

$PYTHON - "$RESULTS_DIR" "$STAGING" << 'PYEOF'
import json, os, glob, sys
from collections import OrderedDict

RESULTS_DIR = sys.argv[1]
STAGING = sys.argv[2]
SEEDS = [1, 4, 42]

# ── Achievement tier mapping (authoritative, from train_craftax.py) ──
TIER_MAP = {}
for name in [
    'collect_wood', 'place_table', 'eat_cow', 'collect_sapling',
    'collect_drink', 'make_wood_pickaxe', 'make_wood_sword',
    'place_plant', 'eat_plant',
]:
    TIER_MAP[name] = 0
for name in [
    'defeat_zombie', 'collect_stone', 'place_stone',
    'defeat_skeleton', 'make_stone_pickaxe', 'make_stone_sword',
    'wake_up', 'place_furnace', 'collect_coal', 'eat_bat', 'eat_snail',
]:
    TIER_MAP[name] = 1
for name in [
    'collect_iron', 'make_iron_pickaxe', 'make_iron_sword',
    'make_iron_armour', 'make_arrow', 'make_torch', 'place_torch',
    'make_diamond_sword', 'make_diamond_armour', 'find_bow', 'fire_bow',
]:
    TIER_MAP[name] = 2
for name in [
    'collect_diamond', 'make_diamond_pickaxe', 'collect_sapphire',
    'collect_ruby', 'enter_gnomish_mines', 'enter_dungeon', 'enter_sewers',
    'enter_vault', 'enter_troll_mines', 'defeat_gnome_warrior',
    'defeat_gnome_archer', 'defeat_orc_solider', 'defeat_orc_mage',
    'defeat_lizard', 'defeat_kobold', 'learn_fireball', 'cast_fireball',
    'learn_iceball', 'cast_iceball', 'open_chest', 'drink_potion',
    'enchant_sword', 'enchant_armour',
]:
    TIER_MAP[name] = 3
for name in [
    'enter_fire_realm', 'enter_ice_realm', 'enter_graveyard',
    'defeat_troll', 'defeat_deep_thing', 'defeat_pigman',
    'defeat_fire_elemental', 'defeat_frost_troll', 'defeat_ice_elemental',
    'defeat_knight', 'defeat_archer', 'damage_necromancer', 'defeat_necromancer',
]:
    TIER_MAP[name] = 4


def load_achievement_names():
    """Load authoritative achievement names from first available summary."""
    for p in sorted(glob.glob(f"{RESULTS_DIR}/*/craftax_*/metrics_summary.json")):
        d = json.load(open(p))
        if "achievement_names" in d:
            return d["achievement_names"]
    # Fallback: hardcoded correct order (Craftax enum sorted by value)
    return [
        'collect_wood', 'place_table', 'eat_cow', 'collect_sapling',
        'collect_drink', 'make_wood_pickaxe', 'make_wood_sword',
        'place_plant', 'defeat_zombie', 'collect_stone', 'place_stone',
        'eat_plant', 'defeat_skeleton', 'make_stone_pickaxe',
        'make_stone_sword', 'wake_up', 'place_furnace', 'collect_coal',
        'collect_iron', 'collect_diamond', 'make_iron_pickaxe',
        'make_iron_sword', 'make_arrow', 'make_torch', 'place_torch',
        'make_diamond_sword', 'make_iron_armour', 'make_diamond_armour',
        'enter_gnomish_mines', 'enter_dungeon', 'enter_sewers', 'enter_vault',
        'enter_troll_mines', 'enter_fire_realm', 'enter_ice_realm',
        'enter_graveyard', 'defeat_gnome_warrior', 'defeat_gnome_archer',
        'defeat_orc_solider', 'defeat_orc_mage', 'defeat_lizard',
        'defeat_kobold', 'defeat_troll', 'defeat_deep_thing', 'defeat_pigman',
        'defeat_fire_elemental', 'defeat_frost_troll', 'defeat_ice_elemental',
        'damage_necromancer', 'defeat_necromancer', 'eat_bat', 'eat_snail',
        'find_bow', 'fire_bow', 'collect_sapphire', 'learn_fireball',
        'cast_fireball', 'learn_iceball', 'cast_iceball', 'collect_ruby',
        'make_diamond_pickaxe', 'open_chest', 'drink_potion', 'enchant_sword',
        'enchant_armour', 'defeat_knight', 'defeat_archer',
    ]


def build_experiment_summary(exp_id, ach_names):
    """Build structured summary for one experiment config across all seeds."""
    seed_data = []

    for seed in SEEDS:
        seed_dir = os.path.join(RESULTS_DIR, f"{exp_id}_seed{seed}")
        if not os.path.isdir(seed_dir):
            continue

        summary_files = glob.glob(f"{seed_dir}/*/metrics_summary.json")
        online_files = glob.glob(f"{seed_dir}/*/online_metrics.jsonl")

        if not summary_files or not online_files:
            continue

        summary = json.load(open(summary_files[0]))
        online_records = open(online_files[0]).readlines()
        last_record = json.loads(online_records[-1])
        first_record = json.loads(online_records[0])

        per_ach = last_record.get("per_achievement_rates", [0.0] * 67)
        achievement_details = []
        for i, rate in enumerate(per_ach):
            name = ach_names[i] if i < len(ach_names) else f"achievement_{i}"
            achievement_details.append({
                "index": i,
                "name": name,
                "tier": TIER_MAP.get(name, -1),
                "success_rate": round(rate, 6),
            })

        tier_rates = {t: [] for t in range(5)}
        for ad in achievement_details:
            if ad["tier"] >= 0:
                tier_rates[ad["tier"]].append(ad["success_rate"])
        tier_summary = {}
        for t in range(5):
            vals = tier_rates[t]
            tier_summary[f"tier_{t}"] = {
                "n_achievements": len(vals),
                "mean_rate": round(sum(vals) / len(vals), 6) if vals else 0.0,
                "n_unlocked": sum(1 for v in vals if v > 0.01),
            }

        seed_entry = {
            "seed": seed,
            "total_episodes": len(online_records),
            "total_steps": last_record.get("step", 0),
            "step_range": [first_record.get("step", 0), last_record.get("step", 0)],
            "final_metrics": {
                "mean_return": summary.get("mean_return"),
                "success_rate": summary.get("success_rate"),
                "max_achievement_depth": summary.get("max_achievement_depth"),
                "personal_best_depth": summary.get("personal_best_depth"),
                "mean_achievement_rate": round(sum(per_ach) / len(per_ach), 6) if per_ach else 0.0,
                "num_achievements_unlocked": sum(1 for r in per_ach if r > 0.01),
                "aggregate_forgetting": last_record.get("aggregate_forgetting"),
                "depth_mean": last_record.get("depth_mean"),
            },
            "tier_summary": tier_summary,
            "per_achievement_rates": achievement_details,
        }
        seed_data.append(seed_entry)

    if not seed_data:
        return None

    def mean_of(key):
        vals = [s["final_metrics"][key] for s in seed_data if s["final_metrics"][key] is not None]
        return round(sum(vals) / len(vals), 6) if vals else None

    def std_of(key):
        vals = [s["final_metrics"][key] for s in seed_data if s["final_metrics"][key] is not None]
        if len(vals) < 2:
            return 0.0
        m = sum(vals) / len(vals)
        return round((sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5, 6)

    aggregated = {
        "mean_return": {"mean": mean_of("mean_return"), "std": std_of("mean_return")},
        "mean_achievement_rate": {"mean": mean_of("mean_achievement_rate"), "std": std_of("mean_achievement_rate")},
        "num_achievements_unlocked": {"mean": mean_of("num_achievements_unlocked"), "std": std_of("num_achievements_unlocked")},
        "aggregate_forgetting": {"mean": mean_of("aggregate_forgetting"), "std": std_of("aggregate_forgetting")},
        "max_achievement_depth": {"mean": mean_of("max_achievement_depth"), "std": std_of("max_achievement_depth")},
    }

    return {
        "experiment_id": exp_id,
        "n_seeds": len(seed_data),
        "seeds": [s["seed"] for s in seed_data],
        "aggregated": aggregated,
        "per_seed": seed_data,
    }


# ── Discover experiments ──
ach_names = load_achievement_names()
exp_ids = set()
for p in sorted(os.listdir(RESULTS_DIR)):
    full = os.path.join(RESULTS_DIR, p)
    if os.path.isdir(full):
        for seed in SEEDS:
            if p.endswith(f"_seed{seed}"):
                exp_ids.add(p[: -len(f"_seed{seed}")])
exp_ids = sorted(exp_ids)

if not exp_ids:
    print(f"  WARNING: No experiments found in {RESULTS_DIR}")
    print(f"  Directory contents: {os.listdir(RESULTS_DIR)}")
    sys.exit(0)

# ── Build and save summaries ──
summaries_dir = os.path.join(STAGING, "summaries")
os.makedirs(summaries_dir, exist_ok=True)

all_experiments = []
for exp_id in exp_ids:
    summary = build_experiment_summary(exp_id, ach_names)
    if summary is None:
        print(f"  {exp_id}: SKIPPED (no data)")
        continue

    out_path = os.path.join(summaries_dir, f"{exp_id}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    all_experiments.append(summary)
    print(f"  {exp_id}: {summary['n_seeds']} seeds, "
          f"mean_ach_rate={summary['aggregated']['mean_achievement_rate']['mean']:.4f} "
          f"+/- {summary['aggregated']['mean_achievement_rate']['std']:.4f}")

# ── Save combined index ──
index = {
    "platform": "autodl",
    "results_type": "10M_steps",
    "n_experiments": len(all_experiments),
    "achievement_names": ach_names,
    "tier_sizes": {f"tier_{t}": sum(1 for n in ach_names if TIER_MAP.get(n) == t) for t in range(5)},
    "experiments": [{
        "experiment_id": e["experiment_id"],
        "n_seeds": e["n_seeds"],
        "aggregated": e["aggregated"],
    } for e in all_experiments],
}
with open(os.path.join(STAGING, "experiment_index.json"), "w") as f:
    json.dump(index, f, indent=2)

print(f"\n  {len(all_experiments)} experiments summarised")
PYEOF

echo "  Done"

# ── 3. Copy raw online_metrics.jsonl files ────────────────────
echo "[3/4] Copying raw online_metrics.jsonl files..."
RAW_DIR="$STAGING/raw_metrics"
mkdir -p "$RAW_DIR"

for exp_dir in "$RESULTS_DIR"/*/; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")
    jsonl=$(find "$exp_dir" -name "online_metrics.jsonl" -print -quit 2>/dev/null)
    if [ -n "$jsonl" ]; then
        cp "$jsonl" "$RAW_DIR/${exp_name}_online_metrics.jsonl"
    fi
done
N_JSONL=$(ls "$RAW_DIR"/*.jsonl 2>/dev/null | wc -l)
echo "  ${N_JSONL} JSONL files copied"

# ── 4. Copy supplementary files ───────────────────────────────
echo "[4/4] Copying supplementary files..."
# Copy any manifest if present
find "$RESULTS_DIR" -maxdepth 1 -name "experiment_manifest.json" -exec cp {} "$STAGING/" \; 2>/dev/null || true
cp -f Report.tex "$STAGING/" 2>/dev/null || true
cp -f experiment_results/plot_neurips_figures.py "$STAGING/" 2>/dev/null || true

# ── Create archive ────────────────────────────────────────────
echo ""
echo "Creating archive..."
tar czf "$ARCHIVE" -C "$PACK_DIR" "staging_${TIMESTAMP}"

# Clean staging
rm -rf "$STAGING"

SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo ""
echo "========================================"
echo "  Archive ready: $ARCHIVE"
echo "  Size: $SIZE"
echo "========================================"
echo ""
echo "To upload to Google Drive, download first:"
echo "  scp user@autodl-host:$(pwd)/$ARCHIVE ."

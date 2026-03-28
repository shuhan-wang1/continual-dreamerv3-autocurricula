#!/bin/bash
# =============================================================
# Package experiment results for Google Drive (Myriad version)
#
# Collects:
#   - All generated figures (PDF + PNG)
#   - A structured per-experiment JSON summary
#   - Raw online_metrics.jsonl for each run
#   - Experiment manifest, configs, and paper draft
#
# Output:  experiment_results/packaged/craftax_results_myriad_<date>.tar.gz
#
# Usage:   bash package_results_myriad.sh
# =============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

RESULTS_DIR="experiment_results/ablation"
FIGURES_DIR="experiment_results/figures"
PACK_DIR="experiment_results/packaged"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
STAGING="${PACK_DIR}/staging_${TIMESTAMP}"
ARCHIVE="${PACK_DIR}/craftax_results_myriad_${TIMESTAMP}.tar.gz"

PYTHON="${HOME}/.conda/envs/dreamer311/bin/python"

mkdir -p "$PACK_DIR" "$STAGING"

echo "========================================"
echo "  Package Myriad Experiment Results"
echo "  $(date)"
echo "========================================"

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

STAGING="$STAGING" $PYTHON << 'PYEOF'
import json, os, glob, sys
from collections import OrderedDict

RESULTS_DIR = "experiment_results/ablation"
STAGING = os.environ["STAGING"]
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
    return [f"achievement_{i}" for i in range(67)]


def load_jsonl_last_n(path, n=100):
    """Load last n records from a JSONL file."""
    lines = open(path).readlines()
    return [json.loads(l) for l in lines[-n:]]


def build_experiment_summary(exp_id, ach_names):
    """Build structured summary for one experiment config across all seeds."""
    seed_data = []

    for seed in SEEDS:
        seed_dir = os.path.join(RESULTS_DIR, f"{exp_id}_seed{seed}")
        if not os.path.isdir(seed_dir):
            continue

        # Find the nested DreamerV3 logdir
        summary_files = glob.glob(f"{seed_dir}/*/metrics_summary.json")
        online_files = glob.glob(f"{seed_dir}/*/online_metrics.jsonl")
        config_files = glob.glob(f"{seed_dir}/*/config.yaml")

        if not summary_files or not online_files:
            continue

        summary = json.load(open(summary_files[0]))
        online_records = open(online_files[0]).readlines()
        last_record = json.loads(online_records[-1])
        first_record = json.loads(online_records[0])

        # Per-achievement final rates with names
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

        # Per-tier aggregated rates
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

    # Aggregate across seeds
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


# ── Load manifest for descriptions ──
manifest_path = os.path.join(RESULTS_DIR, "experiment_manifest.json")
manifest = {}
if os.path.exists(manifest_path):
    raw = json.load(open(manifest_path))
    # Extract unique exp_id -> description
    for run_key, run_info in raw.get("runs", {}).items():
        eid = run_info.get("exp_id", "")
        if eid and eid not in manifest:
            manifest[eid] = {
                "group": run_info.get("group", ""),
                "description": run_info.get("desc", ""),
            }

# ── Discover experiments ──
ach_names = load_achievement_names()
exp_ids = set()
for p in sorted(os.listdir(RESULTS_DIR)):
    if os.path.isdir(os.path.join(RESULTS_DIR, p)):
        for seed in SEEDS:
            if p.endswith(f"_seed{seed}"):
                exp_ids.add(p[: -len(f"_seed{seed}")])
exp_ids = sorted(exp_ids)

# ── Build and save summaries ──
summaries_dir = os.path.join(STAGING, "summaries")
os.makedirs(summaries_dir, exist_ok=True)

all_experiments = []
for exp_id in exp_ids:
    summary = build_experiment_summary(exp_id, ach_names)
    if summary is None:
        continue

    # Add manifest info
    if exp_id in manifest:
        summary["group"] = manifest[exp_id]["group"]
        summary["description"] = manifest[exp_id]["description"]

    # Save individual experiment summary
    out_path = os.path.join(summaries_dir, f"{exp_id}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    all_experiments.append(summary)
    print(f"  {exp_id}: {summary['n_seeds']} seeds, "
          f"mean_ach_rate={summary['aggregated']['mean_achievement_rate']['mean']:.4f} "
          f"+/- {summary['aggregated']['mean_achievement_rate']['std']:.4f}")

# ── Save combined index ──
index = {
    "platform": "myriad",
    "results_type": "ablation_1M",
    "n_experiments": len(all_experiments),
    "achievement_names": ach_names,
    "tier_sizes": {f"tier_{t}": sum(1 for n in ach_names if TIER_MAP.get(n) == t) for t in range(5)},
    "experiments": [{
        "experiment_id": e["experiment_id"],
        "group": e.get("group", ""),
        "description": e.get("description", ""),
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
    exp_name=$(basename "$exp_dir")
    jsonl=$(find "$exp_dir" -name "online_metrics.jsonl" -print -quit 2>/dev/null)
    if [ -n "$jsonl" ]; then
        cp "$jsonl" "$RAW_DIR/${exp_name}_online_metrics.jsonl"
    fi
done
echo "  $(ls "$RAW_DIR"/*.jsonl 2>/dev/null | wc -l) JSONL files copied"

# ── 4. Copy supplementary files ───────────────────────────────
echo "[4/4] Copying supplementary files..."
cp -f "$RESULTS_DIR/experiment_manifest.json" "$STAGING/" 2>/dev/null || true
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
echo "To download:"
echo "  scp ucab327@myriad.rc.ucl.ac.uk:$(pwd)/$ARCHIVE ."

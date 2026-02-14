#!/usr/bin/env python3
"""
Fix achievement_depth and derived rolling-window fields in online_metrics.jsonl.

The raw data arrays (achievements, per_achievement_rates, etc.) are indexed by
Craftax Achievement enum value (0-21) and are already correct.  However,
achievement_depth was always -1 because the VectorDriver stripped log/ keys.

This script:
  1. Scans logs/craftax_dreamerv3-{cl|original}-{1|4|42}/ (same layout as
     analyze_metrics.py).
  2. Recomputes achievement_depth and all rolling-window derived fields
     (depth_mean, frontier_rate, personal_best_depth, agg_max_depth,
     per_task_max_depth, per_achievement_forgetting, aggregate_forgetting).
  3. Backs up originals to *.bak, then overwrites in-place.

Usage:
    cd experiment_results
    python remap_achievements.py
    python remap_achievements.py --logs-dir /path/to/logs
"""

import json
import os
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np

# ── Correct Craftax Achievement enum names (values 0-21) ─────────────
ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling",       # 0-3
    "collect_drink", "make_wood_pickaxe", "make_wood_sword",           # 4-6
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",    # 7-10
    "eat_plant", "defeat_skeleton", "make_stone_pickaxe",              # 11-13
    "make_stone_sword", "wake_up", "place_furnace", "collect_coal",    # 14-17
    "collect_iron", "collect_diamond", "make_iron_pickaxe",            # 18-20
    "make_iron_sword",                                                 # 21
]

NUM_ACH = len(ACHIEVEMENT_NAMES)  # 22

# Tier assignments for depth calculation
TIERS = {
    "collect_wood": 0, "place_table": 0, "eat_cow": 0, "collect_sapling": 0,
    "collect_drink": 0, "make_wood_pickaxe": 0, "make_wood_sword": 0,
    "place_plant": 0, "eat_plant": 0,
    "defeat_zombie": 1, "collect_stone": 1, "place_stone": 1,
    "defeat_skeleton": 1, "make_stone_pickaxe": 1, "make_stone_sword": 1,
    "wake_up": 1, "place_furnace": 1, "collect_coal": 1,
    "collect_iron": 2, "make_iron_pickaxe": 2, "make_iron_sword": 2,
    "collect_diamond": 3,
}


def compute_depth(achievements):
    """Compute max achievement tier from a boolean/int list."""
    max_tier = -1
    for i, achieved in enumerate(achievements):
        if achieved and i < NUM_ACH:
            tier = TIERS.get(ACHIEVEMENT_NAMES[i], -1)
            if tier > max_tier:
                max_tier = tier
    return max_tier


# ── Directory layout (mirrors analyze_metrics.py) ─────────────────────
METHODS = ["cl", "original"]
SEEDS = [1, 4, 42]


def fix_online_jsonl(path, window_size=100):
    """Fix a single online_metrics.jsonl in-place (with .bak backup)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"    [SKIP] empty file")
        return

    n_ach = len(records[0].get("achievements", []))
    print(f"    records: {len(records)}, achievement vector size: {n_ach}")

    # Rolling-window state for recomputation
    depth_window = deque(maxlen=window_size)
    max_depth_lifetime = -1
    personal_best = -1
    ach_history = deque(maxlen=window_size)
    peak_rates = np.zeros(NUM_ACH, dtype=np.float64)

    depth_changed = 0
    for rec in records:
        ach = rec.get("achievements", [])

        # Recompute achievement_depth
        depth = compute_depth(ach)
        if depth != rec.get("achievement_depth", -1):
            depth_changed += 1
        rec["achievement_depth"] = depth

        # Rolling depth stats
        depth_window.append(depth)
        if depth > max_depth_lifetime:
            max_depth_lifetime = depth
        if depth > personal_best:
            personal_best = depth

        rec["depth_mean"] = float(np.mean(list(depth_window)))
        rec["personal_best_depth"] = personal_best
        rec["agg_max_depth"] = float(max_depth_lifetime)
        rec["per_task_max_depth"] = [max_depth_lifetime]

        # Recompute frontier_rate and forgetting
        ach_bool = np.array(ach[:NUM_ACH], dtype=bool)
        ach_history.append(ach_bool)
        if max_depth_lifetime >= 0:
            arr = np.array(list(ach_history), dtype=np.float32)
            current_rates = arr.mean(axis=0)
            frontier_idx = [
                i for i in range(NUM_ACH)
                if TIERS.get(ACHIEVEMENT_NAMES[i], -1) == max_depth_lifetime
            ]
            if frontier_idx:
                rec["frontier_rate"] = float(
                    np.mean([current_rates[i] for i in frontier_idx]))
            else:
                rec["frontier_rate"] = 0.0

            peak_rates = np.maximum(peak_rates, current_rates)
            forgetting = np.maximum(0.0, peak_rates - current_rates)
            rec["per_achievement_forgetting"] = forgetting.tolist()
            rec["aggregate_forgetting"] = float(forgetting.mean())
        else:
            rec["frontier_rate"] = 0.0

    # Backup original then overwrite
    backup = str(path) + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"    backup: {backup}")

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"    achievement_depth changed: {depth_changed}/{len(records)}")
    print(f"    final personal_best_depth: {personal_best}")


def fix_summary_json(path):
    """Fix a single metrics_summary.json in-place (with .bak backup)."""
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    summary["achievement_names"] = ACHIEVEMENT_NAMES

    # Recompute depths from rates
    rates = summary.get("per_task_achievement_rates", [[]])
    if rates and rates[0]:
        max_depth = -1
        for i, rate in enumerate(rates[0]):
            if rate > 0.01 and i < NUM_ACH:
                tier = TIERS.get(ACHIEVEMENT_NAMES[i], -1)
                max_depth = max(max_depth, tier)
        summary["max_achievement_depth"] = max_depth
        summary["personal_best_depth"] = max_depth

    backup = str(path) + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"    fixed metrics_summary.json")


def main():
    logs_dir = Path("logs")
    if "--logs-dir" in sys.argv:
        idx = sys.argv.index("--logs-dir")
        if idx + 1 < len(sys.argv):
            logs_dir = Path(sys.argv[idx + 1])

    print("=" * 70)
    print("Craftax Achievement Fix Tool")
    print("=" * 70)
    print(f"\nLogs directory: {logs_dir.resolve()}")
    print(f"\nCorrect achievement names (Craftax enum 0-21):")
    for i, name in enumerate(ACHIEVEMENT_NAMES):
        print(f"  [{i:2d}] {name}")
    print()

    if not logs_dir.exists():
        print(f"ERROR: {logs_dir} does not exist.")
        print(f"  Expected: {logs_dir}/craftax_dreamerv3-cl-1/online_metrics.jsonl")
        return

    found = 0
    for method in METHODS:
        for seed in SEEDS:
            folder = logs_dir / f"craftax_dreamerv3-{method}-{seed}"
            if not folder.exists():
                print(f"  [SKIP] {folder} -- not found")
                continue

            found += 1
            print(f"\n  === {folder.name} ===")

            online_path = folder / "online_metrics.jsonl"
            if online_path.exists():
                fix_online_jsonl(online_path)
            else:
                print(f"    [SKIP] online_metrics.jsonl not found")

            summary_path = folder / "metrics_summary.json"
            if summary_path.exists():
                fix_summary_json(summary_path)
            else:
                print(f"    [SKIP] metrics_summary.json not found")

    print(f"\n{'=' * 70}")
    print(f"Processed {found}/6 run folders.")
    if found > 0:
        print("Originals backed up to *.bak files.")
        print("You can now run analyze_metrics.py directly.")
    print("=" * 70)


if __name__ == "__main__":
    main()

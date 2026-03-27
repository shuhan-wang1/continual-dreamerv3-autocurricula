#!/usr/bin/env python3
"""Load all records from each 10m experiment's online_metrics.jsonl and print:
1. Per-run summary (records, step, mean/last achievement count, mean/last return)
2. Best trajectory: which achievements were unlocked
3. Per-achievement unlock probability across all timesteps
"""

import json
import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent / "10m"

# 67 Craftax achievements in enum-value order (matches the boolean vector in JSONL)
ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling",
    "collect_drink", "make_wood_pickaxe", "make_wood_sword",
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",
    "eat_plant", "defeat_skeleton", "collect_coal", "make_stone_pickaxe",
    "make_stone_sword", "wake_up", "place_furnace", "collect_iron",
    "make_iron_pickaxe", "make_iron_sword", "collect_diamond",
    "make_diamond_pickaxe", "make_diamond_sword",
    "make_iron_armour", "make_diamond_armour",
    "make_arrow", "make_torch", "place_torch",
    "eat_bat", "eat_snail", "find_bow", "fire_bow",
    "collect_sapphire", "collect_ruby",
    "enter_gnomish_mines", "enter_dungeon", "enter_sewers",
    "enter_vault", "enter_troll_mines",
    "defeat_gnome_warrior", "defeat_gnome_archer",
    "defeat_orc_solider", "defeat_orc_mage",
    "defeat_lizard", "defeat_kobold",
    "learn_fireball", "cast_fireball", "learn_iceball", "cast_iceball",
    "open_chest", "drink_potion", "enchant_sword", "enchant_armour",
    "enter_fire_realm", "enter_ice_realm", "enter_graveyard",
    "defeat_troll", "defeat_deep_thing", "defeat_pigman",
    "defeat_fire_elemental", "defeat_frost_troll", "defeat_ice_elemental",
    "defeat_knight", "defeat_archer",
    "damage_necromancer", "defeat_necromancer",
]


def get_achievement_names(n: int) -> list[str]:
    if n == len(ACHIEVEMENT_NAMES):
        return ACHIEVEMENT_NAMES
    return [f"achievement_{i}" for i in range(n)]


def load_jsonl(path: pathlib.Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_run(run_name: str, records: list[dict]):
    if not records:
        print(f"\n{'='*80}")
        print(f"  {run_name}  (empty)")
        return

    ach_vectors = [r.get("achievements", []) for r in records]
    n_ach = len(ach_vectors[0]) if ach_vectors else 0
    names = get_achievement_names(n_ach)

    ach_counts = [sum(1 for v in vec if v) for vec in ach_vectors]
    return_means = [r.get("return_mean", 0.0) for r in records]

    mean_ach = sum(ach_counts) / len(ach_counts)
    last_ach = ach_counts[-1]
    mean_ret = sum(return_means) / len(return_means)
    last_ret = return_means[-1]
    last_step = records[-1].get("step", "?")

    # --- Header ---
    print(f"\n{'='*80}")
    print(f"  {run_name}")
    print(f"{'='*80}")
    print(f"  Records: {len(records)}   LastStep: {last_step}")
    print(f"  MeanAch: {mean_ach:.2f}   LastAch: {last_ach}")
    print(f"  MeanReturn: {mean_ret:.4f}   LastReturn: {last_ret:.4f}")

    # --- Best trajectory ---
    best_idx = max(range(len(ach_counts)), key=lambda i: ach_counts[i])
    best_vec = ach_vectors[best_idx]
    best_step = records[best_idx].get("step", "?")
    unlocked = [names[i] for i, v in enumerate(best_vec) if v]
    print(f"\n  Best trajectory (step {best_step}, {len(unlocked)}/{n_ach} unlocked):")
    for name in unlocked:
        print(f"    + {name}")

    # --- Per-achievement unlock probability ---
    n_records = len(ach_vectors)
    unlock_counts = [0] * n_ach
    for vec in ach_vectors:
        for i, v in enumerate(vec):
            if v:
                unlock_counts[i] += 1

    probs = [(names[i], unlock_counts[i] / n_records) for i in range(n_ach)]
    probs.sort(key=lambda x: -x[1])

    print(f"\n  Achievement unlock probability (across {n_records} timesteps):")
    print(f"  {'Achievement':<35} {'Prob':>8} {'Count':>8}")
    print(f"  {'-'*55}")
    for name, prob in probs:
        cnt = int(prob * n_records)
        bar = "#" * int(prob * 30)
        print(f"  {name:<35} {prob:>7.1%} {cnt:>8}  {bar}")


def main():
    if not BASE_DIR.is_dir():
        print(f"Directory not found: {BASE_DIR}")
        sys.exit(1)

    jsonl_files = sorted(BASE_DIR.glob("**/online_metrics.jsonl"))
    if not jsonl_files:
        print(f"No online_metrics.jsonl found under {BASE_DIR}")
        sys.exit(1)

    for path in jsonl_files:
        run_name = str(path.relative_to(BASE_DIR))
        records = load_jsonl(path)
        analyze_run(run_name, records)


if __name__ == "__main__":
    main()

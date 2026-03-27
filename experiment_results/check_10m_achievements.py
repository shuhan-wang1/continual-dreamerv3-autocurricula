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

# 67 Craftax achievements indexed by enum value
# Source: https://github.com/MichaelTMatthews/Craftax/blob/main/craftax/craftax/constants.py
ACHIEVEMENT_NAMES = [
    "collect_wood",          # 0
    "place_table",           # 1
    "eat_cow",               # 2
    "collect_sapling",       # 3
    "collect_drink",         # 4
    "make_wood_pickaxe",     # 5
    "make_wood_sword",       # 6
    "place_plant",           # 7
    "defeat_zombie",         # 8
    "collect_stone",         # 9
    "place_stone",           # 10
    "eat_plant",             # 11
    "defeat_skeleton",       # 12
    "make_stone_pickaxe",    # 13
    "make_stone_sword",      # 14
    "wake_up",               # 15
    "place_furnace",         # 16
    "collect_coal",          # 17
    "collect_iron",          # 18
    "collect_diamond",       # 19
    "make_iron_pickaxe",     # 20
    "make_iron_sword",       # 21
    "make_arrow",            # 22
    "make_torch",            # 23
    "place_torch",           # 24
    "make_diamond_sword",    # 25
    "make_iron_armour",      # 26
    "make_diamond_armour",   # 27
    "enter_gnomish_mines",   # 28
    "enter_dungeon",         # 29
    "enter_sewers",          # 30
    "enter_vault",           # 31
    "enter_troll_mines",     # 32
    "enter_fire_realm",      # 33
    "enter_ice_realm",       # 34
    "enter_graveyard",       # 35
    "defeat_gnome_warrior",  # 36
    "defeat_gnome_archer",   # 37
    "defeat_orc_solider",    # 38
    "defeat_orc_mage",       # 39
    "defeat_lizard",         # 40
    "defeat_kobold",         # 41
    "defeat_troll",          # 42
    "defeat_deep_thing",     # 43
    "defeat_pigman",         # 44
    "defeat_fire_elemental", # 45
    "defeat_frost_troll",    # 46
    "defeat_ice_elemental",  # 47
    "damage_necromancer",    # 48
    "defeat_necromancer",    # 49
    "eat_bat",               # 50
    "eat_snail",             # 51
    "find_bow",              # 52
    "fire_bow",              # 53
    "collect_sapphire",      # 54
    "learn_fireball",        # 55
    "cast_fireball",         # 56
    "learn_iceball",         # 57
    "cast_iceball",          # 58
    "collect_ruby",          # 59
    "make_diamond_pickaxe",  # 60
    "open_chest",            # 61
    "drink_potion",          # 62
    "enchant_sword",         # 63
    "enchant_armour",        # 64
    "defeat_knight",         # 65
    "defeat_archer",         # 66
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

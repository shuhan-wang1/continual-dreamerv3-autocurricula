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


def get_achievement_names(n: int) -> list[str]:
    """Try to load names from Craftax enum; fall back to index-based names."""
    try:
        from craftax.craftax_env import CraftaxAchievement
        names = [a.name.lower() for a in sorted(CraftaxAchievement, key=lambda a: a.value)]
        if len(names) == n:
            return names
    except Exception:
        pass
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

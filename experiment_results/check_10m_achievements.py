#!/usr/bin/env python3
"""Load all records from each 10m experiment's online_metrics.jsonl
and print mean return-mean-achievements (avg number of True in achievements)."""

import json
import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent / "10m"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    if not BASE_DIR.is_dir():
        print(f"Directory not found: {BASE_DIR}")
        sys.exit(1)

    jsonl_files = sorted(BASE_DIR.glob("*/online_metrics.jsonl"))
    if not jsonl_files:
        print(f"No online_metrics.jsonl found under {BASE_DIR}")
        sys.exit(1)

    print(f"{'Run':<40} {'Records':>8} {'LastStep':>10} {'MeanAch':>10} {'LastAch':>10} {'MeanReturn':>12} {'LastReturn':>12}")
    print("-" * 106)

    for path in jsonl_files:
        run_name = path.parent.name
        records = load_jsonl(path)
        if not records:
            print(f"{run_name:<40} {'(empty)':>8}")
            continue

        ach_counts = [sum(1 for v in r.get("achievements", []) if v) for r in records]
        return_means = [r.get("return_mean", 0.0) for r in records]

        mean_ach = sum(ach_counts) / len(ach_counts)
        last_ach = ach_counts[-1]
        mean_ret = sum(return_means) / len(return_means)
        last_ret = return_means[-1]
        last_step = records[-1].get("step", "?")

        print(f"{run_name:<40} {len(records):>8} {last_step:>10} {mean_ach:>10.2f} {last_ach:>10} {mean_ret:>12.4f} {last_ret:>12.4f}")


if __name__ == "__main__":
    main()

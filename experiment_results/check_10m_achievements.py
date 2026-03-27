#!/usr/bin/env python3
"""Read the last record of each 10m experiment's online_metrics.jsonl
and print the return-mean-achievements (number of True in achievements vector)."""

import json
import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent / "10m"


def last_json_line(path: pathlib.Path) -> dict | None:
    """Return the last non-empty line of a JSONL file as a dict."""
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    return json.loads(last) if last else None


def main():
    if not BASE_DIR.is_dir():
        print(f"Directory not found: {BASE_DIR}")
        sys.exit(1)

    jsonl_files = sorted(BASE_DIR.glob("*/online_metrics.jsonl"))
    if not jsonl_files:
        print(f"No online_metrics.jsonl found under {BASE_DIR}")
        sys.exit(1)

    print(f"{'Run':<40} {'Step':>10} {'Achievements':>14} {'Return Mean':>12}")
    print("-" * 80)

    for path in jsonl_files:
        run_name = path.parent.name
        record = last_json_line(path)
        if record is None:
            print(f"{run_name:<40} {'(empty)':>10}")
            continue
        achievements = record.get("achievements", [])
        num_true = sum(1 for v in achievements if v)
        step = record.get("step", "?")
        return_mean = record.get("return_mean", float("nan"))
        print(f"{run_name:<40} {step:>10} {num_true:>14} {return_mean:>12.4f}")


if __name__ == "__main__":
    main()

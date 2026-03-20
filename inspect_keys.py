#!/usr/bin/env python3
"""Quick inspection: dump all unique keys from JSONL metrics files."""
import json, sys, os
from collections import Counter

def inspect(path):
    if not os.path.exists(path):
        print(f"  {path}: NOT FOUND")
        return
    keys = Counter()
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                for k in r:
                    keys[k] += 1
                n += 1
            except json.JSONDecodeError:
                continue
    print(f"\n  {path}: {n} records, {len(keys)} unique keys")
    # Group by category
    mask_keys = {k: v for k, v in keys.items() if "mask" in k.lower()}
    loss_keys = {k: v for k, v in keys.items() if "loss" in k.lower()}
    reward_keys = {k: v for k, v in keys.items() if any(x in k.lower() for x in ("reward", "r_intr", "r_spatial", "r_craft", "rew"))}
    episode_keys = {k: v for k, v in keys.items() if k.startswith("episode/")}
    other_keys = {k: v for k, v in keys.items() if k not in mask_keys and k not in loss_keys and k not in reward_keys and k not in episode_keys}

    for label, group in [("MASK", mask_keys), ("LOSS", loss_keys), ("REWARD", reward_keys), ("EPISODE/", episode_keys), ("OTHER", other_keys)]:
        if group:
            print(f"    [{label}] ({len(group)} keys)")
            for k, cnt in sorted(group.items()):
                pct = cnt / n * 100
                print(f"      {k:<55} {cnt:>6} records ({pct:>5.1f}%)")

if __name__ == "__main__":
    logdir = sys.argv[1] if len(sys.argv) > 1 else "."
    if os.path.isfile(logdir):
        logdir = os.path.dirname(logdir)
    # Find nested craftax_ dir
    for entry in os.listdir(logdir):
        sub = os.path.join(logdir, entry)
        if os.path.isdir(sub) and entry.startswith("craftax_"):
            logdir = sub
            break
    print(f"Logdir: {logdir}")
    inspect(os.path.join(logdir, "online_metrics.jsonl"))
    inspect(os.path.join(logdir, "metrics.jsonl"))

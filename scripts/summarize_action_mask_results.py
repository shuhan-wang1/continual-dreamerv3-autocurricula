#!/usr/bin/env python3
"""
Summarize action mask experiment results.
Usage: python scripts/summarize_action_mask_results.py logs/action_mask_YYYYMMDD_HHMMSS
"""
import json
import sys
from pathlib import Path


def main():
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs")
    if not log_dir.exists():
        print(f"Dir not found: {log_dir}")
        return

    print(f"\n=== Action Mask Results: {log_dir} ===\n")
    for sub in sorted(log_dir.iterdir()):
        if not sub.is_dir():
            continue
        metrics_path = sub / "metrics.jsonl"
        summary_path = sub / "metrics_summary.json"
        if not metrics_path.exists():
            continue
        name = sub.name
        # Parse last N lines for recent scores
        lines = metrics_path.read_text().strip().split("\n")
        if not lines:
            continue
        scores = []
        for line in lines[-100:]:
            try:
                d = json.loads(line)
                if "episode/score" in d:
                    scores.append(d["episode/score"])
            except Exception:
                pass
        mean_score = sum(scores) / len(scores) if scores else 0
        print(f"{name}: last_100_ep_mean_score={mean_score:.3f} (n={len(scores)})")
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            print(f"  summary: mean_return={s.get('mean_return', 'N/A')}, max_depth={s.get('max_achievement_depth', 'N/A')}")
    print()


if __name__ == "__main__":
    main()

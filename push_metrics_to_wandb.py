#!/usr/bin/env python3
"""
Push existing ablation experiment metrics to W&B.

Reads online_metrics.jsonl from each experiment directory and uploads
all scalar metrics to W&B, recreating runs with the correct
project / group / name / config structure.

Usage:
    python push_metrics_to_wandb.py                 # upload all
    python push_metrics_to_wandb.py --dry_run       # preview only
    python push_metrics_to_wandb.py --only A2_p2e   # filter by experiment ID prefix
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import wandb
import yaml

# ---------------------------------------------------------------------------
# Constants (must match run_ablation.py / train_craftax.py)
# ---------------------------------------------------------------------------
BASE_LOGDIR = "experiment_results/ablation"
WANDB_PROJECT = "craftax-ablation"
# The default env for single-task ablation training
DEFAULT_ENV = "CraftaxSymbolic-v1"

# Keys that are lists / arrays — skip for wandb scalar logging
SKIP_KEYS = frozenset({
    "achievements", "per_achievement_rates", "per_achievement_forgetting",
    "score_distribution", "per_task_return_mean", "per_task_success_rate",
    "per_task_max_return", "per_task_max_depth", "per_task_episodes",
    "per_task_achievement_rates", "replay/depth_distribution",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_run_dir(dirname):
    """Extract (exp_id, seed) from a directory name like 'A2_p2e_seed42'."""
    m = re.match(r"^(.+)_seed(\d+)$", dirname)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def find_online_metrics(run_dir):
    """Find online_metrics.jsonl in run_dir or its nested craftax_* subdir."""
    # Direct
    p = os.path.join(run_dir, "online_metrics.jsonl")
    if os.path.exists(p):
        return p
    # Nested craftax_<tag><seed>/
    for entry in os.listdir(run_dir):
        nested = os.path.join(run_dir, entry, "online_metrics.jsonl")
        if os.path.exists(nested):
            return nested
    return None


def find_config_yaml(run_dir):
    """Find config.yaml in run_dir or nested subdir."""
    p = os.path.join(run_dir, "config.yaml")
    if os.path.exists(p):
        return p
    for entry in os.listdir(run_dir):
        nested = os.path.join(run_dir, entry, "config.yaml")
        if os.path.exists(nested):
            return nested
    return None


def load_config(config_path):
    """Load config.yaml as a flat dict for wandb.config."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def read_jsonl_records(filepath):
    """Yield (record_dict) for each line in a JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"  WARN: bad JSON at line {lineno}, skipping")


def record_to_scalars(record):
    """Extract only numeric scalars from a CraftaxMetrics record."""
    out = {}
    for k, v in record.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, (int, float)):
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Push ablation metrics to W&B")
    parser.add_argument("--base_logdir", default=BASE_LOGDIR)
    parser.add_argument("--wandb_project", default=WANDB_PROJECT)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be uploaded without touching W&B.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated experiment ID prefixes to include.")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (team/user). Defaults to your default entity.")
    args = parser.parse_args()

    base = Path(args.base_logdir)
    if not base.exists():
        print(f"ERROR: {base} does not exist.")
        sys.exit(1)

    # Load manifest for metadata
    manifest_path = base / "experiment_manifest.json"
    manifest_runs = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest_runs = json.load(f).get("runs", {})

    # Discover all run directories
    only_prefixes = [t.strip() for t in args.only.split(",")] if args.only else None
    runs_to_upload = []

    for entry in sorted(os.listdir(base)):
        entry_path = base / entry
        if not entry_path.is_dir():
            continue
        exp_id, seed = parse_run_dir(entry)
        if exp_id is None:
            continue
        if only_prefixes:
            if not any(exp_id == p or exp_id.startswith(p) for p in only_prefixes):
                continue

        metrics_path = find_online_metrics(str(entry_path))
        if metrics_path is None:
            continue

        config_path = find_config_yaml(str(entry_path))
        run_key = f"{exp_id}_seed{seed}"
        manifest_info = manifest_runs.get(run_key, {})

        runs_to_upload.append({
            "run_key": run_key,
            "exp_id": exp_id,
            "seed": seed,
            "group": manifest_info.get("group", exp_id[0]),
            "desc": manifest_info.get("desc", ""),
            "metrics_path": metrics_path,
            "config_path": config_path,
        })

    if not runs_to_upload:
        print("No runs with metrics found.")
        return

    print(f"Found {len(runs_to_upload)} runs with metrics to upload.\n")

    for i, run_info in enumerate(runs_to_upload, 1):
        exp_id = run_info["exp_id"]
        seed = run_info["seed"]
        tag = f"{exp_id}{seed}"
        run_name = f"DreamerV3_craftax_single-env={DEFAULT_ENV}_{tag}"
        wandb_group = exp_id

        # Count records
        n_records = sum(1 for _ in read_jsonl_records(run_info["metrics_path"]))

        # Preview last step
        last_step = 0
        for rec in read_jsonl_records(run_info["metrics_path"]):
            last_step = rec.get("step", last_step)

        print(f"[{i}/{len(runs_to_upload)}] {run_info['run_key']}")
        print(f"  Records: {n_records}  |  Last step: {last_step:,}")
        print(f"  W&B: project={args.wandb_project}  group={wandb_group}  name={run_name}")

        if args.dry_run:
            print(f"  [DRY RUN] Would upload {n_records} records.\n")
            continue

        # Load training config
        config_dict = {}
        if run_info["config_path"]:
            config_dict = load_config(run_info["config_path"])
        config_dict["exp_id"] = exp_id
        config_dict["seed"] = seed
        config_dict["desc"] = run_info["desc"]

        # Create W&B run
        run = wandb.init(
            project=args.wandb_project,
            entity=args.entity,
            group=wandb_group,
            name=run_name,
            id=f"{wandb_group}_{tag}",
            config=config_dict,
            reinit=True,
            resume="allow",
            tags=["backfill", f"group-{run_info['group']}", f"seed-{seed}"],
            notes=f"Backfilled from {run_info['metrics_path']}. {run_info['desc']}",
        )

        # Upload all records
        logged = 0
        t0 = time.time()
        for record in read_jsonl_records(run_info["metrics_path"]):
            step = record.get("step", 0)
            scalars = record_to_scalars(record)
            if scalars:
                wandb.log(scalars, step=step)
                logged += 1

        elapsed = time.time() - t0
        print(f"  Uploaded {logged} records in {elapsed:.1f}s")

        wandb.finish()
        print()

    print("Done.")


if __name__ == "__main__":
    main()

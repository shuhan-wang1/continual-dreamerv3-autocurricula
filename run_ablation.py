#!/usr/bin/env python3
"""
Ablation & Comparative Experiment Runner
=========================================
Runs a systematic ablation study for:
  1. Spatial-counting + craft-novelty intrinsic rewards (Eq.10-13)
  2. P2E ensemble disagreement
  3. NLR replay with corrected novelty formula (Eq.9)

Experiment groups:
  A  Core comparison      (baseline, P2E, intrinsic, P2E+intrinsic)
  B  Component ablation   (craft_weight sensitivity)
  C  Reward scale          (alpha_i sensitivity: 0.01, 0.3, 1.0 around default 0.1)
  D  Replay strategy       (50:50, NLR, NLU — privileged & non-privileged)
  E  Final model           (NLR replay + Spatial+Craft intrinsic)

Output directory structure:
  experiment_results/ablation/
  ├── experiment_manifest.json        # run metadata & status for all experiments
  ├── A1_baseline_seed1/
  │   └── craftax_A1_baseline/        # DreamerV3 logdir
  │       ├── config.yaml             # full training config snapshot
  │       ├── metrics.jsonl           # DreamerV3 training metrics (loss, reward, etc.)
  │       ├── online_metrics.jsonl    # per-episode CL metrics (achievements, forgetting)
  │       ├── metrics_summary.json    # aggregated achievement stats
  │       ├── nlr_args.yaml           # NLR/NLU config (if enabled)
  │       └── ckpt/                   # model weights (kept after run)
  ├── A1_baseline_seed4/
  │   └── ...
  └── ...

Usage:
  python run_ablation.py                          # run all 42 experiments
  python run_ablation.py --dry_run                # print commands only
  python run_ablation.py --only A                 # run group A only (12 runs)
  python run_ablation.py --only A1,A2             # run specific experiments
  python run_ablation.py --skip D                 # skip group D
  python run_ablation.py --wandb_mode disabled    # no wandb logging
  python run_ablation.py --gpu 0                  # use GPU 0
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# Default training hyperparameters (applied to every experiment)
# ============================================================================
DEFAULTS = {
    "steps":           1_000_000,
    "batch_size":      16,
    "batch_length":    64,
    "envs":            16,
    "model_size":      "25m",
    "wandb_proj_name": "craftax-ablation",
    "wandb_mode":      "online",
}

SEEDS = [1, 4, 42]

BASE_LOGDIR = "experiment_results/ablation"

# ============================================================================
# Experiment registry
# ============================================================================
# Each entry:  (experiment_id, description, {extra CLI args})
# CLI args are *on top of* DEFAULTS.  Use True/False for store_true flags.

EXPERIMENTS = OrderedDict()

# ---------- Group A: Core Comparison ----------
EXPERIMENTS["A1_baseline"] = {
    "group": "A",
    "desc": "Pure DreamerV3 (no P2E, no intrinsic)",
    "args": {
        "no_plan2explore": True,
    },
}
EXPERIMENTS["A2_p2e"] = {
    "group": "A",
    "desc": "DreamerV3 + P2E (current default)",
    "args": {
        "plan2explore": True,
    },
}
EXPERIMENTS["A3_intrinsic"] = {
    "group": "A",
    "desc": "DreamerV3 + Spatial+Craft intrinsic (no P2E)",
    "args": {
        "no_plan2explore": True,
        "intrinsic_spatial": True,
        "alpha_i": 0.1,
        "alpha_e": 1.0,
        "craft_weight": 1.0,
    },
}
EXPERIMENTS["A4_p2e_intrinsic"] = {
    "group": "A",
    "desc": "DreamerV3 + P2E + Spatial+Craft intrinsic",
    "args": {
        "plan2explore": True,
        "intrinsic_spatial": True,
        "alpha_i": 0.1,
        "alpha_e": 1.0,
        "craft_weight": 1.0,
    },
}

# ---------- Group B: Component Ablation (craft_weight) ----------
EXPERIMENTS["B1_spatial_only"] = {
    "group": "B",
    "desc": "Spatial counting only (craft_weight=0, no P2E)",
    "args": {
        "no_plan2explore": True,
        "intrinsic_spatial": True,
        "alpha_i": 0.1,
        "alpha_e": 1.0,
        "craft_weight": 0.0,
    },
}
EXPERIMENTS["B2_craft_light"] = {
    "group": "B",
    "desc": "Light craft-novelty (craft_weight=0.5, no P2E)",
    "args": {
        "no_plan2explore": True,
        "intrinsic_spatial": True,
        "alpha_i": 0.1,
        "alpha_e": 1.0,
        "craft_weight": 0.5,
    },
}
EXPERIMENTS["B3_craft_heavy"] = {
    "group": "B",
    "desc": "Heavy craft-novelty (craft_weight=2.0, no P2E)",
    "args": {
        "no_plan2explore": True,
        "intrinsic_spatial": True,
        "alpha_i": 0.1,
        "alpha_e": 1.0,
        "craft_weight": 2.0,
    },
}

# ---------- Group C: Reward Scale Sensitivity ----------
EXPERIMENTS["C1_tiny_intrinsic"] = {
    "group": "C",
    "desc": "Tiny intrinsic (alpha_i=0.01, alpha_e=1.0, no P2E)",
    "args": {
        "no_plan2explore": True,
        "intrinsic_spatial": True,
        "alpha_i": 0.01,
        "alpha_e": 1.0,
        "craft_weight": 1.0,
    },
}
EXPERIMENTS["C2_high_intrinsic"] = {
    "group": "C",
    "desc": "High intrinsic (alpha_i=0.3, alpha_e=1.0, no P2E)",
    "args": {
        "no_plan2explore": True,
        "intrinsic_spatial": True,
        "alpha_i": 0.3,
        "alpha_e": 1.0,
        "craft_weight": 1.0,
    },
}
EXPERIMENTS["C3_equal_weight"] = {
    "group": "C",
    "desc": "Equal weight (alpha_i=1.0, alpha_e=1.0, no P2E)",
    "args": {
        "no_plan2explore": True,
        "intrinsic_spatial": True,
        "alpha_i": 1.0,
        "alpha_e": 1.0,
        "craft_weight": 1.0,
    },
}

# ---------- Group D: Replay Strategy Comparison ----------
# Pure baseline (no P2E, no intrinsic) — isolate the replay strategy effect.
# The default 50:50 reservoir+recent baseline is already covered by A1_baseline,
# so Group D only tests the 4 NLR/NLU variants against it.
# Group E combines the best replay strategy (NLR) with intrinsic rewards.

EXPERIMENTS["D1_nlr"] = {
    "group": "D",
    "desc": "NLR non-privileged (2D grid novelty-learnability-recency)",
    "args": {
        "no_plan2explore": True,
        "nlr_sampling": True,
    },
}
EXPERIMENTS["D2_nlu"] = {
    "group": "D",
    "desc": "NLU non-privileged (2D grid novelty-learnability-uniform)",
    "args": {
        "no_plan2explore": True,
        "nlu_sampling": True,
    },
}
EXPERIMENTS["D3_nlr_priv"] = {
    "group": "D",
    "desc": "NLR privileged (per-achievement novelty-learnability-recency)",
    "args": {
        "no_plan2explore": True,
        "nlr_privileged_sampling": True,
    },
}
EXPERIMENTS["D4_nlu_priv"] = {
    "group": "D",
    "desc": "NLU privileged (per-achievement novelty-learnability-uniform)",
    "args": {
        "no_plan2explore": True,
        "nlu_privileged_sampling": True,
    },
}

# ---------- Group E: Final Model (NLR + Intrinsic) ----------
EXPERIMENTS["E1_nlr_intrinsic"] = {
    "group": "E",
    "desc": "Final model: NLR replay + Spatial+Craft intrinsic (no P2E)",
    "args": {
        "no_plan2explore": True,
        "nlr_sampling": True,
        "intrinsic_spatial": True,
        "alpha_i": 0.1,
        "alpha_e": 1.0,
        "craft_weight": 1.0,
    },
}


# ============================================================================
# Helpers
# ============================================================================

def build_command(exp_id, exp_cfg, seed, base_logdir, wandb_mode, extra_defaults=None):
    """Build the full CLI command list for subprocess."""
    # Merge defaults
    merged = dict(DEFAULTS)
    if extra_defaults:
        merged.update(extra_defaults)

    # Experiment-specific overrides
    exp_args = exp_cfg["args"]

    # Build logdir: experiment_results/ablation/{exp_id}_seed{seed}
    logdir = os.path.join(base_logdir, f"{exp_id}_seed{seed}")

    cmd = [sys.executable, "train_craftax.py"]

    # --- Fixed args ---
    cmd += ["--seed", str(seed)]
    cmd += ["--tag", f"{exp_id}"]
    cmd += ["--logdir", logdir]
    cmd += ["--steps", str(merged["steps"])]
    cmd += ["--batch_size", str(merged["batch_size"])]
    cmd += ["--batch_length", str(merged["batch_length"])]
    cmd += ["--envs", str(merged["envs"])]
    cmd += ["--model_size", str(merged["model_size"])]

    # wandb
    cmd += ["--wandb_proj_name", merged["wandb_proj_name"]]
    cmd += ["--wandb_group", exp_id]
    cmd += ["--wandb_mode", wandb_mode]

    # --- Experiment-specific args ---
    for key, val in exp_args.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
            # If False, we skip (it's the default)
        else:
            cmd += [flag, str(val)]

    return cmd, logdir


def is_run_complete(logdir):
    """Check if a run already completed by looking for metrics in nested logdir."""
    if not os.path.exists(logdir):
        return False
    # DreamerV3 writes into a nested dir: logdir/craftax_<tag>/
    # Check both the direct logdir and nested subdirectories
    for search_dir in [logdir] + [
        os.path.join(logdir, d) for d in os.listdir(logdir)
        if os.path.isdir(os.path.join(logdir, d))
    ]:
        metrics_path = os.path.join(search_dir, "metrics.jsonl")
        if os.path.exists(metrics_path) and os.path.getsize(metrics_path) > 100:
            return True
    return False


def cleanup_run(logdir):
    """Remove replay buffer after each run to save disk. Keep checkpoints and logs."""
    # The actual DreamerV3 logdir is nested: logdir/craftax_{tag}{seed}/
    # Check both the direct logdir and any nested subdirectories
    dirs_to_check = [logdir]
    # Also scan for nested craftax_* dirs
    if os.path.exists(logdir):
        for entry in os.listdir(logdir):
            subpath = os.path.join(logdir, entry)
            if os.path.isdir(subpath):
                dirs_to_check.append(subpath)

    cleaned_bytes = 0
    for dirpath in dirs_to_check:
        # Only remove replay buffer; keep ckpt (model weights) and all logs
        target = os.path.join(dirpath, "replay")
        if os.path.exists(target):
            for root, dirs, files in os.walk(target):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        cleaned_bytes += os.path.getsize(fp)
                    except OSError:
                        pass
            shutil.rmtree(target, ignore_errors=True)

    return cleaned_bytes


def format_duration(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_bytes(n):
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def filter_experiments(experiments, only=None, skip=None):
    """Filter experiments based on --only and --skip flags."""
    if only:
        tokens = [t.strip() for t in only.split(",")]
        filtered = OrderedDict()
        for exp_id, cfg in experiments.items():
            # Match by exact ID or by group prefix
            for tok in tokens:
                if exp_id == tok or exp_id.startswith(tok) or cfg["group"] == tok:
                    filtered[exp_id] = cfg
                    break
        return filtered

    if skip:
        tokens = [t.strip() for t in skip.split(",")]
        filtered = OrderedDict()
        for exp_id, cfg in experiments.items():
            should_skip = False
            for tok in tokens:
                if exp_id == tok or exp_id.startswith(tok) or cfg["group"] == tok:
                    should_skip = True
                    break
            if not should_skip:
                filtered[exp_id] = cfg
        return filtered

    return experiments


def print_experiment_table(experiments, seeds):
    """Print a formatted table of all experiments."""
    total = len(experiments) * len(seeds)
    print("\n" + "=" * 80)
    print("  ABLATION EXPERIMENT SUITE")
    print("=" * 80)
    print(f"  Total: {len(experiments)} configs x {len(seeds)} seeds = {total} runs")
    print(f"  Seeds: {seeds}")
    print("-" * 80)
    print(f"  {'ID':<25} {'Group':>5}  {'Description'}")
    print("-" * 80)
    current_group = None
    for exp_id, cfg in experiments.items():
        if cfg["group"] != current_group:
            current_group = cfg["group"]
            if current_group == "A":
                print(f"  --- Group A: Core Comparison ---")
            elif current_group == "B":
                print(f"  --- Group B: Component Ablation (craft_weight) ---")
            elif current_group == "C":
                print(f"  --- Group C: Reward Scale Sensitivity ---")
            elif current_group == "D":
                print(f"  --- Group D: Replay Strategy Comparison ---")
            elif current_group == "E":
                print(f"  --- Group E: Final Model (NLR + Intrinsic) ---")
        print(f"  {exp_id:<25} [{cfg['group']}]    {cfg['desc']}")
    print("=" * 80 + "\n")


def save_manifest(manifest_path, manifest):
    """Save experiment manifest to JSON."""
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation & Comparative Experiment Runner for Craftax DreamerV3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only these experiments (comma-separated IDs or group letters). "
                             "E.g., --only A or --only A1_baseline,B2_craft_light")
    parser.add_argument("--skip", type=str, default=None,
                        help="Skip these experiments (comma-separated IDs or group letters). "
                             "E.g., --skip D or --skip C1_low_intrinsic")
    parser.add_argument("--seeds", type=str, default="1,4,42",
                        help="Comma-separated seeds (default: 1,4,42).")
    parser.add_argument("--base_logdir", type=str, default=BASE_LOGDIR,
                        help=f"Base directory for all logs (default: {BASE_LOGDIR}).")
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"],
                        help="Wandb mode (default: online).")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU ID (sets CUDA_VISIBLE_DEVICES). E.g., --gpu 0")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Don't delete replay buffer after each run (keeps replay + ckpt).")
    parser.add_argument("--resume_failed", action="store_true",
                        help="Re-run experiments that failed (non-zero exit code).")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Filter experiments
    experiments = filter_experiments(EXPERIMENTS, only=args.only, skip=args.skip)
    if not experiments:
        print("ERROR: No experiments selected. Check --only / --skip flags.")
        sys.exit(1)

    # Print experiment table
    print_experiment_table(experiments, seeds)

    total_runs = len(experiments) * len(seeds)

    if args.dry_run:
        print("[DRY RUN] Commands that would be executed:\n")
        run_idx = 0
        for exp_id, exp_cfg in experiments.items():
            for seed in seeds:
                run_idx += 1
                cmd, logdir = build_command(
                    exp_id, exp_cfg, seed, args.base_logdir, args.wandb_mode)
                skip_note = " [SKIP - already complete]" if is_run_complete(logdir) else ""
                print(f"  [{run_idx:02d}/{total_runs}] {exp_id} seed={seed}{skip_note}")
                print(f"    logdir: {logdir}")
                print(f"    cmd: {' '.join(cmd)}")
                print()
        print(f"[DRY RUN] Total: {total_runs} runs. No commands executed.")
        return

    # --- Actual execution ---
    base_logdir = Path(args.base_logdir)
    base_logdir.mkdir(parents=True, exist_ok=True)

    # Manifest for tracking all runs
    manifest_path = base_logdir / "experiment_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "created": datetime.now().isoformat(),
            "defaults": DEFAULTS,
            "seeds": seeds,
            "runs": {},
        }

    # Environment for subprocess
    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU: CUDA_VISIBLE_DEVICES={args.gpu}")

    run_idx = 0
    completed = 0
    skipped = 0
    failed = 0
    total_cleaned = 0
    run_times = []
    start_time = time.time()

    print(f"\nStarting {total_runs} experiment runs...")
    print(f"Logs will be saved to: {base_logdir.resolve()}\n")

    for exp_id, exp_cfg in experiments.items():
        for seed in seeds:
            run_idx += 1
            run_key = f"{exp_id}_seed{seed}"

            cmd, logdir = build_command(
                exp_id, exp_cfg, seed, args.base_logdir, args.wandb_mode)

            # Skip if already completed
            if is_run_complete(logdir):
                prev = manifest.get("runs", {}).get(run_key, {})
                prev_status = prev.get("status", "unknown")
                if prev_status == "failed" and args.resume_failed:
                    pass  # Re-run failed experiments
                else:
                    skipped += 1
                    print(f"[{run_idx:02d}/{total_runs}] SKIP {run_key} "
                          f"(already complete, status={prev_status})")
                    continue

            # ETA calculation
            if run_times:
                avg_time = sum(run_times) / len(run_times)
                remaining = total_runs - run_idx + 1
                eta = avg_time * remaining
                eta_str = f"  ETA: {format_duration(eta)}"
            else:
                eta_str = ""

            print(f"\n{'=' * 70}")
            print(f"[{run_idx:02d}/{total_runs}] {run_key}{eta_str}")
            print(f"  Desc: {exp_cfg['desc']}")
            print(f"  Logdir: {logdir}")
            print(f"  Cmd: {' '.join(cmd)}")
            print(f"{'=' * 70}")

            # Create logdir
            os.makedirs(logdir, exist_ok=True)

            # Record start
            run_start = time.time()
            manifest["runs"][run_key] = {
                "exp_id": exp_id,
                "group": exp_cfg["group"],
                "desc": exp_cfg["desc"],
                "seed": seed,
                "logdir": logdir,
                "cmd": " ".join(cmd),
                "started": datetime.now().isoformat(),
                "status": "running",
            }
            save_manifest(manifest_path, manifest)

            # Execute
            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    # Stream output to terminal in real-time
                    stdout=None,  # inherit stdout
                    stderr=None,  # inherit stderr
                )
                run_duration = time.time() - run_start
                run_times.append(run_duration)

                if result.returncode == 0:
                    completed += 1
                    status = "completed"
                    print(f"\n  [OK] {run_key} completed in {format_duration(run_duration)}")
                else:
                    failed += 1
                    status = "failed"
                    print(f"\n  [FAIL] {run_key} exited with code {result.returncode} "
                          f"after {format_duration(run_duration)}")

            except KeyboardInterrupt:
                run_duration = time.time() - run_start
                manifest["runs"][run_key]["status"] = "interrupted"
                manifest["runs"][run_key]["duration_s"] = run_duration
                save_manifest(manifest_path, manifest)
                print(f"\n\n  [INTERRUPTED] Saving manifest and exiting...")
                print(f"  Resume with: python run_ablation.py {' '.join(sys.argv[1:])}")
                sys.exit(130)

            except Exception as e:
                run_duration = time.time() - run_start
                failed += 1
                status = "error"
                print(f"\n  [ERROR] {run_key}: {e}")

            # Update manifest
            manifest["runs"][run_key].update({
                "status": status,
                "duration_s": round(run_duration, 1),
                "finished": datetime.now().isoformat(),
                "return_code": result.returncode if "result" in dir() else -1,
            })
            save_manifest(manifest_path, manifest)

            # Cleanup replay/ckpt to save disk
            if not args.no_cleanup and status == "completed":
                cleaned = cleanup_run(logdir)
                total_cleaned += cleaned
                if cleaned > 0:
                    print(f"  Cleaned {format_bytes(cleaned)} (replay buffer removed, ckpt kept)")

    # --- Final summary ---
    total_time = time.time() - start_time
    print(f"\n\n{'=' * 70}")
    print(f"  ABLATION SUITE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total runs:    {total_runs}")
    print(f"  Completed:     {completed}")
    print(f"  Skipped:       {skipped}")
    print(f"  Failed:        {failed}")
    print(f"  Total time:    {format_duration(total_time)}")
    if total_cleaned > 0:
        print(f"  Disk cleaned:  {format_bytes(total_cleaned)}")
    print(f"  Manifest:      {manifest_path.resolve()}")
    print(f"  Logs at:       {base_logdir.resolve()}")
    print(f"{'=' * 70}")

    # List saved artifacts per run
    print(f"\n  Saved artifacts per run:")
    for run_key, run_info in manifest.get("runs", {}).items():
        if run_info.get("status") == "completed":
            logdir = run_info["logdir"]
            artifacts = []
            for root, dirs, files in os.walk(logdir):
                for f in files:
                    if f in ("metrics.jsonl", "online_metrics.jsonl",
                             "metrics_summary.json", "config.yaml", "nlr_args.yaml"):
                        artifacts.append(f)
                if "ckpt" in dirs:
                    artifacts.append("ckpt/")
            if artifacts:
                print(f"    {run_key}: {', '.join(sorted(set(artifacts)))}")

    print(f"\n  Directory layout:")
    print(f"    {base_logdir.resolve()}/")
    print(f"    ├── experiment_manifest.json")
    print(f"    └── <exp_id>_seed<N>/craftax_<exp_id>/")
    print(f"        ├── config.yaml, metrics.jsonl, online_metrics.jsonl")
    print(f"        ├── metrics_summary.json, nlr_args.yaml")
    print(f"        └── ckpt/  (model weights)")

    if failed > 0:
        print(f"\n  WARNING: {failed} runs failed. Re-run with --resume_failed to retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()

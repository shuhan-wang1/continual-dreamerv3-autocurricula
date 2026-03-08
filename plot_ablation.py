#!/usr/bin/env python3
"""
Ablation Study Plotting & Analysis
====================================
Companion to run_ablation.py. Loads experiment results, aggregates across
seeds, and produces comprehensive visualizations + numerical summaries.

Requirements:
  - experiment_manifest.json must exist in the base logdir
  - Only experiments with ALL 3 seeds completed are included

Usage:
  python plot_ablation.py                                    # default logdir
  python plot_ablation.py --base_logdir experiment_results/ablation
  python plot_ablation.py --only A                           # plot group A only
  python plot_ablation.py --outdir ./ablation_plots          # custom output dir
  python plot_ablation.py --format pdf                       # save as PDF
"""

import argparse
import json
import math
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ============================================================================
# Constants
# ============================================================================

REQUIRED_SEEDS = 3  # must have exactly this many completed seeds

BASE_LOGDIR = "experiment_results/ablation"

# Color palette for experiment groups
GROUP_COLORS = {
    "A": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    "B": ["#9467bd", "#8c564b", "#e377c2"],
    "C": ["#7f7f7f", "#bcbd22", "#17becf"],
    "D": ["#aec7e8", "#ffbb78", "#98df8a"],
}

# Descriptive labels for groups
GROUP_LABELS = {
    "A": "Core Comparison",
    "B": "Component Ablation (craft_weight)",
    "C": "Reward Scale Sensitivity (alpha_i)",
    "D": "Replay Strategy (NLR)",
}

# Key online metrics to plot
ONLINE_SCALAR_METRICS = [
    ("score", "Episode Return"),
    ("achievement_depth", "Achievement Depth"),
    ("return_mean", "Windowed Mean Return"),
    ("success_rate", "Success Rate"),
    ("depth_mean", "Mean Achievement Depth"),
    ("aggregate_forgetting", "Aggregate Forgetting"),
    ("frontier_rate", "Frontier Rate"),
    ("personal_best_depth", "Personal Best Depth"),
    ("r_intr", "Combined Intrinsic Reward"),
    ("r_spatial", "Spatial Intrinsic Reward"),
    ("r_craft", "Craft Intrinsic Reward"),
]

# Key training loss metrics
TRAINING_LOSS_METRICS = [
    ("loss/dyn", "Dynamics Loss"),
    ("loss/rep", "Representation Loss"),
    ("loss/rew", "Reward Loss"),
    ("loss/con", "Continuation Loss"),
    ("loss/policy", "Policy Loss"),
    ("loss/value", "Value Loss"),
]

P2E_METRICS = [
    ("p2e/intr_rew", "P2E Intrinsic Reward"),
    ("p2e/extr_rew", "P2E Extrinsic Reward"),
    ("p2e/combined_rew", "P2E Combined Reward"),
    ("p2e/epistemic_std", "Epistemic Uncertainty"),
]


# ============================================================================
# Data Loading
# ============================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file, returning list of dicts."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def find_nested_logdir(run_logdir: str) -> Optional[str]:
    """Find the actual DreamerV3 nested logdir (e.g., craftax_A1_baseline/)."""
    if not os.path.exists(run_logdir):
        return None
    # Check for nested craftax_* directories
    for entry in os.listdir(run_logdir):
        subpath = os.path.join(run_logdir, entry)
        if os.path.isdir(subpath) and entry.startswith("craftax_"):
            return subpath
    # Fallback: check if files exist directly in run_logdir
    if os.path.exists(os.path.join(run_logdir, "metrics.jsonl")):
        return run_logdir
    if os.path.exists(os.path.join(run_logdir, "online_metrics.jsonl")):
        return run_logdir
    return None


def load_manifest(base_logdir: str) -> Dict[str, Any]:
    """Load and validate the experiment manifest."""
    manifest_path = os.path.join(base_logdir, "experiment_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run run_ablation.py first to generate experiment results.")
        sys.exit(1)
    with open(manifest_path, "r") as f:
        return json.load(f)


def get_completed_experiments(manifest: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Group runs by experiment ID.  Only return experiments where ALL required
    seeds completed successfully.

    Returns:
        {exp_id: {"group": str, "desc": str, "seeds": {seed: run_info}}}
    """
    runs = manifest.get("runs", {})
    # Group by exp_id
    by_exp: Dict[str, Dict] = defaultdict(lambda: {"seeds": {}})
    for run_key, run_info in runs.items():
        if run_info.get("status") != "completed":
            continue
        exp_id = run_info["exp_id"]
        seed = run_info["seed"]
        by_exp[exp_id]["group"] = run_info.get("group", "?")
        by_exp[exp_id]["desc"] = run_info.get("desc", "")
        by_exp[exp_id]["seeds"][seed] = run_info

    # Filter: only keep experiments with all required seeds
    complete = OrderedDict()
    skipped = []
    for exp_id in sorted(by_exp.keys()):
        info = by_exp[exp_id]
        n_seeds = len(info["seeds"])
        if n_seeds >= REQUIRED_SEEDS:
            complete[exp_id] = info
        else:
            skipped.append((exp_id, n_seeds))

    if skipped:
        print(f"\nSkipped experiments (incomplete seeds):")
        for eid, ns in skipped:
            print(f"  {eid}: {ns}/{REQUIRED_SEEDS} seeds completed")
    print(f"\nIncluded experiments: {len(complete)}")
    for eid, info in complete.items():
        seeds = sorted(info["seeds"].keys())
        print(f"  {eid} [{info['group']}]: seeds {seeds}")

    return complete


def load_experiment_data(
    experiments: Dict[str, Dict], base_logdir: str
) -> Dict[str, Dict]:
    """
    Load all metrics for each experiment across seeds.

    Returns:
        {exp_id: {
            "group": str, "desc": str,
            "online": {seed: [records]},
            "training": {seed: [records]},
            "summary": {seed: dict},
        }}
    """
    data = {}
    for exp_id, info in experiments.items():
        exp_data = {
            "group": info["group"],
            "desc": info["desc"],
            "online": {},
            "training": {},
            "summary": {},
        }
        for seed, run_info in info["seeds"].items():
            logdir = run_info["logdir"]
            nested = find_nested_logdir(logdir)
            if nested is None:
                print(f"  WARNING: No nested logdir for {exp_id} seed={seed} "
                      f"at {logdir}")
                continue

            # Load online_metrics.jsonl
            online_path = os.path.join(nested, "online_metrics.jsonl")
            exp_data["online"][seed] = load_jsonl(online_path)

            # Load metrics.jsonl (training losses)
            training_path = os.path.join(nested, "metrics.jsonl")
            exp_data["training"][seed] = load_jsonl(training_path)

            # Load metrics_summary.json
            summary_path = os.path.join(nested, "metrics_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    exp_data["summary"][seed] = json.load(f)

        data[exp_id] = exp_data
    return data


# ============================================================================
# Data Processing Utilities
# ============================================================================

def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def extract_scalar_series(
    records: List[Dict], step_key: str, value_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (steps, values) arrays from a list of records."""
    steps, vals = [], []
    for r in records:
        s = r.get(step_key)
        v = r.get(value_key)
        if s is not None and v is not None:
            steps.append(int(s))
            vals.append(safe_float(v))
    return np.array(steps, dtype=np.float64), np.array(vals, dtype=np.float64)


def interpolate_to_common_steps(
    all_steps: List[np.ndarray], all_vals: List[np.ndarray],
    n_points: int = 500
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Interpolate multiple series onto a common step grid.
    Returns (common_steps, list_of_interpolated_values).
    """
    if not all_steps or all(len(s) == 0 for s in all_steps):
        return np.array([]), []

    # Find the common range
    min_step = max(s[0] for s in all_steps if len(s) > 0)
    max_step = min(s[-1] for s in all_steps if len(s) > 0)
    if min_step >= max_step:
        # Fallback: use union range
        min_step = min(s[0] for s in all_steps if len(s) > 0)
        max_step = max(s[-1] for s in all_steps if len(s) > 0)

    common_steps = np.linspace(min_step, max_step, n_points)
    interpolated = []
    for steps, vals in zip(all_steps, all_vals):
        if len(steps) == 0:
            interpolated.append(np.full(n_points, np.nan))
            continue
        interp_vals = np.interp(common_steps, steps, vals)
        interpolated.append(interp_vals)
    return common_steps, interpolated


def aggregate_across_seeds(
    exp_data: Dict, metric_key: str, step_key: str = "step",
    source: str = "online", n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a single experiment, aggregate a scalar metric across all seeds.
    Returns (steps, mean, std).
    """
    all_steps, all_vals = [], []
    records_by_seed = exp_data[source]
    for seed, records in records_by_seed.items():
        if not records:
            continue
        steps, vals = extract_scalar_series(records, step_key, metric_key)
        if len(steps) > 0:
            all_steps.append(steps)
            all_vals.append(vals)

    if not all_steps:
        return np.array([]), np.array([]), np.array([])

    common_steps, interpolated = interpolate_to_common_steps(
        all_steps, all_vals, n_points
    )
    if len(interpolated) == 0:
        return np.array([]), np.array([]), np.array([])

    stacked = np.stack(interpolated, axis=0)  # (n_seeds, n_points)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return common_steps, mean, std


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving-average smoothing."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    # Pad to avoid edge effects
    padded = np.concatenate([
        np.full(window // 2, values[0]),
        values,
        np.full(window // 2, values[-1]),
    ])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(values)]


# ============================================================================
# Plotting Functions
# ============================================================================

def get_exp_color(exp_id: str, group: str, idx_in_group: int) -> str:
    """Get color for an experiment based on its group."""
    colors = GROUP_COLORS.get(group, ["#333333"])
    return colors[idx_in_group % len(colors)]


def plot_group_learning_curves(
    data: Dict, metric_key: str, metric_label: str,
    group: str, outdir: str, fmt: str = "png",
    smooth_window: int = 20, source: str = "online",
    step_key: str = "step",
):
    """Plot learning curves for all experiments in a group, mean±std across seeds."""
    # Filter experiments in this group
    group_exps = {
        eid: d for eid, d in data.items() if d["group"] == group
    }
    if not group_exps:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    idx = 0
    for exp_id, exp_data in group_exps.items():
        steps, mean, std = aggregate_across_seeds(
            exp_data, metric_key, step_key=step_key, source=source
        )
        if len(steps) == 0:
            continue
        color = get_exp_color(exp_id, group, idx)
        label = exp_id
        sm = smooth(mean, smooth_window)
        ax.plot(steps, sm, color=color, linewidth=1.8, label=label)
        ax.fill_between(
            steps, smooth(mean - std, smooth_window),
            smooth(mean + std, smooth_window),
            color=color, alpha=0.15,
        )
        idx += 1

    ax.set_title(f"Group {group}: {GROUP_LABELS.get(group, '')} — {metric_label}",
                 fontsize=13)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"group_{group}_{metric_key.replace('/', '_')}.{fmt}"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def plot_cross_group_comparison(
    data: Dict, metric_key: str, metric_label: str,
    outdir: str, fmt: str = "png",
    smooth_window: int = 20, source: str = "online",
    step_key: str = "step",
):
    """Plot all experiments on a single figure for cross-group comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    global_idx = 0
    group_idx_map = defaultdict(int)

    for exp_id, exp_data in data.items():
        group = exp_data["group"]
        idx = group_idx_map[group]
        group_idx_map[group] += 1

        steps, mean, std = aggregate_across_seeds(
            exp_data, metric_key, step_key=step_key, source=source
        )
        if len(steps) == 0:
            continue
        color = get_exp_color(exp_id, group, idx)
        sm = smooth(mean, smooth_window)
        ax.plot(steps, sm, color=color, linewidth=1.5, label=exp_id)
        ax.fill_between(
            steps, smooth(mean - std, smooth_window),
            smooth(mean + std, smooth_window),
            color=color, alpha=0.1,
        )
        global_idx += 1

    ax.set_title(f"All Experiments — {metric_label}", fontsize=13)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"all_{metric_key.replace('/', '_')}.{fmt}"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def plot_final_bar_chart(
    data: Dict, metric_key: str, metric_label: str,
    outdir: str, fmt: str = "png", source: str = "online",
):
    """Bar chart of final metric values across all experiments (mean±std across seeds)."""
    exp_ids = []
    means = []
    stds = []
    groups = []

    for exp_id, exp_data in data.items():
        # Get final value from each seed
        seed_finals = []
        records_by_seed = exp_data[source]
        for seed, records in records_by_seed.items():
            if not records:
                continue
            # Get last record's value
            last = records[-1]
            val = safe_float(last.get(metric_key))
            if not math.isnan(val):
                seed_finals.append(val)

        if seed_finals:
            exp_ids.append(exp_id)
            means.append(np.mean(seed_finals))
            stds.append(np.std(seed_finals))
            groups.append(exp_data["group"])

    if not exp_ids:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(exp_ids) * 0.9), 5))
    x = np.arange(len(exp_ids))
    colors = []
    group_idx_map = defaultdict(int)
    for eid, g in zip(exp_ids, groups):
        idx = group_idx_map[g]
        group_idx_map[g] += 1
        colors.append(get_exp_color(eid, g, idx))

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f"Final {metric_label} (mean ± std across seeds)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
            f"{m:.3f}", ha="center", va="bottom", fontsize=7,
        )

    fig.tight_layout()
    fname = f"bar_{metric_key.replace('/', '_')}.{fmt}"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def plot_achievement_heatmap(
    data: Dict, outdir: str, fmt: str = "png",
):
    """
    Heatmap of per-achievement rates at the end of training,
    averaged across seeds, for all experiments.
    """
    exp_ids = []
    all_rates = []

    for exp_id, exp_data in data.items():
        seed_rates = []
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            last = records[-1]
            rates = last.get("per_achievement_rates")
            if rates and isinstance(rates, list):
                seed_rates.append(np.array(rates, dtype=np.float64))
        if seed_rates:
            exp_ids.append(exp_id)
            all_rates.append(np.nanmean(np.stack(seed_rates, axis=0), axis=0))

    if not exp_ids or not all_rates:
        return

    matrix = np.stack(all_rates, axis=0)  # (n_exps, n_achievements)
    n_ach = matrix.shape[1]

    # Try to get achievement names from summary
    ach_names = None
    for exp_data in data.values():
        for seed, summary in exp_data["summary"].items():
            if "achievement_names" in summary:
                ach_names = summary["achievement_names"]
                break
        if ach_names:
            break
    if ach_names is None:
        ach_names = [f"ach_{i}" for i in range(n_ach)]

    # Filter to achievements with non-zero rates in at least one experiment
    active_mask = np.any(matrix > 0.01, axis=0)
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        return

    matrix_filtered = matrix[:, active_indices]
    names_filtered = [ach_names[i] for i in active_indices]

    fig_height = max(4, len(exp_ids) * 0.5)
    fig_width = max(10, len(names_filtered) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix_filtered, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=1)
    ax.set_yticks(range(len(exp_ids)))
    ax.set_yticklabels(exp_ids, fontsize=8)
    ax.set_xticks(range(len(names_filtered)))
    ax.set_xticklabels(names_filtered, rotation=90, fontsize=7, ha="center")
    ax.set_title("Per-Achievement Success Rates (final, mean across seeds)",
                 fontsize=12)
    fig.colorbar(im, ax=ax, label="Rate", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"achievement_heatmap.{fmt}"), dpi=150)
    plt.close(fig)


def plot_score_distribution(
    data: Dict, outdir: str, fmt: str = "png",
):
    """Stacked bar chart of final score distribution per experiment."""
    exp_ids = []
    all_dists = []
    tier_labels = ["Tier -1", "Tier 0", "Tier 1", "Tier 2", "Tier 3", "Tier 4"]

    for exp_id, exp_data in data.items():
        seed_dists = []
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            last = records[-1]
            dist = last.get("score_distribution")
            if dist and isinstance(dist, list):
                seed_dists.append(np.array(dist, dtype=np.float64))
        if seed_dists:
            exp_ids.append(exp_id)
            all_dists.append(np.nanmean(np.stack(seed_dists, axis=0), axis=0))

    if not exp_ids:
        return

    matrix = np.stack(all_dists, axis=0)  # (n_exps, n_tiers)
    n_tiers = matrix.shape[1]
    actual_labels = tier_labels[:n_tiers] if n_tiers <= len(tier_labels) else \
        [f"Tier {i}" for i in range(n_tiers)]

    fig, ax = plt.subplots(figsize=(max(10, len(exp_ids) * 0.9), 5))
    x = np.arange(len(exp_ids))
    bottom = np.zeros(len(exp_ids))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_tiers))

    for i in range(n_tiers):
        ax.bar(x, matrix[:, i], bottom=bottom, color=cmap[i],
               label=actual_labels[i], edgecolor="white", linewidth=0.3)
        bottom += matrix[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fraction", fontsize=11)
    ax.set_title("Score Distribution (final, mean across seeds)", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"score_distribution.{fmt}"), dpi=150)
    plt.close(fig)


def plot_training_losses(
    data: Dict, outdir: str, fmt: str = "png",
    smooth_window: int = 50,
):
    """Plot training loss curves for each group."""
    groups = sorted(set(d["group"] for d in data.values()))

    for group in groups:
        group_exps = {eid: d for eid, d in data.items() if d["group"] == group}
        if not group_exps:
            continue

        # Determine which loss metrics have data
        available_losses = []
        for mk, ml in TRAINING_LOSS_METRICS:
            has_data = False
            for exp_data in group_exps.values():
                for seed, records in exp_data["training"].items():
                    if any(mk in r for r in records[:10]):
                        has_data = True
                        break
                if has_data:
                    break
            if has_data:
                available_losses.append((mk, ml))

        if not available_losses:
            continue

        n_metrics = len(available_losses)
        ncols = min(3, n_metrics)
        nrows = math.ceil(n_metrics / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        if n_metrics == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for mi, (mk, ml) in enumerate(available_losses):
            row, col = mi // ncols, mi % ncols
            ax = axes[row, col]
            idx = 0
            group_idx_map = defaultdict(int)
            for exp_id, exp_data in group_exps.items():
                g_idx = group_idx_map[group]
                group_idx_map[group] += 1
                steps, mean, std = aggregate_across_seeds(
                    exp_data, mk, step_key="step", source="training"
                )
                if len(steps) == 0:
                    continue
                color = get_exp_color(exp_id, group, g_idx)
                sm = smooth(mean, smooth_window)
                ax.plot(steps, sm, color=color, linewidth=1.3, label=exp_id)
                ax.fill_between(
                    steps, smooth(mean - std, smooth_window),
                    smooth(mean + std, smooth_window),
                    color=color, alpha=0.1,
                )
                idx += 1

            ax.set_title(ml, fontsize=10)
            ax.set_xlabel("Step", fontsize=9)
            ax.grid(True, alpha=0.3)
            if mi == 0:
                ax.legend(fontsize=7, loc="best")

        # Hide empty subplots
        for mi in range(n_metrics, nrows * ncols):
            row, col = mi // ncols, mi % ncols
            axes[row, col].set_visible(False)

        fig.suptitle(
            f"Group {group}: {GROUP_LABELS.get(group, '')} — Training Losses",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(outdir, f"group_{group}_losses.{fmt}"), dpi=150)
        plt.close(fig)


def plot_p2e_metrics(
    data: Dict, outdir: str, fmt: str = "png",
    smooth_window: int = 50,
):
    """Plot P2E-specific metrics for experiments that use P2E."""
    p2e_exps = {}
    for eid, ed in data.items():
        # Check if any training record has P2E keys
        has_p2e = False
        for seed, records in ed["training"].items():
            if any("p2e/intr_rew" in r for r in records[:10]):
                has_p2e = True
                break
        if has_p2e:
            p2e_exps[eid] = ed

    if not p2e_exps:
        return

    available = []
    for mk, ml in P2E_METRICS:
        has_data = False
        for ed in p2e_exps.values():
            for seed, records in ed["training"].items():
                if any(mk in r for r in records[:10]):
                    has_data = True
                    break
            if has_data:
                break
        if has_data:
            available.append((mk, ml))

    if not available:
        return

    n = len(available)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for mi, (mk, ml) in enumerate(available):
        row, col = mi // ncols, mi % ncols
        ax = axes[row, col]
        idx = 0
        for exp_id, exp_data in p2e_exps.items():
            group = exp_data["group"]
            steps, mean, std = aggregate_across_seeds(
                exp_data, mk, step_key="step", source="training"
            )
            if len(steps) == 0:
                continue
            color = get_exp_color(exp_id, group, idx)
            sm = smooth(mean, smooth_window)
            ax.plot(steps, sm, linewidth=1.3, label=exp_id, color=color)
            ax.fill_between(
                steps, smooth(mean - std, smooth_window),
                smooth(mean + std, smooth_window),
                color=color, alpha=0.1,
            )
            idx += 1
        ax.set_title(ml, fontsize=10)
        ax.set_xlabel("Step", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    for mi in range(n, nrows * ncols):
        row, col = mi // ncols, mi % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("Plan2Explore (P2E) Metrics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"p2e_metrics.{fmt}"), dpi=150)
    plt.close(fig)


def plot_forgetting_per_achievement(
    data: Dict, outdir: str, fmt: str = "png",
):
    """Bar chart showing per-achievement forgetting across experiments."""
    exp_ids = []
    all_forget = []

    for exp_id, exp_data in data.items():
        seed_forgets = []
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            last = records[-1]
            fgt = last.get("per_achievement_forgetting")
            if fgt and isinstance(fgt, list):
                seed_forgets.append(np.array(fgt, dtype=np.float64))
        if seed_forgets:
            exp_ids.append(exp_id)
            all_forget.append(np.nanmean(np.stack(seed_forgets, axis=0), axis=0))

    if not exp_ids or not all_forget:
        return

    matrix = np.stack(all_forget, axis=0)  # (n_exps, n_ach)
    # Mean forgetting per experiment (already have aggregate_forgetting, but this is per-ach detail)
    mean_per_exp = np.nanmean(matrix, axis=1)

    # Also plot the top-10 most forgotten achievements
    mean_per_ach = np.nanmean(matrix, axis=0)
    top_indices = np.argsort(mean_per_ach)[::-1][:10]

    # Get achievement names
    ach_names = None
    for exp_data in data.values():
        for seed, summary in exp_data["summary"].items():
            if "achievement_names" in summary:
                ach_names = summary["achievement_names"]
                break
        if ach_names:
            break

    if ach_names and len(top_indices) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(top_indices))
        width = 0.8 / len(exp_ids)
        for i, (exp_id, row) in enumerate(zip(exp_ids, matrix)):
            offset = (i - len(exp_ids) / 2 + 0.5) * width
            vals = row[top_indices]
            ax.bar(x + offset, vals, width, label=exp_id, alpha=0.8)

        names = [ach_names[j] if j < len(ach_names) else f"ach_{j}"
                 for j in top_indices]
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Forgetting", fontsize=11)
        ax.set_title("Top-10 Most Forgotten Achievements (mean across seeds)",
                     fontsize=13)
        ax.legend(fontsize=7, loc="best", ncol=2)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"forgetting_top10.{fmt}"), dpi=150)
        plt.close(fig)


def plot_summary_dashboard(
    data: Dict, outdir: str, fmt: str = "png",
):
    """
    Single-page dashboard with the 4 most important final metrics as bar charts.
    """
    key_metrics = [
        ("return_mean", "Final Mean Return"),
        ("success_rate", "Final Success Rate"),
        ("aggregate_forgetting", "Final Aggregate Forgetting"),
        ("personal_best_depth", "Personal Best Depth"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for mi, (mk, ml) in enumerate(key_metrics):
        ax = axes[mi]
        exp_ids = []
        means = []
        stds = []
        colors = []
        group_idx_map = defaultdict(int)

        for exp_id, exp_data in data.items():
            seed_vals = []
            for seed, records in exp_data["online"].items():
                if not records:
                    continue
                last = records[-1]
                val = safe_float(last.get(mk))
                if not math.isnan(val):
                    seed_vals.append(val)
            if seed_vals:
                exp_ids.append(exp_id)
                means.append(np.mean(seed_vals))
                stds.append(np.std(seed_vals))
                g = exp_data["group"]
                idx = group_idx_map[g]
                group_idx_map[g] += 1
                colors.append(get_exp_color(exp_id, g, idx))

        if not exp_ids:
            ax.set_visible(False)
            continue

        x = np.arange(len(exp_ids))
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                      edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_ids, rotation=45, ha="right", fontsize=8)
        ax.set_title(ml, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, m in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.3f}", ha="center", va="bottom", fontsize=6,
            )

    fig.suptitle("Ablation Study — Summary Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"summary_dashboard.{fmt}"), dpi=150)
    plt.close(fig)


# ============================================================================
# Numerical Summary
# ============================================================================

def print_numerical_summary(data: Dict, outdir: str):
    """Print and save a comprehensive numerical summary table."""
    lines = []
    sep = "=" * 120

    lines.append(sep)
    lines.append("  ABLATION STUDY — NUMERICAL SUMMARY")
    lines.append(sep)
    lines.append(f"  Experiments included: {len(data)} "
                 f"(each with {REQUIRED_SEEDS} seeds)")
    lines.append("")

    # Table header
    header = (
        f"  {'Experiment':<25} {'Group':>5} "
        f"{'Return':>14} {'SuccRate':>14} {'Forgetting':>14} "
        f"{'MaxDepth':>10} {'FrontRate':>14}"
    )
    lines.append(header)
    lines.append("  " + "-" * 116)

    summary_rows = []
    for exp_id, exp_data in data.items():
        row = {"exp_id": exp_id, "group": exp_data["group"],
               "desc": exp_data["desc"]}
        for mk in ["return_mean", "success_rate", "aggregate_forgetting",
                    "personal_best_depth", "frontier_rate"]:
            seed_vals = []
            for seed, records in exp_data["online"].items():
                if not records:
                    continue
                last = records[-1]
                val = safe_float(last.get(mk))
                if not math.isnan(val):
                    seed_vals.append(val)
            if seed_vals:
                row[f"{mk}_mean"] = np.mean(seed_vals)
                row[f"{mk}_std"] = np.std(seed_vals)
            else:
                row[f"{mk}_mean"] = float("nan")
                row[f"{mk}_std"] = float("nan")
        summary_rows.append(row)

        def fmt_val(key):
            m = row.get(f"{key}_mean", float("nan"))
            s = row.get(f"{key}_std", float("nan"))
            if math.isnan(m):
                return "N/A".rjust(14)
            return f"{m:.4f}±{s:.4f}".rjust(14)

        line = (
            f"  {exp_id:<25} [{exp_data['group']}]  "
            f"{fmt_val('return_mean')} {fmt_val('success_rate')} "
            f"{fmt_val('aggregate_forgetting')} "
            f"{row.get('personal_best_depth_mean', float('nan')):>10.1f} "
            f"{fmt_val('frontier_rate')}"
        )
        lines.append(line)

    lines.append("  " + "-" * 116)

    # Per-group analysis
    lines.append("")
    lines.append(sep)
    lines.append("  PER-GROUP ANALYSIS")
    lines.append(sep)

    groups = sorted(set(r["group"] for r in summary_rows))
    for group in groups:
        group_rows = [r for r in summary_rows if r["group"] == group]
        lines.append(f"\n  --- Group {group}: {GROUP_LABELS.get(group, '')} ---")

        # Find best in group for each metric
        for metric_name, higher_better in [
            ("return_mean", True),
            ("success_rate", True),
            ("aggregate_forgetting", False),
            ("frontier_rate", True),
        ]:
            vals = [(r["exp_id"], r.get(f"{metric_name}_mean", float("nan")))
                    for r in group_rows]
            vals = [(eid, v) for eid, v in vals if not math.isnan(v)]
            if not vals:
                continue
            if higher_better:
                best_id, best_val = max(vals, key=lambda x: x[1])
            else:
                best_id, best_val = min(vals, key=lambda x: x[1])
            label = metric_name.replace("_", " ").title()
            direction = "highest" if higher_better else "lowest"
            lines.append(f"    Best {label} ({direction}): {best_id} = {best_val:.4f}")

    # Achievement summary from metrics_summary.json
    lines.append("")
    lines.append(sep)
    lines.append("  ACHIEVEMENT SUMMARY (from metrics_summary.json)")
    lines.append(sep)
    for exp_id, exp_data in data.items():
        seed_summaries = exp_data["summary"]
        if not seed_summaries:
            continue
        mean_returns = []
        max_depths = []
        for seed, summary in seed_summaries.items():
            mr = summary.get("mean_return")
            md = summary.get("max_achievement_depth")
            if mr is not None:
                mean_returns.append(float(mr))
            if md is not None:
                max_depths.append(float(md))
        if mean_returns or max_depths:
            mr_str = (f"{np.mean(mean_returns):.4f}±{np.std(mean_returns):.4f}"
                      if mean_returns else "N/A")
            md_str = (f"{np.mean(max_depths):.1f}±{np.std(max_depths):.1f}"
                      if max_depths else "N/A")
            lines.append(f"  {exp_id:<25}  mean_return={mr_str}  "
                         f"max_depth={md_str}")

    # Intrinsic reward summary
    lines.append("")
    lines.append(sep)
    lines.append("  INTRINSIC REWARD SUMMARY")
    lines.append(sep)
    for exp_id, exp_data in data.items():
        seed_intr = []
        seed_spatial = []
        seed_craft = []
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            last = records[-1]
            ri = safe_float(last.get("r_intr"))
            rs = safe_float(last.get("r_spatial"))
            rc = safe_float(last.get("r_craft"))
            if not math.isnan(ri):
                seed_intr.append(ri)
            if not math.isnan(rs):
                seed_spatial.append(rs)
            if not math.isnan(rc):
                seed_craft.append(rc)
        if seed_intr:
            lines.append(
                f"  {exp_id:<25}  r_intr={np.mean(seed_intr):.4f}±{np.std(seed_intr):.4f}  "
                f"r_spatial={np.mean(seed_spatial):.4f}±{np.std(seed_spatial):.4f}  "
                f"r_craft={np.mean(seed_craft):.4f}±{np.std(seed_craft):.4f}"
            )

    lines.append("")
    lines.append(sep)

    # Print to stdout
    summary_text = "\n".join(lines)
    print(summary_text)

    # Save to file
    summary_path = os.path.join(outdir, "numerical_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nNumerical summary saved to: {summary_path}")

    # Also save as JSON for programmatic access
    json_summary = {
        "experiments": {},
        "per_group_best": {},
    }
    for row in summary_rows:
        eid = row["exp_id"]
        json_summary["experiments"][eid] = {
            k: v for k, v in row.items() if k not in ("exp_id",)
        }
    for group in groups:
        group_rows = [r for r in summary_rows if r["group"] == group]
        best = {}
        for mk, hb in [("return_mean", True), ("success_rate", True),
                        ("aggregate_forgetting", False), ("frontier_rate", True)]:
            vals = [(r["exp_id"], r.get(f"{mk}_mean", float("nan")))
                    for r in group_rows]
            vals = [(eid, v) for eid, v in vals if not math.isnan(v)]
            if vals:
                if hb:
                    best_id, best_val = max(vals, key=lambda x: x[1])
                else:
                    best_id, best_val = min(vals, key=lambda x: x[1])
                best[mk] = {"exp_id": best_id, "value": best_val}
        json_summary["per_group_best"][group] = best

    json_path = os.path.join(outdir, "numerical_summary.json")
    with open(json_path, "w") as f:
        json.dump(json_summary, f, indent=2, default=str)
    print(f"JSON summary saved to: {json_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot ablation study results with seed aggregation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base_logdir", type=str, default=BASE_LOGDIR,
                        help=f"Base directory with experiment_manifest.json "
                             f"(default: {BASE_LOGDIR})")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for plots (default: <base_logdir>/plots)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output image format (default: png)")
    parser.add_argument("--only", type=str, default=None,
                        help="Plot only these groups (comma-separated, e.g., A,B)")
    parser.add_argument("--smooth", type=int, default=20,
                        help="Smoothing window for learning curves (default: 20)")
    parser.add_argument("--n_points", type=int, default=500,
                        help="Number of interpolation points (default: 500)")
    parser.add_argument("--no_training_losses", action="store_true",
                        help="Skip training loss plots (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("  ABLATION STUDY — PLOT & ANALYSIS")
    print("=" * 70)

    # 1. Load manifest and filter to complete experiments
    manifest = load_manifest(args.base_logdir)
    experiments = get_completed_experiments(manifest)

    if not experiments:
        print("\nERROR: No experiments with all seeds completed. Nothing to plot.")
        sys.exit(1)

    # Filter by group if requested
    if args.only:
        groups = [g.strip() for g in args.only.split(",")]
        experiments = OrderedDict(
            (eid, info) for eid, info in experiments.items()
            if info["group"] in groups
        )
        print(f"\nFiltered to groups {groups}: {len(experiments)} experiments")

    # 2. Load all data
    print("\nLoading experiment data...")
    data = load_experiment_data(experiments, args.base_logdir)

    # Verify we have data
    total_online = sum(
        sum(1 for r in ed["online"].values() if r)
        for ed in data.values()
    )
    total_training = sum(
        sum(1 for r in ed["training"].values() if r)
        for ed in data.values()
    )
    print(f"  Loaded {total_online} online_metrics files, "
          f"{total_training} training metrics files")

    if total_online == 0 and total_training == 0:
        print("\nERROR: No metric data found in any experiment directory.")
        sys.exit(1)

    # 3. Setup output directory
    outdir = args.outdir or os.path.join(args.base_logdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fmt = args.format
    sw = args.smooth

    # 4. Generate all plots
    groups = sorted(set(d["group"] for d in data.values()))
    print(f"\nGenerating plots for groups: {groups}")
    print(f"Output: {os.path.abspath(outdir)}/")

    # --- 4a. Summary dashboard ---
    print("  [1/8] Summary dashboard...")
    plot_summary_dashboard(data, outdir, fmt)

    # --- 4b. Per-group learning curves for each online metric ---
    print("  [2/8] Per-group learning curves...")
    for group in groups:
        for mk, ml in ONLINE_SCALAR_METRICS:
            plot_group_learning_curves(
                data, mk, ml, group, outdir, fmt,
                smooth_window=sw, source="online",
            )

    # --- 4c. Cross-group comparison for key metrics ---
    print("  [3/8] Cross-group comparison curves...")
    key_cross_metrics = [
        ("score", "Episode Return"),
        ("return_mean", "Windowed Mean Return"),
        ("success_rate", "Success Rate"),
        ("aggregate_forgetting", "Aggregate Forgetting"),
        ("frontier_rate", "Frontier Rate"),
        ("personal_best_depth", "Personal Best Depth"),
    ]
    for mk, ml in key_cross_metrics:
        plot_cross_group_comparison(
            data, mk, ml, outdir, fmt, smooth_window=sw,
        )

    # --- 4d. Final-value bar charts ---
    print("  [4/8] Final-value bar charts...")
    bar_metrics = [
        ("return_mean", "Mean Return"),
        ("success_rate", "Success Rate"),
        ("aggregate_forgetting", "Aggregate Forgetting"),
        ("personal_best_depth", "Personal Best Depth"),
        ("frontier_rate", "Frontier Rate"),
        ("depth_mean", "Mean Achievement Depth"),
    ]
    for mk, ml in bar_metrics:
        plot_final_bar_chart(data, mk, ml, outdir, fmt, source="online")

    # --- 4e. Achievement heatmap ---
    print("  [5/8] Achievement heatmap...")
    plot_achievement_heatmap(data, outdir, fmt)

    # --- 4f. Score distribution ---
    print("  [6/8] Score distribution...")
    plot_score_distribution(data, outdir, fmt)

    # --- 4g. Training losses ---
    if not args.no_training_losses:
        print("  [7/8] Training loss curves...")
        plot_training_losses(data, outdir, fmt, smooth_window=50)
        print("  [7b/8] P2E metrics...")
        plot_p2e_metrics(data, outdir, fmt, smooth_window=50)
    else:
        print("  [7/8] Training losses skipped (--no_training_losses)")

    # --- 4h. Forgetting per achievement ---
    print("  [8/8] Per-achievement forgetting...")
    plot_forgetting_per_achievement(data, outdir, fmt)

    # 5. Numerical summary
    print("\n" + "=" * 70)
    print_numerical_summary(data, outdir)

    # 6. Print file listing
    print(f"\nAll outputs saved to: {os.path.abspath(outdir)}/")
    plot_files = sorted(
        f for f in os.listdir(outdir)
        if f.endswith(f".{fmt}") or f.endswith(".txt") or f.endswith(".json")
    )
    print(f"Generated {len(plot_files)} files:")
    for f in plot_files:
        size = os.path.getsize(os.path.join(outdir, f))
        print(f"  {f} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

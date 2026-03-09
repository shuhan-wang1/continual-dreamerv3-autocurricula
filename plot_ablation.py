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

# ---------------------------------------------------------------------------
# Hardcoded Craftax achievement names in enum-value order (sorted by value).
# This is the canonical mapping: one-hot index i -> CRAFTAX_ACHIEVEMENT_NAMES[i].
# Source: craftax.craftax.constants.Achievement enum (67 achievements, values 0-66).
# ---------------------------------------------------------------------------
CRAFTAX_ACHIEVEMENT_NAMES = [
    # value 0-7
    "collect_wood",        # 0
    "place_table",         # 1
    "eat_cow",             # 2
    "collect_sapling",     # 3
    "collect_drink",       # 4
    "make_wood_pickaxe",   # 5
    "make_wood_sword",     # 6
    "place_plant",         # 7
    # value 8-17
    "defeat_zombie",       # 8
    "collect_stone",       # 9
    "place_stone",         # 10
    "eat_plant",           # 11
    "defeat_skeleton",     # 12
    "make_stone_pickaxe",  # 13
    "make_stone_sword",    # 14
    "wake_up",             # 15
    "place_furnace",       # 16
    "collect_coal",        # 17
    # value 18-24
    "collect_iron",        # 18
    "collect_diamond",     # 19
    "make_iron_pickaxe",   # 20
    "make_iron_sword",     # 21
    "make_arrow",          # 22
    "make_torch",          # 23
    "place_torch",         # 24
    # value 25-35
    "make_diamond_sword",  # 25
    "make_iron_armour",    # 26
    "make_diamond_armour", # 27
    "enter_gnomish_mines", # 28
    "enter_dungeon",       # 29
    "enter_sewers",        # 30
    "enter_vault",         # 31
    "enter_troll_mines",   # 32
    "enter_fire_realm",    # 33
    "enter_ice_realm",     # 34
    "enter_graveyard",     # 35
    # value 36-49
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
    # value 50-53
    "eat_bat",             # 50
    "eat_snail",           # 51
    "find_bow",            # 52
    "fire_bow",            # 53
    # value 54-58
    "collect_sapphire",    # 54
    "learn_fireball",      # 55
    "cast_fireball",       # 56
    "learn_iceball",       # 57
    "cast_iceball",        # 58
    # value 59-66
    "collect_ruby",        # 59
    "make_diamond_pickaxe",# 60
    "open_chest",          # 61
    "drink_potion",        # 62
    "enchant_sword",       # 63
    "enchant_armour",      # 64
    "defeat_knight",       # 65
    "defeat_archer",       # 66
]
NUM_CRAFTAX_ACHIEVEMENTS = len(CRAFTAX_ACHIEVEMENT_NAMES)  # 67

# ---------------------------------------------------------------------------
# Key online metrics to plot (focused on what matters for 1M-step ablation)
# No depth metrics: 1M steps is too short for meaningful depth progression.
# ---------------------------------------------------------------------------
ONLINE_SCALAR_METRICS = [
    # Core performance
    ("score", "Episode Return"),
    ("return_mean", "Windowed Mean Return"),
    # Intrinsic rewards
    ("r_intr", "Combined Intrinsic Reward"),
    ("r_spatial", "Spatial Intrinsic Reward"),
    ("r_craft", "Craft Intrinsic Reward"),
    # Continual learning
    ("aggregate_forgetting", "Aggregate Forgetting"),
]

# ---------------------------------------------------------------------------
# Training loss metrics (include obs loss, td-error)
# ---------------------------------------------------------------------------
TRAINING_LOSS_METRICS = [
    ("loss/dyn", "Dynamics Loss"),
    ("loss/rep", "Representation Loss"),
    ("loss/obs", "Observation Loss"),
    ("loss/rew", "Reward Loss"),
    ("loss/con", "Continuation Loss"),
    ("loss/policy", "Policy Loss"),
    ("loss/value", "Value Loss"),
    ("td_error/mean", "TD-Error (Mean)"),
    ("td_error/max", "TD-Error (Max)"),
]

# ---------------------------------------------------------------------------
# P2E / exploration metrics
# ---------------------------------------------------------------------------
P2E_METRICS = [
    ("p2e/intrinsic_reward", "P2E Intrinsic Reward"),
    ("p2e/extrinsic_reward", "P2E Extrinsic Reward"),
    ("p2e/ensemble_disagreement", "P2E Ensemble Disagreement"),
    ("explore/dream_accuracy", "World Model Dream Accuracy"),
    ("explore/intr_extr_ratio", "Intrinsic / Extrinsic Ratio"),
]

# ---------------------------------------------------------------------------
# Replay buffer diagnostics
# ---------------------------------------------------------------------------
REPLAY_METRICS = [
    ("replay/buffer_size", "Replay Buffer Size"),
    ("replay/mean_td_error", "Replay Mean TD-Error"),
    ("replay/mean_episode_age", "Replay Mean Episode Age"),
]


# ============================================================================
# Achievement name helpers
# ============================================================================

def get_achievement_names(data: Dict) -> List[str]:
    """Get achievement names, preferring metrics_summary.json, falling back to hardcoded list."""
    # Try to get from any experiment's summary
    for exp_data in data.values():
        for seed, summary in exp_data.get("summary", {}).items():
            names = summary.get("achievement_names")
            if names and isinstance(names, list) and len(names) > 0:
                return names
    # Fallback to hardcoded canonical list
    return CRAFTAX_ACHIEVEMENT_NAMES


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
    for entry in os.listdir(run_logdir):
        subpath = os.path.join(run_logdir, entry)
        if os.path.isdir(subpath) and entry.startswith("craftax_"):
            return subpath
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
    """
    runs = manifest.get("runs", {})
    by_exp: Dict[str, Dict] = defaultdict(lambda: {"seeds": {}})
    for run_key, run_info in runs.items():
        if run_info.get("status") != "completed":
            continue
        exp_id = run_info["exp_id"]
        seed = run_info["seed"]
        by_exp[exp_id]["group"] = run_info.get("group", "?")
        by_exp[exp_id]["desc"] = run_info.get("desc", "")
        by_exp[exp_id]["seeds"][seed] = run_info

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
    """Load all metrics for each experiment across seeds."""
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

            online_path = os.path.join(nested, "online_metrics.jsonl")
            exp_data["online"][seed] = load_jsonl(online_path)

            training_path = os.path.join(nested, "metrics.jsonl")
            exp_data["training"][seed] = load_jsonl(training_path)

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
    """Interpolate multiple series onto a common step grid."""
    if not all_steps or all(len(s) == 0 for s in all_steps):
        return np.array([]), []

    min_step = max(s[0] for s in all_steps if len(s) > 0)
    max_step = min(s[-1] for s in all_steps if len(s) > 0)
    if min_step >= max_step:
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
    """Aggregate a scalar metric across all seeds. Returns (steps, mean, std)."""
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

    stacked = np.stack(interpolated, axis=0)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return common_steps, mean, std


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving-average smoothing."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    padded = np.concatenate([
        np.full(window // 2, values[0]),
        values,
        np.full(window // 2, values[-1]),
    ])
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(values)]


def count_achievements_unlocked(achievements_vec: list) -> int:
    """Count number of True/1 entries in an achievements vector."""
    if not achievements_vec:
        return 0
    return int(sum(1 for a in achievements_vec if a))


def get_achievement_names_for_vec(achievements_vec: list, ach_names: List[str]) -> List[str]:
    """Return the names of unlocked achievements given a one-hot vector."""
    names = []
    for i, a in enumerate(achievements_vec):
        if a and i < len(ach_names):
            names.append(ach_names[i])
    return names


def extract_max_achievements_series(
    records: List[Dict], step_key: str = "step"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (steps, max_achievements_unlocked_so_far) from online records."""
    steps, vals = [], []
    max_so_far = 0
    for r in records:
        s = r.get(step_key)
        ach = r.get("achievements")
        if s is None:
            continue
        if ach and isinstance(ach, list):
            n = count_achievements_unlocked(ach)
            max_so_far = max(max_so_far, n)
        steps.append(int(s))
        vals.append(max_so_far)
    return np.array(steps, dtype=np.float64), np.array(vals, dtype=np.float64)


def extract_max_score_series(
    records: List[Dict], step_key: str = "step"
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (steps, running_max_score) from online records."""
    steps, vals = [], []
    max_score = float("-inf")
    for r in records:
        s = r.get(step_key)
        score = r.get("score")
        if s is None or score is None:
            continue
        max_score = max(max_score, float(score))
        steps.append(int(s))
        vals.append(max_score)
    return np.array(steps, dtype=np.float64), np.array(vals, dtype=np.float64)


def aggregate_derived_series(
    exp_data: Dict, extract_fn, n_points: int = 500
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a derived series (from extract_fn) across seeds."""
    all_steps, all_vals = [], []
    for seed, records in exp_data["online"].items():
        if not records:
            continue
        steps, vals = extract_fn(records)
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
    stacked = np.stack(interpolated, axis=0)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return common_steps, mean, std


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
    """Plot learning curves for all experiments in a group, mean +/- std across seeds."""
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
        sm = smooth(mean, smooth_window)
        ax.plot(steps, sm, color=color, linewidth=1.8, label=exp_id)
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


def plot_group_derived_curves(
    data: Dict, extract_fn, metric_label: str,
    group: str, outdir: str, fname_suffix: str,
    fmt: str = "png", smooth_window: int = 1,
):
    """Plot derived (non-standard) learning curves for a group."""
    group_exps = {
        eid: d for eid, d in data.items() if d["group"] == group
    }
    if not group_exps:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    idx = 0
    for exp_id, exp_data in group_exps.items():
        steps, mean, std = aggregate_derived_series(exp_data, extract_fn)
        if len(steps) == 0:
            continue
        color = get_exp_color(exp_id, group, idx)
        sm = smooth(mean, smooth_window) if smooth_window > 1 else mean
        ax.plot(steps, sm, color=color, linewidth=1.8, label=exp_id)
        ax.fill_between(
            steps, mean - std, mean + std,
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

    fname = f"group_{group}_{fname_suffix}.{fmt}"
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

    ax.set_title(f"All Experiments — {metric_label}", fontsize=13)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"all_{metric_key.replace('/', '_')}.{fmt}"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def plot_cross_group_derived(
    data: Dict, extract_fn, metric_label: str,
    outdir: str, fname_suffix: str,
    fmt: str = "png", smooth_window: int = 1,
):
    """Cross-group comparison for derived series."""
    fig, ax = plt.subplots(figsize=(12, 6))
    group_idx_map = defaultdict(int)

    for exp_id, exp_data in data.items():
        group = exp_data["group"]
        idx = group_idx_map[group]
        group_idx_map[group] += 1

        steps, mean, std = aggregate_derived_series(exp_data, extract_fn)
        if len(steps) == 0:
            continue
        color = get_exp_color(exp_id, group, idx)
        sm = smooth(mean, smooth_window) if smooth_window > 1 else mean
        ax.plot(steps, sm, color=color, linewidth=1.5, label=exp_id)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.1)

    ax.set_title(f"All Experiments — {metric_label}", fontsize=13)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = f"all_{fname_suffix}.{fmt}"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def plot_final_bar_chart(
    data: Dict, metric_key: str, metric_label: str,
    outdir: str, fmt: str = "png", source: str = "online",
):
    """Bar chart of final metric values across all experiments (mean +/- std across seeds)."""
    exp_ids, means, stds, groups = [], [], [], []

    for exp_id, exp_data in data.items():
        seed_finals = []
        for seed, records in exp_data[source].items():
            if not records:
                continue
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
    ax.set_title(f"Final {metric_label} (mean +/- std across seeds)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
            f"{m:.3f}", ha="center", va="bottom", fontsize=7,
        )
    fig.tight_layout()
    fname = f"bar_{metric_key.replace('/', '_')}.{fmt}"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def plot_final_bar_derived(
    data: Dict, extract_final_fn, metric_label: str,
    outdir: str, fname_suffix: str, fmt: str = "png",
):
    """Bar chart for a derived final metric (computed from raw records)."""
    exp_ids, means, stds, groups = [], [], [], []

    for exp_id, exp_data in data.items():
        seed_vals = []
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            val = extract_final_fn(records)
            if val is not None and not math.isnan(val):
                seed_vals.append(val)
        if seed_vals:
            exp_ids.append(exp_id)
            means.append(np.mean(seed_vals))
            stds.append(np.std(seed_vals))
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
    ax.set_title(f"{metric_label} (mean +/- std across seeds)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
            f"{m:.2f}", ha="center", va="bottom", fontsize=7,
        )
    fig.tight_layout()
    fname = f"bar_{fname_suffix}.{fmt}"
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)


def plot_achievement_heatmap(data: Dict, outdir: str, fmt: str = "png"):
    """
    Heatmap of per-achievement rates at the end of training,
    averaged across seeds, for all experiments.
    Uses canonical achievement names from get_achievement_names().
    """
    ach_names = get_achievement_names(data)

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

    # Use canonical names; pad/truncate to match vector length
    if len(ach_names) < n_ach:
        ach_names = ach_names + [f"ach_{i}" for i in range(len(ach_names), n_ach)]
    ach_names = ach_names[:n_ach]

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


def plot_score_distribution(data: Dict, outdir: str, fmt: str = "png"):
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

    matrix = np.stack(all_dists, axis=0)
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


def plot_training_losses(data: Dict, outdir: str, fmt: str = "png",
                         smooth_window: int = 50):
    """Plot training loss curves for each group."""
    groups = sorted(set(d["group"] for d in data.values()))

    for group in groups:
        group_exps = {eid: d for eid, d in data.items() if d["group"] == group}
        if not group_exps:
            continue

        available_losses = []
        for mk, ml in TRAINING_LOSS_METRICS:
            has_data = False
            for exp_data in group_exps.values():
                for seed, records in exp_data["online"].items():
                    if any(mk in r for r in records[:20]):
                        has_data = True
                        break
                if has_data:
                    break
                for seed, records in exp_data["training"].items():
                    if any(mk in r for r in records[:20]):
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
            group_idx_map = defaultdict(int)
            for exp_id, exp_data in group_exps.items():
                g_idx = group_idx_map[group]
                group_idx_map[group] += 1
                steps, mean, std = aggregate_across_seeds(
                    exp_data, mk, step_key="step", source="online"
                )
                if len(steps) == 0:
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
            ax.set_title(ml, fontsize=10)
            ax.set_xlabel("Step", fontsize=9)
            ax.grid(True, alpha=0.3)
            if mi == 0:
                ax.legend(fontsize=7, loc="best")

        for mi in range(n_metrics, nrows * ncols):
            row, col = mi // ncols, mi % ncols
            axes[row, col].set_visible(False)

        fig.suptitle(
            f"Group {group}: {GROUP_LABELS.get(group, '')} — Training Losses & TD-Error",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(outdir, f"group_{group}_losses.{fmt}"), dpi=150)
        plt.close(fig)


def plot_p2e_metrics(data: Dict, outdir: str, fmt: str = "png",
                     smooth_window: int = 50):
    """Plot P2E-specific metrics for experiments that use P2E."""
    p2e_exps = {}
    for eid, ed in data.items():
        has_p2e = False
        for seed, records in ed["online"].items():
            if any("p2e/intrinsic_reward" in r or "p2e/ensemble_disagreement" in r
                   for r in records[:20]):
                has_p2e = True
                break
        if not has_p2e:
            for seed, records in ed["training"].items():
                if any("p2e/intr_rew" in r or "p2e/intrinsic_reward" in r
                       for r in records[:20]):
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
            for seed, records in ed["online"].items():
                if any(mk in r for r in records[:20]):
                    has_data = True
                    break
            if has_data:
                break
            for seed, records in ed["training"].items():
                if any(mk in r for r in records[:20]):
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
                exp_data, mk, step_key="step", source="online"
            )
            if len(steps) == 0:
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

    fig.suptitle("Plan2Explore (P2E) & Exploration Metrics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"p2e_metrics.{fmt}"), dpi=150)
    plt.close(fig)


def plot_replay_metrics(data: Dict, outdir: str, fmt: str = "png",
                        smooth_window: int = 50):
    """Plot replay buffer diagnostic metrics."""
    available = []
    for mk, ml in REPLAY_METRICS:
        has_data = False
        for ed in data.values():
            for seed, records in ed["online"].items():
                if any(mk in r for r in records[:20]):
                    has_data = True
                    break
            if has_data:
                break
        if has_data:
            available.append((mk, ml))

    if not available:
        return

    n = len(available)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for mi, (mk, ml) in enumerate(available):
        row, col = mi // ncols, mi % ncols
        ax = axes[row, col]
        group_idx_map = defaultdict(int)
        for exp_id, exp_data in data.items():
            group = exp_data["group"]
            idx = group_idx_map[group]
            group_idx_map[group] += 1
            steps, mean, std = aggregate_across_seeds(
                exp_data, mk, step_key="step", source="online"
            )
            if len(steps) == 0:
                continue
            color = get_exp_color(exp_id, group, idx)
            sm = smooth(mean, smooth_window)
            ax.plot(steps, sm, color=color, linewidth=1.3, label=exp_id)
            ax.fill_between(
                steps, smooth(mean - std, smooth_window),
                smooth(mean + std, smooth_window),
                color=color, alpha=0.1,
            )
        ax.set_title(ml, fontsize=10)
        ax.set_xlabel("Step", fontsize=9)
        ax.grid(True, alpha=0.3)
        if mi == 0:
            ax.legend(fontsize=7, loc="best")

    for mi in range(n, nrows * ncols):
        row, col = mi // ncols, mi % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("Replay Buffer Diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"replay_diagnostics.{fmt}"), dpi=150)
    plt.close(fig)


def plot_summary_dashboard(data: Dict, outdir: str, fmt: str = "png"):
    """
    Single-page dashboard with the 6 most important final metrics.
    Focused on 1M-step ablation: score, achievements, intrinsic rewards.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # --- Panel 0: Final Mean Return (bar) ---
    _dashboard_bar(axes[0], data, "return_mean", "Final Mean Return")

    # --- Panel 1: Max Score (bar, derived) ---
    def _final_max_score(records):
        max_s = float("-inf")
        for r in records:
            s = r.get("score")
            if s is not None:
                max_s = max(max_s, float(s))
        return max_s if max_s > float("-inf") else None
    _dashboard_bar_derived(axes[1], data, _final_max_score, "Max Episode Score")

    # --- Panel 2: Max Achievements Unlocked (bar, derived) ---
    def _final_max_ach(records):
        mx = 0
        for r in records:
            ach = r.get("achievements")
            if ach and isinstance(ach, list):
                mx = max(mx, count_achievements_unlocked(ach))
        return float(mx)
    _dashboard_bar_derived(axes[2], data, _final_max_ach,
                           "Max Achievements Unlocked")

    # --- Panel 3: Combined Intrinsic Reward (bar) ---
    _dashboard_bar(axes[3], data, "r_intr", "Final Intrinsic Reward (combined)")

    # --- Panel 4: Aggregate Forgetting (bar) ---
    _dashboard_bar(axes[4], data, "aggregate_forgetting", "Aggregate Forgetting")

    # --- Panel 5: Spatial Intrinsic Reward (bar) ---
    _dashboard_bar(axes[5], data, "r_spatial", "Final Spatial Intrinsic Reward")

    fig.suptitle("Ablation Study — Summary Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"summary_dashboard.{fmt}"), dpi=150)
    plt.close(fig)


def _dashboard_bar(ax, data, metric_key, title):
    """Helper: draw a bar subplot for a standard metric."""
    exp_ids, means, stds, colors = [], [], [], []
    group_idx_map = defaultdict(int)
    for exp_id, exp_data in data.items():
        seed_vals = []
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            last = records[-1]
            val = safe_float(last.get(metric_key))
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
        return

    x = np.arange(len(exp_ids))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.3f}", ha="center", va="bottom", fontsize=6)


def _dashboard_bar_derived(ax, data, extract_fn, title):
    """Helper: draw a bar subplot for a derived metric."""
    exp_ids, means, stds, colors = [], [], [], []
    group_idx_map = defaultdict(int)
    for exp_id, exp_data in data.items():
        seed_vals = []
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            val = extract_fn(records)
            if val is not None and not math.isnan(val):
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
        return

    x = np.arange(len(exp_ids))
    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.2f}", ha="center", va="bottom", fontsize=6)


# ============================================================================
# Numerical Summary
# ============================================================================

def print_numerical_summary(data: Dict, outdir: str):
    """Print and save a comprehensive numerical summary table."""
    ach_names = get_achievement_names(data)

    lines = []
    sep = "=" * 130

    lines.append(sep)
    lines.append("  ABLATION STUDY — NUMERICAL SUMMARY")
    lines.append(sep)
    lines.append(f"  Experiments included: {len(data)} "
                 f"(each with {REQUIRED_SEEDS} seeds)")
    lines.append("")

    header = (
        f"  {'Experiment':<25} {'Grp':>3} "
        f"{'Return':>14} {'MaxScore':>14} {'MaxAch':>8} "
        f"{'Forgetting':>14} {'r_intr':>14}"
    )
    lines.append(header)
    lines.append("  " + "-" * 126)

    summary_rows = []
    for exp_id, exp_data in data.items():
        row = {"exp_id": exp_id, "group": exp_data["group"],
               "desc": exp_data["desc"]}

        # Standard final-record metrics
        for mk in ["return_mean", "aggregate_forgetting", "r_intr"]:
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

        # Derived: max score across entire run
        seed_max_scores = []
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            ms = max((safe_float(r.get("score")) for r in records), default=float("nan"))
            if not math.isnan(ms):
                seed_max_scores.append(ms)
        if seed_max_scores:
            row["max_score_mean"] = np.mean(seed_max_scores)
            row["max_score_std"] = np.std(seed_max_scores)
        else:
            row["max_score_mean"] = float("nan")
            row["max_score_std"] = float("nan")

        # Derived: max achievements unlocked (with names)
        seed_max_ach = []
        seed_best_ach_vec = None
        for seed, records in exp_data["online"].items():
            if not records:
                continue
            mx = 0
            best_vec = None
            for r in records:
                ach = r.get("achievements")
                if ach and isinstance(ach, list):
                    n = count_achievements_unlocked(ach)
                    if n > mx:
                        mx = n
                        best_vec = ach
            seed_max_ach.append(float(mx))
            if best_vec is not None:
                if seed_best_ach_vec is None:
                    seed_best_ach_vec = best_vec
                elif count_achievements_unlocked(best_vec) > count_achievements_unlocked(seed_best_ach_vec):
                    seed_best_ach_vec = best_vec
        if seed_max_ach:
            row["max_ach_mean"] = np.mean(seed_max_ach)
            row["max_ach_std"] = np.std(seed_max_ach)
            if seed_best_ach_vec:
                row["best_ach_names"] = get_achievement_names_for_vec(
                    seed_best_ach_vec, ach_names)
        else:
            row["max_ach_mean"] = float("nan")
            row["max_ach_std"] = float("nan")

        summary_rows.append(row)

        def fmt_val(key):
            m = row.get(f"{key}_mean", float("nan"))
            s = row.get(f"{key}_std", float("nan"))
            if math.isnan(m):
                return "N/A".rjust(14)
            return f"{m:.4f}+/-{s:.4f}".rjust(14)

        line = (
            f"  {exp_id:<25} [{exp_data['group']}] "
            f"{fmt_val('return_mean')} {fmt_val('max_score')} "
            f"{row.get('max_ach_mean', float('nan')):>6.1f}+/-{row.get('max_ach_std', float('nan')):.1f} "
            f"{fmt_val('aggregate_forgetting')} "
            f"{fmt_val('r_intr')}"
        )
        lines.append(line)

    lines.append("  " + "-" * 126)

    # Per-group analysis
    lines.append("")
    lines.append(sep)
    lines.append("  PER-GROUP ANALYSIS")
    lines.append(sep)

    groups = sorted(set(r["group"] for r in summary_rows))
    for group in groups:
        group_rows = [r for r in summary_rows if r["group"] == group]
        lines.append(f"\n  --- Group {group}: {GROUP_LABELS.get(group, '')} ---")

        for metric_name, higher_better, label in [
            ("return_mean", True, "Mean Return"),
            ("max_score", True, "Max Score"),
            ("max_ach", True, "Max Achievements Unlocked"),
            ("aggregate_forgetting", False, "Aggregate Forgetting"),
            ("r_intr", True, "Intrinsic Reward"),
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
            direction = "highest" if higher_better else "lowest"
            lines.append(f"    Best {label} ({direction}): {best_id} = {best_val:.4f}")

    # Achievement detail: which achievements each experiment unlocked
    lines.append("")
    lines.append(sep)
    lines.append("  ACHIEVEMENT DETAILS (best single-seed achievement set)")
    lines.append(sep)
    for row in summary_rows:
        names = row.get("best_ach_names", [])
        if names:
            lines.append(f"  {row['exp_id']:<25}  [{len(names)} unlocked]: {', '.join(names)}")
        else:
            lines.append(f"  {row['exp_id']:<25}  [0 unlocked]")

    # Intrinsic reward summary
    lines.append("")
    lines.append(sep)
    lines.append("  INTRINSIC REWARD SUMMARY")
    lines.append(sep)
    for exp_id, exp_data in data.items():
        seed_intr, seed_spatial, seed_craft = [], [], []
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
                f"  {exp_id:<25}  r_intr={np.mean(seed_intr):.4f}+/-{np.std(seed_intr):.4f}  "
                f"r_spatial={np.mean(seed_spatial):.4f}+/-{np.std(seed_spatial):.4f}  "
                f"r_craft={np.mean(seed_craft):.4f}+/-{np.std(seed_craft):.4f}"
            )

    lines.append("")
    lines.append(sep)

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = os.path.join(outdir, "numerical_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nNumerical summary saved to: {summary_path}")

    # JSON summary
    json_summary = {
        "experiments": {},
        "per_group_best": {},
    }
    for row in summary_rows:
        eid = row["exp_id"]
        # Convert best_ach_names list for JSON (skip non-serializable)
        row_json = {}
        for k, v in row.items():
            if k == "exp_id":
                continue
            if isinstance(v, (list, str, int, float, bool)):
                row_json[k] = v
            else:
                row_json[k] = str(v)
        json_summary["experiments"][eid] = row_json

    for group in groups:
        group_rows = [r for r in summary_rows if r["group"] == group]
        best = {}
        for mk, hb in [("return_mean", True), ("max_score", True),
                        ("max_ach", True), ("aggregate_forgetting", False)]:
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

    manifest = load_manifest(args.base_logdir)
    experiments = get_completed_experiments(manifest)

    if not experiments:
        print("\nERROR: No experiments with all seeds completed. Nothing to plot.")
        sys.exit(1)

    if args.only:
        groups = [g.strip() for g in args.only.split(",")]
        experiments = OrderedDict(
            (eid, info) for eid, info in experiments.items()
            if info["group"] in groups
        )
        print(f"\nFiltered to groups {groups}: {len(experiments)} experiments")

    print("\nLoading experiment data...")
    data = load_experiment_data(experiments, args.base_logdir)

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

    outdir = args.outdir or os.path.join(args.base_logdir, "plots")
    os.makedirs(outdir, exist_ok=True)
    fmt = args.format
    sw = args.smooth

    groups = sorted(set(d["group"] for d in data.values()))
    print(f"\nGenerating plots for groups: {groups}")
    print(f"Output: {os.path.abspath(outdir)}/")

    # --- 1. Summary dashboard ---
    print("  [1/9] Summary dashboard...")
    plot_summary_dashboard(data, outdir, fmt)

    # --- 2. Per-group learning curves for standard online metrics ---
    print("  [2/9] Per-group learning curves (online metrics)...")
    for group in groups:
        for mk, ml in ONLINE_SCALAR_METRICS:
            plot_group_learning_curves(
                data, mk, ml, group, outdir, fmt,
                smooth_window=sw, source="online",
            )

    # --- 3. Per-group DERIVED curves (max score, max achievements) ---
    print("  [3/9] Per-group derived curves (max score, max achievements)...")
    for group in groups:
        plot_group_derived_curves(
            data, extract_max_score_series,
            "Running Max Score", group, outdir, "max_score",
            fmt=fmt, smooth_window=1,
        )
        plot_group_derived_curves(
            data, extract_max_achievements_series,
            "Max Achievements Unlocked", group, outdir, "max_achievements",
            fmt=fmt, smooth_window=1,
        )

    # --- 4. Cross-group comparison for key metrics ---
    print("  [4/9] Cross-group comparison curves...")
    key_cross_metrics = [
        ("score", "Episode Return"),
        ("return_mean", "Windowed Mean Return"),
        ("aggregate_forgetting", "Aggregate Forgetting"),
        ("r_intr", "Combined Intrinsic Reward"),
    ]
    for mk, ml in key_cross_metrics:
        plot_cross_group_comparison(
            data, mk, ml, outdir, fmt, smooth_window=sw,
        )
    plot_cross_group_derived(
        data, extract_max_score_series, "Running Max Score",
        outdir, "max_score", fmt=fmt,
    )
    plot_cross_group_derived(
        data, extract_max_achievements_series, "Max Achievements Unlocked",
        outdir, "max_achievements", fmt=fmt,
    )

    # --- 5. Final-value bar charts ---
    print("  [5/9] Final-value bar charts...")
    bar_metrics = [
        ("return_mean", "Mean Return"),
        ("aggregate_forgetting", "Aggregate Forgetting"),
        ("r_intr", "Combined Intrinsic Reward"),
        ("r_spatial", "Spatial Intrinsic Reward"),
        ("r_craft", "Craft Intrinsic Reward"),
    ]
    for mk, ml in bar_metrics:
        plot_final_bar_chart(data, mk, ml, outdir, fmt, source="online")

    # Derived bar charts
    def _final_max_score(records):
        return max((safe_float(r.get("score")) for r in records), default=float("nan"))

    def _final_max_ach(records):
        mx = 0
        for r in records:
            ach = r.get("achievements")
            if ach and isinstance(ach, list):
                mx = max(mx, count_achievements_unlocked(ach))
        return float(mx)

    plot_final_bar_derived(data, _final_max_score, "Max Episode Score",
                           outdir, "max_score", fmt=fmt)
    plot_final_bar_derived(data, _final_max_ach, "Max Achievements Unlocked",
                           outdir, "max_achievements", fmt=fmt)

    # --- 6. Achievement heatmap ---
    print("  [6/9] Achievement heatmap...")
    plot_achievement_heatmap(data, outdir, fmt)

    # --- 7. Score distribution ---
    print("  [7/9] Score distribution...")
    plot_score_distribution(data, outdir, fmt)

    # --- 8. Training losses + P2E + replay ---
    if not args.no_training_losses:
        print("  [8/9] Training loss curves (incl. obs, td-error)...")
        plot_training_losses(data, outdir, fmt, smooth_window=50)
        print("  [8b/9] P2E & exploration metrics...")
        plot_p2e_metrics(data, outdir, fmt, smooth_window=50)
        print("  [8c/9] Replay buffer diagnostics...")
        plot_replay_metrics(data, outdir, fmt, smooth_window=50)
    else:
        print("  [8/9] Training losses skipped (--no_training_losses)")

    # --- 9. Numerical summary ---
    print("  [9/9] Numerical summary...")
    print("\n" + "=" * 70)
    print_numerical_summary(data, outdir)

    # Print file listing
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

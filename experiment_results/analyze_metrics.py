#!/usr/bin/env python3
"""
Comparative Analysis: Continual Learning vs Original DreamerV3 on Craftax
=========================================================================
Loads metrics from 6 experiment folders (2 methods × 3 seeds), computes
mean ± std across seeds, and produces publication-quality figures with
statistical annotations (paired t-tests / Welch's t at final evaluation).
"""

import json
import os
import pathlib
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────
# Stats Report Writer
# ──────────────────────────────────────────────────────────────────────

class StatsReport:
    """Accumulates statistics text for all figures and writes to a file."""

    def __init__(self):
        self.sections = []

    def add_section(self, title, body):
        self.sections.append((title, body))

    def write(self, path):
        sep = "=" * 80
        with open(path, "w") as f:
            f.write(sep + "\n")
            f.write("STATISTICAL ANALYSIS REPORT\n")
            f.write("Craftax: DreamerV3-CL vs DreamerV3-Original\n")
            f.write(f"Seeds: {SEEDS}  |  Significance level: {SIGNIFICANCE_LEVEL}\n")
            f.write(sep + "\n\n")
            for title, body in self.sections:
                f.write("-" * 80 + "\n")
                f.write(f"  {title}\n")
                f.write("-" * 80 + "\n")
                f.write(body + "\n\n")
            f.write(sep + "\n")
            f.write("END OF REPORT\n")
            f.write(sep + "\n")


# Global report instance
report = StatsReport()


def welch_t(a, b):
    """Welch's t-test. Returns (t_stat, p_value, significance_string)."""
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, "N/A (insufficient samples)"
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    return t, p, sig


def fmt_dist(arr):
    """Format an array as 'mean ± std [min, max]'."""
    if len(arr) == 0:
        return "N/A"
    return f"{arr.mean():.4f} ± {arr.std():.4f}  [min={arr.min():.4f}, max={arr.max():.4f}]"


def fmt_dist_short(arr):
    """Format an array as 'mean ± std'."""
    if len(arr) == 0:
        return "N/A"
    return f"{arr.mean():.4f} ± {arr.std():.4f}"


def curve_endpoint_stats(curves, metric_name):
    """
    Given curves dict from build_curves, compute stats at 25%, 50%, 75%, 100%
    of training and a final-point Welch's t-test. Returns formatted string.
    """
    lines = []
    mats = {}
    ref_grid = None
    for mk in METHODS:
        grid, mat = curves[mk]
        if grid is not None and mat is not None:
            mats[mk] = mat
            ref_grid = grid

    if not mats or ref_grid is None:
        return f"  {metric_name}: No data available.\n"

    checkpoints = {"25%": 0.25, "50%": 0.50, "75%": 0.75, "100%": 1.0}
    n = ref_grid.shape[0]

    lines.append(f"  {metric_name}:")
    header = f"    {'Checkpoint':<12} {'DreamerV3-CL':<28} {'DreamerV3-Original':<28} {'Welch t-test':<20}"
    lines.append(header)
    lines.append("    " + "-" * 88)

    for label, frac in checkpoints.items():
        idx = min(int(frac * (n - 1)), n - 1)
        cl_vals = mats["cl"][:, idx] if "cl" in mats else np.array([])
        orig_vals = mats["original"][:, idx] if "original" in mats else np.array([])
        _, p, sig = welch_t(cl_vals, orig_vals)
        p_str = f"p={p:.4f} ({sig})" if not np.isnan(p) else "N/A"
        lines.append(f"    {label:<12} {fmt_dist_short(cl_vals):<28} {fmt_dist_short(orig_vals):<28} {p_str:<20}")

    # Effect size (Cohen's d) at final point
    if "cl" in mats and "original" in mats:
        cl_f = mats["cl"][:, -1]
        orig_f = mats["original"][:, -1]
        pooled_std = np.sqrt((cl_f.std()**2 + orig_f.std()**2) / 2)
        cohens_d = (cl_f.mean() - orig_f.mean()) / pooled_std if pooled_std > 1e-12 else 0.0
        lines.append(f"    Cohen's d (final): {cohens_d:+.4f}  (+ favours CL, - favours Original)")

    return "\n".join(lines) + "\n"


def compute_auc(grid, matrix):
    """Compute Area Under Curve (trapezoidal) for each seed, return array."""
    aucs = []
    for i in range(matrix.shape[0]):
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        aucs.append(_trapz(matrix[i], grid))
    return np.array(aucs)


def auc_comparison(curves, metric_name):
    """Compute and compare AUC between methods."""
    lines = []
    mats, grids = {}, {}
    for mk in METHODS:
        grid, mat = curves[mk]
        if grid is not None and mat is not None:
            mats[mk] = mat
            grids[mk] = grid

    if len(mats) < 2:
        return f"  {metric_name} AUC: Insufficient data for comparison.\n"

    lines.append(f"  {metric_name} — Area Under Curve (AUC):")
    aucs = {}
    for mk in METHODS:
        auc_arr = compute_auc(grids[mk], mats[mk])
        aucs[mk] = auc_arr
        lines.append(f"    {METHODS[mk]}: {fmt_dist(auc_arr)}")

    if "cl" in aucs and "original" in aucs:
        _, p, sig = welch_t(aucs["cl"], aucs["original"])
        lines.append(f"    AUC Welch t-test: p={p:.4f} ({sig})")

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
LOGS_DIR = pathlib.Path("logs")
METHODS = {"cl": "DreamerV3-CL", "original": "DreamerV3-Original"}
SEEDS = [1, 4, 42]
COLORS = {"cl": "#1f77b4", "original": "#ff7f0e"}
SMOOTH_WINDOW = 50  # rolling window for smoothing curves
INTERP_STEPS = 1000  # number of interpolation points on x-axis
FIGSIZE_WIDE = (14, 5)
FIGSIZE_TALL = (14, 8)
ALPHA_FILL = 0.18
SIGNIFICANCE_LEVEL = 0.05
OUTPUT_DIR = pathlib.Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Correct ordering: matches Craftax Achievement enum values 0-21
ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling",       # 0-3
    "collect_drink", "make_wood_pickaxe", "make_wood_sword",           # 4-6
    "place_plant", "defeat_zombie", "collect_stone", "place_stone",    # 7-10
    "eat_plant", "defeat_skeleton", "make_stone_pickaxe",              # 11-13
    "make_stone_sword", "wake_up", "place_furnace", "collect_coal",    # 14-17
    "collect_iron", "collect_diamond", "make_iron_pickaxe",            # 18-20
    "make_iron_sword",                                                 # 21
]

# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────

def load_jsonl(path):
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_all_runs():
    """
    Returns nested dict:  data[method][seed] = {
        'metrics': [...],          # from metrics.jsonl
        'online': [...],           # from online_metrics.jsonl
        'summary': {...},          # from metrics_summary.json
        'config': <raw string>,    # from config.yaml
    }
    """
    data = defaultdict(dict)
    for method_key in METHODS:
        for seed in SEEDS:
            folder = LOGS_DIR / f"craftax_dreamerv3-{method_key}-{seed}"
            if not folder.exists():
                print(f"  [WARNING] Missing folder: {folder}")
                continue
            run = {}
            metrics_path = folder / "metrics.jsonl"
            online_path = folder / "online_metrics.jsonl"
            summary_path = folder / "metrics_summary.json"
            config_path = folder / "config.yaml"

            if metrics_path.exists():
                run["metrics"] = load_jsonl(metrics_path)
            else:
                run["metrics"] = []
            if online_path.exists():
                run["online"] = load_jsonl(online_path)
            else:
                run["online"] = []
            if summary_path.exists():
                with open(summary_path) as f:
                    run["summary"] = json.load(f)
            else:
                run["summary"] = {}
            if config_path.exists():
                with open(config_path) as f:
                    run["config"] = f.read()
            else:
                run["config"] = ""

            data[method_key][seed] = run
            n_ep = len(run["metrics"])
            n_on = len(run["online"])
            print(f"  Loaded {method_key} seed={seed}: "
                  f"{n_ep} episode entries, {n_on} online entries")
    return data


def extract_scalar_curve(records, step_key, value_key, smooth=SMOOTH_WINDOW):
    """Extract (steps, values) arrays from a list of dicts, with optional smoothing."""
    steps, vals = [], []
    for r in records:
        if step_key in r and value_key in r:
            v = r[value_key]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                steps.append(r[step_key])
                vals.append(v)
    steps = np.array(steps, dtype=np.float64)
    vals = np.array(vals, dtype=np.float64)
    if len(vals) > 0 and smooth > 1:
        vals = uniform_filter1d(vals, size=min(smooth, len(vals)))
    return steps, vals


def interpolate_to_common_grid(all_steps, all_vals, n_points=INTERP_STEPS):
    """Interpolate multiple (steps, vals) curves onto a shared step grid."""
    if not all_steps:
        return None, None
    lo = max(s[0] for s in all_steps)
    hi = min(s[-1] for s in all_steps)
    if lo >= hi:
        hi = max(s[-1] for s in all_steps)
        lo = min(s[0] for s in all_steps)
    grid = np.linspace(lo, hi, n_points)
    interped = []
    for s, v in zip(all_steps, all_vals):
        interped.append(np.interp(grid, s, v))
    return grid, np.array(interped)


# ──────────────────────────────────────────────────────────────────────
# Plotting Helpers
# ──────────────────────────────────────────────────────────────────────

def plot_mean_std(ax, grid, matrix, color, label):
    """Plot mean line with shaded std band."""
    mu = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    ax.plot(grid, mu, color=color, label=label, linewidth=2)
    ax.fill_between(grid, mu - std, mu + std, color=color, alpha=ALPHA_FILL)


def add_significance_annotation(ax, grid, mat_a, mat_b, y_pos=None):
    """Add a significance star at the final step if p < 0.05 (Welch's t-test)."""
    final_a = mat_a[:, -1]
    final_b = mat_b[:, -1]
    if len(final_a) < 2 or len(final_b) < 2:
        return
    t_stat, p_val = stats.ttest_ind(final_a, final_b, equal_var=False)
    if p_val < SIGNIFICANCE_LEVEL:
        if y_pos is None:
            y_pos = max(final_a.mean(), final_b.mean())
        star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
        ax.annotate(
            f"{star}\np={p_val:.4f}",
            xy=(grid[-1], y_pos),
            fontsize=9,
            ha="right",
            va="bottom",
            color="black",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )
    return p_val


def format_ax(ax, title, xlabel="Environment Steps", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_curves(data, source, value_key, step_key="step", smooth=SMOOTH_WINDOW):
    """Build interpolated matrices for both methods."""
    results = {}
    for method_key in METHODS:
        all_s, all_v = [], []
        for seed in SEEDS:
            if seed not in data[method_key]:
                continue
            records = data[method_key][seed][source]
            s, v = extract_scalar_curve(records, step_key, value_key, smooth=smooth)
            if len(s) > 1:
                all_s.append(s)
                all_v.append(v)
        grid, mat = interpolate_to_common_grid(all_s, all_v)
        results[method_key] = (grid, mat)
    return results


# ──────────────────────────────────────────────────────────────────────
# Main Analysis & Figures
# ──────────────────────────────────────────────────────────────────────

def figure_episode_return_and_length(data):
    """Fig 1: Episode return and episode length over training steps."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    stats_lines = []

    for idx, (value_key, ylabel, title) in enumerate([
        ("episode/score", "Episode Return", "Episode Return"),
        ("episode/length", "Episode Length", "Episode Length"),
    ]):
        ax = axes[idx]
        curves = build_curves(data, "metrics", value_key)
        mats = {}
        ref_grid = None
        for mk in METHODS:
            grid, mat = curves[mk]
            if grid is not None and mat is not None:
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk])
                mats[mk] = mat
                ref_grid = grid
        if "cl" in mats and "original" in mats and ref_grid is not None:
            add_significance_annotation(ax, ref_grid, mats["cl"], mats["original"])
        format_ax(ax, title, ylabel=ylabel)

        # --- Stats ---
        stats_lines.append(curve_endpoint_stats(curves, title))
        stats_lines.append(auc_comparison(curves, title))

    # Per-seed final episode return
    stats_lines.append("  Per-seed final episode return (last 50 episodes, smoothed):")
    for mk in METHODS:
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            records = data[mk][seed].get("metrics", [])
            if records:
                last_scores = [r["episode/score"] for r in records[-50:] if "episode/score" in r]
                if last_scores:
                    arr = np.array(last_scores)
                    stats_lines.append(f"    {METHODS[mk]} seed={seed}: {arr.mean():.4f} ± {arr.std():.4f}")

    report.add_section("Fig 01: Episode Return & Length", "\n".join(stats_lines))

    fig.suptitle("Episode Metrics: CL vs Original DreamerV3", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_episode_return_length.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_episode_return_length.png")


def figure_online_return_and_success(data):
    """Fig 2: Rolling return mean and success rate from online_metrics."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    stats_lines = []

    for idx, (value_key, ylabel, title) in enumerate([
        ("return_mean", "Rolling Mean Return", "Online Mean Return (window=100)"),
        ("success_rate", "Success Rate", "Online Success Rate (window=100)"),
    ]):
        ax = axes[idx]
        curves = build_curves(data, "online", value_key, smooth=1)
        mats = {}
        ref_grid = None
        for mk in METHODS:
            grid, mat = curves[mk]
            if grid is not None and mat is not None:
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk])
                mats[mk] = mat
                ref_grid = grid
        if "cl" in mats and "original" in mats and ref_grid is not None:
            add_significance_annotation(ax, ref_grid, mats["cl"], mats["original"])
        format_ax(ax, title, ylabel=ylabel)

        stats_lines.append(curve_endpoint_stats(curves, title))
        stats_lines.append(auc_comparison(curves, title))

    # Steps to reach success_rate thresholds
    stats_lines.append("  Steps to reach success rate thresholds:")
    for threshold in [0.5, 0.8, 0.9, 0.95]:
        for mk in METHODS:
            steps_to = []
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                online = data[mk][seed].get("online", [])
                for r in online:
                    if r.get("success_rate", 0) >= threshold:
                        steps_to.append(r["step"])
                        break
                else:
                    steps_to.append(np.nan)
            arr = np.array(steps_to)
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                stats_lines.append(f"    {METHODS[mk]} → {threshold:.0%}: {valid.mean():.0f} ± {valid.std():.0f} steps ({len(valid)}/{len(arr)} seeds reached)")
            else:
                stats_lines.append(f"    {METHODS[mk]} → {threshold:.0%}: not reached by any seed")

    report.add_section("Fig 02: Online Return & Success Rate", "\n".join(stats_lines))

    fig.suptitle("Online Performance Metrics", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_online_return_success.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 02_online_return_success.png")


def figure_world_model_losses(data):
    """Fig 3: World model losses (obs, reward, continue, dynamics, representation)."""
    loss_keys = [
        ("loss/obs", "Observation Loss"),
        ("loss/rew", "Reward Loss"),
        ("loss/con", "Continue Loss"),
        ("loss/dyn", "Dynamics Loss"),
        ("loss/rep", "Representation Loss"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    stats_lines = []

    for idx, (value_key, title) in enumerate(loss_keys):
        ax = axes[idx]
        curves = build_curves(data, "online", value_key, smooth=30)
        mats = {}
        ref_grid = None
        for mk in METHODS:
            grid, mat = curves[mk]
            if grid is not None and mat is not None:
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk])
                mats[mk] = mat
                ref_grid = grid
        if "cl" in mats and "original" in mats and ref_grid is not None:
            add_significance_annotation(ax, ref_grid, mats["cl"], mats["original"])
        format_ax(ax, title, ylabel="Loss")

        stats_lines.append(curve_endpoint_stats(curves, title))

    # Policy & Value loss stats
    for value_key, ls, label_suffix in [
        ("loss/policy", "-", "Policy"), ("loss/value", "--", "Value")
    ]:
        curves = build_curves(data, "online", value_key, smooth=30)
        for mk in METHODS:
            grid, mat = curves[mk]
            if grid is not None and mat is not None:
                mu = mat.mean(axis=0)
                std = mat.std(axis=0)
                axes[5].plot(grid, mu, color=COLORS[mk], linestyle=ls,
                        label=f"{METHODS[mk]} ({label_suffix})", linewidth=1.5)
                axes[5].fill_between(grid, mu - std, mu + std,
                                color=COLORS[mk], alpha=ALPHA_FILL * 0.7)
        stats_lines.append(curve_endpoint_stats(curves, f"{label_suffix} Loss"))
    format_ax(axes[5], "Policy & Value Loss", ylabel="Loss")

    # Convergence analysis: step where loss drops below 2x final value
    stats_lines.append("\n  Convergence speed (step where loss first drops below 2× final value):")
    for value_key, title in loss_keys:
        for mk in METHODS:
            conv_steps = []
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                online = data[mk][seed].get("online", [])
                vals = [r[value_key] for r in online if value_key in r]
                steps = [r["step"] for r in online if value_key in r]
                if vals:
                    final_val = np.mean(vals[-20:])
                    threshold = 2.0 * final_val
                    for s, v in zip(steps, vals):
                        if v <= threshold:
                            conv_steps.append(s)
                            break
            if conv_steps:
                arr = np.array(conv_steps)
                stats_lines.append(f"    {title} / {METHODS[mk]}: {arr.mean():.0f} ± {arr.std():.0f} steps")

    report.add_section("Fig 03: World Model & Actor-Critic Losses", "\n".join(stats_lines))

    fig.suptitle("World Model & Actor-Critic Losses", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_world_model_losses.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 03_world_model_losses.png")


def figure_td_error(data):
    """Fig 4: TD error mean and max over training."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    stats_lines = []

    for idx, (vk1, vk2, ylabel, title) in enumerate([
        ("td_error_mean", "td_error/mean", "TD Error (mean)", "Mean TD Error"),
        ("td_error_max", "td_error/max", "TD Error (max)", "Max TD Error"),
    ]):
        ax = axes[idx]
        used_key = None
        for value_key in [vk1, vk2]:
            curves = build_curves(data, "online", value_key, smooth=30)
            any_data = False
            for mk in METHODS:
                grid, mat = curves[mk]
                if grid is not None and mat is not None:
                    plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk])
                    any_data = True
            if any_data:
                used_key = value_key
                break
        format_ax(ax, title, ylabel=ylabel)

        # Stats: raw (unsmoothed) final values from online_metrics
        stats_lines.append(f"  {title} (raw values from last 100 online entries):")
        for mk in METHODS:
            seed_means = []
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                online = data[mk][seed].get("online", [])
                vals = []
                for r in online[-100:]:
                    for k in [vk1, vk2]:
                        if k in r and r[k] is not None:
                            vals.append(r[k])
                            break
                if vals:
                    seed_means.append(np.mean(vals))
            if seed_means:
                arr = np.array(seed_means)
                stats_lines.append(f"    {METHODS[mk]}: {fmt_dist(arr)}")
            else:
                stats_lines.append(f"    {METHODS[mk]}: No data")

    report.add_section("Fig 04: Temporal Difference Error", "\n".join(stats_lines))

    fig.suptitle("Temporal Difference Error", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_td_error.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 04_td_error.png")


def figure_exploration_metrics(data):
    """Fig 5: Exploration / P2E metrics."""
    explore_keys = [
        ("explore/imagined_value", "Imagined Value"),
        ("explore/actual_value", "Actual Value"),
        ("explore/dream_accuracy", "Dream Accuracy"),
        ("explore/intr_extr_ratio", "Intrinsic/Extrinsic Ratio"),
        ("p2e/ensemble_disagreement", "Ensemble Disagreement"),
        ("p2e/intrinsic_reward", "Intrinsic Reward"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    stats_lines = []

    for idx, (value_key, title) in enumerate(explore_keys):
        ax = axes[idx]
        curves = build_curves(data, "online", value_key, smooth=30)
        any_data = False
        mats = {}
        ref_grid = None
        for mk in METHODS:
            grid, mat = curves[mk]
            if grid is not None and mat is not None:
                if np.abs(mat).max() < 1e-12:
                    continue
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk])
                mats[mk] = mat
                ref_grid = grid
                any_data = True
        if not any_data:
            ax.text(0.5, 0.5, "No data / all zeros", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
        if "cl" in mats and "original" in mats and ref_grid is not None:
            add_significance_annotation(ax, ref_grid, mats["cl"], mats["original"])
        format_ax(ax, title, ylabel="Value")

        # Stats for each exploration metric
        present_methods = []
        for mk in METHODS:
            grid, mat = curves[mk]
            if grid is not None and mat is not None and np.abs(mat).max() > 1e-12:
                present_methods.append(mk)

        if present_methods:
            stats_lines.append(f"  {title}:")
            for mk in present_methods:
                grid, mat = curves[mk]
                stats_lines.append(f"    {METHODS[mk]}: final={fmt_dist_short(mat[:, -1])}, "
                                   f"peak_mean={mat.mean(axis=0).max():.4f} at step ~{grid[mat.mean(axis=0).argmax()]:.0f}")
            if len(present_methods) == 2:
                _, p, sig = welch_t(mats["cl"][:, -1], mats["original"][:, -1])
                stats_lines.append(f"    Final-point Welch t-test: p={p:.4f} ({sig})")
        else:
            stats_lines.append(f"  {title}: All zeros or no data for both methods.")

    # Value overestimation ratio
    stats_lines.append("\n  Value Overestimation (imagined / actual at final point):")
    for mk in METHODS:
        imag_curves = build_curves(data, "online", "explore/imagined_value", smooth=30)
        act_curves = build_curves(data, "online", "explore/actual_value", smooth=30)
        ig, im = imag_curves[mk]
        ag, am = act_curves[mk]
        if ig is not None and ag is not None and im is not None and am is not None:
            imag_final = im[:, -1]
            act_final = am[:, -1]
            ratio = imag_final / np.maximum(act_final, 1e-8)
            stats_lines.append(f"    {METHODS[mk]}: ratio = {fmt_dist_short(ratio)}")

    report.add_section("Fig 05: Exploration & P2E Metrics", "\n".join(stats_lines))

    fig.suptitle("Exploration & Plan2Explore Metrics", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_exploration_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 05_exploration_metrics.png")


def figure_forgetting_and_frontier(data):
    """Fig 6: Aggregate forgetting and frontier rate over training."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    stats_lines = []

    for idx, (value_key, ylabel, title) in enumerate([
        ("aggregate_forgetting", "Aggregate Forgetting", "Aggregate Forgetting"),
        ("frontier_rate", "Frontier Rate", "Frontier Achievement Rate"),
    ]):
        ax = axes[idx]
        curves = build_curves(data, "online", value_key, smooth=20)
        mats = {}
        ref_grid = None
        for mk in METHODS:
            grid, mat = curves[mk]
            if grid is not None and mat is not None:
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk])
                mats[mk] = mat
                ref_grid = grid
        if "cl" in mats and "original" in mats and ref_grid is not None:
            add_significance_annotation(ax, ref_grid, mats["cl"], mats["original"])
        format_ax(ax, title, ylabel=ylabel)

        stats_lines.append(curve_endpoint_stats(curves, title))

    # Peak forgetting analysis
    stats_lines.append("  Peak aggregate forgetting (max over training):")
    for mk in METHODS:
        curves = build_curves(data, "online", "aggregate_forgetting", smooth=20)
        grid, mat = curves[mk]
        if grid is not None and mat is not None:
            peak_per_seed = mat.max(axis=1)
            peak_step_per_seed = grid[mat.argmax(axis=1)]
            stats_lines.append(f"    {METHODS[mk]}: peak = {fmt_dist_short(peak_per_seed)}, "
                               f"at step ~{peak_step_per_seed.mean():.0f} ± {peak_step_per_seed.std():.0f}")
    if "cl" in mats and "original" in mats:
        cl_peak = build_curves(data, "online", "aggregate_forgetting", smooth=20)["cl"][1].max(axis=1)
        orig_peak = build_curves(data, "online", "aggregate_forgetting", smooth=20)["original"][1].max(axis=1)
        _, p, sig = welch_t(cl_peak, orig_peak)
        stats_lines.append(f"    Peak forgetting Welch t-test: p={p:.4f} ({sig})")

    report.add_section("Fig 06: Forgetting & Frontier Rate", "\n".join(stats_lines))

    fig.suptitle("Continual Learning Diagnostics", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_forgetting_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 06_forgetting_frontier.png")


def figure_per_achievement_rates(data):
    """Fig 7: Per-achievement rate comparison at final evaluation (bar chart)."""
    fig, ax = plt.subplots(figsize=(16, 7))
    stats_lines = []

    final_rates = {}
    for mk in METHODS:
        seed_rates = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            summary = data[mk][seed].get("summary", {})
            rates = summary.get("per_task_achievement_rates", [[]])
            if rates and len(rates[0]) == len(ACHIEVEMENT_NAMES):
                seed_rates.append(rates[0])
        if seed_rates:
            final_rates[mk] = np.array(seed_rates)

    x = np.arange(len(ACHIEVEMENT_NAMES))
    width = 0.35

    for i, mk in enumerate(["cl", "original"]):
        if mk not in final_rates:
            continue
        mat = final_rates[mk]
        mu = mat.mean(axis=0)
        std = mat.std(axis=0)
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, mu, width, yerr=std, label=METHODS[mk],
                       color=COLORS[mk], alpha=0.85, capsize=3, edgecolor="white")

    # Significance per achievement + stats
    stats_lines.append("  Per-achievement rates (mean ± std) and significance tests:")
    stats_lines.append(f"    {'Achievement':<24} {'DreamerV3-CL':<20} {'DreamerV3-Original':<20} {'p-value':<16} {'Cohen d':<10}")
    stats_lines.append("    " + "-" * 90)

    if "cl" in final_rates and "original" in final_rates:
        for j in range(len(ACHIEVEMENT_NAMES)):
            a = final_rates["cl"][:, j]
            b = final_rates["original"][:, j]
            _, p, sig = welch_t(a, b)
            pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
            d = (a.mean() - b.mean()) / pooled if pooled > 1e-8 else 0.0
            p_str = f"{p:.4f} ({sig})" if not np.isnan(p) else "N/A"
            stats_lines.append(f"    {ACHIEVEMENT_NAMES[j]:<24} {fmt_dist_short(a):<20} {fmt_dist_short(b):<20} {p_str:<16} {d:+.3f}")

            if not np.isnan(p) and p < SIGNIFICANCE_LEVEL:
                y_top = max(a.mean() + a.std(), b.mean() + b.std()) + 0.03
                star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                ax.text(j, min(y_top, 1.05), star, ha="center", fontsize=8, fontweight="bold")

    # Aggregate: total number of achievements with rate > thresholds
    for thresh in [0.01, 0.05, 0.10, 0.50]:
        stats_lines.append(f"\n  Achievements with rate > {thresh:.0%}:")
        for mk in ["cl", "original"]:
            if mk in final_rates:
                counts = (final_rates[mk] > thresh).sum(axis=1).astype(float)
                stats_lines.append(f"    {METHODS[mk]}: {fmt_dist_short(counts)}")
        if "cl" in final_rates and "original" in final_rates:
            c = (final_rates["cl"] > thresh).sum(axis=1).astype(float)
            o = (final_rates["original"] > thresh).sum(axis=1).astype(float)
            _, p, sig = welch_t(c, o)
            stats_lines.append(f"    Welch t-test: p={p:.4f} ({sig})")

    report.add_section("Fig 07: Per-Achievement Rates", "\n".join(stats_lines))

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in ACHIEVEMENT_NAMES],
                        fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Achievement Rate", fontsize=12)
    ax.set_title("Per-Achievement Rates at Final Evaluation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_per_achievement_rates.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 07_per_achievement_rates.png")


def figure_per_achievement_forgetting(data):
    """Fig 8: Per-achievement forgetting at final evaluation."""
    fig, ax = plt.subplots(figsize=(16, 7))
    stats_lines = []

    final_forg = {}
    for mk in METHODS:
        seed_forg = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            online = data[mk][seed].get("online", [])
            if online:
                last = online[-1]
                forg = last.get("per_achievement_forgetting", [])
                if len(forg) == len(ACHIEVEMENT_NAMES):
                    seed_forg.append(forg)
        if seed_forg:
            final_forg[mk] = np.array(seed_forg)

    x = np.arange(len(ACHIEVEMENT_NAMES))
    width = 0.35

    for i, mk in enumerate(["cl", "original"]):
        if mk not in final_forg:
            continue
        mat = final_forg[mk]
        mu = mat.mean(axis=0)
        std = mat.std(axis=0)
        offset = -width / 2 + i * width
        ax.bar(x + offset, mu, width, yerr=std, label=METHODS[mk],
               color=COLORS[mk], alpha=0.85, capsize=3, edgecolor="white")

    # Stats table
    stats_lines.append("  Per-achievement forgetting (mean ± std) and significance:")
    stats_lines.append(f"    {'Achievement':<24} {'DreamerV3-CL':<20} {'DreamerV3-Original':<20} {'p-value':<16} {'Winner':<10}")
    stats_lines.append("    " + "-" * 90)
    if "cl" in final_forg and "original" in final_forg:
        for j in range(len(ACHIEVEMENT_NAMES)):
            a = final_forg["cl"][:, j]
            b = final_forg["original"][:, j]
            _, p, sig = welch_t(a, b)
            p_str = f"{p:.4f} ({sig})" if not np.isnan(p) else "N/A"
            winner = "CL" if a.mean() < b.mean() else ("Original" if b.mean() < a.mean() else "Tie")
            if a.mean() == 0 and b.mean() == 0:
                winner = "-"
            stats_lines.append(f"    {ACHIEVEMENT_NAMES[j]:<24} {fmt_dist_short(a):<20} {fmt_dist_short(b):<20} {p_str:<16} {winner:<10}")

        # Aggregate: mean forgetting across all achievements
        cl_agg = final_forg["cl"].mean(axis=1)
        orig_agg = final_forg["original"].mean(axis=1)
        _, p, sig = welch_t(cl_agg, orig_agg)
        stats_lines.append(f"\n  Mean forgetting across all achievements:")
        stats_lines.append(f"    DreamerV3-CL:       {fmt_dist(cl_agg)}")
        stats_lines.append(f"    DreamerV3-Original: {fmt_dist(orig_agg)}")
        stats_lines.append(f"    Welch t-test: p={p:.4f} ({sig})")

        # Top-3 most forgotten achievements per method
        for mk in ["cl", "original"]:
            mu = final_forg[mk].mean(axis=0)
            top3 = np.argsort(mu)[::-1][:3]
            names = [f"{ACHIEVEMENT_NAMES[i]} ({mu[i]:.4f})" for i in top3]
            stats_lines.append(f"  Top-3 most forgotten ({METHODS[mk]}): {', '.join(names)}")

    report.add_section("Fig 08: Per-Achievement Forgetting", "\n".join(stats_lines))

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in ACHIEVEMENT_NAMES],
                        fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Forgetting", fontsize=12)
    ax.set_title("Per-Achievement Forgetting at Final Step", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_per_achievement_forgetting.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08_per_achievement_forgetting.png")


def figure_score_distribution(data):
    """Fig 9: Score distribution at final evaluation."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    bins_labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]
    stats_lines = []

    all_dists = {}
    for idx, mk in enumerate(["cl", "original"]):
        ax = axes[idx]
        seed_dists = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            online = data[mk][seed].get("online", [])
            if online:
                dist = online[-1].get("score_distribution", [])
                if dist:
                    seed_dists.append(dist)
        if seed_dists:
            mat = np.array(seed_dists)
            all_dists[mk] = mat
            mu = mat.mean(axis=0)
            std = mat.std(axis=0)
            x = np.arange(len(mu))
            ax.bar(x, mu, yerr=std, color=COLORS[mk], alpha=0.85, capsize=4,
                   edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(bins_labels[:len(mu)], fontsize=10)

            stats_lines.append(f"  {METHODS[mk]} score distribution (mean ± std across seeds):")
            for b in range(len(mu)):
                lbl = bins_labels[b] if b < len(bins_labels) else f"bin_{b}"
                stats_lines.append(f"    {lbl}: {mu[b]:.4f} ± {std[b]:.4f}")

        ax.set_title(f"{METHODS[mk]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Score Bucket", fontsize=11)
        ax.set_ylabel("Fraction of Episodes", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # KL divergence between distributions
    if "cl" in all_dists and "original" in all_dists:
        cl_mu = all_dists["cl"].mean(axis=0)
        orig_mu = all_dists["original"].mean(axis=0)
        # Add small epsilon for numerical stability
        eps = 1e-8
        kl = np.sum(cl_mu * np.log((cl_mu + eps) / (orig_mu + eps)))
        stats_lines.append(f"\n  KL divergence (CL || Original): {kl:.6f}")

    # Note about score_distribution being per-episode
    stats_lines.append("\n  NOTE: score_distribution is computed from the last episode's window,")
    stats_lines.append("  not a running average. Interpret with caution.")

    report.add_section("Fig 09: Score Distribution", "\n".join(stats_lines))

    fig.suptitle("Score Distribution at Final Evaluation", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_score_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 09_score_distribution.png")


def figure_achievement_heatmap_over_time(data):
    """Fig 10: Achievement rate heatmap over training steps for each method."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    stats_lines = []

    for idx, mk in enumerate(["cl", "original"]):
        ax = axes[idx]
        seed_data = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            online = data[mk][seed].get("online", [])
            steps = []
            rates_over_time = []
            for r in online:
                par = r.get("per_achievement_rates", [])
                if len(par) == len(ACHIEVEMENT_NAMES):
                    steps.append(r["step"])
                    rates_over_time.append(par)
            if steps:
                seed_data.append((np.array(steps), np.array(rates_over_time)))

        if not seed_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        n_grid = 500
        lo = max(sd[0][0] for sd in seed_data)
        hi = min(sd[0][-1] for sd in seed_data)
        grid = np.linspace(lo, hi, n_grid)

        interped = []
        for s, r in seed_data:
            interp_r = np.zeros((n_grid, len(ACHIEVEMENT_NAMES)))
            for j in range(len(ACHIEVEMENT_NAMES)):
                interp_r[:, j] = np.interp(grid, s, r[:, j])
            interped.append(interp_r)
        mean_rates = np.mean(interped, axis=0).T  # (22, n_grid)

        im = ax.imshow(mean_rates, aspect="auto", cmap="YlOrRd",
                        extent=[grid[0], grid[-1], len(ACHIEVEMENT_NAMES) - 0.5, -0.5],
                        vmin=0, vmax=1)
        ax.set_yticks(range(len(ACHIEVEMENT_NAMES)))
        ax.set_yticklabels([n.replace("_", " ") for n in ACHIEVEMENT_NAMES], fontsize=8)
        ax.set_xlabel("Environment Steps", fontsize=11)
        ax.set_title(f"{METHODS[mk]}", fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
        plt.colorbar(im, ax=ax, label="Achievement Rate", shrink=0.8)

        # Stats: first step each achievement exceeds 10% and 50%
        stats_lines.append(f"  {METHODS[mk]} — Step where achievement first exceeds threshold (mean across seeds):")
        stats_lines.append(f"    {'Achievement':<24} {'> 10%':<18} {'> 50%':<18} {'Final rate':<15}")
        stats_lines.append("    " + "-" * 65)
        for j in range(len(ACHIEVEMENT_NAMES)):
            final_rate = mean_rates[j, -1]
            for thresh, label in [(0.10, "> 10%"), (0.50, "> 50%")]:
                exceed_idx = np.where(mean_rates[j] >= thresh)[0]
                if len(exceed_idx) > 0:
                    step_val = grid[exceed_idx[0]]
                    thresh_strs = {label: f"{step_val:.0f}"}
                else:
                    thresh_strs = {label: "never"}
            # Compute both thresholds
            t10_idx = np.where(mean_rates[j] >= 0.10)[0]
            t50_idx = np.where(mean_rates[j] >= 0.50)[0]
            t10 = f"{grid[t10_idx[0]]:.0f}" if len(t10_idx) > 0 else "never"
            t50 = f"{grid[t50_idx[0]]:.0f}" if len(t50_idx) > 0 else "never"
            stats_lines.append(f"    {ACHIEVEMENT_NAMES[j]:<24} {t10:<18} {t50:<18} {final_rate:.4f}")

    report.add_section("Fig 10: Achievement Heatmap Over Training", "\n".join(stats_lines))

    fig.suptitle("Achievement Rate Heatmap Over Training", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_achievement_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 10_achievement_heatmap.png")


def figure_summary_table(data):
    """Fig 11: Summary statistics table with significance tests."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    stats_lines = []

    metrics_to_compare = [
        ("mean_return", "Mean Return"),
        ("success_rate", "Success Rate"),
        ("per_task_lifetime_mean", "Lifetime Mean (task 0)"),
        ("per_task_aggregate_forgetting", "Aggregate Forgetting (task 0)"),
    ]

    rows = []
    stats_lines.append("  Summary statistics from metrics_summary.json:")
    stats_lines.append(f"    {'Metric':<30} {'DreamerV3-CL':<28} {'DreamerV3-Original':<28} {'t-stat':<10} {'p-value':<12} {'Sig':<6} {'Cohen d':<10}")
    stats_lines.append("    " + "-" * 124)

    for key, label in metrics_to_compare:
        vals = {}
        for mk in METHODS:
            seed_vals = []
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                summary = data[mk][seed].get("summary", {})
                v = summary.get(key)
                if v is not None:
                    if isinstance(v, list):
                        v = v[0] if v else None
                    if v is not None:
                        seed_vals.append(v)
            vals[mk] = np.array(seed_vals) if seed_vals else np.array([])

        cl_vals = vals.get("cl", np.array([]))
        orig_vals = vals.get("original", np.array([]))

        cl_str = f"{cl_vals.mean():.4f} ± {cl_vals.std():.4f}" if len(cl_vals) > 0 else "N/A"
        orig_str = f"{orig_vals.mean():.4f} ± {orig_vals.std():.4f}" if len(orig_vals) > 0 else "N/A"

        if len(cl_vals) >= 2 and len(orig_vals) >= 2:
            t, p = stats.ttest_ind(cl_vals, orig_vals, equal_var=False)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            p_str = f"{p:.4f} ({sig})"
            pooled = np.sqrt((cl_vals.std()**2 + orig_vals.std()**2) / 2)
            d = (cl_vals.mean() - orig_vals.mean()) / pooled if pooled > 1e-8 else 0.0
            stats_lines.append(f"    {label:<30} {cl_str:<28} {orig_str:<28} {t:+.4f}    {p:.4f}       {sig:<6} {d:+.4f}")
        else:
            p_str = "N/A"
            stats_lines.append(f"    {label:<30} {cl_str:<28} {orig_str:<28} {'N/A':<10} {'N/A':<12} {'N/A':<6} {'N/A':<10}")

        rows.append([label, cl_str, orig_str, p_str])

    # Per-seed raw values for transparency
    stats_lines.append("\n  Per-seed raw values:")
    for key, label in metrics_to_compare:
        stats_lines.append(f"    {label}:")
        for mk in METHODS:
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                summary = data[mk][seed].get("summary", {})
                v = summary.get(key)
                if isinstance(v, list):
                    v = v[0] if v else None
                stats_lines.append(f"      {METHODS[mk]} seed={seed}: {v}")

    # Achievements unlocked
    cl_unlock = orig_unlock = np.array([])
    cl_str = orig_str = "N/A"
    for mk in METHODS:
        seed_totals = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            summary = data[mk][seed].get("summary", {})
            rates = summary.get("per_task_achievement_rates", [[]])
            if rates and rates[0]:
                seed_totals.append(sum(1 for r in rates[0] if r > 0.01))
        if seed_totals:
            vals_arr = np.array(seed_totals, dtype=float)
            if mk == "cl":
                cl_unlock = vals_arr
                cl_str = f"{vals_arr.mean():.1f} ± {vals_arr.std():.1f}"
            else:
                orig_unlock = vals_arr
                orig_str = f"{vals_arr.mean():.1f} ± {vals_arr.std():.1f}"

    if len(cl_unlock) >= 2 and len(orig_unlock) >= 2:
        _, p = stats.ttest_ind(cl_unlock, orig_unlock, equal_var=False)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        p_str = f"{p:.4f} ({sig})"
    else:
        p_str = "N/A"
    rows.append(["Achievements Unlocked (>1%)", cl_str, orig_str, p_str])
    stats_lines.append(f"\n  Achievements unlocked (>1%): CL={cl_str}, Original={orig_str}, p={p_str}")

    report.add_section("Fig 11: Summary Table", "\n".join(stats_lines))

    col_labels = ["Metric", "DreamerV3-CL", "DreamerV3-Original", "p-value (Welch t)"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                      cellLoc="center", colWidths=[0.28, 0.24, 0.24, 0.24])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    ax.set_title("Summary Statistics with Significance Tests (3 seeds)",
                  fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_summary_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 11_summary_table.png")


def figure_top_achievement_curves(data):
    """Fig 12: Top-8 most active achievements tracked over time."""
    all_final_rates = []
    for mk in METHODS:
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            summary = data[mk][seed].get("summary", {})
            rates = summary.get("per_task_achievement_rates", [[]])
            if rates and len(rates[0]) == len(ACHIEVEMENT_NAMES):
                all_final_rates.append(rates[0])

    if not all_final_rates:
        print("  Skipping top achievement curves (no data)")
        return

    mean_final = np.mean(all_final_rates, axis=0)
    top_indices = np.argsort(mean_final)[::-1][:8]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    stats_lines = []

    stats_lines.append("  Top-8 achievements by mean final rate (across both methods):")
    stats_lines.append(f"    {'Rank':<6} {'Achievement':<24} {'CL final':<20} {'Orig final':<20} {'p-value':<16} {'Cohen d':<10}")
    stats_lines.append("    " + "-" * 96)

    for plot_idx, ach_idx in enumerate(top_indices):
        ax = axes[plot_idx]
        ach_name = ACHIEVEMENT_NAMES[ach_idx]

        mats = {}
        ref_grid = None
        for mk in METHODS:
            all_s, all_v = [], []
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                online = data[mk][seed].get("online", [])
                steps, vals = [], []
                for r in online:
                    par = r.get("per_achievement_rates", [])
                    if len(par) > ach_idx:
                        steps.append(r["step"])
                        vals.append(par[ach_idx])
                if len(steps) > 1:
                    s = np.array(steps, dtype=np.float64)
                    v = uniform_filter1d(np.array(vals, dtype=np.float64),
                                         size=min(20, len(vals)))
                    all_s.append(s)
                    all_v.append(v)
            grid, mat = interpolate_to_common_grid(all_s, all_v)
            if grid is not None and mat is not None:
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk])
                mats[mk] = mat
                ref_grid = grid

        if "cl" in mats and "original" in mats and ref_grid is not None:
            add_significance_annotation(ax, ref_grid, mats["cl"], mats["original"])

        format_ax(ax, ach_name.replace("_", " ").title(), ylabel="Rate")
        ax.set_ylim(-0.05, 1.05)

        # Stats for this achievement
        cl_f = mats["cl"][:, -1] if "cl" in mats else np.array([])
        orig_f = mats["original"][:, -1] if "original" in mats else np.array([])
        _, p, sig = welch_t(cl_f, orig_f)
        pooled = np.sqrt((cl_f.std()**2 + orig_f.std()**2) / 2) if len(cl_f) > 0 and len(orig_f) > 0 else 0
        d = (cl_f.mean() - orig_f.mean()) / pooled if pooled > 1e-8 else 0.0
        p_str = f"{p:.4f} ({sig})" if not np.isnan(p) else "N/A"
        stats_lines.append(f"    {plot_idx+1:<6} {ach_name:<24} {fmt_dist_short(cl_f):<20} {fmt_dist_short(orig_f):<20} {p_str:<16} {d:+.3f}")

    report.add_section("Fig 12: Top-8 Achievement Curves", "\n".join(stats_lines))

    fig.suptitle("Top-8 Achievement Rate Curves Over Training",
                  fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_top_achievement_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 12_top_achievement_curves.png")


def figure_max_return_and_depth(data):
    """Fig 13: Per-task max return and max achievement depth over time."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    stats_lines = []

    for idx, (value_key, ylabel, title) in enumerate([
        ("per_task_max_return", "Max Return", "Per-Task Max Return Over Time"),
        ("per_task_max_depth", "Max Depth", "Per-Task Max Achievement Depth"),
    ]):
        ax = axes[idx]
        mats = {}
        ref_grid = None
        for mk in METHODS:
            all_s, all_v = [], []
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                online = data[mk][seed].get("online", [])
                steps, vals = [], []
                for r in online:
                    v = r.get(value_key, [])
                    if isinstance(v, list) and len(v) > 0:
                        steps.append(r["step"])
                        vals.append(v[0])
                if len(steps) > 1:
                    all_s.append(np.array(steps, dtype=np.float64))
                    all_v.append(np.array(vals, dtype=np.float64))
            grid, mat = interpolate_to_common_grid(all_s, all_v)
            if grid is not None and mat is not None:
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk])
                mats[mk] = mat
                ref_grid = grid
        format_ax(ax, title, ylabel=ylabel)

        # Stats
        stats_lines.append(f"  {title}:")
        for mk in METHODS:
            if mk in mats:
                final_vals = mats[mk][:, -1]
                stats_lines.append(f"    {METHODS[mk]} final: {fmt_dist(final_vals)}")
        if "cl" in mats and "original" in mats:
            _, p, sig = welch_t(mats["cl"][:, -1], mats["original"][:, -1])
            stats_lines.append(f"    Welch t-test: p={p:.4f} ({sig})")

    # Step where max return first reaches various thresholds
    stats_lines.append("\n  Steps to reach max return thresholds:")
    for thresh in [3.0, 5.0, 6.0, 7.0]:
        for mk in METHODS:
            steps_to = []
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                online = data[mk][seed].get("online", [])
                for r in online:
                    v = r.get("per_task_max_return", [])
                    if isinstance(v, list) and len(v) > 0 and v[0] >= thresh:
                        steps_to.append(r["step"])
                        break
                else:
                    steps_to.append(np.nan)
            valid = np.array([s for s in steps_to if not np.isnan(s)])
            if len(valid) > 0:
                stats_lines.append(f"    {METHODS[mk]} → max_return≥{thresh}: {valid.mean():.0f} ± {valid.std():.0f} ({len(valid)}/{len(steps_to)} seeds)")
            else:
                stats_lines.append(f"    {METHODS[mk]} → max_return≥{thresh}: not reached")

    report.add_section("Fig 13: Max Return & Depth", "\n".join(stats_lines))

    fig.suptitle("Performance Envelope", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_max_return_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 13_max_return_depth.png")


def figure_individual_seed_runs(data):
    """Fig 14: Individual seed trajectories (episode return) for transparency."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    seed_linestyles = {1: "-", 4: "--", 42: ":"}
    stats_lines = []

    stats_lines.append("  Per-seed final episode return (smoothed, last value):")
    stats_lines.append(f"    {'Method':<24} {'Seed':<8} {'Final Return':<16} {'Num Episodes':<14} {'Total Steps':<14}")
    stats_lines.append("    " + "-" * 76)

    for idx, mk in enumerate(["cl", "original"]):
        ax = axes[idx]
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            records = data[mk][seed].get("metrics", [])
            s, v = extract_scalar_curve(records, "step", "episode/score", smooth=SMOOTH_WINDOW)
            if len(s) > 0:
                ax.plot(s, v, linestyle=seed_linestyles[seed], color=COLORS[mk],
                        label=f"seed={seed}", alpha=0.8, linewidth=1.5)
                stats_lines.append(f"    {METHODS[mk]:<24} {seed:<8} {v[-1]:<16.4f} {len(records):<14} {s[-1]:<14.0f}")
        format_ax(ax, f"{METHODS[mk]}", ylabel="Episode Return")

    # Inter-seed variance analysis
    stats_lines.append("\n  Inter-seed variance (coefficient of variation at final point):")
    for mk in METHODS:
        final_vals = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            records = data[mk][seed].get("metrics", [])
            s, v = extract_scalar_curve(records, "step", "episode/score", smooth=SMOOTH_WINDOW)
            if len(v) > 0:
                final_vals.append(v[-1])
        if len(final_vals) >= 2:
            arr = np.array(final_vals)
            cv = arr.std() / arr.mean() if arr.mean() > 0 else 0
            stats_lines.append(f"    {METHODS[mk]}: CV = {cv:.4f} (std/mean = {arr.std():.4f}/{arr.mean():.4f})")

    # Rank correlation across seeds
    stats_lines.append("\n  Seed ranking by final return:")
    for mk in METHODS:
        seed_returns = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            records = data[mk][seed].get("metrics", [])
            s, v = extract_scalar_curve(records, "step", "episode/score", smooth=SMOOTH_WINDOW)
            if len(v) > 0:
                seed_returns.append((seed, v[-1]))
        seed_returns.sort(key=lambda x: -x[1])
        ranking = " > ".join([f"seed={s} ({r:.3f})" for s, r in seed_returns])
        stats_lines.append(f"    {METHODS[mk]}: {ranking}")

    report.add_section("Fig 14: Individual Seed Trajectories", "\n".join(stats_lines))

    fig.suptitle("Individual Seed Trajectories (Episode Return)",
                  fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "14_individual_seeds.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 14_individual_seeds.png")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Craftax: DreamerV3-CL vs DreamerV3-Original Comparative Analysis")
    print("=" * 70)
    print(f"\nLoading data from {LOGS_DIR.resolve()} ...")
    data = load_all_runs()

    total_runs = sum(len(v) for v in data.values())
    print(f"\nLoaded {total_runs} / 6 expected runs.\n")
    if total_runs == 0:
        print("ERROR: No data found. Check LOGS_DIR and folder naming.")
        return

    print("Generating figures and statistics...\n")
    figure_episode_return_and_length(data)
    figure_online_return_and_success(data)
    figure_world_model_losses(data)
    figure_td_error(data)
    figure_exploration_metrics(data)
    figure_forgetting_and_frontier(data)
    figure_per_achievement_rates(data)
    figure_per_achievement_forgetting(data)
    figure_score_distribution(data)
    figure_achievement_heatmap_over_time(data)
    figure_summary_table(data)
    figure_top_achievement_curves(data)
    figure_max_return_and_depth(data)
    figure_individual_seed_runs(data)

    # Write the stats report
    report_path = OUTPUT_DIR / "statistics_report.txt"
    report.write(report_path)
    print(f"\n  Saved statistics_report.txt")

    print(f"\nAll figures and stats saved to {OUTPUT_DIR.resolve()}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
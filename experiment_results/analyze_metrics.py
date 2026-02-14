#!/usr/bin/env python3
"""
Comparative Analysis: Continual Learning Experiments on Craftax
================================================================
Auto-discovers experiment methods from the logs/ directory, loads metrics
from all found runs (N methods x M seeds), computes mean +/- std across
seeds, and produces publication-quality figures with statistical annotations
(pairwise Welch's t-tests at final evaluation).

Folder naming convention:
    craftax_dreamerv3-<method_key>-<seed>
    e.g.  craftax_dreamerv3-cl-1, craftax_dreamerv3-original-42,
          craftax_dreamerv3-nlr_sampling-4
"""

import json
import os
import pathlib
import re
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
            methods_str = ", ".join(f"{k}={v}" for k, v in METHODS.items())
            f.write(f"Methods: {methods_str}\n")
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
    """Format an array as 'mean +/- std [min, max]'."""
    if len(arr) == 0:
        return "N/A"
    return f"{arr.mean():.4f} +/- {arr.std():.4f}  [min={arr.min():.4f}, max={arr.max():.4f}]"


def fmt_dist_short(arr):
    """Format an array as 'mean +/- std'."""
    if len(arr) == 0:
        return "N/A"
    return f"{arr.mean():.4f} +/- {arr.std():.4f}"


def curve_endpoint_stats(curves, metric_name):
    """
    Given curves dict from build_curves, compute stats at 25%, 50%, 75%, 100%
    of training and pairwise Welch's t-tests. Returns formatted string.
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
    # Dynamic header based on discovered methods
    method_headers = "  ".join(f"{METHODS[mk]:<28}" for mk in mats)
    header = f"    {'Checkpoint':<12} {method_headers}"
    lines.append(header)
    lines.append("    " + "-" * (12 + 30 * len(mats)))

    for label, frac in checkpoints.items():
        idx = min(int(frac * (n - 1)), n - 1)
        parts = [f"{label:<12}"]
        for mk in mats:
            vals = mats[mk][:, idx]
            parts.append(f"{fmt_dist_short(vals):<28}")
        lines.append("    " + "  ".join(parts))

    # Pairwise Welch's t-test at final point
    method_keys = list(mats.keys())
    if len(method_keys) >= 2:
        lines.append("    Pairwise Welch t-tests at final point:")
        for i in range(len(method_keys)):
            for j in range(i + 1, len(method_keys)):
                mk_a, mk_b = method_keys[i], method_keys[j]
                a_f = mats[mk_a][:, -1]
                b_f = mats[mk_b][:, -1]
                _, p, sig = welch_t(a_f, b_f)
                p_str = f"p={p:.4f} ({sig})" if not np.isnan(p) else "N/A"
                lines.append(f"      {METHODS[mk_a]} vs {METHODS[mk_b]}: {p_str}")

        # Cohen's d for all pairs
        for i in range(len(method_keys)):
            for j in range(i + 1, len(method_keys)):
                mk_a, mk_b = method_keys[i], method_keys[j]
                a_f = mats[mk_a][:, -1]
                b_f = mats[mk_b][:, -1]
                pooled_std = np.sqrt((a_f.std()**2 + b_f.std()**2) / 2)
                cohens_d = (a_f.mean() - b_f.mean()) / pooled_std if pooled_std > 1e-12 else 0.0
                lines.append(f"      Cohen's d ({METHODS[mk_a]} vs {METHODS[mk_b]}): {cohens_d:+.4f}")

    return "\n".join(lines) + "\n"


def compute_auc(grid, matrix):
    """Compute Area Under Curve (trapezoidal) for each seed, return array."""
    aucs = []
    for i in range(matrix.shape[0]):
        _trapz = getattr(np, 'trapezoid', None) or np.trapz
        aucs.append(_trapz(matrix[i], grid))
    return np.array(aucs)


def auc_comparison(curves, metric_name):
    """Compute and compare AUC between all methods (pairwise)."""
    lines = []
    mats, grids = {}, {}
    for mk in METHODS:
        grid, mat = curves[mk]
        if grid is not None and mat is not None:
            mats[mk] = mat
            grids[mk] = grid

    if len(mats) < 2:
        return f"  {metric_name} AUC: Insufficient data for comparison.\n"

    lines.append(f"  {metric_name} -- Area Under Curve (AUC):")
    aucs = {}
    for mk in mats:
        auc_arr = compute_auc(grids[mk], mats[mk])
        aucs[mk] = auc_arr
        lines.append(f"    {METHODS[mk]}: {fmt_dist(auc_arr)}")

    # Pairwise AUC comparisons
    method_keys = list(aucs.keys())
    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            mk_a, mk_b = method_keys[i], method_keys[j]
            _, p, sig = welch_t(aucs[mk_a], aucs[mk_b])
            lines.append(f"    AUC Welch t-test ({METHODS[mk_a]} vs {METHODS[mk_b]}): p={p:.4f} ({sig})")

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
LOGS_DIR = pathlib.Path("logs")
SMOOTH_WINDOW = 50  # rolling window for smoothing curves
INTERP_STEPS = 1000  # number of interpolation points on x-axis
ALPHA_FILL = 0.18
MAX_SIG_ANNOTATIONS = 3  # max pairwise significance annotations on a single plot
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
# Auto-discovery of methods and seeds
# ──────────────────────────────────────────────────────────────────────
# These globals will be populated at startup by discover_methods()

METHODS = {}      # method_key -> display_name, e.g. {"cl": "DreamerV3-CL", ...}
SEEDS = []        # sorted list of discovered seeds
COLORS = {}       # method_key -> hex color
LINESTYLES = {}   # method_key -> linestyle (solid, dashed, dotted, ...)
MARKERS = {}      # method_key -> marker char ('o', 's', 'D', ...)

# A colour palette that is colour-blind-friendly and extends to many methods.
# The first two entries match the original colours for cl and original.
_BASE_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

_BASE_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)),
                    (0, (3, 5, 1, 5, 1, 5)), "-", "--", "-."]
_BASE_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "p"]


def _fig_size(base_w, base_h, n_methods=None, scale_w=True, scale_h=False):
    """Return (w, h) figure size scaled to number of methods.

    For shared-axis plots (all curves overlaid): widen for legend room.
    For per-method subplot plots: caller controls n_cols/n_rows instead.
    """
    if n_methods is None:
        n_methods = len(METHODS)
    w = base_w * max(1.0, n_methods / 2.5) if scale_w else base_w
    h = base_h * max(1.0, n_methods / 2.5) if scale_h else base_h
    return (w, h)


def _method_display_name(key: str) -> str:
    """Convert a method key to a human-readable display name.

    Examples:
        'cl'           -> 'DreamerV3-CL'
        'original'     -> 'DreamerV3-Original'
        'nlr_sampling' -> 'DreamerV3-NLR-Sampling'
    """
    # Keep legacy names for known methods
    _known = {
        "cl": "DreamerV3-CL",
        "original": "DreamerV3-Original",
    }
    if key in _known:
        return _known[key]
    # General rule: title-case, replace underscores with hyphens
    pretty = key.replace("_", "-").title()
    return f"DreamerV3-{pretty}"


def discover_methods(logs_dir: pathlib.Path):
    """Scan *logs_dir* for folders matching ``craftax_dreamerv3-<method>-<seed>``
    and populate the global METHODS, SEEDS, COLORS dicts.

    Also handles the pattern ``craftax_<method>-<seed>`` as a fallback.
    """
    global METHODS, SEEDS, COLORS, LINESTYLES, MARKERS

    if not logs_dir.exists():
        print(f"  [ERROR] Logs directory not found: {logs_dir.resolve()}")
        return

    method_seeds = defaultdict(set)
    # Primary pattern: craftax_dreamerv3-<method>-<seed>
    pattern_primary = re.compile(r"^craftax_dreamerv3-(.+)-(\d+)$")
    # Fallback pattern: craftax_<method>-<seed>  (but NOT craftax_dreamerv3-*)
    pattern_fallback = re.compile(r"^craftax_(.+)-(\d+)$")

    for entry in sorted(logs_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        m = pattern_primary.match(name)
        if m:
            method_key = m.group(1)
            seed = int(m.group(2))
            method_seeds[method_key].add(seed)
            continue
        m = pattern_fallback.match(name)
        if m:
            method_key = m.group(1)
            seed = int(m.group(2))
            # Avoid capturing partial matches like "dreamerv3" itself
            if method_key != "dreamerv3":
                method_seeds[method_key].add(seed)

    if not method_seeds:
        print("  [WARNING] No experiment folders discovered in logs/.")
        return

    # Collect all seeds (union across all methods)
    all_seeds = set()
    for seeds in method_seeds.values():
        all_seeds.update(seeds)
    SEEDS = sorted(all_seeds)

    # Build METHODS dict, preserving a deterministic order:
    # known methods first (cl, original), then alphabetical
    known_order = ["cl", "original"]
    ordered_keys = [k for k in known_order if k in method_seeds]
    remaining = sorted(k for k in method_seeds if k not in known_order)
    ordered_keys.extend(remaining)

    METHODS = {}
    COLORS = {}
    LINESTYLES = {}
    MARKERS = {}
    for i, mk in enumerate(ordered_keys):
        METHODS[mk] = _method_display_name(mk)
        COLORS[mk] = _BASE_PALETTE[i % len(_BASE_PALETTE)]
        LINESTYLES[mk] = _BASE_LINESTYLES[i % len(_BASE_LINESTYLES)]
        MARKERS[mk] = _BASE_MARKERS[i % len(_BASE_MARKERS)]

    print(f"  Discovered {len(METHODS)} methods: {list(METHODS.keys())}")
    print(f"  Seeds: {SEEDS}")
    for mk in METHODS:
        found_seeds = sorted(method_seeds[mk])
        print(f"    {mk}: seeds {found_seeds}")


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
                # Try fallback pattern without dreamerv3 prefix
                folder = LOGS_DIR / f"craftax_{method_key}-{seed}"
                if not folder.exists():
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

def plot_mean_std(ax, grid, matrix, color, label, method_key=None):
    """Plot mean line with shaded std band.

    When *method_key* is given, linestyle and marker are looked up from
    the global dicts so that each method is visually distinct even in
    grayscale or with many overlapping curves.
    """
    mu = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    ls = LINESTYLES.get(method_key, "-") if method_key else "-"
    mk = MARKERS.get(method_key, None) if method_key else None
    # markevery: show a marker every ~5% of the grid points
    me = max(1, len(grid) // 20) if mk else None
    ax.plot(grid, mu, color=color, label=label, linewidth=2,
            linestyle=ls, marker=mk, markevery=me, markersize=4)
    ax.fill_between(grid, mu - std, mu + std, color=color, alpha=ALPHA_FILL)


def add_pairwise_significance(ax, grid, mats, y_pos=None):
    """Add significance stars at the final step for all pairs of methods.

    Only annotates pairs where p < SIGNIFICANCE_LEVEL. Uses a stacked
    offset to avoid overlapping annotations when many pairs are significant.
    """
    method_keys = [mk for mk in METHODS if mk in mats]
    if len(method_keys) < 2:
        return

    annotations = []
    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            mk_a, mk_b = method_keys[i], method_keys[j]
            final_a = mats[mk_a][:, -1]
            final_b = mats[mk_b][:, -1]
            if len(final_a) < 2 or len(final_b) < 2:
                continue
            _, p_val = stats.ttest_ind(final_a, final_b, equal_var=False)
            if p_val < SIGNIFICANCE_LEVEL:
                star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else "*")
                annotations.append((star, p_val, mk_a, mk_b, final_a, final_b))

    if not annotations:
        return

    # Sort by p-value (most significant first) and cap count to avoid clutter
    annotations.sort(key=lambda x: x[1])
    annotations = annotations[:MAX_SIG_ANNOTATIONS]

    # Stack annotations vertically
    base_y = y_pos
    if base_y is None:
        all_final_means = [mats[mk][:, -1].mean() for mk in mats]
        base_y = max(all_final_means)

    y_range = ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] != ax.get_ylim()[0] else 1.0
    for k, (star, p_val, mk_a, mk_b, _, _) in enumerate(annotations):
        y_offset = base_y + (k + 1) * y_range * 0.05
        short_a = METHODS[mk_a].split("-")[-1][:6]
        short_b = METHODS[mk_b].split("-")[-1][:6]
        ax.annotate(
            f"{star} p={p_val:.3f}\n({short_a} vs {short_b})",
            xy=(grid[-1], y_offset),
            fontsize=7,
            ha="right",
            va="bottom",
            color="black",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
        )


def format_ax(ax, title, xlabel="Environment Steps", ylabel=""):
    """Format axis with adaptive legend placement based on number of methods."""
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    n = len(METHODS)
    if n <= 3:
        ax.legend(fontsize=9, loc="best")
    else:
        # Many methods: use smaller font and place legend outside to avoid
        # occluding curves.  ncol keeps it compact.
        ax.legend(fontsize=8, loc="upper left", ncol=max(1, n // 2),
                  framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_curves(data, source, value_key, step_key="step", smooth=SMOOTH_WINDOW):
    """Build interpolated matrices for all methods."""
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


def _method_keys_sorted():
    """Return method keys in deterministic plot order."""
    return list(METHODS.keys())


# ──────────────────────────────────────────────────────────────────────
# Main Analysis & Figures
# ──────────────────────────────────────────────────────────────────────

def figure_episode_return_and_length(data):
    """Fig 1: Episode return and episode length over training steps."""
    fig, axes = plt.subplots(1, 2, figsize=_fig_size(14, 5))
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
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk], method_key=mk)
                mats[mk] = mat
                ref_grid = grid
        if ref_grid is not None and len(mats) >= 2:
            add_pairwise_significance(ax, ref_grid, mats)
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
                    stats_lines.append(f"    {METHODS[mk]} seed={seed}: {arr.mean():.4f} +/- {arr.std():.4f}")

    report.add_section("Fig 01: Episode Return & Length", "\n".join(stats_lines))

    n_methods = len(METHODS)
    fig.suptitle(f"Episode Metrics ({n_methods} methods)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_episode_return_length.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_episode_return_length.png")


def figure_online_return_and_success(data):
    """Fig 2: Rolling return mean and success rate from online_metrics."""
    fig, axes = plt.subplots(1, 2, figsize=_fig_size(14, 5))
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
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk], method_key=mk)
                mats[mk] = mat
                ref_grid = grid
        if ref_grid is not None and len(mats) >= 2:
            add_pairwise_significance(ax, ref_grid, mats)
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
                stats_lines.append(f"    {METHODS[mk]} -> {threshold:.0%}: {valid.mean():.0f} +/- {valid.std():.0f} steps ({len(valid)}/{len(arr)} seeds reached)")
            else:
                stats_lines.append(f"    {METHODS[mk]} -> {threshold:.0%}: not reached by any seed")

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
    fig, axes = plt.subplots(2, 3, figsize=_fig_size(16, 9))
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
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk], method_key=mk)
                mats[mk] = mat
                ref_grid = grid
        if ref_grid is not None and len(mats) >= 2:
            add_pairwise_significance(ax, ref_grid, mats)
        format_ax(ax, title, ylabel="Loss")

        stats_lines.append(curve_endpoint_stats(curves, title))

    # Policy & Value loss on the last subplot
    for value_key, ls, label_suffix in [
        ("loss/policy", "-", "Policy"), ("loss/value", "--", "Value")
    ]:
        curves = build_curves(data, "online", value_key, smooth=30)
        for mk in METHODS:
            grid, mat = curves[mk]
            if grid is not None and mat is not None:
                mu = mat.mean(axis=0)
                std = mat.std(axis=0)
                mk_marker = MARKERS.get(mk)
                me = max(1, len(grid) // 20) if mk_marker else None
                axes[5].plot(grid, mu, color=COLORS[mk], linestyle=ls,
                        label=f"{METHODS[mk]} ({label_suffix})", linewidth=1.5,
                        marker=mk_marker, markevery=me, markersize=3)
                axes[5].fill_between(grid, mu - std, mu + std,
                                color=COLORS[mk], alpha=ALPHA_FILL * 0.7)
        stats_lines.append(curve_endpoint_stats(curves, f"{label_suffix} Loss"))
    format_ax(axes[5], "Policy & Value Loss", ylabel="Loss")

    # Convergence analysis: step where loss drops below 2x final value
    stats_lines.append("\n  Convergence speed (step where loss first drops below 2x final value):")
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
                stats_lines.append(f"    {title} / {METHODS[mk]}: {arr.mean():.0f} +/- {arr.std():.0f} steps")

    report.add_section("Fig 03: World Model & Actor-Critic Losses", "\n".join(stats_lines))

    fig.suptitle("World Model & Actor-Critic Losses", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_world_model_losses.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 03_world_model_losses.png")


def figure_td_error(data):
    """Fig 4: TD error mean and max over training."""
    fig, axes = plt.subplots(1, 2, figsize=_fig_size(14, 5))
    stats_lines = []

    for idx, (vk1, vk2, ylabel, title) in enumerate([
        ("td_error_mean", "td_error/mean", "TD Error (mean)", "Mean TD Error"),
        ("td_error_max", "td_error/max", "TD Error (max)", "Max TD Error"),
    ]):
        ax = axes[idx]
        for value_key in [vk1, vk2]:
            curves = build_curves(data, "online", value_key, smooth=30)
            any_data = False
            for mk in METHODS:
                grid, mat = curves[mk]
                if grid is not None and mat is not None:
                    plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk], method_key=mk)
                    any_data = True
            if any_data:
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
    fig, axes = plt.subplots(2, 3, figsize=_fig_size(16, 9))
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
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk], method_key=mk)
                mats[mk] = mat
                ref_grid = grid
                any_data = True
        if not any_data:
            ax.text(0.5, 0.5, "No data / all zeros", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
        if ref_grid is not None and len(mats) >= 2:
            add_pairwise_significance(ax, ref_grid, mats)
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
            # Pairwise tests
            if len(present_methods) >= 2:
                for i in range(len(present_methods)):
                    for j in range(i + 1, len(present_methods)):
                        mk_a, mk_b = present_methods[i], present_methods[j]
                        _, p, sig = welch_t(mats[mk_a][:, -1], mats[mk_b][:, -1])
                        stats_lines.append(f"    Welch t ({METHODS[mk_a]} vs {METHODS[mk_b]}): p={p:.4f} ({sig})")
        else:
            stats_lines.append(f"  {title}: All zeros or no data for all methods.")

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
    fig, axes = plt.subplots(1, 2, figsize=_fig_size(14, 5))
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
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk], method_key=mk)
                mats[mk] = mat
                ref_grid = grid
        if ref_grid is not None and len(mats) >= 2:
            add_pairwise_significance(ax, ref_grid, mats)
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
                               f"at step ~{peak_step_per_seed.mean():.0f} +/- {peak_step_per_seed.std():.0f}")

    # Pairwise peak forgetting tests
    method_keys = list(METHODS.keys())
    if len(method_keys) >= 2:
        forg_curves = build_curves(data, "online", "aggregate_forgetting", smooth=20)
        peak_data = {}
        for mk in method_keys:
            grid, mat = forg_curves[mk]
            if grid is not None and mat is not None:
                peak_data[mk] = mat.max(axis=1)
        pk_keys = list(peak_data.keys())
        for i in range(len(pk_keys)):
            for j in range(i + 1, len(pk_keys)):
                mk_a, mk_b = pk_keys[i], pk_keys[j]
                _, p, sig = welch_t(peak_data[mk_a], peak_data[mk_b])
                stats_lines.append(f"    Peak forgetting ({METHODS[mk_a]} vs {METHODS[mk_b]}): p={p:.4f} ({sig})")

    report.add_section("Fig 06: Forgetting & Frontier Rate", "\n".join(stats_lines))

    fig.suptitle("Continual Learning Diagnostics", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_forgetting_frontier.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 06_forgetting_frontier.png")


def figure_per_achievement_rates(data):
    """Fig 7: Per-achievement rate comparison at final evaluation (grouped bar chart)."""
    n_methods = len(METHODS)
    bar_w = max(16, 16 + (n_methods - 2) * 2)  # widen for many methods
    fig, ax = plt.subplots(figsize=(bar_w, 7))
    stats_lines = []

    final_rates = {}
    for mk in METHODS:
        seed_rates = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            summary = data[mk][seed].get("summary", {})
            rates = summary.get("per_task_achievement_rates", [[]])
            if rates and len(rates[0]) >= len(ACHIEVEMENT_NAMES):
                seed_rates.append(rates[0][:len(ACHIEVEMENT_NAMES)])
        if seed_rates:
            final_rates[mk] = np.array(seed_rates)

    n_ach = len(ACHIEVEMENT_NAMES)
    x = np.arange(n_ach)
    width = 0.8 / max(n_methods, 1)

    for i, mk in enumerate(_method_keys_sorted()):
        if mk not in final_rates:
            continue
        mat = final_rates[mk][:, :n_ach]
        mu = mat.mean(axis=0)
        std = mat.std(axis=0)
        offset = -0.4 + (i + 0.5) * width
        ec = "black" if n_methods > 3 else "white"
        al = max(0.6, 0.85 - 0.05 * n_methods)
        ax.bar(x + offset, mu, width, yerr=std, label=METHODS[mk],
               color=COLORS[mk], alpha=al, capsize=3, edgecolor=ec,
               linewidth=0.5)

    # Significance per achievement + stats -- pairwise
    stats_lines.append("  Per-achievement rates (mean +/- std) and pairwise significance:")
    header_parts = [f"{'Achievement':<24}"]
    for mk in final_rates:
        header_parts.append(f"{METHODS[mk]:<20}")
    stats_lines.append("    " + " ".join(header_parts))
    stats_lines.append("    " + "-" * (24 + 22 * len(final_rates)))

    method_keys_with_data = [mk for mk in _method_keys_sorted() if mk in final_rates]

    for j in range(len(ACHIEVEMENT_NAMES)):
        parts = [f"{ACHIEVEMENT_NAMES[j]:<24}"]
        for mk in method_keys_with_data:
            vals = final_rates[mk][:, j]
            parts.append(f"{fmt_dist_short(vals):<20}")
        stats_lines.append("    " + " ".join(parts))

    # Pairwise per-achievement significance at top of bars
    if len(method_keys_with_data) >= 2:
        for j in range(len(ACHIEVEMENT_NAMES)):
            pair_results = []
            for i_a in range(len(method_keys_with_data)):
                for i_b in range(i_a + 1, len(method_keys_with_data)):
                    mk_a = method_keys_with_data[i_a]
                    mk_b = method_keys_with_data[i_b]
                    a = final_rates[mk_a][:, j]
                    b = final_rates[mk_b][:, j]
                    _, p, sig = welch_t(a, b)
                    if not np.isnan(p) and p < SIGNIFICANCE_LEVEL:
                        star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                        pair_results.append(star)
            if pair_results:
                y_top = max(final_rates[mk][:, j].mean() + final_rates[mk][:, j].std()
                            for mk in method_keys_with_data) + 0.03
                ax.text(j, min(y_top, 1.05), pair_results[0], ha="center", fontsize=7, fontweight="bold")

    # Aggregate: total number of achievements with rate > thresholds
    for thresh in [0.01, 0.05, 0.10, 0.50]:
        stats_lines.append(f"\n  Achievements with rate > {thresh:.0%}:")
        for mk in method_keys_with_data:
            counts = (final_rates[mk] > thresh).sum(axis=1).astype(float)
            stats_lines.append(f"    {METHODS[mk]}: {fmt_dist_short(counts)}")
        # Pairwise tests
        if len(method_keys_with_data) >= 2:
            for i_a in range(len(method_keys_with_data)):
                for i_b in range(i_a + 1, len(method_keys_with_data)):
                    mk_a = method_keys_with_data[i_a]
                    mk_b = method_keys_with_data[i_b]
                    c_a = (final_rates[mk_a] > thresh).sum(axis=1).astype(float)
                    c_b = (final_rates[mk_b] > thresh).sum(axis=1).astype(float)
                    _, p, sig = welch_t(c_a, c_b)
                    stats_lines.append(f"    {METHODS[mk_a]} vs {METHODS[mk_b]}: p={p:.4f} ({sig})")

    report.add_section("Fig 07: Per-Achievement Rates", "\n".join(stats_lines))

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in ACHIEVEMENT_NAMES],
                        fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Achievement Rate", fontsize=12)
    ax.set_title("Per-Achievement Rates at Final Evaluation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
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
    n_methods = len(METHODS)
    bar_w = max(16, 16 + (n_methods - 2) * 2)
    fig, ax = plt.subplots(figsize=(bar_w, 7))
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
                if len(forg) >= len(ACHIEVEMENT_NAMES):
                    seed_forg.append(forg[:len(ACHIEVEMENT_NAMES)])
        if seed_forg:
            final_forg[mk] = np.array(seed_forg)

    n_ach = len(ACHIEVEMENT_NAMES)
    x = np.arange(n_ach)
    width = 0.8 / max(n_methods, 1)

    for i, mk in enumerate(_method_keys_sorted()):
        if mk not in final_forg:
            continue
        mat = final_forg[mk][:, :n_ach]
        mu = mat.mean(axis=0)
        std = mat.std(axis=0)
        offset = -0.4 + (i + 0.5) * width
        ec = "black" if n_methods > 3 else "white"
        al = max(0.6, 0.85 - 0.05 * n_methods)
        ax.bar(x + offset, mu, width, yerr=std, label=METHODS[mk],
               color=COLORS[mk], alpha=al, capsize=3, edgecolor=ec,
               linewidth=0.5)

    # Stats table
    method_keys_with_data = [mk for mk in _method_keys_sorted() if mk in final_forg]
    stats_lines.append("  Per-achievement forgetting (mean +/- std) and pairwise significance:")
    header_parts = [f"{'Achievement':<24}"]
    for mk in method_keys_with_data:
        header_parts.append(f"{METHODS[mk]:<20}")
    stats_lines.append("    " + " ".join(header_parts))
    stats_lines.append("    " + "-" * (24 + 22 * len(method_keys_with_data)))

    for j in range(len(ACHIEVEMENT_NAMES)):
        parts = [f"{ACHIEVEMENT_NAMES[j]:<24}"]
        for mk in method_keys_with_data:
            vals = final_forg[mk][:, j]
            parts.append(f"{fmt_dist_short(vals):<20}")
        stats_lines.append("    " + " ".join(parts))

    # Aggregate: mean forgetting across all achievements
    if len(method_keys_with_data) >= 2:
        agg_data = {}
        for mk in method_keys_with_data:
            agg_data[mk] = final_forg[mk].mean(axis=1)
        stats_lines.append(f"\n  Mean forgetting across all achievements:")
        for mk in method_keys_with_data:
            stats_lines.append(f"    {METHODS[mk]}: {fmt_dist(agg_data[mk])}")
        for i_a in range(len(method_keys_with_data)):
            for i_b in range(i_a + 1, len(method_keys_with_data)):
                mk_a, mk_b = method_keys_with_data[i_a], method_keys_with_data[i_b]
                _, p, sig = welch_t(agg_data[mk_a], agg_data[mk_b])
                stats_lines.append(f"    {METHODS[mk_a]} vs {METHODS[mk_b]}: p={p:.4f} ({sig})")

    # Top-3 most forgotten achievements per method
    for mk in method_keys_with_data:
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
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "08_per_achievement_forgetting.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 08_per_achievement_forgetting.png")


def figure_score_distribution(data):
    """Fig 9: Score distribution at final evaluation -- one subplot per method."""
    n_methods = len(METHODS)
    n_cols = min(n_methods, 4)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    bins_labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5+"]
    stats_lines = []

    all_dists = {}
    for idx, mk in enumerate(_method_keys_sorted()):
        ax = axes_flat[idx] if idx < len(axes_flat) else None
        if ax is None:
            continue
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

            stats_lines.append(f"  {METHODS[mk]} score distribution (mean +/- std across seeds):")
            for b in range(len(mu)):
                lbl = bins_labels[b] if b < len(bins_labels) else f"bin_{b}"
                stats_lines.append(f"    {lbl}: {mu[b]:.4f} +/- {std[b]:.4f}")

        ax.set_title(f"{METHODS[mk]}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Score Bucket", fontsize=11)
        ax.set_ylabel("Fraction of Episodes", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused axes
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Pairwise KL divergence
    dist_keys = list(all_dists.keys())
    if len(dist_keys) >= 2:
        eps = 1e-8
        for i_a in range(len(dist_keys)):
            for i_b in range(i_a + 1, len(dist_keys)):
                mk_a, mk_b = dist_keys[i_a], dist_keys[i_b]
                mu_a = all_dists[mk_a].mean(axis=0)
                mu_b = all_dists[mk_b].mean(axis=0)
                kl = np.sum(mu_a * np.log((mu_a + eps) / (mu_b + eps)))
                stats_lines.append(f"\n  KL divergence ({METHODS[mk_a]} || {METHODS[mk_b]}): {kl:.6f}")

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
    n_methods = len(METHODS)
    fig, axes = plt.subplots(n_methods, 1, figsize=(16, 6 * n_methods), squeeze=False)
    stats_lines = []

    for idx, mk in enumerate(_method_keys_sorted()):
        ax = axes[idx, 0]
        seed_data = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            online = data[mk][seed].get("online", [])
            steps = []
            rates_over_time = []
            for r in online:
                par = r.get("per_achievement_rates", [])
                if len(par) >= len(ACHIEVEMENT_NAMES):
                    steps.append(r["step"])
                    rates_over_time.append(par[:len(ACHIEVEMENT_NAMES)])
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
        stats_lines.append(f"  {METHODS[mk]} -- Step where achievement first exceeds threshold (mean across seeds):")
        stats_lines.append(f"    {'Achievement':<24} {'> 10%':<18} {'> 50%':<18} {'Final rate':<15}")
        stats_lines.append("    " + "-" * 65)
        for j in range(len(ACHIEVEMENT_NAMES)):
            final_rate = mean_rates[j, -1]
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
    """Fig 11: Summary statistics table with pairwise significance tests."""
    fig, ax = plt.subplots(figsize=(max(14, 4 * len(METHODS) + 6), 8))
    ax.axis("off")
    stats_lines = []

    metrics_to_compare = [
        ("mean_return", "Mean Return"),
        ("success_rate", "Success Rate"),
        ("per_task_lifetime_mean", "Lifetime Mean (task 0)"),
        ("per_task_aggregate_forgetting", "Aggregate Forgetting (task 0)"),
    ]

    method_keys = _method_keys_sorted()
    rows = []

    # Dynamic header
    header_parts = [f"{'Metric':<30}"]
    for mk in method_keys:
        header_parts.append(f"{METHODS[mk]:<28}")
    stats_lines.append("  Summary statistics from metrics_summary.json:")
    stats_lines.append("    " + " ".join(header_parts))
    stats_lines.append("    " + "-" * (30 + 30 * len(method_keys)))

    for key, label in metrics_to_compare:
        vals = {}
        for mk in method_keys:
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

        row = [label]
        line_parts = [f"{label:<30}"]
        for mk in method_keys:
            v = vals.get(mk, np.array([]))
            s = f"{v.mean():.4f} +/- {v.std():.4f}" if len(v) > 0 else "N/A"
            row.append(s)
            line_parts.append(f"{s:<28}")
        stats_lines.append("    " + " ".join(line_parts))

        # Best (smallest) pairwise p-value for the table cell
        best_p = None
        best_sig = "N/A"
        if len(method_keys) >= 2:
            for i_a in range(len(method_keys)):
                for i_b in range(i_a + 1, len(method_keys)):
                    mk_a, mk_b = method_keys[i_a], method_keys[i_b]
                    a = vals.get(mk_a, np.array([]))
                    b = vals.get(mk_b, np.array([]))
                    if len(a) >= 2 and len(b) >= 2:
                        _, p, sig = welch_t(a, b)
                        stats_lines.append(f"      {METHODS[mk_a]} vs {METHODS[mk_b]}: p={p:.4f} ({sig})")
                        if best_p is None or p < best_p:
                            best_p = p
                            best_sig = f"{p:.4f} ({sig})"
        row.append(best_sig if best_p is not None else "N/A")
        rows.append(row)

    # Per-seed raw values for transparency
    stats_lines.append("\n  Per-seed raw values:")
    for key, label in metrics_to_compare:
        stats_lines.append(f"    {label}:")
        for mk in method_keys:
            for seed in SEEDS:
                if seed not in data[mk]:
                    continue
                summary = data[mk][seed].get("summary", {})
                v = summary.get(key)
                if isinstance(v, list):
                    v = v[0] if v else None
                stats_lines.append(f"      {METHODS[mk]} seed={seed}: {v}")

    # Achievements unlocked
    unlock_data = {}
    for mk in method_keys:
        seed_totals = []
        for seed in SEEDS:
            if seed not in data[mk]:
                continue
            summary = data[mk][seed].get("summary", {})
            rates = summary.get("per_task_achievement_rates", [[]])
            if rates and rates[0]:
                seed_totals.append(sum(1 for r in rates[0] if r > 0.01))
        if seed_totals:
            unlock_data[mk] = np.array(seed_totals, dtype=float)

    row = ["Achievements Unlocked (>1%)"]
    for mk in method_keys:
        if mk in unlock_data:
            arr = unlock_data[mk]
            row.append(f"{arr.mean():.1f} +/- {arr.std():.1f}")
        else:
            row.append("N/A")
    row.append("")  # placeholder for p-value column
    rows.append(row)

    if len(unlock_data) >= 2:
        uk_keys = list(unlock_data.keys())
        for i_a in range(len(uk_keys)):
            for i_b in range(i_a + 1, len(uk_keys)):
                mk_a, mk_b = uk_keys[i_a], uk_keys[i_b]
                _, p, sig = welch_t(unlock_data[mk_a], unlock_data[mk_b])
                stats_lines.append(f"\n  Achievements unlocked {METHODS[mk_a]} vs {METHODS[mk_b]}: p={p:.4f} ({sig})")

    report.add_section("Fig 11: Summary Table", "\n".join(stats_lines))

    # Build table columns dynamically
    col_labels = ["Metric"] + [METHODS[mk] for mk in method_keys] + ["Pairwise p"]
    # Ensure all rows have the right length
    for r in rows:
        while len(r) < len(col_labels):
            r.append("")

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                      cellLoc="center",
                      colWidths=[0.20] + [0.18] * len(method_keys) + [0.14])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(len(rows)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor("#ecf0f1" if i % 2 == 0 else "white")

    ax.set_title(f"Summary Statistics ({len(SEEDS)} seeds, {len(METHODS)} methods)",
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
            if rates and len(rates[0]) >= len(ACHIEVEMENT_NAMES):
                all_final_rates.append(rates[0][:len(ACHIEVEMENT_NAMES)])

    if not all_final_rates:
        print("  Skipping top achievement curves (no data)")
        return

    mean_final = np.mean(all_final_rates, axis=0)
    top_indices = np.argsort(mean_final)[::-1][:8]

    fig, axes = plt.subplots(2, 4, figsize=_fig_size(18, 8))
    axes = axes.flatten()
    stats_lines = []

    stats_lines.append("  Top-8 achievements by mean final rate (across all methods):")
    header_parts = [f"{'Rank':<6}", f"{'Achievement':<24}"]
    for mk in METHODS:
        header_parts.append(f"{METHODS[mk]:<20}")
    stats_lines.append("    " + " ".join(header_parts))
    stats_lines.append("    " + "-" * (30 + 22 * len(METHODS)))

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
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk], method_key=mk)
                mats[mk] = mat
                ref_grid = grid

        if ref_grid is not None and len(mats) >= 2:
            add_pairwise_significance(ax, ref_grid, mats)

        format_ax(ax, ach_name.replace("_", " ").title(), ylabel="Rate")
        ax.set_ylim(-0.05, 1.05)

        # Stats for this achievement
        line_parts = [f"{plot_idx+1:<6}", f"{ach_name:<24}"]
        for mk in METHODS:
            f_vals = mats[mk][:, -1] if mk in mats else np.array([])
            line_parts.append(f"{fmt_dist_short(f_vals):<20}")
        stats_lines.append("    " + " ".join(line_parts))

    report.add_section("Fig 12: Top-8 Achievement Curves", "\n".join(stats_lines))

    fig.suptitle("Top-8 Achievement Rate Curves Over Training",
                  fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "12_top_achievement_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 12_top_achievement_curves.png")


def figure_max_return_and_depth(data):
    """Fig 13: Per-task max return and max achievement depth over time."""
    fig, axes = plt.subplots(1, 2, figsize=_fig_size(14, 5))
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
                    v = r.get(value_key)
                    if v is None:
                        continue
                    # Handle both scalar and list formats
                    if isinstance(v, list):
                        if len(v) > 0:
                            steps.append(r["step"])
                            vals.append(v[0])
                    elif isinstance(v, (int, float)) and not np.isnan(v):
                        steps.append(r["step"])
                        vals.append(v)
                if len(steps) > 1:
                    all_s.append(np.array(steps, dtype=np.float64))
                    all_v.append(np.array(vals, dtype=np.float64))
            grid, mat = interpolate_to_common_grid(all_s, all_v)
            if grid is not None and mat is not None:
                plot_mean_std(ax, grid, mat, COLORS[mk], METHODS[mk], method_key=mk)
                mats[mk] = mat
                ref_grid = grid
        format_ax(ax, title, ylabel=ylabel)

        # Stats
        stats_lines.append(f"  {title}:")
        for mk in METHODS:
            if mk in mats:
                final_vals = mats[mk][:, -1]
                stats_lines.append(f"    {METHODS[mk]} final: {fmt_dist(final_vals)}")
        # Pairwise tests
        mk_with_data = [mk for mk in METHODS if mk in mats]
        if len(mk_with_data) >= 2:
            for i_a in range(len(mk_with_data)):
                for i_b in range(i_a + 1, len(mk_with_data)):
                    mk_a, mk_b = mk_with_data[i_a], mk_with_data[i_b]
                    _, p, sig = welch_t(mats[mk_a][:, -1], mats[mk_b][:, -1])
                    stats_lines.append(f"    {METHODS[mk_a]} vs {METHODS[mk_b]}: p={p:.4f} ({sig})")

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
                    v = r.get("per_task_max_return")
                    if v is None:
                        continue
                    val = None
                    if isinstance(v, list) and len(v) > 0:
                        val = v[0]
                    elif isinstance(v, (int, float)):
                        val = v
                    if val is not None and not np.isnan(val) and val >= thresh:
                        steps_to.append(r["step"])
                        break
                else:
                    steps_to.append(np.nan)
            valid = np.array([s for s in steps_to if not np.isnan(s)])
            if len(valid) > 0:
                stats_lines.append(f"    {METHODS[mk]} -> max_return>={thresh}: {valid.mean():.0f} +/- {valid.std():.0f} ({len(valid)}/{len(steps_to)} seeds)")
            else:
                stats_lines.append(f"    {METHODS[mk]} -> max_return>={thresh}: not reached")

    report.add_section("Fig 13: Max Return & Depth", "\n".join(stats_lines))

    fig.suptitle("Performance Envelope", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_max_return_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 13_max_return_depth.png")


def figure_individual_seed_runs(data):
    """Fig 14: Individual seed trajectories (episode return) for transparency."""
    n_methods = len(METHODS)
    n_cols = min(n_methods, 4)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()
    seed_linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
    stats_lines = []

    stats_lines.append("  Per-seed final episode return (smoothed, last value):")
    stats_lines.append(f"    {'Method':<24} {'Seed':<8} {'Final Return':<16} {'Num Episodes':<14} {'Total Steps':<14}")
    stats_lines.append("    " + "-" * 76)

    for idx, mk in enumerate(_method_keys_sorted()):
        ax = axes_flat[idx] if idx < len(axes_flat) else None
        if ax is None:
            continue
        for s_idx, seed in enumerate(SEEDS):
            if seed not in data[mk]:
                continue
            records = data[mk][seed].get("metrics", [])
            s, v = extract_scalar_curve(records, "step", "episode/score", smooth=SMOOTH_WINDOW)
            ls = seed_linestyles[s_idx % len(seed_linestyles)]
            if len(s) > 0:
                ax.plot(s, v, linestyle=ls, color=COLORS[mk],
                        label=f"seed={seed}", alpha=0.8, linewidth=1.5)
                stats_lines.append(f"    {METHODS[mk]:<24} {seed:<8} {v[-1]:<16.4f} {len(records):<14} {s[-1]:<14.0f}")
        format_ax(ax, f"{METHODS[mk]}", ylabel="Episode Return")

    # Hide unused axes
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

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
    print("Craftax Experiment Comparative Analysis (Auto-Discovery)")
    print("=" * 70)
    print(f"\nScanning {LOGS_DIR.resolve()} for experiment folders ...")
    discover_methods(LOGS_DIR)

    if not METHODS:
        print("ERROR: No methods discovered. Check LOGS_DIR and folder naming.")
        print("  Expected pattern: craftax_dreamerv3-<method>-<seed>")
        return

    print(f"\nLoading data ...")
    data = load_all_runs()

    total_runs = sum(len(v) for v in data.values())
    expected = len(METHODS) * len(SEEDS)
    print(f"\nLoaded {total_runs} / {expected} expected runs.\n")
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

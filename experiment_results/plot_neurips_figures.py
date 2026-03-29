#!/usr/bin/env python3
"""
NeurIPS figure generation for Craftax continual-DreamerV3 ablation study.

Reads online_metrics.jsonl from all ablation runs, aggregates across 3 seeds,
and produces publication-quality figures with mean +/- std uncertainty bands.

Achievement names are loaded from metrics_summary.json (authoritative source)
to avoid the index mismatch in hardcoded fallback lists.

Usage:
    python plot_neurips_figures.py [--results_dir DIR] [--output_dir DIR]
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.ndimage import uniform_filter1d

# ── Configuration ──────────────────────────────────────────────────────────

SEEDS = [1, 4, 42]
INTERP_STEPS = 500          # interpolation grid size for learning curves
SMOOTH_WINDOW = 51          # rolling-window smoothing (odd number)
ALPHA_FILL = 0.20           # transparency for std band
FIG_DPI = 300
FIGSIZE_CURVE = (7, 4.2)
FIGSIZE_BAR = (10, 4.5)
FIGSIZE_HEATMAP = (12, 5)

# Experiment groups for comparison
EXPERIMENT_CONFIGS = {
    # exp_id -> (display_name, colour, linestyle)
    "A0_5050_baseline":  ("50:50 Baseline",       "#1f77b4", "-"),
    "A1_uniform_baseline": ("Uniform Baseline",    "#aec7e8", "--"),
    "A2_p2e":            ("Plan2Explore",           "#ff7f0e", "-"),
    "A3_intrinsic":      ("Intrinsic Only",         "#2ca02c", "-"),
    "A4_p2e_intrinsic":  ("P2E + Intrinsic",        "#98df8a", "--"),
    "B1_spatial_only":   ("Spatial Novelty Only",    "#d62728", "-."),
    "B2_craft_only":     ("Craft Novelty Only",      "#ff9896", "-."),
    "D1_nlr":            ("NLR (non-priv)",          "#9467bd", "-"),
    "D2_nlu":            ("NLU (non-priv)",          "#c5b0d5", "--"),
    "D3_nlr_priv":       ("NLR (privileged)",        "#8c564b", "-"),
    "D4_nlu_priv":       ("NLU (privileged)",        "#c49c94", "--"),
    "E1_nlr_intrinsic":  ("NLR + Intrinsic (ours)",  "#e377c2", "-"),
    "F1_mask_soft":      ("Soft Mask",               "#bcbd22", "-"),
    "F2_mask_hard":      ("Hard Mask",               "#17becf", "--"),
    # G-series: 10M-step extended runs
    "G1v2_mask_intr_nlu":  ("Mask+Intr+NLU (ours)",  "#e41a1c", "-"),
    "G2_baseline_10m":     ("Baseline 10M",           "#377eb8", "--"),
    "G3v3_mask_craft_nlu": ("Mask+Craft+NLU",         "#4daf4a", "-."),
}

# Achievement tier definitions (authoritative, matching train_craftax.py)
_TIER_0 = {
    'collect_wood', 'place_table', 'eat_cow', 'collect_sapling',
    'collect_drink', 'make_wood_pickaxe', 'make_wood_sword',
    'place_plant', 'eat_plant',
}
_TIER_1 = {
    'defeat_zombie', 'collect_stone', 'place_stone',
    'defeat_skeleton', 'make_stone_pickaxe', 'make_stone_sword',
    'wake_up', 'place_furnace', 'collect_coal',
    'eat_bat', 'eat_snail',
}
_TIER_2 = {
    'collect_iron', 'make_iron_pickaxe', 'make_iron_sword',
    'make_iron_armour', 'make_arrow', 'make_torch', 'place_torch',
    'make_diamond_sword', 'make_diamond_armour',
    'find_bow', 'fire_bow',
}
_TIER_3 = {
    'collect_diamond', 'make_diamond_pickaxe',
    'collect_sapphire', 'collect_ruby',
    'enter_gnomish_mines', 'enter_dungeon', 'enter_sewers',
    'enter_vault', 'enter_troll_mines',
    'defeat_gnome_warrior', 'defeat_gnome_archer',
    'defeat_orc_solider', 'defeat_orc_mage',
    'defeat_lizard', 'defeat_kobold',
    'learn_fireball', 'cast_fireball', 'learn_iceball', 'cast_iceball',
    'open_chest', 'drink_potion', 'enchant_sword', 'enchant_armour',
}
_TIER_4 = {
    'enter_fire_realm', 'enter_ice_realm', 'enter_graveyard',
    'defeat_troll', 'defeat_deep_thing', 'defeat_pigman',
    'defeat_fire_elemental', 'defeat_frost_troll', 'defeat_ice_elemental',
    'defeat_knight', 'defeat_archer',
    'damage_necromancer', 'defeat_necromancer',
}
_ALL_TIERS = [_TIER_0, _TIER_1, _TIER_2, _TIER_3, _TIER_4]
TIER_NAMES = ["Tier 0 (Basic)", "Tier 1 (Stone)", "Tier 2 (Iron)",
              "Tier 3 (Dungeon)", "Tier 4 (Endgame)"]


def get_tier(name):
    for i, tier_set in enumerate(_ALL_TIERS):
        if name in tier_set:
            return i
    return 3  # fallback


# ── Data loading ───────────────────────────────────────────────────────────

# Canonical achievement names: one-hot index i -> name.
# Source: craftax.craftax.constants.Achievement enum (67 achievements, values 0-66).
ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling",
    "collect_drink", "make_wood_pickaxe", "make_wood_sword", "place_plant",
    "defeat_zombie", "collect_stone", "place_stone", "eat_plant",
    "defeat_skeleton", "make_stone_pickaxe", "make_stone_sword", "wake_up",
    "place_furnace", "collect_coal", "collect_iron", "collect_diamond",
    "make_iron_pickaxe", "make_iron_sword", "make_arrow", "make_torch",
    "place_torch", "make_diamond_sword", "make_iron_armour", "make_diamond_armour",
    "enter_gnomish_mines", "enter_dungeon", "enter_sewers", "enter_vault",
    "enter_troll_mines", "enter_fire_realm", "enter_ice_realm", "enter_graveyard",
    "defeat_gnome_warrior", "defeat_gnome_archer", "defeat_orc_solider",
    "defeat_orc_mage", "defeat_lizard", "defeat_kobold", "defeat_troll",
    "defeat_deep_thing", "defeat_pigman", "defeat_fire_elemental",
    "defeat_frost_troll", "defeat_ice_elemental", "damage_necromancer",
    "defeat_necromancer", "eat_bat", "eat_snail", "find_bow", "fire_bow",
    "collect_sapphire", "learn_fireball", "cast_fireball", "learn_iceball",
    "cast_iceball", "collect_ruby", "make_diamond_pickaxe", "open_chest",
    "drink_potion", "enchant_sword", "enchant_armour", "defeat_knight",
    "defeat_archer",
]
assert len(ACHIEVEMENT_NAMES) == 67


def find_online_metrics(results_dir, exp_id, seed):
    """Locate online_metrics.jsonl for a given experiment config and seed.

    Supports two layouts:
      1. Flat files:  {results_dir}/{exp_id}_seed{seed}_online_metrics.jsonl
      2. Subdirectory: {results_dir}/{exp_id}_seed{seed}/online_metrics.jsonl
    """
    # Flat file layout (all_results/)
    flat = results_dir / f"{exp_id}_seed{seed}_online_metrics.jsonl"
    if flat.exists():
        return flat
    # Subdirectory layout (original)
    seed_dir = results_dir / f"{exp_id}_seed{seed}"
    if seed_dir.exists():
        jsonl_files = list(seed_dir.rglob("online_metrics.jsonl"))
        if jsonl_files:
            return jsonl_files[0]
    return None


def load_jsonl(path):
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_timeseries(records, key):
    """Extract (steps, values) arrays from a list of metric records."""
    steps, vals = [], []
    for r in records:
        if key in r and r[key] is not None:
            v = r[key]
            if isinstance(v, (int, float)) and np.isfinite(v):
                steps.append(r["step"])
                vals.append(v)
    return np.array(steps, dtype=np.float64), np.array(vals, dtype=np.float64)


def extract_achievement_timeseries(records):
    """Extract per-achievement rate timeseries: (steps, rates_matrix)."""
    steps = []
    all_rates = []
    for r in records:
        if "per_achievement_rates" in r and r["per_achievement_rates"]:
            rates = r["per_achievement_rates"]
            if len(rates) == 67:
                steps.append(r["step"])
                all_rates.append(rates)
    return np.array(steps, dtype=np.float64), np.array(all_rates, dtype=np.float64)


def interpolate_to_grid(steps, vals, grid):
    """Linearly interpolate (steps, vals) onto a uniform grid."""
    if len(steps) < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.interp(grid, steps, vals)


def smooth(vals, window):
    """Apply rolling mean smoothing, handling NaNs at edges."""
    if window <= 1 or len(vals) < window:
        return vals
    return uniform_filter1d(vals, size=window, mode="nearest")


# ── Aggregation across seeds ───────────────────────────────────────────────

def build_seed_curves(results_dir, exp_id, metric_key, grid):
    """Load metric from all seeds, interpolate to grid. Returns (N_seeds, N_grid)."""
    curves = []
    for seed in SEEDS:
        path = find_online_metrics(results_dir, exp_id, seed)
        if path is None:
            continue
        records = load_jsonl(path)
        steps, vals = extract_timeseries(records, metric_key)
        if len(steps) < 10:
            continue
        interp_vals = interpolate_to_grid(steps, vals, grid)
        curves.append(smooth(interp_vals, SMOOTH_WINDOW))
    return np.array(curves) if curves else None


def build_seed_achievement_curves(results_dir, exp_id, grid):
    """Load per-achievement rates from all seeds. Returns (N_seeds, N_grid, 67)."""
    all_seed_data = []
    for seed in SEEDS:
        path = find_online_metrics(results_dir, exp_id, seed)
        if path is None:
            continue
        records = load_jsonl(path)
        steps, rates_matrix = extract_achievement_timeseries(records)
        if len(steps) < 10:
            continue
        # Interpolate each achievement independently
        interp_rates = np.zeros((len(grid), 67))
        for j in range(67):
            interp_rates[:, j] = interpolate_to_grid(steps, rates_matrix[:, j], grid)
        all_seed_data.append(interp_rates)
    return np.array(all_seed_data) if all_seed_data else None


def get_final_achievement_rates(results_dir, exp_id):
    """Get final per-achievement rates from all seeds. Returns (N_seeds, 67)."""
    all_rates = []
    for seed in SEEDS:
        path = find_online_metrics(results_dir, exp_id, seed)
        if path is None:
            continue
        records = load_jsonl(path)
        if not records:
            continue
        last = records[-1]
        rates = last.get("per_achievement_rates", [])
        if len(rates) == 67:
            all_rates.append(rates)
    return np.array(all_rates) if all_rates else None


# ── Plotting helpers ───────────────────────────────────────────────────────

def _style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def plot_curve_with_uncertainty(ax, grid, curves, label, color, linestyle="-"):
    """Plot mean line with shaded std band."""
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    ax.plot(grid, mean, label=label, color=color, linestyle=linestyle, linewidth=1.5)
    ax.fill_between(grid, mean - std, mean + std, alpha=ALPHA_FILL, color=color)


def format_step_axis(ax):
    """Format x-axis to show steps in K or M units."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
    ))


# ── Figure 1: Mean Achievement Success Rate (learning curve) ──────────────

def figure_mean_achievement_rate(results_dir, output_dir, configs, grid):
    """Mean of per_achievement_rates over training, aggregated across seeds."""
    fig, ax = plt.subplots(figsize=FIGSIZE_CURVE)

    for exp_id in configs:
        if exp_id not in EXPERIMENT_CONFIGS:
            continue
        display, color, ls = EXPERIMENT_CONFIGS[exp_id]
        seed_data = build_seed_achievement_curves(results_dir, exp_id, grid)
        if seed_data is None:
            continue
        # seed_data: (seeds, grid, 67) -> mean across achievements -> (seeds, grid)
        mean_ach = np.mean(seed_data, axis=2)
        # Smooth each seed
        for i in range(mean_ach.shape[0]):
            mean_ach[i] = smooth(mean_ach[i], SMOOTH_WINDOW)
        plot_curve_with_uncertainty(ax, grid, mean_ach, display, color, ls)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Achievement Success Rate")
    ax.set_title("Mean Achievement Success Rate over Training")
    format_step_axis(ax)
    ax.legend(loc="best", ncol=2, framealpha=0.9)
    fig.savefig(output_dir / "fig1_mean_achievement_rate.pdf")
    fig.savefig(output_dir / "fig1_mean_achievement_rate.png")
    plt.close(fig)
    print("  [done] fig1_mean_achievement_rate")


# ── Figure 2: Number of Distinct Achievements Unlocked ─────────────────────

def figure_num_achievements_unlocked(results_dir, output_dir, configs, grid):
    """Number of achievements with rate > threshold, over training."""
    THRESHOLD = 0.01  # count achievement as 'unlocked' if rate > 1%
    fig, ax = plt.subplots(figsize=FIGSIZE_CURVE)

    for exp_id in configs:
        if exp_id not in EXPERIMENT_CONFIGS:
            continue
        display, color, ls = EXPERIMENT_CONFIGS[exp_id]
        seed_data = build_seed_achievement_curves(results_dir, exp_id, grid)
        if seed_data is None:
            continue
        # Count achievements above threshold: (seeds, grid)
        num_unlocked = np.sum(seed_data > THRESHOLD, axis=2).astype(np.float64)
        for i in range(num_unlocked.shape[0]):
            num_unlocked[i] = smooth(num_unlocked[i], SMOOTH_WINDOW)
        plot_curve_with_uncertainty(ax, grid, num_unlocked, display, color, ls)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("# Achievements Unlocked (rate > 1%)")
    ax.set_title("Distinct Achievements Discovered over Training")
    format_step_axis(ax)
    ax.legend(loc="best", ncol=2, framealpha=0.9)
    fig.savefig(output_dir / "fig2_num_achievements_unlocked.pdf")
    fig.savefig(output_dir / "fig2_num_achievements_unlocked.png")
    plt.close(fig)
    print("  [done] fig2_num_achievements_unlocked")


# ── Figure 3: Aggregate Forgetting ────────────────────────────────────────

def figure_aggregate_forgetting(results_dir, output_dir, configs, grid):
    """Aggregate forgetting over training."""
    fig, ax = plt.subplots(figsize=FIGSIZE_CURVE)

    for exp_id in configs:
        if exp_id not in EXPERIMENT_CONFIGS:
            continue
        display, color, ls = EXPERIMENT_CONFIGS[exp_id]
        curves = build_seed_curves(results_dir, exp_id, "aggregate_forgetting", grid)
        if curves is None:
            continue
        plot_curve_with_uncertainty(ax, grid, curves, display, color, ls)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Aggregate Forgetting")
    ax.set_title("Aggregate Forgetting over Training")
    format_step_axis(ax)
    ax.legend(loc="best", ncol=2, framealpha=0.9)
    fig.savefig(output_dir / "fig3_aggregate_forgetting.pdf")
    fig.savefig(output_dir / "fig3_aggregate_forgetting.png")
    plt.close(fig)
    print("  [done] fig3_aggregate_forgetting")


# ── Figure 4: Achievement Depth ───────────────────────────────────────────

def figure_achievement_depth(results_dir, output_dir, configs, grid):
    """Mean achievement depth over training."""
    fig, ax = plt.subplots(figsize=FIGSIZE_CURVE)

    for exp_id in configs:
        if exp_id not in EXPERIMENT_CONFIGS:
            continue
        display, color, ls = EXPERIMENT_CONFIGS[exp_id]
        curves = build_seed_curves(results_dir, exp_id, "depth_mean", grid)
        if curves is None:
            continue
        plot_curve_with_uncertainty(ax, grid, curves, display, color, ls)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Achievement Depth (Tier)")
    ax.set_title("Achievement Depth over Training")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(TIER_NAMES)
    format_step_axis(ax)
    ax.legend(loc="best", ncol=2, framealpha=0.9)
    fig.savefig(output_dir / "fig4_achievement_depth.pdf")
    fig.savefig(output_dir / "fig4_achievement_depth.png")
    plt.close(fig)
    print("  [done] fig4_achievement_depth")


# ── Figure 5: Per-Achievement Success Rates (bar chart, grouped by tier) ──

def figure_per_achievement_bars(results_dir, output_dir, configs, ach_names):
    """Grouped bar chart: final per-achievement success rates, coloured by tier."""
    # Select a subset of configs to avoid overcrowding
    show_configs = [c for c in configs if c in EXPERIMENT_CONFIGS]
    n_configs = len(show_configs)
    if n_configs == 0:
        return

    # Gather data: (config, 67) with mean across seeds
    config_means = {}
    config_stds = {}
    for exp_id in show_configs:
        rates = get_final_achievement_rates(results_dir, exp_id)
        if rates is not None:
            config_means[exp_id] = np.mean(rates, axis=0)
            config_stds[exp_id] = np.std(rates, axis=0)

    # Filter to only achievements that at least one config achieves > 0.5%
    active_mask = np.zeros(67, dtype=bool)
    for m in config_means.values():
        active_mask |= (m > 0.005)
    active_idx = np.where(active_mask)[0]

    if len(active_idx) == 0:
        return

    # Sort active achievements by tier, then by index
    tier_of = [get_tier(ach_names[i]) for i in active_idx]
    sort_order = np.argsort(tier_of)
    active_idx = active_idx[sort_order]

    n_ach = len(active_idx)
    bar_width = 0.8 / n_configs
    x = np.arange(n_ach)

    fig, ax = plt.subplots(figsize=(max(10, n_ach * 0.5), 5))

    for k, exp_id in enumerate(show_configs):
        if exp_id not in config_means:
            continue
        display, color, _ = EXPERIMENT_CONFIGS[exp_id]
        means = config_means[exp_id][active_idx]
        stds = config_stds[exp_id][active_idx]
        offset = (k - n_configs / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=display, color=color, alpha=0.85,
               error_kw={"linewidth": 0.7, "capsize": 1.5})

    # Label x-axis with achievement names
    labels = [ach_names[i] for i in active_idx]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Success Rate")
    ax.set_title("Per-Achievement Success Rates (Final Checkpoint)")

    # Add tier separators
    tiers_sorted = [get_tier(ach_names[i]) for i in active_idx]
    for j in range(1, len(tiers_sorted)):
        if tiers_sorted[j] != tiers_sorted[j - 1]:
            ax.axvline(x=j - 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.legend(loc="upper right", ncol=2, fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_per_achievement_bars.pdf")
    fig.savefig(output_dir / "fig5_per_achievement_bars.png")
    plt.close(fig)
    print("  [done] fig5_per_achievement_bars")


# ── Figure 6: Per-Tier Success Rate (learning curves, one subplot per tier)

def figure_per_tier_curves(results_dir, output_dir, configs, ach_names, grid):
    """One subplot per tier showing mean success rate for that tier's achievements."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)

    for tier_idx in range(5):
        ax = axes[tier_idx]
        tier_set = _ALL_TIERS[tier_idx]
        # Find which achievement indices belong to this tier
        tier_ach_idx = [i for i, name in enumerate(ach_names) if name in tier_set]

        if not tier_ach_idx:
            ax.set_title(TIER_NAMES[tier_idx])
            continue

        for exp_id in configs:
            if exp_id not in EXPERIMENT_CONFIGS:
                continue
            display, color, ls = EXPERIMENT_CONFIGS[exp_id]
            seed_data = build_seed_achievement_curves(results_dir, exp_id, grid)
            if seed_data is None:
                continue
            # Mean across achievements in this tier: (seeds, grid)
            tier_rates = np.mean(seed_data[:, :, tier_ach_idx], axis=2)
            for i in range(tier_rates.shape[0]):
                tier_rates[i] = smooth(tier_rates[i], SMOOTH_WINDOW)
            plot_curve_with_uncertainty(ax, grid, tier_rates, display, color, ls)

        ax.set_title(TIER_NAMES[tier_idx], fontsize=10)
        ax.set_xlabel("Steps")
        format_step_axis(ax)
        if tier_idx == 0:
            ax.set_ylabel("Mean Success Rate")

    # Single shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.12), framealpha=0.9)
    fig.suptitle("Per-Tier Achievement Success Rate over Training", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_per_tier_curves.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig6_per_tier_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig6_per_tier_curves")


# ── Figure 7: Achievement Heatmap (single best config) ────────────────────

def figure_achievement_heatmap(results_dir, output_dir, configs, ach_names, grid):
    """Heatmap of per-achievement success rates over training for top configs."""
    # Pick 3 configs: baseline, best non-intrinsic, best overall
    heatmap_configs = ["A0_5050_baseline", "D3_nlr_priv", "E1_nlr_intrinsic"]
    heatmap_configs = [c for c in heatmap_configs if c in configs]

    if not heatmap_configs:
        return

    n_plots = len(heatmap_configs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.0 * n_plots + 0.5))
    if n_plots == 1:
        axes = [axes]

    # Sort achievements by tier for better visualisation
    tier_of = [get_tier(name) for name in ach_names]
    sort_order = np.argsort(tier_of)

    for plot_idx, exp_id in enumerate(heatmap_configs):
        ax = axes[plot_idx]
        seed_data = build_seed_achievement_curves(results_dir, exp_id, grid)
        if seed_data is None:
            ax.set_title(f"{EXPERIMENT_CONFIGS[exp_id][0]} (no data)")
            continue

        # Mean across seeds: (grid, 67), reorder by tier
        mean_rates = np.mean(seed_data, axis=0)[:, sort_order]

        im = ax.imshow(mean_rates.T, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1.0, interpolation="nearest",
                       extent=[grid[0], grid[-1], len(ach_names) - 0.5, -0.5])

        # Y-axis labels (achievement names sorted by tier)
        sorted_names = [ach_names[i] for i in sort_order]
        ax.set_yticks(range(0, 67, 3))
        ax.set_yticklabels([sorted_names[i] for i in range(0, 67, 3)], fontsize=5)
        ax.set_title(EXPERIMENT_CONFIGS[exp_id][0], fontsize=10)
        format_step_axis(ax)

        # Add tier separators
        tiers_sorted = [tier_of[i] for i in sort_order]
        for j in range(1, 67):
            if tiers_sorted[j] != tiers_sorted[j - 1]:
                ax.axhline(y=j - 0.5, color="white", linewidth=1.5)

    fig.colorbar(im, ax=axes, label="Success Rate", shrink=0.6)
    fig.suptitle("Per-Achievement Success Rate Heatmap over Training", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "fig7_achievement_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig7_achievement_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig7_achievement_heatmap")


# ── Figure 8: Summary Bar Chart (final metrics) ──────────────────────────

def figure_summary_bars(results_dir, output_dir, configs):
    """Bar chart comparing final mean_ach_rate, num_unlocked, forgetting across all configs."""
    show_configs = [c for c in configs if c in EXPERIMENT_CONFIGS]
    if not show_configs:
        return

    metrics = {
        "Mean Ach. Rate": [],
        "# Unlocked": [],
        "Forgetting": [],
    }
    labels = []
    errors = {k: [] for k in metrics}

    for exp_id in show_configs:
        rates = get_final_achievement_rates(results_dir, exp_id)
        if rates is None:
            continue

        display, _, _ = EXPERIMENT_CONFIGS[exp_id]
        labels.append(display)

        # Mean achievement rate
        seed_means = np.mean(rates, axis=1)
        metrics["Mean Ach. Rate"].append(np.mean(seed_means))
        errors["Mean Ach. Rate"].append(np.std(seed_means))

        # Num achievements unlocked (> 1%)
        seed_counts = np.sum(rates > 0.01, axis=1).astype(float)
        metrics["# Unlocked"].append(np.mean(seed_counts))
        errors["# Unlocked"].append(np.std(seed_counts))

        # Forgetting (from last record)
        forg_vals = []
        for seed in SEEDS:
            path = find_online_metrics(results_dir, exp_id, seed)
            if path is None:
                continue
            records = load_jsonl(path)
            if records:
                f = records[-1].get("aggregate_forgetting", 0.0)
                if f is not None and np.isfinite(f):
                    forg_vals.append(f)
        metrics["Forgetting"].append(np.mean(forg_vals) if forg_vals else 0.0)
        errors["Forgetting"].append(np.std(forg_vals) if len(forg_vals) > 1 else 0.0)

    n = len(labels)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (title, vals) in zip(axes, metrics.items()):
        errs = errors[title]
        colors = [EXPERIMENT_CONFIGS[c][1] for c in show_configs[:n]]
        x = np.arange(n)
        ax.bar(x, vals, yerr=errs, color=colors, alpha=0.85,
               error_kw={"linewidth": 0.8, "capsize": 2})
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_title(title)

        # Normalise # Unlocked to /67
        if "Unlocked" in title:
            ax.set_ylabel("Count (out of 67)")
        elif "Forgetting" in title:
            ax.set_ylabel("Aggregate Forgetting")
            ax.invert_yaxis()
        else:
            ax.set_ylabel("Rate")

    fig.suptitle("Final Checkpoint Summary (mean +/- std across 3 seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "fig8_summary_bars.pdf")
    fig.savefig(output_dir / "fig8_summary_bars.png")
    plt.close(fig)
    print("  [done] fig8_summary_bars")


# ── Figure 9: Ablation group comparisons ──────────────────────────────────

def figure_ablation_groups(results_dir, output_dir, grid):
    """Separate learning curve panels for each ablation group (1M runs)."""
    groups = {
        "Group A: Core Methods": ["A0_5050_baseline", "A1_uniform_baseline",
                                   "A2_p2e", "A3_intrinsic", "A4_p2e_intrinsic"],
        "Group B: Intrinsic Ablation": ["A0_5050_baseline", "A3_intrinsic",
                                         "B1_spatial_only", "B2_craft_only"],
        "Group D: Replay Strategies": ["A0_5050_baseline", "D1_nlr", "D2_nlu",
                                        "D3_nlr_priv", "D4_nlu_priv"],
        "Group E+F: Combined + Masking": ["A0_5050_baseline", "E1_nlr_intrinsic",
                                           "F1_mask_soft", "F2_mask_hard"],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (group_name, group_configs) in zip(axes, groups.items()):
        for exp_id in group_configs:
            if exp_id not in EXPERIMENT_CONFIGS:
                continue
            display, color, ls = EXPERIMENT_CONFIGS[exp_id]
            seed_data = build_seed_achievement_curves(results_dir, exp_id, grid)
            if seed_data is None:
                continue
            mean_ach = np.mean(seed_data, axis=2)
            for i in range(mean_ach.shape[0]):
                mean_ach[i] = smooth(mean_ach[i], SMOOTH_WINDOW)
            plot_curve_with_uncertainty(ax, grid, mean_ach, display, color, ls)

        ax.set_title(group_name, fontsize=10)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Mean Ach. Rate")
        format_step_axis(ax)
        ax.legend(loc="best", fontsize=7, framealpha=0.9)

    fig.suptitle("Ablation Study: Mean Achievement Rate by Group", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "fig9_ablation_groups.pdf")
    fig.savefig(output_dir / "fig9_ablation_groups.png")
    plt.close(fig)
    print("  [done] fig9_ablation_groups")


# ── Figure 10: Extended 10M Runs ────────────────────────────────────────────

def figure_extended_10m(results_dir, output_dir, long_configs, long_grid):
    """Learning curves for 10M-step extended runs (G-series)."""
    if not long_configs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    metric_info = [
        ("Mean Achievement Rate", None),      # computed from per_achievement_rates
        ("# Achievements Unlocked", None),     # computed from per_achievement_rates
        ("Aggregate Forgetting", "aggregate_forgetting"),
    ]

    for ax, (title, metric_key) in zip(axes, metric_info):
        for exp_id in long_configs:
            if exp_id not in EXPERIMENT_CONFIGS:
                continue
            display, color, ls = EXPERIMENT_CONFIGS[exp_id]

            if metric_key is not None:
                curves = build_seed_curves(results_dir, exp_id, metric_key, long_grid)
                if curves is None:
                    continue
                plot_curve_with_uncertainty(ax, long_grid, curves, display, color, ls)
            else:
                seed_data = build_seed_achievement_curves(results_dir, exp_id, long_grid)
                if seed_data is None:
                    continue
                if "Unlocked" in title:
                    vals = np.sum(seed_data > 0.01, axis=2).astype(np.float64)
                else:
                    vals = np.mean(seed_data, axis=2)
                for i in range(vals.shape[0]):
                    vals[i] = smooth(vals[i], SMOOTH_WINDOW)
                plot_curve_with_uncertainty(ax, long_grid, vals, display, color, ls)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training Steps")
        format_step_axis(ax)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.suptitle("Extended Training (10M Steps): G-Series Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "fig10_extended_10m.pdf")
    fig.savefig(output_dir / "fig10_extended_10m.png")
    plt.close(fig)
    print("  [done] fig10_extended_10m")


# ── Figure 11: 10M Per-Tier Curves ──────────────────────────────────────────

def figure_extended_per_tier(results_dir, output_dir, long_configs, ach_names, long_grid):
    """Per-tier success rate breakdown for 10M-step extended runs."""
    if not long_configs:
        return

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)

    for tier_idx in range(5):
        ax = axes[tier_idx]
        tier_set = _ALL_TIERS[tier_idx]
        tier_ach_idx = [i for i, name in enumerate(ach_names) if name in tier_set]

        if not tier_ach_idx:
            ax.set_title(TIER_NAMES[tier_idx])
            continue

        for exp_id in long_configs:
            if exp_id not in EXPERIMENT_CONFIGS:
                continue
            display, color, ls = EXPERIMENT_CONFIGS[exp_id]
            seed_data = build_seed_achievement_curves(results_dir, exp_id, long_grid)
            if seed_data is None:
                continue
            tier_rates = np.mean(seed_data[:, :, tier_ach_idx], axis=2)
            for i in range(tier_rates.shape[0]):
                tier_rates[i] = smooth(tier_rates[i], SMOOTH_WINDOW)
            plot_curve_with_uncertainty(ax, long_grid, tier_rates, display, color, ls)

        ax.set_title(TIER_NAMES[tier_idx], fontsize=10)
        ax.set_xlabel("Steps")
        format_step_axis(ax)
        if tier_idx == 0:
            ax.set_ylabel("Mean Success Rate")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(long_configs), fontsize=8,
               bbox_to_anchor=(0.5, -0.12), framealpha=0.9)
    fig.suptitle("Extended Training (10M): Per-Tier Achievement Success Rate", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig11_extended_per_tier.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig11_extended_per_tier.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig11_extended_per_tier")


# ── Figure 12: 10M Achievement Heatmap ──────────────────────────────────────

def figure_extended_heatmap(results_dir, output_dir, long_configs, ach_names, long_grid):
    """Achievement heatmap for 10M-step runs."""
    heatmap_configs = [c for c in long_configs if c in EXPERIMENT_CONFIGS]
    if not heatmap_configs:
        return

    n_plots = len(heatmap_configs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.0 * n_plots + 0.5))
    if n_plots == 1:
        axes = [axes]

    tier_of = [get_tier(name) for name in ach_names]
    sort_order = np.argsort(tier_of)

    for plot_idx, exp_id in enumerate(heatmap_configs):
        ax = axes[plot_idx]
        seed_data = build_seed_achievement_curves(results_dir, exp_id, long_grid)
        if seed_data is None:
            ax.set_title(f"{EXPERIMENT_CONFIGS[exp_id][0]} (no data)")
            continue

        mean_rates = np.mean(seed_data, axis=0)[:, sort_order]
        im = ax.imshow(mean_rates.T, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1.0, interpolation="nearest",
                       extent=[long_grid[0], long_grid[-1], len(ach_names) - 0.5, -0.5])

        sorted_names = [ach_names[i] for i in sort_order]
        ax.set_yticks(range(0, 67, 3))
        ax.set_yticklabels([sorted_names[i] for i in range(0, 67, 3)], fontsize=5)
        ax.set_title(EXPERIMENT_CONFIGS[exp_id][0], fontsize=10)
        format_step_axis(ax)

        tiers_sorted = [tier_of[i] for i in sort_order]
        for j in range(1, 67):
            if tiers_sorted[j] != tiers_sorted[j - 1]:
                ax.axhline(y=j - 0.5, color="white", linewidth=1.5)

    fig.colorbar(im, ax=axes, label="Success Rate", shrink=0.6)
    fig.suptitle("Extended Training (10M): Per-Achievement Heatmap", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "fig12_extended_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "fig12_extended_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  [done] fig12_extended_heatmap")


# ── Main ──────────────────────────────────────────────────────────────────

def _get_max_step(results_dir, exp_id):
    """Get the max step for an experiment (from any seed)."""
    max_step = 0
    for seed in SEEDS:
        path = find_online_metrics(results_dir, exp_id, seed)
        if path is None:
            continue
        with open(path, "rb") as f:
            # Seek to end and read last line efficiently
            f.seek(0, 2)
            end = f.tell()
            pos = max(0, end - 4096)
            f.seek(pos)
            lines = f.read().decode("utf-8", errors="replace").strip().split("\n")
            last = json.loads(lines[-1])
            max_step = max(max_step, last["step"])
    return max_step


# 10M-step experiment IDs (G-series)
_LONG_RUN_IDS = {"G1v2_mask_intr_nlu", "G2_baseline_10m", "G3v3_mask_craft_nlu"}


def main():
    parser = argparse.ArgumentParser(description="Generate NeurIPS figures")
    parser.add_argument("--results_dir", type=str,
                        default="all_results",
                        help="Path to results directory")
    parser.add_argument("--output_dir", type=str,
                        default="all_results/figures",
                        help="Path to output directory for figures")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _style()

    ach_names = ACHIEVEMENT_NAMES
    print(f"  {len(ach_names)} achievements (hardcoded from Craftax enum)")

    # Discover available configs (supports both flat files and subdirectories)
    available = set()
    for p in sorted(results_dir.iterdir()):
        name = p.name
        if p.is_file() and name.endswith("_online_metrics.jsonl"):
            base = name.replace("_online_metrics.jsonl", "")
            for seed in SEEDS:
                suffix = f"_seed{seed}"
                if base.endswith(suffix):
                    available.add(base[:-len(suffix)])
        elif p.is_dir() and name != "figures":
            for seed in SEEDS:
                suffix = f"_seed{seed}"
                if name.endswith(suffix):
                    available.add(name[:-len(suffix)])

    all_configs = [c for c in EXPERIMENT_CONFIGS if c in available]
    short_configs = [c for c in all_configs if c not in _LONG_RUN_IDS]
    long_configs = [c for c in all_configs if c in _LONG_RUN_IDS]

    print(f"  Found {len(short_configs)} short-run (1M) configs: {short_configs}")
    print(f"  Found {len(long_configs)} long-run (10M) configs: {long_configs}")

    # Build grids for short and long runs
    short_max = max((_get_max_step(results_dir, c) for c in short_configs), default=0)
    short_grid = np.linspace(0, short_max, INTERP_STEPS)
    print(f"  Short grid: 0 -> {short_max}")

    if long_configs:
        long_max = max(_get_max_step(results_dir, c) for c in long_configs)
        long_grid = np.linspace(0, long_max, INTERP_STEPS * 2)  # finer grid for 10x longer runs
        print(f"  Long grid:  0 -> {long_max}")

    # ── Short-run figures (1M ablation study) ──
    print("\nGenerating 1M ablation figures...")
    figure_mean_achievement_rate(results_dir, output_dir, short_configs, short_grid)
    figure_num_achievements_unlocked(results_dir, output_dir, short_configs, short_grid)
    figure_aggregate_forgetting(results_dir, output_dir, short_configs, short_grid)
    figure_achievement_depth(results_dir, output_dir, short_configs, short_grid)
    figure_per_achievement_bars(results_dir, output_dir, short_configs, ach_names)
    figure_per_tier_curves(results_dir, output_dir, short_configs, ach_names, short_grid)
    figure_achievement_heatmap(results_dir, output_dir, short_configs, ach_names, short_grid)
    figure_summary_bars(results_dir, output_dir, short_configs)
    figure_ablation_groups(results_dir, output_dir, short_grid)

    # ── Long-run figures (10M extended training) ──
    if long_configs:
        print("\nGenerating 10M extended-training figures...")
        figure_extended_10m(results_dir, output_dir, long_configs, long_grid)
        figure_extended_per_tier(results_dir, output_dir, long_configs, ach_names, long_grid)
        figure_extended_heatmap(results_dir, output_dir, long_configs, ach_names, long_grid)
        figure_per_achievement_bars(results_dir, output_dir, long_configs, ach_names)
        # Rename the long-run bar chart
        for ext in ("pdf", "png"):
            src = output_dir / f"fig5_per_achievement_bars.{ext}"
            dst = output_dir / f"fig13_extended_per_achievement_bars.{ext}"
            if src.exists():
                src.rename(dst)
        print("  [done] fig13_extended_per_achievement_bars")

    print(f"\nAll figures saved to: {output_dir}/")


if __name__ == "__main__":
    main()

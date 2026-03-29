#!/usr/bin/env python3
"""
Comprehensive analysis of all Craftax experiment results from all_results/.
Outputs structured tables for NeurIPS paper metrics.

Usage:  python analyze_all_results.py
"""

import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

# ============================================================
# Constants
# ============================================================

RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "all_results"
MAX_SCORE = 226.0

ACHIEVEMENT_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling",
    "collect_drink", "make_wood_pickaxe", "make_wood_sword",
    "place_plant", "defeat_zombie", "collect_stone",
    "place_stone", "eat_plant", "defeat_skeleton", "make_stone_pickaxe",
    "make_stone_sword", "wake_up", "place_furnace", "collect_coal",
    "collect_iron", "collect_diamond", "make_iron_pickaxe", "make_iron_sword",
    "make_arrow", "make_torch", "place_torch", "make_diamond_sword",
    "make_iron_armour", "make_diamond_armour", "enter_gnomish_mines",
    "enter_dungeon", "enter_sewers", "enter_vault", "enter_troll_mines",
    "enter_fire_realm", "enter_ice_realm", "enter_graveyard",
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
NUM_ACH = len(ACHIEVEMENT_NAMES)

# Tier assignments
_TIER_0 = {'collect_wood', 'place_table', 'eat_cow', 'collect_sapling',
            'collect_drink', 'make_wood_pickaxe', 'make_wood_sword',
            'place_plant', 'eat_plant'}
_TIER_1 = {'defeat_zombie', 'collect_stone', 'place_stone',
            'defeat_skeleton', 'make_stone_pickaxe', 'make_stone_sword',
            'wake_up', 'place_furnace', 'collect_coal',
            'eat_bat', 'eat_snail'}
_TIER_2 = {'collect_iron', 'make_iron_pickaxe', 'make_iron_sword',
            'make_iron_armour', 'make_arrow', 'make_torch', 'place_torch',
            'make_diamond_sword', 'make_diamond_armour',
            'find_bow', 'fire_bow'}
_TIER_3 = {'collect_diamond', 'make_diamond_pickaxe',
            'collect_sapphire', 'collect_ruby',
            'enter_gnomish_mines', 'enter_dungeon', 'enter_sewers',
            'enter_vault', 'enter_troll_mines',
            'defeat_gnome_warrior', 'defeat_gnome_archer',
            'defeat_orc_solider', 'defeat_orc_mage',
            'defeat_lizard', 'defeat_kobold',
            'learn_fireball', 'cast_fireball', 'learn_iceball', 'cast_iceball',
            'open_chest', 'drink_potion', 'enchant_sword', 'enchant_armour'}
_TIER_4 = {'enter_fire_realm', 'enter_ice_realm', 'enter_graveyard',
            'defeat_troll', 'defeat_deep_thing', 'defeat_pigman',
            'defeat_fire_elemental', 'defeat_frost_troll', 'defeat_ice_elemental',
            'defeat_knight', 'defeat_archer',
            'damage_necromancer', 'defeat_necromancer'}

TIER_MAP = {}
TIER_INDICES = {t: [] for t in range(5)}  # tier -> list of achievement indices
for i, name in enumerate(ACHIEVEMENT_NAMES):
    if name in _TIER_0:
        TIER_MAP[name] = 0
    elif name in _TIER_1:
        TIER_MAP[name] = 1
    elif name in _TIER_2:
        TIER_MAP[name] = 2
    elif name in _TIER_3:
        TIER_MAP[name] = 3
    elif name in _TIER_4:
        TIER_MAP[name] = 4
    else:
        TIER_MAP[name] = 3
    TIER_INDICES[TIER_MAP[name]].append(i)

# Experiments with intrinsic rewards contaminating return_mean
INTRINSIC_EXPS = {'A3_intrinsic', 'A4_p2e_intrinsic', 'E1_nlr_intrinsic',
                  'G1v2_mask_intr_nlu'}

# Number of final records to average for "final" metrics
FINAL_WINDOW = 50

# ============================================================
# Data Loading
# ============================================================


def discover_experiments():
    """Discover experiment groups and their seed files from all_results/."""
    pattern = re.compile(r'^(.+)_seed(\d+)_online_metrics\.jsonl$')
    groups = defaultdict(dict)  # group_name -> {seed_int: path}
    for f in sorted(RESULTS_DIR.glob("*.jsonl")):
        m = pattern.match(f.name)
        if m:
            group, seed = m.group(1), int(m.group(2))
            groups[group][seed] = f
    return dict(groups)


def load_jsonl(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify_duration(group_name, records):
    """Classify as '1m' or '10m' based on max step."""
    if not records:
        return '1m'
    max_step = records[-1].get('step', 0)
    return '10m' if max_step > 2_000_000 else '1m'


# ============================================================
# Per-seed analysis helpers
# ============================================================


def compute_mean_ach_rate(per_achievement_rates):
    """Mean across 67 achievement success rates."""
    return np.mean(per_achievement_rates)


def count_ever_achieved(records):
    """Count distinct achievements ever unlocked (any record has rate > 0)."""
    ever = np.zeros(NUM_ACH, dtype=bool)
    for r in records:
        rates = r.get('per_achievement_rates', [])
        for i, v in enumerate(rates):
            if v > 0:
                ever[i] = True
    return int(ever.sum())


def count_ever_achieved_from_booleans(records):
    """Count distinct achievements ever True in achievement boolean vectors."""
    ever = np.zeros(NUM_ACH, dtype=bool)
    for r in records:
        ach = r.get('achievements', [])
        for i, v in enumerate(ach):
            if v:
                ever[i] = True
    return int(ever.sum()), ever


def get_per_tier_rates(per_achievement_rates):
    """Compute mean achievement rate per tier (0-4)."""
    rates = np.array(per_achievement_rates)
    tier_rates = {}
    for t in range(5):
        idxs = TIER_INDICES[t]
        if idxs:
            tier_rates[t] = float(np.mean(rates[idxs]))
        else:
            tier_rates[t] = 0.0
    return tier_rates


def analyze_seed(records, is_intrinsic):
    """Analyze a single seed's full trajectory. Returns dict of metrics."""
    if not records:
        return None

    steps = np.array([r['step'] for r in records])
    n = len(records)

    # --- Collect time series ---
    mean_ach_rates = []
    per_ach_rates_ts = []  # (n, 67)
    scores = []
    return_means = []
    agg_forgettings = []
    frontier_rates = []
    personal_best_depths = []
    score_dists = []
    td_error_means = []
    td_error_maxs = []
    mask_infeasible = []
    mask_blocked = []
    per_ach_forgetting_ts = []

    for r in records:
        par = r.get('per_achievement_rates', [0.0] * NUM_ACH)
        mean_ach_rates.append(np.mean(par))
        per_ach_rates_ts.append(par)
        scores.append(r.get('score', 0.0))
        return_means.append(r.get('return_mean', 0.0))
        agg_forgettings.append(r.get('aggregate_forgetting', 0.0))
        frontier_rates.append(r.get('frontier_rate', 0.0))
        personal_best_depths.append(r.get('personal_best_depth', 0))
        score_dists.append(r.get('score_distribution', [0]*6))
        td_error_means.append(r.get('td_error_mean', 0.0))
        td_error_maxs.append(r.get('td_error_max', 0.0))
        per_ach_forgetting_ts.append(r.get('per_achievement_forgetting', [0.0]*NUM_ACH))
        mask_infeasible.append(r.get('mask_infeasible_frac', None))
        mask_blocked.append(r.get('mask_blocked_frac', None))

    per_ach_rates_ts = np.array(per_ach_rates_ts)  # (n, 67)
    per_ach_forgetting_ts = np.array(per_ach_forgetting_ts)  # (n, 67)

    # --- Final window metrics (last FINAL_WINDOW records) ---
    fw = min(FINAL_WINDOW, n)
    final_par = per_ach_rates_ts[-fw:]  # (fw, 67)
    final_mean_ach_rate = float(np.mean(final_par))
    final_per_ach_rate = np.mean(final_par, axis=0)  # (67,)
    final_score = float(np.mean(scores[-fw:]))
    final_return = float(np.mean(return_means[-fw:]))
    final_agg_forgetting = float(np.mean(agg_forgettings[-fw:]))
    final_frontier_rate = float(np.mean(frontier_rates[-fw:]))
    final_depth = int(max(personal_best_depths[-fw:]))
    final_per_ach_forgetting = np.mean(per_ach_forgetting_ts[-fw:], axis=0)

    # Number of achievements ever unlocked (>0 rate at any point)
    num_ever_achieved = int(np.sum(np.max(per_ach_rates_ts, axis=0) > 0))

    # Peak achievement rate across all time
    peak_per_ach_rate = np.max(per_ach_rates_ts, axis=0)  # (67,)
    # Current (final window)
    curr_per_ach_rate = final_per_ach_rate  # (67,)
    # Per-achievement forgetting = peak - current (clamped >= 0)
    computed_forgetting = np.maximum(peak_per_ach_rate - curr_per_ach_rate, 0)

    # Per-tier final rates
    final_tier_rates = get_per_tier_rates(final_per_ach_rate)
    # Per-tier peak rates
    peak_tier_rates = get_per_tier_rates(peak_per_ach_rate)

    # Max achievement depth ever
    max_depth_ever = int(max(personal_best_depths))

    # Score as % of max (only meaningful for extrinsic)
    final_score_pct = final_score / MAX_SCORE * 100

    # TD-error final
    final_td_mean = float(np.mean(td_error_means[-fw:]))
    final_td_max = float(np.mean(td_error_maxs[-fw:]))

    # Mask metrics final (may be None)
    has_mask = mask_infeasible[-1] is not None
    final_mask_infeasible = float(np.mean([x for x in mask_infeasible[-fw:] if x is not None])) if has_mask else None
    final_mask_blocked = float(np.mean([x for x in mask_blocked[-fw:] if x is not None])) if has_mask else None

    # Score distribution final
    final_score_dist = np.mean(score_dists[-fw:], axis=0).tolist()

    # Number of achievements with nonzero rate at the end
    num_active_final = int(np.sum(final_per_ach_rate > 0.01))

    result = {
        'steps': steps,
        'n_records': n,
        'max_step': int(steps[-1]),
        # Final window metrics
        'final_mean_ach_rate': final_mean_ach_rate,
        'final_per_ach_rate': final_per_ach_rate,
        'final_score': final_score,
        'final_return': final_return,
        'final_score_pct': final_score_pct,
        'final_agg_forgetting': final_agg_forgetting,
        'final_frontier_rate': final_frontier_rate,
        'final_depth': final_depth,
        'final_td_mean': final_td_mean,
        'final_td_max': final_td_max,
        'final_mask_infeasible': final_mask_infeasible,
        'final_mask_blocked': final_mask_blocked,
        'final_score_dist': final_score_dist,
        'num_active_final': num_active_final,
        # Lifetime metrics
        'num_ever_achieved': num_ever_achieved,
        'max_depth_ever': max_depth_ever,
        'peak_per_ach_rate': peak_per_ach_rate,
        'computed_forgetting': computed_forgetting,
        'computed_agg_forgetting': float(np.mean(computed_forgetting)),
        # Per-tier
        'final_tier_rates': final_tier_rates,
        'peak_tier_rates': peak_tier_rates,
        # Time series (for trajectory analysis)
        'mean_ach_rate_ts': np.array(mean_ach_rates),
        'per_ach_rates_ts': per_ach_rates_ts,
        'score_ts': np.array(scores),
        'return_mean_ts': np.array(return_means),
        'agg_forgetting_ts': np.array(agg_forgettings),
        'frontier_rate_ts': np.array(frontier_rates),
        'depth_ts': np.array(personal_best_depths),
        'per_ach_forgetting_ts': per_ach_forgetting_ts,
    }
    return result


# ============================================================
# Reporting
# ============================================================

def print_separator(char='=', width=120):
    print(char * width)


def print_header(title, width=120):
    print()
    print_separator('=', width)
    print(f"  {title}")
    print_separator('=', width)


def report_final_table(experiments, duration_filter):
    """Print final metrics summary table for all experiments of given duration."""
    tag = duration_filter.upper()
    print_header(f"FINAL PERFORMANCE SUMMARY — {tag} EXPERIMENTS (last {FINAL_WINDOW} records avg)")

    # Collect data: group -> [seed_results]
    rows = []
    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue
        is_intr = group_name in INTRINSIC_EXPS

        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())
        n_seeds = len(seeds)

        # Aggregate across seeds
        mar = [seed_data[s]['final_mean_ach_rate'] for s in seeds]
        nea = [seed_data[s]['num_ever_achieved'] for s in seeds]
        naf = [seed_data[s]['num_active_final'] for s in seeds]
        depth = [seed_data[s]['max_depth_ever'] for s in seeds]
        frt = [seed_data[s]['final_frontier_rate'] for s in seeds]
        afg = [seed_data[s]['computed_agg_forgetting'] for s in seeds]
        sc = [seed_data[s]['final_score'] for s in seeds]
        sp = [seed_data[s]['final_score_pct'] for s in seeds]
        ret = [seed_data[s]['final_return'] for s in seeds]

        rows.append({
            'name': group_name,
            'is_intr': is_intr,
            'n_seeds': n_seeds,
            'mean_ach_rate': (np.mean(mar), np.std(mar)),
            'num_ever_achieved': (np.mean(nea), np.std(nea)),
            'num_active_final': (np.mean(naf), np.std(naf)),
            'max_depth': (np.mean(depth), np.std(depth)),
            'frontier_rate': (np.mean(frt), np.std(frt)),
            'agg_forgetting': (np.mean(afg), np.std(afg)),
            'score': (np.mean(sc), np.std(sc)),
            'score_pct': (np.mean(sp), np.std(sp)),
            'return': (np.mean(ret), np.std(ret)),
        })

    # Print table
    hdr = (f"{'Experiment':<28} {'MeanAchRate':>14} {'#EverAch':>11} {'#Active':>10} "
           f"{'MaxDepth':>10} {'FrontierR':>11} {'AggForget':>12} "
           f"{'Score%':>10} {'Return':>14}")
    print(hdr)
    print('-' * len(hdr))

    for r in rows:
        mar_str = f"{r['mean_ach_rate'][0]:.4f}±{r['mean_ach_rate'][1]:.4f}"
        nea_str = f"{r['num_ever_achieved'][0]:.1f}±{r['num_ever_achieved'][1]:.1f}"
        naf_str = f"{r['num_active_final'][0]:.1f}±{r['num_active_final'][1]:.1f}"
        dep_str = f"{r['max_depth'][0]:.1f}±{r['max_depth'][1]:.1f}"
        frt_str = f"{r['frontier_rate'][0]:.4f}±{r['frontier_rate'][1]:.4f}"
        afg_str = f"{r['agg_forgetting'][0]:.4f}±{r['agg_forgetting'][1]:.4f}"
        if r['is_intr']:
            scp_str = "N/A(intr)"
            ret_str = "N/A(intr)"
        else:
            scp_str = f"{r['score_pct'][0]:.2f}±{r['score_pct'][1]:.2f}"
            ret_str = f"{r['return'][0]:.3f}±{r['return'][1]:.3f}"
        print(f"{r['name']:<28} {mar_str:>14} {nea_str:>11} {naf_str:>10} "
              f"{dep_str:>10} {frt_str:>11} {afg_str:>12} "
              f"{scp_str:>10} {ret_str:>14}")


def report_per_seed_detail(experiments, duration_filter):
    """Print per-seed detail for each experiment."""
    tag = duration_filter.upper()
    print_header(f"PER-SEED DETAIL — {tag} EXPERIMENTS")

    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue
        is_intr = group_name in INTRINSIC_EXPS
        print(f"\n  --- {group_name} {'[INTRINSIC]' if is_intr else ''} ---")

        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        for s in sorted(seed_data.keys()):
            d = seed_data[s]
            intr_tag = ""
            score_str = f"Score={d['final_score']:.3f} ({d['final_score_pct']:.2f}%)"
            ret_str = f"Return={d['final_return']:.3f}"
            if is_intr:
                score_str += " [contaminated]"
                ret_str += " [contaminated]"
            print(f"  seed={s:>2} | MeanAchRate={d['final_mean_ach_rate']:.4f} | "
                  f"#EverAch={d['num_ever_achieved']:>2} | #Active={d['num_active_final']:>2} | "
                  f"Depth={d['max_depth_ever']} | FrontierR={d['final_frontier_rate']:.3f} | "
                  f"AggForget={d['computed_agg_forgetting']:.4f} | "
                  f"{score_str} | {ret_str} | "
                  f"Steps={d['max_step']}")


def report_per_achievement_final(experiments, duration_filter):
    """Print per-achievement success rates at the final window, averaged across seeds."""
    tag = duration_filter.upper()
    print_header(f"PER-ACHIEVEMENT FINAL RATES — {tag} EXPERIMENTS (top achievements)")

    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue

        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())

        # Average per_ach_rate across seeds
        all_rates = np.stack([seed_data[s]['final_per_ach_rate'] for s in seeds])
        mean_rates = np.mean(all_rates, axis=0)
        std_rates = np.std(all_rates, axis=0)

        # All peak rates across seeds
        all_peak = np.stack([seed_data[s]['peak_per_ach_rate'] for s in seeds])
        mean_peak = np.mean(all_peak, axis=0)

        # Filter to achievements with >0.5% mean rate at any point (peak)
        active_idxs = np.where(mean_peak > 0.005)[0]
        if len(active_idxs) == 0:
            continue

        # Sort by mean final rate descending
        order = active_idxs[np.argsort(-mean_rates[active_idxs])]

        print(f"\n  {group_name} ({len(seeds)} seeds)")
        print(f"  {'Achievement':<30} {'Tier':>4} {'FinalRate':>12} {'PeakRate':>12} {'Forgetting':>12}")
        print(f"  {'-'*74}")
        for idx in order:
            name = ACHIEVEMENT_NAMES[idx]
            tier = TIER_MAP[name]
            fr = f"{mean_rates[idx]:.4f}±{std_rates[idx]:.4f}"
            pr = f"{mean_peak[idx]:.4f}"

            all_forget = np.stack([seed_data[s]['computed_forgetting'] for s in seeds])
            mean_forget = np.mean(all_forget, axis=0)
            fg = f"{mean_forget[idx]:.4f}"

            print(f"  {name:<30} T{tier:>2}  {fr:>12} {pr:>12} {fg:>12}")


def report_per_tier_summary(experiments, duration_filter):
    """Print per-tier achievement rate summary."""
    tag = duration_filter.upper()
    print_header(f"PER-TIER ACHIEVEMENT RATES — {tag} EXPERIMENTS")

    hdr = f"{'Experiment':<28} {'T0-Basic':>12} {'T1-Stone':>12} {'T2-Iron':>12} {'T3-Dungeon':>12} {'T4-Endgame':>12}"
    print(hdr)
    print('-' * len(hdr))

    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue
        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())

        tier_vals = {t: [] for t in range(5)}
        for s in seeds:
            for t in range(5):
                tier_vals[t].append(seed_data[s]['final_tier_rates'][t])

        parts = []
        for t in range(5):
            m, sd = np.mean(tier_vals[t]), np.std(tier_vals[t])
            parts.append(f"{m:.4f}±{sd:.4f}")
        print(f"{group_name:<28} {parts[0]:>12} {parts[1]:>12} {parts[2]:>12} {parts[3]:>12} {parts[4]:>12}")


def report_score_distribution(experiments, duration_filter):
    """Print final score distribution (6 bins by depth tier)."""
    tag = duration_filter.upper()
    print_header(f"SCORE DISTRIBUTION (6 bins) — {tag} EXPERIMENTS")

    print(f"  Bins: [fail, tier0, tier1, tier2, tier3, tier4+]")
    print()

    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue
        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())

        dists = np.stack([seed_data[s]['final_score_dist'] for s in seeds])
        mean_dist = np.mean(dists, axis=0)
        print(f"  {group_name:<28} "
              f"fail={mean_dist[0]:.3f}  T0={mean_dist[1]:.3f}  T1={mean_dist[2]:.3f}  "
              f"T2={mean_dist[3]:.3f}  T3={mean_dist[4]:.3f}  T4+={mean_dist[5]:.3f}")


def report_mask_metrics(experiments, duration_filter):
    """Print mask-specific metrics for experiments that have them."""
    tag = duration_filter.upper()
    print_header(f"MASK METRICS — {tag} EXPERIMENTS")

    has_any = False
    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue
        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())

        # Check if any seed has mask data
        if seed_data[seeds[0]]['final_mask_infeasible'] is None:
            continue
        has_any = True

        mif = [seed_data[s]['final_mask_infeasible'] for s in seeds]
        mbf = [seed_data[s]['final_mask_blocked'] for s in seeds]
        print(f"  {group_name:<28} infeasible_frac={np.mean(mif):.4f}±{np.std(mif):.4f}  "
              f"blocked_frac={np.mean(mbf):.4f}±{np.std(mbf):.4f}")
    if not has_any:
        print("  (no mask experiments in this duration)")


def report_td_error(experiments, duration_filter):
    """Print TD-error statistics for replay ablations."""
    tag = duration_filter.upper()
    print_header(f"TD-ERROR STATISTICS — {tag} EXPERIMENTS")

    hdr = f"{'Experiment':<28} {'TD_mean':>14} {'TD_max':>14}"
    print(hdr)
    print('-' * len(hdr))

    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue
        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())

        tdm = [seed_data[s]['final_td_mean'] for s in seeds]
        tdx = [seed_data[s]['final_td_max'] for s in seeds]
        print(f"{group_name:<28} {np.mean(tdm):.6f}±{np.std(tdm):.6f}  {np.mean(tdx):.6f}±{np.std(tdx):.6f}")


def report_forgetting_detail(experiments, duration_filter):
    """Print detailed forgetting analysis: which achievements are most forgotten."""
    tag = duration_filter.upper()
    print_header(f"FORGETTING DETAIL — {tag} EXPERIMENTS (achievements with forgetting > 0.01)")

    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue
        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())

        all_forget = np.stack([seed_data[s]['computed_forgetting'] for s in seeds])
        mean_forget = np.mean(all_forget, axis=0)

        significant = np.where(mean_forget > 0.01)[0]
        if len(significant) == 0:
            continue

        order = significant[np.argsort(-mean_forget[significant])]

        print(f"\n  {group_name}")
        print(f"  AggForget(computed)={np.mean(mean_forget):.4f}")
        for idx in order:
            name = ACHIEVEMENT_NAMES[idx]
            tier = TIER_MAP[name]
            all_peak = np.stack([seed_data[s]['peak_per_ach_rate'] for s in seeds])
            all_final = np.stack([seed_data[s]['final_per_ach_rate'] for s in seeds])
            pk = np.mean(all_peak[:, idx])
            fn = np.mean(all_final[:, idx])
            fg = mean_forget[idx]
            print(f"    {name:<30} T{tier}  peak={pk:.4f}  final={fn:.4f}  forgot={fg:.4f}")


def report_trajectory_snapshots(experiments, duration_filter):
    """Print trajectory snapshots at key checkpoints for learning dynamics."""
    tag = duration_filter.upper()
    if duration_filter == '1m':
        checkpoints = [100_000, 250_000, 500_000, 750_000, 1_000_000]
    else:
        checkpoints = [500_000, 1_000_000, 2_000_000, 5_000_000, 7_500_000, 10_000_000]

    print_header(f"LEARNING TRAJECTORY SNAPSHOTS — {tag} EXPERIMENTS")
    print(f"  Checkpoints: {checkpoints}")
    print()

    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration')
        if dur != duration_filter:
            continue
        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())

        print(f"  {group_name}")
        print(f"  {'Checkpoint':>12} {'MeanAchRate':>14} {'AggForget':>12} {'FrontierR':>12} {'Depth':>8}")
        print(f"  {'-'*62}")

        for cp in checkpoints:
            mars, afgs, frs, deps = [], [], [], []
            for s in seeds:
                d = seed_data[s]
                # Find closest record to checkpoint
                idx = np.searchsorted(d['steps'], cp)
                idx = min(idx, len(d['steps']) - 1)
                # Small window around checkpoint
                lo = max(0, idx - 5)
                hi = min(len(d['steps']), idx + 5)
                mars.append(float(np.mean(d['mean_ach_rate_ts'][lo:hi])))
                afgs.append(float(np.mean(d['agg_forgetting_ts'][lo:hi])))
                frs.append(float(np.mean(d['frontier_rate_ts'][lo:hi])))
                deps.append(int(max(d['depth_ts'][lo:hi])))

            mar_str = f"{np.mean(mars):.4f}±{np.std(mars):.4f}"
            afg_str = f"{np.mean(afgs):.4f}±{np.std(afgs):.4f}"
            fr_str = f"{np.mean(frs):.4f}±{np.std(frs):.4f}"
            dep_str = f"{np.mean(deps):.1f}±{np.std(deps):.1f}"
            print(f"  {cp:>12,} {mar_str:>14} {afg_str:>12} {fr_str:>12} {dep_str:>8}")
        print()


def report_cross_comparison(experiments):
    """Print a single comparison table ranking all experiments by mean ach rate."""
    print_header("CROSS-EXPERIMENT RANKING BY MEAN ACHIEVEMENT RATE (all durations)")

    rows = []
    for group_name in sorted(experiments.keys()):
        seed_results = experiments[group_name]
        dur = seed_results.get('_duration', '?')
        seed_data = {k: v for k, v in seed_results.items() if k != '_duration'}
        seeds = sorted(seed_data.keys())
        is_intr = group_name in INTRINSIC_EXPS

        mar = [seed_data[s]['final_mean_ach_rate'] for s in seeds]
        nea = [seed_data[s]['num_ever_achieved'] for s in seeds]
        afg = [seed_data[s]['computed_agg_forgetting'] for s in seeds]
        depth = [seed_data[s]['max_depth_ever'] for s in seeds]

        rows.append({
            'name': group_name,
            'dur': dur,
            'is_intr': is_intr,
            'mar_mean': np.mean(mar),
            'mar_std': np.std(mar),
            'nea_mean': np.mean(nea),
            'afg_mean': np.mean(afg),
            'depth_mean': np.mean(depth),
        })

    # Sort by mean ach rate descending
    rows.sort(key=lambda r: -r['mar_mean'])

    hdr = f"{'Rank':>4} {'Experiment':<28} {'Dur':>4} {'Intr':>5} {'MeanAchRate':>14} {'#EverAch':>9} {'AggForget':>11} {'MaxDepth':>9}"
    print(hdr)
    print('-' * len(hdr))
    for i, r in enumerate(rows, 1):
        mar_str = f"{r['mar_mean']:.4f}±{r['mar_std']:.4f}"
        print(f"{i:>4} {r['name']:<28} {r['dur']:>4} {'YES' if r['is_intr'] else 'no':>5} "
              f"{mar_str:>14} {r['nea_mean']:>9.1f} {r['afg_mean']:>11.4f} {r['depth_mean']:>9.1f}")


# ============================================================
# Main
# ============================================================

def main():
    if not RESULTS_DIR.is_dir():
        print(f"ERROR: Directory not found: {RESULTS_DIR}")
        sys.exit(1)

    groups = discover_experiments()
    print(f"Discovered {len(groups)} experiment groups:")
    for g in sorted(groups.keys()):
        seeds = sorted(groups[g].keys())
        print(f"  {g}: seeds {seeds}")

    # Load and analyze all data
    experiments = {}  # group -> {seed: result_dict, '_duration': '1m'|'10m'}
    for group_name in sorted(groups.keys()):
        is_intr = group_name in INTRINSIC_EXPS
        seed_files = groups[group_name]
        experiments[group_name] = {}
        for seed, path in sorted(seed_files.items()):
            print(f"  Loading {group_name} seed={seed}...", end='', flush=True)
            records = load_jsonl(path)
            result = analyze_seed(records, is_intr)
            if result is not None:
                experiments[group_name][seed] = result
                dur = classify_duration(group_name, records)
                experiments[group_name]['_duration'] = dur
            print(f" {len(records)} records, max_step={result['max_step'] if result else 'N/A'}")

    # ---- Reports ----
    for dur in ['1m', '10m']:
        has_dur = any(v.get('_duration') == dur for v in experiments.values())
        if not has_dur:
            continue
        report_final_table(experiments, dur)
        report_per_seed_detail(experiments, dur)
        report_per_tier_summary(experiments, dur)
        report_score_distribution(experiments, dur)
        report_td_error(experiments, dur)
        report_mask_metrics(experiments, dur)
        report_forgetting_detail(experiments, dur)
        report_per_achievement_final(experiments, dur)
        report_trajectory_snapshots(experiments, dur)

    report_cross_comparison(experiments)

    print_header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()

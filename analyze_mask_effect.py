#!/usr/bin/env python3
"""
Action Mask Diagnostic Tool
=============================
Analyzes the mask mechanism itself: loss magnitude, prediction quality,
mask activity, penalty scale, effective action space.  Designed for
parameter tuning, not experiment comparison.

Reads from:
  - online_metrics.jsonl  (mask_penalty_mean, mask_infeasible_frac, ...)
  - metrics.jsonl          (loss/mask_ctx, loss/rew, loss/dyn, ...)

Usage:
  python analyze_mask_effect.py <logdir>
  python analyze_mask_effect.py experiment_results/ablation/F1_mask_soft_seed1/craftax_F1_mask_soft1
  python analyze_mask_effect.py --logdir <path> --outdir mask_diag
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================================
# Data Loading
# ============================================================================

def load_jsonl(path: str) -> List[Dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def find_logdir(path: str) -> str:
    """Resolve to the directory containing metrics files."""
    if os.path.isfile(path):
        return os.path.dirname(path)
    # Check for nested craftax_ subdir
    for entry in os.listdir(path):
        sub = os.path.join(path, entry)
        if os.path.isdir(sub) and entry.startswith("craftax_"):
            return sub
    return path


def resolve_key(records: List[Dict], base: str, prefixes=("", "episode/")) -> Optional[str]:
    """Find the actual key in records, trying multiple prefixes."""
    sample = records[:100]
    for pfx in prefixes:
        candidate = pfx + base
        if any(candidate in r and r[candidate] is not None for r in sample):
            return candidate
    return None


def get_series(records: List[Dict], key: str,
               step_key: str = "step") -> Tuple[np.ndarray, np.ndarray]:
    steps, vals = [], []
    for r in records:
        s = r.get(step_key)
        v = r.get(key)
        if s is not None and v is not None:
            sf = float(s)
            vf = float(v)
            if not math.isnan(vf):
                steps.append(sf)
                vals.append(vf)
    return np.array(steps), np.array(vals)


def smooth(v: np.ndarray, w: int = 30) -> np.ndarray:
    if len(v) <= w:
        return v
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w // 2, v[0]), v, np.full(w // 2, v[-1])])
    return np.convolve(pad, k, mode="valid")[:len(v)]


# ============================================================================
# Diagnostic Panels
# ============================================================================

def diag_loss_comparison(train_records, outdir, fmt):
    """Panel 1: mask_ctx loss vs all other losses — is it too small?"""
    loss_keys = [
        ("loss/mask_ctx", "mask_ctx"),
        ("loss/dyn", "dynamics"),
        ("loss/rep", "representation"),
        ("loss/rew", "reward"),
        ("loss/con", "continuation"),
        ("loss/policy", "policy"),
        ("loss/value", "value"),
    ]

    found = {}
    for raw_key, label in loss_keys:
        actual = resolve_key(train_records, raw_key, prefixes=("", "episode/"))
        if actual:
            s, v = get_series(train_records, actual)
            if len(s) > 0:
                found[label] = (s, v)

    if "mask_ctx" not in found:
        print("    loss/mask_ctx not found in training metrics.")
        return None

    # -- Plot A: all losses on log scale --
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    for label, (s, v) in found.items():
        lw = 2.5 if label == "mask_ctx" else 1.2
        alpha = 1.0 if label == "mask_ctx" else 0.7
        ax.plot(s, smooth(v, 50), linewidth=lw, alpha=alpha, label=label)
    ax.set_yscale("log")
    ax.set_title("All Losses (log scale)", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- Plot B: mask_ctx loss alone with statistics --
    ax = axes[1]
    s, v = found["mask_ctx"]
    ax.plot(s, smooth(v, 30), color="tab:red", linewidth=2, label="mask_ctx loss")
    ax.axhline(np.nanmean(v), color="gray", linestyle="--", alpha=0.5,
               label=f"mean = {np.nanmean(v):.4f}")
    # Mark warmup boundary
    ax.axvline(50000, color="blue", linestyle=":", alpha=0.5, label="warmup end (50k)")
    ax.set_title("mask_ctx Loss Detail", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Diagnostic 1: Loss Magnitude Comparison", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"diag1_loss_comparison.{fmt}"), dpi=150)
    plt.close(fig)

    # -- Numerical report --
    lines = ["=" * 70, "  DIAGNOSTIC 1: Loss Magnitude Comparison", "=" * 70]
    lines.append(f"  {'Loss':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    lines.append("  " + "-" * 68)
    for label, (s, v) in sorted(found.items()):
        lines.append(f"  {label:<20} {np.nanmean(v):>12.6f} {np.nanstd(v):>12.6f} "
                     f"{np.nanmin(v):>12.6f} {np.nanmax(v):>12.6f}")

    # Ratio analysis
    mask_mean = np.nanmean(found["mask_ctx"][1])
    lines.append(f"\n  Ratio analysis (mask_ctx mean = {mask_mean:.6f}):")
    for label, (s, v) in found.items():
        if label != "mask_ctx":
            ratio = mask_mean / max(np.nanmean(v), 1e-12)
            lines.append(f"    mask_ctx / {label:<16} = {ratio:.4f}")

    return "\n".join(lines)


def diag_mask_ctx_convergence(train_records, outdir, fmt):
    """Panel 2: Has mask_ctx head converged? Loss in early vs late training."""
    key = resolve_key(train_records, "loss/mask_ctx", ("", "episode/"))
    if not key:
        return None

    s, v = get_series(train_records, key)
    if len(s) < 100:
        return None

    # Split into quarters
    n = len(v)
    q1, q2, q3, q4 = v[:n//4], v[n//4:n//2], v[n//2:3*n//4], v[3*n//4:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Plot A: loss curve with convergence annotation --
    ax = axes[0]
    ax.plot(s, smooth(v, 50), color="tab:red", linewidth=2)
    # Mark quarters
    for i, (qs, qe, ql, qm) in enumerate([
        (0, n//4, "Q1", np.nanmean(q1)),
        (n//4, n//2, "Q2", np.nanmean(q2)),
        (n//2, 3*n//4, "Q3", np.nanmean(q3)),
        (3*n//4, n, "Q4", np.nanmean(q4)),
    ]):
        ax.axhspan(qm - 0.001, qm + 0.001,
                    xmin=i/4, xmax=(i+1)/4, alpha=0.1, color=f"C{i}")
        ax.text(s[min(qs + 10, n-1)], qm, f" {ql}: {qm:.4f}",
                fontsize=8, va="bottom")
    ax.axvline(50000, color="blue", linestyle=":", alpha=0.5, label="warmup end")
    ax.set_title("mask_ctx Loss Convergence", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Plot B: rolling window statistics --
    ax = axes[1]
    window = max(50, n // 20)
    rolling_mean = np.convolve(v, np.ones(window)/window, mode="valid")
    rolling_std = np.array([np.std(v[max(0,i-window):i+1]) for i in range(len(v))])
    ax.plot(s[:len(rolling_mean)], rolling_mean, color="tab:red", label="rolling mean")
    ax.fill_between(s[:len(rolling_mean)],
                    smooth(v[:len(rolling_mean)] - rolling_std[:len(rolling_mean)], 50),
                    smooth(v[:len(rolling_mean)] + rolling_std[:len(rolling_mean)], 50),
                    alpha=0.15, color="tab:red", label="±1 std")
    ax.set_title("Rolling Statistics", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Diagnostic 2: mask_ctx Head Convergence", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"diag2_convergence.{fmt}"), dpi=150)
    plt.close(fig)

    # Convergence ratio
    late_mean = np.nanmean(q4)
    early_mean = np.nanmean(q1)
    reduction = (early_mean - late_mean) / max(early_mean, 1e-12) * 100

    lines = ["=" * 70, "  DIAGNOSTIC 2: mask_ctx Convergence", "=" * 70]
    lines.append(f"  Q1 (0-25%) mean:    {np.nanmean(q1):.6f}")
    lines.append(f"  Q2 (25-50%) mean:   {np.nanmean(q2):.6f}")
    lines.append(f"  Q3 (50-75%) mean:   {np.nanmean(q3):.6f}")
    lines.append(f"  Q4 (75-100%) mean:  {np.nanmean(q4):.6f}")
    lines.append(f"  Loss reduction Q1→Q4: {reduction:.1f}%")
    lines.append(f"  Converged: {'YES' if reduction > 30 else 'MAYBE' if reduction > 10 else 'NO'}")
    if reduction < 10:
        lines.append("  ⚠ mask_ctx loss barely decreased — head may not be learning.")
        lines.append("    Consider: increase loss scale (currently 1.0), check input features.")

    return "\n".join(lines)


def diag_mask_activity(online_records, outdir, fmt):
    """Panel 3: Mask activity — infeasible fraction, penalty magnitude."""
    metrics = [
        ("mask_penalty_mean", "Mean Penalty", "tab:orange"),
        ("mask_infeasible_frac", "Infeasible Fraction", "tab:red"),
        ("mask_blocked_frac", "Blocked Fraction (hard)", "tab:purple"),
    ]

    found = {}
    for base, label, color in metrics:
        key = resolve_key(online_records, base, ("", "episode/"))
        if key:
            s, v = get_series(online_records, key)
            if len(s) > 0:
                found[label] = (s, v, color)

    if not found:
        print("    No mask activity metrics in online records.")
        return None

    n_panels = len(found)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for i, (label, (s, v, color)) in enumerate(found.items()):
        ax = axes[i]
        ax.plot(s, smooth(v, 30), color=color, linewidth=2)
        ax.axhline(np.nanmean(v), color="gray", linestyle="--", alpha=0.5)
        ax.axvline(50000, color="blue", linestyle=":", alpha=0.5, label="warmup end")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        # Annotate mean
        ax.text(0.98, 0.95, f"mean={np.nanmean(v):.4f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, bbox=dict(boxstyle="round", alpha=0.8,
                facecolor="wheat"))

    fig.suptitle("Diagnostic 3: Mask Activity Over Training", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"diag3_mask_activity.{fmt}"), dpi=150)
    plt.close(fig)

    # Effective action space
    lines = ["=" * 70, "  DIAGNOSTIC 3: Mask Activity", "=" * 70]
    if "Infeasible Fraction" in found:
        s, v, _ = found["Infeasible Fraction"]
        total_actions = 42
        early = v[:max(1, len(v)//10)]
        late = v[max(0, len(v) - len(v)//10):]
        lines.append(f"  Total actions: {total_actions}")
        lines.append(f"  Infeasible fraction (early 10%):  {np.nanmean(early):.3f}"
                     f"  → ~{int(total_actions * np.nanmean(early))} actions masked")
        lines.append(f"  Infeasible fraction (late 10%):   {np.nanmean(late):.3f}"
                     f"  → ~{int(total_actions * np.nanmean(late))} actions masked")
        lines.append(f"  Effective action space early: ~{total_actions - int(total_actions * np.nanmean(early))}")
        lines.append(f"  Effective action space late:  ~{total_actions - int(total_actions * np.nanmean(late))}")
    if "Mean Penalty" in found:
        s, v, _ = found["Mean Penalty"]
        lines.append(f"\n  Mean penalty (overall):   {np.nanmean(v):.4f}")
        lines.append(f"  Mean penalty (early 10%): {np.nanmean(v[:max(1,len(v)//10)]):.4f}")
        lines.append(f"  Mean penalty (late 10%):  {np.nanmean(v[max(0,len(v)-len(v)//10):]):.4f}")
        if np.nanmean(v) < 0.1:
            lines.append("  ⚠ Penalty is very small — consider increasing lambda_penalty.")
        elif np.nanmean(v) > 10:
            lines.append("  ⚠ Penalty is very large — may over-constrain policy exploration.")

    return "\n".join(lines)


def diag_penalty_vs_logits(train_records, online_records, outdir, fmt):
    """Panel 4: Is λ·deficit large enough relative to typical logit magnitudes?"""
    # Policy entropy is a proxy for logit spread
    ent_key = resolve_key(train_records, "policy_entropy",
                          ("", "episode/", "loss/"))
    # Also check for entropy in online records
    if not ent_key:
        ent_key = resolve_key(online_records, "policy_entropy",
                              ("", "episode/", "loss/"))
        records_for_ent = online_records
    else:
        records_for_ent = train_records

    pen_key = resolve_key(online_records, "mask_penalty_mean", ("", "episode/"))
    if not pen_key:
        return None

    s_pen, v_pen = get_series(online_records, pen_key)
    if len(s_pen) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(s_pen, smooth(v_pen, 30), color="tab:orange", linewidth=2,
            label="Mean mask penalty (λ·deficit)")

    if ent_key:
        s_ent, v_ent = get_series(records_for_ent, ent_key)
        if len(s_ent) > 0:
            ax.plot(s_ent, smooth(v_ent, 30), color="tab:blue", linewidth=2,
                    label="Policy entropy", alpha=0.7)

    ax.axvline(50000, color="blue", linestyle=":", alpha=0.5, label="warmup end")
    ax.set_title("Mask Penalty vs Policy Entropy", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"diag4_penalty_scale.{fmt}"), dpi=150)
    plt.close(fig)

    lines = ["=" * 70, "  DIAGNOSTIC 4: Penalty Scale Analysis", "=" * 70]
    lines.append(f"  Mean mask penalty: {np.nanmean(v_pen):.4f}")
    lines.append(f"  For reference: with 42 actions, uniform logits ≈ log(42) ≈ {math.log(42):.2f}")
    lines.append(f"  λ=5.0 with deficit=1 gives penalty=5.0, shifting ~5 logit units")
    if np.nanmean(v_pen) < 0.5:
        lines.append("  ⚠ Average penalty < 0.5 — masking effect is weak.")
        lines.append("    Most actions are feasible (good) OR lambda is too small.")

    return "\n".join(lines)


def diag_before_after_warmup(online_records, outdir, fmt):
    """Panel 5: Performance before vs after dream mask warmup (50k step)."""
    warmup_step = 50000
    metrics = [
        ("score", "Episode Score"),
        ("return_mean", "Mean Return"),
        ("mask_infeasible_frac", "Infeasible Frac"),
    ]

    found = {}
    for base, label in metrics:
        key = resolve_key(online_records, base, ("", "episode/"))
        if key:
            s, v = get_series(online_records, key)
            if len(s) > 0:
                found[label] = (s, v)

    if not found:
        return None

    fig, axes = plt.subplots(1, len(found), figsize=(6 * len(found), 5))
    if len(found) == 1:
        axes = [axes]

    lines = ["=" * 70, "  DIAGNOSTIC 5: Before vs After Dream Mask Warmup", "=" * 70]
    lines.append(f"  Warmup boundary: step {warmup_step}")
    lines.append(f"  {'Metric':<25} {'Before':>12} {'After':>12} {'Delta':>12} {'%Change':>10}")
    lines.append("  " + "-" * 75)

    for i, (label, (s, v)) in enumerate(found.items()):
        ax = axes[i]
        before_mask = s < warmup_step
        after_mask = s >= warmup_step

        if before_mask.any():
            ax.plot(s[before_mask], smooth(v[before_mask], 20),
                    color="tab:gray", linewidth=2, label="before warmup")
        if after_mask.any():
            ax.plot(s[after_mask], smooth(v[after_mask], 20),
                    color="tab:green", linewidth=2, label="after warmup")
        ax.axvline(warmup_step, color="blue", linestyle="--", linewidth=2,
                   label="warmup end", alpha=0.7)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        bm = np.nanmean(v[before_mask]) if before_mask.any() else float("nan")
        am = np.nanmean(v[after_mask]) if after_mask.any() else float("nan")
        delta = am - bm
        pct = delta / abs(bm) * 100 if bm != 0 and not math.isnan(bm) else 0
        lines.append(f"  {label:<25} {bm:>12.4f} {am:>12.4f} {delta:>+12.4f} {pct:>+9.1f}%")

    fig.suptitle("Diagnostic 5: Dream Mask Warmup Effect", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(outdir, f"diag5_warmup_effect.{fmt}"), dpi=150)
    plt.close(fig)

    return "\n".join(lines)


def diag_parameter_recommendations(online_records, train_records):
    """Panel 6: Automated parameter recommendations."""
    lines = ["=" * 70, "  DIAGNOSTIC 6: Parameter Recommendations", "=" * 70]

    # Check mask_ctx loss
    ctx_key = resolve_key(train_records, "loss/mask_ctx", ("", "episode/"))
    if ctx_key:
        _, v = get_series(train_records, ctx_key)
        if len(v) > 0:
            mean_loss = np.nanmean(v)
            late_loss = np.nanmean(v[max(0, len(v) - len(v)//4):])

            # Compare with reward loss
            rew_key = resolve_key(train_records, "loss/rew", ("", "episode/"))
            if rew_key:
                _, rv = get_series(train_records, rew_key)
                if len(rv) > 0:
                    rew_mean = np.nanmean(rv)
                    ratio = mean_loss / max(rew_mean, 1e-12)
                    lines.append(f"\n  [mask_ctx_scale] Current: 1.0")
                    lines.append(f"    mask_ctx loss mean: {mean_loss:.6f}")
                    lines.append(f"    reward loss mean:   {rew_mean:.6f}")
                    lines.append(f"    Ratio (mask_ctx/rew): {ratio:.3f}")
                    if ratio < 0.01:
                        lines.append(f"    → RECOMMENDATION: Increase mask_ctx scale to ~{max(1, int(1/ratio))}x")
                        lines.append(f"      The gradient from mask_ctx is negligible compared to reward.")
                    elif ratio > 100:
                        lines.append(f"    → RECOMMENDATION: Decrease mask_ctx scale to ~{1/ratio:.2f}x")
                        lines.append(f"      mask_ctx loss dominates — may distort representation.")
                    else:
                        lines.append(f"    → OK: mask_ctx loss is in reasonable range relative to reward.")

            if late_loss > mean_loss * 0.9:
                lines.append(f"\n  [convergence] Late loss ({late_loss:.6f}) ≈ mean loss ({mean_loss:.6f})")
                lines.append(f"    → Head may not be converging. Check:")
                lines.append(f"      - Is action_mask_context in obs_space?")
                lines.append(f"      - Are RSSM features informative enough?")
                lines.append(f"      - Consider increasing MLP layers (currently 2).")

    # Check penalty
    pen_key = resolve_key(online_records, "mask_penalty_mean", ("", "episode/"))
    if pen_key:
        _, pv = get_series(online_records, pen_key)
        if len(pv) > 0:
            pen_mean = np.nanmean(pv)
            lines.append(f"\n  [lambda_penalty] Current: 5.0 (soft mode)")
            lines.append(f"    Mean penalty per step: {pen_mean:.4f}")
            if pen_mean < 0.1:
                lines.append(f"    → Most actions are feasible. Penalty is appropriate.")
            elif pen_mean < 1.0:
                lines.append(f"    → Moderate masking. Lambda seems reasonable.")
            elif pen_mean < 5.0:
                lines.append(f"    → Heavy masking. Consider if this is expected for early game.")
            else:
                lines.append(f"    → Very heavy masking. Lambda may be too high for soft mode.")

    # Check infeasible fraction
    inf_key = resolve_key(online_records, "mask_infeasible_frac", ("", "episode/"))
    if inf_key:
        _, iv = get_series(online_records, inf_key)
        if len(iv) > 0:
            inf_mean = np.nanmean(iv)
            lines.append(f"\n  [effective_action_space]")
            lines.append(f"    Mean infeasible fraction: {inf_mean:.3f}")
            lines.append(f"    Effective actions: ~{42 - int(42 * inf_mean)}/{42}")
            if inf_mean > 0.8:
                lines.append(f"    → WARNING: >80% actions masked — agent is very constrained.")
                lines.append(f"      This is normal in early Craftax (no items), but if persistent,")
                lines.append(f"      the agent may not be progressing.")

    # Warmup recommendation
    lines.append(f"\n  [warmup_steps] Current: 50,000 (10k prefill + 40k warmup)")
    if ctx_key:
        _, v = get_series(train_records, ctx_key)
        if len(v) > 100:
            # Find when loss drops to 50% of initial
            early = np.nanmean(v[:max(1, len(v)//20)])
            target = early * 0.5
            steps_arr, _ = get_series(train_records, ctx_key)
            for j in range(len(v)):
                if smooth(v, 50)[min(j, len(v)-1)] < target:
                    conv_step = int(steps_arr[j]) if j < len(steps_arr) else None
                    lines.append(f"    Loss halved at step ~{conv_step}")
                    if conv_step and conv_step < 30000:
                        lines.append(f"    → Could reduce warmup to ~{max(conv_step + 10000, 30000)} steps.")
                    elif conv_step and conv_step > 80000:
                        lines.append(f"    → Consider increasing warmup to ~{conv_step + 20000} steps.")
                    break

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Action mask diagnostic tool.")
    parser.add_argument("logdir", nargs="?", default=None,
                        help="Path to experiment logdir (containing metrics files)")
    parser.add_argument("--logdir", dest="logdir_flag", default=None)
    parser.add_argument("--outdir", default="mask_diag")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    logdir = args.logdir or args.logdir_flag
    if not logdir:
        # Try to find any experiment dir
        for candidate in [
            "experiment_results/ablation",
            "experiment_results",
            "logdir",
        ]:
            if os.path.isdir(candidate):
                logdir = candidate
                break
    if not logdir or not os.path.isdir(logdir):
        print("ERROR: Provide a logdir path containing online_metrics.jsonl / metrics.jsonl")
        sys.exit(1)

    logdir = find_logdir(logdir)
    print("=" * 60)
    print("  ACTION MASK DIAGNOSTIC TOOL")
    print("=" * 60)
    print(f"  Logdir: {os.path.abspath(logdir)}")

    online_path = os.path.join(logdir, "online_metrics.jsonl")
    train_path = os.path.join(logdir, "metrics.jsonl")
    online_records = load_jsonl(online_path)
    train_records = load_jsonl(train_path)

    print(f"  online_metrics.jsonl: {len(online_records)} records")
    print(f"  metrics.jsonl:        {len(train_records)} records")

    if not online_records and not train_records:
        print("\nERROR: No metrics files found.")
        # Try listing available JSONL files
        for f in sorted(os.listdir(logdir)):
            if f.endswith(".jsonl"):
                print(f"  Found: {f}")
        sys.exit(1)

    # Show available keys (sample)
    print("\n  Available keys (sample):")
    for source, name in [(online_records, "online"), (train_records, "training")]:
        if source:
            keys = sorted(set(k for r in source[:10] for k in r.keys()))
            mask_keys = [k for k in keys if "mask" in k.lower()]
            loss_keys = [k for k in keys if "loss" in k.lower()]
            if mask_keys:
                print(f"    [{name}] mask:  {', '.join(mask_keys)}")
            if loss_keys:
                print(f"    [{name}] loss:  {', '.join(loss_keys[:10])}")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    fmt = args.format
    report_parts = []

    # Run diagnostics
    print("\nRunning diagnostics:")

    print("  [1/6] Loss magnitude comparison ...")
    r = diag_loss_comparison(train_records, outdir, fmt)
    if r: report_parts.append(r)

    print("  [2/6] mask_ctx convergence ...")
    r = diag_mask_ctx_convergence(train_records, outdir, fmt)
    if r: report_parts.append(r)

    print("  [3/6] Mask activity metrics ...")
    r = diag_mask_activity(online_records, outdir, fmt)
    if r: report_parts.append(r)

    print("  [4/6] Penalty scale analysis ...")
    r = diag_penalty_vs_logits(train_records, online_records, outdir, fmt)
    if r: report_parts.append(r)

    print("  [5/6] Warmup effect ...")
    r = diag_before_after_warmup(online_records, outdir, fmt)
    if r: report_parts.append(r)

    print("  [6/6] Parameter recommendations ...")
    r = diag_parameter_recommendations(online_records, train_records)
    if r: report_parts.append(r)

    # Save report
    report = "\n\n".join(report_parts)
    print("\n" + report)
    report_path = os.path.join(outdir, "mask_diagnostic_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    outputs = sorted(f for f in os.listdir(outdir))
    print(f"\nGenerated {len(outputs)} files in {os.path.abspath(outdir)}/:")
    for f in outputs:
        sz = os.path.getsize(os.path.join(outdir, f))
        print(f"  {f} ({sz / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

"""Compute per-action logit bias from context array and declarative rules.

Modes:
  - none: no bias
  - soft: bias[a] = -lambda_penalty * deficit
  - hard: bias[a] = -large_negative if any condition fails
"""

from __future__ import annotations

import numpy as np

from .rules import C

try:
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    jnp = None
    _HAS_JAX = False


def compute_logit_bias(context, rules, num_actions, lambda_penalty=5.0,
                       large_negative=1e9, mode="soft"):
    """Backward-compatible wrapper returning only the bias tensor."""
    return compute_mask_details(
        context, rules, num_actions,
        lambda_penalty=lambda_penalty,
        large_negative=large_negative,
        mode=mode,
    )["bias"]


def compute_mask_details(context, rules, num_actions, lambda_penalty=5.0,
                         large_negative=1e9, mode="soft"):
    """Compute per-action deficit, invalid flags, and logit bias.

    Args:
        context: float32 array of shape (CONTEXT_SIZE,) or (B, CONTEXT_SIZE),
                 OR a dict (legacy format, auto-converted).
        rules: ACTION_RULES dict from rules.py.
        num_actions: number of discrete actions.
    """
    xp = _array_module()
    ctx = _ensure_context_array(context, xp)
    batch_shape = ctx.shape[:-1]  # () for unbatched, (B,) for batched
    bias = xp.zeros(batch_shape + (num_actions,), dtype=xp.float32)
    rule_stats = {}

    for name, rule in rules.items():
        deficit = _evaluate_conditions(rule["conditions"], ctx, xp)
        invalid = deficit > 0

        if mode == "soft":
            rule_bias = -float(lambda_penalty) * deficit
        elif mode == "hard":
            rule_bias = xp.where(invalid, -float(large_negative), 0.0)
        else:
            rule_bias = xp.zeros_like(deficit)

        action_id = int(rule["action_id"])
        if action_id < num_actions:
            bias = _set_last_axis(bias, action_id, rule_bias, xp)

        rule_stats[name] = {
            "action_id": action_id,
            "deficit": deficit,
            "invalid": invalid,
            "bias": rule_bias,
        }

    return {"mode": mode, "num_actions": int(num_actions),
            "bias": bias, "rules": rule_stats}


def _evaluate_conditions(conditions, ctx, xp):
    """Evaluate a list of condition tuples and return total deficit."""
    batch_shape = ctx.shape[:-1]
    deficit = xp.zeros(batch_shape, dtype=xp.float32)

    for cond in conditions:
        tag = cond[0]

        if tag == "min":
            # context[idx] >= value; deficit += max(0, value - context[idx])
            idx, value = cond[1], float(cond[2])
            deficit = deficit + xp.maximum(0.0, value - ctx[..., idx])

        elif tag == "below":
            # context[idx] < value; deficit += 1 if context[idx] >= value
            idx, value = cond[1], float(cond[2])
            deficit = deficit + xp.where(ctx[..., idx] >= value, 1.0, 0.0)

        elif tag == "bool":
            # context[idx] > 0.5; deficit += 1 if not true
            idx = cond[1]
            deficit = deficit + xp.where(ctx[..., idx] > 0.5, 0.0, 1.0)

        elif tag == "any_below":
            # any(context[start:end] < value); deficit += 1 if none below
            start, end, value = cond[1], cond[2], float(cond[3])
            slc = ctx[..., start:end]
            has_slot = (slc < value).any(axis=-1)
            deficit = deficit + xp.where(has_slot, 0.0, 1.0)

        elif tag == "sum_pos":
            # sum(context[start:end]) > 0; deficit += 1 if sum == 0
            start, end = cond[1], cond[2]
            s = ctx[..., start:end].sum(axis=-1)
            deficit = deficit + xp.where(s > 0, 0.0, 1.0)

        elif tag == "max_energy_check":
            # energy < 7 + 2 * dexterity
            energy = ctx[..., C.ENERGY]
            max_energy = 7.0 + 2.0 * ctx[..., C.DEXTERITY]
            deficit = deficit + xp.where(energy < max_energy, 0.0, 1.0)

        elif tag == "max_health_check":
            # health < 8 + strength
            health = ctx[..., C.HEALTH]
            max_health = 8.0 + ctx[..., C.STRENGTH]
            deficit = deficit + xp.where(health < max_health, 0.0, 1.0)

        elif tag == "gem_check":
            # ruby >= 1 OR sapphire >= 1
            has_gem = (ctx[..., C.RUBY] >= 1) | (ctx[..., C.SAPPHIRE] >= 1)
            deficit = deficit + xp.where(has_gem, 0.0, 1.0)

        elif tag == "attr_below_max":
            # context[idx] < 5 (max_attribute)
            idx = cond[1]
            deficit = deficit + xp.where(ctx[..., idx] < 5.0, 0.0, 1.0)

    return deficit


def compute_mask_logging_stats(details, rules, raw_logits=None,
                               adjusted_logits=None, large_negative=1e9):
    """Summarize mask diagnostics for logging."""
    xp = _array_module()
    bias = details["bias"]
    batch_shape = bias.shape[:-1]
    penalty = -xp.minimum(bias, 0.0)
    infeasible = (bias < 0.0).astype(xp.float32)
    if details["mode"] == "hard":
        blocked = (bias <= (-0.5 * float(large_negative))).astype(xp.float32)
    else:
        blocked = xp.zeros_like(infeasible)

    stats = empty_mask_logging_stats(batch_shape, rules, context_missing=False)
    stats["mask_penalty_mean"] = penalty.mean(-1)
    stats["mask_infeasible_frac"] = infeasible.mean(-1)
    stats["mask_blocked_frac"] = blocked.mean(-1)

    for name, rule in rules.items():
        suffix = rule["metric_suffix"]
        rd = details["rules"].get(name)
        if rd is not None:
            stats[f"mask/invalid_{suffix}_count"] = rd["invalid"].astype(xp.float32)
            stats[f"mask/deficit_{suffix}"] = rd["deficit"].astype(xp.float32)

    if raw_logits is not None and adjusted_logits is not None:
        # Log probability shift for PLACE_TABLE as a representative action
        pt_rule = rules.get("PLACE_TABLE")
        if pt_rule is not None:
            aid = pt_rule["action_id"]
            if aid < raw_logits.shape[-1]:
                pb = _softmax(raw_logits)
                pa = _softmax(adjusted_logits)
                stats["mask/place_table_prob_before"] = pb[..., aid]
                stats["mask/place_table_prob_after"] = pa[..., aid]
    return stats


def empty_mask_logging_stats(batch_shape, rules, context_missing=False):
    """Create zero-valued logging tensors."""
    xp = _array_module()
    zeros = xp.zeros(batch_shape, dtype=xp.float32)
    stats = {
        "mask_penalty_mean": zeros,
        "mask_infeasible_frac": zeros,
        "mask_blocked_frac": zeros,
        "mask/context_missing": (
            xp.ones(batch_shape, dtype=xp.float32) if context_missing else zeros),
        "mask/place_table_prob_before": zeros,
        "mask/place_table_prob_after": zeros,
    }
    for rule in rules.values():
        s = rule["metric_suffix"]
        stats[f"mask/invalid_{s}_count"] = zeros
        stats[f"mask/deficit_{s}"] = zeros
    return stats


def _ensure_context_array(context, xp):
    """Convert dict-format context (legacy) to flat array, or pass through."""
    if isinstance(context, dict):
        # Legacy dict format — pack into array with zeros for missing fields
        from .rules import CONTEXT_SIZE
        wood = xp.asarray(context.get("wood", 0), dtype=xp.float32)
        batch_shape = wood.shape
        ctx = xp.zeros(batch_shape + (CONTEXT_SIZE,), dtype=xp.float32)
        ctx = _set_last_axis(ctx, C.WOOD, xp.asarray(context.get("wood", 0), dtype=xp.float32), xp)
        ctx = _set_last_axis(ctx, C.STONE, xp.asarray(context.get("stone", 0), dtype=xp.float32), xp)
        ctx = _set_last_axis(ctx, C.COAL, xp.asarray(context.get("coal", 0), dtype=xp.float32), xp)
        ctx = _set_last_axis(ctx, C.IRON, xp.asarray(context.get("iron", 0), dtype=xp.float32), xp)
        near_t = xp.asarray(context.get("near_table", False), dtype=xp.float32)
        near_f = xp.asarray(context.get("near_furnace", False), dtype=xp.float32)
        ctx = _set_last_axis(ctx, C.NEAR_TABLE, near_t, xp)
        ctx = _set_last_axis(ctx, C.NEAR_FURNACE, near_f, xp)
        return ctx
    return xp.asarray(context, dtype=xp.float32)


def _set_last_axis(array, index, value, xp):
    if _HAS_JAX and xp is jnp:
        return array.at[..., index].set(value)
    result = np.array(array, copy=True)
    result[..., index] = value
    return result


def _array_module():
    return jnp if _HAS_JAX else np


def _softmax(logits):
    xp = _array_module()
    shifted = logits - logits.max(-1, keepdims=True)
    exp = xp.exp(shifted)
    return exp / exp.sum(-1, keepdims=True)

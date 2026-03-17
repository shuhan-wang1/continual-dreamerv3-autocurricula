"""Central feasibility and logging utilities for Craftax action masking."""

from __future__ import annotations

import numpy as np

try:
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    jnp = None
    _HAS_JAX = False


def compute_logit_bias(
    context,
    rules,
    num_actions,
    lambda_penalty=5.0,
    large_negative=1e9,
    mode="soft",
):
    """Backward-compatible wrapper that returns only the bias tensor."""
    return compute_mask_details(
        context,
        rules,
        num_actions,
        lambda_penalty=lambda_penalty,
        large_negative=large_negative,
        mode=mode,
    )["bias"]


def compute_mask_details(
    context,
    rules,
    num_actions,
    lambda_penalty=5.0,
    large_negative=1e9,
    mode="soft",
):
    """Compute deficits, invalid flags, and per-action logit bias."""
    xp = _array_module()
    batch_shape = xp.asarray(context["wood"]).shape
    bias = xp.zeros(batch_shape + (num_actions,), dtype=xp.float32)
    rule_stats = {}

    inventory = {
        "wood": xp.asarray(context.get("wood", 0), dtype=xp.float32),
        "stone": xp.asarray(context.get("stone", 0), dtype=xp.float32),
        "coal": xp.asarray(context.get("coal", 0), dtype=xp.float32),
        "iron": xp.asarray(context.get("iron", 0), dtype=xp.float32),
    }
    near_table = xp.asarray(context.get("near_table", True))
    near_furnace = xp.asarray(context.get("near_furnace", True))

    for name, rule in rules.items():
        deficit = xp.zeros(batch_shape, dtype=xp.float32)
        for resource, required in rule["requires"].items():
            current = inventory.get(resource, xp.asarray(0.0, dtype=xp.float32))
            deficit = deficit + xp.maximum(0.0, float(required) - current)
        if rule["needs_table"]:
            deficit = deficit + xp.where(near_table, 0.0, 1.0)
        if rule["needs_furnace"]:
            deficit = deficit + xp.where(near_furnace, 0.0, 1.0)

        invalid = deficit > 0
        if mode == "soft":
            rule_bias = -float(lambda_penalty) * deficit
        elif mode == "hard":
            rule_bias = xp.where(invalid, -float(large_negative), 0.0)
        else:
            rule_bias = xp.zeros_like(deficit, dtype=xp.float32)

        action_id = int(rule["action_id"])
        if action_id < num_actions:
            bias = _set_last_axis_value(bias, action_id, rule_bias)
        else:
            rule_bias = xp.zeros_like(deficit, dtype=xp.float32)

        rule_stats[name] = {
            "action_id": action_id,
            "deficit": deficit,
            "invalid": invalid,
            "bias": rule_bias,
        }

    return {
        "mode": mode,
        "num_actions": int(num_actions),
        "bias": bias,
        "rules": rule_stats,
    }


def compute_mask_logging_stats(details, rules, raw_logits=None, adjusted_logits=None, large_negative=1e9):
    """Summarize rule-level mask diagnostics for logging."""
    xp = _array_module()
    bias = details["bias"]
    batch_shape = bias.shape[:-1]
    penalty = -xp.minimum(bias, 0.0)
    infeasible = (bias < 0.0).astype(xp.float32)
    if details["mode"] == "hard":
        blocked = (bias <= (-0.5 * float(large_negative))).astype(xp.float32)
    else:
        blocked = xp.zeros_like(infeasible, dtype=xp.float32)

    stats = empty_mask_logging_stats(batch_shape, rules, context_missing=False)
    stats["mask_penalty_mean"] = penalty.mean(-1)
    stats["mask_infeasible_frac"] = infeasible.mean(-1)
    stats["mask_blocked_frac"] = blocked.mean(-1)

    for name, rule in rules.items():
        suffix = rule["metric_suffix"]
        rule_details = details["rules"][name]
        stats[f"mask/invalid_{suffix}_count"] = rule_details["invalid"].astype(xp.float32)
        if name == "PLACE_TABLE":
            stats["mask/mean_bias_place_table"] = rule_details["bias"].astype(xp.float32)

    if raw_logits is not None and adjusted_logits is not None:
        place_table_id = rules["PLACE_TABLE"]["action_id"]
        if place_table_id < raw_logits.shape[-1]:
            probs_before = _softmax(raw_logits)
            probs_after = _softmax(adjusted_logits)
            stats["mask/place_table_prob_before"] = probs_before[..., place_table_id]
            stats["mask/place_table_prob_after"] = probs_after[..., place_table_id]
    return stats


def empty_mask_logging_stats(batch_shape, rules, context_missing=False):
    """Create zero-valued logging tensors with the expected metric keys."""
    xp = _array_module()
    zeros = xp.zeros(batch_shape, dtype=xp.float32)
    stats = {
        "mask_penalty_mean": zeros,
        "mask_infeasible_frac": zeros,
        "mask_blocked_frac": zeros,
        "mask/context_missing": xp.ones(batch_shape, dtype=xp.float32) if context_missing else zeros,
        "mask/mean_bias_place_table": zeros,
        "mask/place_table_prob_before": zeros,
        "mask/place_table_prob_after": zeros,
    }
    for rule in rules.values():
        stats[f"mask/invalid_{rule['metric_suffix']}_count"] = zeros
    return stats


def assert_basic_mask_expectations(context, details):
    """Assert the first-pass rule logic required for the early-game experiment."""
    def _as_np(value):
        arr = np.asarray(value)
        return arr.reshape(1) if arr.ndim == 0 else arr

    wood = _as_np(context["wood"])
    near_table = _as_np(context["near_table"]).astype(bool)
    place_table_invalid = _as_np(details["rules"]["PLACE_TABLE"]["invalid"]).astype(bool)
    make_wood_pickaxe_invalid = _as_np(details["rules"]["MAKE_WOOD_PICKAXE"]["invalid"]).astype(bool)
    make_stone_pickaxe_invalid = _as_np(details["rules"]["MAKE_STONE_PICKAXE"]["invalid"]).astype(bool)

    if np.any(wood < 2):
        assert np.all(place_table_invalid[wood < 2]), "wood < 2 must invalidate PLACE_TABLE"
    if np.any(wood >= 2):
        assert np.all(~place_table_invalid[wood >= 2]), "wood >= 2 must allow PLACE_TABLE"
    if np.any(~near_table):
        assert np.all(make_wood_pickaxe_invalid[~near_table]), "no nearby table must invalidate MAKE_WOOD_PICKAXE"
        assert np.all(make_stone_pickaxe_invalid[~near_table]), "no nearby table must invalidate MAKE_STONE_PICKAXE"


def _array_module():
    return jnp if _HAS_JAX else np


def _set_last_axis_value(array, index, value):
    if _HAS_JAX:
        return array.at[..., index].set(value)
    result = np.array(array, copy=True)
    result[..., index] = value
    return result


def _softmax(logits):
    xp = _array_module()
    shifted = logits - logits.max(-1, keepdims=True)
    exp = xp.exp(shifted)
    return exp / exp.sum(-1, keepdims=True)

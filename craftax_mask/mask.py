"""
Compute logit bias from mask context and rules.

Modes:
  - none: no bias
  - soft: bias[a] = -lambda_penalty * deficit
  - hard: bias[a] = -large_negative if deficit > 0 else 0
"""
import numpy as np

try:
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False
    jnp = None


def compute_logit_bias(context, rules, num_actions, lambda_penalty=5.0,
                       large_negative=1e9, mode="soft"):
    """
    Compute per-action bias from context and rules.

    Args:
        context: dict with wood, stone, near_table, near_furnace (scalars or arrays)
        rules: BASIC_ACTION_RULES or similar
        num_actions: int
        lambda_penalty: for soft mode
        large_negative: for hard mode
        mode: "none" | "soft" | "hard"

    Returns:
        bias: array of shape (num_actions,) or (B, num_actions) if context is batched
    """
    if mode == "none":
        if _HAS_JAX:
            return jnp.zeros(num_actions, dtype=jnp.float32)
        return np.zeros(num_actions, dtype=np.float32)

    use_jax = _HAS_JAX
    xp = jnp if use_jax else np

    wood = xp.asarray(context["wood"])
    stone = xp.asarray(context["stone"])
    near_table = xp.asarray(context["near_table"])
    near_furnace = xp.asarray(context["near_furnace"])

    # Infer batch: scalar -> batch=1
    batch = 1
    if hasattr(wood, "ndim") and wood.ndim > 0 and wood.size > 1:
        batch = wood.shape[0]
    if batch == 1 and hasattr(wood, "ndim") and wood.ndim > 0:
        wood = xp.squeeze(wood)
        stone = xp.squeeze(stone)
        near_table = xp.squeeze(near_table)
        near_furnace = xp.squeeze(near_furnace)

    bias = xp.zeros((num_actions,) if batch == 1 else (batch, num_actions), dtype=xp.float32)

    for name, rule in rules.items():
        aid = rule["action_id"]
        if aid >= num_actions:
            continue

        # Resource deficit
        deficit = xp.zeros((), dtype=xp.float32) if batch == 1 else xp.zeros(batch, dtype=xp.float32)
        inv = {"wood": wood, "stone": stone, "coal": context.get("coal", 0), "iron": context.get("iron", 0)}
        for res, req in rule["requires"].items():
            cur = inv.get(res, 0)
            if use_jax:
                cur = xp.asarray(cur)
            deficit = deficit + xp.maximum(0, req - cur)

        if rule["needs_table"]:
            deficit = deficit + xp.where(near_table, 0.0, 1.0)
        if rule["needs_furnace"]:
            deficit = deficit + xp.where(near_furnace, 0.0, 1.0)

        if mode == "soft":
            pen = -lambda_penalty * deficit
        else:
            pen = xp.where(deficit > 0, -large_negative, 0.0)

        if use_jax:
            if batch == 1:
                bias = bias.at[aid].set(pen)
            else:
                bias = bias.at[:, aid].set(pen)
        else:
            if batch == 1:
                bias[aid] = pen
            else:
                bias[:, aid] = pen

    return bias

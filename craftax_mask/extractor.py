"""
Extract action mask context from Craftax EnvState.

Uses structure from craftax/craftax/craftax_state.py:
  - EnvState.inventory: Inventory(wood, stone, coal, iron, ...)
  - EnvState.player_position: jnp.ndarray (x, y)
  - EnvState.player_level: int
  - EnvState.map: jnp.ndarray (num_levels, h, w) - map[level] gives current level tiles
"""
import numpy as np

try:
    import jax.numpy as jnp
    from craftax.craftax.constants import BlockType
    _HAS_CRAFTAX = True
except ImportError:
    _HAS_CRAFTAX = False
    jnp = None
    BlockType = None


def extract_mask_context(state):
    """
    Extract wood, stone, near_table, near_furnace from Craftax EnvState.

    Returns dict with keys: wood, stone, coal, iron, near_table, near_furnace.
    Values are numpy/jax scalars (batch-safe: shape () or (B,)).
    """
    if not _HAS_CRAFTAX:
        return _empty_context()

    try:
        wood = state.inventory.wood
        stone = state.inventory.stone
        coal = getattr(state.inventory, "coal", np.int32(0))
        iron = getattr(state.inventory, "iron", np.int32(0))
    except Exception:
        return _empty_context()

    try:
        # player_position is jnp.ndarray; convert to int for slice
        px = int(np.asarray(state.player_position[0]).item())
        py = int(np.asarray(state.player_position[1]).item())
        level = int(state.player_level)
        map_arr = state.map[level]  # (h, w) tiles for current level
    except Exception:
        return {
            "wood": _to_scalar(wood),
            "stone": _to_scalar(stone),
            "coal": _to_scalar(coal),
            "iron": _to_scalar(iron),
            "near_table": np.array(False),
            "near_furnace": np.array(False),
        }

    h, w = map_arr.shape[0], map_arr.shape[1]
    # 5x5 local window, clamp to map bounds
    start_x = int(np.clip(px - 2, 0, max(0, h - 5)))
    start_y = int(np.clip(py - 2, 0, max(0, w - 5)))
    end_x = min(start_x + 5, h)
    end_y = min(start_y + 5, w)
    local_map = map_arr[start_x:end_x, start_y:end_y]

    craft_table_id = int(BlockType.CRAFTING_TABLE.value)
    furnace_id = int(BlockType.FURNACE.value)
    near_table = np.any(np.asarray(local_map) == craft_table_id)
    near_furnace = np.any(np.asarray(local_map) == furnace_id)

    return {
        "wood": _to_scalar(wood),
        "stone": _to_scalar(stone),
        "coal": _to_scalar(coal),
        "iron": _to_scalar(iron),
        "near_table": np.array(near_table, dtype=bool),
        "near_furnace": np.array(near_furnace, dtype=bool),
    }


def _to_scalar(x):
    if hasattr(x, "item"):
        return np.int32(x.item()) if hasattr(x, "ndim") and x.ndim == 0 else x
    return np.int32(x)


def _empty_context():
    return {
        "wood": np.int32(0),
        "stone": np.int32(0),
        "coal": np.int32(0),
        "iron": np.int32(0),
        "near_table": np.array(True),  # assume OK if we can't extract
        "near_furnace": np.array(True),
    }

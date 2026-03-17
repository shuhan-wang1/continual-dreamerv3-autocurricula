"""Minimal state extraction for action-feasibility masking in Craftax."""

from __future__ import annotations

import numpy as np

WINDOW_RADIUS = 2


def extract_mask_context(state):
    """Extract wood / stone counts and nearby station booleans from EnvState."""
    inventory = _get_field(state, ("inventory",))
    if inventory is None:
        return None

    wood = _inventory_value(inventory, "wood")
    stone = _inventory_value(inventory, "stone")
    coal = _inventory_value(inventory, "coal", default=0)
    iron = _inventory_value(inventory, "iron", default=0)

    if wood is None or stone is None:
        return None

    level_map = _extract_level_map(state)
    player_pos = _extract_player_position(state)
    near_table = False
    near_furnace = False
    if level_map is not None and player_pos is not None:
        local_map = _local_window(level_map, player_pos, WINDOW_RADIUS)
        table_id = _resolve_block_id("CRAFTING_TABLE")
        furnace_id = _resolve_block_id("FURNACE")
        local_np = np.asarray(local_map)
        if table_id is not None:
            near_table = bool(np.any(local_np == table_id))
        if furnace_id is not None:
            near_furnace = bool(np.any(local_np == furnace_id))

    return {
        "wood": np.int32(wood),
        "stone": np.int32(stone),
        "coal": np.int32(coal if coal is not None else 0),
        "iron": np.int32(iron if iron is not None else 0),
        "near_table": np.array(near_table, dtype=bool),
        "near_furnace": np.array(near_furnace, dtype=bool),
    }


def _get_field(obj, names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _inventory_value(inventory, name, default=None):
    value = _get_field(inventory, (name,))
    if value is None:
        return default
    try:
        return int(np.asarray(value).item())
    except ValueError:
        arr = np.asarray(value).reshape(-1)
        if len(arr):
            return int(arr[0])
        return default


def _extract_player_position(state):
    value = _get_field(state, ("player_position", "player_pos", "position"))
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size < 2:
        return None
    return int(arr[0]), int(arr[1])


def _extract_level_map(state):
    map_value = _get_field(state, ("map", "block_map", "tile_map"))
    if map_value is None:
        return None
    arr = np.asarray(map_value)
    if arr.ndim == 2:
        return arr
    if arr.ndim >= 3:
        level = _get_field(state, ("player_level", "current_level", "level"))
        level_idx = int(np.asarray(level).item()) if level is not None else 0
        level_idx = int(np.clip(level_idx, 0, arr.shape[0] - 1))
        return arr[level_idx]
    return None


def _local_window(level_map, position, radius):
    px, py = position
    h, w = level_map.shape[:2]
    x0 = max(0, px - radius)
    x1 = min(h, px + radius + 1)
    y0 = max(0, py - radius)
    y1 = min(w, py + radius + 1)
    return level_map[x0:x1, y0:y1]


def _resolve_block_id(name):
    enum_types = ()
    try:
        from craftax.craftax.constants import BlockType as CraftaxBlockType
        enum_types += (CraftaxBlockType,)
    except ImportError:
        pass
    try:
        from craftax.craftax_classic.constants import BlockType as CraftaxClassicBlockType
        enum_types += (CraftaxClassicBlockType,)
    except ImportError:
        pass

    for enum_type in enum_types:
        member = getattr(enum_type, name, None)
        if member is not None:
            return int(member.value)
    return None

"""Extract action-mask context from Craftax EnvState into a flat array.

Returns a float32 array of shape (CONTEXT_SIZE,) matching the schema in rules.py.
"""

from __future__ import annotations

import numpy as np

from .rules import CONTEXT_SIZE, C

# 8-cell adjacency matching game's CLOSE_BLOCKS (game_logic_utils.py:347-358)
_CLOSE_OFFSETS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),   # cardinal
    (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonal
]

# Player direction -> row/col offset (constants.py DIRECTIONS)
# 0=NOOP(0,0), 1=LEFT(0,-1), 2=RIGHT(0,1), 3=UP(-1,0), 4=DOWN(1,0)
_DIRECTIONS = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

# BlockType IDs resolved at import time, with fallback defaults
_BLOCK_IDS = {}
_SOLID_SET = set()
_CAN_PLACE_ITEM_SET = set()

try:
    from craftax.craftax.constants import BlockType
    _BLOCK_IDS = {bt.name: int(bt.value) for bt in BlockType}
    # Solid blocks that prevent placement (from SOLID_BLOCK_MAPPING in constants.py)
    _SOLID_NAMES = {
        'STONE', 'TREE', 'COAL', 'IRON', 'DIAMOND',
        'CRAFTING_TABLE', 'FURNACE', 'PLANT', 'RIPE_PLANT', 'WALL',
        'WALL_MOSS', 'STALAGMITE', 'RUBY', 'SAPPHIRE', 'CHEST',
        'FOUNTAIN', 'FIRE_TREE', 'ENCHANTMENT_TABLE_FIRE',
        'ENCHANTMENT_TABLE_ICE', 'GRAVE', 'GRAVE2', 'GRAVE3',
        'NECROMANCER',
    }
    _SOLID_SET = {_BLOCK_IDS[n] for n in _SOLID_NAMES if n in _BLOCK_IDS}
    # Blocks that can receive a torch (CAN_PLACE_ITEM_BLOCKS)
    _TORCH_NAMES = {'GRASS', 'SAND', 'PATH', 'FIRE_GRASS', 'ICE_GRASS'}
    _CAN_PLACE_ITEM_SET = {_BLOCK_IDS[n] for n in _TORCH_NAMES if n in _BLOCK_IDS}
except ImportError:
    pass

try:
    from craftax.craftax.constants import ItemType
    _ITEM_NONE = int(ItemType.NONE.value) if hasattr(ItemType, 'NONE') else 0
    _ITEM_LADDER_DOWN = int(ItemType.LADDER_DOWN.value)
    _ITEM_LADDER_UP = int(ItemType.LADDER_UP.value)
except ImportError:
    _ITEM_NONE = 0
    _ITEM_LADDER_DOWN = 2
    _ITEM_LADDER_UP = 3

_ENCHANT_TABLE_IDS = set()
for _n in ('ENCHANTMENT_TABLE_FIRE', 'ENCHANTMENT_TABLE_ICE'):
    if _n in _BLOCK_IDS:
        _ENCHANT_TABLE_IDS.add(_BLOCK_IDS[_n])

_GRASS_ID = _BLOCK_IDS.get('GRASS', None)
_CRAFTING_TABLE_ID = _BLOCK_IDS.get('CRAFTING_TABLE', 11)
_FURNACE_ID = _BLOCK_IDS.get('FURNACE', 12)
_MONSTERS_TO_CLEAR = 8
_MAX_PROJECTILES = 3


def extract_mask_context(state) -> np.ndarray:
    """Extract a flat float32 context array of shape (CONTEXT_SIZE,) from EnvState."""
    ctx = np.zeros(CONTEXT_SIZE, dtype=np.float32)

    # --- Inventory ---
    inv = getattr(state, 'inventory', None)
    if inv is None:
        return ctx  # fallback: all zeros (maximally restrictive)

    ctx[C.WOOD] = _scalar(getattr(inv, 'wood', 0))
    ctx[C.STONE] = _scalar(getattr(inv, 'stone', 0))
    ctx[C.COAL] = _scalar(getattr(inv, 'coal', 0))
    ctx[C.IRON] = _scalar(getattr(inv, 'iron', 0))
    ctx[C.DIAMOND] = _scalar(getattr(inv, 'diamond', 0))
    ctx[C.SAPLING] = _scalar(getattr(inv, 'sapling', 0))
    ctx[C.PICKAXE] = _scalar(getattr(inv, 'pickaxe', 0))
    ctx[C.SWORD] = _scalar(getattr(inv, 'sword', 0))
    ctx[C.BOW] = _scalar(getattr(inv, 'bow', 0))
    ctx[C.ARROWS] = _scalar(getattr(inv, 'arrows', 0))
    ctx[C.TORCHES] = _scalar(getattr(inv, 'torches', 0))
    ctx[C.BOOKS] = _scalar(getattr(inv, 'books', 0))
    ctx[C.RUBY] = _scalar(getattr(inv, 'ruby', 0))
    ctx[C.SAPPHIRE] = _scalar(getattr(inv, 'sapphire', 0))

    # Armour: shape (4,)
    armour = getattr(inv, 'armour', None)
    if armour is not None:
        a = np.asarray(armour).flatten()[:4]
        ctx[C.ARMOUR_0:C.ARMOUR_0 + len(a)] = a

    # Potions: shape (6,)
    potions = getattr(inv, 'potions', None)
    if potions is not None:
        p = np.asarray(potions).flatten()[:6]
        ctx[C.POTION_0:C.POTION_0 + len(p)] = p

    # --- Player state ---
    ctx[C.HEALTH] = _scalar(getattr(state, 'player_health', 0))
    ctx[C.ENERGY] = _scalar(getattr(state, 'player_energy', 0))
    ctx[C.MANA] = _scalar(getattr(state, 'player_mana', 0))
    ctx[C.XP] = _scalar(getattr(state, 'player_xp', 0))

    level = _scalar(getattr(state, 'player_level', 0))
    ctx[C.LEVEL] = level
    level_idx = int(np.clip(level, 0, 8))  # will be bounds-checked against map shape below

    ctx[C.DEXTERITY] = _scalar(getattr(state, 'player_dexterity', 1))
    ctx[C.STRENGTH] = _scalar(getattr(state, 'player_strength', 1))
    ctx[C.INTELLIGENCE] = _scalar(getattr(state, 'player_intelligence', 1))

    # Learned spells: shape (2,)
    spells = getattr(state, 'learned_spells', None)
    if spells is not None:
        s = np.asarray(spells).flatten()[:2]
        ctx[C.SPELL_FIREBALL] = float(s[0]) if len(s) > 0 else 0.0
        ctx[C.SPELL_ICEBALL] = float(s[1]) if len(s) > 1 else 0.0

    # --- Map-based checks ---
    player_pos = getattr(state, 'player_position', None)
    if player_pos is None:
        return ctx
    pos = np.asarray(player_pos).flatten()
    if pos.size < 2:
        return ctx
    px, py = int(pos[0]), int(pos[1])

    player_dir = int(_scalar(getattr(state, 'player_direction', 0)))

    map_arr = getattr(state, 'map', None)
    if map_arr is not None:
        full_map = np.asarray(map_arr)
        if full_map.ndim >= 3:
            if level_idx >= full_map.shape[0]:
                level_idx = full_map.shape[0] - 1
            level_map = full_map[level_idx]
        elif full_map.ndim == 2:
            level_map = full_map
        else:
            level_map = None

        if level_map is not None:
            h, w = level_map.shape[:2]

            # Proximity: exact 8-cell adjacency (matches CLOSE_BLOCKS)
            near_table = False
            near_furnace = False
            for dr, dc in _CLOSE_OFFSETS:
                nr, nc = px + dr, py + dc
                if 0 <= nr < h and 0 <= nc < w:
                    block = int(level_map[nr, nc])
                    if block == _CRAFTING_TABLE_ID:
                        near_table = True
                    if block == _FURNACE_ID:
                        near_furnace = True
            ctx[C.NEAR_TABLE] = float(near_table)
            ctx[C.NEAR_FURNACE] = float(near_furnace)

            # Facing block checks
            dr, dc = _DIRECTIONS.get(player_dir, (0, 0))
            fr, fc = px + dr, py + dc
            if 0 <= fr < h and 0 <= fc < w:
                facing_block = int(level_map[fr, fc])
                facing_is_solid = facing_block in _SOLID_SET

                # Check item_map for items at facing position
                item_map = getattr(state, 'item_map', None)
                facing_has_item = False
                if item_map is not None:
                    im = np.asarray(item_map)
                    if im.ndim >= 3 and level_idx < im.shape[0]:
                        facing_has_item = int(im[level_idx, fr, fc]) != _ITEM_NONE
                    elif im.ndim == 2:
                        facing_has_item = int(im[fr, fc]) != _ITEM_NONE

                ctx[C.FACING_PLACEABLE] = float(not facing_is_solid and not facing_has_item)
                ctx[C.FACING_GRASS] = float(
                    _GRASS_ID is not None and facing_block == _GRASS_ID and not facing_has_item)
                ctx[C.FACING_ENCHANT_TABLE] = float(facing_block in _ENCHANT_TABLE_IDS)
                ctx[C.FACING_FIRE_TABLE] = float(
                    'ENCHANTMENT_TABLE_FIRE' in _BLOCK_IDS
                    and facing_block == _BLOCK_IDS['ENCHANTMENT_TABLE_FIRE'])
                ctx[C.FACING_ICE_TABLE] = float(
                    'ENCHANTMENT_TABLE_ICE' in _BLOCK_IDS
                    and facing_block == _BLOCK_IDS['ENCHANTMENT_TABLE_ICE'])
                ctx[C.FACING_TORCH_PLACEABLE] = float(
                    facing_block in _CAN_PLACE_ITEM_SET and not facing_has_item)

    # --- Ladder checks ---
    item_map = getattr(state, 'item_map', None)
    if item_map is not None:
        im = np.asarray(item_map)
        item_at_player = _ITEM_NONE
        if im.ndim >= 3 and level_idx < im.shape[0]:
            if 0 <= px < im.shape[1] and 0 <= py < im.shape[2]:
                item_at_player = int(im[level_idx, px, py])
        elif im.ndim == 2:
            if 0 <= px < im.shape[0] and 0 <= py < im.shape[1]:
                item_at_player = int(im[px, py])
        ctx[C.ON_LADDER_DOWN] = float(item_at_player == _ITEM_LADDER_DOWN)
        ctx[C.ON_LADDER_UP] = float(item_at_player == _ITEM_LADDER_UP)

    # --- Level cleared ---
    monsters_killed = getattr(state, 'monsters_killed', None)
    if monsters_killed is not None:
        mk = np.asarray(monsters_killed).flatten()
        if level_idx < len(mk):
            ctx[C.LEVEL_CLEARED] = float(int(mk[level_idx]) >= _MONSTERS_TO_CLEAR)

    # --- Projectile slots ---
    proj = getattr(state, 'player_projectiles', None)
    if proj is not None:
        mask = getattr(proj, 'mask', None)
        if mask is not None:
            m = np.asarray(mask)
            if m.ndim >= 2 and level_idx < m.shape[0]:
                active = int(m[level_idx].sum())
                ctx[C.PROJECTILE_SLOTS] = float(active < _MAX_PROJECTILES)

    return ctx


def _scalar(x) -> float:
    """Convert a JAX/numpy scalar to Python float."""
    try:
        return float(np.asarray(x).item())
    except (ValueError, TypeError):
        arr = np.asarray(x).flatten()
        return float(arr[0]) if arr.size > 0 else 0.0

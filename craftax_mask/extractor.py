"""Extract action-mask context from Craftax EnvState into a flat array.

Returns a float32 array of shape (CONTEXT_SIZE,) matching the schema in rules.py.
"""

from __future__ import annotations

import warnings

import numpy as np

from .rules import CONTEXT_SIZE, C

# 8-cell adjacency matching game's CLOSE_BLOCKS (game_logic_utils.py:347-358)
_CLOSE_OFFSETS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),   # cardinal
    (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonal
]
_CLOSE_OFFSETS_ARR = np.array(_CLOSE_OFFSETS, dtype=np.int32)  # (8, 2) for vectorized use

# Player direction -> row/col offset (constants.py DIRECTIONS)
# 0=NOOP(0,0), 1=LEFT(0,-1), 2=RIGHT(0,1), 3=UP(-1,0), 4=DOWN(1,0)
_DIRECTIONS = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

# BlockType IDs resolved at import time, with fallback defaults
_BLOCK_IDS = {}
_SOLID_SET = set()
_CAN_PLACE_ITEM_SET = set()
_CRAFTAX_AVAILABLE = False

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
    _CRAFTAX_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Craftax not installed — action mask context extraction will produce incorrect results",
        RuntimeWarning,
    )

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

_CRAFTING_TABLE_ID = _BLOCK_IDS.get('CRAFTING_TABLE', None)
if _CRAFTING_TABLE_ID is None:
    _CRAFTING_TABLE_ID = 11
    warnings.warn("Using hardcoded CRAFTING_TABLE block ID=11", RuntimeWarning)

_FURNACE_ID = _BLOCK_IDS.get('FURNACE', None)
if _FURNACE_ID is None:
    _FURNACE_ID = 12
    warnings.warn("Using hardcoded FURNACE block ID=12", RuntimeWarning)
_MONSTERS_TO_CLEAR = 8
_MAX_PROJECTILES = 3


def extract_mask_context(state) -> np.ndarray:
    """Extract a flat float32 context array of shape (CONTEXT_SIZE,) from EnvState."""
    if not _CRAFTAX_AVAILABLE:
        raise RuntimeError("Craftax constants required for mask extraction")
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

    # Fetch item_map once for reuse in both facing checks and ladder checks
    item_map = getattr(state, 'item_map', None)
    im = np.asarray(item_map) if item_map is not None else None

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
            # Vectorized: compute all 8 neighbor positions at once
            neighbor_pos = _CLOSE_OFFSETS_ARR + np.array([px, py], dtype=np.int32)  # (8, 2)
            valid = (
                (neighbor_pos[:, 0] >= 0) & (neighbor_pos[:, 0] < h)
                & (neighbor_pos[:, 1] >= 0) & (neighbor_pos[:, 1] < w)
            )
            valid_rows = neighbor_pos[valid, 0]
            valid_cols = neighbor_pos[valid, 1]
            neighbor_blocks = level_map[valid_rows, valid_cols]
            ctx[C.NEAR_TABLE] = float(np.any(neighbor_blocks == _CRAFTING_TABLE_ID))
            ctx[C.NEAR_FURNACE] = float(np.any(neighbor_blocks == _FURNACE_ID))

            # Facing block checks
            if player_dir == 0:
                # NOOP direction: no valid facing tile
                facing_block = -1
                facing_is_solid = False
                facing_has_item = False
                ctx[C.FACING_PLACEABLE] = 0.0
            else:
                dr, dc = _DIRECTIONS.get(player_dir, (0, 0))
                fr, fc = px + dr, py + dc
                if 0 <= fr < h and 0 <= fc < w:
                    facing_block = int(level_map[fr, fc])
                    facing_is_solid = facing_block in _SOLID_SET

                    # Check item_map for items at facing position
                    facing_has_item = False
                    if im is not None:
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
    if im is not None:
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
            elif m.ndim == 1:
                active = int(m.sum())
                ctx[C.PROJECTILE_SLOTS] = float(active < _MAX_PROJECTILES)

    return ctx


def _scalar(x) -> float:
    """Convert a JAX/numpy scalar to Python float."""
    try:
        return float(np.asarray(x).item())
    except (ValueError, TypeError):
        arr = np.asarray(x).flatten()
        if arr.size > 0:
            return float(arr[0])
        warnings.warn(f"_scalar received unconvertible value: {type(x)}", RuntimeWarning)
        return 0.0


# ===================================================================
# JAX-batched extraction — runs entirely on GPU, zero CPU round-trips
# ===================================================================

def _build_block_lut(block_id_set, max_id):
    """Build a float32 lookup table: lut[block_id] = 1.0 if id in set."""
    lut = np.zeros(max_id, dtype=np.float32)
    for bid in block_id_set:
        if 0 <= bid < max_id:
            lut[bid] = 1.0
    return lut


# Lazily-initialized JAX constant arrays (built on first call)
_jax_consts = None


def _get_jax_consts():
    """Build and cache JAX constant arrays for batched extraction."""
    global _jax_consts
    if _jax_consts is not None:
        return _jax_consts

    import jax.numpy as jnp

    max_id = (max(_BLOCK_IDS.values()) + 1) if _BLOCK_IDS else 1
    max_id = max(max_id, 64)  # minimum LUT size

    fire_id = _BLOCK_IDS.get('ENCHANTMENT_TABLE_FIRE', -1)
    ice_id = _BLOCK_IDS.get('ENCHANTMENT_TABLE_ICE', -1)

    _jax_consts = {
        'close_offsets': jnp.array(_CLOSE_OFFSETS_ARR, dtype=jnp.int32),
        'dir_offsets': jnp.array([
            [0, 0],   # 0=NOOP
            [0, -1],  # 1=LEFT
            [0, 1],   # 2=RIGHT
            [-1, 0],  # 3=UP
            [1, 0],   # 4=DOWN
        ], dtype=jnp.int32),
        'solid_lut': jnp.array(
            _build_block_lut(_SOLID_SET, max_id)),
        'can_place_lut': jnp.array(
            _build_block_lut(_CAN_PLACE_ITEM_SET, max_id)),
        'enchant_lut': jnp.array(
            _build_block_lut(_ENCHANT_TABLE_IDS, max_id)),
        'fire_lut': jnp.array(
            _build_block_lut({fire_id} if fire_id >= 0 else set(), max_id)),
        'ice_lut': jnp.array(
            _build_block_lut({ice_id} if ice_id >= 0 else set(), max_id)),
        'grass_id': _GRASS_ID,
        'max_id': max_id,
    }
    return _jax_consts


def extract_mask_context_batch_jax(states):
    """Extract mask context for all envs using pure JAX ops on device.

    Replaces the per-env Python loop + GPU→CPU transfers with a single
    vectorized computation that stays entirely on GPU.

    Args:
        states: Vectorized Craftax EnvState with leading (N,) dim on all leaves.

    Returns:
        jnp.ndarray of shape (N, CONTEXT_SIZE), float32, on device.
    """
    import jax.numpy as jnp

    c = _get_jax_consts()
    inv = states.inventory
    N = states.player_health.shape[0]
    ctx = jnp.zeros((N, CONTEXT_SIZE), dtype=jnp.float32)

    # --- Inventory (indices 0-13) ---
    ctx = ctx.at[:, C.WOOD].set(inv.wood.astype(jnp.float32))
    ctx = ctx.at[:, C.STONE].set(inv.stone.astype(jnp.float32))
    ctx = ctx.at[:, C.COAL].set(inv.coal.astype(jnp.float32))
    ctx = ctx.at[:, C.IRON].set(inv.iron.astype(jnp.float32))
    ctx = ctx.at[:, C.DIAMOND].set(inv.diamond.astype(jnp.float32))
    ctx = ctx.at[:, C.SAPLING].set(inv.sapling.astype(jnp.float32))
    ctx = ctx.at[:, C.PICKAXE].set(inv.pickaxe.astype(jnp.float32))
    ctx = ctx.at[:, C.SWORD].set(inv.sword.astype(jnp.float32))
    ctx = ctx.at[:, C.BOW].set(inv.bow.astype(jnp.float32))
    ctx = ctx.at[:, C.ARROWS].set(inv.arrows.astype(jnp.float32))
    ctx = ctx.at[:, C.TORCHES].set(inv.torches.astype(jnp.float32))
    ctx = ctx.at[:, C.BOOKS].set(inv.books.astype(jnp.float32))
    ctx = ctx.at[:, C.RUBY].set(inv.ruby.astype(jnp.float32))
    ctx = ctx.at[:, C.SAPPHIRE].set(inv.sapphire.astype(jnp.float32))

    # --- Armour (indices 14-17) ---
    armour = inv.armour.astype(jnp.float32)              # (N, 4)
    ctx = ctx.at[:, C.ARMOUR_0:C.ARMOUR_0 + 4].set(armour[:, :4])

    # --- Potions (indices 18-23) ---
    potions = inv.potions.astype(jnp.float32)             # (N, 6)
    ctx = ctx.at[:, C.POTION_0:C.POTION_0 + 6].set(potions[:, :6])

    # --- Player state (indices 24-33) ---
    ctx = ctx.at[:, C.HEALTH].set(
        jnp.asarray(states.player_health, dtype=jnp.float32))
    ctx = ctx.at[:, C.ENERGY].set(
        jnp.asarray(states.player_energy, dtype=jnp.float32))
    ctx = ctx.at[:, C.MANA].set(
        jnp.asarray(states.player_mana, dtype=jnp.float32))
    ctx = ctx.at[:, C.XP].set(
        jnp.asarray(states.player_xp, dtype=jnp.float32))
    ctx = ctx.at[:, C.LEVEL].set(
        jnp.asarray(states.player_level, dtype=jnp.float32))
    ctx = ctx.at[:, C.DEXTERITY].set(
        jnp.asarray(states.player_dexterity, dtype=jnp.float32))
    ctx = ctx.at[:, C.STRENGTH].set(
        jnp.asarray(states.player_strength, dtype=jnp.float32))
    ctx = ctx.at[:, C.INTELLIGENCE].set(
        jnp.asarray(states.player_intelligence, dtype=jnp.float32))

    # --- Learned spells (indices 32-33) ---
    spells = jnp.asarray(states.learned_spells, dtype=jnp.float32)  # (N, 2)
    ctx = ctx.at[:, C.SPELL_FIREBALL].set(spells[:, 0])
    ctx = ctx.at[:, C.SPELL_ICEBALL].set(spells[:, 1])

    # --- Map-based checks ---
    game_map = states.map                                  # (N, L, H, W)
    item_map_full = states.item_map                        # (N, L, H, W)
    L = game_map.shape[1]
    H, W = game_map.shape[2], game_map.shape[3]

    level_idx = jnp.clip(
        jnp.asarray(states.player_level, dtype=jnp.int32), 0, L - 1)   # (N,)
    pos = jnp.asarray(states.player_position, dtype=jnp.int32)         # (N, 2)
    px, py = pos[:, 0], pos[:, 1]
    env_idx = jnp.arange(N)

    # Per-env level slices
    level_maps = game_map[env_idx, level_idx]              # (N, H, W)
    item_maps = item_map_full[env_idx, level_idx]          # (N, H, W)

    # --- Proximity: 8-cell adjacency ---
    nbr = pos[:, None, :] + c['close_offsets'][None, :, :]  # (N, 8, 2)
    nbr_valid = (
        (nbr[:, :, 0] >= 0) & (nbr[:, :, 0] < H) &
        (nbr[:, :, 1] >= 0) & (nbr[:, :, 1] < W)
    )                                                       # (N, 8)
    nr = jnp.clip(nbr[:, :, 0], 0, H - 1)
    nc = jnp.clip(nbr[:, :, 1], 0, W - 1)
    nblocks = level_maps[env_idx[:, None], nr, nc]          # (N, 8)

    ctx = ctx.at[:, C.NEAR_TABLE].set(
        jnp.any(nbr_valid & (nblocks == _CRAFTING_TABLE_ID), axis=1)
        .astype(jnp.float32))
    ctx = ctx.at[:, C.NEAR_FURNACE].set(
        jnp.any(nbr_valid & (nblocks == _FURNACE_ID), axis=1)
        .astype(jnp.float32))

    # --- Facing block ---
    dirs = jnp.asarray(states.player_direction, dtype=jnp.int32)  # (N,)
    # Clamp to valid index range [0, 4]
    dirs_safe = jnp.clip(dirs, 0, 4)
    doff = c['dir_offsets'][dirs_safe]                     # (N, 2)
    fr, fc = px + doff[:, 0], py + doff[:, 1]
    fvalid = (dirs > 0) & (fr >= 0) & (fr < H) & (fc >= 0) & (fc < W)
    fr_s = jnp.clip(fr, 0, H - 1)
    fc_s = jnp.clip(fc, 0, W - 1)
    fblock = level_maps[env_idx, fr_s, fc_s]               # (N,)
    fb_s = jnp.clip(fblock, 0, c['max_id'] - 1)            # safe LUT index

    f_solid = c['solid_lut'][fb_s] > 0.5                    # (N,)
    f_item = item_maps[env_idx, fr_s, fc_s]                 # (N,)
    f_has_item = f_item != _ITEM_NONE

    ctx = ctx.at[:, C.FACING_PLACEABLE].set(
        (fvalid & ~f_solid & ~f_has_item).astype(jnp.float32))

    if c['grass_id'] is not None:
        ctx = ctx.at[:, C.FACING_GRASS].set(
            (fvalid & (fblock == c['grass_id']) & ~f_has_item)
            .astype(jnp.float32))

    ctx = ctx.at[:, C.FACING_ENCHANT_TABLE].set(
        (fvalid & (c['enchant_lut'][fb_s] > 0.5)).astype(jnp.float32))
    ctx = ctx.at[:, C.FACING_FIRE_TABLE].set(
        (fvalid & (c['fire_lut'][fb_s] > 0.5)).astype(jnp.float32))
    ctx = ctx.at[:, C.FACING_ICE_TABLE].set(
        (fvalid & (c['ice_lut'][fb_s] > 0.5)).astype(jnp.float32))
    ctx = ctx.at[:, C.FACING_TORCH_PLACEABLE].set(
        (fvalid & (c['can_place_lut'][fb_s] > 0.5) & ~f_has_item)
        .astype(jnp.float32))

    # --- Ladder checks ---
    px_s = jnp.clip(px, 0, H - 1)
    py_s = jnp.clip(py, 0, W - 1)
    p_item = item_maps[env_idx, px_s, py_s]                 # (N,)
    ctx = ctx.at[:, C.ON_LADDER_DOWN].set(
        (p_item == _ITEM_LADDER_DOWN).astype(jnp.float32))
    ctx = ctx.at[:, C.ON_LADDER_UP].set(
        (p_item == _ITEM_LADDER_UP).astype(jnp.float32))

    # --- Level cleared ---
    mk = jnp.asarray(states.monsters_killed, dtype=jnp.float32)  # (N, L)
    ctx = ctx.at[:, C.LEVEL_CLEARED].set(
        (mk[env_idx, level_idx] >= _MONSTERS_TO_CLEAR).astype(jnp.float32))

    # --- Projectile slots ---
    pm = states.player_projectiles.mask                     # (N, L, P) or (N, P)
    if pm.ndim >= 3:
        lp = pm[env_idx, level_idx]                         # (N, P)
    else:
        lp = pm                                             # (N, P)
    ctx = ctx.at[:, C.PROJECTILE_SLOTS].set(
        (lp.astype(jnp.float32).sum(axis=-1) < _MAX_PROJECTILES)
        .astype(jnp.float32))

    return ctx

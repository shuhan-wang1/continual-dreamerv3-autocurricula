"""Craftax action feasibility rules for all maskable actions (6-42).

Context array schema (CONTEXT_SIZE = 46 float32 values):
  Indices  0-13: inventory [wood, stone, coal, iron, diamond, sapling,
                            pickaxe, sword, bow, arrows, torches, books,
                            ruby, sapphire]
  Indices 14-17: armour[0..3]
  Indices 18-23: potions[0..5]
  Indices 24-28: player state [health, energy, mana, xp, level]
  Indices 29-31: attributes [dexterity, strength, intelligence]
  Indices 32-33: learned_spells [fireball, iceball]
  Indices 34-35: proximity [near_table, near_furnace]
  Indices 36-39: facing [placeable, grass, enchant_table, torch_placeable]
  Indices 40-41: ladder [on_ladder_down, on_ladder_up]
  Index    42:   level_cleared (monsters_killed >= 8)
  Index    43:   projectile_slots_available (active < 3)
"""

from __future__ import annotations

CONTEXT_SIZE = 46

# --- Named indices into the context array ---
class C:
    """Context array index constants."""
    WOOD = 0; STONE = 1; COAL = 2; IRON = 3; DIAMOND = 4; SAPLING = 5
    PICKAXE = 6; SWORD = 7; BOW = 8; ARROWS = 9; TORCHES = 10; BOOKS = 11
    RUBY = 12; SAPPHIRE = 13
    ARMOUR_0 = 14; ARMOUR_1 = 15; ARMOUR_2 = 16; ARMOUR_3 = 17
    POTION_0 = 18; POTION_1 = 19; POTION_2 = 20; POTION_3 = 21
    POTION_4 = 22; POTION_5 = 23
    HEALTH = 24; ENERGY = 25; MANA = 26; XP = 27; LEVEL = 28
    DEXTERITY = 29; STRENGTH = 30; INTELLIGENCE = 31
    SPELL_FIREBALL = 32; SPELL_ICEBALL = 33
    NEAR_TABLE = 34; NEAR_FURNACE = 35
    FACING_PLACEABLE = 36; FACING_GRASS = 37
    FACING_ENCHANT_TABLE = 38; FACING_TORCH_PLACEABLE = 39
    ON_LADDER_DOWN = 40; ON_LADDER_UP = 41
    LEVEL_CLEARED = 42; PROJECTILE_SLOTS = 43
    FACING_FIRE_TABLE = 44; FACING_ICE_TABLE = 45


def _resolve_action_id(name: str, default: int) -> int:
    """Read action ids from Craftax when available and fall back otherwise."""
    try:
        from craftax.craftax.constants import Action as CraftaxAction
        member = getattr(CraftaxAction, name, None)
        if member is not None:
            return int(member.value)
    except ImportError:
        pass
    return int(default)


# Condition types used in rules:
#   ("min", ctx_index, value)           context[idx] >= value
#   ("below", ctx_index, value)         context[idx] < value
#   ("bool", ctx_index)                 context[idx] > 0.5
#   ("any_below", start, end, value)    any(context[start:end] < value)
#   ("sum_pos", start, end)             sum(context[start:end]) > 0
#   ("max_energy_check",)               energy < 7 + 2*dexterity
#   ("max_health_check",)               health < 8 + strength
#   ("gem_check",)                      correct gem for the enchant table being faced
#   ("attr_below_max", ctx_index)       context[idx] < 5 (max_attribute)

ACTION_RULES = {
    # --- Sleep / Rest ---
    "SLEEP": {
        "action_id": _resolve_action_id("SLEEP", 6),
        "metric_suffix": "sleep",
        "conditions": [("max_energy_check",)],
    },
    "REST": {
        "action_id": _resolve_action_id("REST", 17),
        "metric_suffix": "rest",
        "conditions": [("max_health_check",)],
    },

    # --- Placement ---
    "PLACE_STONE": {
        "action_id": _resolve_action_id("PLACE_STONE", 7),
        "metric_suffix": "place_stone",
        "conditions": [("min", C.STONE, 1), ("bool", C.FACING_PLACEABLE)],
    },
    "PLACE_TABLE": {
        "action_id": _resolve_action_id("PLACE_TABLE", 8),
        "metric_suffix": "place_table",
        "conditions": [("min", C.WOOD, 2), ("bool", C.FACING_PLACEABLE)],
    },
    "PLACE_FURNACE": {
        "action_id": _resolve_action_id("PLACE_FURNACE", 9),
        "metric_suffix": "place_furnace",
        "conditions": [("min", C.STONE, 1), ("bool", C.FACING_PLACEABLE)],
    },
    "PLACE_PLANT": {
        "action_id": _resolve_action_id("PLACE_PLANT", 10),
        "metric_suffix": "place_plant",
        "conditions": [("min", C.SAPLING, 1), ("bool", C.FACING_GRASS)],
    },
    "PLACE_TORCH": {
        "action_id": _resolve_action_id("PLACE_TORCH", 28),
        "metric_suffix": "place_torch",
        "conditions": [("min", C.TORCHES, 1), ("bool", C.FACING_TORCH_PLACEABLE)],
    },

    # --- Pickaxe crafting ---
    "MAKE_WOOD_PICKAXE": {
        "action_id": _resolve_action_id("MAKE_WOOD_PICKAXE", 11),
        "metric_suffix": "make_wood_pickaxe",
        "conditions": [
            ("min", C.WOOD, 1), ("bool", C.NEAR_TABLE), ("below", C.PICKAXE, 1),
        ],
    },
    "MAKE_STONE_PICKAXE": {
        "action_id": _resolve_action_id("MAKE_STONE_PICKAXE", 12),
        "metric_suffix": "make_stone_pickaxe",
        "conditions": [
            ("min", C.WOOD, 1), ("min", C.STONE, 1),
            ("bool", C.NEAR_TABLE), ("below", C.PICKAXE, 2),
        ],
    },
    "MAKE_IRON_PICKAXE": {
        "action_id": _resolve_action_id("MAKE_IRON_PICKAXE", 13),
        "metric_suffix": "make_iron_pickaxe",
        "conditions": [
            ("min", C.WOOD, 1), ("min", C.STONE, 1),
            ("min", C.IRON, 1), ("min", C.COAL, 1),
            ("bool", C.NEAR_TABLE), ("bool", C.NEAR_FURNACE),
            ("below", C.PICKAXE, 3),
        ],
    },
    "MAKE_DIAMOND_PICKAXE": {
        "action_id": _resolve_action_id("MAKE_DIAMOND_PICKAXE", 20),
        "metric_suffix": "make_diamond_pickaxe",
        "conditions": [
            ("min", C.WOOD, 1), ("min", C.DIAMOND, 3),
            ("bool", C.NEAR_TABLE), ("below", C.PICKAXE, 4),
        ],
    },

    # --- Sword crafting ---
    "MAKE_WOOD_SWORD": {
        "action_id": _resolve_action_id("MAKE_WOOD_SWORD", 14),
        "metric_suffix": "make_wood_sword",
        "conditions": [
            ("min", C.WOOD, 1), ("bool", C.NEAR_TABLE), ("below", C.SWORD, 1),
        ],
    },
    "MAKE_STONE_SWORD": {
        "action_id": _resolve_action_id("MAKE_STONE_SWORD", 15),
        "metric_suffix": "make_stone_sword",
        "conditions": [
            ("min", C.WOOD, 1), ("min", C.STONE, 1),
            ("bool", C.NEAR_TABLE), ("below", C.SWORD, 2),
        ],
    },
    "MAKE_IRON_SWORD": {
        "action_id": _resolve_action_id("MAKE_IRON_SWORD", 16),
        "metric_suffix": "make_iron_sword",
        "conditions": [
            ("min", C.WOOD, 1), ("min", C.STONE, 1),
            ("min", C.IRON, 1), ("min", C.COAL, 1),
            ("bool", C.NEAR_TABLE), ("bool", C.NEAR_FURNACE),
            ("below", C.SWORD, 3),
        ],
    },
    "MAKE_DIAMOND_SWORD": {
        "action_id": _resolve_action_id("MAKE_DIAMOND_SWORD", 21),
        "metric_suffix": "make_diamond_sword",
        "conditions": [
            ("min", C.WOOD, 1), ("min", C.DIAMOND, 2),
            ("bool", C.NEAR_TABLE), ("below", C.SWORD, 4),
        ],
    },

    # --- Armour crafting ---
    "MAKE_IRON_ARMOUR": {
        "action_id": _resolve_action_id("MAKE_IRON_ARMOUR", 22),
        "metric_suffix": "make_iron_armour",
        "conditions": [
            ("min", C.IRON, 3), ("min", C.COAL, 3),
            ("bool", C.NEAR_TABLE), ("bool", C.NEAR_FURNACE),
            ("any_below", C.ARMOUR_0, C.ARMOUR_3 + 1, 1),
        ],
    },
    "MAKE_DIAMOND_ARMOUR": {
        "action_id": _resolve_action_id("MAKE_DIAMOND_ARMOUR", 23),
        "metric_suffix": "make_diamond_armour",
        "conditions": [
            ("min", C.DIAMOND, 3), ("bool", C.NEAR_TABLE),
            ("any_below", C.ARMOUR_0, C.ARMOUR_3 + 1, 2),
        ],
    },

    # --- Other crafting ---
    # NOTE: MAKE_BOW removed — bows are only obtainable from chests in Craftax,
    # and its action_id=29 collided with DRINK_POTION_RED.
    "MAKE_ARROW": {
        "action_id": _resolve_action_id("MAKE_ARROW", 25),
        "metric_suffix": "make_arrow",
        "conditions": [
            ("min", C.WOOD, 1), ("min", C.STONE, 1),
            ("bool", C.NEAR_TABLE), ("below", C.ARROWS, 99),
        ],
    },
    "MAKE_TORCH": {
        "action_id": _resolve_action_id("MAKE_TORCH", 38),
        "metric_suffix": "make_torch",
        "conditions": [
            ("min", C.WOOD, 1), ("min", C.COAL, 1),
            ("bool", C.NEAR_TABLE), ("below", C.TORCHES, 99),
        ],
    },

    # --- Ranged / Spells ---
    "SHOOT_ARROW": {
        "action_id": _resolve_action_id("SHOOT_ARROW", 24),
        "metric_suffix": "shoot_arrow",
        "conditions": [
            ("min", C.BOW, 1), ("min", C.ARROWS, 1),
            ("bool", C.PROJECTILE_SLOTS),
        ],
    },
    "CAST_FIREBALL": {
        "action_id": _resolve_action_id("CAST_FIREBALL", 26),
        "metric_suffix": "cast_fireball",
        "conditions": [
            ("min", C.MANA, 2), ("bool", C.SPELL_FIREBALL),
            ("bool", C.PROJECTILE_SLOTS),
        ],
    },
    "CAST_ICEBALL": {
        "action_id": _resolve_action_id("CAST_ICEBALL", 27),
        "metric_suffix": "cast_iceball",
        "conditions": [
            ("min", C.MANA, 2), ("bool", C.SPELL_ICEBALL),
            ("bool", C.PROJECTILE_SLOTS),
        ],
    },

    # --- Potions / Books ---
    "DRINK_POTION_RED": {
        "action_id": _resolve_action_id("DRINK_POTION_RED", 29),
        "metric_suffix": "potion_red",
        "conditions": [("min", C.POTION_0, 1)],
    },
    "DRINK_POTION_GREEN": {
        "action_id": _resolve_action_id("DRINK_POTION_GREEN", 30),
        "metric_suffix": "potion_green",
        "conditions": [("min", C.POTION_1, 1)],
    },
    "DRINK_POTION_BLUE": {
        "action_id": _resolve_action_id("DRINK_POTION_BLUE", 31),
        "metric_suffix": "potion_blue",
        "conditions": [("min", C.POTION_2, 1)],
    },
    "DRINK_POTION_PINK": {
        "action_id": _resolve_action_id("DRINK_POTION_PINK", 32),
        "metric_suffix": "potion_pink",
        "conditions": [("min", C.POTION_3, 1)],
    },
    "DRINK_POTION_CYAN": {
        "action_id": _resolve_action_id("DRINK_POTION_CYAN", 33),
        "metric_suffix": "potion_cyan",
        "conditions": [("min", C.POTION_4, 1)],
    },
    "DRINK_POTION_YELLOW": {
        "action_id": _resolve_action_id("DRINK_POTION_YELLOW", 34),
        "metric_suffix": "potion_yellow",
        "conditions": [("min", C.POTION_5, 1)],
    },
    "READ_BOOK": {
        "action_id": _resolve_action_id("READ_BOOK", 35),
        "metric_suffix": "read_book",
        "conditions": [("min", C.BOOKS, 1)],
    },

    # --- Enchanting ---
    "ENCHANT_SWORD": {
        "action_id": _resolve_action_id("ENCHANT_SWORD", 36),
        "metric_suffix": "enchant_sword",
        "conditions": [
            ("bool", C.FACING_ENCHANT_TABLE), ("min", C.MANA, 9),
            ("gem_check",), ("min", C.SWORD, 1),
        ],
    },
    "ENCHANT_ARMOUR": {
        "action_id": _resolve_action_id("ENCHANT_ARMOUR", 37),
        "metric_suffix": "enchant_armour",
        "conditions": [
            ("bool", C.FACING_ENCHANT_TABLE), ("min", C.MANA, 9),
            ("gem_check",), ("sum_pos", C.ARMOUR_0, C.ARMOUR_3 + 1),
        ],
    },
    "ENCHANT_BOW": {
        "action_id": _resolve_action_id("ENCHANT_BOW", 42),
        "metric_suffix": "enchant_bow",
        "conditions": [
            ("bool", C.FACING_ENCHANT_TABLE), ("min", C.MANA, 9),
            ("gem_check",), ("min", C.BOW, 1),
        ],
    },

    # --- Floor transitions ---
    "DESCEND": {
        "action_id": _resolve_action_id("DESCEND", 18),
        "metric_suffix": "descend",
        "conditions": [
            ("bool", C.ON_LADDER_DOWN), ("bool", C.LEVEL_CLEARED),
            ("below", C.LEVEL, 8),
        ],
    },
    "ASCEND": {
        "action_id": _resolve_action_id("ASCEND", 19),
        "metric_suffix": "ascend",
        "conditions": [("bool", C.ON_LADDER_UP), ("min", C.LEVEL, 1)],
    },

    # --- Level-up ---
    "LEVEL_UP_DEXTERITY": {
        "action_id": _resolve_action_id("LEVEL_UP_DEXTERITY", 39),
        "metric_suffix": "lvlup_dex",
        "conditions": [("min", C.XP, 1), ("attr_below_max", C.DEXTERITY)],
    },
    "LEVEL_UP_STRENGTH": {
        "action_id": _resolve_action_id("LEVEL_UP_STRENGTH", 40),
        "metric_suffix": "lvlup_str",
        "conditions": [("min", C.XP, 1), ("attr_below_max", C.STRENGTH)],
    },
    "LEVEL_UP_INTELLIGENCE": {
        "action_id": _resolve_action_id("LEVEL_UP_INTELLIGENCE", 41),
        "metric_suffix": "lvlup_int",
        "conditions": [("min", C.XP, 1), ("attr_below_max", C.INTELLIGENCE)],
    },
}

# Backward compatibility alias
BASIC_ACTION_RULES = ACTION_RULES

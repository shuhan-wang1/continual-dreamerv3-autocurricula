"""
Action feasibility rules for early-game Craftax crafting.

Action indices from craftax.craftax.constants.Action:
  PLACE_TABLE = 8, PLACE_FURNACE = 9,
  MAKE_WOOD_PICKAXE = 11, MAKE_STONE_PICKAXE = 12
"""

# Action indices (craftax/craftax/constants.py -> Action)
ACTION_PLACE_TABLE = 8
ACTION_PLACE_FURNACE = 9
ACTION_MAKE_WOOD_PICKAXE = 11
ACTION_MAKE_STONE_PICKAXE = 12

BASIC_ACTION_RULES = {
    "PLACE_TABLE": {
        "action_id": ACTION_PLACE_TABLE,
        "requires": {"wood": 2},
        "needs_table": False,
        "needs_furnace": False,
    },
    "PLACE_FURNACE": {
        "action_id": ACTION_PLACE_FURNACE,
        "requires": {"stone": 1},
        "needs_table": False,
        "needs_furnace": False,
    },
    "MAKE_WOOD_PICKAXE": {
        "action_id": ACTION_MAKE_WOOD_PICKAXE,
        "requires": {"wood": 1},
        "needs_table": True,
        "needs_furnace": False,
    },
    "MAKE_STONE_PICKAXE": {
        "action_id": ACTION_MAKE_STONE_PICKAXE,
        "requires": {"wood": 1, "stone": 1},
        "needs_table": True,
        "needs_furnace": False,
    },
}

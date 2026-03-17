"""Early-game Craftax feasibility rules used at actor sampling time."""

from __future__ import annotations


def _resolve_action_id(name: str, default: int) -> int:
    """Read action ids from Craftax when available and fall back otherwise."""
    enum_types = ()
    try:
        from craftax.craftax.constants import Action as CraftaxAction
        enum_types += (CraftaxAction,)
    except ImportError:
        pass
    try:
        from craftax.craftax_classic.constants import Action as CraftaxClassicAction
        enum_types += (CraftaxClassicAction,)
    except ImportError:
        pass

    for enum_type in enum_types:
        member = getattr(enum_type, name, None)
        if member is not None:
            return int(member.value)
    return int(default)


ACTION_PLACE_TABLE = _resolve_action_id("PLACE_TABLE", 8)
ACTION_PLACE_FURNACE = _resolve_action_id("PLACE_FURNACE", 9)
ACTION_MAKE_WOOD_PICKAXE = _resolve_action_id("MAKE_WOOD_PICKAXE", 11)
ACTION_MAKE_STONE_PICKAXE = _resolve_action_id("MAKE_STONE_PICKAXE", 12)

RULE_ORDER = (
    "PLACE_TABLE",
    "PLACE_FURNACE",
    "MAKE_WOOD_PICKAXE",
    "MAKE_STONE_PICKAXE",
)

BASIC_ACTION_RULES = {
    "PLACE_TABLE": {
        "action_id": ACTION_PLACE_TABLE,
        "metric_suffix": "place_table",
        "requires": {"wood": 2},
        "needs_table": False,
        "needs_furnace": False,
    },
    "PLACE_FURNACE": {
        "action_id": ACTION_PLACE_FURNACE,
        "metric_suffix": "place_furnace",
        "requires": {"stone": 1},
        "needs_table": False,
        "needs_furnace": False,
    },
    "MAKE_WOOD_PICKAXE": {
        "action_id": ACTION_MAKE_WOOD_PICKAXE,
        "metric_suffix": "make_wood_pickaxe",
        "requires": {"wood": 1},
        "needs_table": True,
        "needs_furnace": False,
    },
    "MAKE_STONE_PICKAXE": {
        "action_id": ACTION_MAKE_STONE_PICKAXE,
        "metric_suffix": "make_stone_pickaxe",
        "requires": {"wood": 1, "stone": 1},
        "needs_table": True,
        "needs_furnace": False,
    },
}

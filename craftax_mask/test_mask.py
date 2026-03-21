"""Tests for the expanded action mask (37 maskable actions).

Run: python -m craftax_mask.test_mask
"""
import numpy as np
from .rules import ACTION_RULES, CONTEXT_SIZE, C
from .mask import compute_logit_bias, compute_mask_details


# Helper: look up action_id from ACTION_RULES by rule name
def _aid(rule_name):
    return ACTION_RULES[rule_name]["action_id"]


def _ctx(**overrides) -> np.ndarray:
    """Create a context array with all zeros except overrides."""
    c = np.zeros(CONTEXT_SIZE, dtype=np.float32)
    for k, v in overrides.items():
        idx = getattr(C, k)
        c[idx] = float(v)
    return c


def _bias(ctx, mode="soft", **kw):
    return np.asarray(compute_logit_bias(ctx, ACTION_RULES, 43, mode=mode, **kw))


# --- Placement actions ---

def test_place_table():
    aid = _aid("PLACE_TABLE")
    # Need wood >= 2 AND facing placeable
    b = _bias(_ctx(WOOD=1, FACING_PLACEABLE=1))
    assert b[aid] < 0, "wood<2 should penalize PLACE_TABLE"
    b = _bias(_ctx(WOOD=2, FACING_PLACEABLE=1))
    assert b[aid] == 0, "wood>=2 + placeable should allow PLACE_TABLE"
    b = _bias(_ctx(WOOD=5, FACING_PLACEABLE=0))
    assert b[aid] < 0, "facing non-placeable should penalize PLACE_TABLE"
    print("test_place_table: OK")


def test_place_furnace():
    aid = _aid("PLACE_FURNACE")
    b = _bias(_ctx(STONE=0, FACING_PLACEABLE=1))
    assert b[aid] < 0
    b = _bias(_ctx(STONE=1, FACING_PLACEABLE=1))
    assert b[aid] == 0
    print("test_place_furnace: OK")


def test_place_plant():
    aid = _aid("PLACE_PLANT")
    b = _bias(_ctx(SAPLING=1, FACING_GRASS=1))
    assert b[aid] == 0
    b = _bias(_ctx(SAPLING=0, FACING_GRASS=1))
    assert b[aid] < 0
    b = _bias(_ctx(SAPLING=1, FACING_GRASS=0))
    assert b[aid] < 0
    print("test_place_plant: OK")


# --- Pickaxe crafting with equipment level checks ---

def test_make_wood_pickaxe():
    aid = _aid("MAKE_WOOD_PICKAXE")
    # Need wood>=1, near_table, pickaxe<1
    b = _bias(_ctx(WOOD=1, NEAR_TABLE=1, PICKAXE=0))
    assert b[aid] == 0, "all conditions met"
    b = _bias(_ctx(WOOD=1, NEAR_TABLE=1, PICKAXE=1))
    assert b[aid] < 0, "already have pickaxe: should penalize"
    b = _bias(_ctx(WOOD=0, NEAR_TABLE=1, PICKAXE=0))
    assert b[aid] < 0, "no wood"
    b = _bias(_ctx(WOOD=1, NEAR_TABLE=0, PICKAXE=0))
    assert b[aid] < 0, "no table"
    print("test_make_wood_pickaxe: OK")


def test_make_stone_pickaxe():
    aid = _aid("MAKE_STONE_PICKAXE")
    b = _bias(_ctx(WOOD=1, STONE=1, NEAR_TABLE=1, PICKAXE=1))
    assert b[aid] == 0, "pickaxe=1 < 2, should be allowed"
    b = _bias(_ctx(WOOD=1, STONE=1, NEAR_TABLE=1, PICKAXE=2))
    assert b[aid] < 0, "pickaxe=2 >= 2, should penalize"
    print("test_make_stone_pickaxe: OK")


def test_make_iron_pickaxe():
    aid = _aid("MAKE_IRON_PICKAXE")
    b = _bias(_ctx(WOOD=1, STONE=1, IRON=1, COAL=1,
                   NEAR_TABLE=1, NEAR_FURNACE=1, PICKAXE=2))
    assert b[aid] == 0
    b = _bias(_ctx(WOOD=1, STONE=1, IRON=1, COAL=1,
                   NEAR_TABLE=1, NEAR_FURNACE=0, PICKAXE=2))
    assert b[aid] < 0, "no furnace"
    b = _bias(_ctx(WOOD=1, STONE=1, IRON=1, COAL=1,
                   NEAR_TABLE=1, NEAR_FURNACE=1, PICKAXE=3))
    assert b[aid] < 0, "already have iron pickaxe"
    print("test_make_iron_pickaxe: OK")


# --- Sword crafting ---

def test_make_wood_sword():
    aid = _aid("MAKE_WOOD_SWORD")
    b = _bias(_ctx(WOOD=1, NEAR_TABLE=1, SWORD=0))
    assert b[aid] == 0
    b = _bias(_ctx(WOOD=1, NEAR_TABLE=1, SWORD=1))
    assert b[aid] < 0, "already have sword"
    print("test_make_wood_sword: OK")


# --- Armour crafting ---

def test_make_iron_armour():
    aid = _aid("MAKE_IRON_ARMOUR")
    # Need iron>=3, coal>=3, near_table+furnace, any armour slot < 1
    ctx = _ctx(IRON=3, COAL=3, NEAR_TABLE=1, NEAR_FURNACE=1)
    # All armour slots are 0 (< 1), so should pass
    b = _bias(ctx)
    assert b[aid] == 0
    # Set all armour to 1 -> no slot available
    ctx[C.ARMOUR_0:C.ARMOUR_3+1] = 1.0
    b = _bias(ctx)
    assert b[aid] < 0, "all slots full"
    print("test_make_iron_armour: OK")


def test_make_diamond_armour():
    aid = _aid("MAKE_DIAMOND_ARMOUR")
    ctx = _ctx(DIAMOND=3, NEAR_TABLE=1)
    ctx[C.ARMOUR_0] = 1.0  # one slot has iron, < 2 -> upgradeable
    b = _bias(ctx)
    assert b[aid] == 0
    ctx[C.ARMOUR_0:C.ARMOUR_3+1] = 2.0  # all diamond
    b = _bias(ctx)
    assert b[aid] < 0
    print("test_make_diamond_armour: OK")


# --- Potions ---

def test_potions():
    potion_rules = [
        "DRINK_POTION_RED", "DRINK_POTION_GREEN", "DRINK_POTION_BLUE",
        "DRINK_POTION_PINK", "DRINK_POTION_YELLOW", "DRINK_POTION_RAINBOW",
    ]
    # Filter to only rules that exist, then extract the correct slot index
    # from each rule's conditions to avoid index misalignment when rules are missing.
    valid_potions = [(name, ACTION_RULES[name]) for name in potion_rules
                     if name in ACTION_RULES]
    for rule_name, rule in valid_potions:
        aid = rule["action_id"]
        # Extract the potion slot index from the rule's conditions
        slot_idx = None
        for cond in rule["conditions"]:
            if cond[0] == "min" and C.POTION_0 <= cond[1] <= C.POTION_5:
                slot_idx = cond[1]
                break
        assert slot_idx is not None, f"Could not find potion slot for {rule_name}"
        ctx = _ctx()
        ctx[slot_idx] = 1.0
        b = _bias(ctx)
        assert b[aid] == 0, f"{rule_name} should be allowed"
        ctx[slot_idx] = 0.0
        b = _bias(ctx)
        assert b[aid] < 0, f"{rule_name} should be penalized when empty"
    print("test_potions: OK")


# --- Spells ---

def test_cast_fireball():
    aid = _aid("CAST_FIREBALL")
    b = _bias(_ctx(MANA=2, SPELL_FIREBALL=1, PROJECTILE_SLOTS=1))
    assert b[aid] == 0
    b = _bias(_ctx(MANA=1, SPELL_FIREBALL=1, PROJECTILE_SLOTS=1))
    assert b[aid] < 0, "not enough mana"
    b = _bias(_ctx(MANA=2, SPELL_FIREBALL=0, PROJECTILE_SLOTS=1))
    assert b[aid] < 0, "spell not learned"
    print("test_cast_fireball: OK")


# --- Enchanting ---

def test_enchant_sword():
    aid = _aid("ENCHANT_SWORD")
    # Fire table + ruby = allowed
    b = _bias(_ctx(FACING_ENCHANT_TABLE=1, FACING_FIRE_TABLE=1, MANA=9, RUBY=1, SWORD=1))
    assert b[aid] == 0, "fire table + ruby should allow"
    # Ice table + sapphire = allowed
    b = _bias(_ctx(FACING_ENCHANT_TABLE=1, FACING_ICE_TABLE=1, MANA=9, SAPPHIRE=1, SWORD=1))
    assert b[aid] == 0, "ice table + sapphire should allow"
    # Fire table + sapphire only (wrong gem) = penalized
    b = _bias(_ctx(FACING_ENCHANT_TABLE=1, FACING_FIRE_TABLE=1, MANA=9, SAPPHIRE=1, RUBY=0, SWORD=1))
    assert b[aid] < 0, "fire table + sapphire (wrong gem) should penalize"
    # Ice table + ruby only (wrong gem) = penalized
    b = _bias(_ctx(FACING_ENCHANT_TABLE=1, FACING_ICE_TABLE=1, MANA=9, RUBY=1, SAPPHIRE=0, SWORD=1))
    assert b[aid] < 0, "ice table + ruby (wrong gem) should penalize"
    # No gems at all = penalized
    b = _bias(_ctx(FACING_ENCHANT_TABLE=1, FACING_FIRE_TABLE=1, MANA=9, RUBY=0, SAPPHIRE=0, SWORD=1))
    assert b[aid] < 0, "no gem"
    # Not facing enchant table = penalized
    b = _bias(_ctx(FACING_ENCHANT_TABLE=0, MANA=9, RUBY=1, SWORD=1))
    assert b[aid] < 0, "not facing enchant table"
    print("test_enchant_sword: OK")


# --- Floor transitions ---

def test_descend():
    aid = _aid("DESCEND")
    b = _bias(_ctx(ON_LADDER_DOWN=1, LEVEL_CLEARED=1, LEVEL=3))
    assert b[aid] == 0
    b = _bias(_ctx(ON_LADDER_DOWN=0, LEVEL_CLEARED=1, LEVEL=3))
    assert b[aid] < 0, "not on ladder"
    b = _bias(_ctx(ON_LADDER_DOWN=1, LEVEL_CLEARED=0, LEVEL=3))
    assert b[aid] < 0, "level not cleared"
    b = _bias(_ctx(ON_LADDER_DOWN=1, LEVEL_CLEARED=1, LEVEL=8))
    assert b[aid] < 0, "already at max level"
    print("test_descend: OK")


def test_ascend():
    aid = _aid("ASCEND")
    b = _bias(_ctx(ON_LADDER_UP=1, LEVEL=1))
    assert b[aid] == 0
    b = _bias(_ctx(ON_LADDER_UP=1, LEVEL=0))
    assert b[aid] < 0, "already at level 0"
    print("test_ascend: OK")


# --- Level-up ---

def test_level_up():
    aid = _aid("LEVEL_UP_DEXTERITY")
    b = _bias(_ctx(XP=1, DEXTERITY=4))
    assert b[aid] == 0, "dex<5, xp>=1"
    b = _bias(_ctx(XP=1, DEXTERITY=5))
    assert b[aid] < 0, "dex=5 (maxed)"
    b = _bias(_ctx(XP=0, DEXTERITY=1))
    assert b[aid] < 0, "no xp"
    print("test_level_up: OK")


# --- Sleep / Rest ---

def test_sleep():
    aid = _aid("SLEEP")
    # energy < max_energy (7 + 2*dex)
    # dex=1 -> max=9, energy=8 < 9 -> allowed
    b = _bias(_ctx(ENERGY=8, DEXTERITY=1))
    assert b[aid] == 0
    # energy=9 >= max=9 -> penalized
    b = _bias(_ctx(ENERGY=9, DEXTERITY=1))
    assert b[aid] < 0
    print("test_sleep: OK")


def test_rest():
    aid = _aid("REST")
    # health < max_health (8 + str)
    # str=1 -> max=9, health=8 < 9 -> allowed
    b = _bias(_ctx(HEALTH=8, STRENGTH=1))
    assert b[aid] == 0
    b = _bias(_ctx(HEALTH=9, STRENGTH=1))
    assert b[aid] < 0
    print("test_rest: OK")


# --- Hard mode ---

def test_hard_mode():
    aid = _aid("PLACE_TABLE")
    b = _bias(_ctx(WOOD=0), mode="hard", large_negative=1e9)
    assert b[aid] < -1e8, "PLACE_TABLE hard blocked"
    b = _bias(_ctx(WOOD=5, FACING_PLACEABLE=1), mode="hard")
    assert b[aid] == 0.0
    print("test_hard_mode: OK")


# --- Batch ---

def test_batched():
    aid = _aid("PLACE_TABLE")
    ctx = np.stack([
        _ctx(WOOD=0, NEAR_TABLE=1, FACING_PLACEABLE=1),
        _ctx(WOOD=5, NEAR_TABLE=1, FACING_PLACEABLE=1),
    ])
    b = _bias(ctx)
    assert b.shape == (2, 43)
    assert b[0, aid] < 0, "batch[0] wood=0"
    assert b[1, aid] == 0, "batch[1] wood=5"
    print("test_batched: OK")


def main():
    test_place_table()
    test_place_furnace()
    test_place_plant()
    test_make_wood_pickaxe()
    test_make_stone_pickaxe()
    test_make_iron_pickaxe()
    test_make_wood_sword()
    test_make_iron_armour()
    test_make_diamond_armour()
    test_potions()
    test_cast_fireball()
    test_enchant_sword()
    test_descend()
    test_ascend()
    test_level_up()
    test_sleep()
    test_rest()
    test_hard_mode()
    test_batched()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()

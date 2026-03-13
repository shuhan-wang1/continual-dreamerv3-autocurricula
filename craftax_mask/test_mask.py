"""
Sanity checks for action mask logic.

Run: python -m craftax_mask.test_mask
"""
import numpy as np


def test_rules():
    from .rules import BASIC_ACTION_RULES, ACTION_PLACE_TABLE, ACTION_MAKE_WOOD_PICKAXE
    assert ACTION_PLACE_TABLE == 8
    assert ACTION_MAKE_WOOD_PICKAXE == 11
    assert BASIC_ACTION_RULES["PLACE_TABLE"]["requires"] == {"wood": 2}
    assert BASIC_ACTION_RULES["MAKE_WOOD_PICKAXE"]["needs_table"] is True
    print("rules: OK")


def test_mask_soft():
    from .mask import compute_logit_bias
    from .rules import BASIC_ACTION_RULES

    # wood < 2 -> PLACE_TABLE (8) should be penalized
    ctx = {"wood": 1, "stone": 0, "near_table": True, "near_furnace": True}
    bias = compute_logit_bias(ctx, BASIC_ACTION_RULES, num_actions=43, mode="soft", lambda_penalty=5.0)
    assert bias[8] < 0, "PLACE_TABLE should get negative bias when wood<2"
    print("mask soft (wood<2, PLACE_TABLE suppressed): OK")

    # wood >= 2 -> PLACE_TABLE no penalty
    ctx2 = {"wood": 2, "stone": 0, "near_table": True, "near_furnace": True}
    bias2 = compute_logit_bias(ctx2, BASIC_ACTION_RULES, num_actions=43, mode="soft")
    assert bias2[8] == 0, "PLACE_TABLE should have no bias when wood>=2"
    print("mask soft (wood>=2, PLACE_TABLE allowed): OK")

    # no nearby table -> MAKE_WOOD_PICKAXE (11) penalized
    ctx3 = {"wood": 5, "stone": 5, "near_table": False, "near_furnace": True}
    bias3 = compute_logit_bias(ctx3, BASIC_ACTION_RULES, num_actions=43, mode="soft")
    assert bias3[11] < 0, "MAKE_WOOD_PICKAXE should be penalized when no table"
    print("mask soft (no table, MAKE_WOOD_PICKAXE suppressed): OK")

    # no nearby table -> MAKE_STONE_PICKAXE (12) penalized
    assert bias3[12] < 0, "MAKE_STONE_PICKAXE should be penalized when no table"
    print("mask soft (no table, MAKE_STONE_PICKAXE suppressed): OK")


def test_mask_hard():
    from .mask import compute_logit_bias
    from .rules import BASIC_ACTION_RULES

    ctx = {"wood": 1, "stone": 0, "near_table": True, "near_furnace": True}
    bias = compute_logit_bias(ctx, BASIC_ACTION_RULES, num_actions=43, mode="hard", large_negative=1e9)
    assert bias[8] < -1e8, "PLACE_TABLE should get large negative in hard mode"
    print("mask hard (wood<2, PLACE_TABLE zero prob): OK")


def main():
    test_rules()
    test_mask_soft()
    test_mask_hard()
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()

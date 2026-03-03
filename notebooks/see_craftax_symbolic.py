"""

see_craftax_symbolic.py

=======================

Decode and inspect a Craftax Symbolic observation vector.



The full observation is a 1-D float32 array of length 8,268:

  [0 … 8,216]  – flattened 9x11x83 map tensor

  [8,217 … 8,267]  – 51-element inventory / status vector



Run modes

---------

  python see_craftax_symbolic.py --schema

      Print the complete index-ranges legend (no env needed).



  python see_craftax_symbolic.py --analyze <obs.npy>

      Decode and pretty-print every field of a saved observation.



  python see_craftax_symbolic.py --map <obs.npy>

      Render an ASCII grid of the 9x11 map view.



  python see_craftax_symbolic.py --analyze <obs.npy> --map <obs.npy>

      Both at once.

"""



from __future__ import annotations

import argparse

import textwrap

from typing import Any



import numpy as np



# -----------------------------------------------------------------------------

# Craftax symbolic-obs constants  (mirrors craftax/craftax/constants.py)

# -----------------------------------------------------------------------------



OBS_ROWS   = 9   # OBS_DIM[0]

OBS_COLS   = 11  # OBS_DIM[1]

OBS_TILES  = OBS_ROWS * OBS_COLS  # 99



BLOCK_TYPES: list[str] = [

    "INVALID",              # 0

    "OUT_OF_BOUNDS",        # 1

    "GRASS",                # 2

    "WATER",                # 3

    "STONE",                # 4

    "TREE",                 # 5

    "WOOD",                 # 6

    "PATH",                 # 7

    "COAL",                 # 8

    "IRON",                 # 9

    "DIAMOND",              # 10

    "CRAFTING_TABLE",       # 11

    "FURNACE",              # 12

    "SAND",                 # 13

    "LAVA",                 # 14

    "PLANT",                # 15

    "RIPE_PLANT",           # 16

    "WALL",                 # 17

    "DARKNESS",             # 18

    "WALL_MOSS",            # 19

    "STALAGMITE",           # 20

    "SAPPHIRE",             # 21

    "RUBY",                 # 22

    "CHEST",                # 23

    "FOUNTAIN",             # 24

    "FIRE_GRASS",           # 25

    "ICE_GRASS",            # 26

    "GRAVEL",               # 27

    "FIRE_TREE",            # 28

    "ICE_SHRUB",            # 29

    "ENCHANTMENT_TABLE_FIRE",   # 30

    "ENCHANTMENT_TABLE_ICE",    # 31

    "NECROMANCER",          # 32

    "GRAVE",                # 33

    "GRAVE2",               # 34

    "GRAVE3",               # 35

    "NECROMANCER_VULNERABLE",   # 36

]



ITEM_TYPES: list[str] = [

    "NONE",                 # 0

    "TORCH",                # 1

    "LADDER_DOWN",          # 2

    "LADDER_UP",            # 3

    "LADDER_DOWN_BLOCKED",  # 4

]



# 5 mob classes x 8 type-IDs = 40 channels

MOB_CLASSES: list[str] = [

    "melee",        # class 0 – Zombie, GnomeWarrior, OrcSoldier, Lizard, Knight, Troll, Pigman, FrostTroll

    "passive",      # class 1 – Cow, Bat, Snail

    "ranged",       # class 2 – Skeleton, GnomeArcher, OrcMage, Kobold, KnightArcher, DeepThing, FireElemental, IceElemental

    "mob_proj",     # class 3 – projectiles fired by mobs

    "player_proj",  # class 4 – projectiles fired by the player

]

MOB_CLASS_MELEE_TYPES:  list[str] = ["Zombie","GnomeWarrior","OrcSoldier","Lizard","Knight","Troll","Pigman","FrostTroll"]

MOB_CLASS_PASSIVE_TYPES: list[str] = ["Cow","Bat","Snail","–","–","–","–","–"]

MOB_CLASS_RANGED_TYPES:  list[str] = ["Skeleton","GnomeArcher","OrcMage","Kobold","KnightArcher","DeepThing","FireElemental","IceElemental"]

MOB_CLASS_MOBPROJ_TYPES: list[str] = ["Arrow","Dagger","Fireball","Slimeball","Arrow2","Slimeball2","Fireball2","Iceball2"]

MOB_CLASS_PLAYERPROJ_TYPES: list[str] = ["Arrow","Fireball","Iceball","–","–","–","–","–"]

MOB_TYPE_NAMES: list[list[str]] = [

    MOB_CLASS_MELEE_TYPES,

    MOB_CLASS_PASSIVE_TYPES,

    MOB_CLASS_RANGED_TYPES,

    MOB_CLASS_MOBPROJ_TYPES,

    MOB_CLASS_PLAYERPROJ_TYPES,

]



# -- Derived map-section sizes -------------------------------------------------

N_BLOCK_CH  = len(BLOCK_TYPES)       # 37

N_ITEM_CH   = len(ITEM_TYPES)        # 5

N_MOB_CH    = len(MOB_CLASSES) * 8   # 40

N_LIGHT_CH  = 1



CHANNELS_PER_TILE = N_BLOCK_CH + N_ITEM_CH + N_MOB_CH + N_LIGHT_CH  # 83

MAP_SIZE          = OBS_TILES * CHANNELS_PER_TILE                    # 8,217



# Channel slices within one tile

CH_BLOCK_START  = 0

CH_BLOCK_END    = N_BLOCK_CH                              # 37

CH_ITEM_START   = CH_BLOCK_END                            # 37

CH_ITEM_END     = CH_ITEM_START + N_ITEM_CH               # 42

CH_MOB_START    = CH_ITEM_END                             # 42

CH_MOB_END      = CH_MOB_START + N_MOB_CH                 # 82

CH_LIGHT        = CH_MOB_END                              # 82



# -- Inventory section (51 scalars, starting at MAP_SIZE) ---------------------

INV_START = MAP_SIZE  # 8,217



POTION_TYPES: list[str] = [

    "health_potion", "mana_potion", "dex_potion",

    "str_potion",    "int_potion",  "speed_potion",

]



ARMOUR_SLOTS: list[str] = ["helmet", "chestplate", "pants", "boots"]

ARMOUR_ENCHANT_VALUES: list[str] = ["none", "fire", "ice"]



DIRECTION_NAMES: list[str] = ["LEFT", "RIGHT", "UP", "DOWN"]



# Ordered list of (local_offset, name, semantics)

INVENTORY_FIELDS: list[tuple[int, str, str]] = [

    # -- raw items --------------------------------------------------------------

    (0,  "wood",             "sqrt(count)/10"),

    (1,  "stone",            "sqrt(count)/10"),

    (2,  "coal",             "sqrt(count)/10"),

    (3,  "iron",             "sqrt(count)/10"),

    (4,  "diamond",          "sqrt(count)/10"),

    (5,  "sapphire",         "sqrt(count)/10"),

    (6,  "ruby",             "sqrt(count)/10"),

    (7,  "sapling",          "sqrt(count)/10"),

    (8,  "torches",          "sqrt(count)/10"),

    (9,  "arrows",           "sqrt(count)/10"),

    (10, "books",            "count/2"),

    (11, "pickaxe_level",    "level/4  (0=none,1=wood,2=stone,3=iron,4=diamond)"),

    (12, "sword_level",      "level/4  (0=none,1=wood,2=stone,3=iron,4=diamond)"),

    (13, "sword_enchantment","0=none,1=fire,2=ice"),

    (14, "bow_enchantment",  "0=none,1=fire,2=ice"),

    (15, "bow",              "0/1 boolean"),

    # -- potions (6 types) ------------------------------------------------------

    (16, "potion_health",    "sqrt(count)/10"),

    (17, "potion_mana",      "sqrt(count)/10"),

    (18, "potion_dex",       "sqrt(count)/10"),

    (19, "potion_str",       "sqrt(count)/10"),

    (20, "potion_int",       "sqrt(count)/10"),

    (21, "potion_speed",     "sqrt(count)/10"),

    # -- player stats -----------------------------------------------------------

    (22, "health",           "/10  (max=10)"),

    (23, "food",             "/10  (max=10)"),

    (24, "drink",            "/10  (max=10)"),

    (25, "energy",           "/10  (max=10)"),

    (26, "mana",             "/10  (max=10)"),

    (27, "xp",               "/10"),

    (28, "dexterity",        "/10"),

    (29, "strength",         "/10"),

    (30, "intelligence",     "/10"),

    # -- direction one-hot ------------------------------------------------------

    (31, "facing_LEFT",      "one-hot"),

    (32, "facing_RIGHT",     "one-hot"),

    (33, "facing_UP",        "one-hot"),

    (34, "facing_DOWN",      "one-hot"),

    # -- armour levels ----------------------------------------------------------

    (35, "armour_helmet",    "level/2  (0=none,1=iron,2=diamond)"),

    (36, "armour_chestplate","level/2"),

    (37, "armour_pants",     "level/2"),

    (38, "armour_boots",     "level/2"),

    # -- armour enchantments ----------------------------------------------------

    (39, "armour_ench_helmet",    "0=none,1=fire,2=ice"),

    (40, "armour_ench_chestplate","0=none,1=fire,2=ice"),

    (41, "armour_ench_pants",     "0=none,1=fire,2=ice"),

    (42, "armour_ench_boots",     "0=none,1=fire,2=ice"),

    # -- global special values --------------------------------------------------

    (43, "light_level",      "0=dark,1=lit"),

    (44, "is_sleeping",      "boolean"),

    (45, "is_resting",       "boolean"),

    (46, "learned_fireball", "boolean"),

    (47, "learned_iceball",  "boolean"),

    (48, "player_level",     "/10  (0=surface,1=gnomish mines,…)"),

    (49, "monsters_cleared", "boolean (≥8 monsters killed on current floor)"),

    (50, "boss_vulnerable",  "boolean (necromancer can be damaged)"),

]

assert len(INVENTORY_FIELDS) == 51, "Inventory layout mismatch!"



# -- ASCII glyphs for map rendering --------------------------------------------

BLOCK_GLYPH: dict[str, str] = {

    "INVALID":               "?",

    "OUT_OF_BOUNDS":         "█",

    "GRASS":                 "·",

    "WATER":                 "~",

    "STONE":                 ":",

    "TREE":                  "T",

    "WOOD":                  "W",

    "PATH":                  "_",

    "COAL":                  "c",

    "IRON":                  "i",

    "DIAMOND":               "d",

    "CRAFTING_TABLE":        "≡",

    "FURNACE":               "Ω",

    "SAND":                  "s",

    "LAVA":                  "^",

    "PLANT":                 "p",

    "RIPE_PLANT":            "P",

    "WALL":                  "#",

    "DARKNESS":              " ",

    "WALL_MOSS":             "%",

    "STALAGMITE":            "^",

    "SAPPHIRE":              "S",

    "RUBY":                  "R",

    "CHEST":                 "C",

    "FOUNTAIN":              "F",

    "FIRE_GRASS":            "f",

    "ICE_GRASS":             "I",

    "GRAVEL":                "g",

    "FIRE_TREE":             "t",

    "ICE_SHRUB":             "j",

    "ENCHANTMENT_TABLE_FIRE":"E",

    "ENCHANTMENT_TABLE_ICE": "e",

    "NECROMANCER":           "N",

    "GRAVE":                 "+",

    "GRAVE2":                "+",

    "GRAVE3":                "+",

    "NECROMANCER_VULNERABLE":"n",

}



ITEM_GLYPH: dict[str, str] = {

    "NONE":               None,

    "TORCH":              "τ",

    "LADDER_DOWN":        "v",

    "LADDER_UP":          "^",

    "LADDER_DOWN_BLOCKED":"v",

}



MOB_GLYPH: dict[int, str] = {

    # melee (class 0)

    0:  "Z",  # Zombie

    1:  "G",  # GnomeWarrior

    2:  "O",  # OrcSoldier

    3:  "L",  # Lizard

    4:  "K",  # Knight

    5:  "T",  # Troll

    6:  "M",  # Pigman

    7:  "F",  # FrostTroll

    # passive (class 1)

    8:  "C",  # Cow

    9:  "B",  # Bat

    10: "σ",  # Snail

    # ranged (class 2)

    16: "☠",  # Skeleton

    17: "g",  # GnomeArcher

    18: "o",  # OrcMage

    19: "k",  # Kobold

    20: "A",  # KnightArcher

    21: "D",  # DeepThing

    22: "🔥", # FireElemental (may not render in all terminals)

    23: "❄",  # IceElemental

    # projectiles (class 3 → offset 24, class 4 → offset 32)

}



# -----------------------------------------------------------------------------

# Helper utilities

# -----------------------------------------------------------------------------



def _hr(char: str = "-", width: int = 76) -> str:

    return char * width





def _section(title: str) -> str:

    pad = max(0, 74 - len(title))

    return f"\n{'=' * 3} {title} {'=' * pad}\n"





def _fmt_val(v: float, fmt: str = ".4f") -> str:

    return format(float(v), fmt)





# -----------------------------------------------------------------------------

# 1.  Schema printer

# -----------------------------------------------------------------------------



def print_schema() -> None:

    """Print a fully self-contained legend of the observation vector."""



    print(_hr("="))

    print("  CRAFTAX SYMBOLIC OBSERVATION SCHEMA")

    print(f"  Total length : {MAP_SIZE + 51:,}  (map section: {MAP_SIZE:,} + inventory: 51)")

    print(_hr("="))



    # -- Map section -----------------------------------------------------------

    print(_section("MAP SECTION  [indices 0 … {:,}]".format(MAP_SIZE - 1)))



    print(textwrap.dedent(f"""\

        The map is a {OBS_ROWS}x{OBS_COLS} grid centred on the player (player is at row 4, col 5).

        It is stored row-major, then column-major, then channel-major.



        Index formula for a given tile and channel:

            flat_idx = row * {OBS_COLS} * {CHANNELS_PER_TILE}  +  col * {CHANNELS_PER_TILE}  +  channel

            (row ∈ [0,{OBS_ROWS-1}],  col ∈ [0,{OBS_COLS-1}],  channel ∈ [0,{CHANNELS_PER_TILE-1}])



        Tiles outside the world boundary are filled with BLOCK_TYPE=OUT_OF_BOUNDS.

        Tiles in darkness have all channels zeroed except the light channel (channel 82).

    """))



    print(f"  Channels per tile = {CHANNELS_PER_TILE}")

    print(f"  +- channels  0…{CH_BLOCK_END-1:2d}  ({N_BLOCK_CH}) : Block type one-hot  (BlockType enum)")

    print(f"  +- channels {CH_ITEM_START:2d}…{CH_ITEM_END-1:2d}  ({N_ITEM_CH}) : Item type one-hot   (ItemType enum)")

    print(f"  +- channels {CH_MOB_START:2d}…{CH_MOB_END-1:2d}  ({N_MOB_CH}) : Mob presence one-hot (5 classes x 8 types)")

    print(f"  +- channel  {CH_LIGHT:2d}       ( 1) : Visibility / light  (1 = visible, 0 = dark)")



    print(f"\n  Block types (channels 0…{N_BLOCK_CH-1})")

    for i, name in enumerate(BLOCK_TYPES):

        glyph = BLOCK_GLYPH.get(name, "?")

        print(f"    [{i:2d}] {name:<30s}  glyph: {glyph}")



    print(f"\n  Item types (channels {CH_ITEM_START}…{CH_ITEM_END-1})")

    for i, name in enumerate(ITEM_TYPES):

        glyph = ITEM_GLYPH.get(name, "?")

        print(f"    [{CH_ITEM_START+i}] {name:<30s}  glyph: {glyph if glyph else '(none)'}")



    print(f"\n  Mob channels (channels {CH_MOB_START}…{CH_MOB_END-1})  — format: channel = class*8 + type_id")

    for c, (cls_name, type_names) in enumerate(zip(MOB_CLASSES, MOB_TYPE_NAMES)):

        base = CH_MOB_START + c * 8

        print(f"    Class {c} [{base}…{base+7}]  {cls_name}")

        for t, tname in enumerate(type_names):

            if tname != "–":

                print(f"      ch {base+t:2d}  type_id={t}  {tname}")



    # -- Inventory section -----------------------------------------------------

    print(_section("INVENTORY / STATUS SECTION  [indices {:,} … {:,}]".format(

        INV_START, INV_START + 50)))



    print(f"  {'offset':>6}  {'abs index':>9}  {'field name':<25}  semantics")

    print(f"  {_hr('-',6)}  {_hr('-',9)}  {_hr('-',25)}  {_hr('-',30)}")

    for off, name, sem in INVENTORY_FIELDS:

        abs_idx = INV_START + off

        print(f"  {off:6d}  {abs_idx:9,d}  {name:<25}  {sem}")



    print()

    print(_hr("="))





# -----------------------------------------------------------------------------

# 2.  Decode a single observation vector

# -----------------------------------------------------------------------------



def decode_obs(obs: np.ndarray) -> dict[str, Any]:

    """Return a structured dict with all decoded fields from `obs`."""

    obs = np.asarray(obs, dtype=np.float32).ravel()

    if obs.shape[0] != MAP_SIZE + 51:

        raise ValueError(

            f"Expected obs length {MAP_SIZE+51}, got {obs.shape[0]}"

        )



    # -- Inventory fields ------------------------------------------------------

    inv = obs[INV_START : INV_START + 51]

    decoded_inv: dict[str, float] = {}

    for off, name, _ in INVENTORY_FIELDS:

        decoded_inv[name] = float(inv[off])



    # -- Reconstruct map (9x11) with argmax per three channel groups -----------

    map_flat = obs[:MAP_SIZE].reshape(OBS_ROWS, OBS_COLS, CHANNELS_PER_TILE)



    block_ids = np.argmax(map_flat[:, :, CH_BLOCK_START:CH_BLOCK_END], axis=-1)  # (9,11)

    item_ids  = np.argmax(map_flat[:, :, CH_ITEM_START:CH_ITEM_END],   axis=-1)

    light     = map_flat[:, :, CH_LIGHT]



    # build mob presence per cell (list of active mob names)

    mob_channels = map_flat[:, :, CH_MOB_START:CH_MOB_END]  # (9,11,40)

    mob_present: list[list[list[str]]] = [

        [[] for _ in range(OBS_COLS)] for _ in range(OBS_ROWS)

    ]

    for r in range(OBS_ROWS):

        for c in range(OBS_COLS):

            for cls_idx, type_names in enumerate(MOB_TYPE_NAMES):

                for t_idx, tname in enumerate(type_names):

                    ch = cls_idx * 8 + t_idx

                    if mob_channels[r, c, ch] > 0.5:

                        mob_present[r][c].append(tname)



    return {

        "inventory": decoded_inv,

        "block_ids": block_ids,

        "item_ids":  item_ids,

        "light":     light,

        "mob_present": mob_present,

        "raw_obs": obs,

    }





def print_inventory(decoded: dict[str, Any]) -> None:

    inv = decoded["inventory"]



    def _h(title: str) -> None:

        print(f"\n  -- {title}")



    def _row(name: str, val: float, hint: str = "") -> None:

        bar_len = int(min(1.0, val) * 20)

        bar = "█" * bar_len + "░" * (20 - bar_len)

        print(f"    {name:<25} {val:6.3f}  [{bar}]  {hint}")



    print(_section("INVENTORY & STATUS"))



    _h("Raw materials  (displayed as sqrt(count)/10; multiply^2 x100 to estimate count)")

    for k in ["wood","stone","coal","iron","diamond","sapphire","ruby","sapling","torches","arrows"]:

        approx_count = round((inv[k] * 10) ** 2)

        _row(k, inv[k], f"≈{approx_count} items")



    _h("Equipment")

    _row("books",            inv["books"],            "x2 for actual count")

    _row("pickaxe_level",    inv["pickaxe_level"],     f"level≈{inv['pickaxe_level']*4:.1f}  (0=none,4=diamond)")

    _row("sword_level",      inv["sword_level"],       f"level≈{inv['sword_level']*4:.1f}")

    _row("sword_enchantment",inv["sword_enchantment"], "0=none,1=fire,2=ice")

    _row("bow",              inv["bow"],               "0/1")

    _row("bow_enchantment",  inv["bow_enchantment"],   "0=none,1=fire,2=ice")



    _h("Potions  (sqrt(count)/10)")

    for k in ["potion_health","potion_mana","potion_dex","potion_str","potion_int","potion_speed"]:

        _row(k, inv[k])



    _h("Player stats")

    _row("health",       inv["health"],       f"≈{inv['health']*10:.1f}/10")

    _row("food",         inv["food"],         f"≈{inv['food']*10:.1f}/10")

    _row("drink",        inv["drink"],        f"≈{inv['drink']*10:.1f}/10")

    _row("energy",       inv["energy"],       f"≈{inv['energy']*10:.1f}/10")

    _row("mana",         inv["mana"],         f"≈{inv['mana']*10:.1f}/10")

    _row("xp",           inv["xp"],           f"≈{inv['xp']*10:.1f}")

    _row("dexterity",    inv["dexterity"],    f"≈{inv['dexterity']*10:.1f}")

    _row("strength",     inv["strength"],     f"≈{inv['strength']*10:.1f}")

    _row("intelligence", inv["intelligence"], f"≈{inv['intelligence']*10:.1f}")



    _h("Facing direction")

    for d in DIRECTION_NAMES:

        k = f"facing_{d}"

        print(f"    {k:<25} {inv[k]:.0f}  {'<' if inv[k]>0.5 else ''}")



    _h("Armour  (valuex2 = level: 0=none,1=iron,2=diamond)")

    for slot in ARMOUR_SLOTS:

        level = round(inv[f"armour_{slot}"] * 2)

        ench  = round(inv[f"armour_ench_{slot}"])

        ench_name = ARMOUR_ENCHANT_VALUES[ench] if ench < len(ARMOUR_ENCHANT_VALUES) else str(ench)

        print(f"    {slot:<25} level={level}  enchantment={ench_name}")



    _h("World / special")

    print(f"    {'light_level':<25} {inv['light_level']:.0f}  (1=lit)")

    print(f"    {'is_sleeping':<25} {inv['is_sleeping']:.0f}")

    print(f"    {'is_resting':<25}  {inv['is_resting']:.0f}")

    print(f"    {'learned_fireball':<25} {inv['learned_fireball']:.0f}")

    print(f"    {'learned_iceball':<25}  {inv['learned_iceball']:.0f}")

    depth = round(inv["player_level"] * 10)

    depth_names = {0:"surface",1:"gnomish mines",2:"dungeon",

                   3:"sewers",4:"vaults",5:"troll mines",6:"fire realm",

                   7:"ice realm",8:"graveyard"}

    print(f"    {'player_depth':<25} {depth}  ({depth_names.get(depth,'?')})")

    print(f"    {'monsters_cleared':<25} {inv['monsters_cleared']:.0f}")

    print(f"    {'boss_vulnerable':<25}  {inv['boss_vulnerable']:.0f}")





def print_map(decoded: dict[str, Any], show_legend: bool = True) -> None:

    """Render the 9x11 view as an ASCII grid. Player is at centre (row 4, col 5)."""

    block_ids  = decoded["block_ids"]

    item_ids   = decoded["item_ids"]

    light      = decoded["light"]

    mob_present = decoded["mob_present"]



    print(_section("MAP VIEW  (9 rows x 11 cols — player '@' at centre row 4, col 5)"))



    # Column header

    col_header = "     " + "".join(f"{c:2d}" for c in range(OBS_COLS))

    print(col_header)

    print("     " + _hr("-", OBS_COLS * 2))



    for r in range(OBS_ROWS):

        row_chars = []

        for c in range(OBS_COLS):

            if r == OBS_ROWS // 2 and c == OBS_COLS // 2:

                glyph = "@"  # player centre

            elif light[r, c] < 0.5:

                glyph = " "  # dark

            else:

                # prioritize: mob > item > block

                mobs = mob_present[r][c]

                if mobs:

                    glyph = "M"  # simplified mob glyph

                elif item_ids[r, c] != 0:

                    glyph = ITEM_GLYPH.get(ITEM_TYPES[item_ids[r, c]], "?") or "?"

                else:

                    block_name = BLOCK_TYPES[block_ids[r, c]]

                    glyph = BLOCK_GLYPH.get(block_name, "?")

            row_chars.append(glyph)



        row_label = f" r{r:1d} |"

        row_str = " ".join(row_chars)

        print(f"{row_label} {row_str}")



    print()



    if show_legend:

        print("  Legend:")

        legend_items = [

            ("@", "Player"),

            ("·", "Grass"),

            ("#", "Wall"),

            ("T", "Tree"),

            ("~", "Water"),

            (":", "Stone"),

            ("^", "Lava/Stalagmite"),

            ("c", "Coal"),

            ("i", "Iron"),

            ("d", "Diamond"),

            ("≡", "Crafting Table"),

            ("Ω", "Furnace"),

            ("M", "Mob (any)"),

            ("v","Ladder down"),

            ("^","Ladder up"),

            ("C", "Chest"),

            (" ","Dark / out-of-range"),

        ]

        for glyph, name in legend_items:

            print(f"    {glyph!r:<5}  {name}")



        # Mob details if any mob is present

        all_mobs: list[str] = []

        for r in range(OBS_ROWS):

            for c in range(OBS_COLS):

                all_mobs.extend(mob_present[r][c])

        if all_mobs:

            print(f"\n  Mobs in view: {', '.join(sorted(set(all_mobs)))}")





def print_map_detailed(decoded: dict[str, Any]) -> None:

    """Print a per-cell breakdown of active channels (verbose)."""

    block_ids  = decoded["block_ids"]

    item_ids   = decoded["item_ids"]

    light      = decoded["light"]

    mob_present = decoded["mob_present"]



    print(_section("DETAILED CELL BREAKDOWN"))

    for r in range(OBS_ROWS):

        for c in range(OBS_COLS):

            label = f"  tile ({r},{c})"

            if r == OBS_ROWS // 2 and c == OBS_COLS // 2:

                label += " [PLAYER]"

            block_name = BLOCK_TYPES[block_ids[r, c]]

            item_name  = ITEM_TYPES[item_ids[r, c]]

            lit        = bool(light[r, c] > 0.5)

            mobs       = mob_present[r][c]

            if not lit and block_name == "DARKNESS":

                continue  # skip dark / trivial cells

            parts = [f"block={block_name}"]

            if item_name != "NONE":

                parts.append(f"item={item_name}")

            if mobs:

                parts.append(f"mobs=[{', '.join(mobs)}]")

            parts.append(f"lit={lit}")

            print(f"{label}: {', '.join(parts)}")





# -----------------------------------------------------------------------------

# 3.  High-level analysis entry point

# -----------------------------------------------------------------------------



def analyze(obs: np.ndarray, verbose_cells: bool = False) -> None:

    """Full analysis of a single observation vector."""

    obs = np.asarray(obs, dtype=np.float32).ravel()

    print(_hr("="))

    print(f"  Observation length: {len(obs):,}")

    print(f"  Value range: [{obs.min():.4f}, {obs.max():.4f}]   mean={obs.mean():.4f}")

    print(_hr("="))



    decoded = decode_obs(obs)

    print_inventory(decoded)

    print_map(decoded, show_legend=True)

    if verbose_cells:

        print_map_detailed(decoded)





# -----------------------------------------------------------------------------

# 4.  CLI

# -----------------------------------------------------------------------------




def _load_obs(path: str) -> np.ndarray:
    """Load an observation from a .npy file (single vector or batched)."""
    import os as _os
    if not _os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path!r}\n"
            "\n"
            "  To capture a real Craftax obs, run:\n"
            "    python notebooks/see_craftax_symbolic.py --generate obs.npy\n"
            "  Take N random steps first (e.g. 200):\n"
            "    python notebooks/see_craftax_symbolic.py --generate obs.npy --steps 200\n"
            "\n"
            "  To test with a synthetic observation (no env needed), run:\n"
            "    python notebooks/see_craftax_symbolic.py --demo\n"
        )
    arr = np.load(path)
    if arr.ndim > 1:
        print(f"[info] Loaded batch of shape {arr.shape}; using first element.")
        arr = arr[0]
    return arr.ravel()


def generate_obs(save_path: str, env_name: str = "CraftaxSymbolic-v1",
                 num_steps: int = 0, seed: int = 42) -> np.ndarray:
    """
    Generate a real Craftax symbolic observation and save it as a .npy file.

    Parameters
    ----------
    save_path  : Path to write the .npy file.
    env_name   : CraftaxSymbolic-v1 or CraftaxClassicSymbolic-v1.
    num_steps  : Random steps to take before capturing obs (0 = reset obs).
    seed       : RNG seed.
    """
    print("[generate] Importing JAX / Craftax ...")
    try:
        import jax
    except ImportError:
        raise ImportError("JAX is required. Use --demo for a synthetic obs instead.")
    try:
        if "Classic" in env_name:
            from craftax.craftax_classic.envs.craftax_symbolic_env import (
                CraftaxClassicSymbolicEnv as EnvCls,
            )
        else:
            from craftax.craftax.envs.craftax_symbolic_env import (
                CraftaxSymbolicEnv as EnvCls,
            )
    except ImportError:
        raise ImportError("Craftax not found. Run:  pip install craftax")

    env = EnvCls()
    params = env.default_params
    rng = jax.random.PRNGKey(seed)
    n_actions = env.action_space(params).n

    print(f"[generate] Resetting {env_name} ...")
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, params)

    for step_i in range(num_steps):
        rng, step_rng, act_rng = jax.random.split(rng, 3)
        action = int(jax.random.randint(act_rng, (), 0, n_actions))
        obs, state, reward, done, info = env.step(step_rng, state, action, params)
        if (step_i + 1) % 50 == 0:
            print(f"  step {step_i + 1}/{num_steps}")
        if done:
            print(f"  Episode ended at step {step_i + 1}, resetting.")
            rng, reset_rng = jax.random.split(rng)
            obs, state = env.reset(reset_rng, params)

    obs_np = np.asarray(obs, dtype=np.float32).ravel()
    np.save(save_path, obs_np)
    print(f"[generate] Saved obs shape={obs_np.shape} to {save_path!r}")
    return obs_np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--schema", action="store_true",
        help="Print the full observation-vector schema and exit.",
    )
    parser.add_argument(
        "--generate", metavar="OUT.npy", nargs="?", const="obs.npy",
        help="Generate a real Craftax obs, save to OUT.npy (default: obs.npy), "
             "then analyze it.",
    )
    parser.add_argument(
        "--steps", type=int, default=0, metavar="N",
        help="(with --generate) Random actions before capturing obs (default: 0).",
    )
    parser.add_argument(
        "--env", default="CraftaxSymbolic-v1",
        choices=["CraftaxSymbolic-v1", "CraftaxClassicSymbolic-v1"],
        help="(with --generate) Craftax symbolic env variant.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for --generate / --demo (default: 42).",
    )
    parser.add_argument(
        "--analyze", metavar="OBS.npy",
        help="Decode and pretty-print every field of a saved observation.",
    )
    parser.add_argument(
        "--map", metavar="OBS.npy",
        help="Render an ASCII map grid from the given observation file.",
    )
    parser.add_argument(
        "--verbose-cells", action="store_true",
        help="(with --analyze / --generate) Print per-cell channel breakdown.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Analyze a synthetic random obs (no Craftax env needed).",
    )
    args = parser.parse_args()

    ran_something = False

    if args.schema:
        print_schema()
        ran_something = True

    if args.generate:
        obs = generate_obs(
            args.generate,
            env_name=args.env,
            num_steps=args.steps,
            seed=args.seed,
        )
        print()
        analyze(obs, verbose_cells=args.verbose_cells)
        ran_something = True

    if args.analyze:
        obs = _load_obs(args.analyze)
        analyze(obs, verbose_cells=args.verbose_cells)
        ran_something = True
    elif args.map:
        obs = _load_obs(args.map)
        decoded = decode_obs(obs)
        print_map(decoded)
        ran_something = True

    if args.demo:
        print("\n[demo] Generating a random synthetic observation ...\n")
        rng = np.random.default_rng(args.seed)
        obs = rng.random(MAP_SIZE + 51).astype(np.float32)
        analyze(obs)
        ran_something = True

    if not ran_something:
        parser.print_help()
        print()
        print("Quick-start:")
        print("  --generate obs.npy                   capture a real obs then analyze")
        print("  --generate obs.npy --steps 200        take 200 random steps first")
        print("  --generate obs.npy --env CraftaxClassicSymbolic-v1")
        print("  --analyze  obs.npy                   analyze a saved obs")
        print("  --analyze  obs.npy --verbose-cells    + per-cell channel dump")
        print("  --map      obs.npy                   ASCII map only")
        print("  --schema                             print full vector layout")
        print("  --demo                               test with a synthetic obs")


if __name__ == "__main__":
    main()

"""Canonical 13-group split of the 64-D G1 proprio vector.

Layout (single source of truth, mirroring the G1 actuator order:
legs(12) + waist(3) + arms(14))::

    proprio[ 0:29] = jp        (rad)        legs(12) + waist(3) + arms(14)
    proprio[29:31] = hand_bits (binary)     left/right
    proprio[31:60] = jv        (rad/s)      same partition as jp
    proprio[60:64] = quat_xyzw (pelvis)

The 13 groups below mirror the actuator partition used by Isaac-GR00T's
``unitree_g1_sonic`` embodiment
(``gr00t/configs/data/embodiment_configs.py:67-114``), extended to cover the
extra ``jv`` and ``quat`` segments that Kappa's proprio carries.

The split is structural: it lets per-group normalization schemes be swapped
in later (sin/cos for ``quat``, mean-std for ``*_vel``, pass-through for
``*_hand_bit``) without changing the on-disk format. Per-dim ``q01/q99``
values themselves are bit-identical to a flat 64-D computation.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

G1_PROPRIO_DIM = 64

# Ordered (name, start, end) with end exclusive. Concatenating slices in this
# order MUST reconstruct the full 64-D proprio.
PROPRIO_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("left_leg_pos",   0,  6),
    ("right_leg_pos",  6, 12),
    ("waist_pos",     12, 15),
    ("left_arm_pos",  15, 22),
    ("right_arm_pos", 22, 29),
    ("left_hand_bit", 29, 30),
    ("right_hand_bit", 30, 31),
    ("left_leg_vel",  31, 37),
    ("right_leg_vel", 37, 43),
    ("waist_vel",     43, 46),
    ("left_arm_vel",  46, 53),
    ("right_arm_vel", 53, 60),
    ("quat",          60, 64),
)

# Sanity: the table above must tile [0, 64) exactly.
_check_total = sum(end - start for _, start, end in PROPRIO_GROUPS)
assert _check_total == G1_PROPRIO_DIM, (
    f"PROPRIO_GROUPS dims sum to {_check_total}, expected {G1_PROPRIO_DIM}"
)
_check_cursor = 0
for _name, _start, _end in PROPRIO_GROUPS:
    assert _start == _check_cursor, (
        f"PROPRIO_GROUPS gap/overlap at group {_name!r}: start={_start}, expected {_check_cursor}"
    )
    _check_cursor = _end
del _check_total, _check_cursor


def assemble_flat_from_groups(
    grouped: Mapping[str, Mapping[str, list | np.ndarray]],
    stat_keys: tuple[str, ...] = ("min", "max", "q01", "q99"),
) -> dict[str, np.ndarray]:
    """Concat per-group stats into flat 64-D vectors in layout order.

    ``grouped`` maps group name -> {stat_key -> per-dim list/array}, matching
    the on-disk grouped JSON layout. Returns ``{stat_key: (64,) np.float64}``.

    Raises ``KeyError`` (with the offending group named) if any expected
    group or stat key is missing, or ``ValueError`` if a per-group array
    length disagrees with the layout slice width.
    """
    out: dict[str, np.ndarray] = {}
    for stat_key in stat_keys:
        parts: list[np.ndarray] = []
        for name, start, end in PROPRIO_GROUPS:
            section = grouped.get(name)
            if section is None:
                raise KeyError(
                    f"proprio stats missing group {name!r} (expected for "
                    f"the {G1_PROPRIO_DIM}-D layout)"
                )
            if stat_key not in section:
                raise KeyError(
                    f"proprio stats group {name!r} missing stat {stat_key!r}"
                )
            arr = np.asarray(section[stat_key], dtype=np.float64)
            expected = end - start
            if arr.shape != (expected,):
                raise ValueError(
                    f"proprio stats group {name!r} stat {stat_key!r}: "
                    f"shape {arr.shape}, expected ({expected},)"
                )
            parts.append(arr)
        out[stat_key] = np.concatenate(parts, axis=0)
        if out[stat_key].shape != (G1_PROPRIO_DIM,):
            raise ValueError(
                f"assembled flat {stat_key!r} has shape {out[stat_key].shape}, "
                f"expected ({G1_PROPRIO_DIM},)"
            )
    return out

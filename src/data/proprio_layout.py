"""Canonical 8-group split of the 34-D ``unitree_g1_sonic`` proprio vector.

Layout (single source of truth, matching Isaac-GR00T's
``unitree_g1_sonic`` embodiment in
``gr00t/configs/data/embodiment_configs.py:67-114``)::

    proprio[ 0: 6] = left_leg            (rad)
    proprio[ 6:12] = right_leg           (rad)
    proprio[12:15] = waist               (rad)
    proprio[15:22] = left_arm            (rad)
    proprio[22:29] = right_arm           (rad)
    proprio[29:30] = left_hand           (binary open/close)
    proprio[30:31] = right_hand          (binary open/close)
    proprio[31:34] = projected_gravity   (unitless, base frame)

The names + ordering are re-exported from
``src.data.kappa_modality_configs`` so anything still importing
``PROPRIO_GROUPS`` / ``G1_PROPRIO_DIM`` stays in lockstep with the
canonical embodiment config.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from src.data.kappa_modality_configs import (
    G1_SONIC_PROPRIO_DIM,
    SONIC_STATE_LAYOUT,
)


G1_PROPRIO_DIM = G1_SONIC_PROPRIO_DIM
PROPRIO_GROUPS: tuple[tuple[str, int, int], ...] = SONIC_STATE_LAYOUT


def assemble_flat_from_groups(
    grouped: Mapping[str, Mapping[str, list | np.ndarray]],
    stat_keys: tuple[str, ...] = ("min", "max", "q01", "q99"),
) -> dict[str, np.ndarray]:
    """Concat per-group stats into flat 34-D vectors in layout order.

    ``grouped`` maps group name -> {stat_key -> per-dim list/array}, matching
    the on-disk grouped JSON layout. Returns ``{stat_key: (34,) np.float64}``.
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

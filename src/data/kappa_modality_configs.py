"""Embodiment-level modality configs for Kappa.

The ``unitree_g1_sonic`` block mirrors Isaac-GR00T's entry of the same name
(``gr00t/configs/data/embodiment_configs.py:67-114``). Only the ``state``
modality is needed in Kappa — video / language / action go through Kappa's
own SONIC latent pipeline, not the GR00T modality processors.
"""
from __future__ import annotations

from src.data.modality_types import ModalityConfig


UNITREE_G1_SONIC_STATE = ModalityConfig(
    delta_indices=(0,),
    modality_keys=(
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "projected_gravity",
    ),
    sin_cos_embedding_keys=None,
    mean_std_embedding_keys=None,
)


KAPPA_MODALITY_CONFIGS: dict[str, dict[str, ModalityConfig]] = {
    "unitree_g1_sonic": {
        "state": UNITREE_G1_SONIC_STATE,
    },
}


# Flat (name, start, end) layout of the 34-D sonic proprio vector, in
# ``modality_keys`` order so the dataset dump scripts, ``StateNormalizer.
# normalize_flat``, deploy ``_assemble_proprio`` and the unit tests share a
# single source of truth.
SONIC_STATE_LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("left_leg",          0,  6),
    ("right_leg",         6, 12),
    ("waist",            12, 15),
    ("left_arm",         15, 22),
    ("right_arm",        22, 29),
    ("left_hand",        29, 30),
    ("right_hand",       30, 31),
    ("projected_gravity", 31, 34),
)

G1_SONIC_PROPRIO_DIM = 34

_total = sum(e - s for _, s, e in SONIC_STATE_LAYOUT)
assert _total == G1_SONIC_PROPRIO_DIM, (
    f"SONIC_STATE_LAYOUT dims sum to {_total}, expected {G1_SONIC_PROPRIO_DIM}"
)
_cursor = 0
for _name, _start, _end in SONIC_STATE_LAYOUT:
    assert _start == _cursor, (
        f"SONIC_STATE_LAYOUT gap/overlap at {_name!r}: start={_start}, expected {_cursor}"
    )
    _cursor = _end
del _total, _cursor

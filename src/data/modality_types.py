"""Minimal modality types mirroring Isaac-GR00T (no gr00t dependency).

Field names match ``Isaac-GR00T/gr00t/configs/data/types.py`` and
``embodiment_configs.py`` 1:1 so config dicts can be copy-pasted across the
two trees. ``tuple`` is used instead of ``list`` to keep the dataclasses
frozen / hashable for downstream dispatch caches.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionRepresentation(str, Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    DELTA = "delta"


class ActionType(str, Enum):
    EEF = "eef"
    NON_EEF = "non_eef"


class ActionFormat(str, Enum):
    DEFAULT = "default"
    XYZ_ROT6D = "xyz_rot6d"
    XYZ_ROTVEC = "xyz_rotvec"


@dataclass(frozen=True)
class ActionConfig:
    rep: ActionRepresentation
    type: ActionType
    format: ActionFormat
    state_key: str | None = None


@dataclass(frozen=True)
class ModalityConfig:
    delta_indices: tuple[int, ...]
    modality_keys: tuple[str, ...]
    sin_cos_embedding_keys: tuple[str, ...] | None = None
    mean_std_embedding_keys: tuple[str, ...] | None = None
    action_configs: tuple[ActionConfig, ...] | None = None

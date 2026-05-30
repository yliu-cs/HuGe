"""Canonical schemas for Kappa's internal 66-D SONIC action format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


MOTION_TOKEN_DIM = 64
HAND_DIM = 2
ACTION_DIM = MOTION_TOKEN_DIM + HAND_DIM
ACTION_HORIZON = 50
CONTROL_FREQ_HZ = 50
PROPRIO_DIM = 34
HAND_POLARITY = "closed_1_open_0"
ACTION_SCHEMA_VERSION = "kappa_sonic_66_v1"
PROPRIO_LAYOUT_VERSION = "unitree_g1_sonic v1"

PROPRIO_LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("left_leg", 0, 6),
    ("right_leg", 6, 12),
    ("waist", 12, 15),
    ("left_arm", 15, 22),
    ("right_arm", 22, 29),
    ("left_hand", 29, 30),
    ("right_hand", 30, 31),
    ("projected_gravity", 31, 34),
)


@dataclass(frozen=True)
class ActionSchema:
    schema_version: str = ACTION_SCHEMA_VERSION
    action_dim: int = ACTION_DIM
    motion_token_dim: int = MOTION_TOKEN_DIM
    hand_dim: int = HAND_DIM
    hand_polarity: str = HAND_POLARITY
    action_horizon: int = ACTION_HORIZON
    control_freq_hz: int = CONTROL_FREQ_HZ

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": self.schema_version,
            "action_dim": self.action_dim,
            "motion_token_dim": self.motion_token_dim,
            "hand_dim": self.hand_dim,
            "hand_polarity": self.hand_polarity,
            "action_horizon": self.action_horizon,
            "control_freq_hz": self.control_freq_hz,
        }


def validate_proprio_layout(layout: tuple[tuple[str, int, int], ...] = PROPRIO_LAYOUT) -> None:
    cursor = 0
    for name, start, end in layout:
        if start != cursor:
            raise ValueError(f"proprio layout gap/overlap at {name}: {start} != {cursor}")
        if end <= start:
            raise ValueError(f"proprio layout group {name} has empty range")
        cursor = end
    if cursor != PROPRIO_DIM:
        raise ValueError(f"proprio layout ends at {cursor}, expected {PROPRIO_DIM}")


def validate_action_array(action: np.ndarray, *, allow_64d: bool = False) -> None:
    if action.ndim < 2:
        raise ValueError(f"action must be at least 2-D [..., D], got {action.shape}")
    dim = action.shape[-1]
    valid_dims = (ACTION_DIM, MOTION_TOKEN_DIM) if allow_64d else (ACTION_DIM,)
    if dim not in valid_dims:
        raise ValueError(f"action dim {dim} is not in {valid_dims}")
    if not np.isfinite(action[..., : min(dim, MOTION_TOKEN_DIM)]).all():
        raise ValueError("action motion token contains NaN or Inf")


def pad_egodex_action(action64: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad EgoDex 64-D SONIC latents into the 66-D container.

    Returns ``(action66, mask66)`` with hand dims masked out.
    """
    validate_action_array(action64, allow_64d=True)
    if action64.shape[-1] == ACTION_DIM:
        action = action64.astype(np.float32, copy=False)
    else:
        action = np.zeros((*action64.shape[:-1], ACTION_DIM), dtype=np.float32)
        action[..., :MOTION_TOKEN_DIM] = action64.astype(np.float32, copy=False)
    mask = np.ones_like(action, dtype=np.float32)
    mask[..., MOTION_TOKEN_DIM:] = 0.0
    return action, mask


def split_action_dict(action: np.ndarray) -> dict[str, np.ndarray]:
    validate_action_array(action)
    return {
        "action.motion_token": action[..., :MOTION_TOKEN_DIM],
        "action.hand_binary": action[..., MOTION_TOKEN_DIM:ACTION_DIM],
    }


def combine_action_dict(parts: Mapping[str, np.ndarray]) -> np.ndarray:
    motion = parts.get("action.motion_token", parts.get("motion_token"))
    hand = parts.get("action.hand_binary", parts.get("hand_binary_action"))
    if motion is None or hand is None:
        raise KeyError("expected motion token and hand binary action keys")
    if motion.shape[:-1] != hand.shape[:-1]:
        raise ValueError(f"motion/hand shape prefix mismatch: {motion.shape} vs {hand.shape}")
    if motion.shape[-1] != MOTION_TOKEN_DIM or hand.shape[-1] != HAND_DIM:
        raise ValueError(f"expected motion 64 and hand 2 dims, got {motion.shape[-1]}, {hand.shape[-1]}")
    return np.concatenate([motion, hand], axis=-1).astype(np.float32, copy=False)


def hand_binary_is_valid(action: np.ndarray, atol: float = 1e-4) -> bool:
    validate_action_array(action)
    hand = action[..., MOTION_TOKEN_DIM:ACTION_DIM]
    return bool(np.all((np.abs(hand - 0.0) <= atol) | (np.abs(hand - 1.0) <= atol)))


validate_proprio_layout()

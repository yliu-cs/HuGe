"""Deployment observation parsing."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from kappa.data.schemas import PROPRIO_DIM


def projected_gravity_from_base_quat(quat_wxyz: np.ndarray | None) -> np.ndarray:
    if quat_wxyz is None:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    q = q / norm
    rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
    return rot.inv().apply(np.asarray([0.0, 0.0, -1.0])).astype(np.float32)


def _resolve(obs: dict[str, Any], key: str) -> Any:
    """Resolve a dotted key, accepting flat, nested, and bare-tail layouts.

    ``state.base_quat`` matches ``obs["state.base_quat"]`` (flat),
    ``obs["state"]["base_quat"]`` (nested), or ``obs["base_quat"]`` (top level).
    """
    if key in obs:
        return obs[key]
    head, _, tail = key.partition(".")
    sub = obs.get(head)
    if isinstance(sub, dict) and tail and tail in sub:
        return sub[tail]
    if tail and tail in obs:
        return obs[tail]
    return None


def _get(obs: dict[str, Any], key: str, width: int, default: float = 0.0) -> np.ndarray:
    val = _resolve(obs, key)
    arr = np.asarray(val if val is not None else np.full((width,), default), dtype=np.float32).reshape(-1)
    if arr.size < width:
        arr = np.pad(arr, (0, width - arr.size))
    return arr[:width]


def _resolve_projected_gravity(observation: dict[str, Any]) -> np.ndarray:
    """Prefer a directly-supplied projected_gravity, else derive it from base_quat.

    The deploy client/state-tap may provide either the projected gravity vector
    or the raw base quaternion (flat or nested under ``state``). Earlier code only
    looked up the flat ``base_quat`` key and silently fell back to the upright
    default ``[0, 0, -1]``, freezing the body-orientation channel during walking.
    """
    grav = _resolve(observation, "state.projected_gravity")
    if grav is not None:
        arr = np.asarray(grav, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size))
        return arr[:3]
    return projected_gravity_from_base_quat(_resolve(observation, "state.base_quat"))


def assemble_proprio(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "state" in observation and not isinstance(observation["state"], dict):
        arr = np.asarray(observation["state"], dtype=np.float32)
        if arr.shape[-1] == PROPRIO_DIM:
            return arr.reshape(1, PROPRIO_DIM), np.ones((1, 1), dtype=np.float32)
    hand = _get(observation, "state.hand_binary_state", 2)
    parts = [
        _get(observation, "state.left_leg", 6),
        _get(observation, "state.right_leg", 6),
        _get(observation, "state.waist", 3),
        _get(observation, "state.left_arm", 7),
        _get(observation, "state.right_arm", 7),
        hand[0:1],
        hand[1:2],
        _resolve_projected_gravity(observation),
    ]
    return np.concatenate(parts, axis=0).reshape(1, PROPRIO_DIM).astype(np.float32), np.ones((1, 1), dtype=np.float32)

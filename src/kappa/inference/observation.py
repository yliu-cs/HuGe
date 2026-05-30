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


def _get(obs: dict[str, Any], key: str, width: int, default: float = 0.0) -> np.ndarray:
    val = obs.get(key)
    if val is None and key.startswith("state."):
        val = obs.get("state", {}).get(key.split(".", 1)[1]) if isinstance(obs.get("state"), dict) else None
    arr = np.asarray(val if val is not None else np.full((width,), default), dtype=np.float32).reshape(-1)
    if arr.size < width:
        arr = np.pad(arr, (0, width - arr.size))
    return arr[:width]


def assemble_proprio(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "state" in observation and not isinstance(observation["state"], dict):
        arr = np.asarray(observation["state"], dtype=np.float32)
        if arr.shape[-1] == PROPRIO_DIM:
            return arr.reshape(1, PROPRIO_DIM), np.ones((1, 1), dtype=np.float32)
    hand = _get(observation, "state.hand_binary_state", 2)
    base_quat = observation.get("state.base_quat", observation.get("base_quat"))
    parts = [
        _get(observation, "state.left_leg", 6),
        _get(observation, "state.right_leg", 6),
        _get(observation, "state.waist", 3),
        _get(observation, "state.left_arm", 7),
        _get(observation, "state.right_arm", 7),
        hand[0:1],
        hand[1:2],
        projected_gravity_from_base_quat(base_quat),
    ]
    return np.concatenate(parts, axis=0).reshape(1, PROPRIO_DIM).astype(np.float32), np.ones((1, 1), dtype=np.float32)

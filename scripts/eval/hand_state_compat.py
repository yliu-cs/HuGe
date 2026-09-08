"""Reproduce the cola training hand labels without changing the GR00T client."""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from typing import Any, Iterator

import numpy as np


LEGACY_MODE = "legacy-inspire-six-channel"
FORMULA_VERSION = "inspire_max_close_first6_ge_0.5_v1"
COMPAT_NOTE = "Reproduces training labels, including thumb rotation; not a measurement of grasp success."


def legacy_inspire_hand_state(left: Any, right: Any) -> np.ndarray:
    """Keep the original exporter formula, including the rotation channel."""
    values = []
    for name, joints in (("left_hand_q", left), ("right_hand_q", right)):
        q = np.asarray(joints, dtype=np.float64)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError(f"Invalid {name}: expected finite [7] Inspire feedback")
        values.append(float(np.max(1.0 - np.clip(q[:6], 0.0, 1.0)) >= 0.5))
    return np.asarray(values, dtype=np.float32)


def validate_compat_api(api: Any) -> None:
    parameters = inspect.signature(api.prepare_observation_from_sensors).parameters
    required = {"camera_subscriber", "state_subscriber", "robot_model", "language_prompt", "hand_type", "log_errors"}
    if not required.issubset(parameters):
        raise RuntimeError("Unsupported VLA observation assembly signature for hand-state compatibility")


class HandStateObservation(dict):
    """Audit data is a Python attribute, never an observation wire field."""

    def __init__(self, observation: dict[str, Any], audit: dict[str, Any]):
        super().__init__(observation)
        self.hand_state_compat = audit


class _StateSnapshot:
    def __init__(self, subscriber: Any):
        self.subscriber = subscriber
        self.raw: dict[str, np.ndarray] | None = None
        self.after: np.ndarray | None = None

    def get_msg(self, *args: Any, **kwargs: Any) -> Any:
        self.raw = None
        self.after = None
        message = self.subscriber.get_msg(*args, **kwargs)
        if message is None:
            return None
        raw = {key: np.asarray(message.get(key)).copy() for key in ("left_hand_q", "right_hand_q")}
        self.after = legacy_inspire_hand_state(raw["left_hand_q"], raw["right_hand_q"])
        self.raw = raw
        # The original assembly mutates left_hand_q. Give it private arrays.
        return {**message, **{key: value.copy() for key, value in raw.items()}}


@contextmanager
def hand_state_compat_hooks(api: Any) -> Iterator[None]:
    original = api.prepare_observation_from_sensors

    def prepare(camera_subscriber: Any, state_subscriber: Any, robot_model: Any,
                language_prompt: str, hand_type: str = "dex3", log_errors: bool = False) -> Any:
        if hand_type != "inspire":
            raise ValueError("Legacy hand-state compatibility requires hand_type='inspire'")
        snapshot = _StateSnapshot(state_subscriber)
        observation = original(camera_subscriber, snapshot, robot_model, language_prompt,
                               hand_type=hand_type, log_errors=log_errors)
        if observation is None:
            return None
        if snapshot.raw is None or snapshot.after is None:
            raise RuntimeError("Observation has no matching raw hand feedback snapshot")
        state = dict(observation["state"])
        before = np.asarray(state["hand_binary_state"]).reshape(2).copy()
        after = snapshot.after
        state["hand_binary_state"] = after[None, None].copy()
        if "token_state" in state:
            token = np.asarray(state["token_state"]).copy()
            if token.shape != (1, 1, 66):
                raise ValueError(f"Expected assembled token_state [1,1,66], got {token.shape}")
            token[..., 64:66] = after
            state["token_state"] = token
        audit = {"mode": LEGACY_MODE, "formula_version": FORMULA_VERSION,
                 **{key: value.tolist() for key, value in snapshot.raw.items()},
                 "before": before.tolist(), "after": after.tolist()}
        return HandStateObservation({**observation, "state": state}, audit)

    api.prepare_observation_from_sensors = prepare
    try:
        yield
    finally:
        api.prepare_observation_from_sensors = original

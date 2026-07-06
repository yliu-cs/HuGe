"""Adapter from Huge's 66-D chunks to SONIC latent protocol v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from huge.data.schemas import ACTION_DIM, MOTION_TOKEN_DIM, split_action_dict, validate_action_array


OPEN_HAND_7D = np.zeros((7,), dtype=np.float32)
CLOSED_HAND_7D = np.ones((7,), dtype=np.float32)


@dataclass
class HandHysteresis:
    close_threshold: float = 0.6
    open_threshold: float = 0.4
    state: np.ndarray = field(default_factory=lambda: np.zeros((2,), dtype=np.float32))

    def update(self, prob: np.ndarray) -> np.ndarray:
        p = np.asarray(prob, dtype=np.float32).reshape(2)
        self.state = np.where(
            p >= self.close_threshold,
            1.0,
            np.where(p <= self.open_threshold, 0.0, self.state),
        ).astype(np.float32)
        return self.state.copy()

    def reset(self) -> None:
        self.state[:] = 0.0


class SonicAdapter:
    def __init__(
        self,
        *,
        hand_mode: str = "joints7",
        open_hand: np.ndarray | None = None,
        closed_hand: np.ndarray | None = None,
        token_abs_limit: float = 1.25,
    ) -> None:
        if hand_mode not in {"binary", "joints7"}:
            raise ValueError("hand_mode must be 'binary' or 'joints7'")
        self.hand_mode = hand_mode
        self.open_hand = OPEN_HAND_7D if open_hand is None else np.asarray(open_hand, dtype=np.float32)
        self.closed_hand = CLOSED_HAND_7D if closed_hand is None else np.asarray(closed_hand, dtype=np.float32)
        if self.open_hand.shape != (7,) or self.closed_hand.shape != (7,):
            raise ValueError("open_hand and closed_hand must be [7]")
        self.token_abs_limit = float(token_abs_limit)
        self.hysteresis = HandHysteresis()
        self.frame_counter = 0

    def reset(self) -> None:
        self.hysteresis.reset()
        self.frame_counter = 0

    def _hand_to_joints(self, hand_binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = self.closed_hand if hand_binary[0] >= 0.5 else self.open_hand
        right = self.closed_hand if hand_binary[1] >= 0.5 else self.open_hand
        return left[None, :].astype(np.float32), right[None, :].astype(np.float32)

    def validate_motion_token(self, token: np.ndarray) -> None:
        if token.shape != (MOTION_TOKEN_DIM,):
            raise ValueError(f"motion token shape {token.shape}, expected ({MOTION_TOKEN_DIM},)")
        if not np.isfinite(token).all():
            raise ValueError("motion token contains NaN or Inf")
        max_abs = float(np.max(np.abs(token)))
        if max_abs > self.token_abs_limit:
            raise ValueError(f"motion token abs max {max_abs:.4f} exceeds {self.token_abs_limit}")

    def action_to_parts(self, action: np.ndarray, chunk_index: int) -> dict[str, np.ndarray]:
        arr = np.asarray(action, dtype=np.float32)
        validate_action_array(arr)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
            raise ValueError(f"expected action chunk [T, {ACTION_DIM}], got {arr.shape}")
        idx = int(np.clip(chunk_index, 0, arr.shape[0] - 1))
        parts = split_action_dict(arr)
        token = np.asarray(parts["action.motion_token"][idx], dtype=np.float32)
        self.validate_motion_token(token)
        hand_raw = np.asarray(parts["action.hand_binary"][idx], dtype=np.float32)
        if hand_raw.min(initial=0.0) < 0.0 or hand_raw.max(initial=0.0) > 1.0:
            hand_prob = 1.0 / (1.0 + np.exp(-hand_raw))
        else:
            hand_prob = hand_raw
        hand_binary = self.hysteresis.update(hand_prob)
        return {"motion_token": token, "hand_binary": hand_binary}

    def to_protocol_v4(self, action: np.ndarray, chunk_index: int) -> dict[str, Any]:
        parts = self.action_to_parts(action, chunk_index)
        token = parts["motion_token"][None, :].astype(np.float32)
        hand_binary = parts["hand_binary"]
        msg: dict[str, Any] = {
            "topic": "pose",
            "version": 4,
            "pose_data": {
                "token_state": token,
                "frame_index": np.asarray([self.frame_counter], dtype=np.int64),
            },
        }
        if self.hand_mode == "binary":
            msg["pose_data"]["hand_binary"] = hand_binary[None, :].astype(np.float32)
        else:
            left, right = self._hand_to_joints(hand_binary)
            msg["pose_data"]["left_hand_joints"] = left
            msg["pose_data"]["right_hand_joints"] = right
        self.frame_counter += 1
        return msg

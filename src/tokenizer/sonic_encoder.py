"""SONIC encoder wrapper for Universal Latent Token generation.

Output token = [64-D SONIC latent | 1-D left-hand binary | 1-D right-hand binary].
The two hand bits are derived from `observation.hand_joints[LEFT/RIGHT_JOINT_IDX]` via
two global percentile thresholds on |theta| plus an Octo-style reverse causal scan that
labels transition frames with the next definitive state (see scripts/calc_hand_thresholds.py).
"""
from __future__ import annotations

import json
import os

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

ort.set_default_logger_severity(3)

OBS_DIM = 1762
SONIC_LATENT_DIM = 64
HAND_BITS = 2
TOKEN_DIM = SONIC_LATENT_DIM + HAND_BITS  # 66
JP_OFFSET = 4
JV_OFFSET = 294
ANCHOR_OFFSET = 601
LOOKAHEAD_K = 10
LOOKAHEAD_STRIDE = 5

LEFT_HAND_JOINT_IDX = 4
RIGHT_HAND_JOINT_IDX = 11

MUJOCO_TO_ISAACLAB = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]

DEFAULT_ANGLES_MUJOCO = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    out = np.stack([q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3], q_wxyz[..., 0]], axis=-1)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=out.dtype)
    return np.where(norms < 1e-10, identity, out)


def _build_lookahead_index(n: int) -> np.ndarray:
    base = np.arange(n, dtype=np.int64)[:, None]
    offsets = (LOOKAHEAD_STRIDE * np.arange(LOOKAHEAD_K, dtype=np.int64))[None, :]
    return np.minimum(base + offsets, n - 1)


def _vectorized_anchor(odom_quat_wxyz: np.ndarray) -> np.ndarray:
    n = len(odom_quat_wxyz)
    idx = _build_lookahead_index(n)  # (n, K)

    quat_xyzw = _quat_wxyz_to_xyzw(odom_quat_wxyz.astype(np.float64))
    rots = R.from_quat(quat_xyzw)  # (n,)
    cur = rots[np.arange(n)]
    tgt = rots[idx.reshape(-1)]
    cur_inv = cur.inv()
    cur_inv_rep = R.from_quat(np.repeat(cur_inv.as_quat(), LOOKAHEAD_K, axis=0))
    rel = cur_inv_rep * tgt
    mat = rel.as_matrix().reshape(n, LOOKAHEAD_K, 3, 3)
    return mat[..., :, :2].reshape(n, LOOKAHEAD_K * 6).astype(np.float32)


def _build_encoder_batch(
    joint_pos_mujoco: np.ndarray,
    odom_quat_wxyz: np.ndarray,
    imu_gyro: np.ndarray,
) -> np.ndarray:
    n = len(joint_pos_mujoco)
    jp = joint_pos_mujoco.astype(np.float64, copy=False)
    jv = np.zeros_like(jp)
    jv[1:] = (jp[1:] - jp[:-1]) * 30.0

    jp_isaac = jp[:, MUJOCO_TO_ISAACLAB].astype(np.float32)
    jv_isaac = jv[:, MUJOCO_TO_ISAACLAB].astype(np.float32)

    idx = _build_lookahead_index(n)  # (n, K)
    jp_win = jp_isaac[idx].reshape(n, LOOKAHEAD_K * 29)
    jv_win = jv_isaac[idx].reshape(n, LOOKAHEAD_K * 29)
    anchor = _vectorized_anchor(odom_quat_wxyz)

    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    # obs[:, 0:4] is the mode one-hot region; G1 mode → obs[:, 0] = 0 (already zeros)
    obs[:, JP_OFFSET:JP_OFFSET + 290] = jp_win
    obs[:, JV_OFFSET:JV_OFFSET + 290] = jv_win
    # obs[:, 584:600] left at zero (legacy 10+1+6 region)
    obs[:, ANCHOR_OFFSET:ANCHOR_OFFSET + 60] = anchor
    return obs


def _make_providers(device_id: int | None) -> list:
    if device_id is None:
        return ["CPUExecutionProvider"]
    return [("CUDAExecutionProvider", {"device_id": int(device_id)}), "CPUExecutionProvider"]


HandThresholds = tuple[float, float, float, float]
"""(left_low, left_high, right_low, right_high) thresholds on |theta|."""


def load_hand_thresholds(threshold_json_path: str) -> HandThresholds:
    """Read (left_low, left_high, right_low, right_high) from the meta JSON written
    by scripts/calc_hand_thresholds.py.
    """
    with open(threshold_json_path) as f:
        meta = json.load(f)
    if meta.get("schema_version") != 2:
        raise ValueError(
            f"hand_threshold.json schema_version={meta.get('schema_version')} unsupported; "
            f"expected 2. Re-run scripts/calc_hand_thresholds.py."
        )
    return (
        float(meta["left"]["low_abs"]),
        float(meta["left"]["high_abs"]),
        float(meta["right"]["low_abs"]),
        float(meta["right"]["high_abs"]),
    )


def _resolve_with_scan(abs_theta: np.ndarray, low: float, high: float) -> np.ndarray:
    """Octo-style three-state classification + reverse propagation.

    Frames with |theta|>high are 'closed' (1), |theta|<low are 'open' (0); intermediate
    frames inherit the *next* definitive label along time (so a transition into closed is
    labeled closed throughout). Any leading intermediate run is then filled forward from
    the first definitive state. If no frame is definitive (rare), defaults to 0 (open).
    """
    n = abs_theta.shape[0]
    out = np.where(abs_theta > high, 1.0, np.where(abs_theta < low, 0.0, np.nan)).astype(np.float32)
    # Reverse scan: NaN takes the carry from the right.
    last = np.nan
    for i in range(n - 1, -1, -1):
        v = out[i]
        if np.isnan(v):
            out[i] = last
        else:
            last = v
    # Forward fill any remaining leading NaN run.
    if np.isnan(out).any():
        first_idx = np.argmax(~np.isnan(out)) if (~np.isnan(out)).any() else -1
        first_val = float(out[first_idx]) if first_idx >= 0 else 0.0
        out = np.where(np.isnan(out), first_val, out).astype(np.float32)
    return out


def binarize_hands(hand_joints: np.ndarray, thresholds: HandThresholds) -> np.ndarray:
    """Map (n, 14) raw hand_joints to (n, 2) binary [left_closed, right_closed] using
    two-threshold + reverse-scan smoothing. Thresholds come from
    scripts/calc_hand_thresholds.py.
    """
    if hand_joints.ndim != 2 or hand_joints.shape[1] <= max(LEFT_HAND_JOINT_IDX, RIGHT_HAND_JOINT_IDX):
        raise ValueError(f"hand_joints must be (n, >=14), got {hand_joints.shape}")
    l_low, l_high, r_low, r_high = thresholds
    left = _resolve_with_scan(np.abs(hand_joints[:, LEFT_HAND_JOINT_IDX]), l_low, l_high)
    right = _resolve_with_scan(np.abs(hand_joints[:, RIGHT_HAND_JOINT_IDX]), r_low, r_high)
    return np.stack([left, right], axis=1).astype(np.float32)


class SonicEncoder:
    def __init__(self, encoder_path: str, device_id: int | None = 0):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            encoder_path, sess_options=sess_opts, providers=_make_providers(device_id),
        )
        active = self.session.get_providers()
        if device_id is not None and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"Requested GPU {device_id} but ORT fell back to {active}. "
                "Make sure torch is imported before onnxruntime so cuDNN is discoverable, "
                "and that onnxruntime-gpu is installed."
            )
        self.device_id = device_id

    def encode_episode(
        self,
        joint_pos_mujoco: np.ndarray,
        odom_quat: np.ndarray,
        imu_gyro: np.ndarray,
        hand_joints: np.ndarray,
        hand_thresholds: HandThresholds,
    ) -> np.ndarray:
        obs = _build_encoder_batch(joint_pos_mujoco, odom_quat, imu_gyro)
        n = len(obs)
        if hand_joints.shape[0] != n:
            raise ValueError(
                f"hand_joints length {hand_joints.shape[0]} != joint_pos length {n}"
            )
        tokens = np.empty((n, TOKEN_DIM), dtype=np.float32)
        run = self.session.run
        for i in range(n):
            tokens[i, :SONIC_LATENT_DIM] = run(None, {"obs_dict": obs[i:i + 1]})[0][0]
        tokens[:, SONIC_LATENT_DIM:] = binarize_hands(hand_joints, hand_thresholds)
        return tokens

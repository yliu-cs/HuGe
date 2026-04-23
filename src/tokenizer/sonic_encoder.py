"""SONIC encoder wrapper for Universal Latent Token generation."""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R

ort.set_default_logger_severity(3)

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


def _quat_wxyz_to_xyzw(q):
    xyzw = np.array([q[1], q[2], q[3], q[0]])
    if np.linalg.norm(xyzw) < 1e-10:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return xyzw


def _compute_anchor_orientation(robot_quat_wxyz, ref_quat_wxyz, current_frame, total_frames):
    result = np.zeros(60, dtype=np.float32)
    robot_rot = R.from_quat(_quat_wxyz_to_xyzw(robot_quat_wxyz))
    for f in range(10):
        target = min(current_frame + f * 5, total_frames - 1)
        ref_rot = R.from_quat(_quat_wxyz_to_xyzw(ref_quat_wxyz[target]))
        rel_rot = robot_rot.inv() * ref_rot
        mat = rel_rot.as_matrix()
        result[f * 6:(f + 1) * 6] = mat[:, :2].flatten()
    return result


def _build_encoder_input(
    joint_pos_isaac, joint_vel_isaac, odom_quat, current_frame, total_frames,
):
    obs = np.zeros(1762, dtype=np.float32)
    offset = 0

    obs[offset] = 0.0  # G1 mode
    offset += 4

    for f in range(10):
        t = min(current_frame + f * 5, total_frames - 1)
        obs[offset + f * 29:offset + (f + 1) * 29] = joint_pos_isaac[t]
    offset += 290

    for f in range(10):
        t = min(current_frame + f * 5, total_frames - 1)
        obs[offset + f * 29:offset + (f + 1) * 29] = joint_vel_isaac[t]
    offset += 290

    offset += 10 + 1 + 6

    anchor = _compute_anchor_orientation(
        odom_quat[current_frame], odom_quat,
        current_frame=current_frame, total_frames=total_frames,
    )
    obs[offset:offset + 60] = anchor
    offset += 60

    return obs.reshape(1, -1)


class SonicEncoder:
    def __init__(self, encoder_path: str):
        self.session = ort.InferenceSession(encoder_path)

    def encode_episode(
        self,
        joint_pos_mujoco: np.ndarray,
        odom_quat: np.ndarray,
        imu_gyro: np.ndarray,
    ) -> np.ndarray:
        n = len(joint_pos_mujoco)
        joint_vel = np.zeros_like(joint_pos_mujoco)
        joint_vel[1:] = (joint_pos_mujoco[1:] - joint_pos_mujoco[:-1]) * 30.0

        jp_isaac = joint_pos_mujoco[:, MUJOCO_TO_ISAACLAB]
        jv_isaac = joint_vel[:, MUJOCO_TO_ISAACLAB]

        tokens = np.zeros((n, 64), dtype=np.float32)
        for i in range(n):
            enc_input = _build_encoder_input(jp_isaac, jv_isaac, odom_quat, i, n)
            tokens[i] = self.session.run(None, {"obs_dict": enc_input})[0][0]
        return tokens

    def encode_frames(
        self,
        joint_pos_mujoco: np.ndarray,
        odom_quat: np.ndarray,
        imu_gyro: np.ndarray,
        frame_indices: list[int],
    ) -> np.ndarray:
        n = len(joint_pos_mujoco)
        joint_vel = np.zeros_like(joint_pos_mujoco)
        joint_vel[1:] = (joint_pos_mujoco[1:] - joint_pos_mujoco[:-1]) * 30.0

        jp_isaac = joint_pos_mujoco[:, MUJOCO_TO_ISAACLAB]
        jv_isaac = joint_vel[:, MUJOCO_TO_ISAACLAB]

        tokens = np.zeros((len(frame_indices), 64), dtype=np.float32)
        for j, i in enumerate(frame_indices):
            enc_input = _build_encoder_input(jp_isaac, jv_isaac, odom_quat, i, n)
            tokens[j] = self.session.run(None, {"obs_dict": enc_input})[0][0]
        return tokens

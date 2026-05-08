"""ONNX SONIC decoder wrapper for closed-loop sim validation.

The deployment-time decoder (`resource/model/sonic/model_decoder.onnx`) takes a
994-D obs vector and emits a 29-D IsaacLab-order action delta. The eval
pipeline calls :meth:`SonicDecoder.step` once per control tick with the
current SONIC token (taken from a Kappa policy chunk) and a snapshot of the
robot state; it returns both the raw IsaacLab action and the corresponding
29-D MuJoCo joint targets.

Obs layout (matches ``resource/model/sonic/observation_config.yaml`` and the
runtime obs registry in ``gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp``)::

    [token_state(64),
     his_base_angular_velocity_10frame_step1(30),       # 10 × 3
     his_body_joint_positions_10frame_step1(290),       # 10 × 29
     his_body_joint_velocities_10frame_step1(290),      # 10 × 29
     his_last_actions_10frame_step1(290),               # 10 × 29
     his_gravity_dir_10frame_step1(30)]                 # 10 × 3   = 994

History buffers are oldest-first (the runtime's ``newest_first=false``
default). Per-joint quantities are in IsaacLab order; ``body_q`` / ``body_dq``
subtract the IsaacLab-ordered default standing angles to match the
deployment-time pre-processing in
``GatherRobotStateToLogger`` (line ~2826 of g1_deploy_onnx_ref.cpp).

Action de-coding mirrors ``CreatePolicyCommand`` in the same file:

    q_target_mujoco[i] = default_angles_mujoco[i]
                       + g1_action_scale_mujoco[i]
                       * action_isaaclab[isaaclab_to_mujoco[i]]
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Joint mapping arrays
# ---------------------------------------------------------------------------
# Both copied verbatim from
# humanoid everyday/.../gear_sonic_deploy/.../include/policy_parameters.hpp.
# The naming convention mirrors the C++ side:
#   ``isaaclab_to_mujoco[i]`` — given a MuJoCo index ``i``, returns the
#       corresponding IsaacLab index (i.e., values are IsaacLab indices,
#       indexed by MuJoCo position).
#   ``mujoco_to_isaaclab[i]`` — given an IsaacLab index ``i``, returns the
#       corresponding MuJoCo index.

ISAACLAB_TO_MUJOCO = np.asarray([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
], dtype=np.int64)

MUJOCO_TO_ISAACLAB = np.asarray([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int64)

# ---------------------------------------------------------------------------
# Default standing pose (MuJoCo joint order). Identical to ``DEFAULT_QPOS``
# in src/eval/g1_mujoco_env.py.
# ---------------------------------------------------------------------------
DEFAULT_ANGLES_MUJOCO = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Per-motor action scale, MuJoCo joint order. Reproduces the constants in
# policy_parameters.hpp so the decoder output matches deployment-time control:
#   scale = 0.25 * effort_limit / stiffness
#   stiffness = armature * (10 * 2π)^2
# ---------------------------------------------------------------------------
_NATURAL_FREQ = 10.0 * 2.0 * np.pi


def _scale(armature: float, effort_limit: float) -> float:
    stiffness = armature * _NATURAL_FREQ * _NATURAL_FREQ
    return 0.25 * effort_limit / stiffness


_S_5020 = _scale(0.003609725, 25.0)
_S_7520_14 = _scale(0.010177520, 88.0)
_S_7520_22 = _scale(0.025101925, 139.0)
_S_4010 = _scale(0.00425, 5.0)

G1_ACTION_SCALE_MUJOCO = np.array([
    _S_7520_22, _S_7520_22, _S_7520_14, _S_7520_22, _S_5020, _S_5020,
    _S_7520_22, _S_7520_22, _S_7520_14, _S_7520_22, _S_5020, _S_5020,
    _S_7520_14, _S_5020, _S_5020,
    _S_5020, _S_5020, _S_5020, _S_5020, _S_5020, _S_4010, _S_4010,
    _S_5020, _S_5020, _S_5020, _S_5020, _S_5020, _S_4010, _S_4010,
], dtype=np.float64)

# ---------------------------------------------------------------------------
# Decoder shape constants
# ---------------------------------------------------------------------------
HISTORY_FRAMES = 10
TOKEN_DIM = 64  # only the 64-D SONIC latent goes into ``token_state``; the
                # two trailing hand bits in a Kappa SONIC token are routed
                # to a separate hand controller and are not part of the
                # decoder's obs.
OBS_DIM = 994


def _quat_inv_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` (3,) by the inverse of unit quaternion ``quat_wxyz``.

    Used to turn world-frame gravity into the body frame, mirroring the C++
    helper ``quat_rotate(quat_conjugate(base_quat), {0,0,-1})``.
    """
    w, x, y, z = quat_wxyz
    qx, qy, qz = -x, -y, -z
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = w * qx, w * qy, w * qz
    R = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)
    return R @ vec.astype(np.float64)


class SonicDecoder:
    """Stateful wrapper around the deployment SONIC decoder.

    Maintains a 10-frame rolling history of robot state and decoder actions in
    IsaacLab order. The caller is responsible for pushing a fresh state
    snapshot before each :meth:`step` call.
    """

    def __init__(self, decoder_path: str | Path, device_id: int | None = 0):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        providers: list = []
        if device_id is not None:
            providers.append(("CUDAExecutionProvider", {"device_id": int(device_id)}))
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(
            str(decoder_path), sess_options=sess_opts, providers=providers,
        )
        active = self.session.get_providers()
        if device_id is not None and "CUDAExecutionProvider" not in active:
            print(
                f"[sonic-decoder] requested CUDA device {device_id} but ORT fell back "
                f"to {active}. Decoder will run on CPU."
            )
        # Fixed-shape obs buffer; rezeroed every step to avoid stale data
        # leaking into a partial layout if a future obs layout shrinks.
        self._obs_buf = np.zeros((1, OBS_DIM), dtype=np.float32)
        self._reset_history(np.zeros(29, dtype=np.float64))

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _reset_history(self, body_q_il: np.ndarray) -> None:
        zero29 = np.zeros(29, dtype=np.float64)
        zero3 = np.zeros(3, dtype=np.float64)
        gravity_init = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self._hist_q = deque([body_q_il.astype(np.float64).copy() for _ in range(HISTORY_FRAMES)], maxlen=HISTORY_FRAMES)
        self._hist_dq = deque([zero29.copy() for _ in range(HISTORY_FRAMES)], maxlen=HISTORY_FRAMES)
        self._hist_act = deque([zero29.copy() for _ in range(HISTORY_FRAMES)], maxlen=HISTORY_FRAMES)
        self._hist_ang = deque([zero3.copy() for _ in range(HISTORY_FRAMES)], maxlen=HISTORY_FRAMES)
        self._hist_grav = deque([gravity_init.copy() for _ in range(HISTORY_FRAMES)], maxlen=HISTORY_FRAMES)

    def reset(self, qpos_mujoco: np.ndarray) -> None:
        """Re-initialize the rolling history with the robot at ``qpos_mujoco``
        (29-D MuJoCo joint angles), zero velocity, identity orientation, and
        no last action.
        """
        body_q_il = qpos_mujoco[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAACLAB]
        self._reset_history(body_q_il)

    def push_state(
        self,
        qpos_mujoco: np.ndarray,
        qvel_mujoco: np.ndarray,
        base_ang_vel_body: np.ndarray,
        base_quat_wxyz: np.ndarray,
    ) -> None:
        """Append the current robot state to the rolling history.

        Call once per control tick **before** :meth:`step` so that ``step``
        sees the freshest state.
        """
        body_q_il = qpos_mujoco[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAACLAB]
        body_dq_il = qvel_mujoco[MUJOCO_TO_ISAACLAB]
        gravity_body = _quat_inv_rotate(
            np.asarray(base_quat_wxyz, dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )
        self._hist_q.append(body_q_il.astype(np.float64))
        self._hist_dq.append(body_dq_il.astype(np.float64))
        self._hist_ang.append(np.asarray(base_ang_vel_body, dtype=np.float64).copy())
        self._hist_grav.append(gravity_body)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_obs(self, sonic_token_64: np.ndarray) -> np.ndarray:
        buf = self._obs_buf
        offset = 0
        buf[0, offset:offset + 64] = sonic_token_64
        offset += 64
        for f in range(HISTORY_FRAMES):
            buf[0, offset:offset + 3] = self._hist_ang[f]
            offset += 3
        for f in range(HISTORY_FRAMES):
            buf[0, offset:offset + 29] = self._hist_q[f]
            offset += 29
        for f in range(HISTORY_FRAMES):
            buf[0, offset:offset + 29] = self._hist_dq[f]
            offset += 29
        for f in range(HISTORY_FRAMES):
            buf[0, offset:offset + 29] = self._hist_act[f]
            offset += 29
        for f in range(HISTORY_FRAMES):
            buf[0, offset:offset + 3] = self._hist_grav[f]
            offset += 3
        assert offset == OBS_DIM, f"obs offset {offset} != {OBS_DIM}"
        return buf

    def step(self, sonic_token: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode one control tick.

        Parameters
        ----------
        sonic_token : np.ndarray
            66-D SONIC token (or just the 64-D latent). Only the first 64 dims
            are fed to the decoder; the two trailing hand bits are ignored.

        Returns
        -------
        action_isaaclab : np.ndarray, shape (29,), float32
            Raw decoder output, IsaacLab joint order. Pushed into the
            rolling ``last_action`` history before returning.
        q_target_mujoco : np.ndarray, shape (29,), float32
            Absolute joint-position targets in MuJoCo joint order, suitable
            for direct consumption by the env's PD controller.
        """
        latent = np.asarray(sonic_token, dtype=np.float32)[:TOKEN_DIM]
        obs = self._build_obs(latent)
        out = self.session.run(["action"], {"obs_dict": obs})[0][0]  # (29,)
        action_il = out.astype(np.float64)
        self._hist_act.append(action_il.copy())
        scaled_mj = action_il[ISAACLAB_TO_MUJOCO] * G1_ACTION_SCALE_MUJOCO
        q_target_mj = DEFAULT_ANGLES_MUJOCO + scaled_mj
        return action_il.astype(np.float32), q_target_mj.astype(np.float32)

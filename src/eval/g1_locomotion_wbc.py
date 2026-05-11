"""Decoupled lower-body locomotion controller (GR00T WBC) loader.

Wraps the pretrained ``GR00T-WholeBodyControl-{Balance,Walk}.onnx`` networks
for use as a balance/locomotion head inside ``scripts/eval/sim_validate.py``.

The networks consume a 516-D observation (86-D ``single_obs`` × 6 history
frames) and produce a 15-D action that maps to absolute joint targets via
``target = action * 0.25 + WBC_DEFAULT_ANGLES_15`` for legs (12) + waist (3).
Arms are *not* controlled here — the upstream policy (Kappa or a placeholder)
drives them directly through the env's PD loop.

This mirrors ``run_mujoco_gear_wbc.py`` from the GR00T-WholeBodyControl repo
with two adaptations for offline replay:

  * No keyboard listener — ``loco_cmd``, ``height_cmd``, ``rpy_cmd`` are
    passed in per ``step()`` call by the caller.
  * Balance vs Walk switch happens automatically based on ``||loco_cmd[:3]||``
    (threshold matches the deployment script).
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort


# --------------------------- WBC observation constants --------------------------
# Lifted from decoupled_wbc/sim2mujoco/resources/robots/g1/g1_gear_wbc.yaml.
# These are training-time hyperparameters baked into the ONNX policies; do not
# change without retraining.

WBC_DEFAULT_ANGLES_15 = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     0.0, 0.0, 0.0,
], dtype=np.float32)

ACTION_SCALE = 0.25
CMD_SCALE = np.array([2.0, 2.0, 0.5], dtype=np.float32)
ANG_VEL_SCALE = 0.5
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05

OBS_HISTORY_LEN = 6
SINGLE_OBS_DIM = 86
NUM_OBS = SINGLE_OBS_DIM * OBS_HISTORY_LEN  # 516
NUM_ACTIONS = 15
NUM_JOINTS = 29

# Above this ||loco_cmd[:3]|| (body-frame m/s + rad/s), switch from Balance to
# Walk. Matches run_mujoco_gear_wbc.py's hardcoded gate.
WALK_THRESHOLD = 0.05


def _gravity_orientation_body(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project world gravity (0,0,-1) into the body frame defined by a wxyz quat.

    Equivalent to ``quat_rotate_inverse(q, [0,0,-1])`` in the deployment script.
    """
    w, x, y, z = quat_wxyz
    # Inverse rotation: g_body = R(q)^T @ g_world
    return np.array([
        -2.0 * (x * z - w * y),
        -2.0 * (y * z + w * x),
        -(w * w - x * x - y * y + z * z),
    ], dtype=np.float32)


class G1LocomotionWBC:
    """GR00T decoupled WBC (lower-body locomotion only)."""

    def __init__(
        self,
        balance_onnx_path: str | Path,
        walk_onnx_path: str | Path,
        device_id: Optional[int] = None,
    ):
        providers = []
        if device_id is not None:
            providers.append(("CUDAExecutionProvider", {"device_id": int(device_id)}))
        providers.append("CPUExecutionProvider")
        self._balance = ort.InferenceSession(str(balance_onnx_path), providers=providers)
        self._walk = ort.InferenceSession(str(walk_onnx_path), providers=providers)
        self._balance_in = self._balance.get_inputs()[0].name
        self._walk_in = self._walk.get_inputs()[0].name

        self._history: collections.deque[np.ndarray] = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * OBS_HISTORY_LEN,
            maxlen=OBS_HISTORY_LEN,
        )
        self._last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self._last_target = WBC_DEFAULT_ANGLES_15.copy()
        self._last_mode = "balance"

    def reset(self) -> None:
        """Clear obs history + last_action. Call before starting a new rollout."""
        self._history = collections.deque(
            [np.zeros(SINGLE_OBS_DIM, dtype=np.float32)] * OBS_HISTORY_LEN,
            maxlen=OBS_HISTORY_LEN,
        )
        self._last_action[:] = 0.0
        self._last_target = WBC_DEFAULT_ANGLES_15.copy()
        self._last_mode = "balance"

    @property
    def last_target_15(self) -> np.ndarray:
        return self._last_target

    @property
    def last_mode(self) -> str:
        return self._last_mode

    def step(
        self,
        qpos_29: np.ndarray,
        qvel_29: np.ndarray,
        base_quat_wxyz: np.ndarray,
        base_omega_body: np.ndarray,
        loco_cmd_3: np.ndarray,
        height_cmd: float,
        rpy_cmd_3: np.ndarray,
    ) -> np.ndarray:
        """Run one WBC inference and return the 15-D target joint angles.

        Joint indexing matches ``JOINT_NAMES_29`` from ``g1_mujoco_env``:
        legs (0..11), waist (12..14). Arm joints (15..28) are ignored here.
        """
        single_obs = self._build_single_obs(
            qpos_29, qvel_29, base_quat_wxyz, base_omega_body,
            loco_cmd_3, height_cmd, rpy_cmd_3,
        )
        self._history.append(single_obs)
        obs_flat = np.concatenate(list(self._history)).astype(np.float32)[None, :]

        loco_norm = float(np.linalg.norm(np.asarray(loco_cmd_3, dtype=np.float32)))
        if loco_norm <= WALK_THRESHOLD:
            sess, in_name = self._balance, self._balance_in
            self._last_mode = "balance"
        else:
            sess, in_name = self._walk, self._walk_in
            self._last_mode = "walk"
        action = sess.run(None, {in_name: obs_flat})[0].reshape(-1).astype(np.float32)
        if action.shape != (NUM_ACTIONS,):
            raise RuntimeError(f"WBC output shape {action.shape} != ({NUM_ACTIONS},)")
        self._last_action = action
        self._last_target = action * ACTION_SCALE + WBC_DEFAULT_ANGLES_15
        return self._last_target.copy()

    def _build_single_obs(
        self,
        qpos_29: np.ndarray,
        qvel_29: np.ndarray,
        base_quat_wxyz: np.ndarray,
        base_omega_body: np.ndarray,
        loco_cmd_3: np.ndarray,
        height_cmd: float,
        rpy_cmd_3: np.ndarray,
    ) -> np.ndarray:
        """Build the 86-D per-step observation; see run_mujoco_gear_wbc.py:225-237.

        Layout (offsets in single_obs):
            [0:3]   loco_cmd * cmd_scale
            [3]     height_cmd
            [4:7]   rpy_cmd
            [7:10]  base_omega_body * ang_vel_scale
            [10:13] gravity_orientation in body frame
            [13:42] (qpos_29 - default_padded_29) * dof_pos_scale
            [42:71] qvel_29 * dof_vel_scale
            [71:86] last_action (15D)
        """
        qj = np.asarray(qpos_29, dtype=np.float32)
        dqj = np.asarray(qvel_29, dtype=np.float32)
        if qj.shape != (NUM_JOINTS,) or dqj.shape != (NUM_JOINTS,):
            raise ValueError(f"qpos/qvel must be (29,); got {qj.shape}/{dqj.shape}")

        # default_angles is 15-D in the WBC config; pad zeros for arms (16..28)
        # so the qj_scaled subtraction is a no-op on arm channels (which are not
        # used by the WBC anyway but still part of the obs vector).
        default_padded = np.zeros(NUM_JOINTS, dtype=np.float32)
        default_padded[: WBC_DEFAULT_ANGLES_15.shape[0]] = WBC_DEFAULT_ANGLES_15

        qj_scaled = (qj - default_padded) * DOF_POS_SCALE
        dqj_scaled = dqj * DOF_VEL_SCALE
        omega_scaled = np.asarray(base_omega_body, dtype=np.float32) * ANG_VEL_SCALE
        grav = _gravity_orientation_body(np.asarray(base_quat_wxyz, dtype=np.float32))

        loco_scaled = np.asarray(loco_cmd_3, dtype=np.float32) * CMD_SCALE
        rpy = np.asarray(rpy_cmd_3, dtype=np.float32)

        single_obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        single_obs[0:3] = loco_scaled
        single_obs[3] = float(height_cmd)
        single_obs[4:7] = rpy
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = grav
        single_obs[13 : 13 + NUM_JOINTS] = qj_scaled
        single_obs[13 + NUM_JOINTS : 13 + 2 * NUM_JOINTS] = dqj_scaled
        single_obs[13 + 2 * NUM_JOINTS : 13 + 2 * NUM_JOINTS + NUM_ACTIONS] = self._last_action
        return single_obs

"""Closed-loop sim → 64-D proprio assembler for the Kappa policy.

Mirrors the training cache layout produced by
``scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py``:

    proprio[ 0:29] = jp        (rad, MuJoCo joint order)
    proprio[29:31] = hand_bits (binary; 0/1)
    proprio[31:60] = jv        (rad/s, finite-diff of jp at the training rate)
    proprio[60:64] = quat_xyzw (pelvis orientation, "as if in HE-world frame")

Two pieces of work the assembler does on top of the env:

  1. Finite-diff jv. The env's ``joint_vel()`` is MuJoCo's continuous-time
     velocity (``data.qvel``), which has a different noise/bandwidth profile
     than the training cache's ``(jp[t]-jp[t-1])*fps`` recipe. Feeding
     ``data.qvel`` into the policy leaves the jv channel slightly OOD even
     when everything else is matched. We compute ``(jp_now - jp_prev) / dt``
     using the env's baseline latched after each outer control tick, so the
     units stay rad/s and the value reflects "what the encoder would have
     reported between sample t-1 and t."

  2. Quat re-basing. The dataset stores pelvis quat in HE-world frame, which
     is whatever direction the operator was facing at episode start. The sim
     spawns at identity quat regardless of episode. Without re-basing the
     policy sees a quat distribution where every episode starts at identity,
     which is not how training data looked. We compute a constant alignment
     quat ``q_align = q_dataset_init ⊗ inv(q_sim_spawn)`` at construction and
     left-multiply every reported sim quat by it. The sim still runs upright
     in its own world (the WBC sees the un-rotated quat through
     ``env.root_quat_wxyz`` — assembler is read-only for the simulator);
     ``q_align`` is purely a relabeling for the policy-facing vector.

  3. Hand bits. Sourced from ``env._last_hand_bits``, which the env's
     ``step`` / ``step_with_wbc`` already updates from the most recently
     commanded action. The caller is expected to have seeded it via
     ``env.reset(init_hand_bits=...)`` so the first inference doesn't see a
     stale zero.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.eval.g1_mujoco_env import G1MujocoEnv


def _qmul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw-ordered quaternions, double precision."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _qinv_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


class SimProprioAssembler:
    """Build the policy-facing 64-D proprio vector from live sim state.

    Lifecycle:

        env.reset(qpos_init=..., init_hand_bits=hand_gt0)
        assembler = SimProprioAssembler(env, q_dataset_init_xyzw)
        # WBC warmup loop, every tick:
        #   env.step_with_wbc(...)
        #   env.latch_proprio_baseline()
        # Main loop, every tick:
        #   if need_inference:
        #       proprio_raw = assembler.get()
        #       policy.predict_chunk(..., proprio_64_raw=proprio_raw)
        #   env.step_with_wbc(...)
        #   env.latch_proprio_baseline()

    The "latch after step" pattern (rather than "latch before step") means
    ``get()`` always uses the jp at the end of the *previous* tick as t-1,
    matching the training pipeline where jv at frame t is computed from jp
    at frames t and t-1.
    """

    # The training cache jv was built at 50 Hz (calc_sonic_tokens.py target_fps
    # for HE). We report rad/s directly so any outer ctrl rate is consistent
    # in magnitude; the only sensitive distribution feature is bandwidth, and
    # at typical 30–50 Hz outer rates the difference is small.
    TRAIN_FPS: float = 50.0

    def __init__(self, env: "G1MujocoEnv", q_dataset_init_xyzw: np.ndarray):
        self.env = env
        q_dataset0 = np.asarray(q_dataset_init_xyzw, dtype=np.float64)
        if q_dataset0.shape != (4,):
            raise ValueError(
                f"q_dataset_init_xyzw must be (4,), got {q_dataset0.shape}"
            )
        # Normalize defensively — dataset quat occasionally drifts off-unit by
        # ~1e-6 after resampling and compounding through ``q_align`` magnifies
        # the error.
        q_dataset0 /= max(float(np.linalg.norm(q_dataset0)), 1e-12)
        q_sim0 = env.root_quat_xyzw().astype(np.float64)
        q_sim0 /= max(float(np.linalg.norm(q_sim0)), 1e-12)
        self.q_align = _qmul_xyzw(q_dataset0, _qinv_xyzw(q_sim0))
        # Renormalize the composed alignment so downstream products stay unit.
        self.q_align /= max(float(np.linalg.norm(self.q_align)), 1e-12)

    @classmethod
    def from_wxyz(
        cls, env: "G1MujocoEnv", q_dataset_init_wxyz: np.ndarray,
    ) -> "SimProprioAssembler":
        """Convenience: build from the dataset's wxyz quat (the format
        ``HEEpisodeSource.pelvis_pose_at`` returns at indices 3..7)."""
        return cls(env, _wxyz_to_xyzw(np.asarray(q_dataset_init_wxyz)))

    def get(self) -> np.ndarray:
        """Assemble the 64-D proprio vector for the current sim state.

        Requires ``env.latch_proprio_baseline()`` to have been called at
        least once after the last ``env.reset`` (the reset itself latches a
        baseline, so the first call after reset already has a valid t-1).
        """
        env = self.env
        if env._jp_baseline is None or env._jp_baseline_t is None:
            raise RuntimeError(
                "SimProprioAssembler.get() called before any proprio baseline "
                "was latched; call env.reset() (which latches automatically) "
                "or env.latch_proprio_baseline() first."
            )
        jp_now = env.joint_pos().astype(np.float64)
        t_now = float(env.data.time)
        dt = t_now - float(env._jp_baseline_t)
        if dt < 1e-9:
            # Two get() calls without an intervening step / latch. Best we can
            # do is report zero jv (matches t=0 frame in the training cache,
            # where jv[0] is also zero by construction).
            jv = np.zeros(29, dtype=np.float32)
        else:
            jv = ((jp_now - env._jp_baseline.astype(np.float64)) / dt).astype(np.float32)

        q_sim = env.root_quat_xyzw().astype(np.float64)
        q_report = _qmul_xyzw(self.q_align, q_sim)
        q_report /= max(float(np.linalg.norm(q_report)), 1e-12)

        return env.proprio_64_with_jv(
            jv_29=jv,
            quat_xyzw=q_report.astype(np.float32),
        )

"""Shared utilities for SONIC token preprocessing scripts.

Common pieces used by both ``scripts/preprocess/humanoid-everyday/`` and
``scripts/preprocess/g1-wbt/`` (and any future source):

* :func:`resample_linear` / :func:`resample_quat_wxyz` — 30→50 Hz upsampling
  with bit-identical behavior across sources so the SONIC token distribution
  doesn't shift between datasets.
* :func:`atomic_save_npy` — write ``.npy`` via ``<final>.tmp.<pid>`` then
  rename, so interrupted runs never leave a half-written file that the
  resume logic would skip.
* :func:`build_loco_cmd` / :func:`pelvis_roll_pitch` — derive the 7-D WBC
  command channel (navigate_cmd[3] + height[1] + rpy[3]) from pelvis
  pos/quat at the SONIC timeline. Mirrors ``episode_source._build_loco_cmd``
  so offline cache and sim_validate's GT-derivation produce identical values.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def resample_linear(arr: np.ndarray, n_src: int, n_tgt: int) -> np.ndarray:
    """Linear interpolation along axis 0 from length ``n_src`` to ``n_tgt``.

    Preserves dtype. Identity when sizes match; broadcasts when ``n_src == 1``.
    """
    if n_tgt == n_src:
        return arr
    if n_src == 1:
        return np.repeat(arr, n_tgt, axis=0)
    pos = np.linspace(0.0, n_src - 1, n_tgt)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n_src - 1)
    frac = (pos - lo).reshape((n_tgt,) + (1,) * (arr.ndim - 1))
    out = (1.0 - frac) * arr[lo] + frac * arr[hi]
    return out.astype(arr.dtype, copy=False)


def resample_quat_wxyz(q: np.ndarray, n_src: int, n_tgt: int) -> np.ndarray:
    """Lerp + L2-renormalize WXYZ unit quaternions with sign canonicalization.

    SLERP would be exact, but at 30→50 Hz the per-step rotation is small
    (<<1 rad) so canonicalized lerp + renorm is within rounding. Sign-flip
    pass guarantees we don't take the long way around the hypersphere.
    """
    if n_tgt == n_src:
        return q
    q = q.astype(np.float64, copy=True)
    if n_src > 1:
        dots = np.sum(q[1:] * q[:-1], axis=-1)
        flips = np.cumprod(np.where(dots < 0.0, -1.0, 1.0))
        q[1:] *= flips[:, None]
    out = resample_linear(q, n_src, n_tgt)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    safe = norms >= 1e-10
    normalized = np.divide(out, norms, out=np.zeros_like(out), where=safe)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    out = np.where(safe, normalized, identity)
    return out.astype(np.float32, copy=False)


def atomic_save_npy(arr: Any, final_path: str) -> None:
    """Atomically write ``arr`` to ``final_path`` (a path ending in ``.npy``).

    Writes via ``<final>.tmp.<pid>`` and ``os.replace`` — racing ranks /
    interrupted runs never observe a half-written file. ``np.save`` is
    careful to honor a path with the ``.npy`` suffix already present (no
    double-suffix), so the temp path can carry it directly.
    """
    if not final_path.endswith(".npy"):
        raise ValueError(f"atomic_save_npy: final_path must end with .npy, got {final_path!r}")
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    np.save(tmp_path, arr)
    # np.save appends .npy when the suffix is missing; tmp_path already
    # ends in ".tmp.<pid>" without .npy, so the actual file written is
    # tmp_path + ".npy". Locate whichever exists.
    written = tmp_path if os.path.exists(tmp_path) else f"{tmp_path}.npy"
    os.replace(written, final_path)


# ---------------------------------------------------------------------------
# WBC command channel derivation (loco_cmd / height / rpy)
# ---------------------------------------------------------------------------

# WBC ONNX training-time range. Values outside push the policy OOD, so we clip
# both the training target (here) and the runtime command (sim_validate).
WBC_VX_CLIP = 1.0
WBC_VY_CLIP = 1.0
WBC_WZ_CLIP = 1.5


def pelvis_roll_pitch(quat_wxyz: np.ndarray) -> np.ndarray:
    """Extract (roll, pitch) from a (T, 4) WXYZ pelvis quaternion array.

    Yaw is intentionally dropped here — the WBC takes torso roll/pitch as
    direct commands while yaw is handled internally by the locomotion policy
    (and HE only stores pelvis, not torso, so a meaningful yaw command would
    require an extra fk pass through waist joints).
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    pitch_arg = np.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0)
    pitch = np.arcsin(pitch_arg)
    return np.stack([roll, pitch], axis=-1).astype(np.float32)


def build_loco_cmd(
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
    fps: float,
    window: int = 2,
) -> np.ndarray:
    """Body-frame (vx, vy, ωz) from pelvis pos+quat via centered finite diff.

    Pipeline mirrors ``src/eval/episode_source.py:_build_loco_cmd``:
      1. unwrap yaw from quat to remove ±π jumps;
      2. centered finite diff window=2 frames (5 frames total) for linear vel
         and yaw rate;
      3. project world-frame planar velocity into body frame at frame ``i``;
      4. clip to WBC training range (±1.0 m/s linear, ±1.5 rad/s yaw).

    Returns (T, 3) float32.
    """
    pos = np.asarray(pos_xyz, dtype=np.float64)
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    T = pos.shape[0]
    dt = 1.0 / float(fps)

    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    yaw_unwrapped = np.unwrap(yaw)

    W = int(window)
    v_world = np.zeros((T, 2), dtype=np.float64)
    wz = np.zeros(T, dtype=np.float64)
    for i in range(T):
        i0 = max(0, i - W)
        i1 = min(T - 1, i + W)
        span = (i1 - i0) * dt
        if span <= 0:
            continue
        v_world[i] = (pos[i1, :2] - pos[i0, :2]) / span
        wz[i] = (yaw_unwrapped[i1] - yaw_unwrapped[i0]) / span

    c, s = np.cos(yaw), np.sin(yaw)
    vx_body = v_world[:, 0] * c + v_world[:, 1] * s
    vy_body = -v_world[:, 0] * s + v_world[:, 1] * c

    cmd = np.stack([vx_body, vy_body, wz], axis=1)
    cmd[:, 0] = np.clip(cmd[:, 0], -WBC_VX_CLIP, WBC_VX_CLIP)
    cmd[:, 1] = np.clip(cmd[:, 1], -WBC_VY_CLIP, WBC_VY_CLIP)
    cmd[:, 2] = np.clip(cmd[:, 2], -WBC_WZ_CLIP, WBC_WZ_CLIP)
    return cmd.astype(np.float32)


def build_g1_action_23(
    jp: np.ndarray,            # (T, 29) float
    hand_bits: np.ndarray,     # (T, 2)  float
    pos_xyz: np.ndarray,       # (T, 3)  float
    quat_wxyz: np.ndarray,     # (T, 4)  float
    fps: float,
) -> np.ndarray:
    """Assemble the 23-D G1 action target.

    Layout:
        [0:14]   arms (jp[15:29])
        [14:16]  hand_bits
        [16:19]  navigate_cmd(vx_b, vy_b, ωz)
        [19:20]  base_height (absolute pelvis z, m)
        [20:23]  rpy_cmd(roll, pitch, yaw=0)

    Legs+waist (jp[0:15]) are intentionally dropped — WBC drives those joints
    at deployment; keeping them in the action target only adds gradient noise
    on shared expert parameters (auto-memory: ``project_kappa_leg_pred_uncorrelated``).
    The full body pose still reaches the model via the 64-D proprio.
    """
    arms = np.asarray(jp[:, 15:29], dtype=np.float32)
    bits = np.asarray(hand_bits, dtype=np.float32)
    nav = build_loco_cmd(pos_xyz, quat_wxyz, fps=fps)
    height = np.asarray(pos_xyz[:, 2:3], dtype=np.float32)
    rp = pelvis_roll_pitch(quat_wxyz)
    yaw_cmd = np.zeros((rp.shape[0], 1), dtype=np.float32)
    rpy = np.concatenate([rp, yaw_cmd], axis=-1)
    return np.concatenate([arms, bits, nav, height, rpy], axis=-1).astype(np.float32)

"""Physics-simulation validation for the Kappa model (SONIC whole-body).

Pipeline
--------
Each sampled HE episode spins up MuJoCo G1 (29-DoF, sim_dt=0.005, free-base).
The ``sonic`` head drives all 29 joints via PD position tracking from the
SonicDecoder ONNX output — SONIC is the whole-body action representation,
so no WBC sits underneath it. The legacy ``gt_wbc`` / ``wbc_idle`` heads
still use the GR00T-WholeBodyControl Balance + Walk ONNX as a comparison
baseline / sanity check. A 0.5 s warmup phase at the start of each rollout
lets the WBC's obs history fill (for ``gt_wbc`` / ``wbc_idle``); metrics
over the warmup window are reported separately (``warmup_steps`` in the
json).

Heads
-----

  * **wbc_idle** — 30 s standstill sanity. No episode content used; verifies
    the WBC keeps the robot upright in the Kappa env. Use this to debug WBC
    integration regressions.
  * **gt_wbc** — WBC + GT arms / hand_bits (from the dataset's proprio cache)
    + GT-derived ``loco_cmd`` from pelvis odometry. The "perfect upper body
    with WBC legs" baseline: drift / pitch / fell here are pure controller
    error, no model influence.
  * **sonic** — Kappa's SONIC token chunk decoded through the SonicDecoder
    ONNX into 29-D joint targets, then PD-tracked across the whole body via
    ``env.step(action_31)``. No WBC, no GT-derived locomotion command — the
    deployment-mirror eval.

Shared inputs across all model heads:

  1. Egocentric video frames pulled at every policy query (training-time
     image transform: resize-short-224 + center-crop 224×320 in PIL space).
  2. 64-D proprio. By default this is pulled from the 50 Hz HE training
     cache (``--proprio-source cache``) — anchors image/text/proprio to the
     training distribution and serves as the OOD-free debug path. Pass
     ``--proprio-source sim`` on the model head to assemble proprio live
     from sim state via :class:`src.eval.sim_proprio.SimProprioAssembler`
     (deployment-faithful closed-loop channel).
  3. Policy queries every ``--s-min`` control steps with RTC prefix
     substitution to smooth chunk seams.

Per-episode outputs land under ``<out-dir>/episode_<idx>/`` with each file
prefixed by head name. The source HE video is copied to
``<out-dir>/episode_<idx>/source.mp4``. After all episodes a single
``summary.json`` aggregates per-head metrics.

Usage
-----
::

    # Walking episode (head-to-head vs WBC baseline)
    uv run python scripts/eval/sim_validate.py \
        --heads gt_wbc sonic \
        --episode-indices 7831 --episode-seconds 8 \
        --out-dir .runs/eval/$(date +%Y%m%d_%H%M%S)

    # WBC integration sanity check
    uv run python scripts/eval/sim_validate.py \
        --heads wbc_idle --episode-indices 7831 --episode-seconds 30 \
        --out-dir .runs/eval/wbc_idle_$(date +%s)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Pick a headless GL backend before any mujoco import side-effects from
# downstream modules. Mirrors the env wrapper's logic so the script also
# works when invoked directly without the env being imported first.
if not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mediapy as media
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.episode_source import FrameStream, HEEpisodeSource
from src.eval.g1_mujoco_env import DEFAULT_QPOS, G1MujocoEnv
from src.eval.g1_locomotion_wbc import G1LocomotionWBC, WBC_DEFAULT_ANGLES_15
from src.eval.kappa_policy import KappaPolicy, KappaPolicyConfig
from src.eval.sim_proprio import SimProprioAssembler
from src.eval.sonic_decoder import SonicDecoder


# ------------------------------- helpers -----------------------------------

# Match the training-time model transform (resize short to 224, then center
# crop to (H=224, W=320)). Done in PIL space; the Qwen3VL processor then
# applies its own normalization on the output.
_TRAIN_IMG_TRANSFORM = v2.Compose([
    v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop((224, 320)),
])


def _preprocess_frame(img: Image.Image) -> Image.Image:
    return _TRAIN_IMG_TRANSFORM(img)


def _quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _saturate_image_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


# Minimal WXYZ-quaternion helpers used to re-base the dataset's pelvis pose
# onto the env's spawn frame for the GT-replay head. Kept inline (rather than
# pulled from scipy) so this script stays a single self-contained file.

def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector ``v`` by unit quaternion ``q`` (wxyz)."""
    qw, qx, qy, qz = q
    vw_q = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    out = _quat_mul(_quat_mul(q, vw_q), _quat_inv(q))
    return out[1:]


def _proprio_diff_entry(
    step_idx: int,
    t_he: float,
    p_cache: np.ndarray,
    p_sim: np.ndarray,
) -> dict:
    """Per-tick channel-wise delta between training cache and live-sim
    proprio. Quat distance is reported as ``1 - |<q_cache, q_sim>|`` so
    sign flips don't show up as 2.0 spikes.
    """
    jp_diff = np.abs(p_cache[:29] - p_sim[:29])
    hand_diff = np.abs(p_cache[29:31] - p_sim[29:31])
    jv_diff = np.abs(p_cache[31:60] - p_sim[31:60])
    q_dot = abs(float(np.dot(p_cache[60:64], p_sim[60:64])))
    return {
        "step_idx": int(step_idx),
        "t_he": float(t_he),
        "jp_max": float(jp_diff.max()),
        "jp_l2": float(np.linalg.norm(p_cache[:29] - p_sim[:29])),
        "hand_max": float(hand_diff.max()),
        "jv_max": float(jv_diff.max()),
        "jv_l2": float(np.linalg.norm(p_cache[31:60] - p_sim[31:60])),
        "quat_err": float(1.0 - q_dot),
    }


def _write_proprio_diff_log(
    out_dir: Path,
    head: str,
    records: list[dict],
) -> None:
    """Dump diff records as ``<head>_proprio_diff.npz`` and print a one-line
    summary so head-to-head runs can be eyeballed without unpacking the file.
    """
    if not records:
        return
    out_path = out_dir / f"{head}_proprio_diff.npz"
    arrs = {k: np.asarray([r[k] for r in records], dtype=np.float32)
            for k in records[0].keys()}
    np.savez_compressed(out_path, **arrs)
    jp99 = float(np.quantile(arrs["jp_max"], 0.99))
    jv99 = float(np.quantile(arrs["jv_max"], 0.99))
    q99 = float(np.quantile(arrs["quat_err"], 0.99))
    print(
        f"[diff/{head}] n={len(records)}  jp_max_p99={jp99:.4f}rad  "
        f"jv_max_p99={jv99:.3f}rad/s  quat_err_p99={q99:.2e}"
    )


# ---------------------------- Rollouts -------------------------------------
#
# All rollout heads run free-base MuJoCo. ``wbc_idle`` and ``gt_wbc`` attach
# the GR00T-WholeBodyControl Balance + Walk ONNX to drive legs+waist. The
# ``sonic`` head runs pure PD position tracking across all 29 joints from
# the SonicDecoder ONNX output — no WBC underneath.


def _run_wbc_idle_rollout(
    *,
    env: G1MujocoEnv,
    wbc: G1LocomotionWBC,
    out_dir: Path,
    seconds: float = 30.0,
    warmup_seconds: float = 0.5,
) -> dict:
    """Sanity rollout for the WBC head.

    Resets the env to the WBC's training-time standing pose
    (``WBC_DEFAULT_ANGLES_15`` for legs+waist, ``DEFAULT_QPOS[15:29]`` for arms),
    attaches the WBC, then runs a free-base rollout with zero loco_cmd for
    ``seconds`` of sim time. The first ``warmup_seconds`` are recorded but
    flagged in the metrics so they don't pollute "is balance stable" judgement.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wbc.reset()

    # WBC was trained from a slightly-bent standing pose; spawn there so its
    # first observation lands inside the training distribution. Arms stay at
    # Kappa's default since the WBC doesn't see them anyway.
    standing_qpos = np.concatenate([WBC_DEFAULT_ANGLES_15.astype(np.float64),
                                    DEFAULT_QPOS[15:29].astype(np.float64)])
    env.reset(qpos_init=standing_qpos)
    env.enable_wbc_mode(wbc)

    ctrl_dt = env.ctrl_dt
    n_total = int(round(seconds / ctrl_dt))
    n_warmup = int(round(warmup_seconds / ctrl_dt))

    arm_target = DEFAULT_QPOS[15:29].astype(np.float32)
    hand_bits = np.zeros(2, dtype=np.float32)
    loco_cmd = np.zeros(3, dtype=np.float64)
    rpy_cmd = np.zeros(3, dtype=np.float64)
    height_cmd = 0.74

    sim_qpos = np.zeros((n_total, 29), dtype=np.float32)
    sim_qvel = np.zeros((n_total, 29), dtype=np.float32)
    root_zs = np.zeros(n_total, dtype=np.float32)
    rpys = np.zeros((n_total, 3), dtype=np.float32)
    contact_totals = np.zeros(n_total, dtype=np.float32)
    wbc_targets = np.zeros((n_total, 15), dtype=np.float32)
    inference_ms: list[float] = []
    fell_at_step: int | None = None

    track_path = out_dir / "wbc_idle_track.mp4"
    track_frames: list[np.ndarray] = []

    for step_idx in range(n_total):
        t0 = time.perf_counter()
        stats = env.step_with_wbc(
            arm_target_14=arm_target,
            hand_bits_2=hand_bits,
            loco_cmd_3=loco_cmd,
            height_cmd=height_cmd,
            rpy_cmd_3=rpy_cmd,
        )
        inference_ms.append((time.perf_counter() - t0) * 1000.0)
        sim_qpos[step_idx] = stats.qpos
        sim_qvel[step_idx] = stats.qvel
        root_zs[step_idx] = stats.root_z
        rpys[step_idx] = stats.root_rpy
        contact_totals[step_idx] = stats.contact_force_total
        wbc_targets[step_idx] = wbc.last_target_15
        if stats.fell and fell_at_step is None and step_idx >= n_warmup:
            fell_at_step = step_idx
        track_frames.append(_saturate_image_uint8(env.render("track")))

    media.write_video(track_path, track_frames, fps=int(round(1 / ctrl_dt)))

    # Post-warmup window is the meaningful balance test.
    post = slice(n_warmup, n_total)
    root_z_post = root_zs[post]
    rpy_post = rpys[post]
    metrics = {
        "head": "wbc_idle",
        "video_path_track": str(track_path.relative_to(out_dir.parent)),
        "control_steps": int(n_total),
        "warmup_steps": int(n_warmup),
        "duration_s": float(n_total * ctrl_dt),
        "sim_dt": float(env.sim_dt),
        "fixed_base": bool(env.fixed_base),
        "fell": bool(fell_at_step is not None),
        "fell_at_step": int(fell_at_step) if fell_at_step is not None else None,
        "root_z_post_warmup_mean": float(root_z_post.mean()),
        "root_z_post_warmup_min": float(root_z_post.min()),
        "root_z_post_warmup_max": float(root_z_post.max()),
        "root_z_post_warmup_drift": float(root_z_post.max() - root_z_post.min()),
        "roll_post_warmup_p99": float(np.quantile(np.abs(rpy_post[:, 0]), 0.99)),
        "pitch_post_warmup_p99": float(np.quantile(np.abs(rpy_post[:, 1]), 0.99)),
        "wbc_step_ms_mean": float(np.mean(inference_ms)),
        "wbc_step_ms_p99": _quantile(np.asarray(inference_ms), 0.99),
        "max_contact_force_total": float(contact_totals.max()),
    }

    (out_dir / "wbc_idle_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "wbc_idle_trajectory.npz",
        sim_qpos=sim_qpos,
        sim_qvel=sim_qvel,
        root_z=root_zs,
        rpy=rpys,
        contact_total=contact_totals,
        wbc_targets=wbc_targets,
        wbc_step_ms=np.asarray(inference_ms, dtype=np.float32),
        warmup_steps=np.asarray(n_warmup),
    )
    return metrics


def _run_gt_wbc_rollout(
    *,
    env: G1MujocoEnv,
    wbc: G1LocomotionWBC,
    episode,
    source: HEEpisodeSource,
    out_dir: Path,
    seconds: float,
    warmup_seconds: float = 0.5,
) -> dict:
    """WBC head with GT-derived loco_cmd + GT arm targets.

    Tests the "WBC + perfect upper body" baseline on a real walking episode:
      * legs+waist driven by the WBC ONNX
      * loco_cmd_3 = ``source.loco_cmd_at(t)`` (body-frame velocity reconstructed
        from pelvis odometry)
      * arms (joints 15..28) driven by ``source.proprio_at(t)[15:29]`` (dataset
        GT joint position)
      * hand_bits taken from ``source.proprio_at(t)[29:31]`` (dataset GT)

    A 0.5 s warmup phase holds the WBC standing pose with arms interpolated
    toward GT frame-0; the actual episode rollout starts after warmup. The
    pelvis x/y displacement vs the dataset's pelvis odometry is reported as
    ``base_xy_drift_*``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wbc.reset()

    ep_idx = episode.episode_index
    # 64-D proprio layout: jp[0:29] | hand_bits[29:31] | jv[31:60] | quat[60:64].
    # GT arms = jp[15:29]; GT hand_bits = proprio[29:31].
    proprio_gt0 = source.proprio_at(ep_idx, 0.0)
    arm_gt0 = proprio_gt0[15:29].astype(np.float32)
    hand_gt0 = proprio_gt0[29:31].astype(np.float32)

    standing_qpos = np.concatenate([
        WBC_DEFAULT_ANGLES_15.astype(np.float64),
        DEFAULT_QPOS[15:29].astype(np.float64),
    ])
    env.reset(qpos_init=standing_qpos)
    env.enable_wbc_mode(wbc)

    ctrl_dt = env.ctrl_dt
    n_total = int(round(seconds / ctrl_dt))
    n_warmup = int(round(warmup_seconds / ctrl_dt))

    # Pelvis spawn frame for translating HE-world odometry into sim space.
    he_init = source.pelvis_pose_at(ep_idx, 0.0)
    gt_he_init_pos = he_init[:3].astype(np.float64)
    gt_he_init_quat_inv = _quat_inv(he_init[3:7].astype(np.float64))
    base_init = env._base_qpos_init
    gt_base_spawn_pos = base_init[:3].astype(np.float64).copy()
    gt_base_spawn_quat = base_init[3:7].astype(np.float64).copy()

    sim_qpos = np.zeros((n_total, 29), dtype=np.float32)
    sim_qvel = np.zeros((n_total, 29), dtype=np.float32)
    cmd_arm = np.zeros((n_total, 14), dtype=np.float32)
    wbc_target = np.zeros((n_total, 15), dtype=np.float32)
    loco_cmds = np.zeros((n_total, 3), dtype=np.float32)
    base_xy_sim = np.zeros((n_total, 2), dtype=np.float32)
    base_xy_gt = np.zeros((n_total, 2), dtype=np.float32)
    root_zs = np.zeros(n_total, dtype=np.float32)
    rpys = np.zeros((n_total, 3), dtype=np.float32)
    contact_totals = np.zeros(n_total, dtype=np.float32)
    inference_ms: list[float] = []
    fell_at_step: int | None = None

    track_path = out_dir / "gt_wbc_track.mp4"
    track_frames: list[np.ndarray] = []

    for step_idx in range(n_total):
        t_warmup = step_idx < n_warmup
        # During warmup: hold loco_cmd=0, interpolate arms from default → GT[0].
        # After warmup: pull from dataset at t = (step_idx - n_warmup) * ctrl_dt.
        if t_warmup:
            blend = (step_idx + 1) / max(n_warmup, 1)
            arm_target = (
                (1.0 - blend) * DEFAULT_QPOS[15:29].astype(np.float32)
                + blend * arm_gt0
            )
            hand_bits = blend * hand_gt0
            loco_cmd = np.zeros(3, dtype=np.float64)
            t_he = 0.0
        else:
            t_he = (step_idx - n_warmup) * ctrl_dt
            proprio_t = source.proprio_at(ep_idx, t_he)
            arm_target = proprio_t[15:29].astype(np.float32)
            hand_bits = proprio_t[29:31].astype(np.float32)
            loco_cmd = source.loco_cmd_at(ep_idx, t_he).astype(np.float64)

        t0 = time.perf_counter()
        stats = env.step_with_wbc(
            arm_target_14=arm_target,
            hand_bits_2=hand_bits,
            loco_cmd_3=loco_cmd,
            height_cmd=0.74,
            rpy_cmd_3=np.zeros(3, dtype=np.float64),
        )
        inference_ms.append((time.perf_counter() - t0) * 1000.0)

        sim_qpos[step_idx] = stats.qpos
        sim_qvel[step_idx] = stats.qvel
        cmd_arm[step_idx] = arm_target
        wbc_target[step_idx] = wbc.last_target_15
        loco_cmds[step_idx] = loco_cmd.astype(np.float32)
        base_xy_sim[step_idx] = env.data.qpos[env._free_qposadr : env._free_qposadr + 2]
        # Map HE-world pelvis pos at the same logical time into the sim spawn
        # frame so the drift comparison is in the same coordinates.
        he_pos = source.pelvis_pose_at(ep_idx, t_he)[:3].astype(np.float64)
        local_delta = _quat_rotate(gt_he_init_quat_inv, he_pos - gt_he_init_pos)
        sim_pos_gt = gt_base_spawn_pos + _quat_rotate(gt_base_spawn_quat, local_delta)
        base_xy_gt[step_idx] = sim_pos_gt[:2]
        root_zs[step_idx] = stats.root_z
        rpys[step_idx] = stats.root_rpy
        contact_totals[step_idx] = stats.contact_force_total
        if stats.fell and fell_at_step is None and not t_warmup:
            fell_at_step = step_idx
        track_frames.append(_saturate_image_uint8(env.render("track")))

    media.write_video(track_path, track_frames, fps=int(round(1 / ctrl_dt)))

    post = slice(n_warmup, n_total)
    drift_xy = np.linalg.norm(base_xy_sim[post] - base_xy_gt[post], axis=1)

    metrics = {
        "head": "gt_wbc",
        "episode_index": episode.episode_index,
        "instruction": episode.instruction,
        "video_path_track": str(track_path.relative_to(out_dir.parent)),
        "control_steps": int(n_total),
        "warmup_steps": int(n_warmup),
        "duration_s": float(n_total * ctrl_dt),
        "sim_dt": float(env.sim_dt),
        "fixed_base": bool(env.fixed_base),
        "fell": bool(fell_at_step is not None),
        "fell_at_step": int(fell_at_step) if fell_at_step is not None else None,
        "root_z_post_warmup_mean": float(root_zs[post].mean()),
        "root_z_post_warmup_min": float(root_zs[post].min()),
        "roll_post_warmup_p99": float(np.quantile(np.abs(rpys[post, 0]), 0.99)),
        "pitch_post_warmup_p99": float(np.quantile(np.abs(rpys[post, 1]), 0.99)),
        "base_xy_drift_p50": float(np.quantile(drift_xy, 0.5)),
        "base_xy_drift_p99": float(np.quantile(drift_xy, 0.99)),
        "base_xy_drift_final": float(drift_xy[-1]),
        "loco_cmd_vx_p99": float(np.quantile(np.abs(loco_cmds[post, 0]), 0.99)),
        "loco_cmd_wz_p99": float(np.quantile(np.abs(loco_cmds[post, 2]), 0.99)),
        "wbc_step_ms_mean": float(np.mean(inference_ms)),
        "max_contact_force_total": float(contact_totals.max()),
    }

    (out_dir / "gt_wbc_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "gt_wbc_trajectory.npz",
        sim_qpos=sim_qpos,
        sim_qvel=sim_qvel,
        cmd_arm=cmd_arm,
        wbc_target=wbc_target,
        loco_cmds=loco_cmds,
        base_xy_sim=base_xy_sim,
        base_xy_gt=base_xy_gt,
        root_z=root_zs,
        rpy=rpys,
        contact_total=contact_totals,
        wbc_step_ms=np.asarray(inference_ms, dtype=np.float32),
        warmup_steps=np.asarray(n_warmup),
    )
    return metrics



def _run_sonic_rollout(
    *,
    env: G1MujocoEnv,
    policy: KappaPolicy,
    sonic_decoder: SonicDecoder,
    episode,
    source: HEEpisodeSource,
    out_dir: Path,
    seconds: float,
    s_min: int,
    delay_init: int,
    warmup_seconds: float = 0.5,
    proprio_source: str = "sim",
    proprio_diff_log: bool = False,
) -> dict:
    """Whole-body SONIC rollout (no WBC).

    Kappa → 66-D sonic token → SonicDecoder ONNX → 29-D joint targets;
    these are the model's complete action output. The 29-D targets plus
    the 2-D hand bits go through ``env.step(action_31)`` for PD position
    tracking on all joints — no WBC, no GT-derived locomotion command.
    This is the deployment-mirror eval: SONIC is the whole-body action
    representation, so we don't layer anything underneath it.

    Known caveat (auto-memory: ``project_sonic_rate_mismatch.md``): the
    deployment SONIC decoder was trained at 50 Hz robot-state history but
    our rollout feeds it at the env outer rate.

    ``proprio_source`` selects where the policy-facing 64-D proprio comes from:

      * ``"cache"`` — pull from the 50 Hz HE training cache via
        ``source.proprio_at``. Anchors image/text/proprio to the training
        distribution; useful as the OOD-free debug path.
      * ``"sim"`` (default) — assemble live from sim state via
        ``SimProprioAssembler``. Deployment-faithful closed-loop channel.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ep_idx = episode.episode_index

    env.reset(qpos_init=DEFAULT_QPOS[:29].astype(np.float64))
    sonic_decoder.reset(env.joint_pos())

    ctrl_dt = env.ctrl_dt
    n_total = int(round(seconds / ctrl_dt))
    n_warmup = int(round(warmup_seconds / ctrl_dt))
    n_inference_period = max(1, int(s_min))
    H = policy.mcfg.action_chunk_size

    # Pelvis spawn-frame transform for ``base_xy_drift`` diagnostics only —
    # GT pelvis pose never enters the control loop.
    he_init = source.pelvis_pose_at(ep_idx, 0.0)
    gt_he_init_pos = he_init[:3].astype(np.float64)
    gt_he_init_quat_inv = _quat_inv(he_init[3:7].astype(np.float64))
    base_init = env._base_qpos_init
    gt_base_spawn_pos = base_init[:3].astype(np.float64).copy()
    gt_base_spawn_quat = base_init[3:7].astype(np.float64).copy()

    assembler: SimProprioAssembler | None = None
    if proprio_source == "sim" or proprio_diff_log:
        assembler = SimProprioAssembler.from_wxyz(env, he_init[3:7])
    diff_records: list[dict] = []

    stream = FrameStream(episode.video_path)
    S_cur: np.ndarray | None = None
    cur_idx = 0
    last_delay = int(delay_init)
    inference_ms: list[float] = []
    decoder_ms: list[float] = []

    # cmd_actions logs the (29 joint targets + 2 hand_bits) sent to the env.
    # gt_actions is the GT full-body jp + hand_bits from the dataset's
    # proprio cache (31-D = jp[29] + hand_bits[2]) so fit error can be
    # computed for both arms and legs.
    cmd_actions = np.zeros((n_total, 31), dtype=np.float32)
    gt_actions = np.zeros((n_total, 31), dtype=np.float32)
    sim_qpos = np.zeros((n_total, 29), dtype=np.float32)
    sim_qvel = np.zeros((n_total, 29), dtype=np.float32)
    base_xy_sim = np.zeros((n_total, 2), dtype=np.float32)
    base_xy_gt = np.zeros((n_total, 2), dtype=np.float32)
    root_zs = np.zeros(n_total, dtype=np.float32)
    rpys = np.zeros((n_total, 3), dtype=np.float32)
    contact_totals = np.zeros(n_total, dtype=np.float32)
    fell_at_step: int | None = None

    track_path = out_dir / "sonic_track.mp4"
    track_frames: list[np.ndarray] = []

    DEFAULT_HAND = np.zeros(2, dtype=np.float32)

    try:
        for step_idx in range(n_total):
            in_warmup = step_idx < n_warmup
            if in_warmup:
                action_31 = np.zeros(31, dtype=np.float32)
                action_31[:29] = DEFAULT_QPOS[:29].astype(np.float32)
                action_31[29:31] = DEFAULT_HAND
                t_he = 0.0
                cmd_actions[step_idx] = action_31
            else:
                t_he = (step_idx - n_warmup) * ctrl_dt
                need_inference = (S_cur is None) or (cur_idx >= n_inference_period)
                if need_inference:
                    obs_frame = _preprocess_frame(stream.frame_at(t_he))
                    proprio_cache = source.proprio_at(ep_idx, t_he)
                    proprio_sim = (
                        assembler.get() if assembler is not None else None
                    )
                    proprio_raw = (
                        proprio_sim if proprio_source == "sim" else proprio_cache
                    )
                    if proprio_diff_log and proprio_sim is not None:
                        diff_records.append(_proprio_diff_entry(
                            step_idx, t_he, proprio_cache, proprio_sim,
                        ))
                    t0 = time.perf_counter()
                    S_chunk = policy.predict_chunk(
                        image=obs_frame,
                        text=episode.instruction,
                        proprio_64_raw=proprio_raw,
                        prev_sonic=S_cur if S_cur is not None else None,
                        delay_sonic=last_delay if S_cur is not None else 0,
                    )
                    inference_ms.append((time.perf_counter() - t0) * 1000.0)
                    if S_cur is not None:
                        delay = min(cur_idx, H - 1)
                        S_chunk[:delay] = S_cur[H - delay:]
                        last_delay = delay if delay > 0 else last_delay
                    S_cur = S_chunk
                    cur_idx = 0

                # Push observed robot state, then decode the sonic token.
                sonic_decoder.push_state(
                    qpos_mujoco=env.joint_pos(),
                    qvel_mujoco=env.joint_vel(),
                    base_ang_vel_body=env.base_angular_velocity_body(),
                    base_quat_wxyz=env.root_quat_wxyz(),
                )
                d0 = time.perf_counter()
                _, q_target_mj = sonic_decoder.step(S_cur[cur_idx])
                decoder_ms.append((time.perf_counter() - d0) * 1000.0)
                hand_bits = np.asarray(S_cur[cur_idx, -2:], dtype=np.float32)
                action_31 = np.zeros(31, dtype=np.float32)
                action_31[:29] = q_target_mj.astype(np.float32)
                action_31[29:31] = hand_bits
                cur_idx += 1
                cmd_actions[step_idx] = action_31

            stats = env.step(action_31)
            env.latch_proprio_baseline()

            sim_qpos[step_idx] = stats.qpos
            sim_qvel[step_idx] = stats.qvel
            base_xy_sim[step_idx] = env.data.qpos[env._free_qposadr : env._free_qposadr + 2]
            he_pos = source.pelvis_pose_at(ep_idx, t_he)[:3].astype(np.float64)
            local_delta = _quat_rotate(gt_he_init_quat_inv, he_pos - gt_he_init_pos)
            sim_pos_gt = gt_base_spawn_pos + _quat_rotate(gt_base_spawn_quat, local_delta)
            base_xy_gt[step_idx] = sim_pos_gt[:2]
            root_zs[step_idx] = stats.root_z
            rpys[step_idx] = stats.root_rpy
            contact_totals[step_idx] = stats.contact_force_total
            try:
                gt_proprio_t = source.proprio_at(ep_idx, t_he)
                gt_actions[step_idx, 0:29] = gt_proprio_t[0:29]
                gt_actions[step_idx, 29:31] = gt_proprio_t[29:31]
            except FileNotFoundError:
                pass
            if stats.fell and fell_at_step is None and not in_warmup:
                fell_at_step = step_idx
            track_frames.append(_saturate_image_uint8(env.render("track")))
    finally:
        stream.close()

    media.write_video(track_path, track_frames, fps=int(round(1 / ctrl_dt)))

    post = slice(n_warmup, n_total)
    body_cmd_post = cmd_actions[post, 0:29]
    body_gt_post = gt_actions[post, 0:29]
    body_sim_post = sim_qpos[post, 0:29]
    arm_fit_err = np.abs(body_cmd_post[:, 15:29] - body_gt_post[:, 15:29])
    leg_fit_err = np.abs(body_cmd_post[:, 0:15] - body_gt_post[:, 0:15])
    arm_track_err = np.abs(body_sim_post[:, 15:29] - body_cmd_post[:, 15:29])
    leg_track_err = np.abs(body_sim_post[:, 0:15] - body_cmd_post[:, 0:15])
    arm_diff = (
        np.abs(np.diff(body_cmd_post[:, 15:29], axis=0))
        if body_cmd_post.shape[0] > 1 else np.zeros((1, 14))
    )
    drift_xy = np.linalg.norm(base_xy_sim[post] - base_xy_gt[post], axis=1)

    metrics = {
        "head": "sonic",
        "episode_index": episode.episode_index,
        "instruction": episode.instruction,
        "video_path_track": str(track_path.relative_to(out_dir.parent)),
        "control_steps": int(n_total),
        "warmup_steps": int(n_warmup),
        "duration_s": float(n_total * ctrl_dt),
        "sim_dt": float(env.sim_dt),
        "fixed_base": bool(env.fixed_base),
        "proprio_source": proprio_source,
        "fell": bool(fell_at_step is not None),
        "fell_at_step": int(fell_at_step) if fell_at_step is not None else None,
        "n_inferences": int(len(inference_ms)),
        "inference_ms_mean": float(np.mean(inference_ms)) if inference_ms else None,
        "inference_ms_p99": _quantile(np.asarray(inference_ms), 0.99) if inference_ms else None,
        "decoder_ms_mean": float(np.mean(decoder_ms)) if decoder_ms else None,
        "decoder_ms_p99": _quantile(np.asarray(decoder_ms), 0.99) if decoder_ms else None,
        "arm_fit_err_per_joint_p50": _quantile(arm_fit_err, 0.5),
        "arm_fit_err_per_joint_p99": _quantile(arm_fit_err, 0.99),
        "leg_fit_err_per_joint_p50": _quantile(leg_fit_err, 0.5),
        "leg_fit_err_per_joint_p99": _quantile(leg_fit_err, 0.99),
        "arm_tracking_err_p50": _quantile(arm_track_err, 0.5),
        "arm_tracking_err_p99": _quantile(arm_track_err, 0.99),
        "leg_tracking_err_p50": _quantile(leg_track_err, 0.5),
        "leg_tracking_err_p99": _quantile(leg_track_err, 0.99),
        "arm_action_diff_p99": _quantile(arm_diff, 0.99),
        "root_z_post_warmup_mean": float(root_zs[post].mean()),
        "root_z_post_warmup_min": float(root_zs[post].min()),
        "roll_post_warmup_p99": float(np.quantile(np.abs(rpys[post, 0]), 0.99)),
        "pitch_post_warmup_p99": float(np.quantile(np.abs(rpys[post, 1]), 0.99)),
        "base_xy_drift_p50": float(np.quantile(drift_xy, 0.5)),
        "base_xy_drift_final": float(drift_xy[-1]),
        "max_contact_force_total": float(contact_totals.max()),
    }
    if proprio_diff_log and diff_records:
        diff_jp = np.asarray([r["jp_max"] for r in diff_records], dtype=np.float32)
        diff_jv = np.asarray([r["jv_max"] for r in diff_records], dtype=np.float32)
        diff_q = np.asarray([r["quat_err"] for r in diff_records], dtype=np.float32)
        metrics["proprio_diff_jp_p99"] = float(np.quantile(diff_jp, 0.99))
        metrics["proprio_diff_jv_p99"] = float(np.quantile(diff_jv, 0.99))
        metrics["proprio_diff_quat_p99"] = float(np.quantile(diff_q, 0.99))
    _write_proprio_diff_log(out_dir, "sonic", diff_records)

    (out_dir / "sonic_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "sonic_trajectory.npz",
        cmd_actions=cmd_actions,
        gt_actions=gt_actions,
        sim_qpos=sim_qpos,
        sim_qvel=sim_qvel,
        base_xy_sim=base_xy_sim,
        base_xy_gt=base_xy_gt,
        root_z=root_zs,
        rpy=rpys,
        contact_total=contact_totals,
        inference_ms=np.asarray(inference_ms, dtype=np.float32),
        decoder_ms=np.asarray(decoder_ms, dtype=np.float32),
        warmup_steps=np.asarray(n_warmup),
    )
    return metrics




# ----------------------------- main loop -----------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--checkpoint", default=str(
            ROOT / ".runs/kappa/exp.kappa.b256.gpus8.2605131448/checkpoints/ckpt_2000/model.safetensors"
        ),
        help="Path to the Post-Train / Fine-Tune safetensors checkpoint.",
    )
    ap.add_argument("--backbone", default=str(ROOT / "resource/model/Qwen3-VL-2B-Instruct"))
    ap.add_argument(
        "--proprio-stats", default=str(ROOT / "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json"),
    )
    ap.add_argument(
        "--proprio-cache-root",
        default=str(ROOT / "resource/dataset/humanoid-everyday-cache/g1_proprio"),
        help="Directory of cached training-time proprio .npy (one per episode, "
             "50Hz, 64D).",
    )
    ap.add_argument(
        "--sonic-decoder",
        default=str(ROOT / "resource/model/sonic/model_decoder.onnx"),
        help="Path to the deployment SONIC decoder ONNX. Used for the "
             "second rollout per episode.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--episode-seconds", type=float, default=None,
                    help="Cap rollout length per episode (seconds). When "
                         "unset (default), each episode runs for its full HE "
                         "duration (n_frames / 30Hz).")
    ap.add_argument("--ctrl-hz", type=float, default=30.0)
    # 0.005 s matches the WBC's training-time sim_dt (sub-WBC inner loop runs
    # at 200 Hz). The legacy PD-only family used 0.002 s; we no longer support
    # PD-only heads, so 0.005 is the only relevant setting.
    ap.add_argument("--sim-dt", type=float, default=0.005)
    ap.add_argument("--s-min", type=int, default=8,
                    help="Re-query the policy every this many control steps. "
                         "8 ≈ 4Hz inference at ctrl_hz=30.")
    ap.add_argument("--delay-init", type=int, default=4,
                    help="Initial RTC prefix delay (overwritten by per-step "
                         "consumed count after the first chunk).")
    ap.add_argument("--num-inference-steps", type=int, default=4)
    ap.add_argument("--no-rtc", dest="use_rtc", action="store_false", default=True,
                    help="Disable RTC prefix substitution (run vanilla "
                         "chunk-by-chunk inference).")
    ap.add_argument("--heads", nargs="+",
                    default=["gt_wbc", "sonic"],
                    choices=["wbc_idle", "gt_wbc", "sonic"],
                    help="Which control heads to roll out per episode. "
                         "'wbc_idle' attaches the GR00T WBC ONNX and verifies "
                         "it can hold the robot standing for --episode-seconds "
                         "(default 30s) with zero loco_cmd; ignores episode "
                         "content. 'gt_wbc' is the perfect-upper-body baseline "
                         "with WBC driving legs+waist (GT arms + GT-derived "
                         "loco_cmd). 'sonic' decodes the model's SONIC token "
                         "chunk into 29-D joint targets and drives the whole "
                         "body via PD position tracking — no WBC.")
    ap.add_argument(
        "--wbc-balance-onnx",
        default=str(ROOT / "humanoid everyday/GR00T-WholeBodyControl/decoupled_wbc/"
                    "sim2mujoco/resources/robots/g1/policy/"
                    "GR00T-WholeBodyControl-Balance.onnx"),
        help="Path to the GR00T WBC Balance ONNX (used when ||loco_cmd|| <= 0.05).",
    )
    ap.add_argument(
        "--wbc-walk-onnx",
        default=str(ROOT / "humanoid everyday/GR00T-WholeBodyControl/decoupled_wbc/"
                    "sim2mujoco/resources/robots/g1/policy/"
                    "GR00T-WholeBodyControl-Walk.onnx"),
        help="Path to the GR00T WBC Walk ONNX (used when ||loco_cmd|| > 0.05).",
    )
    ap.add_argument(
        "--proprio-source", choices=["cache", "sim"], default="sim",
        help="Where the policy-facing 64-D proprio comes from for the "
             "sonic head. 'sim' (default) assembles live from sim state "
             "via SimProprioAssembler — the deployment-faithful closed-"
             "loop channel. 'cache' pulls from the 50 Hz HE training "
             "cache; useful only as an OOD-free debug path.",
    )
    ap.add_argument(
        "--proprio-diff-log", action="store_true", default=False,
        help="On the sonic head, compute BOTH the cache and the sim "
             "proprio at every inference tick and log per-channel deltas "
             "to <out-dir>/.../sonic_proprio_diff.npz. Used as a "
             "regression instrument when validating the SimProprioAssembler.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-indices", type=int, nargs="*", default=[7831],
                    help="Override the random sample with explicit HE "
                         "episode indices.")
    ap.add_argument(
        "--out-dir", default=str(ROOT / ".runs/eval/latest"),
        help="Output root; per-rollout files land under "
             "<out-dir>/episode_<idx>/ with head-prefixed filenames.",
    )
    args = ap.parse_args()

    if args.proprio_diff_log:
        policy_heads = {"sonic"}
        if not (set(args.heads) & policy_heads):
            ap.error(
                "--proprio-diff-log requires 'sonic' in --heads "
                "(it only emits diff records on the policy-driven head)."
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load policy (only if a model head is requested) ----
    model_heads = {"sonic"}
    pol: KappaPolicy | None = None
    if any(h in model_heads for h in args.heads):
        print(f"[main] loading policy from {args.checkpoint}")
        pol = KappaPolicy(KappaPolicyConfig(
            weights=args.checkpoint,
            backbone=args.backbone,
            proprio_stats=args.proprio_stats,
            use_proprio=True,
            use_rtc=args.use_rtc,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        ))

    # ---- load WBC if requested ----
    wbc: G1LocomotionWBC | None = None
    if any(h in {"wbc_idle", "gt_wbc"} for h in args.heads):
        bal = Path(args.wbc_balance_onnx)
        wlk = Path(args.wbc_walk_onnx)
        for p in (bal, wlk):
            if not p.exists():
                raise FileNotFoundError(
                    f"WBC ONNX missing: {p}. Pass --wbc-balance-onnx / "
                    f"--wbc-walk-onnx or drop 'wbc_idle' from --heads."
                )
        device_id = None
        if args.device.startswith("cuda"):
            try:
                device_id = int(args.device.split(":")[1]) if ":" in args.device else 0
            except ValueError:
                device_id = 0
        wbc = G1LocomotionWBC(balance_onnx_path=bal, walk_onnx_path=wlk, device_id=device_id)
        print(f"[main] loaded WBC: balance={bal.name} walk={wlk.name}")

    sonic_decoder = None
    if "sonic" in args.heads:
        decoder_path = Path(args.sonic_decoder)
        if not decoder_path.exists():
            raise FileNotFoundError(
                f"SONIC decoder ONNX missing: {decoder_path}. Pass "
                f"--sonic-decoder or drop 'sonic' from --heads."
            )
        # Decoder rides on its own ORT session; CUDAExecutionProvider here is
        # fine to share the same physical GPU as the policy's torch context.
        device_id = None
        if args.device.startswith("cuda"):
            try:
                device_id = int(args.device.split(":")[1]) if ":" in args.device else 0
            except ValueError:
                device_id = 0
        sonic_decoder = SonicDecoder(decoder_path, device_id=device_id)
        print(f"[main] loaded SONIC decoder from {decoder_path}")

    # ---- pick episodes ----
    src = HEEpisodeSource(
        proprio_cache_root=args.proprio_cache_root,
        seed=args.seed,
    )
    if args.episode_indices:
        episodes = [src.by_index(i) for i in args.episode_indices]
    else:
        episodes = src.sample(args.num_episodes, seed=args.seed)
    print(f"[main] {len(episodes)} episodes selected: {[e.episode_index for e in episodes]}")

    # ---- build env (single instance, reset between rollouts) ----
    # All supported heads use the WBC, which requires a free-base sim with
    # sim_dt=0.005. PD-only heads (which needed fixed-base / sim_dt=0.002)
    # were removed alongside the 31→23 schema collapse.
    env = G1MujocoEnv(
        sim_dt=args.sim_dt,
        ctrl_dt=1.0 / args.ctrl_hz,
        fixed_base=False,
        enable_offscreen=True,
        # Both render dims must be divisible by libx264's macro_block_size=16
        # to avoid imageio's auto-resize fallback during MP4 muxing.
        render_width=480,
        render_height=352,
    )

    # ---- run rollouts ----
    all_metrics: list[dict] = []
    for ep in episodes:
        ep_dir = out_dir / f"episode_{ep.episode_index:06d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # Stash the source HE mp4 once per episode for visual comparison.
        source_video_dst = ep_dir / "source.mp4"
        if not source_video_dst.exists():
            shutil.copyfile(ep.video_path, source_video_dst)
        ep_seconds = (
            min(ep.duration_s, args.episode_seconds)
            if args.episode_seconds is not None
            else ep.duration_s
        )
        for head in args.heads:
            if head == "wbc_idle":
                m = _run_wbc_idle_rollout(
                    env=env,
                    wbc=wbc,
                    out_dir=ep_dir,
                    seconds=args.episode_seconds if args.episode_seconds is not None else 30.0,
                    warmup_seconds=0.5,
                )
                print(
                    f"[ep {ep.episode_index}/wbc_idle] "
                    f"fell={m['fell']}  "
                    f"root_z_drift={m['root_z_post_warmup_drift']*100:.1f}cm  "
                    f"pitch_p99={m['pitch_post_warmup_p99']:.3f}rad  "
                    f"wbc_ms_mean={m['wbc_step_ms_mean']:.1f}"
                )
                all_metrics.append(m)
                continue
            if head == "gt_wbc":
                m = _run_gt_wbc_rollout(
                    env=env,
                    wbc=wbc,
                    episode=ep,
                    source=src,
                    out_dir=ep_dir,
                    seconds=ep_seconds,
                    warmup_seconds=0.5,
                )
                print(
                    f"[ep {ep.episode_index}/gt_wbc] "
                    f"fell={m['fell']}  "
                    f"base_drift_p50={m['base_xy_drift_p50']*100:.1f}cm "
                    f"final={m['base_xy_drift_final']*100:.1f}cm  "
                    f"pitch_p99={m['pitch_post_warmup_p99']:.3f}rad"
                )
                all_metrics.append(m)
                continue
            if head == "sonic":
                m = _run_sonic_rollout(
                    env=env,
                    policy=pol,
                    sonic_decoder=sonic_decoder,
                    episode=ep,
                    source=src,
                    out_dir=ep_dir,
                    seconds=ep_seconds,
                    s_min=args.s_min,
                    delay_init=args.delay_init,
                    warmup_seconds=0.5,
                    proprio_source=args.proprio_source,
                    proprio_diff_log=args.proprio_diff_log,
                )
                print(
                    f"[ep {ep.episode_index}/sonic] "
                    f"fell={m['fell']}  "
                    f"arm_fit_p99={m['arm_fit_err_per_joint_p99']:.3f}rad  "
                    f"leg_fit_p99={m['leg_fit_err_per_joint_p99']:.3f}rad  "
                    f"arm_track_p99={m['arm_tracking_err_p99']:.3f}rad  "
                    f"base_drift_final={m['base_xy_drift_final']*100:.1f}cm  "
                    f"pitch_p99={m['pitch_post_warmup_p99']:.3f}rad"
                )
                all_metrics.append(m)
                continue
            # argparse's choices= guarantees we never reach here.
            raise RuntimeError(f"Unknown head {head!r} (dispatcher unreachable)")

    # ---- aggregate summary (per-head averages) ----
    keys_to_avg = (
        "arm_fit_err_per_joint_p50", "arm_fit_err_per_joint_p99",
        "leg_fit_err_per_joint_p50", "leg_fit_err_per_joint_p99",
        "arm_tracking_err_p50", "arm_tracking_err_p99",
        "leg_tracking_err_p50", "leg_tracking_err_p99",
        "arm_action_diff_p99",
        "base_xy_drift_p50", "base_xy_drift_final",
        "root_z_post_warmup_mean",
        "roll_post_warmup_p99", "pitch_post_warmup_p99",
        "max_contact_force_total",
        "inference_ms_mean", "inference_ms_p99",
        "decoder_ms_mean", "decoder_ms_p99",
        "wbc_step_ms_mean", "wbc_step_ms_p99",
    )
    summary = {
        "checkpoint": args.checkpoint,
        "heads": args.heads,
        "num_episodes": len(episodes),
        "episode_seconds_cap": args.episode_seconds,
        "use_rtc": args.use_rtc,
        "proprio_source": args.proprio_source,
        "proprio_diff_log": args.proprio_diff_log,
        "num_inference_steps": args.num_inference_steps,
        "ctrl_hz": args.ctrl_hz,
        "sim_dt": args.sim_dt,
        "s_min": args.s_min,
        "per_episode": all_metrics,
        "by_head": {},
    }
    for head in args.heads:
        head_metrics = [m for m in all_metrics if m["head"] == head]
        head_summary: dict = {
            "fell_count": int(sum(1 for m in head_metrics if m["fell"])),
            "n_rollouts": len(head_metrics),
        }
        for k in keys_to_avg:
            vals = [m[k] for m in head_metrics if m.get(k) is not None]
            head_summary[f"avg_{k}"] = float(np.mean(vals)) if vals else None
        summary["by_head"][head] = head_summary

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print(f"[main] summary saved to {summary_path}")
    for head in args.heads:
        s = summary["by_head"][head]
        print(f"  [{head}]")
        if head == "wbc_idle":
            head_metrics = [m for m in all_metrics if m["head"] == "wbc_idle"]
            drifts = [m["root_z_post_warmup_drift"] for m in head_metrics]
            pitches = [m["pitch_post_warmup_p99"] for m in head_metrics]
            print(f"    avg root_z drift (post-warmup): {np.mean(drifts)*100:.1f} cm")
            print(f"    avg pitch p99   (post-warmup):  {np.mean(pitches):.3f} rad")
            print(f"    fell_count:                     {s['fell_count']} / {s['n_rollouts']}")
            continue
        if head == "gt_wbc":
            head_metrics = [m for m in all_metrics if m["head"] == "gt_wbc"]
            drift_p50 = [m["base_xy_drift_p50"] for m in head_metrics]
            drift_final = [m["base_xy_drift_final"] for m in head_metrics]
            pitches = [m["pitch_post_warmup_p99"] for m in head_metrics]
            print(f"    avg base_xy drift p50: {np.mean(drift_p50)*100:.1f} cm")
            print(f"    avg base_xy drift final: {np.mean(drift_final)*100:.1f} cm")
            print(f"    avg pitch p99 (post-warmup):  {np.mean(pitches):.3f} rad")
            print(f"    fell_count:           {s['fell_count']} / {s['n_rollouts']}")
            continue
        if head == "sonic":
            head_metrics = [m for m in all_metrics if m["head"] == head]
            arm_fit99 = [m["arm_fit_err_per_joint_p99"] for m in head_metrics]
            leg_fit99 = [m["leg_fit_err_per_joint_p99"] for m in head_metrics]
            arm_trk99 = [m["arm_tracking_err_p99"] for m in head_metrics]
            leg_trk99 = [m["leg_tracking_err_p99"] for m in head_metrics]
            drift_final = [m["base_xy_drift_final"] for m in head_metrics]
            pitches = [m["pitch_post_warmup_p99"] for m in head_metrics]
            print(f"    avg arm fit_err_per_joint_p99:        {np.mean(arm_fit99):.4f} rad")
            print(f"    avg leg fit_err_per_joint_p99:        {np.mean(leg_fit99):.4f} rad")
            print(f"    avg arm tracking_err_p99:             {np.mean(arm_trk99):.4f} rad")
            print(f"    avg leg tracking_err_p99:             {np.mean(leg_trk99):.4f} rad")
            print(f"    avg base_xy drift final:              {np.mean(drift_final)*100:.1f} cm")
            print(f"    avg pitch p99 (post-warmup):          {np.mean(pitches):.3f} rad")
            print(f"    fell_count:                           {s['fell_count']} / {s['n_rollouts']}")
            continue
        raise RuntimeError(f"Unknown head {head!r} in summary block")


if __name__ == "__main__":
    main()

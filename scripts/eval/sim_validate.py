"""Physics-simulation validation for the Kappa post-train model.

Pipeline
--------
For each sampled HE episode the runner spins up MuJoCo G1 (29-DoF) and replays
the chosen ``--heads`` against it. Heads fall into two families that differ in
whether the rig has a balance controller (WBC) attached:

WBC family (sim_dt=0.005, free-base, walking-capable)
.....................................................

These heads attach the GR00T-WholeBodyControl Balance + Walk ONNX as a
lower-body locomotion controller. Legs+waist are driven by the WBC; arms by
the upstream policy (or GT). Both ``loco_cmd`` and the initial standing pose
match the WBC training distribution. A 0.5 s warmup phase at the start of
each rollout lets the WBC's 6-frame obs history fill; metrics over the
warmup window are reported separately (``warmup_steps`` in the json).

  * **wbc_idle** — 30 s standstill sanity. No episode content used; verifies
    the WBC keeps the robot upright in the Kappa env. Use this to debug WBC
    integration regressions.
  * **gt_wbc** — WBC + GT arms (``g1_action_at(t)[15:29]``) + GT-derived
    ``loco_cmd`` from pelvis odometry. The "perfect upper body" baseline:
    drift / pitch / fell here are pure controller error, no model influence.
  * **g1_expert_wbc** — WBC + Kappa's G1-head arm predictions (chunk[15:29]).
    The primary walking-episode evaluation channel; head-to-head against
    ``gt_wbc`` separates model error from controller error.
  * **sonic_wbc** — WBC + Kappa's SONIC chunk routed through SonicDecoder →
    arms only [15:29]. Mirrors the long-form deployment path. Note: arm
    tracking is expected to be worse than ``g1_expert_wbc`` due to the
    deployment decoder's 50 Hz training vs our 28.6 Hz outer rate (see
    auto-memory ``project_sonic_rate_mismatch``).

PD-only family (sim_dt=0.002, free-base or fixed-base)
......................................................

No WBC. These heads use the env's PD-position controller on all 29 joints.
Free-base walking will topple in ~1 s (see auto-memory
``project_offline_rig_no_wbc``); use ``--fixed-base`` for joint-only fit
quality or restrict these heads to non-walking episodes.

  * **g1_expert** — de-normalized 31-D G1 chunk → PD on all 29 joints.
  * **sonic** — SONIC chunk → SonicDecoder → 29-D joint targets → PD on all.
  * **gt_physics** — dataset GT 31-D → PD on all 29 joints. The rig-vs-model
    attribution baseline: if this falls on a walking episode, no PD-only
    head can succeed on that episode and you should be evaluating with the
    WBC family.
  * **gt_replay** — dataset GT 31-D written kinematically into qpos with the
    floating base teleported to the recorded pelvis odometry. Pure
    visualization; no physics, no balance.

Shared inputs across all model heads:

  1. Egocentric video frames pulled at every policy query (training-time
     image transform: resize-short-224 + center-crop 224×320 in PIL space).
  2. 50 Hz cached training-time proprio from the HE proprio cache.
  3. Policy queries every ``--s-min`` control steps with RTC prefix
     substitution to smooth chunk seams.

Per-episode outputs land under ``<out-dir>/episode_<idx>/`` with each file
prefixed by head name (e.g. ``g1_expert_track.mp4``,
``g1_expert_trajectory.{npz,png}``, ``g1_expert_metrics.json``). The source
HE video is copied to ``<out-dir>/episode_<idx>/source.mp4``. After all
episodes a single ``summary.json`` aggregates per-head metrics.

Usage
-----
::

    # Walking episode (use WBC heads — head-to-head vs perfect-arm baseline)
    uv run python scripts/eval/sim_validate.py \
        --heads gt_wbc g1_expert_wbc \
        --episode-indices 7831 --episode-seconds 8 \
        --out-dir .runs/eval/$(date +%Y%m%d_%H%M%S)

    # Fixed-base joint-fit evaluation (PD heads, no WBC needed)
    uv run python scripts/eval/sim_validate.py \
        --heads g1_expert sonic --fixed-base \
        --num-episodes 3 --episode-seconds 8 \
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


# ---------------------------- run a single ep ------------------------------


def _run_one_rollout(
    *,
    head: str,                 # "g1_expert" | "sonic" | "gt_replay" | "gt_physics"
    episode,
    source: HEEpisodeSource,
    policy: KappaPolicy | None,
    env: G1MujocoEnv,
    sonic_decoder: SonicDecoder | None,
    episode_seconds: float,
    s_min: int,
    delay_init: int,
    out_dir: Path,
    init_qpos: np.ndarray | None = None,
) -> dict:
    """Run a single rollout under one control head and return metrics.

    Heads:

    * ``g1_expert`` — env consumes the de-normalized G1 chunk directly.
    * ``sonic`` — each control tick runs the SONIC decoder over
      ``(sonic_token, robot_state_history)`` → 29-D IsaacLab action → 29-D
      MuJoCo joint targets. Shares the policy chunks with ``g1_expert``.
    * ``gt_replay`` — pulls the dataset's 31-D ground-truth action straight
      from the training cache and writes it into the env *kinematically*
      (no PD, no physics) with the floating base teleported to the recorded
      pelvis odometry. Useful as a "what the labels look like in MuJoCo"
      reference. Tracking and balance metrics are degenerate by construction.
    * ``gt_physics`` — same dataset GT actions as ``gt_replay``, but fed
      through the env's PD loop and full ``mj_step`` integration (identical
      control path to ``g1_expert``). This is the apples-to-apples baseline
      that tells you whether the *rig itself* (PD-only, no WBC) can keep the
      floating base upright on this episode's joint trajectory. If
      ``gt_physics`` also falls, model fit error is not the bottleneck —
      a balance controller is missing from the offline simulation harness.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    is_gt_replay = head == "gt_replay"
    is_gt_physics = head == "gt_physics"
    is_gt = is_gt_replay or is_gt_physics
    ctrl_dt = env.ctrl_dt
    n_total = int(round(episode_seconds / ctrl_dt))

    if is_gt:
        H = 0
        n_inference_period = 1
        stream = None
        print(f"[ep {episode.episode_index}/{head}] GT replay  ctrl_steps={n_total}")
    else:
        assert policy is not None, "model heads require a loaded policy"
        H = policy.mcfg.action_chunk_size
        n_inference_period = max(1, int(s_min))
        stream = FrameStream(episode.video_path)
        print(
            f"[ep {episode.episode_index}/{head}] frames={stream.n_frames} "
            f"({stream.n_frames / stream.fps:.1f}s)  ctrl_steps={n_total}  "
            f"chunk_H={H}  s_min={n_inference_period}"
        )

    # Spawn the robot at the episode's actual frame-0 pose rather than the
    # env's default standing pose. Without this, the PD controller sees a step
    # input on tick 0 (DEFAULT_QPOS → GT[0] differs by up to ~0.37 rad on
    # elbows) and yanks the joints, destabilizing free-base sims and producing
    # a misleading "fit error" spike for the model heads.
    env.reset(qpos_init=init_qpos)
    if head == "sonic":
        assert sonic_decoder is not None
        sonic_decoder.reset(env.joint_pos())

    # For the GT-replay head, set up a frame-0-relative transform from the
    # dataset's HE-world pelvis pose into the env's spawn frame. Without
    # this, the kinematic step would leave the floating base pinned and a
    # walking episode would look like in-place stepping. With it, the
    # rendered base position/orientation tracks the recorded odometry while
    # still starting at the env's spawn xyz (so the feet stay near the
    # floor and the tracking camera frames the robot from a sane angle).
    gt_base_spawn_pos: np.ndarray | None = None
    gt_base_spawn_quat: np.ndarray | None = None
    gt_he_init_pos: np.ndarray | None = None
    gt_he_init_quat_inv: np.ndarray | None = None
    gt_base_pose_available = False
    if is_gt_replay:
        try:
            he_init = source.pelvis_pose_at(episode.episode_index, 0.0)
            gt_he_init_pos = he_init[:3].astype(np.float64)
            gt_he_init_quat_inv = _quat_inv(he_init[3:7].astype(np.float64))
            base_init = env._base_qpos_init  # [x, y, z, qw, qx, qy, qz]
            gt_base_spawn_pos = base_init[:3].astype(np.float64).copy()
            gt_base_spawn_quat = base_init[3:7].astype(np.float64).copy()
            gt_base_pose_available = True
        except (FileNotFoundError, KeyError) as exc:
            print(
                f"[ep {episode.episode_index}/{head}] pelvis pose unavailable "
                f"({exc}); GT replay will keep the base pinned to spawn."
            )

    A_cur_norm: np.ndarray | None = None       # (H, 31) normalized G1 chunk
    A_cur_denorm: np.ndarray | None = None     # (H, 31) raw G1 chunk
    S_cur: np.ndarray | None = None            # (H, sonic_dim) SONIC chunk
    cur_idx = 0
    last_delay = int(delay_init)

    cmd_actions = np.zeros((n_total, 31), dtype=np.float32)
    gt_actions = np.zeros((n_total, 31), dtype=np.float32)
    sim_qpos = np.zeros((n_total, 29), dtype=np.float32)
    sim_qvel = np.zeros((n_total, 29), dtype=np.float32)
    root_zs = np.zeros(n_total, dtype=np.float32)
    rpys = np.zeros((n_total, 3), dtype=np.float32)
    contact_totals = np.zeros(n_total, dtype=np.float32)
    limit_violations = np.zeros(n_total, dtype=np.int32)
    inference_ms: list[float] = []
    decoder_ms: list[float] = []
    fell_at_step: int | None = None

    # GT actions are aligned to sim time t = step_idx * ctrl_dt and pulled
    # from the same 50 Hz cache the policy was trained on. ``g1_action_at``
    # clamps t past the episode length, so an over-cap rollout shows the last
    # GT frame rather than failing.
    try:
        gt_available = source.g1_action_at(episode.episode_index, 0.0) is not None
    except FileNotFoundError as exc:
        print(f"[ep {episode.episode_index}/{head}] GT actions unavailable: {exc}")
        gt_available = False

    track_path = out_dir / f"{head}_track.mp4"
    track_frames: list[np.ndarray] = []

    try:
        for step_idx in range(n_total):
            t = step_idx * ctrl_dt

            if is_gt:
                action = source.g1_action_at(episode.episode_index, t)
            else:
                need_inference = (A_cur_denorm is None) or (cur_idx >= n_inference_period)
                if need_inference:
                    obs_frame = _preprocess_frame(stream.frame_at(t))
                    # Pull proprio from the same training cache that fed the model
                    # at train time (50Hz, 64D). Using the simulator's pinned-base
                    # default-pose proprio here is OOD and produces flailing
                    # commands at the arm/wrist; sourcing from the cache keeps
                    # image+text+proprio all anchored to the training distribution.
                    proprio_raw = source.proprio_at(episode.episode_index, t)
                    t0 = time.perf_counter()
                    A_norm, A_denorm, S_chunk = policy.predict_chunk(
                        image=obs_frame,
                        text=episode.instruction,
                        proprio_64_raw=proprio_raw,
                        prev_g1_norm=A_cur_norm if A_cur_norm is not None else None,
                        delay=last_delay if A_cur_norm is not None else 0,
                        prev_sonic=S_cur if S_cur is not None else None,
                        delay_sonic=last_delay if S_cur is not None else 0,
                    )
                    inference_ms.append((time.perf_counter() - t0) * 1000.0)

                    if A_cur_norm is not None:
                        delay = min(cur_idx, H - 1)
                        A_norm[:delay] = A_cur_norm[H - delay:]
                        A_denorm[:delay] = A_cur_denorm[H - delay:]
                        if S_cur is not None:
                            S_chunk[:delay] = S_cur[H - delay:]
                        last_delay = delay if delay > 0 else last_delay
                    A_cur_norm = A_norm
                    A_cur_denorm = A_denorm
                    S_cur = S_chunk
                    cur_idx = 0

                # Pick the action for this control tick.
                if head == "g1_expert":
                    action = A_cur_denorm[cur_idx].astype(np.float32)
                else:  # head == "sonic"
                    # Push the just-observed robot state into the decoder history
                    # before stepping it (so the obs reflects "now" not "now-1").
                    sonic_decoder.push_state(
                        qpos_mujoco=env.joint_pos(),
                        qvel_mujoco=env.joint_vel(),
                        base_ang_vel_body=env.base_angular_velocity_body(),
                        base_quat_wxyz=env.root_quat_wxyz(),
                    )
                    d0 = time.perf_counter()
                    _, q_target_mj = sonic_decoder.step(S_cur[cur_idx])
                    decoder_ms.append((time.perf_counter() - d0) * 1000.0)
                    # Hand bits come from the SONIC token's last 2 dims; the env
                    # records them but the simplified 29-DoF body has no fingers.
                    hand_bits = np.asarray(S_cur[cur_idx, -2:], dtype=np.float32)
                    action = np.concatenate([q_target_mj, hand_bits]).astype(np.float32)
                cur_idx += 1

            # gt_replay is a *visualization* of the dataset's joint
            # trajectory: bypass PD/physics and write qpos directly so the
            # rendered video shows exactly the recorded joint angles. The
            # offline rig has no WBC, so PD-tracking GT targets with the
            # legs hanging from a pinned base would otherwise produce
            # large oscillations on the under-damped distal joints. Model
            # heads keep PD tracking because they're testing physical
            # executability of model outputs.
            if is_gt_replay:
                base_qpos = None
                if gt_base_pose_available:
                    he_pose = source.pelvis_pose_at(episode.episode_index, t)
                    he_pos = he_pose[:3].astype(np.float64)
                    he_quat = he_pose[3:7].astype(np.float64)
                    # Map the HE-world delta into the robot's local frame at
                    # episode start, then back into the sim's spawn frame
                    # (whose quat is the MJCF default — currently identity).
                    local_delta = _quat_rotate(
                        gt_he_init_quat_inv, he_pos - gt_he_init_pos,
                    )
                    sim_pos = gt_base_spawn_pos + _quat_rotate(
                        gt_base_spawn_quat, local_delta,
                    )
                    rel_quat = _quat_mul(gt_he_init_quat_inv, he_quat)
                    sim_quat = _quat_mul(gt_base_spawn_quat, rel_quat)
                    # Renormalize defensively — repeated mul drift is tiny but
                    # MuJoCo expects a unit quat for the free joint.
                    sim_quat /= max(float(np.linalg.norm(sim_quat)), 1e-12)
                    base_qpos = np.concatenate([sim_pos, sim_quat])
                stats = env.kinematic_step(action, base_qpos=base_qpos)
            else:
                stats = env.step(action)
            cmd_actions[step_idx] = action
            if gt_available:
                gt_actions[step_idx] = source.g1_action_at(episode.episode_index, t)
            sim_qpos[step_idx] = stats.qpos
            sim_qvel[step_idx] = stats.qvel
            root_zs[step_idx] = stats.root_z
            rpys[step_idx] = stats.root_rpy
            contact_totals[step_idx] = stats.contact_force_total
            limit_violations[step_idx] = stats.joint_limit_violation
            if stats.fell and fell_at_step is None:
                fell_at_step = step_idx

            track_frames.append(_saturate_image_uint8(env.render("track")))

    finally:
        if stream is not None:
            stream.close()

    media.write_video(track_path, track_frames, fps=int(round(1 / ctrl_dt)))

    # ---- per-episode metrics ----
    cmd_jp = cmd_actions[:, :29]
    cmd_diff = np.abs(np.diff(cmd_jp, axis=0))
    cmd_diff_norm = np.linalg.norm(np.diff(cmd_jp, axis=0), axis=1)
    track_err_abs = np.abs(sim_qpos - cmd_jp)

    # Fit-error metrics (cmd vs dataset GT). Only meaningful for the
    # ``g1_expert`` head, where ``cmd_actions`` IS the model's G1 prediction
    # in the same coordinate frame as ``g1_action.npy``. For the SONIC head
    # the comparison is informational (different control space).
    if gt_available:
        gt_jp = gt_actions[:, :29]
        fit_err_abs = np.abs(cmd_jp - gt_jp)
        fit_err_l2 = np.linalg.norm(cmd_jp - gt_jp, axis=1)
        fit_grip_err = float(np.mean(np.abs(cmd_actions[:, 29:] - gt_actions[:, 29:])))
    else:
        fit_err_abs = None
        fit_err_l2 = None
        fit_grip_err = None

    # Action range vs training stats (only meaningful for the G1 head; the
    # SONIC head's commands are not produced from the action_min/max
    # normalization, so the saturation reading is informational). For
    # ``gt_replay`` the comparison uses whichever stats happen to be loaded —
    # the policy's bounds, since GT is the model's training target so these
    # bounds were fit to the same distribution.
    if policy is not None:
        cmd_norm = (cmd_actions - policy.action_min) / (policy.action_max - policy.action_min) * 2.0 - 1.0
    else:
        cmd_norm = np.zeros_like(cmd_actions)

    metrics = {
        "head": head,
        "episode_index": episode.episode_index,
        "instruction": episode.instruction,
        "video_path_track": str(track_path.relative_to(out_dir.parent)),
        "control_steps": int(n_total),
        "duration_s": float(n_total * ctrl_dt),
        "n_inferences": int(len(inference_ms)),
        "inference_ms_mean": float(np.mean(inference_ms)) if inference_ms else None,
        "inference_ms_p50": _quantile(np.asarray(inference_ms), 0.5),
        "inference_ms_p90": _quantile(np.asarray(inference_ms), 0.9),
        "inference_ms_max": float(np.max(inference_ms)) if inference_ms else None,
        "decoder_ms_mean": float(np.mean(decoder_ms)) if decoder_ms else None,
        "decoder_ms_p99": _quantile(np.asarray(decoder_ms), 0.99) if decoder_ms else None,
        "action_diff_per_joint_p50": _quantile(cmd_diff, 0.5),
        "action_diff_per_joint_p99": _quantile(cmd_diff, 0.99),
        "action_diff_l2_p50": _quantile(cmd_diff_norm, 0.5),
        "action_diff_l2_p99": _quantile(cmd_diff_norm, 0.99),
        "tracking_err_p50": _quantile(track_err_abs, 0.5),
        "tracking_err_p99": _quantile(track_err_abs, 0.99),
        "fit_err_per_joint_p50": _quantile(fit_err_abs, 0.5) if fit_err_abs is not None else None,
        "fit_err_per_joint_p99": _quantile(fit_err_abs, 0.99) if fit_err_abs is not None else None,
        "fit_err_l2_p50": _quantile(fit_err_l2, 0.5) if fit_err_l2 is not None else None,
        "fit_err_l2_p99": _quantile(fit_err_l2, 0.99) if fit_err_l2 is not None else None,
        "fit_grip_mae": fit_grip_err,
        "cmd_norm_min": float(cmd_norm.min()),
        "cmd_norm_max": float(cmd_norm.max()),
        "cmd_norm_pct_outside_pm1": float(np.mean(np.abs(cmd_norm) > 1.0)) * 100.0,
        "fixed_base": bool(env.fixed_base),
        "fell": bool(fell_at_step is not None),
        "fell_at_step": int(fell_at_step) if fell_at_step is not None else None,
        "limit_violation_step_count": int((limit_violations > 0).sum()),
        "limit_violation_step_pct": float((limit_violations > 0).mean() * 100.0),
        "max_contact_force_total": float(contact_totals.max()),
    }

    (out_dir / f"{head}_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / f"{head}_trajectory.npz",
        cmd_actions=cmd_actions,
        gt_actions=gt_actions if gt_available else np.zeros_like(cmd_actions),
        gt_available=np.asarray(gt_available),
        sim_qpos=sim_qpos,
        sim_qvel=sim_qvel,
        root_z=root_zs,
        rpy=rpys,
        contact_total=contact_totals,
        limit_violations=limit_violations,
        inference_ms=np.asarray(inference_ms, dtype=np.float32),
        decoder_ms=np.asarray(decoder_ms, dtype=np.float32),
    )
    # For ``gt_*`` heads cmd IS the GT, so plotting both lines is degenerate;
    # those plots just show the input trajectory.
    plot_gt = gt_actions if (gt_available and not is_gt) else None
    _plot_trajectory(
        out_dir / f"{head}_trajectory.png",
        cmd_actions,
        ctrl_dt,
        head=head,
        gt_actions=plot_gt,
    )
    return metrics


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
      * arms (joints 15..28) driven by ``g1_action_at(t)[15:29]`` (dataset GT)
      * hand_bits taken from the same GT action

    A 0.5 s warmup phase holds the WBC standing pose with arms interpolated
    toward GT frame-0; the actual episode rollout starts after warmup. The
    pelvis x/y displacement vs the dataset's pelvis odometry is reported as
    ``base_xy_drift_*``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wbc.reset()

    ep_idx = episode.episode_index
    arm_gt0 = source.g1_action_at(ep_idx, 0.0)[15:29].astype(np.float32)
    hand_gt0 = source.g1_action_at(ep_idx, 0.0)[29:31].astype(np.float32)

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
            arm_target = source.g1_action_at(ep_idx, t_he)[15:29].astype(np.float32)
            hand_bits = source.g1_action_at(ep_idx, t_he)[29:31].astype(np.float32)
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


def _run_g1_expert_wbc_rollout(
    *,
    env: G1MujocoEnv,
    wbc: G1LocomotionWBC,
    policy: KappaPolicy,
    episode,
    source: HEEpisodeSource,
    out_dir: Path,
    seconds: float,
    s_min: int,
    delay_init: int,
    warmup_seconds: float = 0.5,
) -> dict:
    """Kappa-driven arms + WBC-driven legs/waist + GT-derived loco_cmd.

    Same Kappa chunk inference path as ``g1_expert`` (RTC prefix substitution,
    50 Hz proprio cache, training-time image transform) but only the arm
    channels [15:29] + hand_bits[29:31] of the predicted 31-D chunk are
    consumed; legs+waist are taken from the WBC. This is the "WBC + Kappa
    upper-body" deployment-mirror evaluation channel.

    Warmup: 0.5 s of WBC standstill with arms blended from ``DEFAULT_QPOS[15:29]``
    toward the dataset's GT arm pose at t=0. The Kappa policy is NOT queried
    during warmup — its first inference happens at the start of the episode
    so its proprio history doesn't include the warmup transient.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wbc.reset()

    ep_idx = episode.episode_index
    arm_gt0 = source.g1_action_at(ep_idx, 0.0)[15:29].astype(np.float32)
    hand_gt0 = source.g1_action_at(ep_idx, 0.0)[29:31].astype(np.float32)

    standing_qpos = np.concatenate([
        WBC_DEFAULT_ANGLES_15.astype(np.float64),
        DEFAULT_QPOS[15:29].astype(np.float64),
    ])
    env.reset(qpos_init=standing_qpos)
    env.enable_wbc_mode(wbc)

    ctrl_dt = env.ctrl_dt
    n_total = int(round(seconds / ctrl_dt))
    n_warmup = int(round(warmup_seconds / ctrl_dt))
    n_inference_period = max(1, int(s_min))
    H = policy.mcfg.action_chunk_size

    # Pelvis spawn-frame transform (same as gt_wbc / gt_replay) so we can
    # compare the sim pelvis xy against the dataset odometry in one frame.
    he_init = source.pelvis_pose_at(ep_idx, 0.0)
    gt_he_init_pos = he_init[:3].astype(np.float64)
    gt_he_init_quat_inv = _quat_inv(he_init[3:7].astype(np.float64))
    base_init = env._base_qpos_init
    gt_base_spawn_pos = base_init[:3].astype(np.float64).copy()
    gt_base_spawn_quat = base_init[3:7].astype(np.float64).copy()

    stream = FrameStream(episode.video_path)
    A_cur_norm: np.ndarray | None = None
    A_cur_denorm: np.ndarray | None = None
    cur_idx = 0
    last_delay = int(delay_init)
    inference_ms: list[float] = []

    cmd_actions = np.zeros((n_total, 31), dtype=np.float32)
    gt_actions = np.zeros((n_total, 31), dtype=np.float32)
    sim_qpos = np.zeros((n_total, 29), dtype=np.float32)
    sim_qvel = np.zeros((n_total, 29), dtype=np.float32)
    wbc_target = np.zeros((n_total, 15), dtype=np.float32)
    loco_cmds = np.zeros((n_total, 3), dtype=np.float32)
    base_xy_sim = np.zeros((n_total, 2), dtype=np.float32)
    base_xy_gt = np.zeros((n_total, 2), dtype=np.float32)
    root_zs = np.zeros(n_total, dtype=np.float32)
    rpys = np.zeros((n_total, 3), dtype=np.float32)
    contact_totals = np.zeros(n_total, dtype=np.float32)
    fell_at_step: int | None = None

    track_path = out_dir / "g1_expert_wbc_track.mp4"
    track_frames: list[np.ndarray] = []

    try:
        for step_idx in range(n_total):
            in_warmup = step_idx < n_warmup

            if in_warmup:
                blend = (step_idx + 1) / max(n_warmup, 1)
                arm_target = (
                    (1.0 - blend) * DEFAULT_QPOS[15:29].astype(np.float32)
                    + blend * arm_gt0
                )
                hand_bits = blend * hand_gt0
                loco_cmd = np.zeros(3, dtype=np.float64)
                t_he = 0.0
                # Record predictions as the warmup blend itself so plots remain
                # continuous; not "the model" but reflects the actual commanded
                # arm target.
                cmd_actions[step_idx, :15] = wbc.last_target_15
                cmd_actions[step_idx, 15:29] = arm_target
                cmd_actions[step_idx, 29:31] = hand_bits
            else:
                t_he = (step_idx - n_warmup) * ctrl_dt
                need_inference = (A_cur_denorm is None) or (cur_idx >= n_inference_period)
                if need_inference:
                    obs_frame = _preprocess_frame(stream.frame_at(t_he))
                    proprio_raw = source.proprio_at(ep_idx, t_he)
                    t0 = time.perf_counter()
                    A_norm, A_denorm, _ = policy.predict_chunk(
                        image=obs_frame,
                        text=episode.instruction,
                        proprio_64_raw=proprio_raw,
                        prev_g1_norm=A_cur_norm if A_cur_norm is not None else None,
                        delay=last_delay if A_cur_norm is not None else 0,
                        prev_sonic=None,
                        delay_sonic=0,
                    )
                    inference_ms.append((time.perf_counter() - t0) * 1000.0)
                    if A_cur_norm is not None:
                        delay = min(cur_idx, H - 1)
                        A_norm[:delay] = A_cur_norm[H - delay:]
                        A_denorm[:delay] = A_cur_denorm[H - delay:]
                        last_delay = delay if delay > 0 else last_delay
                    A_cur_norm = A_norm
                    A_cur_denorm = A_denorm
                    cur_idx = 0

                action_31 = A_cur_denorm[cur_idx].astype(np.float32)
                cur_idx += 1
                cmd_actions[step_idx] = action_31
                arm_target = action_31[15:29]
                hand_bits = action_31[29:31]
                loco_cmd = source.loco_cmd_at(ep_idx, t_he).astype(np.float64)

            stats = env.step_with_wbc(
                arm_target_14=arm_target,
                hand_bits_2=hand_bits,
                loco_cmd_3=loco_cmd,
                height_cmd=0.74,
                rpy_cmd_3=np.zeros(3, dtype=np.float64),
            )

            sim_qpos[step_idx] = stats.qpos
            sim_qvel[step_idx] = stats.qvel
            wbc_target[step_idx] = wbc.last_target_15
            loco_cmds[step_idx] = loco_cmd.astype(np.float32)
            base_xy_sim[step_idx] = env.data.qpos[env._free_qposadr : env._free_qposadr + 2]
            he_pos = source.pelvis_pose_at(ep_idx, t_he)[:3].astype(np.float64)
            local_delta = _quat_rotate(gt_he_init_quat_inv, he_pos - gt_he_init_pos)
            sim_pos_gt = gt_base_spawn_pos + _quat_rotate(gt_base_spawn_quat, local_delta)
            base_xy_gt[step_idx] = sim_pos_gt[:2]
            root_zs[step_idx] = stats.root_z
            rpys[step_idx] = stats.root_rpy
            contact_totals[step_idx] = stats.contact_force_total
            try:
                gt_actions[step_idx] = source.g1_action_at(ep_idx, t_he)
            except FileNotFoundError:
                pass
            if stats.fell and fell_at_step is None and not in_warmup:
                fell_at_step = step_idx
            track_frames.append(_saturate_image_uint8(env.render("track")))
    finally:
        stream.close()

    media.write_video(track_path, track_frames, fps=int(round(1 / ctrl_dt)))

    post = slice(n_warmup, n_total)
    arm_cmd_post = cmd_actions[post, 15:29]
    arm_gt_post = gt_actions[post, 15:29]
    arm_sim_post = sim_qpos[post, 15:29]
    arm_fit_err = np.abs(arm_cmd_post - arm_gt_post)
    arm_track_err = np.abs(arm_sim_post - arm_cmd_post)
    arm_diff = np.abs(np.diff(arm_cmd_post, axis=0)) if arm_cmd_post.shape[0] > 1 else np.zeros((1, 14))
    drift_xy = np.linalg.norm(base_xy_sim[post] - base_xy_gt[post], axis=1)

    metrics = {
        "head": "g1_expert_wbc",
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
        "n_inferences": int(len(inference_ms)),
        "inference_ms_mean": float(np.mean(inference_ms)) if inference_ms else None,
        "inference_ms_p99": _quantile(np.asarray(inference_ms), 0.99) if inference_ms else None,
        "arm_fit_err_per_joint_p50": _quantile(arm_fit_err, 0.5),
        "arm_fit_err_per_joint_p99": _quantile(arm_fit_err, 0.99),
        "arm_tracking_err_p50": _quantile(arm_track_err, 0.5),
        "arm_tracking_err_p99": _quantile(arm_track_err, 0.99),
        "arm_action_diff_p99": _quantile(arm_diff, 0.99),
        "root_z_post_warmup_mean": float(root_zs[post].mean()),
        "root_z_post_warmup_min": float(root_zs[post].min()),
        "roll_post_warmup_p99": float(np.quantile(np.abs(rpys[post, 0]), 0.99)),
        "pitch_post_warmup_p99": float(np.quantile(np.abs(rpys[post, 1]), 0.99)),
        "base_xy_drift_p50": float(np.quantile(drift_xy, 0.5)),
        "base_xy_drift_final": float(drift_xy[-1]),
        "loco_cmd_vx_p99": float(np.quantile(np.abs(loco_cmds[post, 0]), 0.99)),
        "loco_cmd_wz_p99": float(np.quantile(np.abs(loco_cmds[post, 2]), 0.99)),
        "max_contact_force_total": float(contact_totals.max()),
    }

    (out_dir / "g1_expert_wbc_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "g1_expert_wbc_trajectory.npz",
        cmd_actions=cmd_actions,
        gt_actions=gt_actions,
        sim_qpos=sim_qpos,
        sim_qvel=sim_qvel,
        wbc_target=wbc_target,
        loco_cmds=loco_cmds,
        base_xy_sim=base_xy_sim,
        base_xy_gt=base_xy_gt,
        root_z=root_zs,
        rpy=rpys,
        contact_total=contact_totals,
        inference_ms=np.asarray(inference_ms, dtype=np.float32),
        warmup_steps=np.asarray(n_warmup),
    )
    return metrics


def _run_sonic_wbc_rollout(
    *,
    env: G1MujocoEnv,
    wbc: G1LocomotionWBC,
    policy: KappaPolicy,
    sonic_decoder: SonicDecoder,
    episode,
    source: HEEpisodeSource,
    out_dir: Path,
    seconds: float,
    s_min: int,
    delay_init: int,
    warmup_seconds: float = 0.5,
) -> dict:
    """Kappa SONIC chunk → SonicDecoder → arms + WBC legs/waist + GT loco_cmd.

    Same control split as ``g1_expert_wbc`` but the arm targets come from the
    SONIC deployment decoder rather than directly from the policy's G1 chunk.
    Mirrors the long-form deployment path (Kappa → 64-D sonic token →
    SonicDecoder ONNX → 29-D action) but only consumes the arm channels
    [15:29] + hand_bits from the sonic token's last 2 dims.

    Known caveat (auto-memory: ``project_sonic_rate_mismatch.md``): the
    deployment SONIC decoder was trained at 50 Hz robot-state history but our
    rollout feeds it at the WBC outer rate (~28.6 Hz). Arm tracking error is
    expected to be larger than ``g1_expert_wbc`` for this reason — this is a
    rate-mismatch artifact, not a fit-quality problem.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wbc.reset()

    ep_idx = episode.episode_index
    arm_gt0 = source.g1_action_at(ep_idx, 0.0)[15:29].astype(np.float32)
    hand_gt0 = source.g1_action_at(ep_idx, 0.0)[29:31].astype(np.float32)

    standing_qpos = np.concatenate([
        WBC_DEFAULT_ANGLES_15.astype(np.float64),
        DEFAULT_QPOS[15:29].astype(np.float64),
    ])
    env.reset(qpos_init=standing_qpos)
    env.enable_wbc_mode(wbc)
    sonic_decoder.reset(env.joint_pos())

    ctrl_dt = env.ctrl_dt
    n_total = int(round(seconds / ctrl_dt))
    n_warmup = int(round(warmup_seconds / ctrl_dt))
    n_inference_period = max(1, int(s_min))
    H = policy.mcfg.action_chunk_size

    he_init = source.pelvis_pose_at(ep_idx, 0.0)
    gt_he_init_pos = he_init[:3].astype(np.float64)
    gt_he_init_quat_inv = _quat_inv(he_init[3:7].astype(np.float64))
    base_init = env._base_qpos_init
    gt_base_spawn_pos = base_init[:3].astype(np.float64).copy()
    gt_base_spawn_quat = base_init[3:7].astype(np.float64).copy()

    stream = FrameStream(episode.video_path)
    A_cur_norm: np.ndarray | None = None
    S_cur: np.ndarray | None = None
    cur_idx = 0
    last_delay = int(delay_init)
    inference_ms: list[float] = []
    decoder_ms: list[float] = []

    cmd_actions = np.zeros((n_total, 31), dtype=np.float32)
    gt_actions = np.zeros((n_total, 31), dtype=np.float32)
    sim_qpos = np.zeros((n_total, 29), dtype=np.float32)
    sim_qvel = np.zeros((n_total, 29), dtype=np.float32)
    wbc_target = np.zeros((n_total, 15), dtype=np.float32)
    loco_cmds = np.zeros((n_total, 3), dtype=np.float32)
    base_xy_sim = np.zeros((n_total, 2), dtype=np.float32)
    base_xy_gt = np.zeros((n_total, 2), dtype=np.float32)
    root_zs = np.zeros(n_total, dtype=np.float32)
    rpys = np.zeros((n_total, 3), dtype=np.float32)
    contact_totals = np.zeros(n_total, dtype=np.float32)
    fell_at_step: int | None = None

    track_path = out_dir / "sonic_wbc_track.mp4"
    track_frames: list[np.ndarray] = []

    try:
        for step_idx in range(n_total):
            in_warmup = step_idx < n_warmup
            if in_warmup:
                blend = (step_idx + 1) / max(n_warmup, 1)
                arm_target = (
                    (1.0 - blend) * DEFAULT_QPOS[15:29].astype(np.float32)
                    + blend * arm_gt0
                )
                hand_bits = blend * hand_gt0
                loco_cmd = np.zeros(3, dtype=np.float64)
                t_he = 0.0
                cmd_actions[step_idx, :15] = wbc.last_target_15
                cmd_actions[step_idx, 15:29] = arm_target
                cmd_actions[step_idx, 29:31] = hand_bits
            else:
                t_he = (step_idx - n_warmup) * ctrl_dt
                need_inference = (S_cur is None) or (cur_idx >= n_inference_period)
                if need_inference:
                    obs_frame = _preprocess_frame(stream.frame_at(t_he))
                    proprio_raw = source.proprio_at(ep_idx, t_he)
                    t0 = time.perf_counter()
                    A_norm, _, S_chunk = policy.predict_chunk(
                        image=obs_frame,
                        text=episode.instruction,
                        proprio_64_raw=proprio_raw,
                        prev_g1_norm=A_cur_norm if A_cur_norm is not None else None,
                        delay=last_delay if A_cur_norm is not None else 0,
                        prev_sonic=S_cur if S_cur is not None else None,
                        delay_sonic=last_delay if S_cur is not None else 0,
                    )
                    inference_ms.append((time.perf_counter() - t0) * 1000.0)
                    if A_cur_norm is not None:
                        delay = min(cur_idx, H - 1)
                        A_norm[:delay] = A_cur_norm[H - delay:]
                        S_chunk[:delay] = S_cur[H - delay:]
                        last_delay = delay if delay > 0 else last_delay
                    A_cur_norm = A_norm
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
                cur_idx += 1
                cmd_actions[step_idx, :29] = q_target_mj
                cmd_actions[step_idx, 29:31] = hand_bits
                arm_target = q_target_mj[15:29].astype(np.float32)
                loco_cmd = source.loco_cmd_at(ep_idx, t_he).astype(np.float64)

            stats = env.step_with_wbc(
                arm_target_14=arm_target,
                hand_bits_2=hand_bits,
                loco_cmd_3=loco_cmd,
                height_cmd=0.74,
                rpy_cmd_3=np.zeros(3, dtype=np.float64),
            )

            sim_qpos[step_idx] = stats.qpos
            sim_qvel[step_idx] = stats.qvel
            wbc_target[step_idx] = wbc.last_target_15
            loco_cmds[step_idx] = loco_cmd.astype(np.float32)
            base_xy_sim[step_idx] = env.data.qpos[env._free_qposadr : env._free_qposadr + 2]
            he_pos = source.pelvis_pose_at(ep_idx, t_he)[:3].astype(np.float64)
            local_delta = _quat_rotate(gt_he_init_quat_inv, he_pos - gt_he_init_pos)
            sim_pos_gt = gt_base_spawn_pos + _quat_rotate(gt_base_spawn_quat, local_delta)
            base_xy_gt[step_idx] = sim_pos_gt[:2]
            root_zs[step_idx] = stats.root_z
            rpys[step_idx] = stats.root_rpy
            contact_totals[step_idx] = stats.contact_force_total
            try:
                gt_actions[step_idx] = source.g1_action_at(ep_idx, t_he)
            except FileNotFoundError:
                pass
            if stats.fell and fell_at_step is None and not in_warmup:
                fell_at_step = step_idx
            track_frames.append(_saturate_image_uint8(env.render("track")))
    finally:
        stream.close()

    media.write_video(track_path, track_frames, fps=int(round(1 / ctrl_dt)))

    post = slice(n_warmup, n_total)
    arm_cmd_post = cmd_actions[post, 15:29]
    arm_gt_post = gt_actions[post, 15:29]
    arm_sim_post = sim_qpos[post, 15:29]
    arm_fit_err = np.abs(arm_cmd_post - arm_gt_post)
    arm_track_err = np.abs(arm_sim_post - arm_cmd_post)
    arm_diff = np.abs(np.diff(arm_cmd_post, axis=0)) if arm_cmd_post.shape[0] > 1 else np.zeros((1, 14))
    drift_xy = np.linalg.norm(base_xy_sim[post] - base_xy_gt[post], axis=1)

    metrics = {
        "head": "sonic_wbc",
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
        "n_inferences": int(len(inference_ms)),
        "inference_ms_mean": float(np.mean(inference_ms)) if inference_ms else None,
        "inference_ms_p99": _quantile(np.asarray(inference_ms), 0.99) if inference_ms else None,
        "decoder_ms_mean": float(np.mean(decoder_ms)) if decoder_ms else None,
        "decoder_ms_p99": _quantile(np.asarray(decoder_ms), 0.99) if decoder_ms else None,
        "arm_fit_err_per_joint_p50": _quantile(arm_fit_err, 0.5),
        "arm_fit_err_per_joint_p99": _quantile(arm_fit_err, 0.99),
        "arm_tracking_err_p50": _quantile(arm_track_err, 0.5),
        "arm_tracking_err_p99": _quantile(arm_track_err, 0.99),
        "arm_action_diff_p99": _quantile(arm_diff, 0.99),
        "root_z_post_warmup_mean": float(root_zs[post].mean()),
        "root_z_post_warmup_min": float(root_zs[post].min()),
        "roll_post_warmup_p99": float(np.quantile(np.abs(rpys[post, 0]), 0.99)),
        "pitch_post_warmup_p99": float(np.quantile(np.abs(rpys[post, 1]), 0.99)),
        "base_xy_drift_p50": float(np.quantile(drift_xy, 0.5)),
        "base_xy_drift_final": float(drift_xy[-1]),
        "loco_cmd_vx_p99": float(np.quantile(np.abs(loco_cmds[post, 0]), 0.99)),
        "loco_cmd_wz_p99": float(np.quantile(np.abs(loco_cmds[post, 2]), 0.99)),
        "max_contact_force_total": float(contact_totals.max()),
    }

    (out_dir / "sonic_wbc_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "sonic_wbc_trajectory.npz",
        cmd_actions=cmd_actions,
        gt_actions=gt_actions,
        sim_qpos=sim_qpos,
        sim_qvel=sim_qvel,
        wbc_target=wbc_target,
        loco_cmds=loco_cmds,
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


def _plot_trajectory(
    out_path: Path,
    cmd_actions: np.ndarray,  # (T, 31)
    ctrl_dt: float,
    head: str = "g1_expert",
    gt_actions: np.ndarray | None = None,  # (T, 31), training-set GT
) -> None:
    """Plot the 29 body joints over time: model command vs dataset GT.

    Sim qpos is intentionally omitted — this plot is for *fit quality*
    (how close cmd is to the supervised target), not tracking quality.
    """
    from matplotlib.lines import Line2D

    T = cmd_actions.shape[0]
    t = np.arange(T) * ctrl_dt
    groups: list[tuple[str, slice]] = [
        ("legs (12)", slice(0, 12)),
        ("waist (3)", slice(12, 15)),
        ("left arm (7)", slice(15, 22)),
        ("right arm (7)", slice(22, 29)),
    ]
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 10), sharex=True)
    # Per-joint plots: same color for cmd and GT of a given joint, two
    # styles to disambiguate. cmd on top so the model output is the
    # dominant trace.
    for ax, (label, sl) in zip(axes, groups):
        cmd = cmd_actions[:, sl]
        gt = gt_actions[:, sl] if gt_actions is not None else None
        for ji in range(cmd.shape[1]):
            color = palette[ji % len(palette)]
            if gt is not None:
                ax.plot(t, gt[:, ji], ":", color=color, alpha=0.7, linewidth=1.1)
            ax.plot(t, cmd[:, ji], "-", color=color, alpha=0.9, linewidth=0.9)
        ax.set_ylabel(f"{label} (rad)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")

    # Global style legend so readers know what each linestyle means without
    # peeking at every subplot's docs.
    style_handles = [Line2D([0], [0], color="k", linestyle="-", label="cmd (model)")]
    if gt_actions is not None:
        style_handles.append(Line2D([0], [0], color="k", linestyle=":", label="GT (dataset)"))
    fig.legend(handles=style_handles, loc="upper right",
               ncol=len(style_handles), fontsize=9, frameon=False)

    if gt_actions is not None:
        mae = float(np.mean(np.abs(cmd_actions[:, :29] - gt_actions[:, :29])))
        title = f"[{head}] cmd vs dataset GT  (joint MAE={mae:.4f} rad)"
    else:
        title = f"[{head}] action trajectory"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------- main loop -----------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--checkpoint", default=str(
            ROOT / ".runs/kappa/exp.kappa.b256.gpus8.2605102217/checkpoints/ckpt_2000/model.safetensors"
        ),
        help="Path to the post-train safetensors checkpoint.",
    )
    ap.add_argument("--backbone", default=str(ROOT / "resource/model/Qwen3-VL-2B-Instruct"))
    ap.add_argument(
        "--action-stats", default=str(ROOT / "resource/dataset/G1_WBT-cache/meta/g1_action_stats.json"),
    )
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
        "--g1-action-cache-root",
        default=str(ROOT / "resource/dataset/humanoid-everyday-cache/g1_actions"),
        help="Directory of cached training-time G1 actions .npy (one per "
             "episode, 50Hz, 31D = jp(29) + hand_bits(2)). Used to overlay "
             "ground-truth actions on the trajectory plot.",
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
    ap.add_argument("--sim-dt", type=float, default=0.002)
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
    ap.add_argument("--fixed-base", dest="fixed_base",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="Pin the pelvis in space so the rollout isolates "
                         "joint-level tracking. Default OFF so the rollout "
                         "exercises real floating-base dynamics — most "
                         "policies need to stay upright to be useful, and "
                         "the fixed-base 'hovering with hanging legs' mode "
                         "loads the distal joints with unrealistic torques. "
                         "Pass --fixed-base when isolating joint-level "
                         "tracking from balance.")
    ap.add_argument("--heads", nargs="+",
                    default=["g1_expert", "sonic", "gt_replay"],
                    choices=["g1_expert", "sonic", "gt_replay", "gt_physics", "wbc_idle", "gt_wbc", "g1_expert_wbc", "sonic_wbc"],
                    help="Which control heads to roll out per episode. "
                         "'gt_replay' kinematically replays the dataset's "
                         "GT actions with the base teleported to the recorded "
                         "odometry (no physics). 'gt_physics' feeds the same "
                         "GT actions through the env's PD + mj_step path (same "
                         "as g1_expert) so you can see whether the rig itself "
                         "can balance on the recorded joint trajectory without "
                         "a WBC. 'wbc_idle' attaches the GR00T WBC ONNX and "
                         "verifies it can hold the robot standing for "
                         "--episode-seconds (default 30s) with zero loco_cmd; "
                         "ignores episode content.")
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load policy (only if a model head is requested) ----
    model_heads = {"g1_expert", "sonic", "g1_expert_wbc", "sonic_wbc"}
    pol: KappaPolicy | None = None
    if any(h in model_heads for h in args.heads):
        print(f"[main] loading policy from {args.checkpoint}")
        pol = KappaPolicy(KappaPolicyConfig(
            weights=args.checkpoint,
            backbone=args.backbone,
            action_stats=args.action_stats,
            proprio_stats=args.proprio_stats,
            use_proprio=True,
            use_rtc=args.use_rtc,
            rtc_on_sonic=True,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
        ))

    # ---- load WBC if requested ----
    wbc: G1LocomotionWBC | None = None
    if any(h in {"wbc_idle", "gt_wbc", "g1_expert_wbc", "sonic_wbc"} for h in args.heads):
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
    if any(h in {"sonic", "sonic_wbc"} for h in args.heads):
        decoder_path = Path(args.sonic_decoder)
        if not decoder_path.exists():
            raise FileNotFoundError(
                f"SONIC decoder ONNX missing: {decoder_path}. Pass "
                f"--sonic-decoder or drop 'sonic'/'sonic_wbc' from --heads."
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
        g1_action_cache_root=args.g1_action_cache_root,
        seed=args.seed,
    )
    if args.episode_indices:
        episodes = [src.by_index(i) for i in args.episode_indices]
    else:
        episodes = src.sample(args.num_episodes, seed=args.seed)
    print(f"[main] {len(episodes)} episodes selected: {[e.episode_index for e in episodes]}")

    # ---- build env (single instance, reset between rollouts) ----
    env = G1MujocoEnv(
        sim_dt=args.sim_dt,
        ctrl_dt=1.0 / args.ctrl_hz,
        fixed_base=args.fixed_base,
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
        # Spawn pose: the dataset's frame-0 joint angles. Shared across all
        # heads in this episode so their rollouts begin from identical state.
        try:
            init_qpos = src.g1_action_at(ep.episode_index, 0.0)[:29].astype(np.float64)
        except FileNotFoundError:
            init_qpos = None
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
            if head == "g1_expert_wbc":
                m = _run_g1_expert_wbc_rollout(
                    env=env,
                    wbc=wbc,
                    policy=pol,
                    episode=ep,
                    source=src,
                    out_dir=ep_dir,
                    seconds=ep_seconds,
                    s_min=args.s_min,
                    delay_init=args.delay_init,
                    warmup_seconds=0.5,
                )
                print(
                    f"[ep {ep.episode_index}/g1_expert_wbc] "
                    f"fell={m['fell']}  "
                    f"arm_fit_p99={m['arm_fit_err_per_joint_p99']:.3f}rad  "
                    f"arm_track_p99={m['arm_tracking_err_p99']:.3f}rad  "
                    f"base_drift_final={m['base_xy_drift_final']*100:.1f}cm  "
                    f"pitch_p99={m['pitch_post_warmup_p99']:.3f}rad"
                )
                all_metrics.append(m)
                continue
            if head == "sonic_wbc":
                m = _run_sonic_wbc_rollout(
                    env=env,
                    wbc=wbc,
                    policy=pol,
                    sonic_decoder=sonic_decoder,
                    episode=ep,
                    source=src,
                    out_dir=ep_dir,
                    seconds=ep_seconds,
                    s_min=args.s_min,
                    delay_init=args.delay_init,
                    warmup_seconds=0.5,
                )
                print(
                    f"[ep {ep.episode_index}/sonic_wbc] "
                    f"fell={m['fell']}  "
                    f"arm_fit_p99={m['arm_fit_err_per_joint_p99']:.3f}rad  "
                    f"arm_track_p99={m['arm_tracking_err_p99']:.3f}rad  "
                    f"base_drift_final={m['base_xy_drift_final']*100:.1f}cm  "
                    f"pitch_p99={m['pitch_post_warmup_p99']:.3f}rad"
                )
                all_metrics.append(m)
                continue
            m = _run_one_rollout(
                head=head,
                episode=ep,
                source=src,
                policy=pol,
                env=env,
                sonic_decoder=sonic_decoder,
                episode_seconds=ep_seconds,
                s_min=args.s_min,
                delay_init=args.delay_init,
                out_dir=ep_dir,
                init_qpos=init_qpos,
            )
            infer_ms = m.get("inference_ms_mean")
            infer_str = f"{infer_ms:.0f}" if infer_ms is not None else "n/a"
            print(
                f"[ep {ep.episode_index}/{head}] "
                f"action_diff p50={m['action_diff_per_joint_p50']:.4f}rad "
                f"p99={m['action_diff_per_joint_p99']:.4f}rad  "
                f"sat_pct={m['cmd_norm_pct_outside_pm1']:.1f}%  "
                f"fell={m['fell']}  "
                f"infer_ms_mean={infer_str}"
            )
            all_metrics.append(m)

    # ---- aggregate summary (per-head averages) ----
    keys_to_avg = (
        "action_diff_per_joint_p50", "action_diff_per_joint_p99",
        "action_diff_l2_p50", "action_diff_l2_p99",
        "tracking_err_p50", "tracking_err_p99",
        "fit_err_per_joint_p50", "fit_err_per_joint_p99",
        "fit_err_l2_p50", "fit_err_l2_p99",
        "fit_grip_mae",
        "cmd_norm_pct_outside_pm1",
        "limit_violation_step_pct",
        "max_contact_force_total",
        "inference_ms_mean", "inference_ms_p50", "inference_ms_p90", "inference_ms_max",
        "decoder_ms_mean", "decoder_ms_p99",
    )
    summary = {
        "checkpoint": args.checkpoint,
        "heads": args.heads,
        "num_episodes": len(episodes),
        "episode_seconds_cap": args.episode_seconds,
        "use_rtc": args.use_rtc,
        "fixed_base": args.fixed_base,
        "num_inference_steps": args.num_inference_steps,
        "ctrl_hz": args.ctrl_hz,
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
        if head in {"g1_expert_wbc", "sonic_wbc"}:
            head_metrics = [m for m in all_metrics if m["head"] == head]
            fit99 = [m["arm_fit_err_per_joint_p99"] for m in head_metrics]
            trk99 = [m["arm_tracking_err_p99"] for m in head_metrics]
            drift_final = [m["base_xy_drift_final"] for m in head_metrics]
            pitches = [m["pitch_post_warmup_p99"] for m in head_metrics]
            print(f"    avg arm fit_err_per_joint_p99: {np.mean(fit99):.4f} rad")
            print(f"    avg arm tracking_err_p99:      {np.mean(trk99):.4f} rad")
            print(f"    avg base_xy drift final:       {np.mean(drift_final)*100:.1f} cm")
            print(f"    avg pitch p99 (post-warmup):   {np.mean(pitches):.3f} rad")
            print(f"    fell_count:                    {s['fell_count']} / {s['n_rollouts']}")
            continue
        print(f"    avg action_diff_per_joint_p50: {s['avg_action_diff_per_joint_p50']:.4f} rad")
        print(f"    avg action_diff_per_joint_p99: {s['avg_action_diff_per_joint_p99']:.4f} rad")
        if s.get("avg_fit_err_per_joint_p50") is not None:
            print(f"    avg fit_err_per_joint_p50:     {s['avg_fit_err_per_joint_p50']:.4f} rad")
            print(f"    avg fit_err_per_joint_p99:     {s['avg_fit_err_per_joint_p99']:.4f} rad")
        print(f"    avg cmd_norm_pct_outside_pm1:  {s['avg_cmd_norm_pct_outside_pm1']:.2f} %")
        print(f"    avg limit_violation_step_pct:  {s['avg_limit_violation_step_pct']:.2f} %")
        print(f"    fell_count:                    {s['fell_count']} / {s['n_rollouts']}")


if __name__ == "__main__":
    main()

"""Physics-simulation validation for the Kappa post-train model.

Pipeline
--------
For each sampled HE episode the runner queries the Kappa policy and runs the
predicted chunks against MuJoCo G1 (29-DoF) twice — once per output head — so
the two control paths can be compared side-by-side under identical
observations:

  * **g1_expert** — the third MoT expert's normalized 31-D joint+grip targets
    are de-normalized via the cached G1 action stats and fed directly into
    the env's PD-position controller. Mirrors the deployment-time fast path.
  * **sonic** — the predicted 64-D SONIC latent token is paired with the
    simulator's rolling robot-state history and fed through the deployment
    SONIC decoder ONNX (`resource/model/sonic/model_decoder.onnx`); the
    decoder's 29-D IsaacLab-order action delta is converted to absolute
    joint targets via ``default_angles_mujoco + g1_action_scale * action``
    and applied through the same PD loop. Mirrors the long-form deploy path
    that goes through the SONIC tokenizer.

Both rollouts share:

  1. Egocentric video frames pulled at every policy query from the source
     HE episode (so the policy's visual obs distribution stays anchored to
     training);
  2. 50Hz cached training-time proprio sourced from the HE proprio cache
     (so the visual / textual / proprio inputs all match training);
  3. Policy queries every ``--s-min`` control steps with RTC prefix
     substitution to smooth chunk seams (G1 always; SONIC when
     ``rtc_on_sonic`` is set).

Per-episode outputs land under ``<out-dir>/episode_<idx>/{g1_expert,sonic}/``;
each subdirectory holds ``rollout_track.mp4``, ``rollout_ego.mp4``,
``trajectory.{npz,png}``, and ``metrics.json``. The source HE video is also
copied to ``<out-dir>/episode_<idx>/source.mp4``. After all episodes a
single ``summary.json`` aggregates per-head action smoothness, tracking
error, joint-limit violations, and inference latency.

Usage
-----
::

    .venv/bin/python scripts/eval/sim_validate.py \
        --checkpoint .runs/kappa/exp.kappa.b2048.gpus8.2605070812/checkpoints/ckpt_34650/model.safetensors \
        --num-episodes 3 --episode-seconds 8 \
        --out-dir .runs/eval/$(date +%Y%m%d_%H%M%S)
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

import imageio
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
from src.eval.g1_mujoco_env import G1MujocoEnv
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


# ---------------------------- run a single ep ------------------------------


def _run_one_rollout(
    *,
    head: str,                 # "g1_expert" | "sonic" | "gt_replay"
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
      (no PD, no physics). The rendered video is exactly the dataset's
      joint trajectory; useful as a "what the labels look like in MuJoCo"
      reference. Tracking metrics are degenerate by construction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    is_gt_replay = head == "gt_replay"
    ctrl_dt = env.ctrl_dt
    n_total = int(round(episode_seconds / ctrl_dt))

    if is_gt_replay:
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

    track_path = out_dir / "rollout_track.mp4"
    ego_path = out_dir / "rollout_ego.mp4"
    track_writer = imageio.get_writer(str(track_path), fps=int(round(1 / ctrl_dt)), codec="libx264")
    ego_writer = imageio.get_writer(str(ego_path), fps=int(round(1 / ctrl_dt)), codec="libx264")

    try:
        for step_idx in range(n_total):
            t = step_idx * ctrl_dt

            if is_gt_replay:
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
                stats = env.kinematic_step(action)
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

            track_writer.append_data(_saturate_image_uint8(env.render("track")))
            ego_writer.append_data(_saturate_image_uint8(env.render("ego_view")))

    finally:
        track_writer.close()
        ego_writer.close()
        if stream is not None:
            stream.close()

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
        "video_path_track": str(track_path.relative_to(out_dir.parent.parent)),
        "video_path_ego": str(ego_path.relative_to(out_dir.parent.parent)),
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

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        out_dir / "trajectory.npz",
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
    # For ``gt_replay`` cmd IS the GT, so plotting both lines is degenerate;
    # the gt_replay plot just shows the input trajectory.
    plot_gt = gt_actions if (gt_available and not is_gt_replay) else None
    _plot_trajectory(
        out_dir / "trajectory.png",
        cmd_actions,
        ctrl_dt,
        head=head,
        gt_actions=plot_gt,
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
            ROOT / ".runs/kappa/exp.kappa.b2048.gpus8.2605070812/checkpoints/ckpt_34650/model.safetensors"
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
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Pin the pelvis in space so the rollout isolates "
                         "joint-level tracking. Default ON because the offline "
                         "rig has no WBC, so floating-base rollouts (--no-fixed-"
                         "base) tumble within ~1s regardless of joint-command "
                         "quality — even GT replay falls. Use --no-fixed-base "
                         "only when you specifically want to study base "
                         "stability.")
    ap.add_argument("--heads", nargs="+", default=["g1_expert", "sonic", "gt_replay"],
                    choices=["g1_expert", "sonic", "gt_replay"],
                    help="Which control heads to roll out per episode. "
                         "'gt_replay' feeds the dataset's ground-truth actions "
                         "straight into the env so you can see what the labels "
                         "look like under MuJoCo physics.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-indices", type=int, nargs="*", default=[7831],
                    help="Override the random sample with explicit HE "
                         "episode indices.")
    ap.add_argument(
        "--out-dir", default=str(ROOT / ".runs/eval/latest"),
        help="Output root; per-rollout files land under "
             "<out-dir>/episode_<idx>/<head>/.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load policy ----
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
            head_dir = ep_dir / head
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
                out_dir=head_dir,
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

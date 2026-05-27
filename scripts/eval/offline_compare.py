"""Offline Kappa inference vs ground-truth visualisation.

Picks one or more frames from a recorded episode, runs the trained Kappa
policy on each (using the same flow-matching sampler as the deploy server),
and plots predicted SONIC tokens against the ground-truth tokens already
materialised by ``scripts/preprocess/proprietary/calc_sonic_tokens.py``.

Usage
-----
    cd /home/kinetra-5090-01/code/liuyang/Kappa
    .venv/bin/python scripts/eval/offline_compare.py \\
        --ckpt-path .runs/kappa/exp.kappa.b1024.gpus8.2605262100/checkpoints/ckpt_3000 \\
        --data-root resource/dataset/proprietary/real/2026-05-23 \\
        --episode-idx 0 \\
        --num-windows 4 \\
        --out-dir .runs/eval/offline_compare/2026-05-23_ep0

Each evaluated window writes a single PNG containing the input image, the
predicted-vs-GT trajectories for a handful of representative motion dims,
the predicted-vs-GT hand-binary curves, and a per-dim absolute-error heatmap
over the prediction horizon. A summary PNG aggregates per-dim mean abs error
across all windows.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.run_kappa_server import (  # noqa: E402
    _detect_proprio_in_ckpt,
    _detect_sonic_dim_from_ckpt,
    _maybe_load_proprio_bounds,
    _normalize_proprio,
    _resolve_ckpt_file,
)
from src.data.video_reader import decode_frame  # noqa: E402


# =============================================================================
# Dataset access (one episode at a time, no DataLoader machinery)
# =============================================================================


@dataclass
class _Episode:
    ep_idx: int
    task: str
    sonic_tokens: np.ndarray   # (T, sonic_dim) float32, already normalised
    proprio: np.ndarray | None  # (T, 64) float32, raw (un-normalised)
    video_path: str
    n_frames: int


def _load_episode(data_root: Path, ep_idx: int, want_proprio: bool) -> _Episode:
    tokens_path = data_root / "sonic_tokens" / f"episode_{ep_idx:06d}.npy"
    if not tokens_path.is_file():
        raise FileNotFoundError(
            f"sonic_tokens npy missing: {tokens_path}. Did you run "
            "scripts/preprocess/proprietary/calc_sonic_tokens.py?"
        )
    tokens = np.load(tokens_path, mmap_mode="r").astype(np.float32, copy=False)
    n = int(tokens.shape[0])

    proprio: np.ndarray | None = None
    if want_proprio:
        prop_path = data_root / "g1_proprio" / f"episode_{ep_idx:06d}.npy"
        if not prop_path.is_file():
            raise FileNotFoundError(f"g1_proprio npy missing: {prop_path}")
        proprio = np.load(prop_path, mmap_mode="r").astype(np.float32, copy=False)
        if proprio.shape[0] != n:
            raise ValueError(
                f"proprio rows ({proprio.shape[0]}) disagree with tokens rows ({n})"
            )

    chunk_dir = ep_idx // 1000
    video_path = (
        data_root
        / f"videos/chunk-{chunk_dir:03d}/observation.images.ego_view"
        / f"episode_{ep_idx:06d}.mp4"
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"video missing: {video_path}")

    tasks_path = data_root / "meta" / "tasks.jsonl"
    task_map: dict[int, str] = {}
    if tasks_path.is_file():
        with open(tasks_path) as f:
            for line in f:
                row = json.loads(line)
                text = row.get("task") or row.get("description") or "perform the task"
                task_map[int(row["task_index"])] = str(text)

    ep_task = 0
    parquet = (
        data_root / f"data/chunk-{chunk_dir:03d}" / f"episode_{ep_idx:06d}.parquet"
    )
    if parquet.is_file():
        try:
            import polars as pl
            df = pl.read_parquet(str(parquet), columns=["task_index"])
            if df.height > 0:
                ep_task = int(df["task_index"][0])
        except Exception:
            ep_task = 0

    task_text = task_map.get(ep_task, "perform the task")
    return _Episode(
        ep_idx=ep_idx,
        task=task_text,
        sonic_tokens=tokens,
        proprio=proprio,
        video_path=str(video_path),
        n_frames=n,
    )


# =============================================================================
# Policy wrapper (mirrors KappaPolicy._sample_sonic without ZMQ scaffolding)
# =============================================================================


class _OfflinePolicy:
    def __init__(
        self,
        ckpt_path: str,
        base_vlm_path: str,
        device: str,
        action_horizon: int,
        num_inference_steps: int,
        use_proprio: bool,
        proprio_normalize_mode: str,
        seed: int | None,
    ):
        self.device = torch.device(device)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        ckpt_file = _resolve_ckpt_file(ckpt_path)
        print(f"[offline] resolved checkpoint: {ckpt_file}")

        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_file), device="cpu")
        has_proprio_in_ckpt = _detect_proprio_in_ckpt(state_dict)
        sonic_dim = _detect_sonic_dim_from_ckpt(state_dict, fallback=66)
        print(
            f"[offline] ckpt detected: sonic_dim={sonic_dim}, "
            f"use_proprio_in_ckpt={has_proprio_in_ckpt}"
        )

        from src.models.kappa_model import KappaModelWrapper
        wrapper = KappaModelWrapper(
            model_name_or_path=base_vlm_path,
            tune_vision=False,
            tune_llm=False,
            tune_mlp=False,
            device=self.device,
        )
        self.use_proprio = bool(use_proprio) and has_proprio_in_ckpt
        wrapper.convert_to_mot(
            sonic_dim=sonic_dim,
            action_loss_weight=1.0,
            use_proprio=has_proprio_in_ckpt,
            proprio_dim=64,
        )
        result = wrapper.model.load_state_dict(state_dict, strict=False)
        print(
            f"[offline] loaded {len(state_dict) - len(result.unexpected_keys)}/"
            f"{len(state_dict)} tensors "
            f"(missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)})"
        )
        wrapper.model.eval()
        wrapper.model.to(device=self.device, dtype=torch.bfloat16)

        self.wrapper = wrapper
        self.model = wrapper.model
        self.processor = wrapper.processor
        self.sonic_dim = int(sonic_dim)
        self.motion_dim = 64
        self.hand_dim = max(0, sonic_dim - 64)

        self.proprio_bounds = (
            _maybe_load_proprio_bounds(ckpt_file, 64, proprio_normalize_mode)
            if self.use_proprio else None
        )
        if self.use_proprio and self.proprio_bounds is None:
            print(
                "[offline] WARNING: proprio in ckpt but no proprio_stats.json found "
                "next to model.safetensors — proprio passed raw."
            )

    def _build_vlm_inputs(self, image_np: np.ndarray, prompt: str) -> dict[str, torch.Tensor]:
        from PIL import Image
        pil = Image.fromarray(image_np)
        conv = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False,
        )
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info([conv], image_patch_size=16)
        except Exception:
            image_inputs, video_inputs = [pil], None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        out: dict[str, torch.Tensor] = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
        }
        if "pixel_values" in inputs:
            out["pixel_values"] = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)
        if "image_grid_thw" in inputs:
            out["image_grid_thw"] = inputs["image_grid_thw"].to(self.device)
        return out

    @torch.no_grad()
    def sample(
        self,
        image_np: np.ndarray,
        prompt: str,
        proprio_raw: np.ndarray | None,
    ) -> np.ndarray:
        vlm_inputs = self._build_vlm_inputs(image_np, prompt)

        T_action = self.action_horizon
        T_p = 1 if (self.use_proprio and proprio_raw is not None) else 0

        proprio_t: torch.Tensor | None = None
        if T_p > 0:
            proprio_norm = _normalize_proprio(proprio_raw, self.proprio_bounds)
            proprio_t = torch.from_numpy(
                np.asarray(proprio_norm, dtype=np.float32).reshape(1, -1)
            ).to(self.device, dtype=torch.bfloat16)

        cache = self.model.encode_latent(
            input_ids=vlm_inputs["input_ids"],
            attention_mask=vlm_inputs.get("attention_mask"),
            pixel_values=vlm_inputs.get("pixel_values"),
            image_grid_thw=vlm_inputs.get("image_grid_thw"),
            T_action=T_action,
            T_proprio=T_p,
            proprio=proprio_t,
        )

        x = torch.randn(
            1, T_action, self.sonic_dim,
            device=self.device, dtype=torch.bfloat16,
        )
        steps = max(1, self.num_inference_steps)
        dt = 1.0 / steps
        for i in range(steps):
            t = 1.0 - i * dt
            timesteps = torch.full((1,), t, device=self.device, dtype=torch.float32)
            v = self.model.denoise_step(cache, x, timesteps)
            x = x - dt * v.to(x.dtype)
        return x.float().cpu().numpy()[0]  # (T_action, sonic_dim)


# =============================================================================
# Plotting
# =============================================================================


def _pick_dims(n: int, k: int) -> list[int]:
    """Evenly-spaced indices in [0, n)."""
    if n <= k:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]


def _plot_window(
    out_path: Path,
    image_np: np.ndarray,
    prompt: str,
    ep_idx: int,
    t: int,
    pred: np.ndarray,    # (T_action, sonic_dim)
    gt: np.ndarray,      # (T_eff, sonic_dim) — T_eff may be < T_action if tail-clipped
    motion_dim: int,
    hand_dim: int,
    motion_dims_to_show: list[int],
) -> None:
    T_action = pred.shape[0]
    T_eff = gt.shape[0]
    horizon = np.arange(T_action)

    # Figure layout: 1 image row + dim plots + hand row + heatmap.
    n_motion = len(motion_dims_to_show)
    n_rows = 1 + n_motion + (1 if hand_dim > 0 else 0) + 1
    fig = plt.figure(figsize=(11, 1.6 * n_rows + 1.5), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 2, height_ratios=[2.2] + [1.0] * (n_rows - 1))

    # --- input image + title
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_np)
    ax_img.set_title(f"input frame  (ep {ep_idx}, t={t})", fontsize=10)
    ax_img.axis("off")
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")
    ax_text.text(
        0.0, 0.95,
        f"prompt:\n{prompt}\n\nhorizon: {T_action} (GT visible: {T_eff})\n"
        f"motion_dim: {motion_dim}, hand_dim: {hand_dim}",
        ha="left", va="top", fontsize=9, family="monospace",
        transform=ax_text.transAxes,
    )

    # --- motion dim line plots
    for i, d in enumerate(motion_dims_to_show):
        ax = fig.add_subplot(gs[1 + i, :])
        ax.plot(horizon, pred[:, d], color="tab:red", lw=1.4, label="pred")
        ax.plot(np.arange(T_eff), gt[:, d], color="tab:blue", lw=1.4, alpha=0.85, label="gt")
        ax.set_ylabel(f"dim {d}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
        if i < n_motion - 1:
            ax.set_xticklabels([])

    row_cursor = 1 + n_motion

    # --- hand-binary (predicted continuous vs GT step)
    if hand_dim > 0:
        ax = fig.add_subplot(gs[row_cursor, :])
        for h in range(hand_dim):
            d = motion_dim + h
            label_p = f"pred hand{h}" if h == 0 else None
            label_g = f"gt hand{h}" if h == 0 else None
            ax.plot(horizon, pred[:, d], lw=1.4, alpha=0.85,
                    color=("tab:red" if h == 0 else "tab:orange"), label=label_p)
            ax.step(np.arange(T_eff), gt[:, d], where="post", lw=1.4, alpha=0.85,
                    color=("tab:blue" if h == 0 else "tab:cyan"), label=label_g)
        ax.set_ylabel("hand bits", fontsize=9)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        row_cursor += 1

    # --- per-dim |error| heatmap (T_eff x sonic_dim)
    ax = fig.add_subplot(gs[row_cursor, :])
    err = np.abs(pred[:T_eff] - gt)
    im = ax.imshow(
        err.T, aspect="auto", origin="lower", interpolation="nearest",
        cmap="magma", vmin=0.0, vmax=max(0.05, float(err.max())),
    )
    ax.set_xlabel("step in horizon")
    ax.set_ylabel("sonic dim")
    ax.set_title("|pred − gt|", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85)

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_full_episode(
    out_path: Path,
    timesteps: list[int],
    preds: list[np.ndarray],   # each (T_i, sonic_dim), T_i may be < H at tail
    gt_full: np.ndarray,       # (n_frames, sonic_dim)
    motion_dim: int,
    hand_dim: int,
    motion_dims_to_show: list[int],
    ep_idx: int,
    prompt: str,
) -> None:
    """Stitch per-window predictions into a single (n_frames, sonic_dim)
    array and plot pred-vs-GT timelines over the entire episode."""
    n_frames, sonic_dim = gt_full.shape
    pred_full = np.full((n_frames, sonic_dim), np.nan, dtype=np.float32)
    for t, p in zip(timesteps, preds):
        end = min(t + p.shape[0], n_frames)
        pred_full[t:end] = p[: end - t]

    boundaries = sorted(set(int(t) for t in timesteps if 0 < t < n_frames))

    n_motion = len(motion_dims_to_show)
    n_rows = n_motion + (1 if hand_dim > 0 else 0) + 2  # +error curve + heatmap
    fig = plt.figure(figsize=(14, 1.4 * n_rows + 0.6), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 1)
    times = np.arange(n_frames)

    for i, d in enumerate(motion_dims_to_show):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(times, gt_full[:, d], color="tab:blue", lw=1.0, alpha=0.85, label="gt")
        ax.plot(times, pred_full[:, d], color="tab:red", lw=1.0, label="pred")
        for b in boundaries:
            ax.axvline(b, color="gray", lw=0.4, alpha=0.35)
        ax.set_ylabel(f"dim {d}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(
                f"full-episode pred vs gt — ep {ep_idx}, prompt={prompt!r}",
                fontsize=10,
            )
        if i < n_motion - 1:
            ax.set_xticklabels([])

    row = n_motion
    if hand_dim > 0:
        ax = fig.add_subplot(gs[row, 0])
        colors_p = ("tab:red", "tab:orange")
        colors_g = ("tab:blue", "tab:cyan")
        for h in range(hand_dim):
            d = motion_dim + h
            ax.step(times, gt_full[:, d], where="post",
                    color=colors_g[h % len(colors_g)], lw=1.0, alpha=0.85,
                    label=f"gt hand{h}")
            ax.plot(times, pred_full[:, d],
                    color=colors_p[h % len(colors_p)], lw=1.0,
                    label=f"pred hand{h}")
        for b in boundaries:
            ax.axvline(b, color="gray", lw=0.4, alpha=0.35)
        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel("hand bits", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        row += 1

    err_full = np.abs(pred_full - gt_full)
    per_step_err = np.nanmean(err_full, axis=1)
    ax = fig.add_subplot(gs[row, 0])
    ax.plot(times, per_step_err, color="tab:purple", lw=1.0)
    for b in boundaries:
        ax.axvline(b, color="gray", lw=0.4, alpha=0.35)
    ax.set_ylabel("mean |pred − gt|", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels([])
    row += 1

    ax = fig.add_subplot(gs[row, 0])
    vmax = max(0.05, float(np.nanmax(err_full)))
    im = ax.imshow(
        err_full.T, aspect="auto", origin="lower", interpolation="nearest",
        cmap="magma", vmin=0.0, vmax=vmax,
    )
    ax.set_xlabel("episode tick (50 Hz)")
    ax.set_ylabel("sonic dim")
    fig.colorbar(im, ax=ax, shrink=0.85)

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_summary(
    out_path: Path,
    per_window_err: list[np.ndarray],  # each (T_eff_i, sonic_dim)
    motion_dim: int,
    hand_dim: int,
) -> None:
    sonic_dim = per_window_err[0].shape[1]
    # Mean abs error per dim, pooling over all windows + horizon steps.
    flat = np.concatenate(per_window_err, axis=0)  # (sum_T_eff, sonic_dim)
    per_dim = flat.mean(axis=0)
    per_step = [w.mean(axis=1) for w in per_window_err]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.0), constrained_layout=True)
    ax = axes[0]
    ax.bar(np.arange(sonic_dim), per_dim, color="tab:purple", alpha=0.85)
    ax.axvline(motion_dim - 0.5, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("sonic dim (0..motion_dim-1=motion, rest=hand)")
    ax.set_ylabel("mean |pred − gt|")
    ax.set_title("per-dim mean error across all windows")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    for i, e in enumerate(per_step):
        ax.plot(np.arange(e.size), e, lw=1.2, alpha=0.8, label=f"win {i}")
    ax.set_xlabel("step in horizon")
    ax.set_ylabel("mean |pred − gt| over all dims")
    ax.set_title("per-step mean error, per window")
    ax.grid(True, alpha=0.3)
    if len(per_step) <= 8:
        ax.legend(loc="upper left", fontsize=8, ncol=2)

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


@dataclass
class OfflineConfig:
    ckpt_path: str = ".runs/kappa/exp.kappa.b1024.gpus8.2605262100/checkpoints/ckpt_3000"
    """``model.safetensors`` file, ``ckpt_*`` directory, or a run directory."""

    data_root: str = "resource/dataset/proprietary/real/2026-05-23"
    """Root of the proprietary dataset (LeRobot v2.1 layout with sonic_tokens/)."""

    base_vlm_path: str = "resource/model/Qwen3-VL-2B-Instruct"

    episode_idx: int = 0
    """Episode index inside ``data_root``."""

    num_windows: int = 4
    """How many evenly-spaced windows to evaluate across the episode."""

    timesteps: tuple[int, ...] = ()
    """Explicit list of start indices into the episode. Overrides num_windows."""

    action_horizon: int = 50
    num_inference_steps: int = 4

    use_proprio: bool = True
    """If False, skip proprio even when the checkpoint was trained with it."""

    proprio_normalize_mode: str = "bounds_q99"

    device: str = "cuda:0"

    seed: int | None = 0

    motion_dims_to_show: tuple[int, ...] = (0, 8, 16, 24, 32, 40, 48, 56)
    """Subset of motion-token dims to draw line plots for. Defaults to 8 dims
    spread across the 64-D motion-token space."""

    tile_full: bool = False
    """Tile the episode end-to-end with non-overlapping ``action_horizon``-sized
    windows starting at t=0, H, 2H, …. Overrides ``timesteps`` and
    ``num_windows`` and additionally emits a stitched full-episode PNG."""

    per_window_plots: bool = True
    """Emit one PNG per window. Off by default when ``tile_full`` is set and
    there are >8 windows, since the stitched plot becomes the headline."""

    out_dir: str = ".runs/eval/offline_compare"


def main(cfg: OfflineConfig) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    data_root = Path(cfg.data_root).resolve()
    out_dir = Path(cfg.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[offline] data_root: {data_root}")
    print(f"[offline] out_dir:   {out_dir}")

    episode = _load_episode(
        data_root, cfg.episode_idx, want_proprio=cfg.use_proprio,
    )
    print(
        f"[offline] episode {episode.ep_idx}: {episode.n_frames} ticks, "
        f"task={episode.task!r}"
    )

    if cfg.tile_full:
        H = cfg.action_horizon
        timesteps = list(range(0, episode.n_frames, H))
    elif cfg.timesteps:
        timesteps = [int(t) for t in cfg.timesteps]
    else:
        max_t = max(0, episode.n_frames - 1)
        if cfg.num_windows <= 1:
            timesteps = [max_t // 2]
        else:
            timesteps = [
                int(round(i * max_t / (cfg.num_windows - 1)))
                for i in range(cfg.num_windows)
            ]
    timesteps = sorted(set(t for t in timesteps if 0 <= t < episode.n_frames))
    emit_per_window = cfg.per_window_plots and not (cfg.tile_full and len(timesteps) > 8)
    print(
        f"[offline] evaluating {len(timesteps)} timesteps "
        f"(tile_full={cfg.tile_full}, per_window={emit_per_window})"
    )

    policy = _OfflinePolicy(
        ckpt_path=cfg.ckpt_path,
        base_vlm_path=cfg.base_vlm_path,
        device=cfg.device,
        action_horizon=cfg.action_horizon,
        num_inference_steps=cfg.num_inference_steps,
        use_proprio=cfg.use_proprio,
        proprio_normalize_mode=cfg.proprio_normalize_mode,
        seed=cfg.seed,
    )

    motion_dims_to_show = [d for d in cfg.motion_dims_to_show if 0 <= d < policy.motion_dim]
    if not motion_dims_to_show:
        motion_dims_to_show = _pick_dims(policy.motion_dim, 8)

    per_window_err: list[np.ndarray] = []
    per_window_preds: list[np.ndarray] = []
    win_dir = out_dir / "windows" if cfg.tile_full else out_dir
    win_dir.mkdir(parents=True, exist_ok=True)
    for t in timesteps:
        image_np = decode_frame(
            episode.video_path, frame_idx=t, target_size=(320, 224),
        )
        proprio_raw: np.ndarray | None = None
        if policy.use_proprio and episode.proprio is not None:
            proprio_raw = np.asarray(episode.proprio[t], dtype=np.float32)

        pred = policy.sample(image_np, episode.task, proprio_raw)  # (T, sonic_dim)
        if pred.shape[1] != episode.sonic_tokens.shape[1]:
            raise RuntimeError(
                f"pred sonic_dim {pred.shape[1]} != gt sonic_dim "
                f"{episode.sonic_tokens.shape[1]}; checkpoint/dataset mismatch"
            )

        t_end = min(t + pred.shape[0], episode.n_frames)
        gt = np.asarray(episode.sonic_tokens[t:t_end], dtype=np.float32)

        win_msg = ""
        if emit_per_window:
            win_path = win_dir / f"ep{episode.ep_idx:06d}_t{t:06d}.png"
            _plot_window(
                win_path,
                image_np=image_np,
                prompt=episode.task,
                ep_idx=episode.ep_idx,
                t=t,
                pred=pred,
                gt=gt,
                motion_dim=policy.motion_dim,
                hand_dim=policy.hand_dim,
                motion_dims_to_show=motion_dims_to_show,
            )
            win_msg = f" → {win_path.name}"
        per_window_err.append(np.abs(pred[: gt.shape[0]] - gt))
        per_window_preds.append(pred[: gt.shape[0]])
        print(
            f"[offline]   t={t:>5d}{win_msg}  "
            f"L1/dim={per_window_err[-1].mean():.4f}"
        )

    if per_window_err:
        summary_path = out_dir / f"ep{episode.ep_idx:06d}_summary.png"
        _plot_summary(
            summary_path,
            per_window_err=per_window_err,
            motion_dim=policy.motion_dim,
            hand_dim=policy.hand_dim,
        )
        flat = np.concatenate(per_window_err, axis=0)
        print(
            f"[offline] summary → {summary_path.name}  "
            f"overall L1/dim={flat.mean():.4f}, "
            f"motion={flat[:, :policy.motion_dim].mean():.4f}, "
            f"hand={flat[:, policy.motion_dim:].mean():.4f}"
        )

    if cfg.tile_full:
        full_path = out_dir / f"ep{episode.ep_idx:06d}_full.png"
        _plot_full_episode(
            full_path,
            timesteps=timesteps,
            preds=per_window_preds,
            gt_full=np.asarray(episode.sonic_tokens, dtype=np.float32),
            motion_dim=policy.motion_dim,
            hand_dim=policy.hand_dim,
            motion_dims_to_show=motion_dims_to_show,
            ep_idx=episode.ep_idx,
            prompt=episode.task,
        )
        print(f"[offline] full-episode → {full_path.name}")


if __name__ == "__main__":
    main(tyro.cli(OfflineConfig))

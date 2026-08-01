#!/usr/bin/env python3
"""Slice and visualize AgiBotWorld SONIC planner/token preview artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CALC_PROPRIO_DIR = REPO_ROOT / "scripts/preprocess/calc_proprio"
if str(CALC_PROPRIO_DIR) not in sys.path:
    sys.path.insert(0, str(CALC_PROPRIO_DIR))

import run as replay


def _arrays(trajectory: replay.PlannerTrajectory) -> dict[str, np.ndarray]:
    return {
        "mode": trajectory.mode.astype(np.float32),
        "movement": trajectory.movement.astype(np.float32),
        "facing": trajectory.facing.astype(np.float32),
        "speed": trajectory.speed.astype(np.float32),
        "height": trajectory.height.astype(np.float32),
        "vr_position": trajectory.vr_position.astype(np.float32),
        "vr_orientation": trajectory.vr_orientation.astype(np.float32),
        "hand_state": trajectory.hand_state.astype(np.float32),
    }


def _slice(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.output_trajectory is None:
        return None
    trajectory = replay.load_planner_trajectory(args.trajectory)
    end = min(args.start + args.frames, trajectory.rows)
    if not 0 <= args.start < end <= trajectory.rows:
        raise ValueError(f"invalid slice [{args.start}, {end}) for {trajectory.rows} rows")
    arrays = {
        name: value[args.start:end]
        for name, value in _arrays(trajectory).items()
    }
    replay.atomic_npz(args.output_trajectory, arrays)
    sliced = replay.load_planner_trajectory(args.output_trajectory)
    return {
        "source_trajectory": str(args.trajectory),
        "output_trajectory": str(args.output_trajectory),
        "start": args.start,
        "end": end,
        "rows": sliced.rows,
        "trajectory_check": replay.validate_planner_trajectory(sliced),
    }


def _plot(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.output_dir is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = replay.load_planner_trajectory(args.trajectory)
    positions = trajectory.vr_position.reshape(trajectory.rows, 3, 3)
    time = np.arange(trajectory.rows, dtype=np.float32) / 50.0
    names = ("left", "right", "head")
    colors = ("tab:blue", "tab:orange", "tab:green")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    projections = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    for ax, (a, b, la, lb) in zip(axes, projections):
        for point, color, name in zip(range(3), colors, names):
            ax.plot(positions[:, point, a], positions[:, point, b], color=color, label=name)
            ax.scatter(positions[0, point, a], positions[0, point, b], color=color, s=16)
        ax.set_xlabel(f"{la} (m)")
        ax.set_ylabel(f"{lb} (m)")
        ax.set_title(f"VR 3pt {la}{lb}")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    path_3pt = args.output_dir / "trajectory_3pt.png"
    fig.savefig(path_3pt, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, trajectory.height)
    axes[0].set_ylabel("height (m)")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time, trajectory.mode)
    axes[1].set_ylabel("mode")
    axes[1].grid(True, alpha=0.25)
    axes[2].step(time, trajectory.hand_state[:, 0], where="post", label="left")
    axes[2].step(time, trajectory.hand_state[:, 1], where="post", label="right")
    axes[2].set_ylabel("closed")
    axes[2].set_xlabel("time (s)")
    axes[2].set_yticks([0, 1])
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.25)
    path_controls = args.output_dir / "planner_controls.png"
    fig.savefig(path_controls, dpi=160)
    plt.close(fig)

    token_summary = None
    token_plots: list[str] = []
    if args.token is not None:
        token = np.load(args.token)
        if token.ndim != 2 or token.shape[1] != 66:
            raise ValueError(f"expected token shape [T,66], got {token.shape}")
        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
        image = ax.imshow(token.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
        ax.set_title("SONIC token preview")
        ax.set_xlabel("frame")
        ax.set_ylabel("dimension")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
        path_heatmap = args.output_dir / "sonic_token_heatmap.png"
        fig.savefig(path_heatmap, dpi=160)
        plt.close(fig)
        token_plots.append(str(path_heatmap))

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
        axes[0].plot(np.arange(len(token)) / 50.0, np.linalg.norm(token[:, :64], axis=1))
        axes[0].set_ylabel("||token[:64]||")
        axes[0].grid(True, alpha=0.25)
        axes[1].step(np.arange(len(token)) / 50.0, token[:, 64], where="post", label="left")
        axes[1].step(np.arange(len(token)) / 50.0, token[:, 65], where="post", label="right")
        axes[1].set_ylabel("closed")
        axes[1].set_xlabel("time (s)")
        axes[1].set_yticks([0, 1])
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.25)
        path_token_series = args.output_dir / "sonic_token_series.png"
        fig.savefig(path_token_series, dpi=160)
        plt.close(fig)
        token_plots.append(str(path_token_series))
        token_summary = {
            "path": str(args.token),
            "shape": list(token.shape),
            "dtype": str(token.dtype),
            "finite": bool(np.isfinite(token).all()),
            "motion_token_min": float(token[:, :64].min()),
            "motion_token_max": float(token[:, :64].max()),
            "hand_unique": np.unique(token[:, 64:66]).tolist(),
        }

    proprio_summary = None
    if args.proprio is not None:
        proprio = np.load(args.proprio)
        proprio_summary = {
            "path": str(args.proprio),
            "shape": list(proprio.shape),
            "dtype": str(proprio.dtype),
            "finite": bool(np.isfinite(proprio).all()),
        }

    summary = {
        "trajectory": {
            "path": str(args.trajectory),
            "rows": trajectory.rows,
            "height_min": float(trajectory.height.min()),
            "height_max": float(trajectory.height.max()),
            "mode_unique": np.unique(trajectory.mode).tolist(),
            "hand_unique": np.unique(trajectory.hand_state).tolist(),
        },
        "plots": [str(path_3pt), str(path_controls), *token_plots],
        "token": token_summary,
        "proprio": proprio_summary,
    }
    (args.output_dir / "preview_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-trajectory", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--token", type=Path)
    parser.add_argument("--proprio", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    payload = {"slice": _slice(args), "plot": _plot(args)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

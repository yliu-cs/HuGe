"""Plot loss curves from a three-stage Huge training run."""

from __future__ import annotations

import argparse
import math
import mmap
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np


_PROGRESS_RE = re.compile(
    rb"stage([123]) steps:[^\r\n]*?\|\s*(\d+)/\d+\s*\[[^\r\n]*?loss=([^,\]\s]+)"
)


def _latest_log(run_root: Path, stage: int) -> Path:
    logs = sorted((run_root / f"stage{stage}").glob("*/train.log"))
    if not logs:
        raise FileNotFoundError(f"no stage{stage} train.log found under {run_root}")
    return logs[-1]


def _read_loss(log_path: Path, stage: int) -> tuple[np.ndarray, np.ndarray]:
    points: dict[int, float] = {}
    with log_path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for match in _PROGRESS_RE.finditer(data):
            if int(match.group(1)) != stage:
                continue
            step = int(match.group(2))
            loss = float(match.group(3))
            if not math.isfinite(loss):
                raise ValueError(f"non-finite loss at stage{stage} step {step}: {loss}")
            points[step] = loss  # tqdm writes stale and current postfixes; the last one is current.
    if not points:
        raise ValueError(f"no stage{stage} loss records found in {log_path}")
    steps = np.fromiter(sorted(points), dtype=np.int64)
    losses = np.fromiter((points[step] for step in steps), dtype=np.float64)
    return steps, losses


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(values))
    if window == 1:
        return values
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return (cumulative[window:] - cumulative[:-window]) / window


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="directory containing stage1/, stage2/, and stage3/")
    parser.add_argument("--output", type=Path, default=None, help="output PNG (default: RUN_ROOT/loss.png)")
    parser.add_argument("--smooth", type=int, default=500, help="moving-average window (default: 500)")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be at least 1")

    run_root = args.run_root.expanduser().resolve()
    output = (args.output or run_root / "loss.png").expanduser().resolve()
    curves = []
    for stage in range(1, 4):
        log_path = _latest_log(run_root, stage)
        steps, losses = _read_loss(log_path, stage)
        curves.append((stage, log_path, steps, losses))

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for ax, color, (stage, log_path, steps, losses) in zip(axes, colors, curves, strict=True):
        stride = max(1, math.ceil(len(steps) / 5000))
        ax.plot(steps[::stride], losses[::stride], color=color, alpha=0.18, linewidth=0.6, label="raw")
        smoothed = _moving_average(losses, args.smooth)
        smooth_steps = steps[min(args.smooth, len(steps)) - 1 :]
        ax.plot(smooth_steps, smoothed, color=color, linewidth=1.5, label=f"mean ({min(args.smooth, len(steps))} steps)")
        ax.set_title(f"Stage {stage} — {log_path.parent.name}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
        print(f"stage{stage}: {len(steps)} points, loss {losses[0]:.4f} -> {losses[-1]:.4f}, {log_path}")

    fig.suptitle(f"Training loss — {run_root}", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

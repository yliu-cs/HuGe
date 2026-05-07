"""Plot training loss curves from metrics.jsonl.

Usage:
    .venv/bin/python scripts/plot/plot_loss.py <run_name> [--output-dir figure] [--smooth N]

Example:
    .venv/bin/python scripts/plot/plot_loss.py exp.kappa.b1024.gpus8.2604301611
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(metrics_path: str) -> list[dict]:
    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    # Pad edges to avoid boundary effects
    padded = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def find_metrics_file(run_name: str) -> str:
    # Search under .runs/ for the run_name directory
    runs_root = Path(".runs")
    candidates = list(runs_root.glob(f"*/{run_name}/metrics.jsonl"))
    if not candidates:
        # Try direct path: .runs/kappa/{run_name}/metrics.jsonl
        direct = runs_root / "kappa" / run_name / "metrics.jsonl"
        if direct.exists():
            return str(direct)
        print(f"Error: no metrics.jsonl found for run '{run_name}'", file=sys.stderr)
        print(f"  Searched: {runs_root}/*/{run_name}/metrics.jsonl", file=sys.stderr)
        sys.exit(1)
    return str(candidates[0])


def plot_loss(run_name: str, output_dir: str, smooth_window: int):
    metrics_path = find_metrics_file(run_name)
    print(f"Loading: {metrics_path}")
    records = load_metrics(metrics_path)
    print(f"  {len(records)} steps")

    steps = np.array([r["step"] for r in records])

    # Collect all loss keys (train/loss, eval/loss, etc.)
    loss_keys = sorted(k for k in records[0] if k.endswith("/loss"))

    fig, ax = plt.subplots(figsize=(10, 5))
    for key in loss_keys:
        values = np.array([r[key] for r in records])
        raw_line, = ax.plot(steps, values, alpha=0.2, linewidth=0.5)
        smoothed = smooth(values, smooth_window)
        ax.plot(steps, smoothed, label=key, color=raw_line.get_color(), linewidth=1.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss — {run_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_name}_loss.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from metrics.jsonl")
    parser.add_argument("run_name", help="Run name (directory under .runs/<task>/)")
    parser.add_argument("--output-dir", default="figure", help="Output directory for plots (default: figure)")
    parser.add_argument("--smooth", type=int, default=100, help="Smoothing window size (default: 100)")
    args = parser.parse_args()

    plot_loss(args.run_name, args.output_dir, args.smooth)


if __name__ == "__main__":
    main()

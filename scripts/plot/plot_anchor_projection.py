#!/usr/bin/env python3
"""Plot PCA projections of the anchor-projection feature arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
from scipy.sparse.linalg import svds
from scipy.stats import gaussian_kde


ENCODERS = (
    ("wanvae", "Wan2.2"),
    ("videomae", "VideoMAE v2"),
    ("vjepa2", "V-JEPA 2"),
)
GROUPS = (
    ("human_original", "Ego", "#F16C23", "o"),
    ("processed_final", "Synthesized", "#1B7C3D", "s"),
    ("real_demo", "Real-World G1", "#2B6A99", "v"),
)


def _project(arrays: list[np.ndarray]) -> np.ndarray:
    features = np.concatenate(arrays).astype(np.float64, copy=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features /= np.maximum(norms, np.finfo(features.dtype).eps)
    features -= features.mean(axis=0, keepdims=True)
    left, singular_values, _ = svds(features, k=2, random_state=1222)
    order = np.argsort(singular_values)[::-1]
    return left[:, order] * singular_values[order]


def _plot_kde(ax, points: np.ndarray, grid: tuple[np.ndarray, np.ndarray], color: str, marker: str) -> None:
    grid_x, grid_y = grid
    positions = np.vstack((grid_x.ravel(), grid_y.ravel()))
    density = gaussian_kde(points.T, bw_method=0.4)(positions).reshape(grid_x.shape)
    levels = density.max() * np.linspace(0.02, 1.0, 9)
    fill_colors = [to_rgba(color, alpha) for alpha in np.linspace(0.03, 0.2, len(levels) - 1)]
    ax.contourf(grid_x, grid_y, density, levels=levels, colors=fill_colors, antialiased=True, zorder=1)
    ax.contour(grid_x, grid_y, density, levels=levels, colors=color, linewidths=0.7, alpha=0.7, zorder=2)
    ax.scatter(points[:, 0], points[:, 1], s=7, marker=marker, color=color, alpha=0.25, edgecolors="none", zorder=3)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / ".runs" / "anchor_projection_data",
        help="directory containing the extracted .npy files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / ".runs" / "figure" / "anchor_projection.png",
        help="output PNG; a same-named PDF is also saved",
    )
    args = parser.parse_args()

    plt.rc("font", family="Times New Roman", size=14)
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 4), constrained_layout=True, gridspec_kw={"wspace": 0.20})

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=10,
            label=label,
        )
        for _, label, color, marker in GROUPS
    ]
    for ax, (encoder, title) in zip(axes, ENCODERS, strict=True):
        arrays = [np.load(args.data_root / f"{encoder}_{suffix}.npy") for suffix, *_ in GROUPS]
        if any(array.ndim != 2 for array in arrays) or len({array.shape[1] for array in arrays}) != 1:
            raise ValueError(f"{encoder} arrays must be 2-D and share the same feature dimension")

        projection = _project(arrays)
        padding = np.ptp(projection, axis=0) * 0.08
        lower = projection.min(axis=0) - padding
        upper = projection.max(axis=0) + padding
        grid = np.meshgrid(
            np.linspace(lower[0], upper[0], 160),
            np.linspace(lower[1], upper[1], 160),
        )
        start = 0
        for array, (_, label, color, marker) in zip(arrays, GROUPS, strict=True):
            end = start + len(array)
            _plot_kde(ax, projection[start:end], grid, color, marker)
            start = end

        ax.set_facecolor("#FAFBFC")
        ax.set_xlabel(title, labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.legend(handles=handles, loc="outside upper center", ncol=3, prop={"size": 12})

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}, {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

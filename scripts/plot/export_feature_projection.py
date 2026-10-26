#!/usr/bin/env python3
"""Export real feature projections for the buildless HuGe project website.

This uses the preprocessing in ``plot_anchor_projection.py``: concatenate all
available groups for each encoder, L2-normalize each row, center the pooled
features, then project onto the first two principal components. NumPy's full
SVD computes the same principal subspace as the original script's truncated
SVD; a deterministic loading sign removes arbitrary axis flips.

The output contains every sample, not a synthetic illustration or subsample.
The arrays do not contain sample IDs, so row indices are local to each group
and must not be interpreted as cross-group correspondences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ENCODERS = (
    ("wanvae", "Wan2.2"),
    ("videomae", "VideoMAE V2"),
    ("vjepa2", "V-JEPA 2"),
)
GROUPS = (
    ("human_original", "Raw ego", "#F16C23", "circle"),
    ("processed_final", "Visually adapted", "#1B7C3D", "square"),
    ("real_demo", "Real robot", "#2B6A99", "triangle"),
)
KDE_BANDWIDTH = 0.4


def project(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return pooled PCA scores and explained variance fractions."""
    features = np.concatenate(arrays).astype(np.float64, copy=False)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features /= np.maximum(norms, np.finfo(features.dtype).eps)
    features -= features.mean(axis=0, keepdims=True)
    left, singular_values, loadings = np.linalg.svd(features, full_matrices=False)
    scores = left[:, :2] * singular_values[:2]
    # SVD axes are sign-ambiguous. Orient each toward its largest loading.
    for component in range(2):
        largest = np.argmax(np.abs(loadings[component]))
        if loadings[component, largest] < 0:
            scores[:, component] *= -1
    explained = singular_values[:2] ** 2 / np.sum(singular_values**2)
    return scores, explained


def kde(points: np.ndarray, bounds: np.ndarray, size: int) -> dict:
    """Evaluate the original bw_method=0.4 Gaussian KDE using NumPy only.

    SciPy gaussian_kde uses the unbiased sample covariance multiplied by the
    square of the bandwidth factor. The global normalization cancels when
    dividing each density by its maximum, as done for the original contours.
    """
    covariance = np.cov(points, rowvar=False) * KDE_BANDWIDTH**2
    inverse = np.linalg.inv(covariance)
    grid_x, grid_y = np.meshgrid(
        np.linspace(bounds[0], bounds[1], size),
        np.linspace(bounds[2], bounds[3], size),
    )
    positions = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    density = np.empty(len(positions), dtype=np.float64)
    # Bound temporary memory even if the source datasets grow.
    for start in range(0, len(positions), 256):
        delta = positions[start : start + 256, None, :] - points[None, :, :]
        distance = np.einsum("bni,ij,bnj->bn", delta, inverse, delta)
        density[start : start + 256] = np.exp(-0.5 * distance).mean(axis=1)
    density /= density.max()
    return {
        "size": size,
        "bandwidth": KDE_BANDWIDTH,
        "levels": np.linspace(0.02, 1.0, 9).round(4).tolist(),
        "values": density.round(4).tolist(),
    }


def export_encoder(data_root: Path, encoder: str, label: str, grid_size: int) -> dict:
    arrays, available, missing = [], [], []
    for group, *_ in GROUPS:
        path = data_root / f"{encoder}_{group}.npy"
        if not path.is_file():
            missing.append(group)
            continue
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or len(array) < 3 or not np.isfinite(array).all():
            raise ValueError(f"{path.name}: expected at least 3 finite feature rows")
        arrays.append(array)
        available.append((group, path.name))
    if len(arrays) < 2:
        raise ValueError(f"{encoder}: at least two groups are required")
    if len({array.shape[1] for array in arrays}) != 1:
        raise ValueError(f"{encoder}: feature dimensions must match across groups")

    scores, explained = project(arrays)
    padding = np.ptp(scores, axis=0) * 0.08
    lower, upper = scores.min(axis=0) - padding, scores.max(axis=0) + padding
    bounds = np.array([lower[0], upper[0], lower[1], upper[1]])
    groups, start = [], 0
    for array, (group, source) in zip(arrays, available):
        points = scores[start : start + len(array)]
        groups.append(
            {
                "id": group,
                "source": source,
                "count": len(array),
                "points": points.round(6).tolist(),
                "mean": points.mean(axis=0).round(6).tolist(),
                "covariance": np.cov(points, rowvar=False).round(8).tolist(),
                "density": kde(points, bounds, grid_size),
            }
        )
        start += len(array)
    return {
        "id": encoder,
        "label": label,
        "dimensions": arrays[0].shape[1],
        "count": len(scores),
        "complete": not missing,
        "missingGroups": missing,
        "fitGroups": [group for group, _ in available],
        "bounds": bounds.round(8).tolist(),
        "explainedVariance": explained.round(6).tolist(),
        "groups": groups,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=repo_root / "assets" / "feature-projection-data.js"
    )
    parser.add_argument("--grid-size", type=int, default=80)
    args = parser.parse_args()
    if args.grid_size < 16 or args.grid_size > 160:
        parser.error("--grid-size must be between 16 and 160")
    encoders = [
        export_encoder(args.data_root.expanduser(), encoder, label, args.grid_size)
        for encoder, label in ENCODERS
    ]
    data = {
        "version": 1,
        "method": {
            "projection": "PCA",
            "normalization": "L2 per sample",
            "centering": "Mean of all available groups, separately per encoder",
            "components": 2,
            "solver": "NumPy full SVD, deterministic largest-loading-positive signs",
            "sourceScript": "scripts/plot/plot_anchor_projection.py",
            "sampleIdentity": "Group-local array row; no cross-group sample IDs supplied",
            "density": "Gaussian KDE, bandwidth factor 0.4, per-group peak normalization",
            "grid": "Row-major: y increases from ymin to ymax; x increases from xmin to xmax",
            "densityLevels": "Fractions of each group's peak density; not confidence regions",
        },
        "groups": [
            {"id": group, "label": label, "color": color, "marker": marker}
            for group, label, color, marker in GROUPS
        ],
        "encoders": encoders,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    output.write_text(
        "// Generated by scripts/plot/export_feature_projection.py; real projected samples.\n"
        "window.HUGE_FEATURE_DATA = " + payload + ";\n",
        encoding="utf-8",
    )
    print(f"Exported {sum(encoder['count'] for encoder in encoders)} samples to {output}")
    for encoder in encoders:
        if encoder["missingGroups"]:
            print(f"{encoder['label']}: missing {', '.join(encoder['missingGroups'])}; PCA fits available groups only")


if __name__ == "__main__":
    main()

"""Strict grouped statistics for Huge's 34-D G1 proprioception."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from huge.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT, PROPRIO_LAYOUT_VERSION


def _grouped_stats(
    flat: np.ndarray,
    *,
    show_progress: bool = False,
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    groups = tqdm(
        PROPRIO_LAYOUT,
        desc="Compute stats groups",
        unit="group",
        disable=not show_progress,
    )
    for name, start, end in groups:
        part = flat[:, start:end]
        q01, q99 = np.quantile(part, (0.01, 0.99), axis=0)
        result[name] = {
            "min": part.min(axis=0).astype(float).tolist(),
            "max": part.max(axis=0).astype(float).tolist(),
            "mean": part.mean(axis=0).astype(float).tolist(),
            "std": part.std(axis=0).astype(float).tolist(),
            "q01": q01.astype(float).tolist(),
            "q99": q99.astype(float).tolist(),
        }
    return result


def compute_proprio_stats(
    sources: dict[str, Path],
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Compute one strict stats payload over all configured sources."""
    if not sources:
        raise ValueError("at least one proprio source is required")

    source_paths: dict[str, list[Path]] = {}
    for tag, source_root in sources.items():
        root = Path(source_root)
        paths = sorted(path for path in root.rglob("episode_*.npy") if path.is_file()) if root.is_dir() else []
        if not paths:
            raise ValueError(f"{tag}: no episode_*.npy files under {root}")
        source_paths[tag] = paths

    arrays: list[np.ndarray] = []
    counts: dict[str, dict[str, int]] = {}
    progress = tqdm(
        total=sum(len(paths) for paths in source_paths.values()),
        desc="Load union stats",
        unit="file",
        disable=not show_progress,
    )
    try:
        for tag, paths in source_paths.items():
            frame_count = 0
            for path in paths:
                array = np.load(path, mmap_mode="r")
                if array.ndim != 2 or array.shape[1] != PROPRIO_DIM:
                    raise ValueError(f"{tag}: {path} expected [T,{PROPRIO_DIM}], got {array.shape}")
                if not np.isfinite(array).all():
                    raise ValueError(f"{tag}: nonfinite values in {path}")
                arrays.append(np.asarray(array, dtype=np.float32))
                frame_count += int(array.shape[0])
                progress.update()
            counts[tag] = {"n_episodes": len(paths), "n_frames": frame_count}
    finally:
        progress.close()

    flat = np.concatenate(arrays, axis=0)
    return {
        "unitree_g1_sonic": {
            "state": _grouped_stats(flat, show_progress=show_progress)
        },
        "_meta": {
            "sources": counts,
            "n_frames": int(flat.shape[0]),
            "layout": PROPRIO_LAYOUT_VERSION,
            "g1_proprio_dim": PROPRIO_DIM,
        },
    }


def write_proprio_stats(
    sources: dict[str, Path],
    out: Path,
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Compute and atomically write a stats payload."""
    payload = compute_proprio_stats(sources, show_progress=show_progress)
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload

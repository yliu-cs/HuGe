"""Compute per-group min/max/q01/q99 stats for a Kappa proprietary dataset.

Reads every ``g1_proprio/episode_*.npy`` under ``--dataset-root`` (each a
``(T, 64)`` float32 array in the canonical layout from
``src/data/proprio_layout.py``), splits the 64-D vector into the 13 named
body-part groups defined there, and writes a grouped-schema JSON to
``<dataset-root>/meta/proprio_stats.json``.

The loader (``src/data/tokenized_video_dataset.py:_load_normalize_bounds``)
auto-detects this grouped schema vs the legacy flat schema used by HE/WBT
(``resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json``). The HE+WBT
producer (``scripts/preprocess/calc_g1_proprio_stats.py``) is untouched —
this script is per-dataset and only computes from the dataset you point it
at, so each fine-tune dataset can carry its own distribution.

Per-dim ``q01/q99`` values are bit-identical to a flat 64-D computation on
the same episodes; the grouped schema only changes file structure (enables
future per-group normalization-scheme dispatch, e.g. sin/cos for ``quat``).

Example::

    .venv/bin/python scripts/preprocess/proprietary/calc_proprio_stats.py \\
        --dataset-root resource/dataset/proprietary/real/2026-05-23
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running both as a module (``python -m ...``) and as a script (the
# common case for preprocess scripts). When executed directly the package
# root needs to be on sys.path so ``src.data.proprio_layout`` imports.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.proprio_layout import (  # noqa: E402
    G1_PROPRIO_DIM,
    PROPRIO_GROUPS,
)


def _iter_proprio_files(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(glob.glob(os.path.join(root, "episode_*.npy")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-root", required=True,
        help="Dataset root directory; reads <root>/g1_proprio/episode_*.npy.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path. Default: <dataset-root>/meta/proprio_stats.json.",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    proprio_dir = os.path.join(dataset_root, "g1_proprio")
    files = _iter_proprio_files(proprio_dir)
    if not files:
        raise SystemExit(
            f"No g1_proprio episodes found under {proprio_dir!r}. "
            f"Run preprocess/proprietary/calc_sonic_tokens.py first."
        )

    bufs: list[np.ndarray] = []
    for path in tqdm(files, desc="Loading g1_proprio for stats", unit="ep"):
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception as e:
            print(f"[!] skip {path}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] != G1_PROPRIO_DIM:
            print(
                f"[!] skip {path}: shape {arr.shape}, expected (T, {G1_PROPRIO_DIM})"
            )
            continue
        bufs.append(np.asarray(arr, dtype=np.float32))
    if not bufs:
        raise SystemExit("All proprio files filtered out.")
    flat = np.concatenate(bufs, axis=0)
    n_episodes = len(bufs)
    n_frames = int(flat.shape[0])
    print(f"Stacked {n_episodes} episodes -> {n_frames} rows × {G1_PROPRIO_DIM} dims")

    groups_out: dict[str, dict[str, list[float]]] = {}
    print(f"\n{'group':22s}{'dim':>4s}  {'min':>10s} {'max':>10s} {'q01':>10s} {'q99':>10s}")
    for name, start, end in PROPRIO_GROUPS:
        sub = flat[:, start:end]
        qvals = np.quantile(sub, (0.01, 0.99), axis=0)
        mn = sub.min(axis=0)
        mx = sub.max(axis=0)
        groups_out[name] = {
            "min": mn.astype(np.float64).tolist(),
            "max": mx.astype(np.float64).tolist(),
            "q01": qvals[0].astype(np.float64).tolist(),
            "q99": qvals[1].astype(np.float64).tolist(),
        }
        print(
            f"{name:22s}{end - start:>4d}  "
            f"{float(mn.min()):>10.3f} {float(mx.max()):>10.3f} "
            f"{float(qvals[0].min()):>10.3f} {float(qvals[1].max()):>10.3f}"
        )

    payload = {
        "proprio": groups_out,
        "_meta": {
            "dataset_root": dataset_root,
            "n_episodes": n_episodes,
            "n_frames": n_frames,
            "layout": "13-group v1",
            "g1_proprio_dim": G1_PROPRIO_DIM,
        },
    }

    out_path = args.out or os.path.join(dataset_root, "meta", "proprio_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)
    print(f"\nWrote proprio stats -> {out_path}")


if __name__ == "__main__":
    main()

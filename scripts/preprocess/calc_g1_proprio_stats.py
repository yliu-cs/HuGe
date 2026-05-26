"""Compute per-group min/max + q01/q99 stats for HE + WBT G1 proprio.

Mirrors ``scripts/preprocess/proprietary/calc_proprio_stats.py``: scans every
``g1_proprio/**/episode_*.npy`` under the HE and WBT cache roots, splits the
64-D vector into the 13 named body-part groups defined in
``src.data.proprio_layout``, and writes a grouped-schema JSON.

Two outputs are produced so the dataset-side loader (``HEDataset`` /
``WBTDataset``) can read the same combined stats it has historically used:

  * ``<wbt-cache-root>/meta/g1_proprio_stats.json`` (default), which is what
    ``posttrain_he_wbt_kappa_config`` points at for the combined HE+WBT stats.

Override the output via ``--out`` if you only want HE-only or WBT-only stats.

Example::

    .venv/bin/python scripts/preprocess/calc_g1_proprio_stats.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

# Allow running both as a module and as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.proprio_layout import (  # noqa: E402
    G1_PROPRIO_DIM,
    PROPRIO_GROUPS,
)


def _iter_proprio_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    pattern = os.path.join(root, "**", "episode_*.npy")
    return sorted(glob.glob(pattern, recursive=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--he-cache-root", default="resource/dataset/humanoid-everyday-cache",
    )
    parser.add_argument(
        "--wbt-cache-root", default="resource/dataset/G1_WBT-cache",
    )
    parser.add_argument(
        "--out", default=None,
        help="Defaults to <wbt-cache-root>/meta/g1_proprio_stats.json",
    )
    args = parser.parse_args()

    files: List[str] = []
    for root in (args.he_cache_root, args.wbt_cache_root):
        if not os.path.isdir(root):
            continue
        sub = os.path.join(root, "g1_proprio")
        files.extend(_iter_proprio_files(sub))
    if not files:
        raise SystemExit(
            "No g1_proprio episodes found under either cache root. "
            "Re-run calc_sonic_tokens.py / calc_wbt_sonic_tokens.py first."
        )

    bufs: list[np.ndarray] = []
    for path in tqdm(files, desc="Loading g1_proprio for stats", unit="ep"):
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception as e:
            print(f"[!] skip {path}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] != G1_PROPRIO_DIM:
            print(f"[!] skip {path}: shape {arr.shape}, expected (T, {G1_PROPRIO_DIM})")
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
            "he_cache_root": args.he_cache_root,
            "wbt_cache_root": args.wbt_cache_root,
            "n_episodes": n_episodes,
            "n_frames": n_frames,
            "layout": "13-group v1",
            "g1_proprio_dim": G1_PROPRIO_DIM,
        },
    }

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(args.wbt_cache_root, "meta", "g1_proprio_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)
    print(f"\nWrote G1 proprio stats -> {out_path}")


if __name__ == "__main__":
    main()

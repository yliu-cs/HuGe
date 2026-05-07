"""Compute per-dim min/max + q01/q99 stats for the 64-D G1 proprio vector.

64-D layout: ``[joint_pos(29) | hand_bits(2) | joint_vel(29) | odom_quat_xyzw(4)]``.

The 4-D quaternion tail and the 2-D hand-bit pair both occupy ``[-1, 1]`` /
``[0, 1]`` ranges respectively, so the per-dim ``ill_mask`` in the normalizer
will pass them through cleanly. Velocities have a wider, dataset-dependent
range so q01/q99 trimming will usually help.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import numpy as np
from tqdm import tqdm


G1_PROPRIO_DIM = 64


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
    print(f"Stacked {len(bufs)} episodes -> {flat.shape[0]} rows × {G1_PROPRIO_DIM} dims")

    qvals = np.quantile(flat, (0.01, 0.99), axis=0)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)

    stats = {
        "proprio": {
            "min": mn.astype(np.float64).tolist(),
            "max": mx.astype(np.float64).tolist(),
            "q01": qvals[0].astype(np.float64).tolist(),
            "q99": qvals[1].astype(np.float64).tolist(),
        }
    }

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(args.wbt_cache_root, "meta", "g1_proprio_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp, out_path)
    print(f"Wrote G1 proprio stats -> {out_path}")
    for k in ("min", "max", "q01", "q99"):
        v = np.asarray(stats["proprio"][k])
        print(f"  {k}: shape={v.shape}, min={v.min():.3f}, max={v.max():.3f}")


if __name__ == "__main__":
    main()

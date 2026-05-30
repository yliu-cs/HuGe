"""Compute grouped 34-D proprio stats for proprietary caches."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from kappa.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT, PROPRIO_LAYOUT_VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prop-root", default="resource/dataset/proprietary/real/cola_merged_cleaned_0530")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    files = sorted(glob.glob(str(Path(args.prop_root) / "g1_proprio" / "**" / "episode_*.npy"), recursive=True))
    arrays: list[np.ndarray] = []
    for path in tqdm(files, desc="PROP proprio"):
        arr = np.load(path, mmap_mode="r")
        if arr.ndim == 2 and arr.shape[1] == PROPRIO_DIM:
            arrays.append(np.asarray(arr, dtype=np.float32))
    if not arrays:
        raise SystemExit("no proprietary g1_proprio episodes found")
    flat = np.concatenate(arrays, axis=0)
    stats: dict[str, dict[str, list[float]]] = {}
    for name, start, end in PROPRIO_LAYOUT:
        part = flat[:, start:end]
        q01, q99 = np.quantile(part, (0.01, 0.99), axis=0)
        stats[name] = {
            "min": part.min(axis=0).astype(float).tolist(),
            "max": part.max(axis=0).astype(float).tolist(),
            "mean": part.mean(axis=0).astype(float).tolist(),
            "std": part.std(axis=0).astype(float).tolist(),
            "q01": q01.astype(float).tolist(),
            "q99": q99.astype(float).tolist(),
        }
    out = Path(args.out) if args.out else Path(args.prop_root) / "meta" / "proprio_stats_sonic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps({"unitree_g1_sonic": {"state": stats}, "_meta": {"layout": PROPRIO_LAYOUT_VERSION}}, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

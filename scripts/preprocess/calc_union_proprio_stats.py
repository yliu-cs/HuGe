"""Compute grouped 34-D proprio stats over HE + WBT + proprietary roots."""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from huge.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT, PROPRIO_LAYOUT_VERSION


def _files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", "episode_*.npy"), recursive=True)) if os.path.isdir(root) else []


def _stats(flat: np.ndarray) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    for name, start, end in PROPRIO_LAYOUT:
        part = flat[:, start:end]
        q01, q99 = np.quantile(part, (0.01, 0.99), axis=0)
        out[name] = {
            "min": part.min(axis=0).astype(float).tolist(),
            "max": part.max(axis=0).astype(float).tolist(),
            "mean": part.mean(axis=0).astype(float).tolist(),
            "std": part.std(axis=0).astype(float).tolist(),
            "q01": q01.astype(float).tolist(),
            "q99": q99.astype(float).tolist(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--he-cache-root", default="resource/dataset/humanoid-everyday-cache")
    parser.add_argument("--wbt-source-root", default="resource/dataset/G1_WBT")
    parser.add_argument("--prop-root", default="resource/dataset/proprietary/real/cola_merged_cleaned_0530")
    parser.add_argument("--out", default="resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json")
    args = parser.parse_args()
    roots = {
        "HE": Path(args.he_cache_root) / "g1_proprio",
        "WBT": Path(args.wbt_source_root) / "g1_proprio",
        "PROP": Path(args.prop_root) / "g1_proprio",
    }
    arrays: list[np.ndarray] = []
    counts: dict[str, dict[str, int]] = {}
    for tag, root in roots.items():
        n_ep = 0
        n_fr = 0
        for path in tqdm(_files(str(root)), desc=f"{tag} proprio"):
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != PROPRIO_DIM:
                print(f"skip {path}: expected [T,{PROPRIO_DIM}], got {arr.shape}")
                continue
            arrays.append(np.asarray(arr, dtype=np.float32))
            n_ep += 1
            n_fr += int(arr.shape[0])
        counts[tag] = {"n_episodes": n_ep, "n_frames": n_fr}
    if not arrays:
        raise SystemExit("no proprio episode_*.npy files found")
    flat = np.concatenate(arrays, axis=0)
    payload = {
        "unitree_g1_sonic": {"state": _stats(flat)},
        "_meta": {
            "sources": counts,
            "n_frames": int(flat.shape[0]),
            "layout": PROPRIO_LAYOUT_VERSION,
            "g1_proprio_dim": PROPRIO_DIM,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

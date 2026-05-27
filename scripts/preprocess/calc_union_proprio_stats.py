"""Compute 13-group proprio stats from the UNION of HE + WBT + PROP episodes.

Produces the normalization space for Stage 3 (PROP fine-tune). Union means
``arms`` / ``quat`` envelopes inherit WBT's wide manipulation workspace while
``legs`` / ``waist`` envelopes inherit PROP's squat / torso-rotation range,
so deploy-time poses on the real robot stop landing OOD vs the model's
learned conditioning.

Stage 2 (HE+WBT post-train) still uses ``G1_WBT-cache/meta/g1_proprio_stats.json``;
the union file is written to the PROP dump's ``meta/`` directory because it
exists to serve fine-tune and downstream deploy / sim / eval.

Example::

    .venv/bin/python scripts/preprocess/calc_union_proprio_stats.py
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
        "--he-cache-root",
        default="resource/dataset/humanoid-everyday-cache",
        help="Root containing g1_proprio/**/episode_*.npy for HE.",
    )
    parser.add_argument(
        "--wbt-source-root",
        default="resource/dataset/G1_WBT",
        help=(
            "Root containing g1_proprio/**/episode_*.npy for WBT. Note this "
            "is the source dataset symlink (G1_WBT), NOT the derived cache "
            "(G1_WBT-cache) -- the cache dir only holds stats JSONs."
        ),
    )
    parser.add_argument(
        "--prop-root",
        default="resource/dataset/proprietary/real/2026-05-23",
        help="Root containing g1_proprio/**/episode_*.npy for PROP.",
    )
    parser.add_argument(
        "--out",
        default=(
            "resource/dataset/proprietary/real/2026-05-23/meta/"
            "proprio_stats_union.json"
        ),
        help=(
            "Output JSON path. Defaults to the PROP dump's meta/ since these "
            "stats are owned by Stage 3, not Stage 2."
        ),
    )
    args = parser.parse_args()

    sources = (
        ("HE",   os.path.join(args.he_cache_root,   "g1_proprio")),
        ("WBT",  os.path.join(args.wbt_source_root, "g1_proprio")),
        ("PROP", os.path.join(args.prop_root,       "g1_proprio")),
    )

    bufs: list[np.ndarray] = []
    counts: dict[str, tuple[int, int]] = {}
    for tag, root in sources:
        files = _iter_proprio_files(root)
        n_ep = n_fr = 0
        for path in tqdm(files, desc=f"{tag:4s} g1_proprio", unit="ep"):
            try:
                arr = np.load(path, mmap_mode="r")
            except Exception as e:
                print(f"[!] skip {path}: {e}")
                continue
            if arr.ndim != 2 or arr.shape[1] != G1_PROPRIO_DIM:
                print(f"[!] skip {path}: shape {arr.shape}, expected (T, {G1_PROPRIO_DIM})")
                continue
            bufs.append(np.asarray(arr, dtype=np.float32))
            n_ep += 1
            n_fr += arr.shape[0]
        counts[tag] = (n_ep, n_fr)

    if not bufs:
        raise SystemExit(
            "No proprio episodes found across HE / WBT / PROP roots. "
            "Re-check the --*-root flags."
        )

    flat = np.concatenate(bufs, axis=0)
    total_frames = int(flat.shape[0])

    print(f"\nUnion: {total_frames} rows x {G1_PROPRIO_DIM} dims")
    for tag, (ne, nf) in counts.items():
        print(f"  {tag:5s}: {ne:5d} ep / {nf:9d} fr")

    groups_out: dict[str, dict[str, list[float]]] = {}
    print(f"\n{'group':22s}{'dim':>4s}  {'min':>10s} {'max':>10s} "
          f"{'q01':>10s} {'q99':>10s}")
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
            "sources": {
                tag: {"n_episodes": ne, "n_frames": nf}
                for tag, (ne, nf) in counts.items()
            },
            "total_frames": total_frames,
            "layout": "13-group v1",
            "g1_proprio_dim": G1_PROPRIO_DIM,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = f"{args.out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, args.out)
    print(f"\nWrote union proprio stats -> {args.out}")


if __name__ == "__main__":
    main()

"""Compute sonic proprio stats from the UNION of HE + WBT + PROP episodes.

Produces the normalization space for Stage 3 (PROP fine-tune). Union means
``arms`` / ``projected_gravity`` envelopes inherit WBT's wide manipulation
workspace while ``legs`` / ``waist`` envelopes inherit PROP's squat /
torso-rotation range, so deploy-time poses on the real robot stop landing
OOD vs the model's learned conditioning.

Stage 2 (HE+WBT post-train) still uses
``G1_WBT-cache/meta/g1_proprio_stats_sonic.json``; the union file is written
to the PROP dump's ``meta/`` directory because it exists to serve fine-tune
and downstream deploy / sim / eval.

Schema: nested ``{embodiment_tag: {modality: {joint_group: {stat_key: list}}}}``
matching Isaac-GR00T. Stats emitted: ``min / max / mean / std / q01 / q99``.

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

from src.data.kappa_modality_configs import (  # noqa: E402
    G1_SONIC_PROPRIO_DIM,
    SONIC_STATE_LAYOUT,
)

SONIC_EMBODIMENT_TAG = "unitree_g1_sonic"
STAT_KEYS: tuple[str, ...] = ("min", "max", "mean", "std", "q01", "q99")

HAND_BINARY_SOURCES: dict[str, str] = {
    "HE": "binarize_hands(observation.hand_joints)",
    "WBT": "compute_hand_bits(hand_state, hand_type)",
    "PROP/real/2026-05-23": "observation.hand_binary_state",
    "PROP/real/2026-05-16": "binarize_hands(dex3)",
    "PROP/simple/2026-05-21": "binarize_hands(dex3)",
}


def _iter_proprio_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    pattern = os.path.join(root, "**", "episode_*.npy")
    return sorted(glob.glob(pattern, recursive=True))


def _per_group_stats(flat: np.ndarray) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    print(
        f"\n{'group':22s}{'dim':>4s}  "
        f"{'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s} "
        f"{'q01':>10s} {'q99':>10s}"
    )
    for name, start, end in SONIC_STATE_LAYOUT:
        sub = flat[:, start:end]
        qvals = np.quantile(sub, (0.01, 0.99), axis=0)
        mn = sub.min(axis=0)
        mx = sub.max(axis=0)
        mean = sub.mean(axis=0)
        std = sub.std(axis=0)
        out[name] = {
            "min": mn.astype(np.float64).tolist(),
            "max": mx.astype(np.float64).tolist(),
            "mean": mean.astype(np.float64).tolist(),
            "std": std.astype(np.float64).tolist(),
            "q01": qvals[0].astype(np.float64).tolist(),
            "q99": qvals[1].astype(np.float64).tolist(),
        }
        print(
            f"{name:22s}{end - start:>4d}  "
            f"{float(mn.min()):>10.3f} {float(mx.max()):>10.3f} "
            f"{float(mean.mean()):>10.3f} {float(std.mean()):>10.3f} "
            f"{float(qvals[0].min()):>10.3f} {float(qvals[1].max()):>10.3f}"
        )
    return out


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
            "proprio_stats_union_sonic.json"
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
            if arr.ndim != 2 or arr.shape[1] != G1_SONIC_PROPRIO_DIM:
                print(
                    f"[!] skip {path}: shape {arr.shape}, "
                    f"expected (T, {G1_SONIC_PROPRIO_DIM})"
                )
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

    print(f"\nUnion: {total_frames} rows x {G1_SONIC_PROPRIO_DIM} dims")
    for tag, (ne, nf) in counts.items():
        print(f"  {tag:5s}: {ne:5d} ep / {nf:9d} fr")

    groups_out = _per_group_stats(flat)

    payload = {
        SONIC_EMBODIMENT_TAG: {"state": groups_out},
        "_meta": {
            "sources": {
                tag: {"n_episodes": ne, "n_frames": nf}
                for tag, (ne, nf) in counts.items()
            },
            "total_frames": total_frames,
            "layout": "unitree_g1_sonic v1",
            "g1_proprio_dim": G1_SONIC_PROPRIO_DIM,
            "stats_schema": {"keys": list(STAT_KEYS)},
            "hand_binary_source": HAND_BINARY_SOURCES,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = f"{args.out}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, args.out)
    print(f"\nWrote union sonic proprio stats -> {args.out}")


if __name__ == "__main__":
    main()

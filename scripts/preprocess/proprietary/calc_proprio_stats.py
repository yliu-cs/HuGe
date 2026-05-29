"""[Diagnostic] Per-group sonic proprio stats on a single proprietary dump.

NOTE: fine-tune training (``finetune_prop_ep_kappa_config``) no longer reads
the JSON this script produces by default. The fine-tune stage now inherits
the union HE+WBT+PROP stats
(``resource/dataset/proprietary/real/2026-05-23/meta/proprio_stats_union_sonic.json``)
so the normalization space stays coupled to what Stage 2/3 conditioned on.
This script is kept as a one-off audit / distribution-shift inspector for a
single proprietary set; running it does NOT affect any training run by
default.

Reads every ``g1_proprio/episode_*.npy`` under ``--dataset-root`` (each a
``(T, 34)`` float32 array in the canonical
``src.data.kappa_modality_configs.SONIC_STATE_LAYOUT`` order), splits by
group, and writes a nested-schema JSON to
``<dataset-root>/meta/proprio_stats_sonic.json``.

The schema matches Isaac-GR00T
``{embodiment_tag: {modality: {joint_group: {stat_key: list}}}}`` so a
manual wire-up via ``--data.proprio-stats-path`` is interchangeable with the
union file (same loader, same StateNormalizer).

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

# Allow running both as a module (``python -m ...``) and as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.kappa_modality_configs import (  # noqa: E402
    G1_SONIC_PROPRIO_DIM,
    SONIC_STATE_LAYOUT,
)

SONIC_EMBODIMENT_TAG = "unitree_g1_sonic"
STAT_KEYS: tuple[str, ...] = ("min", "max", "mean", "std", "q01", "q99")

# Per-dump hand bit provenance — written into ``_meta`` so a stats file is
# self-describing about which path produced the 1-D left/right_hand slots.
HAND_BINARY_SOURCES: dict[str, str] = {
    "real/2026-05-23": "observation.hand_binary_state",
    "real/2026-05-16": "binarize_hands(dex3)",
    "simple/2026-05-21": "binarize_hands(dex3)",
}


def _iter_proprio_files(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted(glob.glob(os.path.join(root, "episode_*.npy")))


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


def _guess_hand_binary_source(dataset_root: str) -> str:
    norm = os.path.normpath(dataset_root)
    for key, val in HAND_BINARY_SOURCES.items():
        if key in norm:
            return val
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-root", required=True,
        help="Dataset root directory; reads <root>/g1_proprio/episode_*.npy.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output path. Default: <dataset-root>/meta/proprio_stats_sonic.json.",
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
        if arr.ndim != 2 or arr.shape[1] != G1_SONIC_PROPRIO_DIM:
            print(
                f"[!] skip {path}: shape {arr.shape}, "
                f"expected (T, {G1_SONIC_PROPRIO_DIM})"
            )
            continue
        bufs.append(np.asarray(arr, dtype=np.float32))
    if not bufs:
        raise SystemExit("All proprio files filtered out.")
    flat = np.concatenate(bufs, axis=0)
    n_episodes = len(bufs)
    n_frames = int(flat.shape[0])
    print(
        f"Stacked {n_episodes} episodes -> {n_frames} rows × "
        f"{G1_SONIC_PROPRIO_DIM} dims"
    )

    groups_out = _per_group_stats(flat)

    payload = {
        SONIC_EMBODIMENT_TAG: {"state": groups_out},
        "_meta": {
            "dataset_root": dataset_root,
            "n_episodes": n_episodes,
            "n_frames": n_frames,
            "layout": "unitree_g1_sonic v1",
            "g1_proprio_dim": G1_SONIC_PROPRIO_DIM,
            "stats_schema": {"keys": list(STAT_KEYS)},
            "hand_binary_source": _guess_hand_binary_source(dataset_root),
        },
    }

    out_path = args.out or os.path.join(
        dataset_root, "meta", "proprio_stats_sonic.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)
    print(f"\nWrote sonic proprio stats -> {out_path}")


if __name__ == "__main__":
    main()

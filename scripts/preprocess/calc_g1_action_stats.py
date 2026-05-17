"""Compute per-dim min/max + q01/q99 stats for the 23-D G1 action target.

Reads ``g1_actions/episode_*.npy`` from both Humanoid-Everyday and G1-WBT
caches (already produced by ``calc_sonic_tokens.py`` / ``calc_wbt_sonic_tokens.py``)
and dumps a single shared stats JSON consumed by ``WBTDataset`` / ``HEDataset``
(via ``KappaModelConfig.g1_action_stats_path``) at training time.

Schema follows Psi0's ``ActionStateTransform`` with two overlapping bound
families so the dataset can pick at load time:

  {"action": {"min": [...23], "max": [...23], "q01": [...23], "q99": [...23]}}

23-D layout (see ``src.data.preprocess.build_g1_action_23``):
  [0:14]  arms                     (rad)
  [14:16] hand_bits                ∈ {0, 1}
  [16:19] navigate_cmd(vx, vy, ωz) clipped to (±1, ±1, ±1.5)
  [19:20] base_height              (m, absolute pelvis z)
  [20:23] rpy_cmd(roll, pitch, 0)  (rad)

Hand-bit columns (idx 14, 15) stay in {0, 1}; their min == 0 / max == 1
keeps them outside the ``ill_mask`` threshold so the normalizer maps them
straight to ``2*x - 1 ∈ {-1, +1}``. The yaw column (idx 22) is a constant
zero today — the dataset's ``_load_normalize_bounds`` ill-mask path collapses
constant cols safely (subtract-lo path, no division by zero).

Posttrain uses ``normalize_mode="bounds_q99"`` so use the q01/q99 columns
for the loco_cmd channels rather than min/max (odometry long-tail).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import numpy as np
from tqdm import tqdm


G1_ACTION_DIM = 23


def _iter_action_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    pattern = os.path.join(root, "**", "episode_*.npy")
    return sorted(glob.glob(pattern, recursive=True))


def _streaming_quantile_estimate(
    files: List[str], dim: int, qs: tuple[float, ...], chunk_episodes: int = 256,
) -> np.ndarray:
    """Approximate per-dim ``qs`` quantiles by concatenating reservoir-style.

    Sample size per file is the full episode (50 Hz tokens are cheap; a 60s
    episode is 3000 rows × 31 cols = ~360 KB float32). For 1M total rows
    the concat hits ~120 MB which still fits in RAM on the preprocessor.
    """
    bufs: list[np.ndarray] = []
    for path in tqdm(files, desc="Loading g1_actions for stats", unit="ep"):
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception as e:
            print(f"[!] skip {path}: {e}")
            continue
        if arr.ndim != 2 or arr.shape[1] != dim:
            print(f"[!] skip {path}: shape {arr.shape}, expected (T, {dim})")
            continue
        bufs.append(np.asarray(arr, dtype=np.float32))
    if not bufs:
        raise SystemExit("No g1_actions files found.")
    flat = np.concatenate(bufs, axis=0)
    print(f"Stacked {len(bufs)} episodes -> {flat.shape[0]} rows × {dim} dims")
    qvals = np.quantile(flat, qs, axis=0)
    mn = flat.min(axis=0)
    mx = flat.max(axis=0)
    return mn, mx, qvals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--he-cache-root", default="resource/dataset/humanoid-everyday-cache",
        help="HE cache root (parent of g1_actions/).",
    )
    parser.add_argument(
        "--wbt-cache-root", default="resource/dataset/G1_WBT-cache",
        help="WBT cache root (parent of g1_actions/).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output stats JSON path. Defaults to <wbt-cache-root>/meta/g1_action_stats.json",
    )
    args = parser.parse_args()

    files: List[str] = []
    for root in (args.he_cache_root, args.wbt_cache_root):
        if not os.path.isdir(root):
            continue
        sub = os.path.join(root, "g1_actions")
        files.extend(_iter_action_files(sub))
    if not files:
        raise SystemExit(
            "No g1_actions episodes found under either cache root. "
            "Re-run calc_sonic_tokens.py / calc_wbt_sonic_tokens.py first."
        )

    mn, mx, qvals = _streaming_quantile_estimate(
        files, G1_ACTION_DIM, qs=(0.01, 0.99),
    )
    q01 = qvals[0]
    q99 = qvals[1]

    stats = {
        "action": {
            "min": mn.astype(np.float64).tolist(),
            "max": mx.astype(np.float64).tolist(),
            "q01": q01.astype(np.float64).tolist(),
            "q99": q99.astype(np.float64).tolist(),
        }
    }

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(args.wbt_cache_root, "meta", "g1_action_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(stats, f, indent=2)
    os.replace(tmp, out_path)
    print(f"Wrote G1 action stats -> {out_path}")
    for k in ("min", "max", "q01", "q99"):
        v = np.asarray(stats["action"][k])
        print(f"  {k}: shape={v.shape}, min={v.min():.3f}, max={v.max():.3f}")


if __name__ == "__main__":
    main()

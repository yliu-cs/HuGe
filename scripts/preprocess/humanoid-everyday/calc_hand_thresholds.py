"""Compute global open/close thresholds for left/right hand observation joints.

Scans every episode of humanoid-everyday, gathers `observation.hand_joints[LEFT_JOINT_IDX]`
and `observation.hand_joints[RIGHT_JOINT_IDX]`, and writes two percentile-based thresholds
of |theta| (`low_abs` / `high_abs`) to `<cache_root>/meta/hand_threshold.json`.

The thresholds are consumed by `src/tokenizer/sonic_encoder.binarize_hands`, which uses an
Octo-style three-state rule:
    closed (1) iff |theta| > high_abs
    open   (0) iff |theta| < low_abs
    intermediate frames inherit the next definitive state along the time axis (reverse
    causal scan). This removes label chatter near a single threshold and gives temporally
    coherent labels within an episode.

The selected joints (LEFT=4, RIGHT=11) were chosen empirically: across a 300-episode
sample they had the largest per-episode dynamic range (p90 ≈ 1.65 rad), which is the
clearest signal that the joint actually flexes during grasping vs idle.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import signal

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")


LEFT_JOINT_IDX = 4
RIGHT_JOINT_IDX = 11
SCHEMA_VERSION = 3


def _worker(parquet_path: str):
    import numpy as np
    import polars as pl

    df = pl.read_parquet(parquet_path, columns=["observation.hand_joints"])
    arr = np.asarray(df["observation.hand_joints"].to_list(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= max(LEFT_JOINT_IDX, RIGHT_JOINT_IDX):
        return None
    return arr[:, LEFT_JOINT_IDX].copy(), arr[:, RIGHT_JOINT_IDX].copy()


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def load_robot_type_map(data_root: str) -> dict[int, str]:
    """Read meta/episodes.jsonl and return ``{episode_index: robot_type}``.

    humanoid-everyday mixes G1 (14D hand) and H1 (12D hand) — their joint
    ranges differ, so percentiles taken across the union don't characterize
    either embodiment. The threshold pipeline must be run per-body.
    """
    path = os.path.join(data_root, "meta", "episodes.jsonl")
    out: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            out[int(entry["episode_index"])] = str(entry.get("robot_type", "")).lower()
    return out


def _episode_index_from_path(p: str) -> int:
    return int(os.path.basename(p)[len("episode_"):-len(".parquet")])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute global hand open/close thresholds.")
    ap.add_argument("--data-root", default="resource/dataset/humanoid-everyday")
    ap.add_argument(
        "--cache-root", default=None,
        help="Writable cache dir; defaults to <data-root>-cache if data-root is read-only, "
             "else <data-root>. Threshold is written to <cache-root>/meta/hand_threshold.json.",
    )
    ap.add_argument(
        "--low-pct", type=float, default=30.0,
        help="Percentile of |theta| below which a frame is definitely 'open'. Default 30.",
    )
    ap.add_argument(
        "--high-pct", type=float, default=70.0,
        help="Percentile of |theta| above which a frame is definitely 'closed'. Default 70.",
    )
    ap.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument(
        "--robot-type", default="g1",
        help="Filter episodes by meta/episodes.jsonl:robot_type before sampling joints. "
             "Default 'g1' (G1 hand_joints is 14D vs H1's 12D, and their ranges differ; "
             "mixing them produces percentiles that fit neither body). Pass 'all' to disable.",
    )
    args = ap.parse_args()

    if not (0.0 < args.low_pct < args.high_pct < 100.0):
        raise SystemExit(
            f"Invalid percentiles: low={args.low_pct}, high={args.high_pct}. "
            f"Require 0 < low < high < 100."
        )

    cache_root = args.cache_root
    if cache_root is None:
        cache_root = args.data_root if os.access(args.data_root, os.W_OK) else f"{args.data_root}-cache"

    import numpy as np
    from tqdm import tqdm

    paths = sorted(glob.glob(os.path.join(args.data_root, "data", "chunk-*", "episode_*.parquet")))
    if not paths:
        raise SystemExit(f"No parquet files under {args.data_root}/data")

    discovered = len(paths)
    robot_type_filter = args.robot_type.lower()
    if robot_type_filter != "all":
        rt_map = load_robot_type_map(args.data_root)
        paths = [p for p in paths if rt_map.get(_episode_index_from_path(p)) == robot_type_filter]
        if not paths:
            raise SystemExit(
                f"No episodes left after robot_type='{robot_type_filter}' filter "
                f"(scanned {discovered})."
            )
        print(
            f"Robot-type filter='{robot_type_filter}': kept {len(paths)} / {discovered} episodes "
            f"({discovered - len(paths)} dropped)."
        )

    print(f"Scanning {len(paths)} episodes with {args.num_workers} workers ...")
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []

    with mp.Pool(args.num_workers, initializer=_init_worker) as pool:
        for result in tqdm(pool.imap_unordered(_worker, paths, chunksize=8), total=len(paths)):
            if result is None:
                continue
            l, r = result
            left_chunks.append(l)
            right_chunks.append(r)

    left = np.concatenate(left_chunks)
    right = np.concatenate(right_chunks)
    print(f"Total frames: left={left.shape[0]} right={right.shape[0]}")

    left_abs = np.abs(left)
    right_abs = np.abs(right)
    left_low = float(np.percentile(left_abs, args.low_pct))
    left_high = float(np.percentile(left_abs, args.high_pct))
    right_low = float(np.percentile(right_abs, args.low_pct))
    right_high = float(np.percentile(right_abs, args.high_pct))

    def _stats(name: str, x: np.ndarray, low: float, high: float) -> dict:
        ax = np.abs(x)
        closed_frac = float((ax > high).mean())
        open_frac = float((ax < low).mean())
        intermediate_frac = 1.0 - closed_frac - open_frac
        return {
            "joint_index": name,
            "low_abs": low,
            "high_abs": high,
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "abs_p10": float(np.percentile(ax, 10)),
            "abs_p25": float(np.percentile(ax, 25)),
            "abs_p50": float(np.percentile(ax, 50)),
            "abs_p75": float(np.percentile(ax, 75)),
            "abs_p90": float(np.percentile(ax, 90)),
            "open_fraction": open_frac,
            "intermediate_fraction": intermediate_frac,
            "closed_fraction": closed_frac,
        }

    out = {
        "schema_version": SCHEMA_VERSION,
        "robot_type": robot_type_filter,
        "low_pct": args.low_pct,
        "high_pct": args.high_pct,
        "left": {
            "hand_joints_index": LEFT_JOINT_IDX,
            **_stats(f"hand_joints[{LEFT_JOINT_IDX}]", left, left_low, left_high),
        },
        "right": {
            "hand_joints_index": RIGHT_JOINT_IDX,
            **_stats(f"hand_joints[{RIGHT_JOINT_IDX}]", right, right_low, right_high),
        },
        "rule": (
            "closed=1 if |theta|>high_abs; open=0 if |theta|<low_abs; intermediate frames "
            "are filled with the next definitive state via a reverse scan, then any leading "
            "intermediate run is filled forward from the first definitive state."
        ),
        "n_frames": int(left.shape[0]),
        "n_episodes": len(paths),
    }

    out_path = os.path.join(cache_root, "meta", "hand_threshold.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

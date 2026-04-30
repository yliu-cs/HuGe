"""Compute open/close thresholds for G1-WBT hand_state, separately per hand_type.

The same Octo-style two-threshold + reverse-scan rule used for HE
(scripts/preprocess/humanoid-everyday/calc_hand_thresholds.py) applies, but the
column layout of ``observation.state.hand_state`` differs by hand vendor:

* Inspire (12D): per analysis report §2.4, thumb open/close lives at index 4
  (left) and 10 (right) — that's the dimension with the largest dynamic range.
* BrainCo (12D): same vendor convention, thumb open/close at 0 (left) / 6 (right).
* Dex1 (2D): single gripper scalar per hand at 0 (left) / 1 (right).

We compute per-hand_type ``(low_abs, high_abs)`` thresholds at the same P30/P70
percentiles that HE uses, scanning every ``data/chunk-*/file-*.parquet`` of the
matching subsets. Three JSON files are written under
``<cache_root>/meta/hand_threshold_<hand_type>.json`` with the same
``schema_version=3`` shape as the HE threshold JSON, so
``src.tokenizer.sonic_encoder.load_hand_thresholds`` can read them unchanged.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

# Make sibling import (_subsets.py) work regardless of CWD.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _subsets import HAND_INDEX_TABLE, Subset, iter_data_files, list_subsets  # noqa: E402

SCHEMA_VERSION = 3


def _worker(args: tuple[str, int, int]) -> "tuple[bytes, bytes] | None":
    """Read one parquet shard and return (left_abs, right_abs) bytes for the
    pair of indices configured for this hand_type. Returns ``None`` if the
    parquet is missing the column or has unexpected width.
    """
    parquet_path, left_idx, right_idx = args
    import numpy as np
    import polars as pl

    df = pl.read_parquet(parquet_path, columns=["observation.state.hand_state"])
    arr = np.asarray(df["observation.state.hand_state"].to_list(), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] <= max(left_idx, right_idx):
        return None
    # Bytes round-trip dodges the pickle overhead of returning ndarrays from a
    # multiprocessing.Pool worker — important when scanning ~5k parquet shards.
    return (
        np.abs(arr[:, left_idx]).astype(np.float32).tobytes(),
        np.abs(arr[:, right_idx]).astype(np.float32).tobytes(),
    )


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _stats(name: str, x, low: float, high: float) -> dict:
    import numpy as np

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


def _process_hand_type(
    hand_type: str,
    subsets: list[Subset],
    data_root: str,
    cache_root: str,
    low_pct: float,
    high_pct: float,
    num_workers: int,
) -> str | None:
    """Scan all parquet shards for one hand_type, fit thresholds, write JSON."""
    import numpy as np
    from tqdm import tqdm

    if not subsets:
        return None

    left_idx, right_idx = HAND_INDEX_TABLE[hand_type]

    tasks: list[tuple[str, int, int]] = []
    for sub in subsets:
        for _, _, parquet_path in iter_data_files(sub, data_root):
            tasks.append((parquet_path, left_idx, right_idx))
    if not tasks:
        print(f"[{hand_type}] no parquet shards under any of "
              f"{[s.name for s in subsets]} — skipping.")
        return None

    print(f"[{hand_type}] scanning {len(tasks)} parquet shards across "
          f"{len(subsets)} subsets with {num_workers} workers ...")
    left_chunks: list[np.ndarray] = []
    right_chunks: list[np.ndarray] = []
    with mp.Pool(num_workers, initializer=_init_worker) as pool:
        for result in tqdm(pool.imap_unordered(_worker, tasks, chunksize=2),
                           total=len(tasks), desc=f"hand={hand_type}"):
            if result is None:
                continue
            lb, rb = result
            left_chunks.append(np.frombuffer(lb, dtype=np.float32))
            right_chunks.append(np.frombuffer(rb, dtype=np.float32))

    if not left_chunks:
        print(f"[{hand_type}] no usable rows — skipping.")
        return None

    left = np.concatenate(left_chunks)
    right = np.concatenate(right_chunks)
    left_low = float(np.percentile(left, low_pct))   # already |theta|
    left_high = float(np.percentile(left, high_pct))
    right_low = float(np.percentile(right, low_pct))
    right_high = float(np.percentile(right, high_pct))

    out = {
        "schema_version": SCHEMA_VERSION,
        "hand_type": hand_type,
        "robot_type": "g1",
        "low_pct": low_pct,
        "high_pct": high_pct,
        "left": {
            "hand_state_index": left_idx,
            **_stats(f"hand_state[{left_idx}]", left, left_low, left_high),
        },
        "right": {
            "hand_state_index": right_idx,
            **_stats(f"hand_state[{right_idx}]", right, right_low, right_high),
        },
        "rule": (
            "closed=1 if |x|>high_abs; open=0 if |x|<low_abs; intermediate frames "
            "are filled with the next definitive state via a reverse scan, then any "
            "leading intermediate run is filled forward from the first definitive state."
        ),
        "subsets": [s.name for s in subsets],
        "n_frames": int(left.shape[0]),
    }

    out_path = os.path.join(cache_root, "meta", f"hand_threshold_{hand_type}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{hand_type}] wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="resource/dataset/G1-WBT")
    ap.add_argument(
        "--cache-root", default=None,
        help="Writable cache dir; defaults to <data-root>-cache if data-root is read-only, "
             "else <data-root>. Thresholds are written to "
             "<cache-root>/meta/hand_threshold_<hand_type>.json (one file per hand_type).",
    )
    ap.add_argument("--low-pct", type=float, default=30.0)
    ap.add_argument("--high-pct", type=float, default=70.0)
    ap.add_argument(
        "--hand-types", nargs="*", default=None,
        help="Restrict to a subset of {inspire, brainco, dex1}. Default: all three.",
    )
    ap.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    if not (0.0 < args.low_pct < args.high_pct < 100.0):
        raise SystemExit(
            f"Invalid percentiles: low={args.low_pct}, high={args.high_pct}."
        )

    cache_root = args.cache_root
    if cache_root is None:
        cache_root = (
            args.data_root if os.access(args.data_root, os.W_OK)
            else f"{args.data_root}-cache"
        )

    all_subsets = list_subsets(args.data_root)
    if not all_subsets:
        raise SystemExit(f"No G1-WBT subsets found under {args.data_root}")

    by_type: dict[str, list[Subset]] = {}
    for s in all_subsets:
        by_type.setdefault(s.hand_type, []).append(s)

    requested = args.hand_types or list(HAND_INDEX_TABLE.keys())
    written: list[str] = []
    for ht in requested:
        if ht not in HAND_INDEX_TABLE:
            print(f"[skip] unknown hand_type={ht!r}")
            continue
        path = _process_hand_type(
            ht, by_type.get(ht, []), args.data_root, cache_root,
            args.low_pct, args.high_pct, args.num_workers,
        )
        if path:
            written.append(path)

    print(f"\nDone. {len(written)} threshold file(s) written.")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()

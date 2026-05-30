"""Materialize proprietary LeRobot motion tokens and 34-D G1 proprio caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from kappa.data.preprocess import atomic_save_npy
from kappa.data.schemas import ACTION_DIM, PROPRIO_DIM


def _episodes(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/*.parquet"))


def _feature_names(root: Path, key: str) -> list[str]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    return list(info.get("features", {}).get(key, {}).get("names") or [])


def _detect_layout(root: Path) -> dict[str, int]:
    names = _feature_names(root, "observation.state")
    return {
        "left_arm": names.index("left_shoulder_pitch_joint"),
        "right_arm": names.index("right_shoulder_pitch_joint"),
    }


def _hand_bits(df: pl.DataFrame, n: int, key: str) -> np.ndarray:
    if key in df.columns:
        arr = np.asarray(df[key].to_list(), dtype=np.float32)
        if arr.shape == (n, 2):
            return arr
    return np.zeros((n, 2), dtype=np.float32)


def _write_episode(parquet: Path, out_root: Path, layout: dict[str, int]) -> None:
    cols = ["action.motion_token", "observation.state", "observation.projected_gravity"]
    optional = ["action.hand_binary", "observation.hand_binary_state"]
    df = pl.read_parquet(parquet, columns=[c for c in cols + optional if c in pl.read_parquet_schema(parquet)])
    motion = np.asarray(df["action.motion_token"].to_list(), dtype=np.float32)
    state = np.asarray(df["observation.state"].to_list(), dtype=np.float32)
    gravity = np.asarray(df["observation.projected_gravity"].to_list(), dtype=np.float32)
    n = motion.shape[0]
    if motion.shape != (n, 64):
        raise ValueError(f"{parquet}: action.motion_token must be [T,64], got {motion.shape}")
    action_hand = _hand_bits(df, n, "action.hand_binary")
    state_hand = _hand_bits(df, n, "observation.hand_binary_state")
    tokens = np.concatenate([motion, action_hand], axis=-1).astype(np.float32)
    if tokens.shape[1] != ACTION_DIM:
        raise ValueError(f"{parquet}: action dim {tokens.shape[1]} != {ACTION_DIM}")
    left_arm = layout["left_arm"]
    right_arm = layout["right_arm"]
    proprio = np.concatenate(
        [
            state[:, 0:6],
            state[:, 6:12],
            state[:, 12:15],
            state[:, left_arm:left_arm + 7],
            state[:, right_arm:right_arm + 7],
            state_hand[:, 0:1],
            state_hand[:, 1:2],
            gravity[:, 0:3],
        ],
        axis=-1,
    ).astype(np.float32)
    if proprio.shape[1] != PROPRIO_DIM:
        raise ValueError(f"{parquet}: proprio dim {proprio.shape[1]} != {PROPRIO_DIM}")
    ep = int(parquet.stem.removeprefix("episode_"))
    atomic_save_npy(tokens, str(out_root / "sonic_tokens" / f"episode_{ep:06d}.npy"))
    atomic_save_npy(proprio, str(out_root / "g1_proprio" / f"episode_{ep:06d}.npy"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-roots", nargs="+", default=["resource/dataset/proprietary/real/2026-05-28"])
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()
    for root_str in args.data_roots:
        root = Path(root_str)
        out_root = Path(args.cache_root) if args.cache_root else root
        layout = _detect_layout(root)
        for parquet in tqdm(_episodes(root), desc=str(root)):
            _write_episode(parquet, out_root, layout)
        print(f"wrote proprietary caches under {out_root}")


if __name__ == "__main__":
    main()

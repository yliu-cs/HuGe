#!/usr/bin/env python3
"""Precompute SONIC latent tokens for all episodes and save as .npy files."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.tokenizer.sonic_encoder import SonicEncoder


def load_episodes(data_root: str) -> dict[int, pd.DataFrame]:
    data_dir = os.path.join(data_root, "data")
    episodes: dict[int, list[pd.DataFrame]] = {}
    for chunk_dir in sorted(os.listdir(data_dir)):
        chunk_path = os.path.join(data_dir, chunk_dir)
        if not os.path.isdir(chunk_path):
            continue
        for parquet_file in sorted(os.listdir(chunk_path)):
            if not parquet_file.endswith(".parquet"):
                continue
            df = pd.read_parquet(os.path.join(chunk_path, parquet_file))
            for ep_idx, group in df.groupby("episode_index"):
                episodes.setdefault(int(ep_idx), []).append(group)

    merged = {}
    for ep_idx in sorted(episodes):
        merged[ep_idx] = pd.concat(episodes[ep_idx], ignore_index=True)
    return merged


def encode_episode(encoder: SonicEncoder, df: pd.DataFrame) -> np.ndarray:
    ep_len = len(df)
    jp = np.zeros((ep_len, 29), dtype=np.float64)
    oq = np.zeros((ep_len, 4), dtype=np.float32)
    ig = np.zeros((ep_len, 3), dtype=np.float32)

    for f in range(ep_len):
        row = df.iloc[f]
        leg = np.array(row["observation.leg_joints"], dtype=np.float64)
        arm = np.array(row["observation.arm_joints"], dtype=np.float64)
        jp[f, :len(leg)] = leg
        jp[f, 15:15 + len(arm)] = arm
        oq[f] = np.array(row["observation.odometry.quat"], dtype=np.float32)
        ig[f] = np.array(row["observation.imu.gyroscope"], dtype=np.float32)

    return encoder.encode_episode(jp, oq, ig)


def main():
    parser = argparse.ArgumentParser(description="Precompute SONIC tokens")
    parser.add_argument("--data-root", default="resource/dataset/humanoid-everyday")
    parser.add_argument("--sonic-model-dir", default="resource/model/sonic")
    args = parser.parse_args()

    out_dir = os.path.join(args.data_root, "sonic_tokens")
    os.makedirs(out_dir, exist_ok=True)

    encoder_path = os.path.join(args.sonic_model_dir, "model_encoder.onnx")
    encoder = SonicEncoder(encoder_path)

    print(f"Loading episodes from {args.data_root}/data ...")
    episodes = load_episodes(args.data_root)
    print(f"Found {len(episodes)} episodes")

    skipped = 0
    for ep_idx, df in tqdm(episodes.items(), desc="Encoding episodes"):
        npy_path = os.path.join(out_dir, f"episode_{ep_idx:06d}.npy")
        if os.path.exists(npy_path):
            skipped += 1
            continue
        tokens = encode_episode(encoder, df)
        np.save(npy_path, tokens)

    print(f"Done. {len(episodes) - skipped} encoded, {skipped} skipped (already exist).")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

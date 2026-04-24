"""Precompute SONIC latent tokens for all episodes and save as .npy files."""
from __future__ import annotations

import argparse
import os
import sys

os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

import numpy as np
import polars as pl
from tqdm import tqdm

from src.tokenizer.sonic_encoder import SonicEncoder


def load_episodes(data_root: str) -> dict[int, pl.DataFrame]:
    data_dir = os.path.join(data_root, "data")
    episodes: dict[int, list[pl.DataFrame]] = {}
    for chunk_dir in sorted(os.listdir(data_dir)):
        chunk_path = os.path.join(data_dir, chunk_dir)
        if not os.path.isdir(chunk_path):
            continue
        for parquet_file in sorted(os.listdir(chunk_path)):
            if not parquet_file.endswith(".parquet"):
                continue
            df = pl.read_parquet(os.path.join(chunk_path, parquet_file))
            for ep_idx_val, group in df.group_by("episode_index"):
                episodes.setdefault(int(ep_idx_val[0]), []).append(group)

    merged = {}
    for ep_idx in sorted(episodes):
        merged[ep_idx] = pl.concat(episodes[ep_idx])
    return merged


def encode_episode(encoder: SonicEncoder, df: pl.DataFrame) -> np.ndarray:
    ep_len = len(df)
    jp = np.zeros((ep_len, 29), dtype=np.float64)
    oq = np.zeros((ep_len, 4), dtype=np.float32)
    ig = np.zeros((ep_len, 3), dtype=np.float32)

    leg_col = df["observation.leg_joints"].to_list()
    arm_col = df["observation.arm_joints"].to_list()
    oq_col = df["observation.odometry.quat"].to_list()
    ig_col = df["observation.imu.gyroscope"].to_list()

    for f in range(ep_len):
        leg = np.array(leg_col[f], dtype=np.float64)
        arm = np.array(arm_col[f], dtype=np.float64)
        jp[f, :len(leg)] = leg
        jp[f, 15:15 + len(arm)] = arm
        oq[f] = np.array(oq_col[f], dtype=np.float32)
        ig[f] = np.array(ig_col[f], dtype=np.float32)

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

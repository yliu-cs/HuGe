"""Humanoid-Everyday dataset with precomputed SONIC latent tokens."""
from __future__ import annotations

import os
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class HEDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        action_chunk_size: int = 10,
        split: str = "train",
    ):
        self.data_root = data_root
        self.action_chunk_size = action_chunk_size

        self.sonic_tokens_dir = os.path.join(data_root, "sonic_tokens")
        if not os.path.isdir(self.sonic_tokens_dir):
            raise FileNotFoundError(
                f"Precomputed sonic tokens not found at {self.sonic_tokens_dir}. "
                "Run `python scripts/precompute_sonic_tokens.py` first."
            )

        tasks_path = os.path.join(data_root, "meta", "tasks.jsonl")
        self.task_map = {}
        if os.path.exists(tasks_path):
            with open(tasks_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self.task_map[entry["task_index"]] = entry["description"]

        with open(os.path.join(data_root, "meta", "info.json")) as f:
            self.info = json.load(f)

        self._load_parquet_data()
        self._token_cache: dict[int, np.ndarray] = {}

    def _load_parquet_data(self):
        data_dir = os.path.join(self.data_root, "data")
        dfs = []
        for chunk_dir in sorted(os.listdir(data_dir)):
            chunk_path = os.path.join(data_dir, chunk_dir)
            if not os.path.isdir(chunk_path):
                continue
            for parquet_file in sorted(os.listdir(chunk_path)):
                if not parquet_file.endswith(".parquet"):
                    continue
                df = pd.read_parquet(os.path.join(chunk_path, parquet_file))
                dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)
        ep_col = self.df["episode_index"]
        self.episode_ids = sorted(ep_col.unique().tolist())

        self.ep_slices: dict[int, tuple[int, int]] = {}
        for ep_idx in self.episode_ids:
            mask = ep_col == ep_idx
            indices = self.df.index[mask]
            self.ep_slices[ep_idx] = (int(indices[0]), int(indices[-1]) + 1)

        self.frame_indices = []
        for ep_idx in self.episode_ids:
            start, end = self.ep_slices[ep_idx]
            ep_len = end - start
            for f in range(ep_len):
                self.frame_indices.append((ep_idx, ep_len, f))

    def _get_sonic_tokens(self, ep_idx: int) -> np.ndarray:
        if ep_idx in self._token_cache:
            return self._token_cache[ep_idx]
        npy_path = os.path.join(self.sonic_tokens_dir, f"episode_{ep_idx:06d}.npy")
        tokens = np.load(npy_path)
        self._token_cache[ep_idx] = tokens
        return tokens

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, ep_len, frame_id = self.frame_indices[idx]
        ep_start = self.ep_slices[ep_idx][0]
        row = self.df.iloc[ep_start + frame_id]

        task_idx = int(row.get("task_index", 0))
        text = self.task_map.get(task_idx, "perform the task")

        video_path = os.path.join(
            self.data_root,
            f"videos/chunk-{ep_idx // 1000:03d}/egocentric/episode_{ep_idx:06d}.mp4",
        )
        pil_image = self._decode_frame(video_path, frame_id)

        sonic_all = self._get_sonic_tokens(ep_idx)
        chunk_indices = [min(frame_id + t, ep_len - 1) for t in range(self.action_chunk_size)]
        sonic_tokens = sonic_all[chunk_indices]
        sonic_mask = np.array(
            [1.0 if frame_id + t < ep_len else 0.0 for t in range(self.action_chunk_size)],
            dtype=np.float32,
        )

        return {
            "image": pil_image,
            "text": text,
            "sonic_tokens": torch.from_numpy(sonic_tokens.copy()),
            "sonic_mask": torch.from_numpy(sonic_mask),
        }

    @staticmethod
    def _decode_frame(video_path: str, frame_idx: int) -> Image.Image:
        import cv2
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return Image.new("RGB", (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

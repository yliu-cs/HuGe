"""Humanoid-Everyday dataset with precomputed SONIC latent tokens."""
from __future__ import annotations

import os
import json
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from PIL import Image


class HEDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        action_chunk_size: int = 10,
        split: str = "train",
        cache_root: str | None = None,
    ):
        self.data_root = data_root
        self.action_chunk_size = action_chunk_size
        # cache_root holds outputs that the read-only HF dataset mount can't host
        # (sonic tokens, dataset index, hand threshold). Auto-fall back to a parallel
        # "<data_root>-cache" dir when data_root itself isn't writable.
        if cache_root is None:
            cache_root = data_root if os.access(data_root, os.W_OK) else f"{data_root}-cache"
        self.cache_root = cache_root

        self.sonic_tokens_dir = os.path.join(self.cache_root, "sonic_tokens")
        if not os.path.isdir(self.sonic_tokens_dir):
            raise FileNotFoundError(
                f"Precomputed sonic tokens not found at {self.sonic_tokens_dir}. "
                "Run `python scripts/calc_sonic_tokens.py` first."
            )

        tasks_path = os.path.join(data_root, "meta", "tasks.jsonl")
        self.task_map: dict[int, str] = {}
        if os.path.exists(tasks_path):
            with open(tasks_path) as f:
                for line in f:
                    entry = json.loads(line)
                    self.task_map[entry["task_index"]] = entry["description"]

        with open(os.path.join(data_root, "meta", "info.json")) as f:
            self.info = json.load(f)

        self._load_data()
        self._token_cache: dict[int, np.ndarray] = {}

    def _collect_parquet_paths(self) -> list[str]:
        data_dir = os.path.join(self.data_root, "data")
        paths = []
        for chunk_dir in sorted(os.listdir(data_dir)):
            chunk_path = os.path.join(data_dir, chunk_dir)
            if not os.path.isdir(chunk_path):
                continue
            for fname in sorted(os.listdir(chunk_path)):
                if fname.endswith(".parquet"):
                    paths.append(os.path.join(chunk_path, fname))
        return paths

    def _load_data(self):
        cache_path = os.path.join(self.cache_root, "meta", "dataset_index.json")

        if os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            self._build_and_cache(cache_path)

    def _load_from_cache(self, cache_path: str):
        with open(cache_path) as f:
            cache = json.load(f)
        self.episode_ids = cache["episode_ids"]
        self.ep_slices = {int(k): tuple(v) for k, v in cache["ep_slices"].items()}
        self.frame_indices = [tuple(x) for x in cache["frame_indices"]]
        self._task_ids = cache["task_ids"]

    def _build_and_cache(self, cache_path: str):
        parquet_paths = self._collect_parquet_paths()

        # Read all parquets with polars (parallel internally, much faster than pandas)
        dfs = [pl.read_parquet(p, columns=["episode_index", "task_index"]) for p in parquet_paths]
        df = pl.concat(dfs)
        del dfs

        # Build episode slices via groupby — handles non-contiguous episodes safely
        self.episode_ids: list[int] = []
        self.ep_slices: dict[int, tuple[int, int]] = {}
        task_per_ep: dict[int, int] = {}
        offset = 0
        for ep_idx_val, group in df.group_by("episode_index", maintain_order=True):
            ep_idx = int(ep_idx_val[0])
            rows = group.height
            self.episode_ids.append(ep_idx)
            self.ep_slices[ep_idx] = (offset, offset + rows)
            task_per_ep[ep_idx] = int(group["task_index"][0])
            offset += rows

        # Build flat frame index and per-frame task IDs
        self.frame_indices: list[tuple[int, int, int]] = []
        self._task_ids: list[int] = []
        for ep_idx in self.episode_ids:
            start, end = self.ep_slices[ep_idx]
            ep_len = end - start
            task_id = task_per_ep[ep_idx]
            for f in range(ep_len):
                self.frame_indices.append((ep_idx, ep_len, f))
                self._task_ids.append(task_id)

        # Persist cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "episode_ids": self.episode_ids,
                    "ep_slices": {str(k): list(v) for k, v in self.ep_slices.items()},
                    "frame_indices": self.frame_indices,
                    "task_ids": self._task_ids,
                },
                f,
            )

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

        task_idx = self._task_ids[idx]
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

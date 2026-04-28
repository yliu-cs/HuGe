"""Humanoid-Everyday dataset with precomputed SONIC latent tokens."""
from __future__ import annotations

import os
import json
from collections import OrderedDict
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
from PIL import Image


# Per-worker LRU cache of mmap'd ``token_state.npy`` views. Keyed by file
# path so each forked DataLoader worker gets its own dict. ~32 episodes
# in flight is enough to amortize the mmap setup cost on shuffled training
# without holding the page cache hostage.
_SONIC_MMAP_CACHE_SIZE = 64
_SONIC_MMAP_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _get_sonic_mmap(path: str) -> np.ndarray:
    arr = _SONIC_MMAP_CACHE.get(path)
    if arr is not None:
        _SONIC_MMAP_CACHE.move_to_end(path)
        return arr
    arr = np.load(path, mmap_mode="r")
    _SONIC_MMAP_CACHE[path] = arr
    if len(_SONIC_MMAP_CACHE) > _SONIC_MMAP_CACHE_SIZE:
        _SONIC_MMAP_CACHE.popitem(last=False)
    return arr


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
        # Index is stored as packed ``int32`` ndarrays in an .npz: at ~2e8
        # frames the prior list-of-tuples layout was ~5 GB Python objects
        # and slow to fork into workers. Four parallel arrays keep the same
        # info under ~3 GB and give O(1) integer access in __getitem__.
        cache_path = os.path.join(self.cache_root, "meta", "dataset_index.npz")

        if os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            self._build_and_cache(cache_path)

    def _load_from_cache(self, cache_path: str):
        with np.load(cache_path) as cache:
            self._frame_ep = cache["frame_ep"].astype(np.int32, copy=False)
            self._frame_ep_len = cache["frame_ep_len"].astype(np.int32, copy=False)
            self._frame_local = cache["frame_local"].astype(np.int32, copy=False)
            self._frame_task = cache["frame_task"].astype(np.int32, copy=False)

    def _build_and_cache(self, cache_path: str):
        parquet_paths = self._collect_parquet_paths()

        # Read all parquets with polars (parallel internally, much faster than pandas)
        dfs = [pl.read_parquet(p, columns=["episode_index", "task_index"]) for p in parquet_paths]
        df = pl.concat(dfs)
        del dfs

        # Build per-episode (length, task) via groupby — handles non-contiguous
        # episodes safely. We discard the global row offset (no longer used).
        ep_ids: list[int] = []
        ep_lens: list[int] = []
        ep_tasks: list[int] = []
        for ep_idx_val, group in df.group_by("episode_index", maintain_order=True):
            ep_ids.append(int(ep_idx_val[0]))
            ep_lens.append(group.height)
            ep_tasks.append(int(group["task_index"][0]))

        total = sum(ep_lens)
        frame_ep = np.empty(total, dtype=np.int32)
        frame_ep_len = np.empty(total, dtype=np.int32)
        frame_local = np.empty(total, dtype=np.int32)
        frame_task = np.empty(total, dtype=np.int32)

        cur = 0
        for ep_id, ep_len, task in zip(ep_ids, ep_lens, ep_tasks):
            sl = slice(cur, cur + ep_len)
            frame_ep[sl] = ep_id
            frame_ep_len[sl] = ep_len
            frame_local[sl] = np.arange(ep_len, dtype=np.int32)
            frame_task[sl] = task
            cur += ep_len

        self._frame_ep = frame_ep
        self._frame_ep_len = frame_ep_len
        self._frame_local = frame_local
        self._frame_task = frame_task

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        np.savez(
            tmp_path,
            frame_ep=frame_ep,
            frame_ep_len=frame_ep_len,
            frame_local=frame_local,
            frame_task=frame_task,
        )
        # ``np.savez`` writes ``<tmp_path>.npz`` when given a path without
        # the suffix; rename it onto the canonical location atomically so
        # racing ranks don't see a half-written index.
        written = tmp_path if os.path.exists(tmp_path) else f"{tmp_path}.npz"
        os.replace(written, cache_path)

    def _get_sonic_tokens(self, ep_idx: int) -> np.ndarray:
        # mmap so the OS page cache is shared across workers/processes; LRU
        # cache the mmap object itself so consecutive samples from the same
        # episode skip the mmap setup syscall.
        npy_path = os.path.join(self.sonic_tokens_dir, f"episode_{ep_idx:06d}.npy")
        return _get_sonic_mmap(npy_path)

    def __len__(self) -> int:
        return int(self._frame_ep.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx = int(self._frame_ep[idx])
        ep_len = int(self._frame_ep_len[idx])
        frame_id = int(self._frame_local[idx])
        task_idx = int(self._frame_task[idx])
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
        from src.data.video_reader import decode_frame

        frame = decode_frame(video_path, frame_idx, target_size=(640, 480))
        return Image.fromarray(frame)

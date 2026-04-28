"""EgoDex dataset (precomputed SONIC tokens, no hand-binary bits).

Source layout::

    <data_root>/episodes/<part>/<task_name>/<ep_id>/
        episode.mp4              # egocentric RGB, 30 fps
        llm_description.txt      # one-line task description
        token_state.npy          # float32 [T, 64], SONIC latents at 50 Hz

The Kappa model uses ``sonic_dim=66`` (64 SONIC + 2 hand-binary bits). EgoDex
tokens are right-padded with two zero columns and the loss is suppressed on
those two padded dimensions via a per-dimension ``sonic_mask``.

Indexing notes:

* Sampling is in 50 Hz token space; the video frame at the same time is
  ``floor(token_idx * num_frames / num_token_rows)`` and clamped at decode time.
* With ~3.35e5 episodes a flat per-frame index would explode, so we keep
  per-episode cumulative offsets and resolve ``__getitem__`` via ``bisect``.
* The first scan is parallelized with threads (``np.load(mmap_mode="r").shape``
  is cheap) and cached as JSON to ``<cache_root>/meta/egodex_index.json``.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_RAW_TOKEN_DIM = 64  # EgoDex token_state width (no hand bits)
# Decoded frames are produced at this (W, H). decord resizes during decode so
# we never materialize the native 1920×1080 frame in worker memory; 256 is the
# smallest size that still leaves headroom for Resize(224)+CenterCrop(224)
# downstream without a second resize step.
_FRAME_SIZE = (256, 256)

# Per-worker LRU cache of mmap'd ``token_state.npy`` views. Avoids re-running
# the np.load mmap path on every __getitem__ when consecutive samples land in
# the same episode (likely under shuffle when episodes are long).
_TOKEN_MMAP_CACHE_SIZE = 64
_TOKEN_MMAP_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _get_token_mmap(path: str) -> np.ndarray:
    arr = _TOKEN_MMAP_CACHE.get(path)
    if arr is not None:
        _TOKEN_MMAP_CACHE.move_to_end(path)
        return arr
    arr = np.load(path, mmap_mode="r")
    _TOKEN_MMAP_CACHE[path] = arr
    if len(_TOKEN_MMAP_CACHE) > _TOKEN_MMAP_CACHE_SIZE:
        _TOKEN_MMAP_CACHE.popitem(last=False)
    return arr


class EgoDexDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        action_chunk_size: int = 10,
        split: str = "train",
        cache_root: str | None = None,
        sonic_dim: int = 66,
        max_episodes: int | None = None,
        index_workers: int = 32,
    ):
        if split != "train":
            raise ValueError(
                "EgoDex's repacked layout only ships a single split; pass split='train'."
            )
        if sonic_dim < _RAW_TOKEN_DIM:
            raise ValueError(
                f"sonic_dim={sonic_dim} smaller than raw EgoDex token width {_RAW_TOKEN_DIM}"
            )

        self.data_root = data_root
        self.action_chunk_size = action_chunk_size
        self.sonic_dim = sonic_dim
        self.episodes_root = os.path.join(data_root, "episodes")

        if cache_root is None:
            cache_root = data_root if os.access(data_root, os.W_OK) else f"{data_root}-cache"
        self.cache_root = cache_root

        self._load_index(index_workers=index_workers)
        if max_episodes is not None:
            self._truncate(max_episodes)

    # -- index ------------------------------------------------------------------

    def _index_path(self) -> str:
        return os.path.join(self.cache_root, "meta", "egodex_index.json")

    def _load_index(self, index_workers: int) -> None:
        idx_path = self._index_path()
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
        else:
            idx = self._build_index(index_workers)
            os.makedirs(os.path.dirname(idx_path), exist_ok=True)
            # Atomic write so racing ranks never observe a half-written file.
            tmp_path = f"{idx_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(idx, f)
            os.replace(tmp_path, idx_path)

        # Index columns live as numpy arrays so DataLoader workers can read
        # them through fork without breaking copy-on-write on every access:
        # Python list/int access bumps refcount and dirties the page, which
        # multiplies the index across all workers. Contiguous numpy buffers
        # have no Python-object refcounts on element access.
        self._descriptions: list[str] = idx["descriptions"]
        self._ep_paths: np.ndarray = np.asarray(idx["paths"], dtype=np.bytes_)
        self._ep_token_rows: np.ndarray = np.asarray(idx["token_rows"], dtype=np.int64)
        self._ep_desc_id: np.ndarray = np.asarray(idx["desc_ids"], dtype=np.int32)
        self._recompute_offsets()

    def _recompute_offsets(self) -> None:
        n = self._ep_token_rows
        if n.size == 0:
            self._ep_offsets = np.zeros(0, dtype=np.int64)
            self._total_rows = 0
            return
        self._ep_offsets = np.empty(n.size, dtype=np.int64)
        self._ep_offsets[0] = 0
        if n.size > 1:
            np.cumsum(n[:-1], out=self._ep_offsets[1:])
        self._total_rows = int(n.sum())

    def _build_index(self, index_workers: int) -> dict[str, list]:
        from tqdm import tqdm

        ep_keys: list[tuple[str, str, str]] = []
        for part in sorted(os.listdir(self.episodes_root)):
            pdir = os.path.join(self.episodes_root, part)
            if not os.path.isdir(pdir):
                continue
            for task in sorted(os.listdir(pdir)):
                tdir = os.path.join(pdir, task)
                if not os.path.isdir(tdir):
                    continue
                for ep in sorted(
                    os.listdir(tdir),
                    key=lambda x: (int(x) if x.isdigit() else 1 << 31, x),
                ):
                    ep_keys.append((part, task, ep))

        def probe(key: tuple[str, str, str]):
            part, task, ep = key
            edir = os.path.join(self.episodes_root, part, task, ep)
            tok_path = os.path.join(edir, "token_state.npy")
            txt_path = os.path.join(edir, "llm_description.txt")
            mp4_path = os.path.join(edir, "episode.mp4")
            try:
                shape = np.load(tok_path, mmap_mode="r").shape
                if shape[0] < 1 or shape[1] != _RAW_TOKEN_DIM:
                    return None
                if not os.path.exists(mp4_path):
                    return None
                with open(txt_path) as f:
                    # Preserve original case to match HE descriptions and the
                    # casing distribution Qwen3-VL was instruction-tuned on.
                    desc = f.read().strip()
            except Exception:
                return None
            return key, int(shape[0]), desc

        results: list[tuple[tuple[str, str, str], int, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, index_workers)) as ex:
            for r in tqdm(
                ex.map(probe, ep_keys),
                total=len(ep_keys),
                desc="Indexing EgoDex",
                unit="ep",
            ):
                if r is not None:
                    results.append(r)

        # Stable order: sort by (part, task, int(ep))
        def _sort_key(item):
            (part, task, ep), _, _ = item
            return (part, task, int(ep) if ep.isdigit() else 1 << 31, ep)

        results.sort(key=_sort_key)

        descs: list[str] = []
        desc_to_id: dict[str, int] = {}
        paths: list[str] = []
        token_rows: list[int] = []
        desc_ids: list[int] = []
        for (part, task, ep), n_tok, desc in results:
            if desc not in desc_to_id:
                desc_to_id[desc] = len(descs)
                descs.append(desc)
            paths.append(f"episodes/{part}/{task}/{ep}")
            token_rows.append(n_tok)
            desc_ids.append(desc_to_id[desc])

        return {
            "descriptions": descs,
            "paths": paths,
            "token_rows": token_rows,
            "desc_ids": desc_ids,
        }

    def _truncate(self, n: int) -> None:
        # numpy slicing returns views; copy() makes them owning so the
        # truncated dataset doesn't keep the full index alive.
        self._ep_paths = self._ep_paths[:n].copy()
        self._ep_token_rows = self._ep_token_rows[:n].copy()
        self._ep_desc_id = self._ep_desc_id[:n].copy()
        self._recompute_offsets()

    # -- sampling ---------------------------------------------------------------

    def __len__(self) -> int:
        return self._total_rows

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._total_rows:
            raise IndexError(idx)
        ep = int(np.searchsorted(self._ep_offsets, idx, side="right")) - 1
        return ep, idx - int(self._ep_offsets[ep])

    def episode_spans(self) -> np.ndarray:
        """Return ``[N, 2]`` int64 array of ``(start_idx, length)`` per episode.

        Consumed by ``BlockShuffleSampler`` to chop the index space into
        episode-aligned contiguous runs. Computed from the cached offsets so
        this is O(N) and never re-reads the index file.
        """
        out = np.empty((self._ep_offsets.size, 2), dtype=np.int64)
        out[:, 0] = self._ep_offsets
        out[:, 1] = self._ep_token_rows.astype(np.int64, copy=False)
        return out

    def _get_tokens(self, ep: int) -> np.ndarray:
        # mmap so 33 万 episode 的 token 文件按需分页读入，OS page cache 在
        # workers 之间共享；mmap 句柄进程内 LRU 缓存避免重复 syscall。
        ep_path = self._ep_paths[ep].decode()
        path = os.path.join(self.data_root, ep_path, "token_state.npy")
        return _get_token_mmap(path)

    @staticmethod
    def _decode_frame(video_path: str, frame_idx: int) -> Image.Image:
        from src.data.video_reader import decode_frame

        frame = decode_frame(video_path, frame_idx, target_size=_FRAME_SIZE)
        return Image.fromarray(frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep, t = self._resolve(idx)
        ep_path = self._ep_paths[ep].decode()
        n_tok = int(self._ep_token_rows[ep])
        text = self._descriptions[int(self._ep_desc_id[ep])]

        # Token chunk: clamp tail by repeating last valid row.
        tokens = self._get_tokens(ep)
        chunk_indices = [min(t + k, n_tok - 1) for k in range(self.action_chunk_size)]
        chunk = tokens[chunk_indices]  # [chunk, 64]

        pad = self.sonic_dim - chunk.shape[-1]
        if pad > 0:
            chunk = np.concatenate(
                [chunk, np.zeros((self.action_chunk_size, pad), dtype=chunk.dtype)],
                axis=-1,
            )

        # Per-time × per-dim mask: time-step valid only when t+k < n_tok,
        # and the trailing `pad` (hand-bit) dims always masked out.
        time_valid = np.array(
            [1.0 if t + k < n_tok else 0.0 for k in range(self.action_chunk_size)],
            dtype=np.float32,
        )
        dim_valid = np.zeros((self.sonic_dim,), dtype=np.float32)
        dim_valid[:_RAW_TOKEN_DIM] = 1.0
        sonic_mask = time_valid[:, None] * dim_valid[None, :]  # [chunk, sonic_dim]

        # Aligned video frame: tokens are 50 Hz, video is 30 Hz.
        # Use linear scaling on the actual token row count rather than a hard
        # 5/3 ratio so we stay correct for any sample we encounter.
        video_path = os.path.join(self.data_root, ep_path, "episode.mp4")
        # Frame index is derived from t/n_tok; per-episode actual video length
        # is enforced inside _decode_frame, avoiding a second probe at index time.
        # We approximate using the 30/50 ratio; the decoder clamps to range.
        frame_idx = int(t * 30 // 50)
        pil_image = self._decode_frame(video_path, frame_idx)

        return {
            "image": pil_image,
            "text": text,
            "sonic_tokens": torch.from_numpy(chunk.copy()),
            "sonic_mask": torch.from_numpy(sonic_mask),
        }

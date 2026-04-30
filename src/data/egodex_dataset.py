"""EgoDex dataset (precomputed SONIC tokens, no hand-binary bits).

Source layout::

    <data_root>/episodes/<part>/<task_name>/<ep_id>/
        episode.mp4              # egocentric RGB, 30 fps
        llm_description.txt      # one-line task description
        token_state.npy          # float32 [T, 64], SONIC latents at 50 Hz

The Kappa model uses ``sonic_dim=66`` (64 SONIC + 2 hand-binary bits). EgoDex
tokens are right-padded with two zero columns and the loss is suppressed on
those two padded dimensions via a per-dim ``sonic_mask``.

Indexing: with ~3.35e5 episodes the first scan is parallelized with threads
(``np.load(mmap_mode='r').shape`` is cheap) and cached as JSON to
``<cache_root>/meta/egodex_index.json``. The base class' bisect-on-cumulative-
offsets resolution avoids materializing a per-frame array (would be ~7 GB).
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.data.tokenized_video_dataset import (
    TokenizedVideoDataset,
    get_token_mmap,
    resolve_writable_cache_root,
)


_RAW_TOKEN_DIM = 64  # EgoDex token_state width (no hand bits)


class EgoDexDataset(TokenizedVideoDataset):
    def __init__(
        self,
        data_root: str,
        action_chunk_size: int = 50,
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
        super().__init__(action_chunk_size=action_chunk_size)

        self.data_root = data_root
        self.sonic_dim = sonic_dim
        self.episodes_root = os.path.join(data_root, "episodes")
        self.cache_root = resolve_writable_cache_root(data_root, cache_root)

        self._load_index(index_workers=index_workers)
        if max_episodes is not None:
            self._truncate(max_episodes)
        self._finalize_index()

    # -- index ---------------------------------------------------------------

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
            tmp_path = f"{idx_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(idx, f)
            os.replace(tmp_path, idx_path)

        # Bytes/int arrays so DataLoader fork() doesn't bump Python refcounts
        # on every access (which would dirty pages and multiply RSS).
        self._descriptions: list[str] = idx["descriptions"]
        self._ep_paths: np.ndarray = np.asarray(idx["paths"], dtype=np.bytes_)
        self._ep_lengths: np.ndarray = np.asarray(idx["token_rows"], dtype=np.int64)
        self._ep_desc_id: np.ndarray = np.asarray(idx["desc_ids"], dtype=np.int32)

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
        self._ep_lengths = self._ep_lengths[:n].copy()
        self._ep_desc_id = self._ep_desc_id[:n].copy()

    # -- hooks ---------------------------------------------------------------

    def _video_for(self, ep: int, t: int) -> tuple[str, int]:
        ep_path = self._ep_paths[ep].decode()
        # Tokens are 50 Hz, video is 30 Hz. Decoder clamps to actual range.
        frame_idx = int(t * 30 // 50)
        return os.path.join(self.data_root, ep_path, "episode.mp4"), frame_idx

    def _text(self, ep: int) -> str:
        return self._descriptions[int(self._ep_desc_id[ep])]

    def _tokens(self, ep: int) -> np.ndarray:
        ep_path = self._ep_paths[ep].decode()
        return get_token_mmap(os.path.join(self.data_root, ep_path, "token_state.npy"))

    def _build_chunk(
        self, ep: int, t: int, tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Override: 64 → ``sonic_dim`` right-pad + per-dim mask zeroing the
        trailing hand-bit columns so the loss ignores them."""
        n = int(self._ep_lengths[ep])
        chunk_indices = [min(t + k, n - 1) for k in range(self.action_chunk_size)]
        chunk = tokens[chunk_indices]  # [chunk, 64]
        pad = self.sonic_dim - chunk.shape[-1]
        if pad > 0:
            chunk = np.concatenate(
                [chunk, np.zeros((self.action_chunk_size, pad), dtype=chunk.dtype)],
                axis=-1,
            )
        time_valid = np.array(
            [1.0 if t + k < n else 0.0 for k in range(self.action_chunk_size)],
            dtype=np.float32,
        )
        dim_valid = np.zeros((self.sonic_dim,), dtype=np.float32)
        dim_valid[:_RAW_TOKEN_DIM] = 1.0
        sonic_mask = time_valid[:, None] * dim_valid[None, :]  # [chunk, sonic_dim]
        return chunk, sonic_mask

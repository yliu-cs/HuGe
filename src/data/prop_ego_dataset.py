"""Proprietary first-person ("ego") dataset for EgoDex-style fine-tuning.

Source layout::

    <data_root>/<task_name>/<ep_id>/
        CameraRecord_<TS>.mp4            # 30 fps stereo side-by-side (unused)
        CameraRecord_<TS>_left.mp4       # 30 fps mono, trimmed to tracking
        CameraRecord_<TS>_right.mp4      # 30 fps mono, trimmed to tracking
        llm_description.txt              # one-line instruction
        trackingData_<TS>.txt            # raw VR/AR head/hand/body tracking
        trackingData_<TS>_latent_tokens_66d.npy  # float32 [T, 66], 50 Hz
        trackingData_<TS>_sonic_v3.npz   # joint_pos/vel/quat/hand state (unused)

This is *human* first-person capture (no robot proprio), so the dataset is
trained the same way the stage-1 EgoDex pre-train consumes EgoDex: image +
SONIC tokens + text only, no proprio, no RTC. The latent token file is
nominally 66-D (64 SONIC + 2 hand-binary bits) but the hand bits come from
an offline thresholded hand-close signal whose distribution does not match
the post-train hand-binary contract; following the EgoDex recipe we zero
the trailing 2 columns and mask them via the per-dim ``sonic_mask`` so the
loss ignores them.

The ``_left.mp4`` mono crop is already trimmed to the tracking window so its
frame range aligns with the token timeline (token 50 Hz → frame 30 fps via
``frame_idx = t * 30 // 50``, identical to EgoDex).
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from src.data.tokenized_video_dataset import (
    TokenizedVideoDataset,
    get_token_mmap,
    resolve_writable_cache_root,
)


_RAW_TOKEN_DIM = 64  # SONIC latent width; trailing 2 hand-bit dims are masked off.
_FULL_TOKEN_DIM = 66

# Matches ``trackingData_<TS>_latent_tokens_66d.npy`` and captures ``<TS>`` so
# the corresponding ``CameraRecord_<TS>_left.mp4`` can be derived without an
# extra directory scan per episode.
_TOKENS_NAME_RE = re.compile(r"^trackingData_(?P<ts>.+)_latent_tokens_66d\.npy$")


class PropEgoDataset(TokenizedVideoDataset):
    """Single-root proprietary ego dataset.

    Multi-root concatenation is handled at the config layer via
    ``ConcatMixDataset``; this class indexes one ``data_root`` (e.g. one
    capture date) at a time.
    """

    def __init__(
        self,
        data_root: str,
        action_chunk_size: int = 50,
        split: str = "train",
        cache_root: str | None = None,
        sonic_dim: int = _FULL_TOKEN_DIM,
        max_episodes: int | None = None,
        index_workers: int = 32,
    ):
        if split != "train":
            raise ValueError(
                "Proprietary ego dumps only ship a single split; pass split='train'."
            )
        if sonic_dim != _FULL_TOKEN_DIM:
            raise ValueError(
                f"sonic_dim={sonic_dim} != {_FULL_TOKEN_DIM}; the Kappa head "
                f"expects the full 66-D SONIC layout. EgoDex-style masking "
                f"zeros the trailing 2 dims at the sample level."
            )
        super().__init__(action_chunk_size=action_chunk_size)

        self.data_root = data_root
        self.sonic_dim = sonic_dim
        self.cache_root = resolve_writable_cache_root(data_root, cache_root)

        self._load_index(index_workers=index_workers)
        if max_episodes is not None:
            self._truncate(max_episodes)
        self._finalize_index()

    # -- index ---------------------------------------------------------------

    def _index_path(self) -> str:
        return os.path.join(self.cache_root, "meta", "prop_ego_index.json")

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

        self._descriptions: list[str] = idx["descriptions"]
        # Stored as relative ``<task>/<ep>`` strings; the timestamp piece
        # ``<TS>`` is kept separately so token / video paths reconstruct
        # without re-scanning the episode directory at sample time.
        self._ep_paths: np.ndarray = np.asarray(idx["paths"], dtype=np.bytes_)
        self._ep_stems: np.ndarray = np.asarray(idx["stems"], dtype=np.bytes_)
        self._ep_lengths: np.ndarray = np.asarray(idx["token_rows"], dtype=np.int64)
        self._ep_desc_id: np.ndarray = np.asarray(idx["desc_ids"], dtype=np.int32)

    def _build_index(self, index_workers: int) -> dict[str, list]:
        from tqdm import tqdm

        ep_keys: list[tuple[str, str]] = []
        for task in sorted(os.listdir(self.data_root)):
            tdir = os.path.join(self.data_root, task)
            if not os.path.isdir(tdir):
                continue
            for ep in sorted(
                os.listdir(tdir),
                key=lambda x: (int(x) if x.isdigit() else 1 << 31, x),
            ):
                if os.path.isdir(os.path.join(tdir, ep)):
                    ep_keys.append((task, ep))

        def probe(key: tuple[str, str]):
            task, ep = key
            edir = os.path.join(self.data_root, task, ep)
            try:
                names = os.listdir(edir)
            except FileNotFoundError:
                return None
            ts: str | None = None
            for fn in names:
                m = _TOKENS_NAME_RE.match(fn)
                if m is not None:
                    ts = m.group("ts")
                    break
            if ts is None:
                return None
            tok_path = os.path.join(edir, f"trackingData_{ts}_latent_tokens_66d.npy")
            mp4_path = os.path.join(edir, f"CameraRecord_{ts}_left.mp4")
            txt_path = os.path.join(edir, "llm_description.txt")
            try:
                shape = np.load(tok_path, mmap_mode="r").shape
                if shape[0] < 1 or shape[1] != _FULL_TOKEN_DIM:
                    return None
                if not os.path.exists(mp4_path) or not os.path.exists(txt_path):
                    return None
                with open(txt_path) as f:
                    desc = f.read().strip()
            except Exception:
                return None
            return key, ts, int(shape[0]), desc

        results: list[tuple[tuple[str, str], str, int, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, index_workers)) as ex:
            for r in tqdm(
                ex.map(probe, ep_keys),
                total=len(ep_keys),
                desc=f"Indexing PropEgo[{os.path.basename(self.data_root.rstrip('/'))}]",
                unit="ep",
            ):
                if r is not None:
                    results.append(r)

        def _sort_key(item):
            (task, ep), _ts, _n, _d = item
            return (task, int(ep) if ep.isdigit() else 1 << 31, ep)

        results.sort(key=_sort_key)

        descs: list[str] = []
        desc_to_id: dict[str, int] = {}
        paths: list[str] = []
        stems: list[str] = []
        token_rows: list[int] = []
        desc_ids: list[int] = []
        for (task, ep), ts, n_tok, desc in results:
            if desc not in desc_to_id:
                desc_to_id[desc] = len(descs)
                descs.append(desc)
            paths.append(f"{task}/{ep}")
            stems.append(ts)
            token_rows.append(n_tok)
            desc_ids.append(desc_to_id[desc])

        return {
            "descriptions": descs,
            "paths": paths,
            "stems": stems,
            "token_rows": token_rows,
            "desc_ids": desc_ids,
        }

    def _truncate(self, n: int) -> None:
        self._ep_paths = self._ep_paths[:n].copy()
        self._ep_stems = self._ep_stems[:n].copy()
        self._ep_lengths = self._ep_lengths[:n].copy()
        self._ep_desc_id = self._ep_desc_id[:n].copy()

    # -- hooks ---------------------------------------------------------------

    def _video_for(self, ep: int, t: int) -> tuple[str, int]:
        ep_path = self._ep_paths[ep].decode()
        ts = self._ep_stems[ep].decode()
        # Tokens are 50 Hz, the trimmed left/right mp4s are 30 fps. Match
        # EgoDex's clamp-on-the-decoder contract.
        frame_idx = int(t * 30 // 50)
        return (
            os.path.join(self.data_root, ep_path, f"CameraRecord_{ts}_left.mp4"),
            frame_idx,
        )

    def _text(self, ep: int) -> str:
        return self._descriptions[int(self._ep_desc_id[ep])]

    def _tokens(self, ep: int) -> np.ndarray:
        ep_path = self._ep_paths[ep].decode()
        ts = self._ep_stems[ep].decode()
        return get_token_mmap(
            os.path.join(
                self.data_root, ep_path, f"trackingData_{ts}_latent_tokens_66d.npy"
            )
        )

    def _build_chunk(
        self, ep: int, t: int, tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """EgoDex-style chunk: keep the first 64 dims, zero the trailing 2
        hand-bit columns, and emit a per-dim ``sonic_mask`` that suppresses
        the loss on those two columns. Mirrors ``EgoDexDataset._build_chunk``
        even though the source tokens are already 66-D — the trailing values
        come from an offline thresholded hand-close signal and aren't
        compatible with the post-train hand-binary contract.
        """
        n = int(self._ep_lengths[ep])
        chunk_indices = [min(t + k, n - 1) for k in range(self.action_chunk_size)]
        chunk = np.asarray(tokens[chunk_indices], dtype=np.float32)
        # Zero the trailing 2 dims so the head never sees the offline hand
        # signal even before masking — important because the model still
        # consumes the chunk as a regression target.
        if chunk.shape[-1] > _RAW_TOKEN_DIM:
            chunk = chunk.copy()
            chunk[:, _RAW_TOKEN_DIM:] = 0.0
        time_valid = np.array(
            [1.0 if t + k < n else 0.0 for k in range(self.action_chunk_size)],
            dtype=np.float32,
        )
        dim_valid = np.zeros((self.sonic_dim,), dtype=np.float32)
        dim_valid[:_RAW_TOKEN_DIM] = 1.0
        sonic_mask = time_valid[:, None] * dim_valid[None, :]
        return chunk, sonic_mask

"""G1-WBT dataset with precomputed SONIC latent tokens.

Layout produced by ``scripts/preprocess/g1-wbt/calc_wbt_sonic_tokens.py``::

    <cache_root>/
        meta/episodes_meta.parquet                       # cross-subset index
        sonic_tokens/<subset>/episode_<NNNNNN>.npy       # [T_50hz, 66] tokens

LeRobot v3 packs many episodes into a single ``data/file-NNN.parquet`` and the
matching ``videos/.../file-NNN.mp4``. The episode's frames inside the mp4 start
at ``round(video_from_ts * 30)`` (30 fps native). Sampling is in 50 Hz token
space; the video frame at the same time is

    frame_in_mp4 = video_from_ts * 30
                 + min(int(t_50 * vid_len_30 / tok_len_50), vid_len_30 - 1)

The MainCamOnly subsets put the LeRobot tree inside an extra ``G1_WB_Dex5_<X>/``
directory and use ``cam_0`` instead of ``head_stereo_left``. Both quirks are
captured in the ``episodes_meta.parquet`` columns (``nested_dir``, ``video_key``).
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from src.data.tokenized_video_dataset import (
    TokenizedVideoDataset,
    get_token_mmap,
    resolve_writable_cache_root,
)


_VIDEO_FPS = 30.0  # native rate of every G1-WBT mp4 (per meta/info.json)


class WBTDataset(TokenizedVideoDataset):
    """G1-WBT loader. Emits the same dict schema as ``HEDataset``.

    Parameters
    ----------
    data_root
        Path to the G1-WBT dataset root (the directory holding the per-subset
        folders). Typically ``resource/dataset/G1-WBT``.
    action_chunk_size
        Number of 50 Hz tokens returned per sample.
    cache_root
        Where ``episodes_meta.parquet`` and ``sonic_tokens/`` live. ``None``
        falls back to ``data_root`` when writable, else ``data_root + '-cache'``.
    subsets
        ``"all"`` (default) or an iterable of subset directory names. Useful
        for restricting training to a subset family (e.g. only Inspire variants).
    """

    def __init__(
        self,
        data_root: str,
        action_chunk_size: int = 50,
        cache_root: str | None = None,
        subsets: "str | list[str]" = "all",
    ) -> None:
        super().__init__(action_chunk_size=action_chunk_size)
        self.data_root = data_root
        self.cache_root = resolve_writable_cache_root(data_root, cache_root)

        meta_path = os.path.join(self.cache_root, "meta", "episodes_meta.parquet")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Missing {meta_path}. Run "
                "`scripts/preprocess/g1-wbt/calc_wbt_sonic_tokens.py` first."
            )
        self.tokens_root = os.path.join(self.cache_root, "sonic_tokens")
        if not os.path.isdir(self.tokens_root):
            raise FileNotFoundError(
                f"Missing token dir {self.tokens_root}. Re-run the preprocess script."
            )

        meta = pl.read_parquet(meta_path)
        if isinstance(subsets, str):
            if subsets != "all":
                raise ValueError(f"subsets must be 'all' or a list, got {subsets!r}")
        else:
            wanted = set(subsets)
            meta = meta.filter(pl.col("subset").is_in(list(wanted)))
            if meta.is_empty():
                raise ValueError(
                    f"No episodes left after filtering on subsets={list(wanted)}"
                )

        self._build_index(meta)
        self._finalize_index()

    # -- index ---------------------------------------------------------------

    def _build_index(self, meta: pl.DataFrame) -> None:
        # Distinct subsets, mapped to compact int8 ids for the per-episode array.
        subset_names = meta["subset"].unique(maintain_order=True).to_list()
        if len(subset_names) > 127:
            raise ValueError(f"Too many subsets ({len(subset_names)}) for int8 id table")
        self._subset_names = subset_names
        sid_map = {n: i for i, n in enumerate(subset_names)}

        # Per-subset routing (video_key, nested_dir) — every row in a given
        # subset shares these by construction; re-derive from the first row to
        # avoid trusting input order.
        per_subset = (
            meta.group_by("subset", maintain_order=True)
            .agg([pl.col("video_key").first(), pl.col("nested_dir").first()])
        )
        self._subset_video_key: list[str] = [""] * len(subset_names)
        self._subset_nested_dir: list[str] = [""] * len(subset_names)
        for row in per_subset.iter_rows(named=True):
            sid = sid_map[row["subset"]]
            self._subset_video_key[sid] = row["video_key"]
            self._subset_nested_dir[sid] = row["nested_dir"] or ""

        # Walk episodes in row order. Probe each token .npy once for its 50 Hz
        # length (cheap with mmap_mode="r" — only opens the header).
        ep_count = meta.height
        ep_subset_id = np.empty(ep_count, dtype=np.int8)
        ep_episode_idx = np.empty(ep_count, dtype=np.int32)
        ep_tok_len = np.empty(ep_count, dtype=np.int32)
        ep_vid_len = np.empty(ep_count, dtype=np.int32)
        ep_video_chunk = np.empty(ep_count, dtype=np.int32)
        ep_video_file = np.empty(ep_count, dtype=np.int32)
        ep_video_frame_offset = np.empty(ep_count, dtype=np.int32)
        ep_task: list[str] = [""] * ep_count

        for i, row in enumerate(meta.iter_rows(named=True)):
            sub = row["subset"]
            sid = sid_map[sub]
            ep_idx = int(row["episode_index"])
            tok_path = os.path.join(self.tokens_root, sub, f"episode_{ep_idx:06d}.npy")
            tok_len = int(np.load(tok_path, mmap_mode="r").shape[0])
            ep_subset_id[i] = sid
            ep_episode_idx[i] = ep_idx
            ep_tok_len[i] = tok_len
            ep_vid_len[i] = int(row["length"])
            ep_video_chunk[i] = int(row["video_chunk"])
            ep_video_file[i] = int(row["video_file"])
            ep_video_frame_offset[i] = int(round(float(row["video_from_ts"]) * _VIDEO_FPS))
            ep_task[i] = row["task"] or "perform the task"

        self._ep_subset_id = ep_subset_id
        self._ep_episode_idx = ep_episode_idx
        self._ep_tok_len = ep_tok_len
        self._ep_vid_len = ep_vid_len
        self._ep_video_chunk = ep_video_chunk
        self._ep_video_file = ep_video_file
        self._ep_video_frame_offset = ep_video_frame_offset
        self._ep_task = ep_task
        self._ep_lengths = ep_tok_len.astype(np.int64, copy=False)

    # -- hooks ---------------------------------------------------------------

    def _video_path(self, subset_id: int, vid_chunk: int, vid_file: int) -> str:
        sub = self._subset_names[subset_id]
        nested = self._subset_nested_dir[subset_id]
        vk = self._subset_video_key[subset_id]
        parts = [self.data_root, sub]
        if nested:
            parts.append(nested)
        parts += [
            "videos", f"observation.images.{vk}",
            f"chunk-{vid_chunk:03d}", f"file-{vid_file:03d}.mp4",
        ]
        return os.path.join(*parts)

    def _video_for(self, ep: int, t: int) -> tuple[str, int]:
        tok_len = int(self._ep_tok_len[ep])
        vid_len = int(self._ep_vid_len[ep])
        # 50 Hz token index -> 30 Hz frame within the per-episode video span,
        # then add the offset that locates the episode inside the multi-episode
        # mp4 (LeRobot v3 packs many episodes per mp4 file).
        local_frame = min(int(t * vid_len / max(tok_len, 1)), max(vid_len - 1, 0))
        frame_idx = int(self._ep_video_frame_offset[ep]) + local_frame
        path = self._video_path(
            int(self._ep_subset_id[ep]),
            int(self._ep_video_chunk[ep]),
            int(self._ep_video_file[ep]),
        )
        return path, frame_idx

    def _text(self, ep: int) -> str:
        return self._ep_task[ep]

    def _tokens(self, ep: int) -> np.ndarray:
        sub = self._subset_names[int(self._ep_subset_id[ep])]
        ep_idx = int(self._ep_episode_idx[ep])
        return get_token_mmap(
            os.path.join(self.tokens_root, sub, f"episode_{ep_idx:06d}.npy")
        )

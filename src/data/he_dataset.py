"""Humanoid-Everyday dataset with precomputed SONIC latent tokens."""
from __future__ import annotations

import json
import os

import numpy as np
import polars as pl

from src.data.kappa_modality_configs import KAPPA_MODALITY_CONFIGS
from src.data.state_normalizer import StateNormalizer
from src.data.tokenized_video_dataset import (
    TokenizedVideoDataset,
    get_token_mmap,
    resolve_writable_cache_root,
)


class HEDataset(TokenizedVideoDataset):
    """One mp4 per episode at ``videos/chunk-<NNN>/egocentric/episode_<NNNNNN>.mp4``."""

    def __init__(
        self,
        data_root: str,
        action_chunk_size: int = 50,
        split: str = "train",
        cache_root: str | None = None,
        robot_type: str = "g1",
        emit_proprio: bool = False,
        stats: dict | None = None,
        use_percentiles: bool = True,
        clip_outliers: bool = True,
        episode_indices: list[int] | None = None,
    ):
        normalizer: StateNormalizer | None = None
        if emit_proprio and stats is not None:
            normalizer = StateNormalizer(
                KAPPA_MODALITY_CONFIGS, stats,
                use_percentiles=use_percentiles,
                clip_outliers=clip_outliers,
            )
        super().__init__(action_chunk_size=action_chunk_size, normalizer=normalizer)
        self.data_root = data_root
        # humanoid-everyday is a G1+H1 mix, but the SONIC token pipeline assumes
        # a single embodiment (29D jp with arm starting at index 15 only matches
        # G1). Default to G1; pass "all" to opt out once the pipeline supports
        # cross-embodiment encoding.
        self.robot_type = robot_type.lower()
        self._episode_filter: set[int] | None = (
            {int(i) for i in episode_indices} if episode_indices else None
        )
        self.cache_root = resolve_writable_cache_root(data_root, cache_root)
        self.sonic_tokens_dir = os.path.join(self.cache_root, "sonic_tokens")
        if not os.path.isdir(self.sonic_tokens_dir):
            raise FileNotFoundError(
                f"Precomputed sonic tokens not found at {self.sonic_tokens_dir}. "
                "Run `python scripts/preprocess/humanoid-everyday/calc_sonic_tokens.py` first."
            )

        self._emit_proprio = emit_proprio
        # ``g1_proprio/`` references the G1 *robot's* joint state cache; the
        # name is preserved across the schema swap to ``unitree_g1_sonic``.
        self.proprio_dir = os.path.join(self.cache_root, "g1_proprio")
        if emit_proprio and not os.path.isdir(self.proprio_dir):
            raise FileNotFoundError(
                f"Missing {self.proprio_dir}; re-run calc_sonic_tokens.py to dump g1_proprio/."
            )

        tasks_path = os.path.join(data_root, "meta", "tasks.jsonl")
        task_map: dict[int, str] = {}
        if os.path.exists(tasks_path):
            with open(tasks_path) as f:
                for line in f:
                    entry = json.loads(line)
                    task_map[entry["task_index"]] = entry["description"]
        self._task_map = task_map

        with open(os.path.join(data_root, "meta", "info.json")) as f:
            self.info = json.load(f)

        self._load_index()

    # -- index ---------------------------------------------------------------

    def _index_path(self) -> str:
        # ``_50hz_<robot_type>`` suffix bumps the cache so old 30 Hz / mixed-body
        # indices don't get reused after upsampling and the G1/H1 split.
        # Bumped from "dataset_index_50hz_*" so old per-frame cache files
        # (different .npz schema) get rebuilt on first load instead of
        # raising KeyError on the new column names.
        suffix = ""
        if self._episode_filter is not None:
            ep_list = "_".join(str(i) for i in sorted(self._episode_filter))
            suffix = f"_filter_ep{ep_list}"
        return os.path.join(
            self.cache_root, "meta",
            f"episode_index_50hz_{self.robot_type}{suffix}.npz",
        )

    def _load_index(self) -> None:
        cache_path = self._index_path()
        if os.path.exists(cache_path):
            with np.load(cache_path) as cache:
                self._ep_episode_idx = cache["ep_episode_idx"].astype(np.int32, copy=False)
                self._ep_tok_len = cache["ep_tok_len"].astype(np.int32, copy=False)
                self._ep_vid_len = cache["ep_vid_len"].astype(np.int32, copy=False)
                self._ep_task = cache["ep_task"].astype(np.int32, copy=False)
        else:
            self._build_index()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            np.savez(
                tmp_path,
                ep_episode_idx=self._ep_episode_idx,
                ep_tok_len=self._ep_tok_len,
                ep_vid_len=self._ep_vid_len,
                ep_task=self._ep_task,
            )
            written = tmp_path if os.path.exists(tmp_path) else f"{tmp_path}.npz"
            os.replace(written, cache_path)

        self._ep_lengths = self._ep_tok_len.astype(np.int64, copy=False)
        self._finalize_index()

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

    def _load_robot_type_map(self) -> dict[int, str]:
        path = os.path.join(self.data_root, "meta", "episodes.jsonl")
        out: dict[int, str] = {}
        with open(path) as f:
            for line in f:
                entry = json.loads(line)
                out[int(entry["episode_index"])] = str(entry.get("robot_type", "")).lower()
        return out

    def _build_index(self) -> None:
        parquet_paths = self._collect_parquet_paths()

        def _ep_idx_from_path(p: str) -> int:
            return int(os.path.basename(p)[len("episode_"):-len(".parquet")])

        if self.robot_type != "all":
            rt_map = self._load_robot_type_map()
            allowed = {idx for idx, rt in rt_map.items() if rt == self.robot_type}
            kept = [p for p in parquet_paths if _ep_idx_from_path(p) in allowed]
            print(
                f"[HEDataset] robot_type='{self.robot_type}': kept "
                f"{len(kept)} / {len(parquet_paths)} episodes."
            )
            parquet_paths = kept

        if self._episode_filter is not None:
            before = len(parquet_paths)
            parquet_paths = [
                p for p in parquet_paths if _ep_idx_from_path(p) in self._episode_filter
            ]
            print(
                f"[HEDataset] episode_indices filter "
                f"{sorted(self._episode_filter)}: kept {len(parquet_paths)} / {before} episodes."
            )
            if not parquet_paths:
                raise ValueError(
                    f"episode_indices={sorted(self._episode_filter)} matched no "
                    f"episodes (after robot_type='{self.robot_type}' filter)."
                )

        dfs = [pl.read_parquet(p, columns=["episode_index", "task_index"]) for p in parquet_paths]
        df = pl.concat(dfs)
        del dfs

        # Per-episode metadata. ``vid_len`` is the parquet/video row count
        # (30 Hz native); ``tok_len`` is the SONIC token row count (50 Hz after
        # calc_sonic_tokens.py's upsampling).
        ep_ids: list[int] = []
        ep_vid_lens: list[int] = []
        ep_tok_lens: list[int] = []
        ep_tasks: list[int] = []
        for ep_idx_val, group in df.group_by("episode_index", maintain_order=True):
            ep_id = int(ep_idx_val[0])
            npy_path = os.path.join(self.sonic_tokens_dir, f"episode_{ep_id:06d}.npy")
            tok_len = int(np.load(npy_path, mmap_mode="r").shape[0])
            ep_ids.append(ep_id)
            ep_vid_lens.append(group.height)
            ep_tok_lens.append(tok_len)
            ep_tasks.append(int(group["task_index"][0]))

        self._ep_episode_idx = np.asarray(ep_ids, dtype=np.int32)
        self._ep_tok_len = np.asarray(ep_tok_lens, dtype=np.int32)
        self._ep_vid_len = np.asarray(ep_vid_lens, dtype=np.int32)
        self._ep_task = np.asarray(ep_tasks, dtype=np.int32)

    # -- hooks ---------------------------------------------------------------

    def _video_for(self, ep: int, t: int) -> tuple[str, int]:
        ep_idx = int(self._ep_episode_idx[ep])
        tok_len = int(self._ep_tok_len[ep])
        vid_len = int(self._ep_vid_len[ep])
        # Use per-episode ratio rather than a fixed 30/50 so off-by-one
        # rounding from the resampler stays in sync with the encoded count.
        frame_idx = min(int(t * vid_len / max(tok_len, 1)), max(vid_len - 1, 0))
        path = os.path.join(
            self.data_root,
            f"videos/chunk-{ep_idx // 1000:03d}/egocentric/episode_{ep_idx:06d}.mp4",
        )
        return path, frame_idx

    def _text(self, ep: int) -> str:
        return self._task_map.get(int(self._ep_task[ep]), "perform the task")

    def _tokens(self, ep: int) -> np.ndarray:
        ep_idx = int(self._ep_episode_idx[ep])
        return get_token_mmap(
            os.path.join(self.sonic_tokens_dir, f"episode_{ep_idx:06d}.npy")
        )

    def _proprio(self, ep: int) -> np.ndarray | None:
        if not self._emit_proprio:
            return None
        ep_idx = int(self._ep_episode_idx[ep])
        return get_token_mmap(
            os.path.join(self.proprio_dir, f"episode_{ep_idx:06d}.npy")
        )

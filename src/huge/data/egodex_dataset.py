"""EgoDex dataset over repacked episodes.

Backup-branch contract:

```
resource/dataset/EgoDex/egodex_repacked_simple/
  episodes/<part>/<task>/<ep_id>/
    episode.mp4
    llm_description.txt
    token_state.npy      # [T, 64] SONIC latents at 50 Hz
```

EgoDex has no G1 proprio and no reliable hand binary labels, so this loader
pads 64-D tokens into the 66-D container and masks the hand dimensions.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from huge.data.episode_selection import (
    SELECTION_ALGORITHM,
    derive_source_seed,
    select_episode_indices,
    selected_episode_digest,
    task_histogram,
)
from huge.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM, pad_egodex_action
from huge.data.transforms import token_to_video_frame_index
from huge.data.video_reader import decode_video_frame


@dataclass(frozen=True)
class EgoDexEpisode:
    episode_dir: Path
    episode_id: str
    token_path: Path
    video_path: Path | None
    description: str
    length: int


@dataclass(frozen=True)
class EgoDexFrame:
    episode_idx: int
    frame_idx: int


def _task_from_episode_path(value: str) -> str:
    parts = Path(value).parts
    if len(parts) < 2:
        raise ValueError(f"cannot derive EgoDex task from episode path {value!r}")
    return parts[-2]


class EgoDexDataset:
    def __init__(
        self,
        data_root: str | Path = "resource/dataset/EgoDex/egodex_repacked_simple",
        *,
        action_horizon: int = ACTION_HORIZON,
        sample_stride: int = 1,
        max_episodes: int | None = None,
        fallback_language: str = "perform the task",
        cache_root: str | Path | None = None,
        index_workers: int = 32,
        episode_fraction: float | None = None,
        episode_selection_seed: int = 7,
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset_id = "egodex"
        self.cache_root = Path(cache_root) if cache_root is not None else self.data_root
        self.action_horizon = int(action_horizon)
        self.sample_stride = int(sample_stride)
        if self.sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive, got {self.sample_stride}")
        self.fallback_language = fallback_language
        self._load_index(index_workers=index_workers)
        source_token_rows = int(self._ep_token_rows.sum())
        source_count = len(self._ep_paths)
        indices = select_episode_indices(
            source_count,
            source_id="egodex",
            base_seed=episode_selection_seed,
            fraction=episode_fraction,
            max_episodes=max_episodes,
        )
        self._select(indices)
        selected_ids = [path.decode() for path in self._ep_paths]
        self._selected_episode_ids = tuple(selected_ids)
        self._finalize_index()
        self._selection_summary = {
            "dataset_id": "egodex",
            "source_id": "egodex",
            "selection_schema_version": 1,
            "algorithm": SELECTION_ALGORITHM,
            "adapter_name": type(self).__name__,
            "adapter_schema_version": 1,
            "data_root": str(self.data_root),
            "cache_root": str(self.cache_root),
            "selection_mode": "prefix_debug" if max_episodes is not None else "fraction",
            "episode_fraction": episode_fraction,
            "requested_fraction": episode_fraction,
            "effective_episode_fraction": len(selected_ids) / source_count,
            "episode_selection_seed": int(episode_selection_seed),
            "base_selection_seed": int(episode_selection_seed),
            "source_seed": derive_source_seed(episode_selection_seed, "egodex"),
            "derived_source_seed": derive_source_seed(episode_selection_seed, "egodex"),
            "available_episodes": source_count,
            "source_episode_count": source_count,
            "selected_episodes": len(selected_ids),
            "selected_episode_count": len(selected_ids),
            "selected_episode_digest": selected_episode_digest(selected_ids),
            "selected_episode_ids_sha256": selected_episode_digest(selected_ids),
            "source_segment_count": source_count,
            "selected_segment_count": len(selected_ids),
            "source_token_rows": source_token_rows,
            "selected_token_rows": int(self._ep_token_rows.sum()),
            "selected_stride_samples": len(self),
            "task_episode_counts": task_histogram(
                selected_ids, [_task_from_episode_path(episode_id) for episode_id in selected_ids]
            ),
        }

    def _index_path(self) -> Path:
        return self.cache_root / "meta" / "egodex_index.json"

    def _load_index(self, *, index_workers: int) -> None:
        index_path = self._index_path()
        if index_path.exists():
            idx = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            idx = self._build_index(index_workers=index_workers)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = index_path.with_suffix(index_path.suffix + f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(idx), encoding="utf-8")
            os.replace(tmp, index_path)

        if "paths" in idx:
            self._descriptions = list(idx["descriptions"])
            self._ep_paths = np.asarray(idx["paths"], dtype=np.bytes_)
            self._ep_token_rows = np.asarray(idx["token_rows"], dtype=np.int64)
            self._ep_desc_id = np.asarray(idx["desc_ids"], dtype=np.int32)
            return

        if "episodes" in idx:
            descriptions: list[str] = []
            paths: list[str] = []
            token_rows: list[int] = []
            desc_ids: list[int] = []
            desc_to_id: dict[str, int] = {}
            episodes_root = self.data_root / "episodes"
            for row in idx["episodes"]:
                token_path = Path(row["token_path"])
                if token_path.is_absolute():
                    ep_dir = token_path.parent
                else:
                    ep_dir = (self.data_root / token_path).parent
                try:
                    rel = ep_dir.relative_to(self.data_root).as_posix()
                except ValueError:
                    rel = ep_dir.relative_to(episodes_root).as_posix()
                    rel = f"episodes/{rel}"
                desc_path = ep_dir / "llm_description.txt"
                desc = desc_path.read_text(encoding="utf-8").strip() if desc_path.exists() else self.fallback_language
                desc = desc or self.fallback_language
                if desc not in desc_to_id:
                    desc_to_id[desc] = len(descriptions)
                    descriptions.append(desc)
                paths.append(rel)
                token_rows.append(int(row["length"]))
                desc_ids.append(desc_to_id[desc])
            self._descriptions = descriptions
            self._ep_paths = np.asarray(paths, dtype=np.bytes_)
            self._ep_token_rows = np.asarray(token_rows, dtype=np.int64)
            self._ep_desc_id = np.asarray(desc_ids, dtype=np.int32)
            return

        raise ValueError(f"unsupported EgoDex index format: {index_path}")

    def _build_index(self, *, index_workers: int) -> dict[str, list]:
        episodes_root = self.data_root / "episodes"
        if not episodes_root.exists():
            raise FileNotFoundError(f"missing EgoDex episodes root: {episodes_root}")
        token_paths = sorted(episodes_root.rglob("token_state.npy"))

        def probe(token_path: Path) -> tuple[str, int, str] | None:
            try:
                arr = np.load(token_path, mmap_mode="r")
                if arr.ndim != 2 or arr.shape[1] != 64:
                    return None
            except Exception:
                return None
            ep_dir = token_path.parent
            desc_path = ep_dir / "llm_description.txt"
            if desc_path.exists():
                desc = desc_path.read_text(encoding="utf-8").strip() or self.fallback_language
            else:
                desc = self.fallback_language
            if not (ep_dir / "episode.mp4").exists():
                return None
            return ep_dir.relative_to(self.data_root).as_posix(), int(arr.shape[0]), desc

        rows: list[tuple[str, int, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, index_workers)) as executor:
            for row in tqdm(executor.map(probe, token_paths), total=len(token_paths), desc="Indexing EgoDex", unit="ep"):
                if row is not None:
                    rows.append(row)
        if not rows:
            raise FileNotFoundError(f"no EgoDex token_state.npy files found under {episodes_root}")

        descriptions: list[str] = []
        desc_to_id: dict[str, int] = {}
        paths: list[str] = []
        token_rows: list[int] = []
        desc_ids: list[int] = []
        for rel_path, length, desc in rows:
            if desc not in desc_to_id:
                desc_to_id[desc] = len(descriptions)
                descriptions.append(desc)
            paths.append(rel_path)
            token_rows.append(length)
            desc_ids.append(desc_to_id[desc])
        return {"descriptions": descriptions, "paths": paths, "token_rows": token_rows, "desc_ids": desc_ids}

    def _select(self, indices: np.ndarray) -> None:
        self._ep_paths = self._ep_paths[indices].copy()
        self._ep_token_rows = self._ep_token_rows[indices].copy()
        self._ep_desc_id = self._ep_desc_id[indices].copy()

    def _finalize_index(self) -> None:
        self._ep_lengths = ((self._ep_token_rows + self.sample_stride - 1) // self.sample_stride).astype(np.int64)
        self._ep_offsets = np.empty(len(self._ep_lengths), dtype=np.int64)
        if len(self._ep_lengths):
            self._ep_offsets[0] = 0
            if len(self._ep_lengths) > 1:
                np.cumsum(self._ep_lengths[:-1], out=self._ep_offsets[1:])
            self._total = int(self._ep_lengths.sum())
        else:
            self._total = 0

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        ep_idx = int(np.searchsorted(self._ep_offsets, idx, side="right")) - 1
        local_idx = idx - int(self._ep_offsets[ep_idx])
        return ep_idx, int(local_idx * self.sample_stride)

    def episode_spans(self) -> np.ndarray:
        spans = np.empty((len(self._ep_lengths), 2), dtype=np.int64)
        spans[:, 0] = self._ep_offsets
        spans[:, 1] = self._ep_lengths
        return spans

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, frame_idx = self._resolve(idx)
        ep_rel = self._ep_paths[ep_idx].decode()
        ep_dir = self.data_root / ep_rel
        token_path = ep_dir / "token_state.npy"
        tokens = np.load(token_path, mmap_mode="r")
        chunk = tokens[frame_idx : frame_idx + self.action_horizon]
        valid = len(chunk)
        action, mask = pad_egodex_action(chunk)
        if valid < self.action_horizon:
            padded = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
            padded_mask = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
            padded[:valid] = action
            padded_mask[:valid] = mask
            action = padded
            mask = padded_mask
        video = ep_dir / "episode.mp4"
        video_frame_idx = None
        if video.exists():
            # Backup branch used int(t * 30 // 50). This equivalent mapping is
            # clamped and does not need video decoding during indexing.
            token_rows = int(self._ep_token_rows[ep_idx])
            video_len_est = max(1, int(round(token_rows * 30 / 50)))
            video_frame_idx = token_to_video_frame_index(frame_idx, token_rows, video_len_est)
        image = decode_video_frame(video, video_frame_idx or 0, missing_ok=False)
        return {
            "dataset_id": "egodex",
            "task_id": _task_from_episode_path(ep_rel),
            "episode_id": ep_rel.removeprefix("episodes/"),
            "frame_idx": frame_idx,
            "video_path": str(video) if video.exists() else None,
            "video_frame_idx": video_frame_idx,
            "language": self._descriptions[int(self._ep_desc_id[ep_idx])],
            "state": np.zeros((PROPRIO_DIM,), dtype=np.float32),
            "state_mask": np.zeros((1,), dtype=np.float32),
            "normalized_actions": action,
            "sonic_mask": mask,
            "image": image,
        }


    def data_contract(self) -> dict[str, Any]:
        index_path = self._index_path()
        digest = hashlib.sha256()
        with index_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return {
            **self._selection_summary,
            "samples": len(self),
            "selected_token_rows": int(self._ep_token_rows.sum()),
            "index_file": {
                "path": str(index_path),
                "size": index_path.stat().st_size,
                "sha256": digest.hexdigest(),
            },
            "vision_source": "episode.mp4",
            "vision_time_mapping": "token_50hz_to_video_30hz",
        }


    def selection_summary(self) -> dict[str, Any]:
        return self.data_contract()

    def selected_episode_ids(self) -> tuple[str, ...]:
        return self._selected_episode_ids

def build_egodex_dataset(data_root: str | Path, **kwargs) -> EgoDexDataset:
    return EgoDexDataset(data_root, **kwargs)

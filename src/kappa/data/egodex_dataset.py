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

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM, pad_egodex_action
from kappa.data.transforms import token_to_video_frame_index


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
    ) -> None:
        self.data_root = Path(data_root)
        self.cache_root = Path(cache_root) if cache_root is not None else self.data_root
        self.action_horizon = int(action_horizon)
        self.sample_stride = int(sample_stride)
        self.fallback_language = fallback_language
        self._load_index(index_workers=index_workers)
        if max_episodes is not None:
            self._truncate(max_episodes)
        self._finalize_index()

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

    def _truncate(self, max_episodes: int) -> None:
        self._ep_paths = self._ep_paths[:max_episodes].copy()
        self._ep_token_rows = self._ep_token_rows[:max_episodes].copy()
        self._ep_desc_id = self._ep_desc_id[:max_episodes].copy()

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
        return {
            "dataset_id": "egodex",
            "task_id": "egodex",
            "episode_id": ep_rel.removeprefix("episodes/"),
            "frame_idx": frame_idx,
            "video_path": str(video) if video.exists() else None,
            "video_frame_idx": video_frame_idx,
            "language": self._descriptions[int(self._ep_desc_id[ep_idx])],
            "state": np.zeros((PROPRIO_DIM,), dtype=np.float32),
            "state_mask": np.zeros((1,), dtype=np.float32),
            "normalized_actions": action,
            "sonic_mask": mask,
            "image": None,
        }


def build_egodex_dataset(data_root: str | Path, **kwargs) -> EgoDexDataset:
    return EgoDexDataset(data_root, **kwargs)

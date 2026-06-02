"""Tokenized SONIC dataset over precomputed 50 Hz npy caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM, pad_egodex_action, validate_action_array
from kappa.data.state_normalizer import StateNormalizer


@dataclass(frozen=True)
class EpisodeRecord:
    dataset_id: str
    episode_id: str
    token_path: Path
    proprio_path: Path | None
    video_path: Path | None
    video_len_hint: int | None
    length: int


@dataclass(frozen=True)
class FrameIndex:
    episode_idx: int
    frame_idx: int


def _find_npy(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.npy") if p.is_file())


def _episode_id(path: Path) -> str:
    return path.stem.replace("episode_", "")


def _candidate_proprio_paths(cache_root: Path, token_path: Path) -> list[Path]:
    rel_parts = token_path.relative_to(cache_root).parts
    candidates: list[Path] = []
    if "sonic_tokens" in rel_parts:
        idx = rel_parts.index("sonic_tokens")
        suffix = Path(*rel_parts[idx + 1 :])
        candidates.append(cache_root / "g1_proprio" / suffix)
    candidates.append(cache_root / "g1_proprio" / token_path.name)
    return candidates


def _first_existing(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _candidate_video_paths(
    cache_root: Path,
    token_path: Path,
    token_len: int,
    *,
    video_len_hint_mode: str = "legacy_30fps",
) -> tuple[Path | None, int | None]:
    """Best-effort video path resolver for HE/WBT/proprietary cache layouts."""
    stem = token_path.stem
    rel_parts = token_path.relative_to(cache_root).parts
    subset = rel_parts[1] if len(rel_parts) >= 3 and rel_parts[0] == "sonic_tokens" else None
    sibling_root = cache_root.parent / cache_root.name.removesuffix("-cache")
    candidates: list[Path] = []
    if cache_root.name == "G1_WBT" and subset is not None:
        for key in ("observation.images.head_stereo_left", "observation.images.cam_0", "observation.images.ego_view"):
            candidates.append(cache_root / "videos" / subset / key / f"{stem}.mp4")
            candidates.append(cache_root / subset / "episode_videos" / key / f"{stem}.mp4")
    elif cache_root.name == "humanoid-everyday-cache":
        try:
            ep_idx = int(stem.replace("episode_", ""))
            candidates.append(sibling_root / "videos" / f"chunk-{ep_idx // 1000:03d}" / "egocentric" / f"{stem}.mp4")
        except ValueError:
            pass
    else:
        candidates.extend([
            cache_root / "videos" / "chunk-000" / "observation.images.ego_view" / f"{stem}.mp4",
            cache_root / "videos" / "observation.images.ego_view" / f"{stem}.mp4",
        ])
    path = _first_existing(candidates)
    if path is None:
        return None, None
    if video_len_hint_mode == "token_1to1":
        return path, token_len
    if video_len_hint_mode != "legacy_30fps":
        raise ValueError(f"unknown video_len_hint_mode {video_len_hint_mode!r}")
    return path, max(1, int(round(token_len * 30 / 50)))


class SonicTokenDataset:
    """Map-style dataset that emits 50-step 66-D action chunks.

    Images are intentionally left as optional metadata in this first pass; the
    Qwen processor/collator can attach frames once video indexing is finalized.
    """

    def __init__(
        self,
        cache_root: str | Path,
        *,
        dataset_id: str,
        task_id: str = "default",
        action_horizon: int = ACTION_HORIZON,
        sample_stride: int = 1,
        proprio_normalizer: StateNormalizer | None = None,
        state_required: bool = True,
        egodex_64d: bool = False,
        language: str = "perform the task",
        language_map: dict[str, str] | None = None,
        max_episodes: int | None = None,
        video_len_hint_mode: str = "legacy_30fps",
    ) -> None:
        self.cache_root = Path(cache_root)
        self.dataset_id = dataset_id
        self.task_id = task_id
        self.action_horizon = int(action_horizon)
        self.sample_stride = int(sample_stride)
        self.proprio_normalizer = proprio_normalizer
        self.state_required = state_required
        self.egodex_64d = egodex_64d
        self.language = language
        self.language_map = language_map or {}
        self.max_episodes = max_episodes
        self.video_len_hint_mode = video_len_hint_mode
        self.episodes = self._discover_episodes()
        self._finalize_index()

    def _discover_episodes(self) -> list[EpisodeRecord]:
        token_root = self.cache_root / "sonic_tokens"
        token_paths = _find_npy(token_root if token_root.exists() else self.cache_root)
        records: list[EpisodeRecord] = []
        for token_path in token_paths:
            token = np.load(token_path, mmap_mode="r")
            if token.ndim != 2:
                raise ValueError(f"{token_path} must be [T, D], got {token.shape}")
            validate_action_array(np.asarray(token[:1]), allow_64d=self.egodex_64d)
            proprio_path = next((p for p in _candidate_proprio_paths(self.cache_root, token_path) if p.exists()), None)
            if self.state_required and proprio_path is None:
                raise FileNotFoundError(f"missing g1_proprio cache for {token_path}")
            if proprio_path is not None:
                proprio = np.load(proprio_path, mmap_mode="r")
                if proprio.ndim != 2 or proprio.shape[1] != PROPRIO_DIM:
                    raise ValueError(f"{proprio_path} must be [T, {PROPRIO_DIM}], got {proprio.shape}")
                if proprio.shape[0] != token.shape[0]:
                    raise ValueError(f"length mismatch {token_path}={token.shape[0]} vs {proprio_path}={proprio.shape[0]}")
            video_path, video_len_hint = _candidate_video_paths(
                self.cache_root,
                token_path,
                int(token.shape[0]),
                video_len_hint_mode=self.video_len_hint_mode,
            )
            records.append(
                EpisodeRecord(
                    dataset_id=self.dataset_id,
                    episode_id=_episode_id(token_path),
                    token_path=token_path,
                    proprio_path=proprio_path,
                    video_path=video_path,
                    video_len_hint=video_len_hint,
                    length=int(token.shape[0]),
                )
            )
            if self.max_episodes is not None and len(records) >= self.max_episodes:
                break
        if not records:
            raise FileNotFoundError(f"no .npy token files found under {self.cache_root}")
        return records

    def _finalize_index(self) -> None:
        self._ep_lengths = np.asarray(
            [(ep.length + self.sample_stride - 1) // self.sample_stride for ep in self.episodes],
            dtype=np.int64,
        )
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

    def __len__(self) -> int:
        return self._total

    def episode_spans(self) -> np.ndarray:
        spans = np.empty((len(self._ep_lengths), 2), dtype=np.int64)
        spans[:, 0] = self._ep_offsets
        spans[:, 1] = self._ep_lengths
        return spans

    def _load_action_chunk(self, ep: EpisodeRecord, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        raw = np.load(ep.token_path)
        chunk = raw[frame_idx : frame_idx + self.action_horizon]
        valid = len(chunk)
        if self.egodex_64d and chunk.shape[-1] == 64:
            action, mask = pad_egodex_action(chunk)
        else:
            validate_action_array(chunk)
            action = chunk.astype(np.float32, copy=False)
            mask = np.ones_like(action, dtype=np.float32)
        if valid < self.action_horizon:
            padded = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
            padded_mask = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
            padded[:valid] = action
            padded_mask[:valid] = mask
            action = padded
            mask = padded_mask
        return action, mask

    def _load_state(self, ep: EpisodeRecord, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if ep.proprio_path is None:
            return np.zeros((PROPRIO_DIM,), dtype=np.float32), np.zeros((1,), dtype=np.float32)
        proprio = np.load(ep.proprio_path)
        state = proprio[min(frame_idx, len(proprio) - 1)].astype(np.float32, copy=False)
        if self.proprio_normalizer is not None:
            state = self.proprio_normalizer.normalize(state)
        return state, np.ones((1,), dtype=np.float32)

    def _video_frame_idx(self, ep: EpisodeRecord, frame_idx: int) -> int | None:
        if ep.video_len_hint is None:
            return None
        return min(int(frame_idx * ep.video_len_hint / max(ep.length, 1)), max(ep.video_len_hint - 1, 0))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, frame_idx = self._resolve(idx)
        ep = self.episodes[ep_idx]
        action, sonic_mask = self._load_action_chunk(ep, frame_idx)
        state, state_mask = self._load_state(ep, frame_idx)
        return {
            "dataset_id": ep.dataset_id,
            "task_id": self.task_id,
            "episode_id": ep.episode_id,
            "frame_idx": frame_idx,
            "language": self.language_map.get(ep.episode_id, self.language),
            "video_path": str(ep.video_path) if ep.video_path else None,
            "video_frame_idx": self._video_frame_idx(ep, frame_idx),
            "state": state,
            "state_mask": state_mask,
            "normalized_actions": action,
            "sonic_mask": sonic_mask,
            "image": None,
        }


class MixtureDataset:
    """Natural-frame-ratio mixture over multiple map-style datasets."""

    def __init__(self, datasets: Sequence[Any]) -> None:
        if not datasets:
            raise ValueError("datasets must be non-empty")
        self.datasets = list(datasets)
        lengths = [len(d) for d in self.datasets]
        self._cum = np.cumsum(lengths)

    def __len__(self) -> int:
        return int(self._cum[-1])

    def episode_spans(self) -> np.ndarray:
        spans: list[np.ndarray] = []
        offset = 0
        for dataset in self.datasets:
            fn = getattr(dataset, "episode_spans", None)
            if fn is not None:
                arr = np.asarray(fn(), dtype=np.int64).copy()
                arr[:, 0] += offset
                spans.append(arr)
            offset += len(dataset)
        return np.concatenate(spans, axis=0) if spans else np.empty((0, 2), dtype=np.int64)

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        dataset_idx = int(np.searchsorted(self._cum, idx, side="right"))
        offset = 0 if dataset_idx == 0 else int(self._cum[dataset_idx - 1])
        return self.datasets[dataset_idx][idx - offset]

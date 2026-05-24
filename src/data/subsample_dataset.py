"""Episode-level random subsample of a token-row-indexed dataset.

Picks ``n_episodes`` whole episodes from a dataset that exposes
``episode_spans() -> [N, 2]`` (i.e., any ``TokenizedVideoDataset`` or
``ConcatMixDataset``) with a fixed seed, and flattens them into a fresh
contiguous index space. The picked episodes keep their original
``(start, length)`` shape; only the global offsets are re-numbered so
``BlockShuffleSampler`` consumes a tight ``[0, sum(lengths))`` range.

Used by data configs that want to dial down training-data volume at
episode granularity (``--data.max-samples N`` ⇒ pick N random episodes
across the union of ``data_roots``). Within each picked episode every
frame is kept, so block-shuffle decode locality is preserved exactly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import Dataset


class EpisodeSubsampleDataset(Dataset):
    """Keep ``n_episodes`` random episodes from ``base``; concat their frames."""

    def __init__(self, base: Dataset, n_episodes: int, seed: int = 0) -> None:
        spans_fn = getattr(base, "episode_spans", None)
        if spans_fn is None:
            raise AttributeError(
                f"{type(base).__name__} lacks episode_spans(); "
                "episode-level subsampling needs the per-episode (start, length) layout."
            )
        spans = np.asarray(spans_fn(), dtype=np.int64)
        if spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError(f"episode_spans() returned shape {spans.shape}, expected [N, 2]")

        n_total = spans.shape[0]
        if n_episodes <= 0:
            raise ValueError(f"n_episodes must be > 0, got {n_episodes}")
        if n_episodes > n_total:
            raise ValueError(
                f"n_episodes={n_episodes} exceeds available episode count {n_total}"
            )

        rng = np.random.default_rng(seed)
        picked = rng.choice(n_total, size=n_episodes, replace=False)
        picked.sort()  # episode order in the new index space matches the original

        self._orig_starts = spans[picked, 0].astype(np.int64, copy=False)
        self._orig_lengths = spans[picked, 1].astype(np.int64, copy=False)
        self._offsets = np.empty(n_episodes + 1, dtype=np.int64)
        self._offsets[0] = 0
        np.cumsum(self._orig_lengths, out=self._offsets[1:])
        self._total = int(self._offsets[-1])
        self._base = base
        self._picked = picked.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        ep = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        t = idx - int(self._offsets[ep])
        return self._base[int(self._orig_starts[ep]) + t]

    def episode_spans(self) -> np.ndarray:
        """Re-numbered ``(start, length)`` over the subsampled index space."""
        out = np.empty((self._orig_lengths.size, 2), dtype=np.int64)
        out[:, 0] = self._offsets[:-1]
        out[:, 1] = self._orig_lengths
        return out

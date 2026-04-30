"""Concat-style mixed dataset that preserves ``episode_spans`` for BlockShuffle.

``torch.utils.data.ConcatDataset`` already implements length-weighted
concatenation, but its ``__getitem__`` is the only thing the trainer can rely
on. ``KappaTrainer`` looks up ``train_dataset.raw_dataset.episode_spans()`` to
build a ``BlockShuffleSampler``; if the raw dataset is a mix we have to stitch
the per-source spans together with the right global offsets, otherwise block
shuffle silently degrades to plain ``shuffle=True`` for one of the sources.

Behavior:

* Sample i in ``[0, len(d_0))`` is drawn from ``d_0[i]``; in ``[len(d_0),
  len(d_0)+len(d_1))`` from ``d_1[i - len(d_0)]``; etc. So natural-frame-ratio
  mixing falls out of plain DataLoader iteration.
* ``episode_spans()`` returns the union of each child's spans with their start
  indices shifted by the cumulative offsets — same ``[N, 2]`` int64 layout
  that ``BlockShuffleSampler`` expects.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from torch.utils.data import Dataset


class ConcatMixDataset(Dataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        self._datasets = list(datasets)
        if not self._datasets:
            raise ValueError("ConcatMixDataset requires >= 1 dataset")
        for i, d in enumerate(self._datasets):
            if not hasattr(d, "__len__"):
                raise TypeError(f"datasets[{i}] is not sized; need __len__ for offsets")

        lens = np.asarray([len(d) for d in self._datasets], dtype=np.int64)
        self._lens = lens
        self._offsets = np.empty(len(lens) + 1, dtype=np.int64)
        self._offsets[0] = 0
        np.cumsum(lens, out=self._offsets[1:])
        self._total = int(self._offsets[-1])

    def __len__(self) -> int:
        return self._total

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        di = int(np.searchsorted(self._offsets, idx, side="right")) - 1
        return di, idx - int(self._offsets[di])

    def __getitem__(self, idx: int) -> Any:
        di, local = self._resolve(idx)
        return self._datasets[di][local]

    def episode_spans(self) -> np.ndarray:
        """Concatenate per-dataset spans with shifted start indices."""
        all_spans: list[np.ndarray] = []
        for di, d in enumerate(self._datasets):
            fn = getattr(d, "episode_spans", None)
            if fn is None:
                raise AttributeError(
                    f"datasets[{di}] ({type(d).__name__}) lacks episode_spans(); "
                    "every component must expose it for BlockShuffle to work end-to-end."
                )
            spans = np.asarray(fn(), dtype=np.int64)
            if spans.ndim != 2 or spans.shape[1] != 2:
                raise ValueError(
                    f"datasets[{di}].episode_spans() returned shape {spans.shape}, "
                    "expected [N, 2]"
                )
            shift = int(self._offsets[di])
            if shift and spans.size:
                spans = spans.copy()
                spans[:, 0] += shift
            all_spans.append(spans)
        return np.concatenate(all_spans, axis=0) if all_spans else np.zeros((0, 2), dtype=np.int64)

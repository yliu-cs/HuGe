"""Episode-aware block shuffle sampler."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class BlockShuffleSampler(Sampler[int]):
    def __init__(self, episode_spans: np.ndarray, block_size: int, generator: torch.Generator | None = None) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        spans = np.asarray(episode_spans, dtype=np.int64)
        if spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError(f"episode_spans must be [N,2], got {spans.shape}")
        self.block_size = int(block_size)
        self.generator = generator
        starts: list[int] = []
        lengths: list[int] = []
        for start, length in spans:
            for offset in range(0, int(length), self.block_size):
                starts.append(int(start + offset))
                lengths.append(min(self.block_size, int(length - offset)))
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.total = int(spans[:, 1].sum())

    def __len__(self) -> int:
        return self.total

    def __iter__(self) -> Iterator[int]:
        if self.generator is not None:
            seed = int(torch.empty((), dtype=torch.int64).random_(generator=self.generator).item())
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        for block_id in rng.permutation(len(self.starts)):
            start = int(self.starts[block_id])
            length = int(self.lengths[block_id])
            yield from range(start, start + length)

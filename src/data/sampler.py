"""Episode-aware block-shuffle sampler.

Random shuffle (``shuffle=True`` in DataLoader) hops across all 3.3e5 EgoDex
episodes uniformly. With H.264 GOP=12 (EgoDex) / 250 (HE), each random seek
costs 23–67 ms because decord must rewind to the previous I-frame and decode
forward — the per-worker ``VideoReader`` LRU never gets a hit.

``BlockShuffleSampler`` keeps shuffle-grade randomness across batches but
preserves *intra-block* sequential access: it chops each episode's row range
into ``block_size``-sized contiguous chunks, globally shuffles the chunk
order, and concatenates. Within one chunk the worker decodes ``next-frame``,
which is ~50× cheaper than a fresh seek (see ``video_reader`` benchmark).

Sized for the typical loader: ``block_size=batch_size`` means one batch
usually comes from one episode, and round-robin worker dispatch keeps each
worker locked onto a single ``VideoReader`` for the duration of the batch.

DDP: do not pre-shard. ``Accelerator.prepare(dataloader)`` wraps the sampler
in ``BatchSamplerShard`` which slices batches across ranks; pre-sharding here
would double the slicing.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Sampler


class BlockShuffleSampler(Sampler[int]):
    """Yield episode-contiguous blocks of indices in shuffled order.

    Parameters
    ----------
    episode_spans
        ``[N, 2]`` int64 array of ``(start_idx, length)``. Concatenation of
        all spans must equal ``range(0, sum(lengths))`` — i.e., the dataset
        is laid out episode-contiguous (both ``EgoDexDataset`` and
        ``HEDataset`` satisfy this by construction).
    block_size
        Maximum size of one contiguous block. Tail of an episode shorter
        than ``block_size`` is kept as a smaller block (no row dropped).
    generator
        Optional ``torch.Generator`` for reproducibility. If ``None`` we use
        a fresh ``np.random.default_rng()`` and accept non-determinism.
    """

    def __init__(
        self,
        episode_spans: np.ndarray,
        block_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        spans = np.ascontiguousarray(episode_spans, dtype=np.int64)
        if spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError(f"episode_spans must be [N, 2], got {spans.shape}")

        self.block_size = int(block_size)
        self.generator = generator
        self._build_block_table(spans)

    def _build_block_table(self, spans: np.ndarray) -> None:
        # For each (start, length), produce ceil(length / block_size) blocks.
        # Vectorized so we don't loop over ~3e5 episodes in Python.
        starts = spans[:, 0]
        lengths = spans[:, 1]
        n_blocks_per_ep = (lengths + self.block_size - 1) // self.block_size
        total_blocks = int(n_blocks_per_ep.sum())

        block_starts = np.empty(total_blocks, dtype=np.int64)
        block_lengths = np.empty(total_blocks, dtype=np.int64)

        # Per-episode block offsets via cumulative sum on n_blocks_per_ep.
        ep_block_offsets = np.empty(spans.shape[0] + 1, dtype=np.int64)
        ep_block_offsets[0] = 0
        np.cumsum(n_blocks_per_ep, out=ep_block_offsets[1:])

        # Within each episode the k-th block covers
        # [start + k*B : start + min((k+1)*B, length)).
        # Implement as: arange(total_blocks) → (ep_id, k_in_ep) via searchsorted.
        block_ids = np.arange(total_blocks, dtype=np.int64)
        ep_ids = np.searchsorted(ep_block_offsets, block_ids, side="right") - 1
        k_in_ep = block_ids - ep_block_offsets[ep_ids]

        ep_starts = starts[ep_ids]
        ep_lengths = lengths[ep_ids]
        block_starts[:] = ep_starts + k_in_ep * self.block_size
        # Tail-block length = length - k*B, capped at block_size.
        remaining = ep_lengths - k_in_ep * self.block_size
        block_lengths[:] = np.minimum(remaining, self.block_size)

        self._block_starts = block_starts
        self._block_lengths = block_lengths
        self._total = int(lengths.sum())

    def __len__(self) -> int:
        return self._total

    def __iter__(self) -> Iterator[int]:
        n_blocks = self._block_starts.size
        if self.generator is not None:
            # Mirror RandomSampler: draw one int64 per call to advance state.
            seed = int(
                torch.empty((), dtype=torch.int64).random_(generator=self.generator).item()
            )
            rng = np.random.default_rng(seed & 0xFFFFFFFFFFFFFFFF)
        else:
            rng = np.random.default_rng()

        perm = rng.permutation(n_blocks)
        starts = self._block_starts
        lengths = self._block_lengths
        for k in perm:
            s = int(starts[k])
            n = int(lengths[k])
            # int() per element is ~2× faster than yielding numpy scalars,
            # which torch.utils.data.DataLoader has to coerce anyway.
            for i in range(s, s + n):
                yield i

"""Per-worker LRU cache of Decord ``VideoReader`` handles.

cv2 random seek (``CAP_PROP_POS_FRAMES``) on H.264 mp4 rewinds to the closest
I-frame and decodes forward — for EgoDex's 1080p / 30 fps clips that costs
5–20 ms per ``__getitem__``, dominating the data pipeline. Decord keeps an
internal index and exposes O(1) random access; together with reader handle
reuse across consecutive ``__getitem__`` calls the per-frame cost drops below
1 ms even with ``num_workers > 0`` re-shuffling.

The cache lives at module scope so each forked DataLoader worker owns its
own dict (Linux ``fork`` => CoW semantics, no cross-worker handle sharing).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np

# Default LRU size: trades RAM for hit-rate on shuffled training. Each open
# decord reader retains an internal frame index plus decoder state, so on
# 1080p EgoDex clips a single handle is several MB; with N ranks × M workers
# the cache is multiplied accordingly. 4 keeps the resident set small while
# still absorbing the within-episode sequential hits that drive the hit-rate.
_DEFAULT_CACHE_SIZE = 4

_READER_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_CACHE_SIZE = _DEFAULT_CACHE_SIZE

# Tiny LRU of decoded frames keyed by (path, frame_idx, target_size). EgoDex
# tokens are 50 Hz on a 30 Hz video, so consecutive token rows often map to the
# same video frame; under BlockShuffleSampler those samples arrive back-to-back
# in the worker, and a 2-slot cache absorbs ~37% of decodes. Frames are stored
# as read-only views so accidental in-place mutation by callers fails loudly.
_DEFAULT_FRAME_CACHE_SIZE = 2
_FRAME_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_FRAME_CACHE_SIZE = _DEFAULT_FRAME_CACHE_SIZE
_FRAME_CACHE_HITS = 0
_FRAME_CACHE_MISSES = 0


def set_cache_size(n: int) -> None:
    global _CACHE_SIZE
    _CACHE_SIZE = max(1, n)
    while len(_READER_CACHE) > _CACHE_SIZE:
        _READER_CACHE.popitem(last=False)


def set_frame_cache_size(n: int) -> None:
    global _FRAME_CACHE_SIZE
    _FRAME_CACHE_SIZE = max(0, n)
    while len(_FRAME_CACHE) > _FRAME_CACHE_SIZE:
        _FRAME_CACHE.popitem(last=False)


def get_frame_cache_stats() -> tuple[int, int]:
    """Return ``(hits, misses)`` for the per-worker decoded-frame cache."""
    return _FRAME_CACHE_HITS, _FRAME_CACHE_MISSES


def reset_frame_cache_stats() -> None:
    global _FRAME_CACHE_HITS, _FRAME_CACHE_MISSES
    _FRAME_CACHE_HITS = 0
    _FRAME_CACHE_MISSES = 0


def _open_reader(path: str, width: Optional[int] = None, height: Optional[int] = None):
    import decord

    # num_threads=1 — DataLoader workers already give us process-level
    # parallelism; per-reader threads only fight each other for the same
    # frames and burn CPU.
    # width/height let decord resize during decode (avoids materializing the
    # full-resolution intermediate frame), shrinking the per-frame buffer ~9×
    # for 1080p → 256.
    kwargs = {"num_threads": 1}
    if width is not None and height is not None:
        kwargs["width"] = width
        kwargs["height"] = height
    return decord.VideoReader(path, **kwargs)


def get_reader(
    path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
):
    """Return a cached ``decord.VideoReader`` for ``path`` (per-worker)."""
    key = (path, width, height)
    reader = _READER_CACHE.get(key)
    if reader is not None:
        _READER_CACHE.move_to_end(key)
        return reader
    reader = _open_reader(path, width, height)
    _READER_CACHE[key] = reader
    if len(_READER_CACHE) > _CACHE_SIZE:
        _READER_CACHE.popitem(last=False)
    return reader


def decode_frame(
    path: str,
    frame_idx: int,
    target_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Decode a single RGB frame.

    ``target_size`` is ``(W, H)``. When set the underlying reader decodes
    directly at that resolution (decord internally resizes), eliminating the
    full-resolution intermediate copy.
    """
    global _FRAME_CACHE_HITS, _FRAME_CACHE_MISSES
    if target_size is not None:
        W, H = target_size
        reader = get_reader(path, W, H)
    else:
        reader = get_reader(path)
    total = len(reader)
    if total == 0:
        H, W = (target_size[1], target_size[0]) if target_size else (480, 640)
        return np.zeros((H, W, 3), dtype=np.uint8)
    frame_idx = max(0, min(frame_idx, total - 1))

    if _FRAME_CACHE_SIZE > 0:
        key = (path, frame_idx, target_size)
        cached = _FRAME_CACHE.get(key)
        if cached is not None:
            _FRAME_CACHE.move_to_end(key)
            _FRAME_CACHE_HITS += 1
            return cached
        _FRAME_CACHE_MISSES += 1
        frame = reader[frame_idx].asnumpy()
        # Mark the cached array read-only so a downstream in-place mutation
        # (PIL.Image.fromarray shares the buffer) fails loudly instead of
        # silently corrupting later cache hits.
        frame.setflags(write=False)
        _FRAME_CACHE[key] = frame
        if len(_FRAME_CACHE) > _FRAME_CACHE_SIZE:
            _FRAME_CACHE.popitem(last=False)
        return frame

    return reader[frame_idx].asnumpy()

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


def set_cache_size(n: int) -> None:
    global _CACHE_SIZE
    _CACHE_SIZE = max(1, n)
    while len(_READER_CACHE) > _CACHE_SIZE:
        _READER_CACHE.popitem(last=False)


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
    frame = reader[frame_idx].asnumpy()  # decord returns RGB at decoded size
    return frame

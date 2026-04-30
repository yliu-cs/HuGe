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


class _PyAVReader:
    """decord-compatible shim over PyAV for codecs decord can't open (AV1).

    Exposes ``__len__`` and ``__getitem__(i)`` returning an object with
    ``.asnumpy()`` so callers don't branch on backend. Seeks are O(GOP) — the
    same granularity decord offers for H.264/H.265 — so the existing reader
    LRU + frame cache + ``BlockShuffleSampler`` all amortize cost identically.
    """

    class _Frame:
        __slots__ = ("_arr",)
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr
        def asnumpy(self) -> np.ndarray:
            return self._arr

    def __init__(self, path: str, width: Optional[int], height: Optional[int]) -> None:
        import av

        self._path = path
        self._width = width
        self._height = height
        # Force libdav1d threading to single — DataLoader workers parallelize
        # at the sample level, oversubscribing here just thrashes the cache.
        opts = {"threads": "1"}
        self._container = av.open(path, options=opts)
        self._stream = self._container.streams.video[0]
        # PyAV stream.frames is the muxer-reported total; sometimes 0 for
        # streams without an index, in which case fall back to time_base*duration.
        self._n = int(self._stream.frames)
        if self._n <= 0:
            dur = self._stream.duration
            tb = self._stream.time_base
            avg = self._stream.average_rate
            if dur is not None and tb is not None and avg is not None:
                self._n = int(round(float(dur * tb) * float(avg)))
        self._iter = None
        self._next_idx = -1  # index of the frame the iterator will yield next

    def __len__(self) -> int:
        return max(0, self._n)

    def _restart_iter(self) -> None:
        self._container.seek(0, stream=self._stream)
        self._iter = self._container.decode(video=0)
        self._next_idx = 0

    def _seek_near(self, target: int) -> None:
        # av.seek wants stream-time-base PTS; convert from frame index using
        # the stream's average rate. seek backward=True snaps to the nearest
        # keyframe at or before the target.
        avg = self._stream.average_rate
        tb = self._stream.time_base
        if avg is None or tb is None or avg == 0:
            self._restart_iter()
            return
        ts = int(target / float(avg) / float(tb))
        self._container.seek(ts, stream=self._stream, any_frame=False, backward=True)
        self._iter = self._container.decode(video=0)
        # We don't know exactly which frame the keyframe corresponds to until
        # we read it; decode-forward will handle the offset using PTS below.
        self._next_idx = -1  # unknown, resolve on the fly

    def __getitem__(self, idx: int) -> "_PyAVReader._Frame":
        target = max(0, min(int(idx), self._n - 1)) if self._n > 0 else 0

        # Sequential case: just step the iterator forward.
        if self._iter is not None and 0 <= self._next_idx <= target and (target - self._next_idx) < 64:
            for _ in range(target - self._next_idx):
                next(self._iter)
            frame = next(self._iter)
            self._next_idx = target + 1
            return _PyAVReader._Frame(self._to_rgb(frame))

        # Random access: seek to the keyframe before target, then decode forward
        # using PTS to count frames precisely.
        self._seek_near(target)
        avg = self._stream.average_rate
        tb = self._stream.time_base
        chosen = None
        last = None
        for fr in self._iter:
            last = fr
            if fr.pts is None:
                continue
            cur_idx = int(round(float(fr.pts * tb) * float(avg))) if (avg and tb) else 0
            if cur_idx >= target:
                chosen = fr
                self._next_idx = cur_idx + 1
                break
        if chosen is None:
            chosen = last  # ran past end; serve the last decoded frame
            self._next_idx = self._n
        return _PyAVReader._Frame(self._to_rgb(chosen))

    def _to_rgb(self, frame) -> np.ndarray:
        if self._width is not None and self._height is not None:
            frame = frame.reformat(width=self._width, height=self._height, format="rgb24")
        else:
            frame = frame.reformat(format="rgb24")
        # to_ndarray returns (H, W, 3) uint8 in RGB layout.
        return np.ascontiguousarray(frame.to_ndarray())


# Codec probe results keyed by path. Cheap (one open() per file) and cached
# for the lifetime of the worker.
_CODEC_CACHE: dict[str, str] = {}


def _probe_codec(path: str) -> str:
    name = _CODEC_CACHE.get(path)
    if name is not None:
        return name
    try:
        import av
        with av.open(path) as c:
            name = c.streams.video[0].codec_context.name or ""
    except Exception:
        name = ""
    _CODEC_CACHE[path] = name
    return name


# decord can't decode AV1 (its bundled ffmpeg lacks libdav1d); route those to
# PyAV transparently. Any other codec keeps the decord fast path.
_PYAV_CODECS = {"av1", "libdav1d"}


def _open_reader(path: str, width: Optional[int] = None, height: Optional[int] = None):
    if _probe_codec(path) in _PYAV_CODECS:
        return _PyAVReader(path, width, height)

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

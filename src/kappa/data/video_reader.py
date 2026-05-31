"""Cached video frame reader for training."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

_READER_CACHE: "OrderedDict[tuple[str, int | None, int | None], object]" = OrderedDict()
_FRAME_CACHE: "OrderedDict[tuple[str, int, tuple[int, int] | None], np.ndarray]" = OrderedDict()
_VIDEO_SIZE_CACHE: "OrderedDict[str, tuple[int, int] | None]" = OrderedDict()
_READER_CACHE_SIZE = 4
_FRAME_CACHE_SIZE = 2
_VIDEO_SIZE_CACHE_SIZE = 1024
DEFAULT_DECODE_LONG_EDGE = 512


def set_cache_size(n: int) -> None:
    global _READER_CACHE_SIZE
    _READER_CACHE_SIZE = max(1, int(n))
    while len(_READER_CACHE) > _READER_CACHE_SIZE:
        _READER_CACHE.popitem(last=False)


def set_frame_cache_size(n: int) -> None:
    global _FRAME_CACHE_SIZE
    _FRAME_CACHE_SIZE = max(0, int(n))
    while len(_FRAME_CACHE) > _FRAME_CACHE_SIZE:
        _FRAME_CACHE.popitem(last=False)


def _zero_image(size: tuple[int, int] = (224, 320)) -> Image.Image:
    h, w = size
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def _open_reader(path: str, width: int | None, height: int | None):
    try:
        import decord

        kwargs = {"num_threads": 1}
        if width is not None and height is not None:
            kwargs["width"] = width
            kwargs["height"] = height
        return decord.VideoReader(path, **kwargs)
    except Exception:
        return None


def _get_reader(path: str, width: int | None, height: int | None):
    key = (path, width, height)
    reader = _READER_CACHE.get(key)
    if reader is not None:
        _READER_CACHE.move_to_end(key)
        return reader
    reader = _open_reader(path, width, height)
    _READER_CACHE[key] = reader
    if len(_READER_CACHE) > _READER_CACHE_SIZE:
        _READER_CACHE.popitem(last=False)
    return reader


def _probe_video_size(path: Path) -> tuple[int, int] | None:
    key = str(path)
    if key in _VIDEO_SIZE_CACHE:
        _VIDEO_SIZE_CACHE.move_to_end(key)
        return _VIDEO_SIZE_CACHE[key]
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            size = None
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            size = (width, height) if width > 0 and height > 0 else None
        cap.release()
    except Exception:
        size = None
    _VIDEO_SIZE_CACHE[key] = size
    if len(_VIDEO_SIZE_CACHE) > _VIDEO_SIZE_CACHE_SIZE:
        _VIDEO_SIZE_CACHE.popitem(last=False)
    return size


def _decode_size_for_long_edge(path: Path, long_edge: int | None) -> tuple[int, int] | None:
    if long_edge is None or long_edge <= 0:
        return None
    size = _probe_video_size(path)
    if size is None:
        return None
    width, height = size
    current_long = max(width, height)
    if current_long <= long_edge:
        return None
    scale = float(long_edge) / float(current_long)
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return out_w, out_h


def _decode_rgb(path: Path, frame_idx: int, size_wh: tuple[int, int] | None) -> np.ndarray:
    width, height = size_wh if size_wh else (None, None)
    reader = _get_reader(str(path), width, height)
    if reader is not None:
        total = len(reader)
        if total <= 0:
            raise RuntimeError(f"empty video {path}")
        idx = max(0, min(int(frame_idx), total - 1))
        key = (str(path), idx, size_wh)
        if _FRAME_CACHE_SIZE > 0 and key in _FRAME_CACHE:
            _FRAME_CACHE.move_to_end(key)
            return _FRAME_CACHE[key]
        frame = reader[idx].asnumpy()
        frame.setflags(write=False)
        if _FRAME_CACHE_SIZE > 0:
            _FRAME_CACHE[key] = frame
            if len(_FRAME_CACHE) > _FRAME_CACHE_SIZE:
                _FRAME_CACHE.popitem(last=False)
        return frame

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = max(0, min(int(frame_idx), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not decode frame {idx} from {path}")
    if size_wh is not None:
        frame = cv2.resize(frame, size_wh, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _resize_long_edge(image: Image.Image, long_edge: int | None) -> Image.Image:
    if long_edge is None or long_edge <= 0:
        return image
    w, h = image.size
    current_long = max(w, h)
    if current_long <= long_edge:
        return image
    scale = float(long_edge) / float(current_long)
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return image.resize(size, Image.Resampling.BICUBIC)


def decode_video_frame(
    path: str | Path | None,
    frame_idx: int,
    *,
    output_size: tuple[int, int] = (224, 320),
    decode_long_edge: int | None = DEFAULT_DECODE_LONG_EDGE,
    missing_ok: bool = True,
) -> Image.Image:
    if path is None:
        if missing_ok:
            return _zero_image(output_size)
        raise FileNotFoundError("video path is None")
    src = Path(path)
    if not src.exists():
        if missing_ok:
            return _zero_image(output_size)
        raise FileNotFoundError(src)
    try:
        decode_size = _decode_size_for_long_edge(src, decode_long_edge)
        arr = _decode_rgb(src, int(frame_idx), decode_size)
        image = Image.fromarray(np.asarray(arr))
        return _resize_long_edge(image, decode_long_edge)
    except Exception:
        if not missing_ok:
            raise
        return _zero_image(output_size)


def image_to_float_array(image: Image.Image) -> np.ndarray:
    return (np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).astype(np.float32)

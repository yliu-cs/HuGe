"""Cached video frame reader for training."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from kappa.data.transforms import resize_short_center_crop


_READER_CACHE: "OrderedDict[tuple[str, int | None, int | None], object]" = OrderedDict()
_FRAME_CACHE: "OrderedDict[tuple[str, int, tuple[int, int] | None], np.ndarray]" = OrderedDict()
_READER_CACHE_SIZE = 4
_FRAME_CACHE_SIZE = 2


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
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def decode_video_frame(
    path: str | Path | None,
    frame_idx: int,
    *,
    output_size: tuple[int, int] = (224, 320),
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
        h, w = output_size
        arr = _decode_rgb(src, int(frame_idx), (w, h))
        image = Image.fromarray(np.asarray(arr))
        if image.size != (w, h):
            image = resize_short_center_crop(image, size=output_size)
        return image
    except Exception:
        if not missing_ok:
            raise
        return _zero_image(output_size)


def image_to_float_array(image: Image.Image) -> np.ndarray:
    return (np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).astype(np.float32)

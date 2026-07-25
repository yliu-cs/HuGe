"""Cached video frame reader for training."""

from __future__ import annotations

import atexit
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

_READER_CACHE: "OrderedDict[tuple[str, int | None, int | None], object]" = OrderedDict()
_FRAME_CACHE: "OrderedDict[tuple[str, int, tuple[int, int] | None], np.ndarray]" = OrderedDict()
_AV_CACHE: "OrderedDict[str, tuple[object, object]]" = OrderedDict()
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
    while len(_AV_CACHE) > _READER_CACHE_SIZE:
        container, _ = _AV_CACHE.popitem(last=False)[1]
        container.close()


def set_frame_cache_size(n: int) -> None:
    global _FRAME_CACHE_SIZE
    _FRAME_CACHE_SIZE = max(0, int(n))
    while len(_FRAME_CACHE) > _FRAME_CACHE_SIZE:
        _FRAME_CACHE.popitem(last=False)


def clear_video_caches() -> None:
    _FRAME_CACHE.clear()
    _READER_CACHE.clear()
    while _AV_CACHE:
        container, _ = _AV_CACHE.popitem(last=False)[1]
        container.close()
    _VIDEO_SIZE_CACHE.clear()


atexit.register(clear_video_caches)


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

def _get_av_reader(path: Path):
    import av

    key = str(path)
    cached = _AV_CACHE.get(key)
    if cached is not None:
        _AV_CACHE.move_to_end(key)
        return cached
    container = av.open(key)
    streams = [stream for stream in container.streams if stream.type == "video"]
    if not streams:
        container.close()
        raise RuntimeError(f"no video stream in {path}")
    cached = (container, streams[0])
    _AV_CACHE[key] = cached
    if len(_AV_CACHE) > _READER_CACHE_SIZE:
        old_container, _ = _AV_CACHE.popitem(last=False)[1]
        old_container.close()
    return cached


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


def _decode_rgb(
    path: Path,
    frame_idx: int,
    size_wh: tuple[int, int] | None,
    *,
    backend: str,
) -> np.ndarray:
    if backend not in {"auto", "decord", "opencv", "pyav"}:
        raise ValueError(f"unsupported video backend {backend!r}")
    if backend == "pyav":
        _, stream = _get_av_reader(path)
        if stream.average_rate is None or float(stream.average_rate) <= 0:
            raise RuntimeError(f"missing average frame rate for {path}")
        return _decode_rgb_at_timestamp(path, frame_idx / float(stream.average_rate), size_wh)
    width, height = size_wh if size_wh else (None, None)
    reader = _get_reader(str(path), width, height) if backend in {"auto", "decord"} else None
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
    if backend == "auto":
        try:
            _, stream = _get_av_reader(path)
            if stream.average_rate is not None and float(stream.average_rate) > 0:
                return _decode_rgb_at_timestamp(path, frame_idx / float(stream.average_rate), size_wh)
        except Exception:
            pass
    if backend == "decord":
        raise RuntimeError(f"decord could not open video {path}")

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

def _decode_rgb_at_timestamp(
    path: Path,
    timestamp_seconds: float,
    size_wh: tuple[int, int] | None,
) -> np.ndarray:
    if timestamp_seconds < 0:
        raise ValueError(f"timestamp_seconds must be non-negative, got {timestamp_seconds}")
    container, stream = _get_av_reader(path)
    time_base = float(stream.time_base)
    if time_base <= 0:
        raise RuntimeError(f"invalid video time base for {path}: {stream.time_base}")
    start_seconds = float((stream.start_time or 0) * stream.time_base)
    if timestamp_seconds < start_seconds:
        raise ValueError(f"timestamp {timestamp_seconds:.6f}s precedes video start {start_seconds:.6f}s")
    if stream.duration is not None:
        end_seconds = start_seconds + float(stream.duration * stream.time_base)
        if timestamp_seconds > end_seconds:
            raise ValueError(f"timestamp {timestamp_seconds:.6f}s exceeds video end {end_seconds:.6f}s")
    target_pts = int(timestamp_seconds / time_base)
    container.seek(max(target_pts, 0), stream=stream, backward=True)
    best_frame = None
    best_distance = float("inf")
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        seconds = float(frame.pts * stream.time_base)
        distance = abs(seconds - timestamp_seconds)
        if distance < best_distance:
            best_frame = frame
            best_distance = distance
        if seconds >= timestamp_seconds:
            break
    if best_frame is None:
        raise RuntimeError(f"could not decode timestamp {timestamp_seconds:.6f}s from {path}")
    array = best_frame.to_ndarray(format="rgb24")
    if size_wh is not None:
        image = Image.fromarray(array).resize(size_wh, Image.Resampling.BICUBIC)
        array = np.asarray(image)
    return array


def decode_video_timestamp(
    path: str | Path | None,
    timestamp_seconds: float,
    *,
    output_size: tuple[int, int] = (224, 320),
    decode_long_edge: int | None = DEFAULT_DECODE_LONG_EDGE,
    missing_ok: bool = True,
    backend: str = "pyav",
) -> Image.Image:
    if backend != "pyav":
        raise ValueError(f"timestamp decoding only supports backend='pyav', got {backend!r}")
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
        arr = _decode_rgb_at_timestamp(src, float(timestamp_seconds), decode_size)
        image = Image.fromarray(np.asarray(arr))
        return _resize_long_edge(image, decode_long_edge)
    except Exception as exc:
        if not missing_ok:
            raise RuntimeError(
                f"video timestamp decode failed: path={src} timestamp={timestamp_seconds:.6f} backend={backend}"
            ) from exc
        return _zero_image(output_size)


def decode_video_frame(
    path: str | Path | None,
    frame_idx: int,
    *,
    output_size: tuple[int, int] = (224, 320),
    decode_long_edge: int | None = DEFAULT_DECODE_LONG_EDGE,
    missing_ok: bool = True,
    backend: str = "auto",
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
        arr = _decode_rgb(src, int(frame_idx), decode_size, backend=backend)
        image = Image.fromarray(np.asarray(arr))
        return _resize_long_edge(image, decode_long_edge)
    except Exception as exc:
        if not missing_ok:
            raise RuntimeError(
                f"video frame decode failed: path={src} frame={frame_idx} backend={backend}"
            ) from exc
        return _zero_image(output_size)


def image_to_float_array(image: Image.Image) -> np.ndarray:
    return (np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).astype(np.float32)

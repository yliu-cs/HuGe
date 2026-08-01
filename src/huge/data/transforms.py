"""Shared preprocessing transforms for token, proprio, and image timelines."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from scipy.spatial.transform import Rotation as Rotation
from scipy.spatial.transform import Slerp


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    """Write an npy file via a same-directory temp file and ``os.replace``."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npy",
        prefix=f".{dst.name}.",
        dir=dst.parent,
        delete=False,
    ) as f:
        tmp = Path(f.name)
        np.save(f, array)
    os.replace(tmp, dst)


def resample_linear(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample continuous arrays along axis 0 with linear interpolation."""
    arr = np.asarray(values)
    if arr.ndim < 1:
        raise ValueError("values must have a time axis")
    if len(arr) == 0:
        return arr.astype(np.float32, copy=False)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    duration = (len(arr) - 1) / source_fps
    target_len = int(round(duration * target_fps)) + 1
    if target_len <= 1:
        return arr[:1].astype(np.float32, copy=False)
    source_t = np.arange(len(arr), dtype=np.float64) / source_fps
    target_t = np.arange(target_len, dtype=np.float64) / target_fps
    flat = arr.reshape(len(arr), -1).astype(np.float64)
    out = np.empty((target_len, flat.shape[1]), dtype=np.float64)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(target_t, source_t, flat[:, i])
    return out.reshape((target_len, *arr.shape[1:])).astype(np.float32)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    identity = np.zeros_like(q)
    identity[..., 0] = 1.0
    return np.where(norm > 1e-10, q / np.maximum(norm, 1e-10), identity)


def resample_quat_wxyz(quat_wxyz: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample WXYZ quaternions with sign alignment, lerp, and renormalization."""
    q = _normalize_quat_wxyz(quat_wxyz)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat_wxyz must be (T, 4), got {q.shape}")
    aligned = q.copy()
    for i in range(1, len(aligned)):
        if np.dot(aligned[i - 1], aligned[i]) < 0.0:
            aligned[i] *= -1.0
    return _normalize_quat_wxyz(resample_linear(aligned, source_fps, target_fps)).astype(np.float32)


def fixed_hz_target_seconds(source_seconds: np.ndarray, target_hz: float) -> np.ndarray:
    """Build Xperience-compatible target times over a timestamp span."""
    source = np.asarray(source_seconds, dtype=np.float64)
    if (
        source.ndim != 1
        or len(source) < 2
        or not np.isfinite(source).all()
        or not np.isclose(source[0], 0.0, atol=1e-9)
        or np.any(np.diff(source) <= 0)
    ):
        raise ValueError("source_seconds must be finite, strictly increasing, and start at zero")
    if not np.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("target_hz must be positive and finite")
    duration = float(source[-1])
    step = 1.0 / target_hz
    target = np.arange(0.0, duration + step * 0.5, step, dtype=np.float64)
    target[-1] = min(target[-1], duration)
    return target


def resample_linear_at_times(
    values: np.ndarray,
    source_seconds: np.ndarray,
    target_seconds: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate arrays along axis 0 using explicit timestamps."""
    source = np.asarray(source_seconds, dtype=np.float64)
    target = np.asarray(target_seconds, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if len(array) != len(source):
        raise ValueError(f"values/source length mismatch: {len(array)} != {len(source)}")
    if (
        target.ndim != 1
        or not len(target)
        or not np.isfinite(target).all()
        or np.any(np.diff(target) <= 0)
        or target[0] < source[0]
        or target[-1] > source[-1]
    ):
        raise ValueError("target_seconds must be finite, increasing, and inside source_seconds")
    flat = array.reshape(len(array), -1)
    result = np.empty((len(target), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(target, source, flat[:, column])
    return result.reshape((len(target), *array.shape[1:])).astype(np.float32)


def resample_quat_wxyz_at_times(
    quaternions_wxyz: np.ndarray,
    source_seconds: np.ndarray,
    target_seconds: np.ndarray,
) -> np.ndarray:
    """SLERP WXYZ quaternions shaped ``[T,...,4]`` at explicit timestamps."""
    source = np.asarray(source_seconds, dtype=np.float64)
    target = np.asarray(target_seconds, dtype=np.float64)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternions.ndim < 2 or quaternions.shape[-1] != 4:
        raise ValueError(f"quaternions_wxyz must be [T,...,4], got {quaternions.shape}")
    if len(quaternions) != len(source):
        raise ValueError(
            f"quaternion/source length mismatch: {len(quaternions)} != {len(source)}"
        )
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("zero quaternion in a preflight-approved segment")
    flat = (quaternions / norms).reshape(len(quaternions), -1, 4)
    result = np.empty((len(target), flat.shape[1], 4), dtype=np.float32)
    for item in range(flat.shape[1]):
        rotations = Rotation.from_quat(flat[:, item], scalar_first=True)
        result[:, item] = Slerp(source, rotations)(target).as_quat(scalar_first=True)
    return result.reshape((len(target), *quaternions.shape[1:]))


def projected_gravity_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Compute gravity direction in the base frame from world->base WXYZ quats."""
    q = _normalize_quat_wxyz(quat_wxyz)
    xyzw = np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)
    rot = Rotation.from_quat(xyzw.reshape(-1, 4))
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    projected = rot.inv().apply(np.repeat(gravity_world[None, :], len(xyzw.reshape(-1, 4)), axis=0))
    return projected.reshape((*q.shape[:-1], 3)).astype(np.float32)


def token_to_video_frame_index(token_idx: int, token_len: int, video_len: int) -> int:
    if token_len <= 0 or video_len <= 0:
        raise ValueError("token_len and video_len must be positive")
    return min(int(token_idx * video_len / token_len), video_len - 1)


def resize_short_center_crop(image: Image.Image, short_side: int = 224, size: tuple[int, int] = (224, 320)) -> Image.Image:
    """Resize by short side, then center-crop to ``(height, width)``."""
    height, width = size
    w, h = image.size
    scale = short_side / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    left = max((new_w - width) // 2, 0)
    top = max((new_h - height) // 2, 0)
    return resized.crop((left, top, left + width, top + height))


def _pad_to_square(image: Image.Image, fill: int = 0) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_top = (side - h) // 2
    pad_right = side - w - pad_left
    pad_bottom = side - h - pad_top
    return ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)


def _fractional_crop(image: Image.Image, crop_fraction: float, *, random_crop: bool) -> Image.Image:
    crop_fraction = float(crop_fraction)
    if crop_fraction <= 0.0 or crop_fraction >= 1.0:
        return image
    w, h = image.size
    crop_w = max(1, int(round(w * crop_fraction)))
    crop_h = max(1, int(round(h * crop_fraction)))
    if random_crop:
        left = random.randint(0, max(w - crop_w, 0))
        top = random.randint(0, max(h - crop_h, 0))
    else:
        left = max((w - crop_w) // 2, 0)
        top = max((h - crop_h) // 2, 0)
    return image.crop((left, top, left + crop_w, top + crop_h))


def _jitter_factor(amount: float) -> float:
    amount = max(float(amount), 0.0)
    return random.uniform(max(0.0, 1.0 - amount), 1.0 + amount)


def _apply_color_jitter(
    image: Image.Image,
    *,
    brightness: float = 0.3,
    contrast: float = 0.4,
    saturation: float = 0.5,
) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(_jitter_factor(brightness))
    image = ImageEnhance.Contrast(image).enhance(_jitter_factor(contrast))
    image = ImageEnhance.Color(image).enhance(_jitter_factor(saturation))
    return image


def gr00t_image_transform(
    image: Image.Image,
    *,
    train: bool,
    target_size: tuple[int, int] = (256, 256),
    crop_fraction: float = 0.95,
    color_jitter: bool = True,
) -> Image.Image:
    """GR00T-style single-view image preprocessing for Huge.

    The transform keeps the policy-side contract simple:
    train uses random fractional crop and color jitter, while eval uses the
    same geometry with a deterministic center crop and no jitter.
    """
    image = _pad_to_square(image)
    image = image.resize(target_size, Image.Resampling.BICUBIC)
    image = _fractional_crop(image, crop_fraction, random_crop=train)
    image = image.resize(target_size, Image.Resampling.BICUBIC)
    if train and color_jitter:
        image = _apply_color_jitter(image)
    return image

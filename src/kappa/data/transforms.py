"""Shared preprocessing transforms for token, proprio, and image timelines."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rotation


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

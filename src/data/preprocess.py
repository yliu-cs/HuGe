"""Shared utilities for SONIC token preprocessing scripts.

Common pieces used by both ``scripts/preprocess/humanoid-everyday/`` and
``scripts/preprocess/g1-wbt/`` (and any future source):

* :func:`resample_linear` / :func:`resample_quat_wxyz` — 30→50 Hz upsampling
  with bit-identical behavior across sources so the SONIC token distribution
  doesn't shift between datasets.
* :func:`atomic_save_npy` — write ``.npy`` via ``<final>.tmp.<pid>`` then
  rename, so interrupted runs never leave a half-written file that the
  resume logic would skip.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


def resample_linear(arr: np.ndarray, n_src: int, n_tgt: int) -> np.ndarray:
    """Linear interpolation along axis 0 from length ``n_src`` to ``n_tgt``.

    Preserves dtype. Identity when sizes match; broadcasts when ``n_src == 1``.
    """
    if n_tgt == n_src:
        return arr
    if n_src == 1:
        return np.repeat(arr, n_tgt, axis=0)
    pos = np.linspace(0.0, n_src - 1, n_tgt)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n_src - 1)
    frac = (pos - lo).reshape((n_tgt,) + (1,) * (arr.ndim - 1))
    out = (1.0 - frac) * arr[lo] + frac * arr[hi]
    return out.astype(arr.dtype, copy=False)


def resample_quat_wxyz(q: np.ndarray, n_src: int, n_tgt: int) -> np.ndarray:
    """Lerp + L2-renormalize WXYZ unit quaternions with sign canonicalization.

    SLERP would be exact, but at 30→50 Hz the per-step rotation is small
    (<<1 rad) so canonicalized lerp + renorm is within rounding. Sign-flip
    pass guarantees we don't take the long way around the hypersphere.
    """
    if n_tgt == n_src:
        return q
    q = q.astype(np.float64, copy=True)
    if n_src > 1:
        dots = np.sum(q[1:] * q[:-1], axis=-1)
        flips = np.cumprod(np.where(dots < 0.0, -1.0, 1.0))
        q[1:] *= flips[:, None]
    out = resample_linear(q, n_src, n_tgt)
    norms = np.linalg.norm(out, axis=-1, keepdims=True)
    safe = norms >= 1e-10
    normalized = np.divide(out, norms, out=np.zeros_like(out), where=safe)
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    out = np.where(safe, normalized, identity)
    return out.astype(np.float32, copy=False)


def atomic_save_npy(arr: Any, final_path: str) -> None:
    """Atomically write ``arr`` to ``final_path`` (a path ending in ``.npy``).

    Writes via ``<final>.tmp.<pid>`` and ``os.replace`` — racing ranks /
    interrupted runs never observe a half-written file. ``np.save`` is
    careful to honor a path with the ``.npy`` suffix already present (no
    double-suffix), so the temp path can carry it directly.
    """
    if not final_path.endswith(".npy"):
        raise ValueError(f"atomic_save_npy: final_path must end with .npy, got {final_path!r}")
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    tmp_path = f"{final_path}.tmp.{os.getpid()}"
    np.save(tmp_path, arr)
    # np.save appends .npy when the suffix is missing; tmp_path already
    # ends in ".tmp.<pid>" without .npy, so the actual file written is
    # tmp_path + ".npy". Locate whichever exists.
    written = tmp_path if os.path.exists(tmp_path) else f"{tmp_path}.npy"
    os.replace(written, final_path)

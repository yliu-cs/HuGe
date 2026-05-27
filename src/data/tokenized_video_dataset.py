"""Common base for SONIC-token + video-frame map-style datasets.

All three current sources (HE, EgoDex, G1-WBT) share the same shape:

* sampling space is 50 Hz SONIC tokens, one row per dataset index;
* each token row maps to a video frame in a 30 fps mp4;
* a sample returns ``{image, text, sonic_tokens, sonic_mask}`` plus an
  optional 64-D ``proprio`` vector when the subclass exposes a ``_proprio(ep)``
  hook.

The differences are surface-level — where the mp4 lives on disk, whether
many episodes are packed into one mp4 (LeRobot v3) or one mp4 per episode,
and (for EgoDex only) that token width is 64 not 66 so the trailing two
"hand-bit" dims need padding + a per-dim mask. This module factors out the
common index/sampling machinery; subclasses implement four hooks.

Subclass contract
-----------------
``__init__`` populates ``self._ep_lengths`` (int64 array of token-row count
per episode) plus any source-specific arrays the hooks need, then calls
``self._finalize_index()``. Hooks:

* ``_video_for(ep, t) -> (path, frame_idx)`` — resolve the mp4 path and
  the absolute frame index inside it.
* ``_text(ep) -> str`` — instruction string.
* ``_tokens(ep) -> np.ndarray`` — full episode token array (mmap fine).
* ``_build_chunk(ep, t, tokens) -> (chunk, sonic_mask)`` — defaults to
  clamp-tail repeat with a 1-D time mask; EgoDex overrides for 64→66
  padding + 2-D per-dim mask.
* ``_proprio(ep) -> np.ndarray | None`` (optional) — 64-D proprio array.

Proprio normalization
---------------------
When ``proprio_stats`` is passed to the constructor the dataset rescales
each dim by ``(lo, hi)`` so that ``lo -> -1`` and ``hi -> +1``. ``mode``
selects which stat keys back ``lo / hi``:

* ``"bounds"``        — ``min / max`` (no train-time saturation by design,
  but raw min/max is sensitive to outliers).
* ``"bounds_q99"``    — ``q01 / q99`` + ``np.clip(..., -1, 1)``. Mirrors
  Isaac-GR00T's default ``use_percentiles=True, clip_outliers=True`` path.
  The 1% tails on each side saturate at ``±1``.
* ``"bounds_q99_noclip"`` — ``q01 / q99``, **no clip**. Values outside the
  envelope are linearly extrapolated to ``|x| > 1`` and the model sees the
  real magnitude (e.g. PROP deep-squat knees keep their dynamic range
  instead of saturating). Matches GR00T's behavior when called with
  ``clip_outliers=False``; GR00T's own G1 finetunes default to clip on.

Degenerate dims (``min == max``) are skipped via ``ill_mask``: ``hi`` is
overwritten with ``lo + 1`` so the transform becomes ``(x - lo)`` and never
divides by zero.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Decoded frames are produced at this (W, H). decord/PyAV resize during
# decode so workers never materialize the native frame. 320×224 matches the
# downstream Resize+CenterCrop pipeline output exactly and both edges are
# multiples of patch_size×merge_size=32, so processor smart_resize is a no-op.
DEFAULT_FRAME_SIZE: tuple[int, int] = (320, 224)


# Per-worker LRU of mmap'd token arrays, shared across all subclasses.
# Keyed by absolute path. Each forked DataLoader worker gets its own dict
# (Linux fork = CoW; no cross-worker handle sharing).
_TOKEN_MMAP_CACHE_SIZE = 64
_TOKEN_MMAP_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def get_token_mmap(path: str) -> np.ndarray:
    """Return ``np.load(path, mmap_mode='r')`` cached per worker."""
    arr = _TOKEN_MMAP_CACHE.get(path)
    if arr is not None:
        _TOKEN_MMAP_CACHE.move_to_end(path)
        return arr
    arr = np.load(path, mmap_mode="r")
    _TOKEN_MMAP_CACHE[path] = arr
    if len(_TOKEN_MMAP_CACHE) > _TOKEN_MMAP_CACHE_SIZE:
        _TOKEN_MMAP_CACHE.popitem(last=False)
    return arr


# Modes that skip the post-rescale clamp. The same (lo, hi) come back as in
# the clipped variant; the dataset's normalize step picks behavior off this
# set so deploy / sim / eval can mirror the choice via the same string.
_NOCLIP_MODES = frozenset({"bounds_q99_noclip"})

# Maps a normalize mode string to the (lo_key, hi_key) pair used to pull
# bounds out of the 13-group stats JSON.
_MODE_TO_KEYS = {
    "bounds":            ("min", "max"),
    "bounds_q99":        ("q01", "q99"),
    "bounds_q99_noclip": ("q01", "q99"),
}


def _load_normalize_bounds(
    stats: dict | None, key: str, dim: int, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pre-compute (lo, hi, ill_mask) from a stats dict.

    Expects ``stats[key]`` in the 13-group schema written by
    ``scripts/preprocess/calc_g1_proprio_stats.py`` (HE+WBT) and
    ``scripts/preprocess/proprietary/calc_proprio_stats.py`` (proprietary):
    ``{<group>: {lo_key: [...], hi_key: [...]}, ...}``. The layout in
    ``src.data.proprio_layout`` concatenates the groups back to a flat
    ``(dim,)`` vector.

    See module docstring for the available modes. Returns ``None`` if
    ``stats`` is missing — datasets fall back to raw pass-through.
    """
    if stats is None:
        return None
    section = stats.get(key)
    if section is None:
        return None
    try:
        lo_key, hi_key = _MODE_TO_KEYS[mode]
    except KeyError as e:
        raise ValueError(f"unknown stats mode {mode!r}") from e
    from src.data.proprio_layout import assemble_flat_from_groups
    assembled = assemble_flat_from_groups(section, stat_keys=(lo_key, hi_key))
    lo = assembled[lo_key].astype(np.float32, copy=False)
    hi = assembled[hi_key].astype(np.float32, copy=False)
    if lo.shape != (dim,) or hi.shape != (dim,):
        raise ValueError(
            f"stats[{key}][{lo_key}/{hi_key}] expected ({dim},), got {lo.shape}/{hi.shape}"
        )
    # Mirror Psi0 ActionStateTransform: degenerate ranges (constant cols) get
    # bypassed by overwriting ``hi`` with 1.0 so the transform becomes (x-lo).
    eps = 1e-4
    ill_mask = np.abs(hi - lo) < eps * (np.abs(hi) + np.abs(lo) + 1e-8)
    hi_safe = np.where(ill_mask, lo + 1.0, hi)
    return lo, hi_safe, ill_mask


def _normalize_to_pm1(
    x: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    clip: bool = True,
) -> np.ndarray:
    """Rescale ``x`` (..., D) to ``[-1, 1]`` using ``(lo, hi, ill_mask)``.

    ``clip=True`` saturates values outside ``[lo, hi]`` at ``±1`` (the
    classic ``bounds`` / ``bounds_q99`` behavior). ``clip=False`` keeps the
    linear extrapolation so the model sees values beyond ``±1`` for tail
    samples — used by ``bounds_q99_noclip``.
    """
    if bounds is None:
        return x
    lo, hi, _ = bounds
    rng = hi - lo
    out = (x - lo) / rng * 2.0 - 1.0
    if clip:
        out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32, copy=False)


class TokenizedVideoDataset(Dataset):
    """Base class. See module docstring for the subclass contract."""

    # Subclass may override (e.g. EgoDex pads 64 → 66).
    sonic_dim: int = 66
    frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE

    # Populated by subclass __init__ before calling _finalize_index().
    _ep_lengths: np.ndarray  # int64 [N_ep], token-row count per episode

    # Proprio configuration. Defaults correspond to "proprio off" —
    # ``__getitem__`` only emits SONIC tokens unless the subclass turns it on
    # via the constructor / config.
    proprio_dim: int = 64
    _proprio_bounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    # Whether to clip post-rescale proprio to ``[-1, 1]``. Subclasses set this
    # off ``normalize_mode`` so ``bounds_q99_noclip`` lets tail values exceed
    # ±1 (real magnitude preserved); legacy ``bounds`` / ``bounds_q99`` keep
    # the historical clamp.
    _proprio_clip: bool = True

    def __init__(self, action_chunk_size: int) -> None:
        super().__init__()
        self.action_chunk_size = action_chunk_size

    # -- index ---------------------------------------------------------------

    def _finalize_index(self) -> None:
        """Compute cumulative offsets + total length from ``self._ep_lengths``."""
        n = self._ep_lengths
        if n.size == 0:
            self._ep_offsets = np.zeros(0, dtype=np.int64)
            self._total = 0
            return
        n64 = n.astype(np.int64, copy=False)
        self._ep_offsets = np.empty(n.size, dtype=np.int64)
        self._ep_offsets[0] = 0
        if n.size > 1:
            np.cumsum(n64[:-1], out=self._ep_offsets[1:])
        self._total = int(n64.sum())

    def __len__(self) -> int:
        return self._total

    def episode_spans(self) -> np.ndarray:
        """``[N, 2]`` int64 ``(start_idx, length)`` per episode (50 Hz tokens).

        Consumed by ``BlockShuffleSampler`` to chop the index space into
        episode-aligned contiguous runs.
        """
        out = np.empty((self._ep_offsets.size, 2), dtype=np.int64)
        out[:, 0] = self._ep_offsets
        out[:, 1] = self._ep_lengths.astype(np.int64, copy=False)
        return out

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        ep = int(np.searchsorted(self._ep_offsets, idx, side="right")) - 1
        return ep, idx - int(self._ep_offsets[ep])

    # -- subclass hooks ------------------------------------------------------

    def _video_for(self, ep: int, t: int) -> tuple[str, int]:
        raise NotImplementedError

    def _text(self, ep: int) -> str:
        raise NotImplementedError

    def _tokens(self, ep: int) -> np.ndarray:
        raise NotImplementedError

    def _build_chunk(
        self, ep: int, t: int, tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Default: last-row-repeat tail + 1-D time mask of length ``action_chunk_size``."""
        n = int(self._ep_lengths[ep])
        chunk_indices = [min(t + k, n - 1) for k in range(self.action_chunk_size)]
        chunk = tokens[chunk_indices]
        mask = np.array(
            [1.0 if t + k < n else 0.0 for k in range(self.action_chunk_size)],
            dtype=np.float32,
        )
        return chunk, mask

    def _proprio(self, ep: int) -> np.ndarray | None:
        """Optional hook for the 64-D proprio vector. Default disabled."""
        return None

    def _build_proprio_at(self, ep: int, t: int, proprio: np.ndarray) -> np.ndarray:
        """Single-row proprio at ``t`` (clamped to last frame), normalized."""
        n = int(self._ep_lengths[ep])
        idx = min(t, n - 1)
        raw = proprio[idx].astype(np.float32, copy=False)
        return _normalize_to_pm1(raw, self._proprio_bounds, clip=self._proprio_clip)

    # -- sampling ------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep, t = self._resolve(idx)
        text = self._text(ep)
        chunk, sonic_mask = self._build_chunk(ep, t, self._tokens(ep))
        video_path, frame_idx = self._video_for(ep, t)

        from src.data.video_reader import decode_frame

        frame = decode_frame(video_path, frame_idx, target_size=self.frame_size)
        out: dict[str, Any] = {
            "image": Image.fromarray(frame),
            "text": text,
            "sonic_tokens": torch.from_numpy(np.ascontiguousarray(chunk)),
            "sonic_mask": torch.from_numpy(np.ascontiguousarray(sonic_mask)),
        }

        # Optional per-tick proprio. Subclasses opt in by returning non-``None``
        # from the hook; the collator guards on key presence.
        proprio_arr = self._proprio(ep)
        if proprio_arr is not None:
            proprio_now = self._build_proprio_at(ep, t, proprio_arr)
            out["proprio"] = torch.from_numpy(np.ascontiguousarray(proprio_now))

        return out


def resolve_writable_cache_root(data_root: str, cache_root: str | None) -> str:
    """Common cache_root resolution: prefer ``data_root`` when writable, else
    fall back to a parallel ``<data_root>-cache`` directory (read-only HF
    dataset mounts can't host generated artifacts)."""
    if cache_root is not None:
        return cache_root
    return data_root if os.access(data_root, os.W_OK) else f"{data_root}-cache"

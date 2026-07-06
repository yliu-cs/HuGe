"""GR00T-compatible grouped state normalizer for Huge's 34-D proprio."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from huge.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT


@dataclass(frozen=True)
class StateNormConfig:
    use_percentiles: bool = True
    clip_outliers: bool = True
    eps: float = 1e-8


def _extract(raw: dict[str, Any]) -> dict[str, Any]:
    if "unitree_g1_sonic" in raw:
        return raw["unitree_g1_sonic"]["state"]
    if "state" in raw:
        return raw["state"]
    return raw


class StateNormalizer:
    """Per-group min/max normalizer matching GR00T dispatch for this schema.

    The current ``unitree_g1_sonic`` config has no sin/cos or mean/std groups,
    so every group uses q01/q99 or min/max bounds. The grouped implementation is
    kept so later embodiment configs can extend the dispatch without changing
    dataset code.
    """

    def __init__(self, stats: dict[str, Any], config: StateNormConfig | None = None) -> None:
        self.stats = _extract(stats)
        self.config = config or StateNormConfig()
        self._params = self._build_params()

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        use_percentiles: bool = True,
        clip_outliers: bool = True,
    ) -> "StateNormalizer":
        with Path(path).open("r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(stats, StateNormConfig(use_percentiles=use_percentiles, clip_outliers=clip_outliers))

    def _build_params(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        lo_key, hi_key = ("q01", "q99") if self.config.use_percentiles else ("min", "max")
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, start, end in PROPRIO_LAYOUT:
            section = self.stats.get(name)
            if section is None:
                raise KeyError(f"missing proprio stats group {name!r}")
            width = end - start
            lo = np.asarray(section[lo_key], dtype=np.float64)
            hi = np.asarray(section[hi_key], dtype=np.float64)
            if lo.shape != (width,) or hi.shape != (width,):
                raise ValueError(f"stats group {name} expected width {width}, got {lo.shape}/{hi.shape}")
            out[name] = lo, hi
        return out

    def normalize_group(self, name: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self._params[name]
        arr = np.asarray(x, dtype=np.float64)
        if self.config.clip_outliers:
            arr = np.clip(arr, lo, hi)
        out = 2.0 * (arr - lo) / (hi - lo + self.config.eps) - 1.0
        return out.astype(np.float32, copy=False)

    def denormalize_group(self, name: str, x: np.ndarray) -> np.ndarray:
        lo, hi = self._params[name]
        out = (np.asarray(x, dtype=np.float64) + 1.0) * 0.5 * (hi - lo) + lo
        return out.astype(np.float32, copy=False)

    def normalize_flat(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"proprio dim {arr.shape[-1]} != {PROPRIO_DIM}")
        parts = [self.normalize_group(name, arr[..., start:end]) for name, start, end in PROPRIO_LAYOUT]
        return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return self.normalize_flat(x)

    def denormalize_flat(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"proprio dim {arr.shape[-1]} != {PROPRIO_DIM}")
        parts = [self.denormalize_group(name, arr[..., start:end]) for name, start, end in PROPRIO_LAYOUT]
        return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

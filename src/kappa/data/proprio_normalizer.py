"""Percentile-bounds normalizer for the 34-D G1 SONIC proprio schema."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kappa.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT, PROPRIO_LAYOUT_VERSION


@dataclass(frozen=True)
class ProprioNormConfig:
    use_percentiles: bool = True
    clip_outliers: bool = True
    eps: float = 1e-8
    layout: str = PROPRIO_LAYOUT_VERSION

    def to_dict(self) -> dict[str, bool | float | str]:
        return {
            "use_percentiles": self.use_percentiles,
            "clip_outliers": self.clip_outliers,
            "eps": self.eps,
            "layout": self.layout,
        }


def _as_float_array(value: Any, expected: int, key: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (expected,):
        raise ValueError(f"stats key {key} has shape {arr.shape}, expected ({expected},)")
    return arr


def _extract_grouped_stats(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "unitree_g1_sonic" in raw:
        return raw["unitree_g1_sonic"]["state"]
    if "state" in raw and isinstance(raw["state"], dict):
        return raw["state"]
    return raw


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ProprioNormalizer:
    """Normalize/denormalize flat proprio arrays with q01/q99 percentile bounds."""

    def __init__(
        self,
        stats: dict[str, Any],
        config: ProprioNormConfig | None = None,
        *,
        stats_path: str | Path | None = None,
        stats_hash: str | None = None,
    ) -> None:
        self.config = config or ProprioNormConfig()
        self.stats_path = str(stats_path) if stats_path is not None else None
        self.stats_hash = stats_hash
        grouped = _extract_grouped_stats(stats)
        self._lo, self._hi = self._assemble_bounds(grouped)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        config: ProprioNormConfig | None = None,
    ) -> "ProprioNormalizer":
        stats_path = Path(path)
        with stats_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(raw, config=config, stats_path=stats_path, stats_hash=_sha256(stats_path))

    def _assemble_bounds(self, grouped: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        lo_key, hi_key = ("q01", "q99") if self.config.use_percentiles else ("min", "max")
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for name, start, end in PROPRIO_LAYOUT:
            width = end - start
            section = grouped.get(name)
            if section is None:
                raise KeyError(f"proprio stats missing group {name!r}")
            lows.append(_as_float_array(section[lo_key], width, f"{name}.{lo_key}"))
            highs.append(_as_float_array(section[hi_key], width, f"{name}.{hi_key}"))
        lo = np.concatenate(lows, axis=0)
        hi = np.concatenate(highs, axis=0)
        if lo.shape != (PROPRIO_DIM,) or hi.shape != (PROPRIO_DIM,):
            raise ValueError(f"assembled bounds have shapes {lo.shape}, {hi.shape}")
        return lo, hi

    @property
    def low(self) -> np.ndarray:
        return self._lo.copy()

    @property
    def high(self) -> np.ndarray:
        return self._hi.copy()

    def normalize(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"proprio dim {arr.shape[-1]} != {PROPRIO_DIM}")
        lo = self._lo
        hi = self._hi
        if self.config.clip_outliers:
            arr = np.clip(arr, lo, hi)
        out = 2.0 * (arr - lo) / (hi - lo + self.config.eps) - 1.0
        return out.astype(np.float32, copy=False)

    def denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        arr = np.asarray(x_norm, dtype=np.float64)
        if arr.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"proprio dim {arr.shape[-1]} != {PROPRIO_DIM}")
        out = (arr + 1.0) * 0.5 * (self._hi - self._lo) + self._lo
        return out.astype(np.float32, copy=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "proprio_layout": self.config.layout,
            "proprio_dim": PROPRIO_DIM,
            "proprio_stats_path": self.stats_path,
            "proprio_stats_hash": self.stats_hash,
            "proprio_use_percentiles": self.config.use_percentiles,
            "proprio_clip_outliers": self.config.clip_outliers,
        }


def _apply_sin_cos_encoding(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.sin(x), np.cos(x)], axis=-1)


class StateNormalizer:
    """GR00T-compatible group-dispatch state normalizer.

    Dispatch order per group is sin/cos -> mean/std -> percentile/minmax.
    Kappa's current ``unitree_g1_sonic v1`` config uses the percentile/minmax
    path for every group, but this class keeps the extensibility of the backup
    branch.
    """

    def __init__(
        self,
        statistics: dict[str, Any],
        *,
        modality_keys: tuple[str, ...] | None = None,
        sin_cos_embedding_keys: tuple[str, ...] | None = None,
        mean_std_embedding_keys: tuple[str, ...] | None = None,
        use_percentiles: bool = True,
        clip_outliers: bool = True,
        eps: float = 1e-8,
    ) -> None:
        self.grouped = _extract_grouped_stats(statistics)
        self.modality_keys = modality_keys or tuple(name for name, _, _ in PROPRIO_LAYOUT)
        self.sin_cos_embedding_keys = set(sin_cos_embedding_keys or ())
        self.mean_std_embedding_keys = set(mean_std_embedding_keys or ())
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.eps = eps
        self._params = self._compute_params()

    @classmethod
    def from_json(cls, path: str | Path, **kwargs: Any) -> "StateNormalizer":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls(json.load(f), **kwargs)

    def _compute_params(self) -> dict[str, dict[str, np.ndarray]]:
        params: dict[str, dict[str, np.ndarray]] = {}
        widths = {name: end - start for name, start, end in PROPRIO_LAYOUT}
        for name in self.modality_keys:
            section = self.grouped[name]
            p: dict[str, np.ndarray] = {}
            if "mean" in section and "std" in section:
                p["mean"] = _as_float_array(section["mean"], widths[name], f"{name}.mean")
                p["std"] = np.maximum(_as_float_array(section["std"], widths[name], f"{name}.std"), self.eps)
            lo_key, hi_key = ("q01", "q99") if self.use_percentiles and "q01" in section and "q99" in section else ("min", "max")
            p["low"] = _as_float_array(section[lo_key], widths[name], f"{name}.{lo_key}")
            p["range"] = np.maximum(_as_float_array(section[hi_key], widths[name], f"{name}.{hi_key}") - p["low"], self.eps)
            params[name] = p
        return params

    def apply_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, arr in state.items():
            x = np.asarray(arr, dtype=np.float64)
            p = self._params[name]
            if name in self.sin_cos_embedding_keys:
                y = _apply_sin_cos_encoding(x)
            elif name in self.mean_std_embedding_keys:
                y = (x - p["mean"]) / p["std"]
            else:
                y = (x - p["low"]) / p["range"] * 2.0 - 1.0
                if self.clip_outliers:
                    y = np.clip(y, -1.0, 1.0)
            out[name] = y.astype(np.float32, copy=False)
        return out

    def normalize_flat(
        self,
        x_flat: np.ndarray,
        layout: tuple[tuple[str, int, int], ...] = PROPRIO_LAYOUT,
    ) -> np.ndarray:
        offsets = {name: (start, end) for name, start, end in layout}
        grouped = {name: x_flat[..., offsets[name][0] : offsets[name][1]] for name in self.modality_keys}
        out = self.apply_state(grouped)
        return np.concatenate([out[name] for name in self.modality_keys], axis=-1).astype(np.float32, copy=False)

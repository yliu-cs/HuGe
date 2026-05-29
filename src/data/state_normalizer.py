"""State normalizer matching GR00T ``StateActionProcessor.apply_state``.

Per-group dispatch order (sin/cos -> mean/std -> min/max) mirrors
``Isaac-GR00T/gr00t/data/processors/state_action_processor.py:222-248``.
Degenerate columns are protected by ``range = max(max-min, 1e-8)`` exactly
as in the upstream ``_compute_normalization_parameters`` (lines 150-151).

The ``unitree_g1_sonic`` embodiment config used by Kappa does NOT populate
``sin_cos_embedding_keys`` or ``mean_std_embedding_keys``, so every group
takes the min/max + optional clip path; both alternative paths are
implemented for forward-compat and tested.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from src.data.modality_types import ModalityConfig


_STAT_KEYS_MINMAX = ("min", "max", "q01", "q99")
_STAT_KEYS_MEANSTD = ("mean", "std")


def _to_array(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


def _apply_sin_cos_encoding(x: np.ndarray) -> np.ndarray:
    """Concat ``[sin(x), cos(x)]`` along the last axis (GR00T semantics)."""
    return np.concatenate([np.sin(x), np.cos(x)], axis=-1)


class StateNormalizer:
    """GR00T-equivalent per-group state normalizer (numpy-only).

    Parameters
    ----------
    modality_configs
        ``{embodiment_tag: {"state": ModalityConfig, ...}}``.
    statistics
        Nested ``{embodiment_tag: {"state": {group_name: {stat_key: list}}}}``
        as written by the ``calc_*_proprio_stats.py`` scripts.
    use_percentiles
        ``True`` (GR00T N1.7 default) picks ``q01 / q99`` instead of ``min /
        max`` for the min/max path.
    clip_outliers
        Saturate the post-rescale min/max output to ``[-1, +1]``.
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: dict[str, dict[str, dict[str, dict[str, list]]]],
        use_percentiles: bool = True,
        clip_outliers: bool = True,
    ) -> None:
        self._modality_configs = modality_configs
        self._statistics = statistics
        self._use_percentiles = use_percentiles
        self._clip_outliers = clip_outliers
        self._norm_params = self._compute_normalization_parameters()

    # -- normalization parameter prep ---------------------------------------

    def _compute_normalization_parameters(self) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
        """Precompute (low, range, mean, std) per (embodiment, modality, group)."""
        out: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
        for emb_tag, modalities in self._modality_configs.items():
            emb_stats = self._statistics.get(emb_tag)
            if emb_stats is None:
                continue
            out[emb_tag] = {}
            for modality_name, modality_cfg in modalities.items():
                m_stats = emb_stats.get(modality_name)
                if m_stats is None:
                    continue
                out[emb_tag][modality_name] = {}
                for group_name in modality_cfg.modality_keys:
                    g = m_stats.get(group_name)
                    if g is None:
                        continue
                    params: dict[str, np.ndarray] = {}
                    if "min" in g and "max" in g:
                        if self._use_percentiles and "q01" in g and "q99" in g:
                            lo = _to_array(g["q01"])
                            hi = _to_array(g["q99"])
                        else:
                            lo = _to_array(g["min"])
                            hi = _to_array(g["max"])
                        rng = np.maximum(hi - lo, 1e-8)
                        params["low"] = lo
                        params["range"] = rng
                    if "mean" in g and "std" in g:
                        params["mean"] = _to_array(g["mean"])
                        params["std"] = np.maximum(_to_array(g["std"]), 1e-8)
                    out[emb_tag][modality_name][group_name] = params
        return out

    # -- internal per-group ops --------------------------------------------

    def _normalize_minmax(
        self, x: np.ndarray, params: dict[str, np.ndarray]
    ) -> np.ndarray:
        out = (x.astype(np.float64) - params["low"]) / params["range"] * 2.0 - 1.0
        if self._clip_outliers:
            out = np.clip(out, -1.0, 1.0)
        return out.astype(np.float32, copy=False)

    def _normalize_meanstd(
        self, x: np.ndarray, params: dict[str, np.ndarray]
    ) -> np.ndarray:
        out = (x.astype(np.float64) - params["mean"]) / params["std"]
        return out.astype(np.float32, copy=False)

    def _unnormalize_minmax(
        self, x: np.ndarray, params: dict[str, np.ndarray]
    ) -> np.ndarray:
        out = (x.astype(np.float64) + 1.0) * 0.5 * params["range"] + params["low"]
        return out.astype(np.float32, copy=False)

    def _unnormalize_meanstd(
        self, x: np.ndarray, params: dict[str, np.ndarray]
    ) -> np.ndarray:
        out = x.astype(np.float64) * params["std"] + params["mean"]
        return out.astype(np.float32, copy=False)

    # -- public API ---------------------------------------------------------

    def apply_state(
        self, state: dict[str, np.ndarray], embodiment_tag: str
    ) -> dict[str, np.ndarray]:
        """Normalize a state dict in place-style. Mirrors GR00T dispatch."""
        cfg = self._modality_configs[embodiment_tag]["state"]
        params = self._norm_params[embodiment_tag]["state"]
        sin_cos_keys = set(cfg.sin_cos_embedding_keys or ())
        mean_std_keys = set(cfg.mean_std_embedding_keys or ())
        out: dict[str, np.ndarray] = {}
        for group_name, arr in state.items():
            if group_name in sin_cos_keys:
                out[group_name] = _apply_sin_cos_encoding(arr).astype(np.float32, copy=False)
            elif group_name in mean_std_keys:
                out[group_name] = self._normalize_meanstd(arr, params[group_name])
            else:
                out[group_name] = self._normalize_minmax(arr, params[group_name])
        return out

    def unapply_state(
        self, state: dict[str, np.ndarray], embodiment_tag: str
    ) -> dict[str, np.ndarray]:
        """Inverse of ``apply_state`` for min/max + mean/std paths."""
        cfg = self._modality_configs[embodiment_tag]["state"]
        params = self._norm_params[embodiment_tag]["state"]
        sin_cos_keys = set(cfg.sin_cos_embedding_keys or ())
        mean_std_keys = set(cfg.mean_std_embedding_keys or ())
        out: dict[str, np.ndarray] = {}
        for group_name, arr in state.items():
            if group_name in sin_cos_keys:
                raise ValueError(
                    f"unapply_state on sin/cos group {group_name!r} is not invertible"
                )
            if group_name in mean_std_keys:
                out[group_name] = self._unnormalize_meanstd(arr, params[group_name])
            else:
                out[group_name] = self._unnormalize_minmax(arr, params[group_name])
        return out

    def normalize_flat(
        self,
        x_flat: np.ndarray,
        embodiment_tag: str,
        layout: tuple[tuple[str, int, int], ...],
    ) -> np.ndarray:
        """Slice a flat ``(..., D_in)`` vector by ``layout``, apply per-group
        normalization, then concat in ``modality_keys`` order.

        ``layout`` order does not have to match ``modality_keys`` order; the
        output respects ``modality_keys`` (= the canonical embodiment order).
        """
        cfg = self._modality_configs[embodiment_tag]["state"]
        offsets = {name: (s, e) for name, s, e in layout}
        for k in cfg.modality_keys:
            if k not in offsets:
                raise KeyError(
                    f"layout missing modality key {k!r} for embodiment "
                    f"{embodiment_tag!r}"
                )
        state = {k: x_flat[..., offsets[k][0] : offsets[k][1]] for k in cfg.modality_keys}
        out_groups = self.apply_state(state, embodiment_tag)
        parts = [out_groups[k] for k in cfg.modality_keys]
        return np.concatenate(parts, axis=-1)

    def output_dim(self, embodiment_tag: str) -> int:
        """Flat output width post-normalization (sin/cos doubles the group)."""
        cfg = self._modality_configs[embodiment_tag]["state"]
        params = self._norm_params[embodiment_tag]["state"]
        sin_cos_keys = set(cfg.sin_cos_embedding_keys or ())
        total = 0
        for k in cfg.modality_keys:
            p = params[k]
            base = int(p.get("low", p.get("mean")).shape[-1])
            total += base * 2 if k in sin_cos_keys else base
        return total

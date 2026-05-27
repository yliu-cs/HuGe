"""Proprietary fine-tune data config (multi-root).

Selects one or more LeRobot v2.1 dumps under ``resource/dataset/proprietary``
for fine-tuning. Each root must already have its SONIC token + proprio cache
materialized via ``scripts/preprocess/proprietary/calc_sonic_tokens.py``.

CLI:

    --data.data-roots resource/dataset/proprietary/real/2026-05-23 \
                      resource/dataset/proprietary/real/2026-05-16

A single root produces a plain :class:`PropDataset`. Multiple roots are
stitched with :class:`ConcatMixDataset`, mirroring ``HEWBTMixDataConfig`` so
``BlockShuffleSampler`` keeps within-episode locality on every source.
"""
from __future__ import annotations

import json
import os
from typing import Any

from pydantic import Field

from src.config.config import DataConfig


# Default to the latest real dump so ``./scripts/train/run_kappa.sh`` lands on
# a real-hand-bits fine-tune without per-stage data overrides. Older dumps
# (e.g. ``real/2026-05-16``) remain accessible via ``--data.data-roots``.
_DEFAULT_DATA_ROOT = "resource/dataset/proprietary/real/2026-05-23"


class PropDataConfig(DataConfig):
    dataset_name: str = "proprietary"
    # List form so the same config drives single- and multi-root fine-tunes.
    # Override on the CLI with ``--data.data-roots PATH1 PATH2``; tyro maps
    # ``list[str]`` to nargs='*'. The legacy single-root config (``data_root``)
    # is no longer accepted — pass a one-element list to keep the old behavior.
    data_roots: list[str] = Field(default_factory=lambda: [_DEFAULT_DATA_ROOT])
    action_chunk_size: int = 50

    # Optional 64-D robot proprio prefix; default on for fine-tune since the
    # post-train checkpoint already has the proprio embedder.
    emit_proprio: bool = True
    proprio_stats_path: str | None = None
    normalize_mode: str = "bounds_q99"

    # Restrict to a subset of episode indices for diagnostic runs. Applied
    # identically to every root in ``data_roots`` — episode index spaces are
    # not unified across roots, so passing e.g. ``[0, 1]`` selects the first
    # two episodes from each root.
    episode_indices: list[int] | None = None

    # Episode-level random subsample applied AFTER the per-root datasets are
    # concatenated. ``--data.max-samples N`` picks N random whole episodes
    # across the union of ``data_roots`` (deterministic via ``subsample_seed``);
    # every frame inside a picked episode is kept, so block-shuffle decode
    # locality is preserved exactly. ``None`` keeps every episode.
    max_samples: int | None = None
    subsample_seed: int = 0

    def _load_stats(self, path: str | None) -> dict | None:
        if path is None or not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import ConcatMixDataset, EpisodeSubsampleDataset, PropDataset
        from src.data.dataset import Dataset

        if not self.data_roots:
            raise ValueError("PropDataConfig.data_roots must contain at least one path")

        stats = self._load_stats(self.proprio_stats_path)
        raws = [
            PropDataset(
                data_root=root,
                action_chunk_size=self.action_chunk_size,
                split=split,
                cache_root=None,  # auto-resolved per-root inside PropDataset
                emit_proprio=self.emit_proprio,
                proprio_stats=stats,
                normalize_mode=self.normalize_mode,
                episode_indices=self.episode_indices,
            )
            for root in self.data_roots
        ]
        raw = raws[0] if len(raws) == 1 else ConcatMixDataset(raws)
        if self.max_samples is not None:
            raw = EpisodeSubsampleDataset(
                raw, n_episodes=self.max_samples, seed=self.subsample_seed,
            )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

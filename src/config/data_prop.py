"""Proprietary 2026-05-16 data config."""
from __future__ import annotations

import json
import os
from typing import Any

from src.config.config import DataConfig


class PropDataConfig(DataConfig):
    dataset_name: str = "proprietary_2026_05_16"
    data_root: str = "resource/dataset/proprietary/2026-05-16"
    cache_root: str | None = None  # auto-resolved at runtime
    action_chunk_size: int = 50

    # Optional 64-D robot proprio prefix; default on for fine-tune since the
    # post-train checkpoint already has the proprio embedder.
    emit_proprio: bool = True
    proprio_stats_path: str | None = None
    normalize_mode: str = "bounds_q99"

    # Restrict to a subset of episodes for diagnostic runs. ``None`` keeps all
    # 60 proprietary episodes.
    episode_indices: list[int] | None = None

    def _load_stats(self, path: str | None) -> dict | None:
        if path is None or not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import PropDataset
        from src.data.dataset import Dataset

        raw = PropDataset(
            data_root=self.data_root,
            action_chunk_size=self.action_chunk_size,
            split=split,
            cache_root=self.cache_root,
            emit_proprio=self.emit_proprio,
            proprio_stats=self._load_stats(self.proprio_stats_path),
            normalize_mode=self.normalize_mode,
            episode_indices=self.episode_indices,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

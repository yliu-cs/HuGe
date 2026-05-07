"""G1-WBT data config."""
from __future__ import annotations

import json
import os
from typing import Any

from src.config.config import DataConfig


class WBTDataConfig(DataConfig):
    dataset_name: str = "g1_wbt"
    data_root: str = "resource/dataset/G1_WBT"
    cache_root: str | None = None
    action_chunk_size: int = 50
    # "all" or list of subset directory names. Restrict to e.g. only Inspire
    # variants by passing the relevant subset names; default uses everything
    # produced by calc_wbt_sonic_tokens.py.
    subsets: str | list[str] = "all"

    emit_g1_actions: bool = False
    emit_proprio: bool = False
    g1_action_stats_path: str | None = None
    g1_proprio_stats_path: str | None = None
    normalize_mode: str = "bounds"

    def _load_stats(self, path: str | None) -> dict | None:
        if path is None or not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import WBTDataset
        from src.data.dataset import Dataset

        raw = WBTDataset(
            data_root=self.data_root,
            action_chunk_size=self.action_chunk_size,
            cache_root=self.cache_root,
            subsets=self.subsets,
            emit_g1_actions=self.emit_g1_actions,
            emit_proprio=self.emit_proprio,
            g1_action_stats=self._load_stats(self.g1_action_stats_path),
            g1_proprio_stats=self._load_stats(self.g1_proprio_stats_path),
            normalize_mode=self.normalize_mode,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

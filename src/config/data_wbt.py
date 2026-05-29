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

    emit_proprio: bool = False
    proprio_stats_path: str | None = None
    use_percentiles: bool = True
    clip_outliers: bool = True

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
            emit_proprio=self.emit_proprio,
            stats=self._load_stats(self.proprio_stats_path),
            use_percentiles=self.use_percentiles,
            clip_outliers=self.clip_outliers,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

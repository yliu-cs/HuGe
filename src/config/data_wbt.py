"""G1-WBT data config."""
from __future__ import annotations

from typing import Any

from src.config.config import DataConfig


class WBTDataConfig(DataConfig):
    dataset_name: str = "g1_wbt"
    data_root: str = "resource/dataset/G1-WBT"
    cache_root: str | None = None
    action_chunk_size: int = 50
    # "all" or list of subset directory names. Restrict to e.g. only Inspire
    # variants by passing the relevant subset names; default uses everything
    # produced by calc_wbt_sonic_tokens.py.
    subsets: str | list[str] = "all"

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import WBTDataset
        from src.data.dataset import Dataset

        raw = WBTDataset(
            data_root=self.data_root,
            action_chunk_size=self.action_chunk_size,
            cache_root=self.cache_root,
            subsets=self.subsets,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

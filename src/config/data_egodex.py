"""EgoDex data config."""
from __future__ import annotations

from typing import Any

from src.config.config import DataConfig


class EgoDexDataConfig(DataConfig):
    dataset_name: str = "egodex"
    data_root: str = "resource/dataset/EgoDex/egodex_repacked_simple"
    cache_root: str | None = None  # auto-resolved at runtime to writable parallel dir
    action_chunk_size: int = 50
    sonic_dim: int = 66
    # Optional cap on # episodes loaded (for fast smoke tests). None = use all.
    max_episodes: int | None = None
    # Number of threads to probe `token_state.npy` shape during the first index
    # build. The cached JSON makes subsequent runs instant.
    index_workers: int = 32

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import EgoDexDataset
        from src.data.dataset import Dataset

        raw = EgoDexDataset(
            data_root=self.data_root,
            action_chunk_size=self.action_chunk_size,
            split=split,
            cache_root=self.cache_root,
            sonic_dim=self.sonic_dim,
            max_episodes=self.max_episodes,
            index_workers=self.index_workers,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

"""Humanoid-Everyday data config."""
from __future__ import annotations

from typing import Any

from src.config.config import DataConfig


class HEDataConfig(DataConfig):
    dataset_name: str = "humanoid_everyday"
    data_root: str = "resource/dataset/humanoid-everyday"
    cache_root: str | None = None  # auto-resolved at runtime to writable parallel dir
    action_chunk_size: int = 50
    # humanoid-everyday is G1+H1 mixed; the SONIC token pipeline assumes one
    # embodiment, so default to G1. Pass "all" to disable the filter.
    robot_type: str = "g1"

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import HEDataset
        from src.data.dataset import Dataset

        raw = HEDataset(
            data_root=self.data_root,
            action_chunk_size=self.action_chunk_size,
            split=split,
            cache_root=self.cache_root,
            robot_type=self.robot_type,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

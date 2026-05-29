"""Mixed data configs (multi-source training)."""
from __future__ import annotations

from typing import Any

from pydantic import Field

from src.config.config import DataConfig
from src.config.data_he import HEDataConfig
from src.config.data_wbt import WBTDataConfig


class HEWBTMixDataConfig(DataConfig):
    """HE-G1 + G1-WBT mix.

    Both child datasets emit identical ``{image, text, sonic_tokens [50,66],
    sonic_mask [50]}`` dicts and share the same transform pipeline, so a single
    ``Dataset(ConcatMixDataset(...), transform)`` is all the trainer needs.
    Sampling is the natural frame ratio (HE-G1 ~1.7M : WBT ~2.48M), and
    BlockShuffle keeps within-episode locality on both sources via
    ``ConcatMixDataset.episode_spans``.
    """
    dataset_name: str = "he_wbt_mix"
    he: HEDataConfig = Field(default_factory=lambda: HEDataConfig(robot_type="g1"))
    wbt: WBTDataConfig = Field(default_factory=WBTDataConfig)

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import ConcatMixDataset, HEDataset, WBTDataset
        from src.data.dataset import Dataset

        raw_he = HEDataset(
            data_root=self.he.data_root,
            action_chunk_size=self.he.action_chunk_size,
            split=split,
            cache_root=self.he.cache_root,
            robot_type=self.he.robot_type,
            emit_proprio=self.he.emit_proprio,
            stats=self.he._load_stats(self.he.proprio_stats_path),
            use_percentiles=self.he.use_percentiles,
            clip_outliers=self.he.clip_outliers,
        )
        raw_wbt = WBTDataset(
            data_root=self.wbt.data_root,
            action_chunk_size=self.wbt.action_chunk_size,
            cache_root=self.wbt.cache_root,
            subsets=self.wbt.subsets,
            emit_proprio=self.wbt.emit_proprio,
            stats=self.wbt._load_stats(self.wbt.proprio_stats_path),
            use_percentiles=self.wbt.use_percentiles,
            clip_outliers=self.wbt.clip_outliers,
        )
        mix = ConcatMixDataset([raw_he, raw_wbt])
        return Dataset(mix, self.transform, transform_kwargs=transform_kwargs)

"""Mixed data configs (multi-source training)."""
from __future__ import annotations

from typing import Any

from pydantic import Field

from src.config.config import DataConfig
from src.config.data_he import HEDataConfig
from src.config.data_wbt import WBTDataConfig


class HEWBTMixDataConfig(DataConfig):
    """Stage 1-2 mix: Humanoid-Everyday G1 + G1-WBT.

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
        )
        raw_wbt = WBTDataset(
            data_root=self.wbt.data_root,
            action_chunk_size=self.wbt.action_chunk_size,
            cache_root=self.wbt.cache_root,
            subsets=self.wbt.subsets,
        )
        mix = ConcatMixDataset([raw_he, raw_wbt])
        return Dataset(mix, self.transform, transform_kwargs=transform_kwargs)

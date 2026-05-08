"""Humanoid-Everyday data config."""
from __future__ import annotations

import json
import os
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

    # Post-train extras. Default off so existing pretrain configs are byte-
    # identical to before this change.
    emit_g1_actions: bool = False
    emit_proprio: bool = False
    g1_action_stats_path: str | None = None
    g1_proprio_stats_path: str | None = None
    normalize_mode: str = "bounds"

    # Restrict the dataset to a fixed list of HE episode indices. ``None``
    # keeps the full split; used by fine-tune configs that overfit a single
    # episode for diagnostic / sanity-check runs.
    episode_indices: list[int] | None = None

    def _load_stats(self, path: str | None) -> dict | None:
        if path is None or not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import HEDataset
        from src.data.dataset import Dataset

        raw = HEDataset(
            data_root=self.data_root,
            action_chunk_size=self.action_chunk_size,
            split=split,
            cache_root=self.cache_root,
            robot_type=self.robot_type,
            emit_g1_actions=self.emit_g1_actions,
            emit_proprio=self.emit_proprio,
            g1_action_stats=self._load_stats(self.g1_action_stats_path),
            g1_proprio_stats=self._load_stats(self.g1_proprio_stats_path),
            normalize_mode=self.normalize_mode,
            episode_indices=self.episode_indices,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

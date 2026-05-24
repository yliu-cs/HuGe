"""Proprietary ego (human first-person) fine-tune data config.

Selects one or more capture-date roots under
``resource/dataset/proprietary/ego`` for EgoDex-style fine-tuning. Each root
is a dump of human first-person trajectories with pre-materialized 66-D
SONIC latent tokens (``trackingData_<TS>_latent_tokens_66d.npy``); no robot
proprio is available, so the data is consumed exactly the way stage-1
pre-train consumes EgoDex (image + tokens + text, no proprio, no RTC).

CLI examples::

    # default single capture date
    ./scripts/train/train_kappa.sh finetune_prop_ego_kappa_config

    # multi-root concat
    ./scripts/train/train_kappa.sh finetune_prop_ego_kappa_config \\
        --data.data-roots resource/dataset/proprietary/ego/2026-5-23 \\
                          resource/dataset/proprietary/ego/2026-6-01
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from src.config.config import DataConfig


_DEFAULT_DATA_ROOT = "resource/dataset/proprietary/ego/2026-5-23"


class PropEgoDataConfig(DataConfig):
    dataset_name: str = "proprietary_ego"
    # List form so the same config drives single- and multi-root fine-tunes.
    # Override on the CLI with ``--data.data-roots PATH1 PATH2``; tyro maps
    # ``list[str]`` to nargs='*'.
    data_roots: list[str] = Field(default_factory=lambda: [_DEFAULT_DATA_ROOT])
    action_chunk_size: int = 50
    sonic_dim: int = 66

    # Optional cap on # episodes loaded per root (for fast smoke tests).
    max_episodes: int | None = None
    # Number of threads used to probe ``*_latent_tokens_66d.npy`` shape during
    # the first index build per root. The cached JSON makes subsequent runs
    # instant.
    index_workers: int = 32

    # Episode-level random subsample applied AFTER per-root concat. Mirrors
    # ``PropDataConfig.max_samples`` — ``--data.max-samples N`` picks N random
    # whole episodes across the union of ``data_roots``.
    max_samples: int | None = None
    subsample_seed: int = 0

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import ConcatMixDataset, EpisodeSubsampleDataset, PropEgoDataset
        from src.data.dataset import Dataset

        if not self.data_roots:
            raise ValueError("PropEgoDataConfig.data_roots must contain at least one path")

        raws = [
            PropEgoDataset(
                data_root=root,
                action_chunk_size=self.action_chunk_size,
                split=split,
                cache_root=None,  # auto-resolved per-root inside PropEgoDataset
                sonic_dim=self.sonic_dim,
                max_episodes=self.max_episodes,
                index_workers=self.index_workers,
            )
            for root in self.data_roots
        ]
        raw = raws[0] if len(raws) == 1 else ConcatMixDataset(raws)
        if self.max_samples is not None:
            raw = EpisodeSubsampleDataset(
                raw, n_episodes=self.max_samples, seed=self.subsample_seed,
            )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)

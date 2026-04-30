"""Kappa config package.

Re-exports the most commonly used configs so callers can write
``from src.config import HEDataConfig`` without knowing the per-source module
layout. New data sources should add a ``data_<source>.py`` and re-export
their config here.
"""
from src.config.config import (
    DataConfig,
    LaunchConfig,
    LoggingConfig,
    ModelConfig,
    TrainConfig,
)
from src.config.data_egodex import EgoDexDataConfig
from src.config.data_he import HEDataConfig
from src.config.data_mix import HEWBTMixDataConfig
from src.config.data_wbt import WBTDataConfig

__all__ = [
    "DataConfig",
    "EgoDexDataConfig",
    "HEDataConfig",
    "HEWBTMixDataConfig",
    "LaunchConfig",
    "LoggingConfig",
    "ModelConfig",
    "TrainConfig",
    "WBTDataConfig",
]

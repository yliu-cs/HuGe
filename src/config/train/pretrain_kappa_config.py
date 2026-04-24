"""Pretrain config for Kappa."""
from src.config.config import (
    LaunchConfig,
    TrainConfig,
    HEDataConfig,
)
from src.config.model_kappa import KappaModelConfig


class DynamicDataConfig(HEDataConfig):
    """Data config for Kappa pretraining with Humanoid-Everyday."""
    pass


class DynamicLaunchConfig(LaunchConfig):
    """Launch config for Kappa pretraining."""
    model: KappaModelConfig
    data: DynamicDataConfig

    train: TrainConfig = TrainConfig(
        name="kappa",
        max_training_steps=10000,
        train_batch_size=16,
        num_workers=0,
        checkpointing_steps=5000,
        validation_steps=1000,
        mixed_precision="bf16",
        learning_rate=1e-5,
    )

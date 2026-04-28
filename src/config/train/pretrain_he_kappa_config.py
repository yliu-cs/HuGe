"""Pretrain config for Kappa on Humanoid-Everyday."""
from pydantic import Field

from src.config.config import (
    HEDataConfig,
    LaunchConfig,
    TrainConfig,
)
from src.config.model_kappa import KappaModelConfig
from src.data.transform import (
    DataTransform,
    HERepackTransform,
    IdentityFieldTransform,
    KappaModelTransform,
)


class DynamicDataTransform(DataTransform):
    repack: HERepackTransform = Field(default_factory=HERepackTransform)
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class DynamicDataConfig(HEDataConfig):
    transform: DynamicDataTransform = Field(default_factory=DynamicDataTransform)


class DynamicLaunchConfig(LaunchConfig):
    model: KappaModelConfig
    data: DynamicDataConfig = Field(default_factory=DynamicDataConfig)

    train: TrainConfig = TrainConfig(
        name="kappa",
        max_training_steps=1000000,
        train_batch_size=96,
        gradient_accumulation_steps=1,
        num_workers=12,
        # HE has GOP=250 H.264 — random seek is the dominant data cost
        # (66 ms/seek measured). Block-shuffling to 1 batch ≈ 1 episode
        # converts these into next-frame decodes (~22× per-sample speedup).
        # Tokens are 50 Hz, video stays at 30 Hz; consecutive token rows
        # land on the same video frame ~40% of the time, so frame_cache_size=2
        # gives a real (small) hit rate.
        block_shuffle_size=64,
        frame_cache_size=2,
        checkpointing_steps=10000,
        max_checkpoints_to_keep=5,
        validation_steps=0,
        mixed_precision="bf16",
        learning_rate=3e-6,
        warmup_steps=500,
        warmup_ratio=0,
    )

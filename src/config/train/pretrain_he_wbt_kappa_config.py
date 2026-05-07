"""Pretrain config for Kappa Stage 1-2 (mixed): HE-G1 + G1-WBT.

Mirrors ``pretrain_he_kappa_config`` but swaps in ``HEWBTMixDataConfig`` so a
single trainer step draws from both sources at the natural frame ratio
(HE-G1 ~1.7M : WBT ~2.48M ≈ 1 : 1.46). BlockShuffle still works because
``ConcatMixDataset`` exposes a unified ``episode_spans`` over the concatenated
index space.
"""
from pydantic import Field

from src.config.config import LaunchConfig, TrainConfig
from src.config.data_mix import HEWBTMixDataConfig
from src.config.model_kappa import KappaModelConfig
from src.data.transform import (
    DataTransform,
    HERepackTransform,
    IdentityFieldTransform,
    KappaModelTransform,
)


class HEWBTDataTransform(DataTransform):
    # HEDataset and WBTDataset emit the same dict schema; the HE-style repack
    # transform is a no-op for already-tensorized inputs and applies cleanly to
    # both sources.
    repack: HERepackTransform = Field(default_factory=HERepackTransform)
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class HEWBTDataConfig(HEWBTMixDataConfig):
    transform: HEWBTDataTransform = Field(default_factory=HEWBTDataTransform)


class HEWBTLaunchConfig(LaunchConfig):
    model: KappaModelConfig
    data: HEWBTDataConfig = Field(default_factory=HEWBTDataConfig)

    train: TrainConfig = TrainConfig(
        name="kappa",
        num_train_epochs=1,
        train_batch_size=128,
        gradient_accumulation_steps=1,
        num_workers=8,
        prefetch_factor=2,
        # Same block_shuffle rationale as the pure HE config: random seek on
        # GOP-heavy mp4s (HE H.264 GOP=250, WBT AV1) is the dominant data cost,
        # and one-batch-per-episode access turns it into next-frame decodes.
        block_shuffle_size=64,
        frame_cache_size=2,
        checkpointing_steps=3000,
        max_checkpoints_to_keep=5,
        validation_steps=0,
        mixed_precision="bf16",
        learning_rate=5e-5,
        warmup_steps=0,
        warmup_ratio=0.01,
    )


LaunchConfig = HEWBTLaunchConfig

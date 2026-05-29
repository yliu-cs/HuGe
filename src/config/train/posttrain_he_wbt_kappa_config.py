"""Post-Train config for Kappa: SONIC tokens on HE-G1 + G1-WBT.

Inherits the EgoDex pretrain checkpoint and continues SONIC-token training on
the HE+WBT mix at the natural frame ratio (HE-G1 ~1.7M : WBT ~2.48M ≈ 1 :
1.46). Proprio (64-D robot joint state) is fed into the latent prefix as an
optional conditioning signal.
"""
from pydantic import Field

from src.config.config import LaunchConfig, TrainConfig
from src.config.data_he import HEDataConfig
from src.config.data_mix import HEWBTMixDataConfig
from src.config.data_wbt import WBTDataConfig
from src.config.model_kappa import KappaModelConfig
from src.data.transform import (
    DataTransform,
    HERepackTransform,
    IdentityFieldTransform,
    KappaModelTransform,
)


# HE+WBT combined proprio stats in the nested ``{"unitree_g1_sonic": {
# "state": ...}}`` schema, written by
# ``scripts/preprocess/calc_g1_proprio_stats.py``. Lives next to the WBT
# cache; the ``g1_`` prefix is kept because the vector describes the G1
# *robot's* joint state. Trainer.save_checkpoint copies this JSON into each
# ckpt_<step>/ so deploy / sim / eval can auto-discover it.
_DEFAULT_PROPRIO_STATS = "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json"


class PostTrainHEWBTDataTransform(DataTransform):
    # HEDataset and WBTDataset emit the same dict schema; the HE-style repack
    # transform is a no-op for already-tensorized inputs and applies cleanly to
    # both sources.
    repack: HERepackTransform = Field(default_factory=HERepackTransform)
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class PostTrainHEWBTDataConfig(HEWBTMixDataConfig):
    dataset_name: str = "he_wbt_posttrain"
    transform: PostTrainHEWBTDataTransform = Field(
        default_factory=PostTrainHEWBTDataTransform,
    )
    he: HEDataConfig = Field(
        default_factory=lambda: HEDataConfig(
            robot_type="g1",
            emit_proprio=True,
            proprio_stats_path=_DEFAULT_PROPRIO_STATS,
            use_percentiles=True,
            clip_outliers=True,
        )
    )
    wbt: WBTDataConfig = Field(
        default_factory=lambda: WBTDataConfig(
            emit_proprio=True,
            proprio_stats_path=_DEFAULT_PROPRIO_STATS,
            use_percentiles=True,
            clip_outliers=True,
        )
    )


class PostTrainHEWBTLaunchConfig(LaunchConfig):
    model: KappaModelConfig = Field(
        default_factory=lambda: KappaModelConfig(
            use_proprio=True,
            proprio_stats_path=_DEFAULT_PROPRIO_STATS,
        )
    )
    data: PostTrainHEWBTDataConfig = Field(
        default_factory=PostTrainHEWBTDataConfig,
    )

    train: TrainConfig = TrainConfig(
        name="kappa",
        num_train_epochs=2,
        train_batch_size=128,
        gradient_accumulation_steps=1,
        num_workers=8,
        prefetch_factor=2,
        # Same block_shuffle rationale as the pure HE config: random seek on
        # GOP-heavy mp4s (HE H.264 GOP=250, WBT AV1) is the dominant data cost,
        # and one-batch-per-episode access turns it into next-frame decodes.
        block_shuffle_size=64,
        frame_cache_size=2,
        checkpointing_steps=10000,
        max_checkpoints_to_keep=5,
        validation_steps=0,
        mixed_precision="bf16",
        learning_rate=3e-5,
        warmup_steps=0,
        warmup_ratio=0.01,
    )


LaunchConfig = PostTrainHEWBTLaunchConfig

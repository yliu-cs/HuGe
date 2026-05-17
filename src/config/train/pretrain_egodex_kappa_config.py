"""Pretrain config for Kappa on EgoDex.

EgoDex is the first stage of pretraining: SONIC tokens are 64-D (no hand-binary
bits). The dataset right-pads tokens to ``sonic_dim=66`` and emits a per-dim
``sonic_mask`` that zeros out the trailing two hand-bit dims, so the loss
ignores them. RTC and proprio are off at this stage.

Stage 2 (HE-G1 + G1-WBT Post-Train) launches via
``posttrain_he_wbt_kappa_config`` with ``--train.pretrained-weights`` pointing
at this run's ``model.safetensors`` (file, ``ckpt_*`` dir, or run dir — latest
ckpt auto-picked). Optimizer, scheduler, and global_step start fresh; weights
are loaded with ``strict=False`` so the architectures must already match
(they do: both share ``KappaModelConfig`` with ``sonic_dim=66`` — the
Post-Train just additionally enables ``use_rtc`` / ``use_proprio``, which
adds new parameters that initialize from scratch).
"""
from pydantic import Field

from src.config.config import LaunchConfig, TrainConfig
from src.config.data_egodex import EgoDexDataConfig
from src.config.model_kappa import KappaModelConfig
from src.data.transform import (
    DataTransform,
    EgodexRepackTransform,
    IdentityFieldTransform,
    KappaModelTransform,
)


class EgoDexDataTransform(DataTransform):
    repack: EgodexRepackTransform = Field(default_factory=EgodexRepackTransform)
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class EgoDexDataConfig(EgoDexDataConfig):
    transform: EgoDexDataTransform = Field(default_factory=EgoDexDataTransform)


class EgoDexLaunchConfig(LaunchConfig):
    model: KappaModelConfig
    data: EgoDexDataConfig = Field(default_factory=EgoDexDataConfig)

    # Effective global batch = train_batch_size × world_size × grad_accum = 64
    # on 1 GPU and 512 on 8 GPUs (same as the previous (64, 1) config on 8 GPUs,
    # but split into micro-batches so a single 24/48 GB GPU can fit one sample
    # at a time without OOM).
    train: TrainConfig = TrainConfig(
        name="kappa",
        num_train_epochs=1,
        train_batch_size=128,
        gradient_accumulation_steps=1,
        num_workers=8,
        prefetch_factor=2,
        # Episode-aware block shuffle: one batch ≈ one episode → decord
        # sequentializes per-worker decode (~21× per-sample speedup measured
        # on this dataset). frame_cache absorbs ~38% of decodes that 50→30Hz
        # token-to-video mapping reuses across adjacent token rows.
        block_shuffle_size=64,
        frame_cache_size=2,
        checkpointing_steps=30000,
        max_checkpoints_to_keep=5,
        validation_steps=0,
        mixed_precision="bf16",
        learning_rate=5e-5,
        warmup_steps=0,
        warmup_ratio=0.01,
    )


LaunchConfig = EgoDexLaunchConfig

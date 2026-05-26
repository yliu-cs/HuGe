"""Fine-tune Kappa on proprietary first-person ("ego") human captures.

This dataset is human first-person video with pre-materialized 66-D SONIC
latent tokens but **no robot proprio state** (the wearer is a person, not a
G1). It is therefore consumed exactly the way stage-1 Pre-Train consumes
EgoDex: image + SONIC tokens + text only, no proprio, no RTC. The EgoDex
recipe — zero + mask the trailing 2 hand-binary dims — is applied at the
``PropEgoDataset._build_chunk`` layer so the loss never sees the offline
thresholded hand-close signal.

Selecting training data
-----------------------

The CLI exposes ``--data.data-roots`` (``list[str]``, nargs='*'). Default is
the ``2026-5-23`` capture day (124 trajectories of
``Grab_the_cola_throw_into_the_bin``). Override to point at other capture
dates or concat several::

    # default single capture date
    ./scripts/train/train_kappa.sh finetune_prop_ego_kappa_config

    # different capture date
    ./scripts/train/train_kappa.sh finetune_prop_ego_kappa_config \\
        --data.data-roots resource/dataset/proprietary/ego/2026-6-01

    # mix two capture dates
    ./scripts/train/train_kappa.sh finetune_prop_ego_kappa_config \\
        --data.data-roots resource/dataset/proprietary/ego/2026-5-23 \\
                          resource/dataset/proprietary/ego/2026-6-01

Checkpoint seeding
------------------

This config does **not** load a default checkpoint — the human ego data is
incompatible with any post-train run that conditions on proprio. Use it
either as a continuation of the EgoDex pre-train run (``--train.pretrained-
weights .runs/kappa/<pretrain-egodex-run>``) or as a standalone training
from scratch.
"""
from __future__ import annotations

from pydantic import Field

from src.config.config import LaunchConfig, TrainConfig
from src.config.data_prop_ego import PropEgoDataConfig
from src.config.model_kappa import KappaModelConfig
from src.data.transform import (
    DataTransform,
    EgodexRepackTransform,
    IdentityFieldTransform,
    KappaModelTransform,
)


class FineTunePropEgoRepackTransform(EgodexRepackTransform):
    # Same repack pipeline (image -> observations list, copy text/sonic_*),
    # only the dataset tag changes so downstream logging can distinguish the
    # source.
    dataset_name: str = "proprietary_ego"


class FineTunePropEgoDataTransform(DataTransform):
    repack: FineTunePropEgoRepackTransform = Field(
        default_factory=FineTunePropEgoRepackTransform
    )
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class FineTunePropEgoDataConfig(PropEgoDataConfig):
    transform: FineTunePropEgoDataTransform = Field(
        default_factory=FineTunePropEgoDataTransform
    )


class FineTunePropEgoLaunchConfig(LaunchConfig):
    model: KappaModelConfig = Field(
        default_factory=lambda: KappaModelConfig(
            # Mirror pretrain_egodex_kappa_config: no proprio.
            use_proprio=False,
        )
    )
    data: FineTunePropEgoDataConfig = Field(default_factory=FineTunePropEgoDataConfig)

    train: TrainConfig = TrainConfig(
        name="kappa",
        max_training_steps=3000,
        train_batch_size=128,
        gradient_accumulation_steps=1,
        num_workers=8,
        prefetch_factor=2,
        # Episode-aware block shuffle: one batch ≈ one episode → decord
        # sequentializes per-worker decode (same speedup the EgoDex pre-train
        # config gets on the 50→30 Hz token-to-video mapping).
        block_shuffle_size=64,
        frame_cache_size=2,
        checkpointing_steps=500,
        max_checkpoints_to_keep=5,
        validation_steps=0,
        mixed_precision="bf16",
        learning_rate=2e-5,
        warmup_steps=0,
        warmup_ratio=0.1,
        # Seeded via --train.pretrained-weights from the CLI (e.g. from the
        # EgoDex pre-train run); nothing wired in by default since the
        # post-train checkpoint's proprio embedder would mismatch this
        # config's no-proprio model.
        pretrained_weights=None,
    )


LaunchConfig = FineTunePropEgoLaunchConfig

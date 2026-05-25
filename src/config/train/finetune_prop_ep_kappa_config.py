"""Fine-tune Kappa on the proprietary 2026-05-16 dataset (all 60 episodes).

Continues training the Post-Train checkpoint (SONIC tokens + RTC + proprio)
on the proprietary set, which ships its own pre-materialized SONIC tokens
via the ``action.motion_token`` parquet column (see
``scripts/preprocess/proprietary/calc_sonic_tokens.py``).

Training target is SONIC tokens with RTC (``use_rtc=True``) + proprio, fully
inherited from Post-Train so the loaded checkpoint stays compatible without
re-initialization. Hand bits are zeroed in the cached tokens because the
proprietary teleop hands are fake.
"""
from __future__ import annotations

from pydantic import Field

from src.config.config import LaunchConfig, TrainConfig
from src.config.data_prop import PropDataConfig
from src.config.model_kappa import KappaModelConfig
from src.data.transform import (
    DataTransform,
    HERepackTransform,
    IdentityFieldTransform,
    KappaModelTransform,
)


# Per-dataset proprio stats, produced from this dataset's own episodes by
# ``scripts/preprocess/proprietary/calc_proprio_stats.py``. The 13-group
# schema is read by ``_load_normalize_bounds`` via ``proprio_layout``.
# Trainer.save_checkpoint copies this JSON into each ckpt_<step>/ so deploy
# / sim / eval can auto-discover it.
_DEFAULT_PROPRIO_STATS = "resource/dataset/proprietary/real/2026-05-23/meta/proprio_stats.json"


class FineTunePropDataTransform(DataTransform):
    # HE repack is source-agnostic (it just normalizes the dict schema before
    # the model transform) — reused here so PropDataset samples drop into
    # ``KappaCollator`` without conditional logic.
    repack: HERepackTransform = Field(default_factory=HERepackTransform)
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class FineTunePropDataConfig(PropDataConfig):
    transform: FineTunePropDataTransform = Field(default_factory=FineTunePropDataTransform)
    emit_proprio: bool = True
    proprio_stats_path: str = _DEFAULT_PROPRIO_STATS
    normalize_mode: str = "bounds_q99"
    # Default ``None`` keeps all 60 proprietary episodes. Override via CLI to
    # restrict to a subset for sanity-check / overfit runs.
    episode_indices: list[int] | None = None


class FineTunePropLaunchConfig(LaunchConfig):
    model: KappaModelConfig = Field(
        default_factory=lambda: KappaModelConfig(
            use_proprio=True,
            use_rtc=True,
            proprio_stats_path=_DEFAULT_PROPRIO_STATS,
        )
    )
    data: FineTunePropDataConfig = Field(default_factory=FineTunePropDataConfig)

    train: TrainConfig = TrainConfig(
        name="kappa",
        max_training_steps=5000,
        train_batch_size=128,
        gradient_accumulation_steps=1,
        num_workers=8,
        prefetch_factor=2,
        block_shuffle_size=64,
        frame_cache_size=2,
        checkpointing_steps=1000,
        max_checkpoints_to_keep=5,
        validation_steps=0,
        mixed_precision="bf16",
        learning_rate=2e-5,
        warmup_steps=0,
        warmup_ratio=0.1,
        # Seeded via --train.pretrained-weights from the CLI (or by
        # scripts/train/train_kappa.sh chaining from posttrain).
        pretrained_weights=None,
    )


LaunchConfig = FineTunePropLaunchConfig

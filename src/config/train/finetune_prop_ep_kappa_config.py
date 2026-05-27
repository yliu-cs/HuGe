"""Fine-tune Kappa on proprietary LeRobot v2.1 dumps (one or more roots).

Continues training the Post-Train checkpoint (SONIC tokens + proprio)
on any subset of dumps under ``resource/dataset/proprietary``. Each root
ships its own pre-materialized SONIC tokens via the ``action.motion_token``
parquet column; run
``python scripts/preprocess/proprietary/calc_sonic_tokens.py --data-roots ROOT [ROOT ...]``
once per root before training to populate the ``sonic_tokens/`` and
``g1_proprio/`` caches.

Selecting training data
-----------------------

The CLI exposes ``--data.data-roots`` (``list[str]``, nargs='*'). Default is
``real/2026-05-23``; ``run_kappa.sh`` can then drive the post-train →
fine-tune chain with no data flags. The original 60-episode
``real/2026-05-16`` set stays available via an explicit ``--data.data-roots``
override. Proprio normalization stats are NOT per-dataset by default — they
are inherited from Post-Train (see ``_DEFAULT_PROPRIO_STATS`` below) so the
normalization space stays coupled across the chain. Override with
``--data.proprio-stats-path`` if you want to A/B per-dataset stats.

  # single root, original behavior
  ./scripts/train/train_kappa.sh finetune_prop_ep_kappa_config

  # different root
  ./scripts/train/train_kappa.sh finetune_prop_ep_kappa_config \
      --data.data-roots resource/dataset/proprietary/real/2026-05-23

  # mix two real roots (concat via ConcatMixDataset, BlockShuffle-safe)
  ./scripts/train/train_kappa.sh finetune_prop_ep_kappa_config \
      --data.data-roots resource/dataset/proprietary/real/2026-05-16 \
                        resource/dataset/proprietary/real/2026-05-23

  # diagnostic single-episode overfit on the first episode of every root
  ./scripts/train/train_kappa.sh finetune_prop_ep_kappa_config \
      --data.episode-indices 0

Training target is SONIC tokens + proprio, fully inherited from Post-Train so
the loaded checkpoint stays compatible without re-initialization. Real hand
bits propagate through the SONIC token / proprio caches when the dump's
parquet has ``action.hand_binary`` (2026-05-23); otherwise they remain zero
(2026-05-16 fake-fingers contract, sim).
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


_DEFAULT_DATA_ROOT = "resource/dataset/proprietary/real/2026-05-23"

# Stage 3 normalizes against the UNION of HE + WBT + PROP, produced by
# ``scripts/preprocess/calc_union_proprio_stats.py``. The envelope inherits
# WBT's wide arm / quat-yaw workspace AND PROP's squat / torso-rotation
# range, so deploy poses stop landing OOD vs the model's conditioning --
# WBT-only would clip squat signals, PROP-only would clip manipulation.
#
# File lives under the PROP dump's ``meta/`` (not ``G1_WBT-cache/``) because
# these stats are a Stage 3 asset: Stage 2 (HE+WBT post-train) still uses
# ``resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json``. The
# Stage 2 -> Stage 3 normalization-space mismatch is covariate shift the
# embedder re-calibrates over the 5k fine-tune steps; we accept it because
# "deploy must squat" is a hard requirement and the alternative
# (Stage 2 re-run with union stats) costs an extra post-train.
#
# The 13-group schema matches ``_load_normalize_bounds`` and
# ``Trainer._bundle_proprio_stats`` copies the file into each ckpt_<step>/
# for deploy / sim / eval auto-discovery. The stats path stays decoupled
# from ``--data.data-roots`` so swapping roots does NOT silently change the
# normalization space; override via ``--data.proprio-stats-path`` only when
# explicitly A/B-ing a different stats file.
_DEFAULT_PROPRIO_STATS = (
    "resource/dataset/proprietary/real/2026-05-23/meta/proprio_stats_union.json"
)


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
    # Default to the most recent real dump so ``./scripts/train/run_kappa.sh``
    # (which forwards args to every stage and can't set per-stage data
    # overrides) lands on a sane post-train -> fine-tune chain. The stats
    # path stays pinned at the Post-Train HE+WBT JSON regardless of which
    # data roots are selected, so swapping roots does NOT silently change
    # the normalization space. Override via
    # ``--data.data-roots PATH1 [PATH2 ...]``; only also pass
    # ``--data.proprio-stats-path`` when you explicitly want per-dataset stats.
    data_roots: list[str] = Field(default_factory=lambda: [_DEFAULT_DATA_ROOT])
    # ``episode_indices=None`` keeps every episode in each selected root;
    # override via ``--data.episode-indices`` for sanity-check / overfit runs.


class FineTunePropLaunchConfig(LaunchConfig):
    model: KappaModelConfig = Field(
        default_factory=lambda: KappaModelConfig(
            use_proprio=True,
            proprio_stats_path=_DEFAULT_PROPRIO_STATS,
        )
    )
    data: FineTunePropDataConfig = Field(default_factory=FineTunePropDataConfig)

    train: TrainConfig = TrainConfig(
        name="kappa",
        max_training_steps=3000,
        train_batch_size=128,
        gradient_accumulation_steps=1,
        num_workers=8,
        prefetch_factor=2,
        block_shuffle_size=64,
        frame_cache_size=2,
        checkpointing_steps=500,
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

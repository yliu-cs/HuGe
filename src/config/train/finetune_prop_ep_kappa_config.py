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

# Inherit the Post-Train HE+WBT stats verbatim instead of refitting on the
# proprietary subset. Each proprietary dump (~tens to ~hundreds of episodes)
# is too small to refit q01/q99 cleanly — narrower bounds shift deploy-time
# proprio OOD vs the model's learned conditioning, which we observed as a
# near-zero motion delta on real-robot rollouts. Keeping the post-train
# stats means fine-tune only updates weights while the normalization space
# stays coupled to what Stage 2 saw. The HE+WBT producer
# (``scripts/preprocess/calc_g1_proprio_stats.py``) now ships the same
# 13-group schema, so ``_load_normalize_bounds`` reads this path the same
# way it would a per-dataset stats file. Trainer.save_checkpoint still
# copies it into each ckpt_<step>/ for deploy / sim / eval auto-discovery.
# Override with ``--data.proprio-stats-path`` (pair with the matching
# ``--data.data-roots``) when running per-dataset A/B comparisons via
# ``scripts/preprocess/proprietary/calc_proprio_stats.py``.
_DEFAULT_PROPRIO_STATS = "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json"


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

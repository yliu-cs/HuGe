"""Post-train config for Kappa: G1 expert + RTC + Proprio on HE-G1 + G1-WBT.

Loads weights from a stage 1-2 pretrain checkpoint (SONIC-only MoT) via
``train.pretrained_weights``, expands the model with the third (G1) expert,
copies SONIC expert weights into G1副本 (``init_g1_from_sonic``), freezes
everything outside the SONIC + G1 expert paths (``freeze_for_post_train``),
and trains with two flow-matching losses + RTC prefix substitution.

Data: ``HEWBTMixDataConfig`` toggled to emit ``g1_actions`` and ``proprio``
alongside the SONIC tokens. Both sources share a single normalization stats
file (``g1_action_stats.json`` / ``g1_proprio_stats.json``) so the [-1, 1]
mapping is byte-identical across HE and WBT.
"""
from __future__ import annotations

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


# Default stats paths sit next to the WBT cache (post-train uses HE+WBT
# combined stats; WBT is typically the larger of the two so the meta dir
# already exists there).
#
# Stats describe the 23-D action schema (arms + hand_bits + loco_cmd); the
# old 31-D file was deleted alongside the cache regen.
_DEFAULT_ACTION_STATS = "resource/dataset/G1_WBT-cache/meta/g1_action_stats.json"
_DEFAULT_PROPRIO_STATS = "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json"


class PostTrainHEWBTDataTransform(DataTransform):
    repack: HERepackTransform = Field(default_factory=HERepackTransform)
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class PostTrainHEWBTDataConfig(HEWBTMixDataConfig):
    dataset_name: str = "he_wbt_posttrain"
    transform: PostTrainHEWBTDataTransform = Field(
        default_factory=PostTrainHEWBTDataTransform,
    )
    # ``bounds_q99`` clips the long-tail odometry-derived loco_cmd channels
    # (post-train action dims 16:23) at the 1/99 percentiles before mapping
    # to [-1, 1]; plain min/max would let rare ±2 m/s spikes squeeze the
    # walking distribution into a tiny fraction of the normalized range.
    he: HEDataConfig = Field(
        default_factory=lambda: HEDataConfig(
            robot_type="g1",
            emit_g1_actions=True,
            emit_proprio=True,
            g1_action_stats_path=_DEFAULT_ACTION_STATS,
            g1_proprio_stats_path=_DEFAULT_PROPRIO_STATS,
            normalize_mode="bounds_q99",
        )
    )
    wbt: WBTDataConfig = Field(
        default_factory=lambda: WBTDataConfig(
            emit_g1_actions=True,
            emit_proprio=True,
            g1_action_stats_path=_DEFAULT_ACTION_STATS,
            g1_proprio_stats_path=_DEFAULT_PROPRIO_STATS,
            normalize_mode="bounds_q99",
        )
    )


class PostTrainHEWBTLaunchConfig(LaunchConfig):
    # ``g1_dim=23`` comes from ``KappaModelConfig`` default (WBC-coupled
    # deployment target). ``action_embedder_g1`` / ``velocity_head_g1`` are
    # randomly initialized; ``init_g1_from_sonic=True`` still copies the
    # per-layer attention/MLP from SONIC since those operate on hidden_size
    # and don't depend on g1_dim.
    model: KappaModelConfig = Field(
        default_factory=lambda: KappaModelConfig(
            use_g1_expert=True,
            use_proprio=True,
            use_rtc=True,
            rtc_on_sonic=True,
            action_loss_weight=0.3,
            g1_loss_weight=1.0,
            init_g1_from_sonic=True,
            freeze_for_post_train=True,
            g1_action_stats_path=_DEFAULT_ACTION_STATS,
            g1_proprio_stats_path=_DEFAULT_PROPRIO_STATS,
        )
    )
    data: PostTrainHEWBTDataConfig = Field(
        default_factory=PostTrainHEWBTDataConfig,
    )

    train: TrainConfig = TrainConfig(
        name="kappa",
        num_train_epochs=10,
        train_batch_size=128,
        gradient_accumulation_steps=2,
        num_workers=8,
        prefetch_factor=2,
        block_shuffle_size=64,
        frame_cache_size=2,
        checkpointing_steps=10000,
        max_checkpoints_to_keep=5,
        validation_steps=0,
        mixed_precision="bf16",
        # Post-train uses ~30% of stage 1-2's lr because most of the network
        # is frozen (only the SONIC + G1 expert副本 update) and the new G1
        # head needs careful warm-up to avoid blowing the SONIC anchor.
        learning_rate=2e-5,
        warmup_steps=0,
        warmup_ratio=0.01,
        # No default seed: pass --train.pretrained-weights at the CLI (or let
        # scripts/train/run_kappa_pipeline.sh thread the previous stage's run
        # dir in automatically).
        pretrained_weights=None,
    )


LaunchConfig = PostTrainHEWBTLaunchConfig

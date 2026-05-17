"""Fine-tune Kappa post-train on a single HE episode (default ep 7831).

Initializes from an already-trained post-train checkpoint (G1 expert + RTC +
Proprio) and continues training only on the filtered HE episodes — useful as a
diagnostic for overfit / data-pipeline sanity, or to specialize the policy to a
specific clip.

Important: ``init_g1_from_sonic`` is ``False`` here because the loaded
checkpoint already contains a trained G1 expert. The trainer copies SONIC→G1
*after* ``load_pretrained_weights`` (``src/trainers/kappa.py``), so flipping
this on would clobber the trained expert.
"""
from __future__ import annotations

from pydantic import Field

from src.config.config import LaunchConfig, TrainConfig
from src.config.data_he import HEDataConfig
from src.config.model_kappa import KappaModelConfig
from src.data.transform import (
    DataTransform,
    HERepackTransform,
    IdentityFieldTransform,
    KappaModelTransform,
)


_DEFAULT_ACTION_STATS = "resource/dataset/G1_WBT-cache/meta/g1_action_stats.json"
_DEFAULT_PROPRIO_STATS = "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json"


class FineTuneHEDataTransform(DataTransform):
    repack: HERepackTransform = Field(default_factory=HERepackTransform)
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class FineTuneHEDataConfig(HEDataConfig):
    transform: FineTuneHEDataTransform = Field(default_factory=FineTuneHEDataTransform)
    robot_type: str = "g1"
    emit_g1_actions: bool = True
    emit_proprio: bool = True
    g1_action_stats_path: str = _DEFAULT_ACTION_STATS
    g1_proprio_stats_path: str = _DEFAULT_PROPRIO_STATS
    normalize_mode: str = "bounds_q99"
    episode_indices: list[int] = Field(default_factory=lambda: [7831])


class FineTuneHELaunchConfig(LaunchConfig):
    model: KappaModelConfig = Field(
        default_factory=lambda: KappaModelConfig(
            use_g1_expert=True,
            use_proprio=True,
            use_rtc=True,
            rtc_on_sonic=True,
            action_loss_weight=0.3,
            g1_loss_weight=1.0,
            init_g1_from_sonic=False,
            freeze_for_post_train=True,
            g1_action_stats_path=_DEFAULT_ACTION_STATS,
            g1_proprio_stats_path=_DEFAULT_PROPRIO_STATS,
        )
    )
    data: FineTuneHEDataConfig = Field(default_factory=FineTuneHEDataConfig)

    train: TrainConfig = TrainConfig(
        name="kappa",
        max_training_steps=2000,
        train_batch_size=32,
        gradient_accumulation_steps=1,
        num_workers=8,
        prefetch_factor=2,
        block_shuffle_size=0,
        frame_cache_size=2,
        checkpointing_steps=1000,
        max_checkpoints_to_keep=5,
        validation_steps=0,
        mixed_precision="bf16",
        learning_rate=1e-5,
        warmup_steps=0,
        warmup_ratio=0.1,
        # No default seed: pass --train.pretrained-weights at the CLI (or let
        # scripts/train/run_kappa_pipeline.sh thread the previous stage's run
        # dir in automatically).
        pretrained_weights=None,
    )


LaunchConfig = FineTuneHELaunchConfig

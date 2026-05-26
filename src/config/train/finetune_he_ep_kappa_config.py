"""Fine-tune Kappa on a single HE episode (default ep 7831), SONIC objective.

Continues training the Post-Train checkpoint (SONIC tokens + proprio) on the
filtered HE episode set — useful as a diagnostic for overfit / data-pipeline
sanity, or to specialize the policy to a specific clip.

The training target is still SONIC tokens; no G1 expert is involved. Proprio
inherits the Post-Train configuration so the loaded weights stay compatible
without re-initialization.
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


_DEFAULT_PROPRIO_STATS = "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json"


class FineTuneHEDataTransform(DataTransform):
    repack: HERepackTransform = Field(default_factory=HERepackTransform)
    field: IdentityFieldTransform = Field(default_factory=IdentityFieldTransform)
    model: KappaModelTransform = Field(default_factory=KappaModelTransform)


class FineTuneHEDataConfig(HEDataConfig):
    transform: FineTuneHEDataTransform = Field(default_factory=FineTuneHEDataTransform)
    robot_type: str = "g1"
    emit_proprio: bool = True
    proprio_stats_path: str = _DEFAULT_PROPRIO_STATS
    normalize_mode: str = "bounds_q99"
    episode_indices: list[int] = Field(default_factory=lambda: [7831])


class FineTuneHELaunchConfig(LaunchConfig):
    model: KappaModelConfig = Field(
        default_factory=lambda: KappaModelConfig(
            use_proprio=True,
            proprio_stats_path=_DEFAULT_PROPRIO_STATS,
        )
    )
    data: FineTuneHEDataConfig = Field(default_factory=FineTuneHEDataConfig)

    train: TrainConfig = TrainConfig(
        name="kappa",
        max_training_steps=2000,
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
        # No default seed: pass --train.pretrained-weights at the CLI (or let
        # scripts/train/train_kappa.sh thread the previous stage's run dir in
        # automatically).
        pretrained_weights=None,
    )


LaunchConfig = FineTuneHELaunchConfig

"""
Configuration system for kappa training.
"""
from __future__ import annotations

import os
import datetime
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from src.data.transform import DataTransform


class LoggingConfig(BaseModel):
    logging_dir: str = "logs"
    report_to: str | None = None
    log_freq: int = 100
    enable_profiling: bool = True
    profiling_window: int = 50


class TrainConfig(BaseModel):
    num_workers: int = 8
    # Per-worker prefetch depth; raise this when the input pipeline is the
    # bottleneck so the GPU never blocks waiting on a batch.
    prefetch_factor: int = 4
    overfit_single_batch: bool = False
    name: str = "kappa"
    resume_from_checkpoint: str | None = None
    skip_resumed_steps: bool = False
    # Weights-only init for two-stage training (e.g. EgoDex -> HE). Path may
    # point to a model.safetensors file, a ckpt_* directory, or a run dir
    # (latest ckpt is auto-selected). Optimizer/scheduler/global_step start
    # fresh. Ignored when `resume_from_checkpoint` resolves successfully.
    pretrained_weights: str | None = None

    hf_token: Union[str, Path] = Path(".hf_token")

    lora: bool = False
    output_dir: str = ".runs"
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float | None = 1.0
    optimizer_foreach: bool | None = None

    train_batch_size: int = 2
    val_batch_size: int = 2
    val_num_batches: int = 10

    checkpointing_steps: int = 100
    max_checkpoints_to_keep: int | None = None
    validation_steps: int = 50

    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    optimizer_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {
            "betas": (0.9, 0.95),
            "eps": 1e-8,
        }
    )
    scheduler_specific_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {}
    )

    data_parallel: str = "ddp"
    sharding_strategy: str = "full-shard"
    enable_gradient_checkpointing: bool = True

    use_deepspeed: bool = True
    deepspeed_config: str | None = None
    zero_stage: int = 2
    offload_optimizer: bool = False
    offload_param: bool = False

    # Exactly one of these must be set; the other must remain None. Defaulting
    # max_training_steps to None (rather than a non-None placeholder) ensures
    # that a config setting only num_train_epochs is honored — otherwise the
    # placeholder wins on the trainer's not-None short-circuit.
    max_training_steps: int | None = None
    num_train_epochs: int | None = None
    warmup_steps: int | None = 0
    warmup_ratio: float | None = 0.1

    @model_validator(mode="after")
    def check_warmup(self) -> Self:
        steps, ratio = self.warmup_steps, self.warmup_ratio
        if (steps is not None and steps > 0) and (ratio is not None and ratio > 0):
            raise ValueError("Only one of warmup_steps or warmup_ratio can be set")
        return self

    @model_validator(mode="after")
    def check_steps(self) -> Self:
        steps, epochs = self.max_training_steps, self.num_train_epochs
        has_steps = steps is not None and steps > 0
        has_epochs = epochs is not None and epochs > 0
        if not has_steps and not has_epochs:
            raise ValueError("At least one of max_training_steps or num_train_epochs must be set")
        if has_steps and has_epochs:
            raise ValueError(
                "Only one of max_training_steps or num_train_epochs can be set; "
                "leaving both non-None makes the source of truth ambiguous."
            )
        return self


class DataConfig(BaseModel):
    # Trainer dispatches on this name; subclasses override.
    dataset_name: str = "humanoid_everyday"
    # Per-sample pipeline: repack (source-specific) -> field (token-space) ->
    # model (image augmentation). Default 224×224 lives on KappaModelTransform.
    transform: DataTransform = Field(default_factory=DataTransform)

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        raise NotImplementedError


class HEDataConfig(DataConfig):
    dataset_name: str = "humanoid_everyday"
    data_root: str = "resource/dataset/humanoid-everyday"
    cache_root: str | None = None  # auto-resolved at runtime to writable parallel dir
    action_chunk_size: int = 50

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import HEDataset
        from src.data.dataset import Dataset

        raw = HEDataset(
            data_root=self.data_root,
            action_chunk_size=self.action_chunk_size,
            split=split,
            cache_root=self.cache_root,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)


class EgoDexDataConfig(DataConfig):
    dataset_name: str = "egodex"
    data_root: str = "resource/dataset/EgoDex/egodex_repacked_simple"
    cache_root: str | None = None  # auto-resolved at runtime to writable parallel dir
    action_chunk_size: int = 50
    sonic_dim: int = 66
    # Optional cap on # episodes loaded (for fast smoke tests). None = use all.
    max_episodes: int | None = None
    # Number of threads to probe `token_state.npy` shape during the first index
    # build. The cached JSON makes subsequent runs instant.
    index_workers: int = 32

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        from src.data import EgoDexDataset
        from src.data.dataset import Dataset

        raw = EgoDexDataset(
            data_root=self.data_root,
            action_chunk_size=self.action_chunk_size,
            split=split,
            cache_root=self.cache_root,
            sonic_dim=self.sonic_dim,
            max_episodes=self.max_episodes,
            index_workers=self.index_workers,
        )
        return Dataset(raw, self.transform, transform_kwargs=transform_kwargs)


class ModelConfig(BaseModel):
    ...


class LaunchConfig(BaseModel):
    exp: str = Field(default="exp", description="Name of the experiment")
    seed: int | None = 178331
    auto_tag_run: bool = False
    eval: bool = False
    debug: bool = False
    timestamp: str | None = None

    log: LoggingConfig = Field(default_factory=LoggingConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    def model_post_init(self, __context: Any) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")

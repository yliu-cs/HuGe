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


class LoggingConfig(BaseModel):
    logging_dir: str = "logs"
    report_to: str | None = None
    log_freq: int = 100
    enable_profiling: bool = True
    profiling_window: int = 50


class TrainConfig(BaseModel):
    num_workers: int = 4
    overfit_single_batch: bool = False
    name: str = "kappa"
    resume_from_checkpoint: str | None = None
    skip_resumed_steps: bool = False

    hf_token: Union[str, Path] = Path(".hf_token")

    lora: bool = False
    output_dir: str = ".runs"
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float | None = None
    optimizer_foreach: bool | None = None

    train_batch_size: int = 2
    val_batch_size: int = 2
    val_num_batches: int = 10

    checkpointing_steps: int = 100
    max_checkpoints_to_keep: int | None = None
    validation_steps: int = 50

    learning_rate: float = 1e-4
    lr_scheduler_type: str = "constant"
    lr_scheduler_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {
            "betas": (0.9, 0.95),
            "weight_decay": 1e-4,
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

    max_training_steps: int | None = 100
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
        if (steps is None or steps == 0) and (epochs is None or epochs == 0):
            raise ValueError("At least one of max_training_steps or num_train_epochs must be set")
        return self


class DataConfig(BaseModel):
    ...


class HEDataConfig(DataConfig):
    data_root: str = "resource/dataset/humanoid-everyday"
    cache_root: str | None = None  # auto-resolved at runtime to writable parallel dir
    action_chunk_size: int = 10


class ModelConfig(BaseModel):
    ...


class LaunchConfig(BaseModel):
    exp: str = Field(default="exp", description="Name of the experiment")
    seed: int | None = None
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

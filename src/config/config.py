"""Base configuration classes for Kappa training.

Per-source data configs live in sibling modules (``data_he.py``,
``data_egodex.py``, ``data_wbt.py``, ``data_mix.py``) — this file only holds
the cross-cutting base types (logging, train loop, abstract data, model,
launch). New data sources should add a new ``data_<source>.py`` rather than
extending this file.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Union

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

    # Block-shuffle sampler: 0 disables (falls back to torch shuffle=True).
    # When >0, indices are emitted in episode-contiguous chunks of this size,
    # turning random video seeks into next-frame decodes. Set to
    # ``train_batch_size`` for "one batch ≈ one episode" behavior.
    block_shuffle_size: int = 0
    # Per-worker decoded-frame LRU. EgoDex's 50→30 Hz token-to-video mapping
    # produces ~37% adjacent-row frame collisions; size=2 absorbs them under
    # block_shuffle_size>0. Set to 0 to disable.
    frame_cache_size: int = 2

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
    """Abstract base for per-source data configs.

    Concrete subclasses live in ``data_<source>.py`` and override
    ``__call__`` to construct the underlying raw dataset and wrap it in
    ``src.data.dataset.Dataset`` with this config's ``transform`` pipeline.
    """
    # Trainer dispatches on this name; subclasses override.
    dataset_name: str = "humanoid_everyday"
    # Per-sample pipeline: repack (source-specific) -> field (token-space) ->
    # model (image augmentation). Default 320×224 (W×H) lives on KappaModelTransform.
    transform: DataTransform = Field(default_factory=DataTransform)

    def __call__(self, split: str = "train", transform_kwargs: dict | None = None) -> Any:
        raise NotImplementedError


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

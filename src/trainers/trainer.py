"""
Base trainer class for kappa.
"""
from __future__ import annotations

import importlib
import math
import os
import random
import re
import shutil
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import get_scheduler
from safetensors.torch import save_file as save_safetensors

from src.utils import initialize_overwatch, snake_to_pascal
overwatch = initialize_overwatch(__name__)


def worker_init_fn(worker_id: int) -> None:
    """Initialize worker random seed."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class Trainer(ABC):
    """Abstract base trainer class."""

    cfg: Any
    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler

    train_dataset: Dataset
    val_dataset: Optional[Dataset]

    def __init__(self, cfg: Any, device: Union[torch.device, int]):
        self.device = torch.device(device)
        self.cfg = cfg

        if cfg.train.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif cfg.train.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.timestamp = self.cfg.timestamp

    @classmethod
    def instantiate(cls, cfg: Any, device: Union[torch.device, int]) -> "Trainer":
        """Instantiate a trainer based on config.train.name."""
        trainer_name = cfg.train.name
        try:
            parts = re.split(r"[-_]", trainer_name)
            trainer_name = "_".join([p.lower() for p in parts])
            module = importlib.import_module(f"src.trainers.{trainer_name}")
            trainer_clazz = getattr(module, f"{snake_to_pascal(trainer_name)}Trainer")
        except Exception as e:
            raise ValueError(
                f"fail to import {trainer_name} from trainers"
            ) from e
        return trainer_clazz(cfg, device)

    @property
    def train_cfg(self):
        return self.cfg.train

    @property
    def log_cfg(self):
        return self.cfg.log

    @property
    def data_cfg(self):
        return self.cfg.data

    @property
    def world_size(self) -> int:
        return overwatch.world_size()

    @property
    def device_train_batch_size(self) -> int:
        return self.cfg.train.train_batch_size

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.cfg.train.gradient_accumulation_steps

    @property
    def global_train_batch_size(self) -> int:
        return (
            self.device_train_batch_size
            * self.world_size
            * self.gradient_accumulation_steps
        )

    @property
    def len_train_dataset(self) -> int:
        if hasattr(self, "_len_train_dataset") and self._len_train_dataset:
            return self._len_train_dataset
        return len(self.train_dataset)

    @len_train_dataset.setter
    def len_train_dataset(self, length: int):
        self._len_train_dataset = length

    @property
    def len_train_dataloader(self) -> int:
        total = self.len_train_dataset / (self.device_train_batch_size * self.world_size)
        if hasattr(self, "train_dataloader_drop_last") and self.train_dataloader_drop_last:
            return math.floor(total)
        return math.ceil(total)

    @property
    def num_steps_per_epoch(self) -> int:
        return max(self.len_train_dataloader // self.gradient_accumulation_steps, 1)

    @property
    def max_training_steps(self) -> int:
        # Treat 0 the same as None ("not set") so configs can opt into the
        # epoch-based path by setting only `num_train_epochs`. The validator
        # already enforces mutual exclusion.
        steps = self.cfg.train.max_training_steps
        if steps is not None and steps > 0:
            return steps
        # `num_steps_per_epoch` is already per-rank optimizer steps:
        # `len_train_dataloader` divides the dataset by `world_size`, so
        # multiplying by `world_size` here would inflate total training by
        # that factor (8× on 8 GPUs).
        return self.num_steps_per_epoch * self.cfg.train.num_train_epochs

    @property
    def num_warmup_steps(self) -> int:
        if self.cfg.train.warmup_steps is not None and self.cfg.train.warmup_steps > 0:
            return self.cfg.train.warmup_steps
        return math.ceil(self.max_training_steps * (self.cfg.train.warmup_ratio or 0.1))

    @property
    def run_name(self) -> str:
        run_name = (
            f"{self.cfg.exp}.{self.train_cfg.name}"
            f".b{self.global_train_batch_size}.gpus{self.world_size}"
        )
        run_name = f"{run_name}.{self.timestamp}"
        if self.cfg.debug:
            run_name = f"debug-{run_name}"
        return run_name

    @property
    def project_dir(self) -> str:
        return os.path.join(self.cfg.train.output_dir, self.train_cfg.name, self.run_name)

    @abstractmethod
    def init_models(self):
        """Initialize the models for training."""

    @abstractmethod
    def create_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        """Create training and validation datasets."""

    @abstractmethod
    def create_dataloaders(
        self, train_dataset: Dataset, val_dataset: Optional[Dataset]
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        """Create training and validation dataloaders."""

    def create_optimizer(self):
        optimizer_kwargs = dict(self.cfg.train.optimizer_kwargs)
        if self.cfg.train.optimizer_foreach is not None:
            optimizer_kwargs["foreach"] = self.cfg.train.optimizer_foreach
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.train.learning_rate,
            **optimizer_kwargs,
        )

    def create_scheduler(self, num_training_steps: int | None = None, optimizer=None):
        if num_training_steps is None:
            num_training_steps = self.max_training_steps
        self.lr_scheduler = get_scheduler(
            name=self.cfg.train.lr_scheduler_type,
            optimizer=optimizer if optimizer is not None else self.optimizer,
            num_warmup_steps=self.num_warmup_steps * self.world_size,
            num_training_steps=num_training_steps * self.world_size,
            scheduler_specific_kwargs=self.cfg.train.scheduler_specific_kwargs,
        )
        return self.lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int | None = None):
        optimizer = self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)

    def step(self, batch_input: dict, global_step: int, local_step: int) -> tuple[bool, dict[str, Any]]:
        """Perform a single training step."""
        self.local_step = local_step
        self.global_step = global_step
        sync_gradients, losses = self.training_step(batch_input)
        return sync_gradients, losses

    @abstractmethod
    def training_step(
        self, batch: dict[str, torch.Tensor | Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Perform a single training step. Return (sync_gradients, losses_dict)."""

    def prepare(self, accelerator: Accelerator) -> DataLoader:
        self.accelerator = accelerator

        if accelerator.state.deepspeed_plugin is not None:
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler, self.train_dataloader
            )
        else:
            self.optimizer, self.lr_scheduler = accelerator.prepare(
                self.optimizer, self.lr_scheduler
            )
            self.train_dataloader = accelerator.prepare(self.train_dataloader)

        overwatch.add_file_handler(self.project_dir)

        if self.cfg.train.overfit_single_batch:
            overwatch.warning("Overfitting a single batch.")
            first_batch = next(iter(self.train_dataloader))
            class SingleBatchLoader:
                def __iter__(self):
                    while True:
                        yield first_batch
                def __len__(self):
                    return 1
            self.train_dataloader = SingleBatchLoader()

        val_dataloader = getattr(self, "val_dataloader", None)
        if val_dataloader is not None:
            self.val_dataloader = accelerator.prepare(self.val_dataloader)

        self.accelerator = accelerator
        return self.train_dataloader

    def load_pretrained_weights(self) -> Optional[str]:
        """Load weights only (no optimizer/scheduler/step state) for two-stage training.

        Resolves ``train.pretrained_weights`` to a ``model.safetensors`` file:
        accepts a direct file path, a ``ckpt_*`` directory, or a run directory
        (in which case the latest ``ckpt_*`` is selected). Must be called after
        ``init_models()`` and before ``prepare()`` so the unwrapped HF model is
        addressable on every rank.
        """
        path_str = self.train_cfg.pretrained_weights
        if path_str is None:
            return None

        from pathlib import Path
        from safetensors.torch import load_file

        p = Path(path_str)
        if p.is_file():
            weights_file = p
        elif p.is_dir():
            if (p / "model.safetensors").is_file():
                weights_file = p / "model.safetensors"
            elif (p / "checkpoints").is_dir():
                ckpt_dirs = [
                    d for d in (p / "checkpoints").iterdir()
                    if d.is_dir() and d.name.startswith("ckpt_")
                ]
                if not ckpt_dirs:
                    raise FileNotFoundError(f"No ckpt_* dirs under {p / 'checkpoints'}")
                ckpt_dirs.sort(key=lambda d: int(d.name.split("_")[1]))
                weights_file = ckpt_dirs[-1] / "model.safetensors"
                if not weights_file.is_file():
                    raise FileNotFoundError(f"Expected {weights_file} to exist")
            else:
                raise FileNotFoundError(
                    f"Cannot resolve pretrained_weights from directory {p}: "
                    "expected model.safetensors or checkpoints/ckpt_*/"
                )
        else:
            raise FileNotFoundError(f"pretrained_weights path does not exist: {p}")

        overwatch.info(f"Loading pretrained weights from {weights_file}")
        state_dict = load_file(str(weights_file))
        result = self.model.load_state_dict(state_dict, strict=False)

        n_loaded = len(state_dict) - len(result.unexpected_keys)
        msg = (
            f"Loaded {n_loaded}/{len(state_dict)} tensors; "
            f"{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected"
        )
        if result.missing_keys or result.unexpected_keys:
            overwatch.warning(msg)
            if result.missing_keys:
                overwatch.warning(f"  First missing keys: {result.missing_keys[:5]}")
            if result.unexpected_keys:
                overwatch.warning(f"  First unexpected keys: {result.unexpected_keys[:5]}")
        else:
            overwatch.info(msg)

        return str(weights_file)

    def resume_from_checkpoint(self) -> tuple[int, Optional[str]]:
        if self.cfg.train.resume_from_checkpoint is None:
            return 0, None

        path = self.cfg.train.resume_from_checkpoint
        if "ckpt_" in path:
            load_path = path
        else:
            if os.path.exists(f"{path}/checkpoints"):
                dirs = os.listdir(f"{path}/checkpoints")
                dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
                load_path = os.path.join(path, "checkpoints", dirs[-1]) if dirs else None
            else:
                load_path = None

        if load_path is None or not os.path.exists(load_path):
            overwatch.warning(f"Checkpoint '{self.cfg.train.resume_from_checkpoint}' not found. Starting fresh.")
            self.cfg.train.resume_from_checkpoint = None
            return 0, None

        overwatch.info(f"Resuming from checkpoint {load_path}")
        self.accelerator.load_state(load_path)
        initial_global_step = int(os.path.basename(load_path).split("_")[1]) + 1
        return initial_global_step, load_path

    def save_checkpoint(self, global_step: int) -> str:
        save_dir = os.path.join(self.project_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_dir = os.path.join(save_dir, f"ckpt_{global_step}")

        self.accelerator.save_state(ckpt_dir)

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state_dict = {}
            seen_ptrs: set[int] = set()
            for k, v in unwrapped_model.state_dict().items():
                ptr = v.data_ptr()
                if ptr in seen_ptrs:
                    v = v.clone()
                seen_ptrs.add(ptr)
                state_dict[k] = v.cpu()
            save_safetensors(state_dict, os.path.join(ckpt_dir, "model.safetensors"))
            self._bundle_proprio_stats(ckpt_dir)

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            max_to_keep = self.train_cfg.max_checkpoints_to_keep or 100
            if max_to_keep > 0:
                ckpt_dirs = [d for d in os.listdir(save_dir) if d.startswith("ckpt_") and os.path.isdir(os.path.join(save_dir, d))]
                ckpt_dirs_sorted = sorted(ckpt_dirs, key=lambda d: int(d.split("_")[1]), reverse=True)
                for old_ckpt in ckpt_dirs_sorted[max_to_keep:]:
                    old_path = os.path.join(save_dir, old_ckpt)
                    try:
                        shutil.rmtree(old_path)
                        overwatch.info(f"Removed old checkpoint: {old_path}")
                    except Exception as e:
                        overwatch.warning(f"Failed to remove old checkpoint: {e}")

        self.accelerator.wait_for_everyone()
        return ckpt_dir

    def _bundle_proprio_stats(self, ckpt_dir: str) -> None:
        """Copy the active dataset's proprio stats JSON into ``ckpt_dir``.

        Discovery order (first existing source wins): top-level
        ``data_cfg.proprio_stats_path`` for single-dataset configs (prop /
        he / wbt), then mix children ``data_cfg.{prop,he,wbt}.proprio_stats_path``.
        This makes each ckpt self-contained so inference / deploy / eval can
        ``discover_proprio_stats(weights_path)`` without a hardcoded global
        default.
        """
        candidates: list[str] = []
        top = getattr(self.data_cfg, "proprio_stats_path", None)
        if top:
            candidates.append(top)
        for child in ("prop", "he", "wbt"):
            sub = getattr(self.data_cfg, child, None)
            sub_path = getattr(sub, "proprio_stats_path", None) if sub is not None else None
            if sub_path:
                candidates.append(sub_path)
        src_path: str | None = next((p for p in candidates if p and os.path.isfile(p)), None)
        if src_path is None:
            overwatch.warning(
                "no proprio_stats_path resolves to an existing file on data_cfg; "
                "ckpt will not bundle proprio_stats.json (inference will fall back to legacy default)."
            )
            return
        dst = os.path.join(ckpt_dir, "proprio_stats.json")
        try:
            shutil.copyfile(src_path, dst)
        except OSError as e:
            overwatch.warning(f"failed to bundle proprio_stats.json into {ckpt_dir}: {e}")

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def log(self, metrics: dict[str, Any], start_time: float | None = None) -> None:
        wandb_dict = copy(metrics)
        for k, v in list(wandb_dict.items()):
            if isinstance(v, (torch.Tensor, np.ndarray)):
                wandb_dict[k] = v.item()
        self.accelerator.log(wandb_dict, step=self.global_step)

    def evaluate(self) -> dict[str, float] | None:
        return None

    def finalize(self):
        pass

    @property
    def lr(self):
        return self.lr_scheduler.get_last_lr()[0]

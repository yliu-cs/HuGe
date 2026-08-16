"""Shared training loop for Stage 1/2/3."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
import traceback
import uuid
import warnings
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, TextIO, TypeVar

import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from huge.data.collator import HugeCollator
from huge.data.datamodule import build_stage_dataset, limit_dataset, normalize_dataset_specs
from huge.data.proprio_stats import validate_proprio_stats_contract
from huge.data.validation_manifest import (
    build_validation_sources,
    ensure_validation_manifest,
    load_validation_manifest,
)
from huge.data.sampler import BlockShuffleSampler
from huge.data.schemas import ActionSchema
from huge.data.video_reader import set_frame_cache_size
from huge.models.qwen_sonicfm import ACTION_HEAD_DEFAULTS, QwenSonicFM, QwenSonicFMConfig
from huge.training.checkpointing import (
    PORTABLE_CHECKPOINT_KIND,
    RESUME_STATE_KIND,
    TRAINER_STATE_SCHEMA_VERSION,
    CheckpointValidationError,
    atomic_publish_directory,
    build_checkpoint_metadata,
    canonical_json_bytes,
    publish_directory_manifest,
    sha256_bytes,
    sha256_file,
    validate_published_directory,
    write_json_atomic,
)
from huge.training.losses import qwen_sonicfm_loss


warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False because encoder_layer\.norm_first was True",
    category=UserWarning,
    module=r"torch\.nn\.modules\.transformer",
)

DEFAULT_DEEPSPEED_CONFIG = "configs/deepspeed/ds_zero2.json"
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
OPTIMIZER_GROUP_ORDER = ("vlm", "projector_base", "action_head")
EXPECTED_FULL_INVENTORY = {
    "qwenpi_v3": {"vlm": 2_127_532_032, "projector_base": 33_636_352, "action_head": 379_082_818},
    "qwengr00t": {"vlm": 2_127_532_032, "projector_base": 2_102_272, "action_head": 379_115_586},
    "qwenoft": {"vlm": 2_127_532_032, "projector_base": 2_102_272, "action_head": 10_641_474},
}

@dataclass(frozen=True)
class StageDefaults:
    learning_rate: float
    batch_size: int
    epochs: int
    sample_stride: int
    checkpointing_steps: int
    max_steps: int | None = None
    weight_decay: float = 0.01
    num_workers: int = 4
    freeze_qwen: bool = False
    qwen_lora: bool = False
    train_qwen_lora: bool = False
    qwen_lora_r: int = 8
    qwen_lora_alpha: int = 16
    qwen_lora_dropout: float = 0.0
    mixed_precision: str = "bf16"
    save_epoch_checkpoints: bool = False
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    use_deepspeed: bool = True
    zero_stage: int = 2
    block_shuffle_size: int = 64
    frame_cache_size: int = 2
    prefetch_factor: int = 2
    drop_last: bool = False
    max_text_tokens: int = 48
    model_max_length: int = 8192
    char_max_length: int = 256
    action_hidden_size: int | None = None
    num_action_layers: int | None = None
    num_attention_heads: int | None = None
    state_dropout_prob: float = 0.2


STAGE_DEFAULTS: dict[str, StageDefaults] = {
    # Large, weakly supervised visual-language pretraining. Hand bits are
    # masked, so the hand loss weight is irrelevant but kept small.
    "stage1": StageDefaults(
        learning_rate=3e-5,
        batch_size=32,
        epochs=1,
        sample_stride=10,
        checkpointing_steps=0,
        warmup_ratio=0.05,
        max_text_tokens=48,
    ),
    # Main whole-body posttraining on HE + WBT + HIW with shared 34-D proprio
    # normalization and complete 66-D motion/hand supervision.
    "stage2": StageDefaults(
        learning_rate=3e-5,
        batch_size=32,
        epochs=1,
        sample_stride=5,
        checkpointing_steps=0,
        warmup_ratio=0.02,
        max_text_tokens=48,
    ),
    # Final adaptation. Stop at 50k successful updates and publish only the
    # final step unless a caller explicitly requests periodic checkpoints.
    "stage3": StageDefaults(
        learning_rate=3e-5,
        batch_size=32,
        epochs=30,
        sample_stride=1,
        checkpointing_steps=0,
        max_steps=50_000,
        weight_decay=0.005,
        warmup_ratio=0.01,
        max_text_tokens=48,
    ),
}

STAGE_ALIASES = {
    "egodex": "stage1",
    "pretrain_egodex": "stage1",
    "he_wbt": "stage2",
    "posttrain_he_wbt": "stage2",
    "prop": "stage3",
    "proprietary": "stage3",
    "finetune_prop": "stage3",
    "proprietary_ego": "prop_ego",
    "finetune_prop_ego": "prop_ego",
}


class StageArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args: list[str] | None = None, namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        provided = set()
        for token in (args if args is not None else sys.argv[1:]):
            if token.startswith("--"):
                key = token.split("=", 1)[0].removeprefix("--").replace("-", "_")
                if key.startswith("no_"):
                    key = key.removeprefix("no_")
                provided.add(key)
        parsed = apply_stage_defaults(parsed, provided)
        try:
            return resolve_and_validate_args(parsed, provided)
        except ValueError as exc:
            self.error(str(exc))


def canonical_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def apply_stage_defaults(args: argparse.Namespace, provided: set[str]) -> argparse.Namespace:
    stage = canonical_stage(args.stage)
    defaults = STAGE_DEFAULTS.get(stage)
    if defaults is None:
        return args
    args.stage = stage
    for key, value in asdict(defaults).items():
        if key not in provided:
            setattr(args, key, value)
    if "batch_size" not in provided and args.model_architecture == "qwenoft" and stage in {"stage2", "stage3"}:
        args.batch_size = 24
    return args


def _optional_path(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return None if normalized.lower() in {"", "none", "null"} else normalized


def _tuning_mode(args: argparse.Namespace) -> str:
    if args.qwen_lora:
        return "lora"
    if args.freeze_qwen:
        return "frozen_vlm"
    return "full"


def resolve_and_validate_args(args: argparse.Namespace, provided: set[str]) -> argparse.Namespace:
    """Resolve compatible CLI defaults, then fail before constructing data or models."""
    args.init_ckpt = _optional_path(args.init_ckpt)
    args.resume_from_checkpoint = _optional_path(args.resume_from_checkpoint)
    if args.init_ckpt is not None and args.resume_from_checkpoint is not None:
        raise ValueError("--init-ckpt and --resume-from-checkpoint are mutually exclusive")

    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be > 0")
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes must be > 0")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("--max-train-samples must be > 0")
    if args.episode_selection_seed is None:
        args.episode_selection_seed = args.seed
    if args.episode_selection_seed < 0:
        raise ValueError("--episode-selection-seed must be >= 0")
    if args.episode_fraction is not None:
        if not math.isfinite(args.episode_fraction) or not 0.0 < args.episode_fraction <= 1.0:
            raise ValueError("--episode-fraction must satisfy 0 < fraction <= 1")
        if args.max_episodes is not None:
            raise ValueError("--episode-fraction and --max-episodes are mutually exclusive")
        if args.stage != "stage1":
            raise ValueError("--episode-fraction is only valid for Stage 1")
    if args.stage1_sources:
        if args.stage != "stage1":
            raise ValueError("--stage1-source is only valid for Stage 1")
        if args.egodex_root is not None:
            raise ValueError("--stage1-source and --egodex-root are mutually exclusive")
        normalize_dataset_specs(args.stage1_sources, stage="stage1")
    if args.humanoid_arena_roots and args.humanoid_arena_task != "all":
        raise ValueError("HumanoidArena training only supports --humanoid-arena-task all")
    if args.stage2_sources:
        if args.stage != "stage2":
            raise ValueError("--stage2-source is only valid for Stage 2")
        normalize_dataset_specs(args.stage2_sources, stage="stage2")

    lr_names = ("vlm_learning_rate", "projector_learning_rate", "action_learning_rate")
    configured_lrs = [getattr(args, name) for name in lr_names]
    configured_count = sum(value is not None for value in configured_lrs)
    if configured_count not in {0, len(lr_names)}:
        raise ValueError(
            "--vlm-learning-rate, --projector-learning-rate, and --action-learning-rate "
            "must be provided together"
        )
    if args.learning_rate is None or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0")
    args.explicit_group_learning_rates = configured_count == len(lr_names)
    if not args.explicit_group_learning_rates:
        configured_lrs = [args.learning_rate] * len(lr_names)
    for name, value in zip(lr_names, configured_lrs, strict=True):
        if value is None or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
        setattr(args, name, float(value))

    if args.min_lr is not None and args.min_lr_rate is not None:
        raise ValueError("--min-lr and --min-lr-rate are mutually exclusive")
    if args.min_lr is not None:
        if args.min_lr < 0:
            raise ValueError("--min-lr must be >= 0")
        if len(set(configured_lrs)) != 1:
            raise ValueError("--min-lr is only valid with the legacy single learning rate")
        args.min_lr_rate = float(args.min_lr) / float(configured_lrs[0])
    if args.min_lr_rate is None:
        args.min_lr_rate = 0.0
    if not 0.0 <= args.min_lr_rate <= 1.0:
        raise ValueError("--min-lr-rate must be in [0, 1]")
    args.resolved_min_lrs = {
        group_name: float(lr) * float(args.min_lr_rate)
        for group_name, lr in zip(OPTIMIZER_GROUP_ORDER, configured_lrs, strict=True)
    }

    if args.epochs is None or args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if args.warmup_ratio is not None and not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1]")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if args.checkpointing_steps is None or args.checkpointing_steps < 0:
        raise ValueError("--checkpointing-steps must be >= 0")
    if args.resume_state_steps is None:
        args.resume_state_steps = args.checkpointing_steps
    if args.resume_state_steps < 0:
        raise ValueError("--resume-state-steps must be >= 0")
    if args.resume_state_keep < 1:
        raise ValueError("--resume-state-keep must be >= 1")
    if args.block_shuffle_size is not None and args.block_shuffle_size < 1:
        raise ValueError("--block-shuffle-size must be >= 1")
    if args.expected_num_processes <= 0:
        raise ValueError("--expected-num-processes must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0")
    if args.max_grad_norm is None or args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be > 0")
    if args.model_architecture not in EXPECTED_FULL_INVENTORY:
        raise ValueError(
            "--model-architecture must be one of "
            f"{sorted(EXPECTED_FULL_INVENTORY)}"
        )
    action_defaults = ACTION_HEAD_DEFAULTS[args.model_architecture]
    if args.action_hidden_size is None:
        args.action_hidden_size = action_defaults.hidden_size
    if args.num_action_layers is None:
        args.num_action_layers = action_defaults.num_layers
    if args.action_ffn_multiplier is None:
        args.action_ffn_multiplier = action_defaults.ffn_multiplier

    is_humanoid_arena = bool(args.humanoid_arena_roots)
    if is_humanoid_arena and args.explicit_group_learning_rates:
        if _tuning_mode(args) != "full":
            raise ValueError("explicit three-group learning rates are only valid for HumanoidArena full tuning")
        if args.gradient_accumulation_steps != 1:
            raise ValueError("HumanoidArena G1 requires --gradient-accumulation-steps=1")
        if not math.isclose(float(args.weight_decay), 0.005, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("HumanoidArena G1 requires --weight-decay=0.005")
    return args


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _is_main_process(accelerator: Any | None) -> bool:
    return accelerator is None or accelerator.is_main_process


def _update_training_progress(
    *,
    pbar: Any,
    accelerator: Any | None,
    global_step: int,
    attempted_batches: int,
    skipped_updates: int,
    loss: torch.Tensor,
    lr_scheduler: Any,
) -> None:
    """Reduce loss on every process, then update the progress bar on main."""
    loss_for_log = loss.detach().float()
    if accelerator is not None:
        loss_for_log = accelerator.reduce(loss_for_log, reduction="mean")
    if not _is_main_process(accelerator):
        return
    current_lrs = list(lr_scheduler.get_last_lr())
    lr_postfix = {
        f"lr_{name}": f"{current_lrs[index]:.2e}"
        for index, name in enumerate(OPTIMIZER_GROUP_ORDER)
        if index < len(current_lrs)
    }
    pbar.update(1)
    pbar.set_postfix(
        step=global_step,
        attempted=attempted_batches,
        skipped=skipped_updates,
        loss=f"{loss_for_log.item():.4f}",
        **lr_postfix,
    )


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _log_main(accelerator: Any | None, message: str) -> None:
    if _is_main_process(accelerator):
        print(message, flush=True)


def _log_rank(accelerator: Any | None, message: str) -> None:
    rank = getattr(accelerator, "process_index", 0)
    local_rank = getattr(accelerator, "local_process_index", 0)
    print(f"{message} rank={rank} local_rank={local_rank} pid={os.getpid()}", flush=True)


def _env_is_main_process() -> bool:
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(rank) == 0
    except ValueError:
        return True


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _sanitize_path_part(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value.strip())
    sanitized = sanitized.strip("._-")
    if not sanitized:
        raise ValueError(f"invalid empty path component from {value!r}")
    return sanitized


def _broadcast_run_id(run_id: str, accelerator: Any | None) -> str:
    if accelerator is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return run_id
    payload: list[str | None] = [run_id if accelerator.is_main_process else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("failed to broadcast run id")
    return str(payload[0])


def _run_dir_name(args: argparse.Namespace, accelerator: Any | None) -> str:
    base = str(args.run_id).strip() if args.run_id else time.strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        base = f"{base}_{args.run_name}"
    return _sanitize_path_part(_broadcast_run_id(base, accelerator))


def _format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    return f"{n / 1_000_000:.1f}M"


@dataclass
class TrainingProgress:
    global_step: int = 0
    epoch: int = 0
    batches_consumed_in_epoch: int = 0
    attempted_batches: int = 0
    skipped_updates: int = 0


@dataclass(frozen=True)
class OptimizerGroupInfo:
    name: str
    tensor_count: int
    numel: int
    peak_lr: float
    floor_lr: float
    parameter_names_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parameter_group_for_name(name: str, architecture: str) -> str | None:
    if name.startswith("qwen."):
        return "vlm"
    if name.startswith("action_head."):
        return "action_head"
    projector_prefixes = {
        "qwenpi_v3": ("vlm_projectors.",),
        "qwengr00t": ("vlm_projector.",),
        "qwenoft": ("vlm_projector.",),
    }
    if any(name.startswith(prefix) for prefix in projector_prefixes[architecture]):
        return "projector_base"
    return None


def build_optimizer_param_groups(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[OptimizerGroupInfo]]:
    """Build identity-disjoint, allowlisted optimizer groups with no fallback."""
    architecture = str(args.model_architecture)
    grouped: dict[str, list[torch.nn.Parameter]] = {name: [] for name in OPTIMIZER_GROUP_ORDER}
    grouped_ids: dict[str, set[int]] = {name: set() for name in OPTIMIZER_GROUP_ORDER}
    aliases: dict[str, dict[int, set[str]]] = {name: {} for name in OPTIMIZER_GROUP_ORDER}
    owner_by_id: dict[int, str] = {}
    unknown: list[str] = []
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = model.named_parameters()
    for parameter_name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        group_name = _parameter_group_for_name(parameter_name, architecture)
        if group_name is None:
            unknown.append(parameter_name)
            continue
        parameter_id = id(parameter)
        previous_owner = owner_by_id.get(parameter_id)
        if previous_owner is not None and previous_owner != group_name:
            raise ValueError(
                f"trainable parameter is owned by multiple optimizer groups: "
                f"name={parameter_name!r} groups={previous_owner!r}/{group_name!r}"
            )
        owner_by_id[parameter_id] = group_name
        aliases[group_name].setdefault(parameter_id, set()).add(parameter_name)
        if parameter_id not in grouped_ids[group_name]:
            grouped_ids[group_name].add(parameter_id)
            grouped[group_name].append(parameter)
    if unknown:
        raise ValueError(f"unknown trainable parameter names outside architecture allowlist: {sorted(unknown)}")

    all_trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    grouped_union = set().union(*(grouped_ids[name] for name in OPTIMIZER_GROUP_ORDER))
    if grouped_union != all_trainable:
        raise ValueError(
            f"optimizer groups do not cover trainable parameters: "
            f"missing_ids={len(all_trainable - grouped_union)} extra_ids={len(grouped_union - all_trainable)}"
        )
    for index, left in enumerate(OPTIMIZER_GROUP_ORDER):
        for right in OPTIMIZER_GROUP_ORDER[index + 1 :]:
            overlap = grouped_ids[left].intersection(grouped_ids[right])
            if overlap:
                raise ValueError(f"optimizer groups overlap by identity: {left}/{right} count={len(overlap)}")

    tuning_mode = _tuning_mode(args)
    required_nonempty = {"projector_base", "action_head"}
    if tuning_mode == "full" or (args.qwen_lora and args.train_qwen_lora):
        required_nonempty.add("vlm")
    empty_required = sorted(name for name in required_nonempty if not grouped[name])
    if empty_required:
        raise ValueError(f"required optimizer groups are empty for {tuning_mode}: {empty_required}")
    if tuning_mode in {"frozen_vlm", "lora"} and not args.train_qwen_lora and grouped["vlm"]:
        raise ValueError(f"{tuning_mode} unexpectedly has trainable VLM parameters")

    lrs = {
        "vlm": float(args.vlm_learning_rate),
        "projector_base": float(args.projector_learning_rate),
        "action_head": float(args.action_learning_rate),
    }
    optimizer_groups: list[dict[str, Any]] = []
    inventory: list[OptimizerGroupInfo] = []
    for group_name in OPTIMIZER_GROUP_ORDER:
        parameters = grouped[group_name]
        if not parameters:
            continue
        alias_names = sorted(name for names in aliases[group_name].values() for name in names)
        info = OptimizerGroupInfo(
            name=group_name,
            tensor_count=len(parameters),
            numel=sum(parameter.numel() for parameter in parameters),
            peak_lr=lrs[group_name],
            floor_lr=lrs[group_name] * float(args.min_lr_rate),
            parameter_names_sha256=sha256_bytes(canonical_json_bytes(alias_names)),
        )
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": info.peak_lr,
                "weight_decay": float(args.weight_decay),
                "group_name": group_name,
            }
        )
        inventory.append(info)

    if args.explicit_group_learning_rates and tuning_mode == "full" and args.humanoid_arena_roots:
        actual = {item.name: item.numel for item in inventory}
        expected = EXPECTED_FULL_INVENTORY[architecture]
        if actual != expected:
            raise ValueError(f"full-model optimizer inventory mismatch for {architecture}: expected={expected} actual={actual}")
    return optimizer_groups, inventory


def _clamped_cosine_factor(current_step: int, *, total_steps: int, warmup_steps: int, min_lr_rate: float) -> float:
    step = min(max(int(current_step), 0), total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0 if step < total_steps else min_lr_rate
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr_rate + (1.0 - min_lr_rate) * cosine


def resolved_warmup_steps(args: argparse.Namespace, total_steps: int) -> int:
    warmup_steps = (
        int(args.warmup_steps)
        if args.warmup_steps is not None and args.warmup_steps > 0
        else math.ceil(total_steps * float(args.warmup_ratio or 0.0))
    )
    if warmup_steps > total_steps:
        raise ValueError(f"warmup steps {warmup_steps} exceed scheduler total {total_steps}")
    return warmup_steps


def resolved_scheduler_steps(
    args: argparse.Namespace,
    *,
    prepared_loader_batches: int,
) -> tuple[int, int]:
    if type(prepared_loader_batches) is not int or prepared_loader_batches <= 0:
        raise ValueError(
            f"prepared train loader must have a positive batch count, got {prepared_loader_batches!r}"
        )
    total_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else prepared_loader_batches * int(args.epochs)
    )
    return total_steps, resolved_warmup_steps(args, total_steps)


def build_logical_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_rate: float,
) -> Any:
    if total_steps <= 0:
        raise ValueError("scheduler total_steps must be > 0")
    normalized = scheduler_type.lower()
    if normalized in {"cosine", "cosine_with_min_lr", "clamped_cosine_with_min_lr"}:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _clamped_cosine_factor(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_lr_rate=min_lr_rate,
            ),
        )
    if min_lr_rate != 0.0:
        raise ValueError(f"--min-lr-rate requires a cosine scheduler, got {scheduler_type!r}")
    from transformers import get_scheduler

    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


class TeeStream:
    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()


class RunLogTee:
    def __init__(self, out_dir: Path, *, enabled: bool) -> None:
        self.out_dir = out_dir
        self.enabled = enabled
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._log_file: TextIO | None = None

    def __enter__(self) -> "RunLogTee":
        if not self.enabled:
            return self
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = (self.out_dir / "train.log").open("a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(self._stdout, self._log_file)  # type: ignore[assignment]
        sys.stderr = TeeStream(self._stderr, self._log_file)  # type: ignore[assignment]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and self._log_file is not None:
            self._log_file.write("\n")
            traceback.print_exception(exc_type, exc, tb, file=self._log_file)
            self._log_file.flush()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        if self._log_file is not None:
            self._log_file.close()


class RunWriterLock:
    """Single-writer marker consumed by both training and evaluation discovery."""

    SCHEMA_VERSION = 1
    FILE_NAME = ".trainer.lock"

    def __init__(self, run_dir: Path, *, enabled: bool) -> None:
        self.run_dir = run_dir.resolve()
        self.enabled = enabled
        self.path = self.run_dir / self.FILE_NAME
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> None:
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "token": self.token,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "run_dir": str(self.run_dir),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        encoded = canonical_json_bytes(payload)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            try:
                existing = self.path.read_text(encoding="utf-8")
            except OSError:
                existing = "<unreadable>"
            raise RuntimeError(
                f"refusing a second writer because {self.path} exists; stale locks require explicit audit: {existing}"
            ) from exc
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                self.path.unlink()
            except OSError:
                pass
            raise
        self.acquired = True

    def release(self) -> None:
        if not self.enabled or not self.acquired:
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict) and payload.get("token") == self.token:
            self.path.unlink()
            self.acquired = False


T = TypeVar("T")


def _main_process_call(accelerator: Any | None, action: Callable[[], T]) -> T:
    if accelerator is None or not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return action()
    payload: list[dict[str, Any] | None] = [None]
    if accelerator.is_main_process:
        try:
            payload[0] = {"ok": True, "value": action()}
        except Exception as exc:
            payload[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    torch.distributed.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error") if isinstance(result, dict) else "rank 0 returned no result"
        raise RuntimeError(str(error))
    return result["value"]  # type: ignore[return-value]


def _payload_fingerprint(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


_RESUME_CONFIG_EXCLUDED_FIELDS = frozenset(
    {
        "init_ckpt",
        "force_data_audit",
        "output_dir",
        "resume_from_checkpoint",
        "run_id",
        "run_name",
    }
)


def _json_contract_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_contract_value(item) for item in value]
    raise ValueError(f"unsupported resolved-config value: {type(value).__name__}")


def _resolved_training_contract(args: argparse.Namespace) -> dict[str, Any]:
    resolved = {
        key: _json_contract_value(value)
        for key, value in sorted(vars(args).items())
        if key not in _RESUME_CONFIG_EXCLUDED_FIELDS
    }
    for stage, field in (("stage1", "stage1_sources"), ("stage2", "stage2_sources")):
        values = getattr(args, field)
        if values:
            resolved[field] = [
                [spec.kind, str(spec.root)]
                for spec in normalize_dataset_specs(values, stage=stage)
            ]
    resolved.update(
        {
            "tuning_mode": _tuning_mode(args),
            "resolved_action_hidden_size": args.action_hidden_size,
            "resolved_num_action_layers": args.num_action_layers,
            "resolved_state_dropout_prob": (
                0.2 if args.state_dropout_prob is None else args.state_dropout_prob
            ),
        }
    )
    return resolved


def _file_contract(path: str | Path, *, required: bool = True) -> dict[str, Any] | None:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"required contract file not found: {source}")
        return None
    return {"path": str(source), "size": source.stat().st_size, "sha256": sha256_file(source)}


def _cumulative_stats_contract(args: argparse.Namespace) -> tuple[str, dict[str, Path], str | None] | None:
    data = Path(args.dataset_root) if args.dataset_root else Path("resource/dataset")
    stage1 = {
        "EGODEX": data / "EgoDex-cache" / "g1_proprio",
        "EGOVERSE": data / "egoverse-retargeted-cache" / "g1_proprio",
        "BEHAVIOR1K": data / "behavior-1k-2025-challenge-cache" / "g1_proprio",
        "XPERIENCE": data / "xperience-10m-cache" / "g1_proprio",
    }
    if args.stage1_sources:
        for spec in normalize_dataset_specs(args.stage1_sources, stage="stage1"):
            stage1[{
                "egodex": "EGODEX",
                "egoverse": "EGOVERSE",
                "behavior1k": "BEHAVIOR1K",
                "xperience": "XPERIENCE",
            }[spec.kind]] = (
                spec.root.with_name(f"{spec.root.name}-cache") / "g1_proprio"
                if spec.kind == "egodex"
                else spec.root / "g1_proprio"
            )
    if args.stage == "stage1":
        return "stage1", stage1, None

    stage12 = {
        **stage1,
        "HE": data / "humanoid-everyday-cache" / "g1_proprio",
        "WBT": data / "G1_WBT" / "g1_proprio",
        "HIW": data / "HIW-500" / "g1_proprio",
    }
    if args.stage2_sources:
        for spec in normalize_dataset_specs(args.stage2_sources, stage="stage2"):
            stage12[{"he": "HE", "wbt": "WBT", "hiw": "HIW"}[spec.kind]] = (
                spec.root.with_name(f"{spec.root.name}-cache") / "g1_proprio"
                if spec.kind == "he"
                else spec.root / "g1_proprio"
            )
    if args.stage == "stage2":
        return "stage12", stage12, None
    if args.stage == "stage3" and args.humanoid_arena_roots:
        if len(args.humanoid_arena_roots) != 1:
            raise ValueError("cumulative HumanoidArena stats require exactly one --humanoid-arena-roots entry")
        return (
            "stage123_humanoid_arena",
            {**stage12, "HUMANOID_ARENA": Path(args.humanoid_arena_roots[0]) / "g1_proprio"},
            "all",
        )
    return None


def _validation_sources(args: argparse.Namespace, expected_sources: dict[str, Path]) -> dict[str, Any]:
    data = Path(args.dataset_root) if args.dataset_root else Path("resource/dataset")
    stage1_roots = (
        {
            spec.kind: spec.root
            for spec in normalize_dataset_specs(args.stage1_sources, stage="stage1")
        }
        if args.stage1_sources
        else None
    )
    stage2_roots = (
        {
            spec.kind: spec.root
            for spec in normalize_dataset_specs(args.stage2_sources, stage="stage2")
        }
        if args.stage2_sources
        else None
    )
    arena_root = Path(args.humanoid_arena_roots[0]) if args.humanoid_arena_roots else None
    available = build_validation_sources(
        data,
        stage1_roots=stage1_roots,
        stage2_roots=stage2_roots,
        humanoid_arena_root=arena_root,
    )
    selected = {tag: available[tag] for tag in expected_sources}
    for tag, expected_root in expected_sources.items():
        if selected[tag].proprio_root.resolve() != expected_root.resolve():
            raise ValueError(f"{tag} validation/stats proprio root mismatch")
    return selected


def _current_validation_tags(args: argparse.Namespace) -> list[str]:
    if args.stage == "stage1":
        kinds = (
            [spec.kind for spec in normalize_dataset_specs(args.stage1_sources, stage="stage1")]
            if args.stage1_sources
            else ["egodex"]
        )
        tags = {
            "egodex": "EGODEX",
            "egoverse": "EGOVERSE",
            "behavior1k": "BEHAVIOR1K",
            "xperience": "XPERIENCE",
        }
        return [tags[kind] for kind in kinds]
    if args.stage == "stage2":
        kinds = (
            [spec.kind for spec in normalize_dataset_specs(args.stage2_sources, stage="stage2")]
            if args.stage2_sources
            else ["he", "wbt"]
        )
        tags = {"he": "HE", "wbt": "WBT", "hiw": "HIW"}
        return [tags[kind] for kind in kinds]
    if args.stage == "stage3" and args.humanoid_arena_roots:
        return ["HUMANOID_ARENA"]
    return []


def _validation_summaries(manifest: dict[str, Any], tags: list[str]) -> list[dict[str, Any]]:
    fields = (
        "dataset_id",
        "n_files",
        "n_frames",
        "missing",
        "extra",
        "header_mismatch",
        "length_mismatch",
        "inventory_sha256",
    )
    return [{key: manifest["sources"][tag][key] for key in fields} for tag in tags]

def _default_cumulative_stats_path(args: argparse.Namespace, stage_scope: str) -> str:
    data = Path(args.dataset_root) if args.dataset_root else Path("resource/dataset")
    filename = {
        "stage1": "stage1_proprio_stats_union_sonic.json",
        "stage12": "stage12_proprio_stats_union_sonic.json",
        "stage123_humanoid_arena": "stage123_humanoid_arena_proprio_stats_union_sonic.json",
    }[stage_scope]
    return str(data / "meta" / filename)


def _snapshot_proprio_stats(source: str | Path, run_dir: Path) -> str:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"proprio stats not found: {source_path}")
    destination = run_dir / "proprio_stats_sonic.json"
    if destination.exists():
        raise FileExistsError(f"new run already has a stats snapshot: {destination}")
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    try:
        shutil.copy2(source_path, temporary)
        if sha256_file(temporary) != sha256_file(source_path):
            raise OSError(f"proprio stats snapshot changed while copying: {source_path}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return str(destination.resolve())


def _validate_proprio_inventory(dataset: Any) -> list[dict[str, Any]] | None:
    adapters = getattr(dataset, "datasets", [dataset])
    summaries: list[dict[str, Any]] = []
    for adapter in adapters:
        validate = getattr(adapter, "validate_proprio_inventory", None)
        if not callable(validate):
            continue
        summary = validate()
        if not isinstance(summary, dict):
            raise ValueError(f"{type(adapter).__name__}.validate_proprio_inventory() returned no summary")
        failures = {
            key: summary.get(key)
            for key in ("missing", "extra", "header_mismatch", "length_mismatch")
            if summary.get(key) != 0
        }
        if failures:
            raise ValueError(
                f"{summary.get('dataset_id', type(adapter).__name__)} proprio inventory failed: "
                f"{failures}; examples={summary.get('examples', [])}"
            )
        summaries.append(summary)
    return summaries or None

def _source_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    source_paths: list[Path] = []
    for relative_root in (
        "src/huge/data",
        "src/huge/models",
        "src/huge/training",
    ):
        source_paths.extend(sorted((repo_root / relative_root).rglob("*.py")))
    source_paths.extend(sorted((repo_root / "scripts/train").glob("*.sh")))
    source_paths.extend(
        path
        for path in (repo_root / "pyproject.toml", repo_root / "uv.lock")
        if path.is_file()
    )
    files = [
        _file_contract(path)
        for path in sorted(set(source_paths), key=lambda item: item.as_posix())
    ]
    base_vlm = Path(args.base_vlm_path).expanduser().resolve()
    model_files: list[dict[str, Any]] = []
    for name in (
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.json",
    ):
        contract = _file_contract(
            base_vlm / name,
            required=bool(args.use_qwen and name == "config.json"),
        )
        if contract is not None:
            model_files.append(contract)
    packages: dict[str, str | None] = {}
    for name in ("accelerate", "deepspeed", "numpy", "torch", "transformers"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    deepspeed = _file_contract(
        args.deepspeed_config,
        required=bool(args.use_deepspeed and args.deepspeed_config),
    )
    payload = {
        "source_git_commit": _git_commit(),
        "files": files,
        "base_vlm_files": model_files,
        "deepspeed_config": deepspeed,
        "software": {
            "python": sys.version,
            "packages": packages,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
    }
    return {**payload, "fingerprint": _payload_fingerprint(payload)}


def _data_contract(
    args: argparse.Namespace,
    *,
    dataset: Any,
    train_length: int,
    stats_path: str | None,
    proprio_inventory: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    source_values = args.stage1_sources if args.stage == "stage1" else args.stage2_sources
    normalized_specs = normalize_dataset_specs(source_values, stage=args.stage) if source_values else []
    root_values = [
        *(args.dataset_root and [args.dataset_root] or []),
        *(args.egodex_root and [args.egodex_root] or []),
        *(spec.root for spec in normalized_specs),
        *(args.prop_roots or []),
        *(args.prop_ego_roots or []),
        *(args.humanoid_arena_roots or []),
    ]
    for raw_root in root_values:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {root}")
        markers = []
        for relative in ("_SUCCESS", "meta/conversion_manifest.json", "meta/language_map.json"):
            marker = _file_contract(root / relative, required=False)
            if marker is not None:
                markers.append(marker)
        roots.append({"path": str(root), "markers": markers})
    normalized_sources = [
        {
            "kind": spec.kind,
            "logical_root": str(spec.root),
            "resolved_root": str(spec.root.expanduser().resolve()),
        }
        for spec in normalized_specs
    ]
    spans_fn = getattr(dataset, "episode_spans", None)
    spans_contract = None
    if callable(spans_fn):
        spans = spans_fn()
        spans_contract = {
            "count": len(spans),
            "sha256": hashlib.sha256(spans.tobytes(order="C")).hexdigest(),
        }
    payload = {
        "dataset_type": type(dataset).__name__,
        "dataset_contract": (
            dataset.data_contract()
            if callable(getattr(dataset, "data_contract", None))
            else None
        ),
        "train_len": int(train_length),
        "normalized_sources": normalized_sources,
        "episode_spans": spans_contract,
        "roots": roots,
        "stats": _file_contract(stats_path, required=True) if stats_path else None,
        "proprio_inventory": proprio_inventory,
        "sample_stride": int(args.sample_stride),
        "max_episodes": args.max_episodes,
        "episode_fraction": args.episode_fraction,
        "episode_selection_seed": args.episode_selection_seed,
        "max_train_samples": args.max_train_samples,
        "task": args.humanoid_arena_task,
        "block_shuffle_size": args.block_shuffle_size,
        "action_horizon": args.action_horizon,
        "drop_last": bool(args.drop_last),
    }
    return {**payload, "fingerprint": _payload_fingerprint(payload)}


def _data_summary_lines(contract: dict[str, Any]) -> list[str]:
    dataset_contract = contract.get("dataset_contract")
    if not isinstance(dataset_contract, dict):
        return []
    sources = (
        dataset_contract.get("sources")
        if isinstance(dataset_contract.get("sources"), list)
        else [dataset_contract]
    )
    total = sum(int(source.get("samples", 0)) for source in sources)
    lines: list[str] = []
    for source in sources:
        samples = int(source.get("samples", 0))
        source_id = source.get("source_id", source.get("dataset_id", source.get("dataset_type")))
        episodes = source.get("selected_episode_count", source.get("episodes", "unknown"))
        ratio = samples / total if total else 0.0
        digest = str(source.get("selected_episode_ids_sha256", ""))[:12] or "n/a"
        segments = source.get("selected_segment_count", source.get("segments", "unknown"))
        token_rows = source.get("selected_token_rows", "unknown")
        line = (
            f"source={source_id} episodes={episodes} segments={segments} "
            f"token_rows={token_rows} samples={samples} ratio={ratio:.6f} "
            f"selection_digest={digest}"
        )
        if "source_fps" in source:
            line += (
                f" source_fps={source['source_fps']} logical_control_hz={source['logical_control_hz']}"
                f" motion_dim={source['motion_dim']} hand_mode={source['hand_mode']}"
                f" state_mode={source['state_mode']} video_backend={source['video_backend']}"
            )
        lines.append(line)
    return lines


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"{label} must be a JSON object: {path}")
    return payload


def _read_resume_latest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CheckpointValidationError(f"resume LATEST is missing or invalid: {path}")
    latest = _read_json_object(path, label="resume LATEST")
    expected_keys = {
        "schema_version",
        "state_dir",
        "global_step",
        "manifest_sha256",
        "publication_file_count",
    }
    if set(latest) != expected_keys:
        raise CheckpointValidationError(f"invalid resume LATEST schema: {sorted(latest)}")
    if (
        type(latest.get("schema_version")) is not int
        or latest.get("schema_version") != TRAINER_STATE_SCHEMA_VERSION
    ):
        raise CheckpointValidationError(
            f"unsupported resume LATEST schema: {latest.get('schema_version')!r}"
        )
    global_step = latest.get("global_step")
    if type(global_step) is not int or global_step < 0:
        raise CheckpointValidationError(f"resume LATEST has invalid global_step: {global_step!r}")
    state_dir = latest.get("state_dir")
    if not isinstance(state_dir, str) or state_dir != f"step_{global_step}":
        raise CheckpointValidationError(f"resume LATEST has invalid state_dir: {state_dir!r}")
    manifest_sha256 = latest.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in manifest_sha256)
    ):
        raise CheckpointValidationError(
            f"resume LATEST has invalid manifest_sha256: {manifest_sha256!r}"
        )
    publication_file_count = latest.get("publication_file_count")
    if type(publication_file_count) is not int or publication_file_count <= 0:
        raise CheckpointValidationError(
            f"resume LATEST has invalid publication_file_count: {publication_file_count!r}"
        )
    return latest


def _resolve_resume_state_path(raw_path: str | Path) -> tuple[Path, dict[str, Any]]:
    requested = Path(raw_path).expanduser()
    if requested.is_symlink():
        raise CheckpointValidationError(f"resume path may not be a symlink: {requested}")
    candidate = requested.resolve()
    if (
        candidate.is_dir()
        and (candidate / "model.safetensors").is_file()
        and not (candidate / "trainer_state.json").is_file()
    ):
        raise CheckpointValidationError(
            f"{candidate} is a portable model checkpoint, not a full resume state; use --init-ckpt for a new run"
        )

    direct_state: Path | None = None
    if candidate.is_dir() and (candidate / "_resume_states").is_dir():
        resume_root = candidate / "_resume_states"
    elif candidate.is_dir() and candidate.name == "_resume_states":
        resume_root = candidate
    elif candidate.is_file():
        if candidate.name != "LATEST.json":
            raise CheckpointValidationError(
                f"resume path must be a run directory, state directory, or LATEST.json: {candidate}"
            )
        resume_root = candidate.parent
    elif candidate.is_dir():
        direct_state = candidate
        resume_root = candidate.parent
    else:
        raise CheckpointValidationError(f"resume path does not exist: {candidate}")
    if resume_root.name != "_resume_states":
        raise CheckpointValidationError(f"resume state must be under an _resume_states directory: {candidate}")

    latest = _read_resume_latest(resume_root / "LATEST.json")
    state_dir = (resume_root / latest["state_dir"]).resolve()
    if state_dir.parent != resume_root.resolve():
        raise CheckpointValidationError(f"resume LATEST escapes its state root: {state_dir}")
    if direct_state is not None and direct_state != state_dir:
        raise CheckpointValidationError(
            f"direct resume state must equal current LATEST: requested={direct_state} latest={state_dir}"
        )
    return state_dir, latest


def _resume_uses_deepspeed(trainer_state: dict[str, Any]) -> bool:
    resolved_config = trainer_state.get("resolved_config")
    if isinstance(resolved_config, dict) and type(resolved_config.get("use_deepspeed")) is bool:
        return bool(resolved_config["use_deepspeed"])
    runtime = trainer_state.get("runtime")
    return isinstance(runtime, dict) and isinstance(runtime.get("deepspeed_config"), str)


def _validate_resume_progress_and_scheduler(path: Path, trainer_state: dict[str, Any]) -> None:
    progress = trainer_state.get("progress")
    expected_progress_fields = {
        "attempted_batches",
        "batches_consumed_in_epoch",
        "epoch",
        "global_step",
        "skipped_updates",
        "successful_updates",
    }
    if not isinstance(progress, dict) or set(progress) != expected_progress_fields:
        raise CheckpointValidationError(
            f"trainer state has invalid progress schema: "
            f"{sorted(progress) if isinstance(progress, dict) else progress!r}"
        )
    for name in expected_progress_fields:
        value = progress[name]
        if type(value) is not int or value < 0:
            raise CheckpointValidationError(f"trainer state progress {name} is invalid: {value!r}")
    global_step = progress["global_step"]
    attempted_batches = progress["attempted_batches"]
    skipped_updates = progress["skipped_updates"]
    if progress["successful_updates"] != global_step:
        raise CheckpointValidationError(
            "trainer state successful_updates must equal global_step: "
            f"{progress['successful_updates']} != {global_step}"
        )
    if attempted_batches != global_step + skipped_updates:
        raise CheckpointValidationError(
            "trainer state attempted_batches must equal successful plus skipped updates: "
            f"{attempted_batches} != {global_step} + {skipped_updates}"
        )
    if progress["batches_consumed_in_epoch"] > attempted_batches:
        raise CheckpointValidationError(
            "trainer state batches_consumed_in_epoch exceeds attempted_batches: "
            f"{progress['batches_consumed_in_epoch']} > {attempted_batches}"
        )

    scheduler = trainer_state.get("scheduler")
    expected_scheduler_fields = {
        "absolute_max_steps",
        "current_lrs",
        "min_lr_rate",
        "total_steps",
        "type",
        "warmup_steps",
    }
    if not isinstance(scheduler, dict) or set(scheduler) != expected_scheduler_fields:
        raise CheckpointValidationError(
            f"trainer state has invalid scheduler schema: "
            f"{sorted(scheduler) if isinstance(scheduler, dict) else scheduler!r}"
        )
    total_steps = scheduler["total_steps"]
    warmup_steps = scheduler["warmup_steps"]
    if type(total_steps) is not int or total_steps <= 0:
        raise CheckpointValidationError(f"trainer state scheduler total_steps is invalid: {total_steps!r}")
    if type(warmup_steps) is not int or warmup_steps < 0 or warmup_steps > total_steps:
        raise CheckpointValidationError(f"trainer state scheduler warmup_steps is invalid: {warmup_steps!r}")
    absolute_max_steps = scheduler["absolute_max_steps"]
    if absolute_max_steps is not None and (
        type(absolute_max_steps) is not int or absolute_max_steps <= 0
    ):
        raise CheckpointValidationError(
            f"trainer state scheduler absolute_max_steps is invalid: {absolute_max_steps!r}"
        )
    min_lr_rate = scheduler["min_lr_rate"]
    if (
        type(min_lr_rate) not in (int, float)
        or not math.isfinite(float(min_lr_rate))
        or not 0.0 <= float(min_lr_rate) <= 1.0
    ):
        raise CheckpointValidationError(
            f"trainer state scheduler min_lr_rate is invalid: {min_lr_rate!r}"
        )
    if not isinstance(scheduler["type"], str) or not scheduler["type"]:
        raise CheckpointValidationError(f"trainer state scheduler type is invalid: {scheduler['type']!r}")

    optimizer_groups = trainer_state.get("optimizer_groups")
    current_lrs = scheduler["current_lrs"]
    if (
        not isinstance(optimizer_groups, list)
        or not optimizer_groups
        or not isinstance(current_lrs, list)
        or len(current_lrs) != len(optimizer_groups)
        or any(
            type(value) not in (int, float)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in current_lrs
        )
    ):
        raise CheckpointValidationError(
            f"trainer state scheduler current_lrs is invalid: {current_lrs!r}"
        )
    peak_lrs = [item.get("peak_lr") if isinstance(item, dict) else None for item in optimizer_groups]
    if any(
        type(value) not in (int, float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in peak_lrs
    ):
        raise CheckpointValidationError(f"trainer state optimizer peak LRs are invalid: {peak_lrs!r}")

    try:
        scheduler_state = torch.load(
            path / "scheduler.bin",
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise CheckpointValidationError(
            f"resume scheduler state is unreadable: {path / 'scheduler.bin'}"
        ) from exc
    if not isinstance(scheduler_state, dict):
        raise CheckpointValidationError("resume scheduler state must be a dictionary")
    if scheduler_state.get("last_epoch") != global_step:
        raise CheckpointValidationError(
            "resume scheduler last_epoch does not match global_step: "
            f"{scheduler_state.get('last_epoch')!r} != {global_step}"
        )
    if scheduler_state.get("_step_count") != global_step + 1:
        raise CheckpointValidationError(
            "resume scheduler _step_count does not match global_step: "
            f"{scheduler_state.get('_step_count')!r} != {global_step + 1}"
        )
    if scheduler_state.get("base_lrs") != peak_lrs:
        raise CheckpointValidationError(
            "resume scheduler base_lrs do not match optimizer peak LRs: "
            f"{scheduler_state.get('base_lrs')!r} != {peak_lrs!r}"
        )
    if scheduler_state.get("_last_lr") != current_lrs:
        raise CheckpointValidationError(
            "resume scheduler _last_lr does not match trainer current_lrs: "
            f"{scheduler_state.get('_last_lr')!r} != {current_lrs!r}"
        )


def _validate_resume_state_contents(path: Path, trainer_state: dict[str, Any]) -> None:
    required_files = {"trainer_state.json", "scheduler.bin"}
    data_state = trainer_state.get("data")
    if isinstance(data_state, dict) and isinstance(data_state.get("sampler"), dict):
        required_files.add("custom_checkpoint_0.pkl")
    missing = sorted(
        name
        for name in required_files
        if not (path / name).is_file() or (path / name).is_symlink()
    )
    if missing:
        raise CheckpointValidationError(f"resume state is missing required Accelerate files: {missing}")
    if _resume_uses_deepspeed(trainer_state):
        latest_tag = path / "latest"
        model_state_root = path / "pytorch_model"
        if (
            not latest_tag.is_file()
            or latest_tag.is_symlink()
            or not model_state_root.is_dir()
            or model_state_root.is_symlink()
        ):
            raise CheckpointValidationError(
                "DeepSpeed resume state requires regular latest and pytorch_model entries"
            )
    else:
        required_accelerate_files = {"model.safetensors", "optimizer.bin"}
        missing_accelerate_files = sorted(
            name
            for name in required_accelerate_files
            if not (path / name).is_file() or (path / name).is_symlink()
        )
        if missing_accelerate_files:
            raise CheckpointValidationError(
                "non-DeepSpeed resume state is missing model/optimizer files: "
                f"{missing_accelerate_files}"
            )


def inspect_resume_state(raw_path: str | Path) -> dict[str, Any]:
    state_dir, latest = _resolve_resume_state_path(raw_path)
    trainer_state_path = state_dir / "trainer_state.json"
    if not trainer_state_path.is_file() or trainer_state_path.is_symlink():
        raise CheckpointValidationError(f"trainer state is missing or invalid: {trainer_state_path}")
    trainer_state = _read_json_object(trainer_state_path, label="trainer state")
    if (
        type(trainer_state.get("schema_version")) is not int
        or trainer_state.get("schema_version") != TRAINER_STATE_SCHEMA_VERSION
        or trainer_state.get("kind") != RESUME_STATE_KIND
    ):
        raise CheckpointValidationError(
            f"unsupported trainer state schema/kind: "
            f"{trainer_state.get('schema_version')!r}/{trainer_state.get('kind')!r}"
        )
    progress = trainer_state.get("progress")
    global_step = progress.get("global_step") if isinstance(progress, dict) else None
    if type(global_step) is not int or global_step < 0:
        raise CheckpointValidationError(f"trainer state has invalid global_step: {global_step!r}")
    runtime = trainer_state.get("runtime")
    world_size = runtime.get("world_size") if isinstance(runtime, dict) else None
    if type(world_size) is not int or world_size <= 0:
        raise CheckpointValidationError(f"trainer state has invalid world_size: {world_size!r}")

    _validate_resume_state_contents(state_dir, trainer_state)
    _validate_resume_progress_and_scheduler(state_dir, trainer_state)
    publication = validate_published_directory(
        state_dir,
        expected_kind=RESUME_STATE_KIND,
        expected_world_size=world_size,
    )
    manifest_path = state_dir / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if latest["manifest_sha256"] != manifest_sha256:
        raise CheckpointValidationError(f"resume LATEST manifest digest mismatch: {state_dir}")
    if latest["global_step"] != global_step:
        raise CheckpointValidationError(
            f"resume LATEST global_step mismatch: latest={latest['global_step']} trainer={global_step}"
        )
    if latest["publication_file_count"] != len(publication["files"]):
        raise CheckpointValidationError(
            "resume LATEST publication_file_count mismatch: "
            f"latest={latest['publication_file_count']} manifest={len(publication['files'])}"
        )

    run_dir = Path(str(trainer_state.get("run_dir", ""))).expanduser().resolve()
    if state_dir.parent.name != "_resume_states" or state_dir.parent.parent.resolve() != run_dir:
        raise CheckpointValidationError(f"resume state is not under its declared run_dir: {state_dir} vs {run_dir}")
    if state_dir.name != f"step_{global_step}":
        raise CheckpointValidationError(f"resume directory step does not match trainer state: {state_dir.name}")
    expected_publication = {
        "global_step": global_step,
        "world_size": world_size,
        "model_architecture": trainer_state.get("model_architecture"),
        "tuning_mode": trainer_state.get("tuning_mode"),
        "run_id": trainer_state.get("run_id"),
    }
    for key, expected in expected_publication.items():
        if publication.get(key) != expected:
            raise CheckpointValidationError(
                f"resume publication {key} mismatch: manifest={publication.get(key)!r} trainer={expected!r}"
            )

    run_files = trainer_state.get("run_files")
    if not isinstance(run_files, dict) or not run_files:
        raise CheckpointValidationError("trainer state has invalid run_files contract")
    for name, expected_sha256 in run_files.items():
        relative = Path(name) if isinstance(name, str) else Path()
        if (
            not isinstance(name, str)
            or relative.is_absolute()
            or not relative.parts
            or relative.as_posix() != name
            or ".." in relative.parts
            or not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise CheckpointValidationError(f"trainer state has invalid run file contract: {name!r}")
        path = run_dir / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha256:
            raise CheckpointValidationError(f"resume run file changed or is missing: {path}")
    return {
        "state_dir": str(state_dir),
        "trainer_state": trainer_state,
        "latest": latest,
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"resume contract mismatch for {label}: state={actual!r} current={expected!r}")


def validate_resume_static(
    trainer_state: dict[str, Any],
    args: argparse.Namespace,
    *,
    source_contract: dict[str, Any],
) -> Path:
    run_dir = Path(str(trainer_state["run_dir"])).resolve()
    expected_parent = (Path(args.output_dir).expanduser().resolve() / args.stage)
    _require_equal("run_dir parent", run_dir.parent, expected_parent)
    _require_equal("stage", trainer_state.get("stage"), args.stage)
    _require_equal("model architecture", trainer_state.get("model_architecture"), args.model_architecture)
    _require_equal("task", trainer_state.get("task"), args.humanoid_arena_task)
    _require_equal("tuning mode", trainer_state.get("tuning_mode"), _tuning_mode(args))
    _require_equal("max_steps", trainer_state.get("scheduler", {}).get("absolute_max_steps"), args.max_steps)
    _require_equal("world size", trainer_state.get("runtime", {}).get("world_size"), args.expected_num_processes)
    _require_equal("per-device batch", trainer_state.get("runtime", {}).get("batch_size"), args.batch_size)
    expected_global_batch = args.batch_size * args.expected_num_processes * args.gradient_accumulation_steps
    _require_equal("global batch", trainer_state.get("runtime", {}).get("global_batch"), expected_global_batch)
    _require_equal(
        "gradient accumulation",
        trainer_state.get("runtime", {}).get("gradient_accumulation_steps"),
        args.gradient_accumulation_steps,
    )
    _require_equal("seed", trainer_state.get("data", {}).get("seed"), args.seed)
    _require_equal("source fingerprint", trainer_state.get("source_fingerprint"), source_contract["fingerprint"])
    _require_equal("mixed precision", trainer_state.get("runtime", {}).get("mixed_precision"), args.mixed_precision)
    _require_equal(
        "DeepSpeed config",
        trainer_state.get("runtime", {}).get("deepspeed_config_sha256"),
        (source_contract.get("deepspeed_config") or {}).get("sha256"),
    )
    _require_equal(
        "resolved training config",
        trainer_state.get("resolved_config"),
        _resolved_training_contract(args),
    )
    expected_lrs = {
        "vlm": args.vlm_learning_rate,
        "projector_base": args.projector_learning_rate,
        "action_head": args.action_learning_rate,
    }
    expected_fields = {
        "floor_lr", "name", "numel", "parameter_names_sha256", "peak_lr", "tensor_count"
    }
    saved_groups = trainer_state.get("optimizer_groups")
    if not isinstance(saved_groups, list) or not saved_groups:
        raise ValueError("resume optimizer_groups must be a non-empty ordered list")
    saved_names: list[str] = []
    for item in saved_groups:
        if not isinstance(item, dict) or set(item) != expected_fields:
            raise ValueError(f"resume optimizer group has invalid schema: {item!r}")
        name = item.get("name")
        digest = item.get("parameter_names_sha256")
        if (
            name not in expected_lrs
            or type(item.get("tensor_count")) is not int
            or item["tensor_count"] <= 0
            or type(item.get("numel")) is not int
            or item["numel"] <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"resume optimizer group is invalid: {item!r}")
        _require_equal(f"optimizer group {name} peak LR", item.get("peak_lr"), expected_lrs[name])
        _require_equal(
            f"optimizer group {name} floor LR",
            item.get("floor_lr"),
            expected_lrs[name] * float(args.min_lr_rate),
        )
        saved_names.append(name)
    if len(saved_names) != len(set(saved_names)):
        raise ValueError(f"resume optimizer group names are duplicated: {saved_names}")
    ordered_saved = [name for name in OPTIMIZER_GROUP_ORDER if name in saved_names]
    _require_equal("optimizer group order", saved_names, ordered_saved)
    if _tuning_mode(args) == "full":
        _require_equal("full optimizer group inventory", saved_names, list(OPTIMIZER_GROUP_ORDER))
    _require_equal("AdamW betas", trainer_state.get("adamw", {}).get("betas"), list(ADAMW_BETAS))
    _require_equal("AdamW eps", trainer_state.get("adamw", {}).get("eps"), ADAMW_EPS)
    _require_equal("AdamW weight_decay", trainer_state.get("adamw", {}).get("weight_decay"), args.weight_decay)
    if args.run_id is not None:
        _require_equal("run_id", trainer_state.get("run_id"), args.run_id)
    if args.run_name is not None:
        raise ValueError("--run-name is not valid with --resume-from-checkpoint")
    progress = trainer_state.get("progress", {})
    global_step = progress.get("global_step")
    epoch = progress.get("epoch")
    batch_offset = progress.get("batches_consumed_in_epoch")
    if (
        type(global_step) is not int or global_step < 0
        or type(epoch) is not int or epoch < 0
        or type(batch_offset) is not int or batch_offset < 0
    ):
        raise ValueError(f"resume progress is invalid: {progress!r}")
    if args.num_workers == 0 and batch_offset > 0:
        raise ValueError(
            "exact mid-epoch resume requires --num-workers>0 so augmentation RNG can be replayed independently"
        )
    if args.max_steps is not None and global_step >= args.max_steps:
        raise ValueError("resume state already reached the absolute max_steps endpoint")
    if args.max_steps is None and epoch >= args.epochs:
        raise ValueError("resume state already reached the configured epoch endpoint")
    if os.path.lexists(run_dir / RunWriterLock.FILE_NAME):
        raise ValueError(
            f"resume run already has a writer marker: {run_dir / RunWriterLock.FILE_NAME}"
        )
    return run_dir


def _prepare_loader_epoch(
    train_loader: Any,
    train_sampler: Sampler[int] | None,
    loader_generator: torch.Generator,
    *,
    seed: int,
    epoch: int,
) -> None:
    loader_generator.manual_seed(int(seed) + int(epoch))
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(epoch)  # type: ignore[attr-defined]
    if hasattr(train_loader, "set_epoch"):
        train_loader.set_epoch(epoch)


def _batch_trace(batch: dict[str, Any]) -> tuple[str, str, str]:
    identities = list(
        zip(
            batch.get("dataset_id", []),
            batch.get("episode_id", []),
            batch.get("frame_idx", []),
            strict=False,
        )
    )
    sources = ",".join(sorted({str(identity[0]) for identity in identities}))
    identity_sha256 = _payload_fingerprint(identities)
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        if torch.is_tensor(value):
            tensor = value.detach().contiguous().cpu()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(canonical_json_bytes(list(tensor.shape)))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for nested_key in sorted(value):
                nested = value[nested_key]
                if torch.is_tensor(nested):
                    tensor = nested.detach().contiguous().cpu()
                    digest.update(f"{key}.{nested_key}".encode("utf-8"))
                    digest.update(str(tensor.dtype).encode("utf-8"))
                    digest.update(canonical_json_bytes(list(tensor.shape)))
                    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return sources, identity_sha256, digest.hexdigest()


def _new_staging_directory(destination: Path, accelerator: Any) -> Path:
    def create() -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.parent / f".staging-{destination.name}-{uuid.uuid4().hex}"
        staging.mkdir()
        return str(staging)

    return Path(_main_process_call(accelerator, create))


def _portable_publication_valid(path: Path, *, args: argparse.Namespace, run_id: str, step: int) -> bool:
    if not path.exists():
        return False
    manifest = validate_published_directory(path, expected_kind=PORTABLE_CHECKPOINT_KIND)
    expected = {
        "global_step": step,
        "model_architecture": args.model_architecture,
        "tuning_mode": _tuning_mode(args),
        "run_id": run_id,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise CheckpointValidationError(
                f"existing portable checkpoint {key} mismatch at {path}: {manifest.get(key)!r} != {value!r}"
            )
    return True


def _validate_portable_contents(path: Path, *, args: argparse.Namespace, run_id: str, step: int) -> None:
    required = {"config.json", "model.safetensors"}
    if args.proprio_stats_path:
        required.update({"proprio_stats_sonic.json", "proprio_norm_config.json"})
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise CheckpointValidationError(f"portable checkpoint is missing required files: {missing}")
    config_payload = _read_json_object(path / "config.json", label="portable config")
    model_config = config_payload.get("config")
    checkpoint_metadata = config_payload.get("metadata")
    if not isinstance(model_config, dict) or not isinstance(checkpoint_metadata, dict):
        raise CheckpointValidationError("portable config requires object fields config and metadata")
    _require_equal("portable config architecture", model_config.get("model_architecture"), args.model_architecture)
    _require_equal("portable metadata architecture", checkpoint_metadata.get("model_architecture"), args.model_architecture)
    _require_equal("portable metadata tuning mode", checkpoint_metadata.get("tuning_mode"), _tuning_mode(args))
    _require_equal("portable metadata run_id", checkpoint_metadata.get("run_id"), run_id)
    _require_equal("portable metadata step", checkpoint_metadata.get("global_step"), step)


def _save_portable_checkpoint(
    model: QwenSonicFM,
    destination: Path,
    *,
    args: argparse.Namespace,
    run_id: str,
    progress: TrainingProgress,
    metadata: dict[str, Any],
    reasons: list[str],
    accelerator: Any,
    proprio_stats_path: str | None,
) -> bool:
    already_complete = _main_process_call(
        accelerator,
        lambda: _portable_publication_valid(destination, args=args, run_id=run_id, step=progress.global_step),
    )
    if already_complete:
        return False
    staging = _new_staging_directory(destination, accelerator)
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)

    def publish() -> bool:
        try:
            unwrapped = accelerator.unwrap_model(model)
            checkpoint_metadata = {
                **metadata,
                "global_step": progress.global_step,
                "epoch": progress.epoch,
                "batches_consumed_in_epoch": progress.batches_consumed_in_epoch,
                "save_reasons": sorted(set(reasons)),
            }
            unwrapped.save_checkpoint(staging, metadata=checkpoint_metadata, state_dict=state_dict)
            if proprio_stats_path:
                source = Path(proprio_stats_path)
                if source.is_file():
                    shutil.copy2(source, staging / "proprio_stats_sonic.json")
                    write_json_atomic(
                        staging / "proprio_norm_config.json",
                        {"use_percentiles": True, "clip_outliers": True},
                    )
            _validate_portable_contents(
                staging,
                args=args,
                run_id=run_id,
                step=progress.global_step,
            )
            publish_directory_manifest(
                staging,
                kind=PORTABLE_CHECKPOINT_KIND,
                metadata={
                    "global_step": progress.global_step,
                    "model_architecture": args.model_architecture,
                    "tuning_mode": _tuning_mode(args),
                    "run_id": run_id,
                    "stage": args.stage,
                    "epoch": progress.epoch,
                    "batches_consumed_in_epoch": progress.batches_consumed_in_epoch,
                    "save_reasons": sorted(set(reasons)),
                },
            )
            validate_published_directory(staging, expected_kind=PORTABLE_CHECKPOINT_KIND)
            atomic_publish_directory(staging, destination)
            return True
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    published = _main_process_call(accelerator, publish)
    accelerator.wait_for_everyone()
    return bool(published)


def _resume_publication_valid(
    path: Path,
    *,
    args: argparse.Namespace,
    run_id: str,
    step: int,
    expected_world_size: int,
    expected_trainer_state: dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    trainer_state_path = path / "trainer_state.json"
    if not trainer_state_path.is_file() or trainer_state_path.is_symlink():
        raise CheckpointValidationError(f"existing resume state has no valid trainer state: {path}")
    trainer_state = _read_json_object(trainer_state_path, label="existing trainer state")
    _validate_resume_state_contents(path, trainer_state)
    _validate_resume_progress_and_scheduler(path, trainer_state)
    manifest = validate_published_directory(
        path,
        expected_kind=RESUME_STATE_KIND,
        expected_world_size=expected_world_size,
    )
    expected = {
        "global_step": step,
        "world_size": expected_world_size,
        "model_architecture": args.model_architecture,
        "tuning_mode": _tuning_mode(args),
        "run_id": run_id,
    }
    trainer_values = {
        "global_step": (
            trainer_state.get("progress", {}).get("global_step")
            if isinstance(trainer_state.get("progress"), dict)
            else None
        ),
        "world_size": (
            trainer_state.get("runtime", {}).get("world_size")
            if isinstance(trainer_state.get("runtime"), dict)
            else None
        ),
        "model_architecture": trainer_state.get("model_architecture"),
        "tuning_mode": trainer_state.get("tuning_mode"),
        "run_id": trainer_state.get("run_id"),
    }
    for key, value in expected.items():
        if manifest.get(key) != value or trainer_values[key] != value:
            raise CheckpointValidationError(
                f"existing resume state {key} mismatch at {path}: "
                f"manifest={manifest.get(key)!r} trainer={trainer_values[key]!r} expected={value!r}"
            )
    existing_identity = {
        key: value for key, value in trainer_state.items() if key != "created_at"
    }
    requested_identity = {
        key: value for key, value in expected_trainer_state.items() if key != "created_at"
    }
    if existing_identity != requested_identity:
        raise CheckpointValidationError(
            "existing resume state differs from the requested same-step trainer state"
        )
    return True


def _resume_latest_payload(
    destination: Path,
    *,
    step: int,
    publication_file_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINER_STATE_SCHEMA_VERSION,
        "state_dir": destination.name,
        "global_step": step,
        "manifest_sha256": sha256_file(destination / "manifest.json"),
        "publication_file_count": publication_file_count,
    }


def _commit_resume_latest(root: Path, payload: dict[str, Any]) -> bool:
    latest_path = root / "LATEST.json"
    if os.path.lexists(latest_path):
        current = _read_resume_latest(latest_path)
        current_step = current["global_step"]
        requested_step = payload["global_step"]
        if current_step > requested_step:
            raise CheckpointValidationError(
                "refusing to move resume LATEST backwards: "
                f"current={current_step} requested={requested_step}"
            )
        if current_step == requested_step:
            if current != payload:
                raise CheckpointValidationError(
                    "resume LATEST conflicts with the completed state at the same step"
                )
            return False
    write_json_atomic(latest_path, payload)
    if _read_resume_latest(latest_path) != payload:
        raise CheckpointValidationError("published resume LATEST does not match its requested payload")
    return True


def _validate_prunable_resume_state(path: Path) -> tuple[int, int]:
    if path.is_symlink():
        raise CheckpointValidationError(f"resume state directory may not be a symlink: {path}")
    trainer_state_path = path / "trainer_state.json"
    if not trainer_state_path.is_file() or trainer_state_path.is_symlink():
        raise CheckpointValidationError(f"resume state has no valid trainer state: {path}")
    trainer_state = _read_json_object(trainer_state_path, label="resume trainer state")
    if (
        trainer_state.get("schema_version") != TRAINER_STATE_SCHEMA_VERSION
        or trainer_state.get("kind") != RESUME_STATE_KIND
    ):
        raise CheckpointValidationError(
            f"unsupported resume trainer state schema/kind at {path}"
        )
    progress = trainer_state.get("progress")
    step = progress.get("global_step") if isinstance(progress, dict) else None
    if type(step) is not int or step < 0 or path.name != f"step_{step}":
        raise CheckpointValidationError(
            f"resume state directory/global_step mismatch at {path}: {step!r}"
        )
    runtime = trainer_state.get("runtime")
    world_size = runtime.get("world_size") if isinstance(runtime, dict) else None
    if type(world_size) is not int or world_size <= 0:
        raise CheckpointValidationError(
            f"resume state has invalid world_size at {path}: {world_size!r}"
        )
    _validate_resume_state_contents(path, trainer_state)
    _validate_resume_progress_and_scheduler(path, trainer_state)
    publication = validate_published_directory(
        path,
        expected_kind=RESUME_STATE_KIND,
        expected_world_size=world_size,
    )
    expected = {
        "global_step": step,
        "world_size": world_size,
        "model_architecture": trainer_state.get("model_architecture"),
        "tuning_mode": trainer_state.get("tuning_mode"),
        "run_id": trainer_state.get("run_id"),
    }
    for key, value in expected.items():
        if publication.get(key) != value:
            raise CheckpointValidationError(
                f"resume publication {key} mismatch at {path}: "
                f"manifest={publication.get(key)!r} trainer={value!r}"
            )
    return step, len(publication["files"])


def _prune_resume_states(root: Path, *, keep: int, latest_step: int) -> None:
    if type(keep) is not int or keep <= 0:
        raise CheckpointValidationError(f"invalid resume retention count: {keep!r}")
    latest = _read_resume_latest(root / "LATEST.json")
    if latest["global_step"] != latest_step:
        raise CheckpointValidationError(
            f"resume retention is not anchored to current LATEST: "
            f"{latest['global_step']} != {latest_step}"
        )

    complete: list[tuple[int, Path, int]] = []
    for path in root.glob("step_*"):
        raw_step = path.name.removeprefix("step_")
        if not raw_step.isdigit():
            continue
        candidate_step = int(raw_step)
        if candidate_step > latest_step:
            continue
        if not path.is_dir():
            continue
        if not (path / "_SUCCESS").is_file() or not (path / "manifest.json").is_file():
            continue
        validated_step, file_count = _validate_prunable_resume_state(path)
        if validated_step != candidate_step:
            raise CheckpointValidationError(
                f"resume retention step mismatch at {path}: {validated_step}"
            )
        complete.append((candidate_step, path, file_count))

    complete.sort()
    current = next((item for item in complete if item[0] == latest_step), None)
    if current is None:
        raise CheckpointValidationError(
            f"current resume LATEST state is not a valid retention candidate: step_{latest_step}"
        )
    expected_latest = _resume_latest_payload(
        current[1],
        step=latest_step,
        publication_file_count=current[2],
    )
    if latest != expected_latest:
        raise CheckpointValidationError(
            "resume retention LATEST does not match its validated current state"
        )
    for _, path, _ in complete[:-keep]:
        shutil.rmtree(path)


def _save_resume_state(
    destination: Path,
    *,
    trainer_state: dict[str, Any],
    args: argparse.Namespace,
    run_id: str,
    accelerator: Any,
) -> bool:
    raw_step = trainer_state.get("progress", {}).get("global_step")
    if type(raw_step) is not int or raw_step < 0:
        raise CheckpointValidationError(f"trainer state has invalid global_step: {raw_step!r}")
    step = raw_step
    runtime = trainer_state.get("runtime")
    world_size = runtime.get("world_size") if isinstance(runtime, dict) else None
    if type(world_size) is not int or world_size <= 0:
        raise CheckpointValidationError(f"trainer state has invalid world_size: {world_size!r}")
    already_complete = _main_process_call(
        accelerator,
        lambda: _resume_publication_valid(
            destination,
            args=args,
            run_id=run_id,
            step=step,
            expected_world_size=world_size,
            expected_trainer_state=trainer_state,
        ),
    )
    if already_complete:
        def commit_existing() -> bool:
            publication = validate_published_directory(
                destination,
                expected_kind=RESUME_STATE_KIND,
                expected_world_size=world_size,
            )
            latest_payload = _resume_latest_payload(
                destination,
                step=step,
                publication_file_count=len(publication["files"]),
            )
            _commit_resume_latest(destination.parent, latest_payload)
            _prune_resume_states(
                destination.parent,
                keep=args.resume_state_keep,
                latest_step=step,
            )
            return False

        committed = _main_process_call(accelerator, commit_existing)
        accelerator.wait_for_everyone()
        return bool(committed)
    staging = _new_staging_directory(destination, accelerator)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(staging))
    accelerator.wait_for_everyone()

    def publish() -> bool:
        try:
            write_json_atomic(staging / "trainer_state.json", trainer_state)
            publish_directory_manifest(
                staging,
                kind=RESUME_STATE_KIND,
                metadata={
                    "global_step": step,
                    "world_size": world_size,
                    "model_architecture": args.model_architecture,
                    "tuning_mode": _tuning_mode(args),
                    "run_id": run_id,
                    "stage": args.stage,
                },
            )
            _validate_resume_state_contents(staging, trainer_state)
            _validate_resume_progress_and_scheduler(staging, trainer_state)
            publication = validate_published_directory(
                staging,
                expected_kind=RESUME_STATE_KIND,
                expected_world_size=world_size,
            )
            atomic_publish_directory(staging, destination)
            latest_payload = _resume_latest_payload(
                destination,
                step=step,
                publication_file_count=len(publication["files"]),
            )
            _commit_resume_latest(destination.parent, latest_payload)
            _prune_resume_states(
                destination.parent,
                keep=args.resume_state_keep,
                latest_step=step,
            )
            return True
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    published = _main_process_call(accelerator, publish)
    accelerator.wait_for_everyone()
    return bool(published)


class SaveCoordinator:
    def __init__(
        self,
        *,
        model: QwenSonicFM,
        args: argparse.Namespace,
        run_dir: Path,
        run_id: str,
        accelerator: Any,
        scheduler: Any,
        scheduler_total_steps: int,
        warmup_steps: int,
        optimizer_inventory: list[OptimizerGroupInfo],
        checkpoint_metadata: dict[str, Any],
        source_contract: dict[str, Any],
        data_contract: dict[str, Any],
        run_file_hashes: dict[str, str],
        train_loader: Any,
        train_sampler: Sampler[int] | None,
        proprio_stats_path: str | None,
    ) -> None:
        self.model = model
        self.args = args
        self.run_dir = run_dir
        self.run_id = run_id
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.scheduler_total_steps = scheduler_total_steps
        self.warmup_steps = warmup_steps
        self.optimizer_inventory = optimizer_inventory
        self.checkpoint_metadata = checkpoint_metadata
        self.source_contract = source_contract
        self.data_contract = data_contract
        self.run_file_hashes = run_file_hashes
        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.proprio_stats_path = proprio_stats_path
        self.portable_steps: set[int] = set()
        self.resume_steps: set[int] = set()

    def _trainer_state(self, progress: TrainingProgress, reasons: list[str]) -> dict[str, Any]:
        world_size = int(self.accelerator.num_processes)
        sampler_state = (
            self.train_sampler.state_dict()
            if isinstance(self.train_sampler, BlockShuffleSampler)
            else None
        )
        return {
            "schema_version": TRAINER_STATE_SCHEMA_VERSION,
            "kind": RESUME_STATE_KIND,
            "stage": self.args.stage,
            "task": self.args.humanoid_arena_task,
            "model_architecture": self.args.model_architecture,
            "tuning_mode": _tuning_mode(self.args),
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "resolved_config": _resolved_training_contract(self.args),
            "progress": {
                **asdict(progress),
                "successful_updates": progress.global_step,
            },
            "runtime": {
                "world_size": world_size,
                "batch_size": self.args.batch_size,
                "global_batch": self.args.batch_size * world_size * self.args.gradient_accumulation_steps,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "mixed_precision": self.args.mixed_precision,
                "deepspeed_config": self.args.deepspeed_config,
                "deepspeed_config_sha256": (
                    self.source_contract.get("deepspeed_config") or {}
                ).get("sha256"),
            },
            "optimizer_groups": [item.to_dict() for item in self.optimizer_inventory],
            "adamw": {
                "betas": list(ADAMW_BETAS),
                "eps": ADAMW_EPS,
                "weight_decay": self.args.weight_decay,
            },
            "scheduler": {
                "type": self.args.lr_scheduler_type,
                "total_steps": self.scheduler_total_steps,
                "absolute_max_steps": self.args.max_steps,
                "warmup_steps": self.warmup_steps,
                "min_lr_rate": self.args.min_lr_rate,
                "current_lrs": list(self.scheduler.get_last_lr()),
            },
            "data": {
                "seed": self.args.seed,
                "loader_batches": len(self.train_loader),
                "dataset_summary": self.data_contract.get("dataset_contract"),
                "sample_stride": self.args.sample_stride,
                "block_shuffle_size": self.args.block_shuffle_size,
                "drop_last": bool(self.args.drop_last),
                "deterministic_worker_epoch_seed": True,
                "persistent_workers": False,
                "sampler": sampler_state,
            },
            "data_fingerprint": self.data_contract["fingerprint"],
            "source_fingerprint": self.source_contract["fingerprint"],
            "run_files": self.run_file_hashes,
            "save_reasons": sorted(set(reasons)),
            "resume_state_steps": self.args.resume_state_steps,
            "resume_state_keep": self.args.resume_state_keep,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def request(
        self,
        progress: TrainingProgress,
        *,
        reasons: list[str],
        portable: bool,
        resume: bool,
    ) -> None:
        step = progress.global_step
        if portable and step not in self.portable_steps:
            destination = self.run_dir / f"step_{step}"
            _log_main(self.accelerator, f"[{self.args.stage}] saving portable checkpoint: {destination}")
            _save_portable_checkpoint(
                self.model,
                destination,
                args=self.args,
                run_id=self.run_id,
                progress=progress,
                metadata=self.checkpoint_metadata,
                reasons=reasons,
                accelerator=self.accelerator,
                proprio_stats_path=self.proprio_stats_path,
            )
            self.portable_steps.add(step)
        if resume and step not in self.resume_steps:
            destination = self.run_dir / "_resume_states" / f"step_{step}"
            _log_main(self.accelerator, f"[{self.args.stage}] saving full resume state: {destination}")
            _save_resume_state(
                destination,
                trainer_state=self._trainer_state(progress, reasons),
                args=self.args,
                run_id=self.run_id,
                accelerator=self.accelerator,
            )
            self.resume_steps.add(step)


def _train_stage_impl(
    args: argparse.Namespace,
    writer_lock_holder: list[RunWriterLock],
) -> Path:
    accelerator = None
    if not args.use_accelerate:
        raise ValueError("training must run with accelerate; use scripts/train/train_huge.sh")
    if args.use_accelerate:
        from accelerate import Accelerator, DeepSpeedPlugin

        accelerator_kwargs: dict[str, Any] = {
            "mixed_precision": None if args.mixed_precision == "no" else args.mixed_precision,
            "step_scheduler_with_optimizer": False,
        }
        if args.use_deepspeed:
            if args.deepspeed_config:
                accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config)
            else:
                accelerator_kwargs["deepspeed_plugin"] = DeepSpeedPlugin(
                    zero_stage=args.zero_stage,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    gradient_clipping=args.max_grad_norm or 0.0,
                    zero3_init_flag=(args.zero_stage == 3),
                    zero3_save_16bit_model=(args.zero_stage == 3),
                )
        accelerator = Accelerator(**accelerator_kwargs)
        if accelerator.num_processes != args.expected_num_processes:
            raise ValueError(
                f"training must run on {args.expected_num_processes} processes; "
                f"got {accelerator.num_processes}. Use scripts/train/train_huge.sh."
            )
        device = accelerator.device
    else:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    source_contract = _main_process_call(accelerator, lambda: _source_contract(args))
    resume_info: dict[str, Any] | None = None
    stats_contract = _cumulative_stats_contract(args)
    requested_stats_path = args.proprio_stats_path
    if args.resume_from_checkpoint is not None:
        resume_info = _main_process_call(
            accelerator,
            lambda: inspect_resume_state(args.resume_from_checkpoint),
        )
        if stats_contract is not None:
            resume_run_dir = Path(str(resume_info["trainer_state"]["run_dir"])).resolve()
            snapshot = resume_run_dir / "proprio_stats_sonic.json"
            if not snapshot.is_file():
                raise FileNotFoundError(f"resume run is missing stats snapshot: {snapshot}")
            if requested_stats_path is not None and sha256_file(requested_stats_path) != sha256_file(snapshot):
                raise ValueError("explicit proprio stats do not match the resume run-local snapshot")
            args.proprio_stats_path = str(snapshot)
        out_dir = validate_resume_static(
            resume_info["trainer_state"],
            args,
            source_contract=source_contract,
        )
    else:
        out_dir = Path(args.output_dir).expanduser().resolve() / args.stage / _run_dir_name(args, accelerator)
    writer_lock = RunWriterLock(out_dir, enabled=_is_main_process(accelerator))
    writer_lock_holder.append(writer_lock)

    _log_main(
        accelerator,
        (
            f"[{args.stage}] start: device={device} processes={getattr(accelerator, 'num_processes', 1)} "
            f"mixed_precision={args.mixed_precision} deepspeed={args.use_deepspeed} "
            f"deepspeed_config={args.deepspeed_config or 'plugin-default'}"
        ),
    )
    stats = args.proprio_stats_path
    external_stats_path = requested_stats_path
    validation_manifest: dict[str, Any] | None = None
    validation_manifest_path: str | None = None
    validated_inventories: dict[str, dict[str, Any]] | None = None
    if stats_contract is not None:
        stage_scope, expected_sources, task_scope = stats_contract
        if stats is None:
            stats = _default_cumulative_stats_path(args, stage_scope)
        validate_proprio_stats_contract(
            stats,
            stage_scope=stage_scope,
            sources=expected_sources,
            task_scope=task_scope,
        )
        if resume_info is None:
            external_stats_path = stats
    elif stats is None and args.stage in {"stage3", "finetune_prop"}:
        stats = "resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json"
        external_stats_path = stats

    if stats_contract is not None:
        validation_sources = _validation_sources(args, expected_sources)
        validation_result = _main_process_call(
            accelerator,
            lambda: ensure_validation_manifest(
                Path(args.dataset_root) if args.dataset_root else Path("resource/dataset"),
                sources=validation_sources,
                stage_scope=stage_scope,
                stats_path=str(stats),
                force=args.force_data_audit,
            ),
        )
        if accelerator is not None:
            accelerator.wait_for_everyone()
        validation_manifest = load_validation_manifest(validation_result["publication"])
        validation_manifest_path = str(Path(validation_result["publication"]) / "manifest.json")
        validated_inventories = validation_manifest["sources"]
        _log_main(
            accelerator,
            f"[{args.stage}] action/proprio validation: manifest={validation_manifest_path} "
            f"audited={validation_result['audited_tags'] or 'none'} "
            f"cached={validation_result['cached_tags'] or 'none'}",
        )
    _main_process_call(accelerator, lambda: writer_lock.acquire() or True)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if stats_contract is not None and resume_info is None:
        stats = _main_process_call(accelerator, lambda: _snapshot_proprio_stats(str(stats), out_dir))
        args.proprio_stats_path = stats
    elif stats_contract is not None:
        stats = args.proprio_stats_path
    _log_main(
        accelerator,
        (
            f"[{args.stage}] building dataset: sample_stride={args.sample_stride} "
            f"max_episodes={args.max_episodes or 'all'} stats={stats or 'none'}"
        ),
    )
    dataset = build_stage_dataset(
        args.stage,
        dataset_root=args.dataset_root,
        sample_stride=args.sample_stride,
        max_episodes=args.max_episodes,
        proprio_stats_path=stats,
        egodex_root=args.egodex_root,
        stage1_sources=args.stage1_sources,
        stage2_sources=args.stage2_sources,
        episode_fraction=args.episode_fraction,
        episode_selection_seed=args.episode_selection_seed,
        prop_roots=args.prop_roots,
        prop_ego_roots=args.prop_ego_roots,
        humanoid_arena_roots=args.humanoid_arena_roots,
        humanoid_arena_task=args.humanoid_arena_task,
        validated_inventories=validated_inventories,
    )
    if accelerator is not None:
        accelerator.wait_for_everyone()
    if validation_manifest is not None:
        proprio_inventory = _validation_summaries(validation_manifest, _current_validation_tags(args))
    else:
        proprio_inventory = _main_process_call(accelerator, lambda: _validate_proprio_inventory(dataset))
    if proprio_inventory:
        for summary in proprio_inventory:
            _log_main(
                accelerator,
                f"[{args.stage}] proprio inventory: source={summary['dataset_id']} "
                f"files={summary['n_files']} frames={summary['n_frames']} mismatches=0",
            )
    train_ds = limit_dataset(dataset, max_samples=args.max_train_samples)
    data_contract = _data_contract(
        args,
        dataset=dataset,
        train_length=len(train_ds),
        stats_path=stats,
        proprio_inventory=proprio_inventory,
    )
    if resume_info is not None:
        _require_equal(
            "data fingerprint",
            resume_info["trainer_state"].get("data_fingerprint"),
            data_contract["fingerprint"],
        )
    for summary_line in _data_summary_lines(data_contract):
        _log_main(accelerator, f"[{args.stage}] data source: {summary_line}")
    _log_main(
        accelerator,
        (
            f"[{args.stage}] dataset ready: train_samples={len(train_ds)} "
            f"validation=disabled data_fingerprint={data_contract['fingerprint']}"
        ),
    )
    collator = HugeCollator(
        use_qwen_processor=args.use_qwen,
        qwen_processor_path=args.base_vlm_path,
        image_transform_mode="train",
        max_text_tokens=args.max_text_tokens,
        model_max_length=args.model_max_length,
        char_max_length=args.char_max_length,
        model_architecture=args.model_architecture,
        action_horizon=args.action_horizon,
        oft_action_token=args.oft_action_token,
    )
    set_frame_cache_size(args.frame_cache_size)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_sampler: Sampler[int] | None = None
    spans_fn = getattr(train_ds, "episode_spans", None)
    if args.block_shuffle_size and spans_fn is not None:
        spans = spans_fn()
        if len(spans):
            train_sampler = BlockShuffleSampler(
                spans,
                block_size=args.block_shuffle_size,
                base_seed=args.seed,
            )
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
        # Restart workers from a seed derived from seed+epoch so replaying an
        # epoch advances Python-random augmentations identically.
        "persistent_workers": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=loader_generator,
        drop_last=args.drop_last,
        **loader_kwargs,
    )
    _log_main(
        accelerator,
        (
            f"[{args.stage}] train loader ready: batches={len(train_loader)} batch_size={args.batch_size} "
            f"num_workers={args.num_workers} drop_last={args.drop_last}"
        ),
    )
    if len(train_loader) == 0:
        raise ValueError(
            f"[{args.stage}] train loader has 0 batches. "
            "Reduce --batch-size, increase --max-episodes/--max-train-samples, or keep --no-drop-last."
        )
    if resume_info is not None:
        saved_data = resume_info["trainer_state"].get("data", {})
        _require_equal("block shuffle size", saved_data.get("block_shuffle_size"), args.block_shuffle_size)
        _require_equal("drop_last", saved_data.get("drop_last"), bool(args.drop_last))
        _require_equal("sample_stride", saved_data.get("sample_stride"), args.sample_stride)
        saved_sampler = saved_data.get("sampler")
        current_sampler = train_sampler.state_dict() if isinstance(train_sampler, BlockShuffleSampler) else None
        if isinstance(saved_sampler, dict) and isinstance(current_sampler, dict):
            saved_sampler = {**saved_sampler, "epoch": 0}
            current_sampler = {**current_sampler, "epoch": 0}
        _require_equal("sampler contract", saved_sampler, current_sampler)
    state_dropout_prob = 0.2 if args.state_dropout_prob is None else args.state_dropout_prob
    _log_main(
        accelerator,
        (
            f"[{args.stage}] building model: base_vlm={args.base_vlm_path} use_qwen={args.use_qwen} "
            f"architecture={args.model_architecture} "
            f"freeze_qwen={args.freeze_qwen} qwen_lora={args.qwen_lora} "
            f"train_qwen_lora={args.train_qwen_lora} action_hidden_size={args.action_hidden_size} "
            f"num_action_layers={args.num_action_layers} state_dropout_prob={state_dropout_prob}"
        ),
    )
    config = QwenSonicFMConfig(
        model_architecture=args.model_architecture,
        action_hidden_size=args.action_hidden_size,
        num_action_layers=args.num_action_layers,
        num_attention_heads=args.num_attention_heads,
        action_ffn_multiplier=args.action_ffn_multiplier,
        action_dropout=args.action_dropout,
        num_future_tokens=args.num_future_tokens,
        state_dropout_prob=state_dropout_prob,
        flow_noise_beta_alpha=args.flow_noise_beta_alpha,
        flow_noise_beta_beta=args.flow_noise_beta_beta,
        flow_noise_s=args.flow_noise_s,
        action_horizon=args.action_horizon,
        base_vlm_path=args.base_vlm_path,
        use_qwen=args.use_qwen,
        freeze_qwen=args.freeze_qwen,
        qwen_lora=args.qwen_lora,
        train_qwen_lora=args.train_qwen_lora,
        qwen_lora_r=args.qwen_lora_r,
        qwen_lora_alpha=args.qwen_lora_alpha,
        qwen_lora_dropout=args.qwen_lora_dropout,
        num_inference_steps=args.num_inference_steps,
        oft_action_token=args.oft_action_token,
        max_text_tokens=args.max_text_tokens,
        model_max_length=args.model_max_length,
        char_max_length=args.char_max_length,
    )
    model = QwenSonicFM(config).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    _log_main(
        accelerator,
        (
            f"[{args.stage}] trainable parameters: "
            f"{_format_param_count(trainable_params)} / {_format_param_count(total_params)}"
        ),
    )
    if args.init_ckpt is not None:
        _log_main(accelerator, f"[{args.stage}] loading init checkpoint: {args.init_ckpt}")
        _load_checkpoint_state(model, args.init_ckpt, accelerator)
    optimizer_param_groups, optimizer_inventory = build_optimizer_param_groups(model, args)
    if resume_info is not None:
        _require_equal(
            "optimizer inventory",
            resume_info["trainer_state"].get("optimizer_groups"),
            [item.to_dict() for item in optimizer_inventory],
        )
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS,
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=accelerator is None and args.mixed_precision == "fp16" and device.type == "cuda",
    )
    if accelerator is not None:
        _log_main(accelerator, f"[{args.stage}] accelerator.prepare starting")
        prepare_debug = _env_flag("HUGE_DEBUG_PREPARE")
        if prepare_debug:
            timeout = int(os.environ.get("HUGE_PREPARE_TRACEBACK_SECONDS", "300"))
            _log_rank(accelerator, f"[{args.stage}] entering accelerator.prepare device={device}")
            faulthandler.dump_traceback_later(timeout, repeat=True)
        try:
            model, optimizer, train_loader = accelerator.prepare(
                model,
                optimizer,
                train_loader,
            )
        finally:
            if prepare_debug:
                faulthandler.cancel_dump_traceback_later()
        if prepare_debug:
            _log_rank(accelerator, f"[{args.stage}] exited accelerator.prepare device={device}")
        _log_main(accelerator, f"[{args.stage}] accelerator.prepare done")
    prepared_loader_batches = len(train_loader)
    scheduler_total_steps, warmup_steps = resolved_scheduler_steps(
        args,
        prepared_loader_batches=prepared_loader_batches,
    )
    lr_scheduler = build_logical_scheduler(
        optimizer,
        scheduler_type=args.lr_scheduler_type,
        total_steps=scheduler_total_steps,
        warmup_steps=warmup_steps,
        min_lr_rate=float(args.min_lr_rate),
    )
    if accelerator is not None:
        # Scheduler stepping stays explicit in the successful-update path. The
        # Accelerator uses step_scheduler_with_optimizer=False; this only
        # registers scheduler.bin for exact save/resume.
        lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)
    if isinstance(train_sampler, BlockShuffleSampler) and accelerator is not None:
        accelerator.register_for_checkpointing(train_sampler)
    if resume_info is not None:
        saved_data = resume_info["trainer_state"].get("data", {})
        _require_equal(
            "prepared loader batches",
            saved_data.get("loader_batches"),
            prepared_loader_batches,
        )
        saved_scheduler = resume_info["trainer_state"].get("scheduler", {})
        _require_equal("scheduler total", saved_scheduler.get("total_steps"), scheduler_total_steps)
        _require_equal("scheduler warmup", saved_scheduler.get("warmup_steps"), warmup_steps)
        _require_equal("scheduler type", saved_scheduler.get("type"), args.lr_scheduler_type)
        _require_equal("scheduler floor", saved_scheduler.get("min_lr_rate"), args.min_lr_rate)
        saved_progress = resume_info["trainer_state"]["progress"]
        saved_epoch = saved_progress["epoch"]
        saved_batch_offset = saved_progress["batches_consumed_in_epoch"]
        if not 0 <= saved_batch_offset < prepared_loader_batches:
            raise ValueError(
                "resume batches_consumed_in_epoch is outside the prepared loader: "
                f"offset={saved_batch_offset} loader_batches={prepared_loader_batches}"
            )
        _require_equal(
            "attempted batch data position",
            saved_progress["attempted_batches"],
            saved_epoch * prepared_loader_batches + saved_batch_offset,
        )

    run_id = (
        str(resume_info["trainer_state"]["run_id"])
        if resume_info is not None
        else out_dir.name
    )
    if resume_info is None:
        data_manifest_payload = {
            **data_contract,
            "stage": args.stage,
            "stats": stats,
            "external_stats": _file_contract(external_stats_path, required=True) if external_stats_path else None,
            "validation": "disabled",
        }
        training_config_payload = {
            **vars(args),
            "run_dir": str(out_dir),
            "resolved_training_contract": _resolved_training_contract(args),
            "run_id": run_id,
            "optimizer_groups": [item.to_dict() for item in optimizer_inventory],
            "scheduler_total_steps": scheduler_total_steps,
            "scheduler_warmup_steps": warmup_steps,
            "source_fingerprint": source_contract["fingerprint"],
        }
        run_file_hashes = _main_process_call(
            accelerator,
            lambda: {
                "data_manifest.json": write_json_atomic(out_dir / "data_manifest.json", data_manifest_payload),
                "training_config.json": write_json_atomic(out_dir / "training_config.json", training_config_payload),
                **({"proprio_stats_sonic.json": sha256_file(out_dir / "proprio_stats_sonic.json")} if stats_contract else {}),
            },
        )
    else:
        run_file_hashes = dict(resume_info["trainer_state"]["run_files"])

    with RunLogTee(out_dir, enabled=_is_main_process(accelerator)):
        _log_main(accelerator, f"[{args.stage}] logging to {out_dir / 'train.log'}")
        metadata = build_checkpoint_metadata(
            model_name_or_path=args.base_vlm_path,
            qwen_processor_path=args.base_vlm_path,
            proprio_stats_path=stats,
            action_schema=ActionSchema().to_dict(),
            action_hidden_size=config.action_hidden_size,
            num_action_layers=config.num_action_layers,
            num_attention_heads=config.num_attention_heads,
            model_architecture=config.model_architecture,
            tuning_mode=_tuning_mode(args),
            run_id=run_id,
            state_dropout_prob=state_dropout_prob,
            image_transform="gr00t_explicit_train_random_eval_center_256",
            freeze_qwen=args.freeze_qwen,
            qwen_lora=args.qwen_lora,
            train_qwen_lora=args.train_qwen_lora,
            qwen_lora_r=args.qwen_lora_r,
            qwen_lora_alpha=args.qwen_lora_alpha,
            qwen_lora_dropout=args.qwen_lora_dropout,
            optimizer_groups=[item.to_dict() for item in optimizer_inventory],
            adamw={"betas": list(ADAMW_BETAS), "eps": ADAMW_EPS, "weight_decay": args.weight_decay},
            scheduler={
                "type": args.lr_scheduler_type,
                "total_steps": scheduler_total_steps,
                "warmup_steps": warmup_steps,
                "min_lr_rate": args.min_lr_rate,
            },
            source_git_commit=source_contract.get("source_git_commit"),
            source_fingerprint=source_contract["fingerprint"],
            data_fingerprint=data_contract["fingerprint"],
        )
        for item in optimizer_inventory:
            _log_main(
                accelerator,
                (
                    f"[{args.stage}] optimizer group={item.name} tensors={item.tensor_count} "
                    f"numel={item.numel} peak_lr={item.peak_lr:.2e} "
                    f"floor_lr={item.floor_lr:.2e} weight_decay={args.weight_decay}"
                ),
            )
        if resume_info is not None:
            accelerator.load_state(resume_info["state_dir"])
            saved_progress = resume_info["trainer_state"]["progress"]
            progress = TrainingProgress(
                global_step=int(saved_progress["global_step"]),
                epoch=int(saved_progress["epoch"]),
                batches_consumed_in_epoch=int(saved_progress["batches_consumed_in_epoch"]),
                attempted_batches=int(saved_progress["attempted_batches"]),
                skipped_updates=int(saved_progress["skipped_updates"]),
            )
            _require_equal("successful updates", saved_progress.get("successful_updates"), progress.global_step)
            _require_equal(
                "scheduler current LRs",
                list(resume_info["trainer_state"]["scheduler"].get("current_lrs", [])),
                list(lr_scheduler.get_last_lr()),
            )
            _log_main(
                accelerator,
                (
                    f"[{args.stage}] resumed state={resume_info['state_dir']} "
                    f"step={progress.global_step} epoch={progress.epoch} "
                    f"batch_offset={progress.batches_consumed_in_epoch}"
                ),
            )
        else:
            progress = TrainingProgress()

        total_train_steps = scheduler_total_steps
        _log_main(
            accelerator,
            (
                f"[{args.stage}] scheduler ready: epochs={args.epochs} total_steps={total_train_steps} "
                f"scheduler_steps={scheduler_total_steps} warmup_steps={warmup_steps} "
                f"checkpointing_steps={args.checkpointing_steps} "
                f"resume_state_steps={args.resume_state_steps}"
            ),
        )

        save_coordinator = SaveCoordinator(
            model=model,
            args=args,
            run_dir=out_dir,
            run_id=run_id,
            accelerator=accelerator,
            scheduler=lr_scheduler,
            scheduler_total_steps=scheduler_total_steps,
            warmup_steps=warmup_steps,
            optimizer_inventory=optimizer_inventory,
            checkpoint_metadata=metadata,
            source_contract=source_contract,
            data_contract=data_contract,
            run_file_hashes=run_file_hashes,
            train_loader=train_loader,
            train_sampler=train_sampler,
            proprio_stats_path=stats,
        )
        train_start = time.time()
        progress_total = total_train_steps
        trace_batches = _env_flag("HUGE_LOG_BATCH_DIGESTS")
        hard_cap_reached = False
        with tqdm(
            total=progress_total,
            initial=progress.global_step,
            desc=f"{args.stage} steps",
            unit="step",
            disable=not _is_main_process(accelerator),
        ) as pbar:
            while True:
                if args.max_steps is not None and progress.global_step >= args.max_steps:
                    hard_cap_reached = True
                    break
                if args.max_steps is None and progress.epoch >= args.epochs:
                    break
                epoch = progress.epoch
                resume_offset = progress.batches_consumed_in_epoch
                _prepare_loader_epoch(
                    train_loader,
                    train_sampler,
                    loader_generator,
                    seed=args.seed,
                    epoch=epoch,
                )
                model.train()
                for batch_index, batch in enumerate(train_loader):
                    if batch_index < resume_offset:
                        if trace_batches:
                            sources, identity_sha256, batch_sha256 = _batch_trace(batch)
                            _log_rank(
                                accelerator,
                                (
                                    f"[{args.stage}] replay_discard epoch={epoch} batch={batch_index} "
                                    f"sources={sources} identity_sha256={identity_sha256} "
                                    f"batch_sha256={batch_sha256}"
                                ),
                            )
                        continue
                    if args.max_steps is not None and progress.global_step >= args.max_steps:
                        hard_cap_reached = True
                        break

                    progress.attempted_batches += 1
                    progress.batches_consumed_in_epoch = batch_index + 1
                    epoch_completed = progress.batches_consumed_in_epoch >= len(train_loader)
                    if trace_batches:
                        sources, identity_sha256, batch_sha256 = _batch_trace(batch)
                        _log_rank(
                            accelerator,
                            (
                                f"[{args.stage}] batch epoch={epoch} batch={batch_index} "
                                f"sources={sources} identity_sha256={identity_sha256} "
                                f"batch_sha256={batch_sha256}"
                            ),
                        )
                    batch = move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    autocast_enabled = (
                        args.mixed_precision == "bf16"
                        and device.type == "cuda"
                        and not args.use_deepspeed
                    )
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                        pred = model(
                            normalized_actions=batch["normalized_actions"],
                            input_ids=batch["input_ids"],
                            images=batch["images"],
                            state=batch["state"],
                            state_mask=batch["state_mask"],
                            qwen_inputs=batch.get("qwen_inputs"),
                            action_token_mask=batch.get("action_token_mask"),
                        )
                        losses = qwen_sonicfm_loss(
                            pred,
                            batch["normalized_actions"],
                            sonic_mask=batch["sonic_mask"],
                        )
                    loss_is_nonfinite = (~torch.isfinite(losses["loss"].detach())).to(torch.int32)
                    loss_is_nonfinite = accelerator.reduce(loss_is_nonfinite, reduction="sum")
                    if bool(loss_is_nonfinite.item()):
                        sources, identity_sha256, batch_sha256 = _batch_trace(batch)
                        _log_rank(
                            accelerator,
                            (
                                f"[{args.stage}] non-finite loss: epoch={epoch} batch={batch_index} "
                                f"sources={sources} identity_sha256={identity_sha256} "
                                f"batch_sha256={batch_sha256}"
                            ),
                        )
                        raise FloatingPointError("non-finite training loss")
                    if scaler.is_enabled():
                        scaler.scale(losses["loss"]).backward()
                        scaler.unscale_(optimizer)
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        accelerator.backward(losses["loss"])
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()

                    update_skipped = bool(accelerator.optimizer_step_was_skipped)
                    if update_skipped:
                        progress.skipped_updates += 1
                        sources, identity_sha256, batch_sha256 = _batch_trace(batch)
                        _log_rank(
                            accelerator,
                            (
                                f"[{args.stage}] non-finite gradients: epoch={epoch} batch={batch_index} "
                                f"sources={sources} identity_sha256={identity_sha256} "
                                f"batch_sha256={batch_sha256}"
                            ),
                        )
                        raise FloatingPointError("optimizer skipped a non-finite gradient update")

                    lr_scheduler.step()
                    progress.global_step += 1
                    if epoch_completed:
                        progress.epoch = epoch + 1
                        progress.batches_consumed_in_epoch = 0
                    _update_training_progress(
                        pbar=pbar,
                        accelerator=accelerator,
                        global_step=progress.global_step,
                        attempted_batches=progress.attempted_batches,
                        skipped_updates=progress.skipped_updates,
                        loss=losses["loss"],
                        lr_scheduler=lr_scheduler,
                    )

                    reasons: list[str] = []
                    portable_due = bool(
                        args.checkpointing_steps
                        and progress.global_step % args.checkpointing_steps == 0
                    )
                    resume_due = bool(
                        args.resume_state_steps
                        and progress.global_step % args.resume_state_steps == 0
                    )
                    if portable_due:
                        reasons.append("interval")
                    if resume_due:
                        reasons.append("resume_interval")
                    if epoch_completed and args.save_epoch_checkpoints:
                        reasons.append("epoch")
                        portable_due = True
                        resume_due = True
                    hard_cap_reached = bool(
                        args.max_steps is not None
                        and progress.global_step >= args.max_steps
                    )
                    final_epoch = bool(
                        args.max_steps is None
                        and epoch_completed
                        and progress.epoch >= args.epochs
                    )
                    if hard_cap_reached or final_epoch:
                        reasons.append("final")
                        portable_due = True
                        resume_due = True
                    if portable_due or resume_due:
                        save_coordinator.request(
                            progress,
                            reasons=reasons,
                            portable=portable_due,
                            resume=resume_due,
                        )
                    if hard_cap_reached:
                        break

                if hard_cap_reached:
                    break
                if progress.epoch == epoch and progress.batches_consumed_in_epoch >= len(train_loader):
                    progress.epoch = epoch + 1
                    progress.batches_consumed_in_epoch = 0

        save_coordinator.request(
            progress,
            reasons=["final"],
            portable=True,
            resume=True,
        )
        _log_main(
            accelerator,
            (
                f"[{args.stage}] done: output_dir={out_dir} successful={progress.global_step} "
                f"attempted={progress.attempted_batches} skipped={progress.skipped_updates} "
                f"total_time={_format_seconds(time.time() - train_start)}"
            ),
        )
    _main_process_call(accelerator, lambda: writer_lock.release() or True)
    accelerator.wait_for_everyone()
    return out_dir


def train_stage(args: argparse.Namespace) -> Path:
    writer_locks: list[RunWriterLock] = []
    try:
        return _train_stage_impl(args, writer_locks)
    finally:
        for writer_lock in reversed(writer_locks):
            writer_lock.release()


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _load_checkpoint_state(model: QwenSonicFM, checkpoint_path: str | Path, accelerator: Any | None = None) -> None:
    from safetensors.torch import load_file

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "model.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    state_dict = load_file(str(ckpt_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        _log_main(accelerator, f"[checkpoint] loaded {ckpt_path} with missing={sorted(missing)} unexpected={sorted(unexpected)}")


def build_argparser(default_stage: str) -> argparse.ArgumentParser:
    p = StageArgumentParser()
    p.add_argument("--stage", default=canonical_stage(default_stage))
    p.add_argument("--output-dir", default=".runs/train")
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--use-accelerate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--expected-num-processes", type=int, default=8)
    p.add_argument("--base-vlm-path", default="resource/model/Qwen3-VL-2B-Instruct")
    p.add_argument("--model-architecture", choices=["qwenpi_v3", "qwengr00t", "qwenoft"], default="qwenpi_v3")
    p.add_argument("--use-qwen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-qwen", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--qwen-lora", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--train-qwen-lora", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--qwen-lora-r", type=int, default=None)
    p.add_argument("--qwen-lora-alpha", type=int, default=None)
    p.add_argument("--qwen-lora-dropout", type=float, default=None)
    p.add_argument("--hidden-size", type=int, default=512)
    p.add_argument("--vlm-hidden-size", type=int, default=512)
    p.add_argument("--action-hidden-size", type=int, default=None)
    p.add_argument("--num-action-layers", type=int, default=None)
    p.add_argument("--num-attention-heads", type=int, default=None)
    p.add_argument("--action-ffn-multiplier", type=int, default=4)
    p.add_argument("--action-dropout", type=float, default=0.1)
    p.add_argument("--num-future-tokens", type=int, default=32)
    p.add_argument("--state-dropout-prob", type=float, default=None)
    p.add_argument("--flow-noise-beta-alpha", type=float, default=1.5)
    p.add_argument("--flow-noise-beta-beta", type=float, default=1.0)
    p.add_argument("--flow-noise-s", type=float, default=0.999)
    p.add_argument("--action-horizon", type=int, default=50)
    p.add_argument("--num-inference-steps", type=int, default=4)
    p.add_argument("--oft-action-token", default="\U0001f50d")
    p.add_argument("--max-text-tokens", type=int, default=None)
    p.add_argument("--model-max-length", type=int, default=None)
    p.add_argument("--char-max-length", type=int, default=None)
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--resume-from-checkpoint", default=None)
    p.add_argument("--proprio-stats-path", default=None)
    p.add_argument("--force-data-audit", action="store_true")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--egodex-root", default=None)
    p.add_argument("--stage1-source", dest="stage1_sources", action="append", nargs=2, metavar=("TYPE", "ROOT"))
    p.add_argument("--stage2-source", dest="stage2_sources", action="append", nargs=2, metavar=("TYPE", "ROOT"))
    p.add_argument("--sample-stride", type=int, default=None)
    p.add_argument("--episode-fraction", type=float, default=None)
    p.add_argument("--episode-selection-seed", type=int, default=None)
    p.add_argument("--prop-roots", nargs="*", default=None)
    p.add_argument("--prop-ego-roots", nargs="*", default=None)
    p.add_argument("--humanoid-arena-roots", nargs="*", default=None)
    p.add_argument("--humanoid-arena-task", default="all")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--vlm-learning-rate", type=float, default=None)
    p.add_argument("--projector-learning-rate", type=float, default=None)
    p.add_argument("--action-learning-rate", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--min-lr-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mixed-precision", choices=["no", "bf16", "fp16"], default="bf16")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=None)
    p.add_argument("--frame-cache-size", type=int, default=None)
    p.add_argument("--block-shuffle-size", type=int, default=None)
    p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--checkpointing-steps", type=int, default=None)
    p.add_argument("--resume-state-steps", type=int, default=None)
    p.add_argument("--resume-state-keep", type=int, default=2)
    p.add_argument("--save-epoch-checkpoints", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--lr-scheduler-type", default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--warmup-ratio", type=float, default=None)
    p.add_argument("--use-deepspeed", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--deepspeed-config", default=DEFAULT_DEEPSPEED_CONFIG)
    p.add_argument("--zero-stage", type=int, default=None)
    p.add_argument("--log-every-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    return p


def main(default_stage: str) -> None:
    args = build_argparser(default_stage).parse_args()
    out = train_stage(args)
    if _env_is_main_process():
        print(f"wrote run {out}", flush=True)

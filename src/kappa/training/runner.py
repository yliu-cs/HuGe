"""Shared training loop for Stage 1/2/3."""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from kappa.data.collator import KappaCollator
from kappa.data.datamodule import build_stage_dataset, limit_dataset, write_split_manifest
from kappa.data.sampler import BlockShuffleSampler
from kappa.data.schemas import ActionSchema
from kappa.data.video_reader import set_frame_cache_size
from kappa.models.qwen_sonicfm import QwenSonicFM, QwenSonicFMConfig
from kappa.training.checkpointing import build_checkpoint_metadata
from kappa.training.losses import qwen_sonicfm_loss


warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False because encoder_layer\.norm_first was True",
    category=UserWarning,
    module=r"torch\.nn\.modules\.transformer",
)

DEFAULT_DEEPSPEED_CONFIG = "configs/deepspeed/ds_zero2.json"

@dataclass(frozen=True)
class StageDefaults:
    learning_rate: float
    batch_size: int
    epochs: int
    sample_stride: int
    checkpointing_steps: int
    weight_decay: float = 0.01
    num_workers: int = 8
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
    action_hidden_size: int = 1024
    num_action_layers: int = 16
    num_attention_heads: int = 16
    state_dropout_prob: float = 0.2


STAGE_DEFAULTS: dict[str, StageDefaults] = {
    # Large, weakly supervised visual-language pretraining. Hand bits are
    # masked, so the hand loss weight is irrelevant but kept small.
    "stage1": StageDefaults(
        learning_rate=1e-4,
        batch_size=96,
        epochs=1,
        sample_stride=10,
        checkpointing_steps=10000,
        warmup_ratio=0.05,
        max_text_tokens=48,
    ),
    # Main whole-body posttraining on HE + WBT. Denser sampling and stronger
    # hand supervision because 66-D targets are fully available.
    "stage2": StageDefaults(
        learning_rate=1e-4,
        batch_size=96,
        epochs=1,
        sample_stride=1,
        checkpointing_steps=1000,
        warmup_ratio=0.05,
        max_text_tokens=48,
    ),
    # Real-robot adaptation. Lower LR, denser sampling, frequent checkpoints.
    "stage3": StageDefaults(
        learning_rate=3e-5,
        batch_size=32,
        epochs=30,
        sample_stride=1,
        checkpointing_steps=10000,
        weight_decay=0.005,
        warmup_ratio=0.1,
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
        return apply_stage_defaults(parsed, provided)


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


def train_stage(args: argparse.Namespace) -> Path:
    accelerator = None
    if not args.use_accelerate:
        raise ValueError("training must run with accelerate; use scripts/train/train_kappa.sh")
    if args.use_accelerate:
        from accelerate import Accelerator, DeepSpeedPlugin

        accelerator_kwargs: dict[str, Any] = {
            "mixed_precision": None if args.mixed_precision == "no" else args.mixed_precision,
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
                f"got {accelerator.num_processes}. Use scripts/train/train_kappa.sh."
            )
        device = accelerator.device
    else:
        device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    _log_main(
        accelerator,
        (
            f"[{args.stage}] start: device={device} processes={getattr(accelerator, 'num_processes', 1)} "
            f"mixed_precision={args.mixed_precision} deepspeed={args.use_deepspeed} "
            f"deepspeed_config={args.deepspeed_config or 'plugin-default'}"
        ),
    )
    stats = args.proprio_stats_path
    if stats is None and args.stage in {"stage2", "posttrain_he_wbt"}:
        stats = "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json"
    if stats is None and args.stage in {"stage3", "finetune_prop"}:
        stats = "resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json"
    _log_main(
        accelerator,
        (
            f"[{args.stage}] building dataset: sample_stride={args.sample_stride} "
            f"max_episodes={args.max_episodes or 'all'} stats={stats or 'none'}"
        ),
    )
    dataset = build_stage_dataset(
        args.stage,
        sample_stride=args.sample_stride,
        max_episodes=args.max_episodes,
        proprio_stats_path=stats,
        prop_roots=args.prop_roots,
        prop_ego_roots=args.prop_ego_roots,
    )
    if accelerator is not None:
        accelerator.wait_for_everyone()
    train_ds = limit_dataset(dataset, max_samples=args.max_train_samples)
    _log_main(
        accelerator,
        (
            f"[{args.stage}] dataset ready: train_samples={len(train_ds)} "
            f"validation=disabled"
        ),
    )
    collator = KappaCollator(
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
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_sampler: Sampler[int] | None = None
    spans_fn = getattr(train_ds, "episode_spans", None)
    if args.block_shuffle_size and spans_fn is not None:
        spans = spans_fn()
        if len(spans):
            train_sampler = BlockShuffleSampler(spans, block_size=args.block_shuffle_size, generator=generator)
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=None if train_sampler is not None else generator,
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
        action_hidden_size=args.action_hidden_size or 1024,
        num_action_layers=args.num_action_layers or 16,
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
    if args.init_ckpt and str(args.init_ckpt).strip().lower() not in {"", "none", "null"}:
        _log_main(accelerator, f"[{args.stage}] loading init checkpoint: {args.init_ckpt}")
        _load_checkpoint_state(model, args.init_ckpt, accelerator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    from transformers import get_scheduler

    scaler = torch.amp.GradScaler("cuda", enabled=accelerator is None and args.mixed_precision == "fp16" and device.type == "cuda")
    out_dir = Path(args.output_dir) / args.stage / _run_dir_name(args, accelerator)
    if _is_main_process(accelerator):
        out_dir.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    with RunLogTee(out_dir, enabled=_is_main_process(accelerator)):
        _log_main(accelerator, f"[{args.stage}] logging to {out_dir / 'train.log'}")
        if _is_main_process(accelerator):
            write_split_manifest(out_dir / "data_manifest.json", dataset, len(train_ds), extra={"stage": args.stage, "stats": stats, "validation": "disabled"})
            (out_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
        metadata = build_checkpoint_metadata(
            model_name_or_path=args.base_vlm_path,
            qwen_processor_path=args.base_vlm_path,
            proprio_stats_path=stats,
            action_schema=ActionSchema().to_dict(),
            action_hidden_size=config.action_hidden_size,
            num_action_layers=config.num_action_layers,
            num_attention_heads=config.num_attention_heads,
            model_architecture=config.model_architecture,
            state_dropout_prob=state_dropout_prob,
            image_transform="gr00t_explicit_train_random_eval_center_256",
            freeze_qwen=args.freeze_qwen,
            qwen_lora=args.qwen_lora,
            train_qwen_lora=args.train_qwen_lora,
            qwen_lora_r=args.qwen_lora_r,
            qwen_lora_alpha=args.qwen_lora_alpha,
            qwen_lora_dropout=args.qwen_lora_dropout,
            source_git_commit=_git_commit(),
        )
        _log_main(accelerator, f"[{args.stage}] optimizer ready: lr={args.learning_rate:.2e} weight_decay={args.weight_decay}")
        scheduler_total_steps = args.max_steps if args.max_steps is not None else len(train_loader) * args.epochs
        warmup_steps = args.warmup_steps if args.warmup_steps and args.warmup_steps > 0 else math.ceil(scheduler_total_steps * (args.warmup_ratio or 0.0))
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=scheduler_total_steps,
        )
        if accelerator is not None:
            _log_main(accelerator, f"[{args.stage}] accelerator.prepare starting")
            prepare_debug = _env_flag("KAPPA_DEBUG_PREPARE")
            if prepare_debug:
                timeout = int(os.environ.get("KAPPA_PREPARE_TRACEBACK_SECONDS", "300"))
                _log_rank(accelerator, f"[{args.stage}] entering accelerator.prepare device={device}")
                faulthandler.dump_traceback_later(timeout, repeat=True)
            try:
                model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
                    model,
                    optimizer,
                    train_loader,
                    lr_scheduler,
                )
            finally:
                if prepare_debug:
                    faulthandler.cancel_dump_traceback_later()
            if prepare_debug:
                _log_rank(accelerator, f"[{args.stage}] exited accelerator.prepare device={device}")
            _log_main(accelerator, f"[{args.stage}] accelerator.prepare done")
        total_train_steps = args.max_steps if args.max_steps is not None else len(train_loader) * args.epochs

        _log_main(
            accelerator,
            (
                f"[{args.stage}] scheduler ready: epochs={args.epochs} total_steps={total_train_steps} "
                f"scheduler_steps={scheduler_total_steps} warmup_steps={warmup_steps} "
                f"checkpointing_steps={args.checkpointing_steps}"
            ),
        )

        global_step = 0
        train_start = time.time()
        progress_total = total_train_steps
        stop_training = False
        with tqdm(
            total=progress_total,
            desc=f"{args.stage} steps",
            unit="step",
            disable=not _is_main_process(accelerator),
        ) as pbar:
            for epoch in range(args.epochs):
                if args.max_steps is not None and global_step >= args.max_steps:
                    break
                model.train()
                for batch in train_loader:
                    if args.max_steps is not None and global_step >= args.max_steps:
                        stop_training = True
                        break
                    batch = move_batch(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    autocast_enabled = args.mixed_precision == "bf16" and device.type == "cuda" and not args.use_deepspeed
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
                        losses = qwen_sonicfm_loss(pred, batch["normalized_actions"], sonic_mask=batch["sonic_mask"])
                    if scaler.is_enabled():
                        scaler.scale(losses["loss"]).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if accelerator is not None:
                            accelerator.backward(losses["loss"])
                        else:
                            losses["loss"].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                    lr_scheduler.step()
                    global_step += 1
                    pbar.update(1)
                    current_loss = float(losses["loss"].detach())
                    current_lr = lr_scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        step=global_step,
                        loss=f"{current_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                    )
                    if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                        _log_main(accelerator, f"[{args.stage}] saving checkpoint: step={global_step} path={out_dir / f'step_{global_step}'}")
                        _save_model(model, out_dir / f"step_{global_step}", {**metadata, "global_step": global_step, "epoch": epoch}, accelerator, stats)
                        _log_main(accelerator, f"[{args.stage}] checkpoint saved: step={global_step}")
                if args.save_epoch_checkpoints:
                    _log_main(accelerator, f"[{args.stage}] saving epoch checkpoint: step={global_step} path={out_dir / f'step_{global_step}'}")
                    _save_model(
                        model,
                        out_dir / f"step_{global_step}",
                        {**metadata, "global_step": global_step, "epoch": epoch},
                        accelerator,
                        stats,
                    )
                    _log_main(accelerator, f"[{args.stage}] epoch checkpoint saved: step={global_step}")
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_training = True
                if stop_training:
                    break
        _log_main(accelerator, f"[{args.stage}] saving final checkpoint: step={global_step} path={out_dir / f'step_{global_step}'}")
        _save_model(model, out_dir / f"step_{global_step}", {**metadata, "global_step": global_step}, accelerator, stats)
        _log_main(accelerator, f"[{args.stage}] done: output_dir={out_dir} total_time={_format_seconds(time.time() - train_start)}")
    return out_dir


def _save_model(model: QwenSonicFM, path: Path, metadata: dict[str, Any], accelerator: Any | None = None, proprio_stats_path: str | None = None) -> None:
    state_dict = None
    if accelerator is not None:
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if not accelerator.is_main_process:
            return
        model = accelerator.unwrap_model(model)
    model.save_checkpoint(path, metadata=metadata, state_dict=state_dict)
    if proprio_stats_path:
        src = Path(proprio_stats_path)
        if src.exists():
            shutil.copy2(src, path / "proprio_stats_sonic.json")
            (path / "proprio_norm_config.json").write_text(
                json.dumps({"use_percentiles": True, "clip_outliers": True}, indent=2),
                encoding="utf-8",
            )


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
    p.add_argument("--proprio-stats-path", default=None)
    p.add_argument("--prop-roots", nargs="*", default=None)
    p.add_argument("--prop-ego-roots", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--mixed-precision", choices=["no", "bf16", "fp16"], default="bf16")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=None)
    p.add_argument("--frame-cache-size", type=int, default=None)
    p.add_argument("--block-shuffle-size", type=int, default=None)
    p.add_argument("--drop-last", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sample-stride", type=int, default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--checkpointing-steps", type=int, default=None)
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

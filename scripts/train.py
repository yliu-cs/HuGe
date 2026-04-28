"""Main training entry point."""

from __future__ import annotations

import datetime
import gc
import json
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration
from tqdm import tqdm
from importlib import import_module

from src.utils import batch_str_to_tensor, seed_everything, initialize_overwatch, flatten, StepTimer

overwatch = initialize_overwatch(__name__)

MAX_TRAINING_EPOCHS = 1_000_000

# Per-worker shm budget. The previous 48MB estimate left no fragmentation
# margin and let a 64MB /dev/shm pretend it could host 1 worker — which
# crashed with a bus error mid-step. 96MB matches what we actually see in
# practice once you factor in decord buffers, mmap caches, prefetch queue,
# and the IPC pickling overhead.
_SHM_PER_WORKER_BYTES = 96 * 1024 * 1024


def _autoscale_num_workers(config) -> None:
    """Cap config.train.num_workers to what /dev/shm can support.

    Tensor handoff between DataLoader workers and the main process always
    routes through /dev/shm on Linux (both file_descriptor and file_system
    sharing strategies). When shm is too small the workers are killed with a
    bus error before the first batch arrives, so we detect this up front
    rather than crashing 18s into training.

    If shm can't safely host even one worker (typical of a stock 64MB
    container tmpfs), fall back to in-process loading (``num_workers=0``).
    That removes shm IPC entirely at the cost of overlap with GPU compute —
    still strictly better than crashing.
    """
    requested = getattr(config.train, "num_workers", 0)
    if requested <= 0:
        return
    try:
        shm_stat = os.statvfs("/dev/shm")
        shm_bytes = shm_stat.f_blocks * shm_stat.f_frsize
    except (FileNotFoundError, OSError):
        return
    affordable = max(0, shm_bytes // _SHM_PER_WORKER_BYTES)
    if affordable >= requested:
        return
    if affordable < 1:
        overwatch.warning(
            f"/dev/shm is {shm_bytes // (1024 * 1024)}MiB; not enough to host "
            f"a DataLoader worker safely (need ~{_SHM_PER_WORKER_BYTES // (1024 * 1024)}MiB "
            f"each). Falling back to num_workers=0 (in-process loading). "
            f"Increase --shm-size on the container to restore parallel loading."
        )
        config.train.num_workers = 0
        return
    overwatch.warning(
        f"/dev/shm is {shm_bytes // (1024 * 1024)}MiB, can only safely host "
        f"{affordable} DataLoader worker(s); downgrading num_workers "
        f"{requested} -> {affordable}. Increase --shm-size on the container "
        f"to restore parallel data loading."
    )
    config.train.num_workers = affordable


def _initialize_accelerator(trainer) -> Accelerator:
    os.makedirs(trainer.project_dir, exist_ok=True)
    logging_dir = os.path.join(trainer.project_dir, trainer.cfg.log.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=trainer.project_dir, logging_dir=logging_dir
    )

    kwargs = dict(
        gradient_accumulation_steps=trainer.cfg.train.gradient_accumulation_steps,
        mixed_precision=trainer.cfg.train.mixed_precision,
        log_with=trainer.cfg.log.report_to,
        project_config=accelerator_project_config,
    )

    train_cfg = trainer.cfg.train
    if train_cfg.use_deepspeed:
        if train_cfg.deepspeed_config:
            ds_plugin = DeepSpeedPlugin(hf_ds_config=train_cfg.deepspeed_config)
        else:
            ds_plugin = DeepSpeedPlugin(
                zero_stage=train_cfg.zero_stage,
                gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
                gradient_clipping=train_cfg.max_grad_norm or 0.0,
                offload_optimizer_device="cpu" if train_cfg.offload_optimizer else "none",
                offload_param_device="cpu" if train_cfg.offload_param else "none",
                zero3_init_flag=(train_cfg.zero_stage == 3),
                zero3_save_16bit_model=(train_cfg.zero_stage == 3),
            )
        kwargs["deepspeed_plugin"] = ds_plugin
        overwatch.info(f"DeepSpeed enabled: ZeRO-{train_cfg.zero_stage}, offload_optimizer={train_cfg.offload_optimizer}, offload_param={train_cfg.offload_param}")

    accelerator = Accelerator(**kwargs)

    if overwatch.is_rank_zero() and not trainer.cfg.eval:
        os.makedirs(trainer.project_dir, exist_ok=True)
        with open(f"{trainer.project_dir}/run_config.json", "w") as f:
            f.write(trainer.cfg.model_dump_json(indent=4))

    trainer.accelerator = accelerator
    return accelerator


def train(config):
    if torch.cuda.is_available():
        torch.cuda.set_device(overwatch.local_rank())
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if config.seed:
        seed_everything(config.seed)

    from src.trainers import Trainer
    trainer = Trainer.instantiate(config, overwatch.local_rank())
    overwatch.info("Initializing models...")
    trainer.init_models()
    trainer.load_pretrained_weights()
    accelerator = _initialize_accelerator(trainer)
    overwatch.add_file_handler(trainer.project_dir)

    overwatch.info(f"Training task: '{config.train.name}'")
    overwatch.info(f"Run name: {trainer.run_name}")

    # Datasets must be built before reading max_training_steps: when a config
    # drives training off num_train_epochs, the property derives total steps
    # from len_train_dataset, which is only assigned inside create_datasets().
    train_dataset, val_dataset = trainer.create_datasets()
    _autoscale_num_workers(config)
    trainer.create_dataloaders(train_dataset, val_dataset)
    overwatch.info(f"Training dataset size: {trainer.len_train_dataset}")
    overwatch.info(f"Max training steps: {trainer.max_training_steps}")

    overwatch.info("Initializing optimizer and scheduler...")
    trainer.create_optimizer_and_scheduler(trainer.max_training_steps)

    overwatch.info(f"Global train batch size: {trainer.global_train_batch_size}")
    overwatch.info(f"Num processes: {trainer.world_size}")

    trainer.prepare(accelerator)
    global_step = initial_global_step = trainer.resume_from_checkpoint()[0]
    epoch_start = global_step // trainer.num_steps_per_epoch

    overwatch.info(f"Accelerator working directory: {trainer.project_dir}")
    trainer.set_train()

    metrics_path = os.path.join(trainer.project_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "a", encoding="utf-8")

    profiling = config.log.enable_profiling
    timer = None
    if profiling:
        timer = StepTimer(
            window_size=config.log.profiling_window,
            use_cuda=torch.cuda.is_available(),
            use_mps=torch.backends.mps.is_available(),
        )
        trainer._step_timer = timer
        overwatch.info(f"Profiling enabled (window={config.log.profiling_window})")

    progress_bar = tqdm(
        range(0, trainer.max_training_steps),
        initial=initial_global_step,
        desc="Training steps",
        disable=not overwatch.is_rank_zero(),
    )

    for epoch in range(epoch_start, MAX_TRAINING_EPOCHS):
        accelerator.wait_for_everyone()
        for local_step, batch in enumerate(trainer.train_dataloader):
            if timer:
                timer.start("step_total")
                timer.start("data_loading")
            batch = batch_str_to_tensor(batch)
            if timer:
                timer.stop("data_loading")

            sync_gradients, losses = trainer.step(batch, global_step, local_step)

            if sync_gradients:
                if timer:
                    timer.start("logging")

                log_metrics = flatten({**losses, "epoch": epoch}, parent_key="train")

                if timer:
                    timer.stop("step_total")
                    timer.sync()
                    log_metrics.update(timer.summary())

                trainer.log(log_metrics)

                if overwatch.is_rank_zero():
                    metrics_file.write(json.dumps({"step": global_step, **log_metrics}) + "\n")
                    metrics_file.flush()

                if (global_step + 1) % config.train.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    save_path = trainer.save_checkpoint(global_step + 1)
                    if overwatch.is_rank_zero():
                        tqdm.write(f"Saved state to {save_path}")

                if config.train.validation_steps > 0 and (
                    global_step % config.train.validation_steps == 0
                ):
                    if timer:
                        timer.stop("logging")
                        timer.sync()
                        timer.start("validation")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    trainer.set_eval()
                    autocast_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
                    with torch.no_grad(), torch.autocast(
                        device_type=autocast_device, dtype=trainer.dtype
                    ):
                        eval_losses = trainer.evaluate()
                        if eval_losses is not None:
                            trainer.log(flatten(eval_losses, parent_key="eval"))
                    trainer.set_train()
                    if timer:
                        timer.stop("validation")
                        timer.sync()
                        timer.start("logging")

                if timer:
                    timer.stop("logging")
                    timer.sync()

                progress_bar.update()
                # Fixed-width scientific notation for lr so warmup (0 → 1e-9 →
                # 1e-7) and post-warmup magnitudes stay visually comparable.
                postfix = {
                    "loss": f"{losses.get('loss', 0):.4f}",
                    "lr": f"{trainer.lr:.3e}",
                }
                progress_bar.set_postfix(postfix, refresh=False)
                global_step += 1

            if global_step >= trainer.max_training_steps:
                break

        if global_step >= trainer.max_training_steps:
            break

    accelerator.wait_for_everyone()
    trainer.finalize()
    progress_bar.close()
    metrics_file.close()

    if timer and overwatch.is_rank_zero():
        overwatch.info("=== Profiling Summary ===")
        for name, avg in sorted(timer.averages().items()):
            overwatch.info(f"  {name}: {avg:.4f}s avg")
    overwatch.info("Training complete!")
    accelerator.end_training()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--local_rank")]

    env_vars = [
        "OMP_NUM_THREADS", "HF_HOME", "TORCH_HOME", "HF_TOKEN",
        "CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"
    ]
    for var in env_vars:
        overwatch.info(f"{var}: {os.environ.get(var, 'Not Set')}")

    overwatch.info("Parsing configuration...")
    try:
        module = import_module(f"src.config.train.{args[0]}")
        DynamicLaunchConfigClass = getattr(module, "DynamicLaunchConfig")
        from tyro import cli
        config = cli(
            DynamicLaunchConfigClass,
            args=args[1:]
        )
    except Exception as e:
        overwatch.error(f"Failed to import config module: {e}")
        raise e

    train(config)

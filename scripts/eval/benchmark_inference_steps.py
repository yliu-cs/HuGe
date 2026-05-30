"""Benchmark Kappa action sampling latency for different inference step counts."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kappa.data.collator import KappaCollator
from kappa.data.datamodule import build_stage_dataset
from kappa.models.qwen_sonicfm import QwenSonicFM
from kappa.training.runner import move_batch


def _default_proprio_stats(stage: str, override: str | None) -> str | None:
    if override:
        return override
    stage = stage.lower()
    if stage in {"stage2", "he_wbt", "posttrain_he_wbt"}:
        return "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json"
    if stage in {"stage3", "prop", "proprietary", "finetune_prop", "prop_ego", "proprietary_ego"}:
        return "resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json"
    return None


def _collator_for(model: QwenSonicFM, use_qwen_processor: bool | None) -> KappaCollator:
    use_qwen = bool(model.config.use_qwen) if use_qwen_processor is None else bool(use_qwen_processor)
    return KappaCollator(
        qwen_processor_path=model.config.base_vlm_path,
        use_qwen_processor=use_qwen,
        model_max_length=model.config.model_max_length,
        max_text_tokens=model.config.max_text_tokens,
        char_max_length=model.config.char_max_length,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _prepare_batch(args: argparse.Namespace, model: QwenSonicFM, device: torch.device) -> dict[str, Any]:
    stats_path = _default_proprio_stats(args.stage, args.proprio_stats_path)
    dataset = build_stage_dataset(
        args.stage,
        sample_stride=args.sample_stride,
        max_episodes=args.max_episodes,
        proprio_stats_path=stats_path,
        prop_roots=args.prop_roots,
        prop_ego_roots=args.prop_ego_roots,
        include_replay=False,
    )
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    if args.start_index < 0 or args.start_index >= len(dataset):
        raise IndexError(f"--start-index {args.start_index} outside [0, {len(dataset) - 1}]")
    end = min(args.start_index + args.batch_size, len(dataset))
    items = [dataset[i] for i in range(args.start_index, end)]
    if len(items) < args.batch_size:
        raise ValueError(f"only {len(items)} samples available from --start-index; lower --batch-size")
    batch = _collator_for(model, args.use_qwen_processor)(items)
    return move_batch(batch, device)


@torch.inference_mode()
def _predict(model: QwenSonicFM, batch: dict[str, Any], steps: int) -> torch.Tensor:
    return model.predict_action(
        input_ids=batch["input_ids"],
        images=batch["images"],
        state=batch["state"],
        state_mask=batch["state_mask"],
        qwen_inputs=batch.get("qwen_inputs"),
        num_inference_steps=steps,
    )


def _measure_steps(
    model: QwenSonicFM,
    batch: dict[str, Any],
    device: torch.device,
    *,
    steps: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        _predict(model, batch, steps)
    _sync(device)

    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _predict(model, batch, steps)
        _sync(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "steps": steps,
        "repeats": repeats,
        "mean_ms": float(np.mean(times_ms)),
        "p50_ms": _percentile(times_ms, 50),
        "p95_ms": _percentile(times_ms, 95),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "--ckpt-path", dest="ckpt", required=True)
    parser.add_argument("--stage", default="stage3")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--proprio-stats-path", default=None)
    parser.add_argument("--prop-roots", nargs="*", default=None)
    parser.add_argument("--prop-ego-roots", nargs="*", default=None)
    parser.add_argument("--use-qwen-processor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--csv", default=None, help="optional path to write benchmark rows as CSV")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    args = _parse_args()

    if any(step <= 0 for step in args.steps):
        raise ValueError("--steps values must be positive")
    if args.batch_size <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--batch-size and --repeats must be positive; --warmup must be non-negative")

    device = torch.device(args.device)
    print(f"[bench] loading checkpoint {args.ckpt}")
    model = QwenSonicFM.from_checkpoint(args.ckpt, map_location=device).to(device).eval()
    batch = _prepare_batch(args, model, device)
    print(
        "[bench] "
        f"device={device} batch_size={args.batch_size} horizon={model.config.action_horizon} "
        f"checkpoint_default_steps={model.config.num_inference_steps}"
    )

    rows = [
        _measure_steps(
            model,
            batch,
            device,
            steps=step,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for step in args.steps
    ]

    print("steps,mean_ms,p50_ms,p95_ms,min_ms,max_ms,repeats")
    for row in rows:
        print(
            f"{row['steps']},{row['mean_ms']:.3f},{row['p50_ms']:.3f},"
            f"{row['p95_ms']:.3f},{row['min_ms']:.3f},{row['max_ms']:.3f},{row['repeats']}"
        )

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["steps", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "repeats"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[bench] wrote {out}")


if __name__ == "__main__":
    main()

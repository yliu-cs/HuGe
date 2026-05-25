#!/usr/bin/env python3
"""Benchmark deploy-faithful Kappa inference latency.

This script mirrors the inference path launched by
``scripts/eval/run_kappa_sim_test.sh``:

  * model loading goes through ``src.inference.kappa_serve_rtc._load_model``;
  * proprio normalization / scheduler setup goes through ``_build_controller``;
  * image preprocessing uses the server's 224x320 deploy transform;
  * timed calls use ``RealTimeChunkController._predict_chunk_rtc``.

It intentionally does not launch WebSocket, bridge, WBC, or MuJoCo. The goal is
to measure Kappa model inference speed without transport / control-loop noise.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.kappa_serve_rtc import (  # noqa: E402
    _TRAIN_IMG_TRANSFORM,
    _build_controller,
    _load_model,
)
from src.inference.kappa_rtc_controller import RealTimeChunkController  # noqa: E402


DEFAULT_CKPT = ".runs/kappa/170R+85E/real/checkpoints/ckpt_1300/model.safetensors"
DEFAULT_BACKBONE = "resource/model/Qwen3-VL-2B-Instruct"
DEFAULT_DATA_ROOT = "resource/dataset/proprietary/real/2026-05-23"
DEFAULT_PROMPT = "Grab the cola, turn left, throw into the trash bin"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _summarize_ms(values_ms: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values_ms),
        "median": statistics.median(values_ms),
        "min": min(values_ms),
        "max": max(values_ms),
        "p90": _percentile(values_ms, 90),
        "p95": _percentile(values_ms, 95),
        "p99": _percentile(values_ms, 99),
    }


def _print_latency_table(title: str, values_ms: list[float]) -> None:
    stats = _summarize_ms(values_ms)
    print(f"\n[{title}] n={len(values_ms)}")
    print(
        "  mean={mean:.2f} ms  p50={median:.2f} ms  p90={p90:.2f} ms  "
        "p95={p95:.2f} ms  p99={p99:.2f} ms  min={min:.2f} ms  max={max:.2f} ms"
        .format(**stats)
    )


def _resolve_proprietary_root(path: str | Path) -> Path:
    root = Path(path)
    if (root / "meta" / "episodes.jsonl").exists():
        return root
    for child in ("2026-05-23", "2026-05-16"):
        dated = root / child
        if (dated / "meta" / "episodes.jsonl").exists():
            return dated
    raise FileNotFoundError(
        f"Could not find proprietary episodes.jsonl under {root}"
    )


def _make_dummy_obs(text: str) -> dict[str, Any]:
    # Server receives raw 480x640 frames, then applies _TRAIN_IMG_TRANSFORM.
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    return {
        "image": _TRAIN_IMG_TRANSFORM(img),
        "text": text,
        "proprio": np.zeros(64, dtype=np.float32),
    }


def _load_dataset_obs(args: argparse.Namespace) -> dict[str, Any]:
    from src.eval.episode_source import FrameStream
    from src.eval.prop_episode_source import PropEpisodeSource

    data_root = _resolve_proprietary_root(args.data_root)
    source = PropEpisodeSource(
        data_root=data_root,
        proprio_cache_root=args.proprio_cache_root or None,
        seed=args.seed,
    )
    episode = source.by_index(args.episode_index)
    t_seconds = float(args.sample_time)

    stream = FrameStream(episode.video_path, fps=episode.video_fps)
    try:
        image = stream.frame_at(max(0.0, min(t_seconds, episode.duration_s)))
    finally:
        stream.close()

    proprio = source.proprio_at(episode.episode_index, t_seconds)
    text = args.prompt or episode.instruction or DEFAULT_PROMPT
    return {
        "image": _TRAIN_IMG_TRANSFORM(image.convert("RGB")),
        "text": text,
        "proprio": proprio,
    }


def _make_server_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        weights=args.weights,
        backbone=args.backbone,
        use_proprio=args.use_proprio,
        proprio_stats=args.proprio_stats,
        normalize_mode=args.normalize_mode,
        ctrl_hz=args.ctrl_hz,
        s_min=args.s_min,
    )


def _load_controller(args: argparse.Namespace) -> RealTimeChunkController:
    server_args = _make_server_args(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper, cfg = _load_model(server_args, device)
    if args.num_inference_steps is not None:
        cfg.num_inference_steps = int(args.num_inference_steps)
    factory = _build_controller(server_args, wrapper, cfg, device)
    controller = factory()
    print(
        "[bench] loaded controller: "
        f"device={controller.device} dtype={controller.dtype} "
        f"H={controller.H} sonic_dim={controller.SONIC_DIM} "
        f"num_inference_steps={controller.num_inference_steps} "
        f"use_proprio={controller.use_proprio}"
    )
    return controller


def _benchmark_rtc_chunks(
    controller: RealTimeChunkController,
    obs: dict[str, Any],
    warmup: int,
    iters: int,
    delay: int,
) -> tuple[list[float], np.ndarray]:
    prev = np.zeros((controller.H, controller.SONIC_DIM), dtype=np.float32)
    device = controller.device

    print(f"[bench] warmup rtc chunks: {warmup}")
    for _ in range(warmup):
        prev = controller._predict_chunk_rtc(obs, prev_sonic=prev, delay=delay)
        _sync(device)

    latencies: list[float] = []
    print(f"[bench] timed rtc chunks: {iters}")
    for i in range(iters):
        start = time.perf_counter()
        prev = controller._predict_chunk_rtc(obs, prev_sonic=prev, delay=delay)
        _sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(elapsed_ms)
        if (i + 1) % max(1, iters // 10) == 0:
            print(f"[bench] {i + 1}/{iters}: {elapsed_ms:.2f} ms")
    return latencies, prev


@torch.no_grad()
def _benchmark_breakdown(
    controller: RealTimeChunkController,
    obs: dict[str, Any],
    prev_sonic: np.ndarray,
    warmup: int,
    iters: int,
    delay: int,
) -> dict[str, list[float]]:
    device = controller.device
    dtype = controller.dtype
    H = controller.cfg.action_chunk_size
    B = 1

    def run_once(record: bool) -> dict[str, float]:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        prompt = controller._build_inputs(obs)
        proprio = controller._proprio_tensor(obs, B=1, dtype=dtype, device=device)
        _sync(device)
        if record:
            timings["preprocess"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        cache = controller.model.encode_latent(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            pixel_values=prompt["pixel_values"],
            image_grid_thw=prompt["image_grid_thw"],
            T_action=H,
            T_proprio=1 if controller.use_proprio else 0,
            proprio=proprio,
        )
        _sync(device)
        if record:
            timings["encode_latent"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        sonic = torch.randn(
            cache["B"], H, controller.SONIC_DIM,
            device=device, dtype=cache["dtype"],
        )
        prev_t = torch.from_numpy(prev_sonic).to(
            device=device, dtype=cache["dtype"],
        ).unsqueeze(0)
        delay_t = torch.tensor([int(delay)], device=device, dtype=torch.long)
        arange_t = torch.arange(H, device=device)
        prefix_mask = arange_t[None, :] < delay_t[:, None]

        controller.scheduler.set_timesteps(
            controller.num_inference_steps, device=device,
        )
        n_train = controller.scheduler.config.num_train_timesteps
        for timestep in controller.scheduler.timesteps:
            t_scalar = (timestep / n_train).to(cache["dtype"])
            t_sonic = t_scalar.expand(cache["B"])
            sonic = torch.where(prefix_mask[..., None], prev_t, sonic)
            v_sonic = controller.model.denoise_step_rtc(
                cache=cache,
                noisy_sonic_tokens=sonic,
                timesteps_sonic=t_sonic,
                prev_sonic=prev_t,
                delay_sonic=delay_t,
            )
            sonic = controller.scheduler.step(v_sonic, timestep, sonic).prev_sample
        sonic = torch.where(prefix_mask[..., None], prev_t, sonic)
        _ = sonic[0].float().cpu().numpy()
        _sync(device)
        if record:
            timings["denoise_loop"] = (time.perf_counter() - t0) * 1000.0
        return timings

    for _ in range(warmup):
        run_once(record=False)

    out = {"preprocess": [], "encode_latent": [], "denoise_loop": []}
    print(f"[bench] timed breakdown iterations: {iters}")
    for _ in range(iters):
        timings = run_once(record=True)
        for key, value in timings.items():
            out[key].append(value)
    return out


def _write_json(
    path: str | None,
    args: argparse.Namespace,
    chunk_ms: list[float],
    breakdown: dict[str, list[float]] | None,
    controller: RealTimeChunkController,
) -> None:
    if not path:
        return
    payload = {
        "args": vars(args),
        "model": {
            "H": controller.H,
            "sonic_dim": controller.SONIC_DIM,
            "num_inference_steps": controller.num_inference_steps,
            "device": str(controller.device),
            "dtype": str(controller.dtype),
            "use_proprio": controller.use_proprio,
        },
        "rtc_chunk_ms": chunk_ms,
        "rtc_chunk_summary_ms": _summarize_ms(chunk_ms),
        "breakdown_ms": breakdown,
        "breakdown_summary_ms": (
            {k: _summarize_ms(v) for k, v in breakdown.items()}
            if breakdown is not None else None
        ),
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[bench] wrote json: {output}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", default=os.environ.get("KAPPA_CKPT", DEFAULT_CKPT))
    ap.add_argument("--backbone", default=os.environ.get("KAPPA_BACKBONE", DEFAULT_BACKBONE))
    ap.add_argument(
        "--proprio-stats",
        default=os.environ.get("KAPPA_PROPRIO_STATS") or None,
        help="Optional override. When omitted, the controller auto-discovers "
             "proprio_stats.json next to the checkpoint (bundled by Trainer).",
    )
    ap.add_argument("--device", default=os.environ.get("KAPPA_DEVICE", "cuda:0"))
    ap.add_argument("--ctrl-hz", type=float, default=float(os.environ.get("CTRL_HZ", "50")))
    ap.add_argument("--s-min", type=int, default=8)
    ap.add_argument("--normalize-mode", default="bounds_q99", choices=["bounds_q99", "bounds"])
    ap.add_argument("--use-proprio", action="store_true", default=True)
    ap.add_argument("--no-proprio", dest="use_proprio", action="store_false")
    ap.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Optional experiment override. Omit to match the server default.",
    )

    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--delay", type=int, default=6)
    ap.add_argument(
        "--breakdown-iters",
        type=int,
        default=10,
        help="Set to 0 to skip preprocess / encode / denoise breakdown.",
    )
    ap.add_argument("--json-out", default=None)

    ap.add_argument(
        "--input-source",
        choices=["offline", "dummy"],
        default="offline",
        help="offline uses the same proprietary cache inputs as sim test.",
    )
    ap.add_argument(
        "--data-root",
        default=os.environ.get("OFFLINE_DATA_ROOT", DEFAULT_DATA_ROOT),
    )
    ap.add_argument(
        "--proprio-cache-root",
        default=os.environ.get("OFFLINE_PROPRIO_CACHE_ROOT", "") or None,
    )
    ap.add_argument(
        "--episode-index",
        type=int,
        default=int(os.environ.get("OFFLINE_EPISODE_INDEX", "123")),
    )
    ap.add_argument("--sample-time", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=int(os.environ.get("OFFLINE_SEED", "0")))
    ap.add_argument("--prompt", default=os.environ.get("BRIDGE_PROMPT", DEFAULT_PROMPT))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Kappa checkpoint not found: {args.weights}")
    if args.use_proprio and args.proprio_stats and not Path(args.proprio_stats).exists():
        raise FileNotFoundError(f"Proprio stats override not found: {args.proprio_stats}")

    torch.set_grad_enabled(False)
    controller = _load_controller(args)
    obs = (
        _load_dataset_obs(args)
        if args.input_source == "offline"
        else _make_dummy_obs(args.prompt)
    )

    chunk_ms, prev = _benchmark_rtc_chunks(
        controller=controller,
        obs=obs,
        warmup=args.warmup,
        iters=args.iters,
        delay=args.delay,
    )
    _print_latency_table("rtc chunk inference", chunk_ms)

    mean_ms = statistics.fmean(chunk_ms)
    chunks_per_s = 1000.0 / mean_ms
    tokens_per_s = chunks_per_s * controller.H
    print(
        f"  throughput={chunks_per_s:.3f} chunks/s  "
        f"{tokens_per_s:.1f} sonic-tokens/s  "
        f"query-rate-at-s_min={args.ctrl_hz / args.s_min:.3f} Hz"
    )

    breakdown = None
    if args.breakdown_iters > 0:
        breakdown = _benchmark_breakdown(
            controller=controller,
            obs=obs,
            prev_sonic=prev,
            warmup=max(1, min(args.warmup, 2)),
            iters=args.breakdown_iters,
            delay=args.delay,
        )
        for key, values in breakdown.items():
            _print_latency_table(key, values)

    _write_json(args.json_out, args, chunk_ms, breakdown, controller)


if __name__ == "__main__":
    main()

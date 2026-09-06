"""Compare complete Real episodes using checkpoint stats and repeatable noise.

Reuses offline_compare.py without modifying it. Each run randomly selects one
complete Real episode shared by every checkpoint under .runs/finetune_real.
Use --dry-run to inspect the selection and paths without loading models or writing.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import sysconfig
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ARCHITECTURES = ("qwenpi_v3", "qwengr00t")


def _ensure_cudnn_library_path() -> None:
    """Prefer the same cuDNN as train_huge.sh before importing PyTorch."""
    cudnn_lib = Path(sysconfig.get_path("purelib")) / "nvidia/cudnn/lib"
    for name in ("libcudnn.so.9", "libcudnn_graph.so.9"):
        if not (cudnn_lib / name).is_file():
            raise FileNotFoundError(f"incomplete virtualenv cuDNN installation: {cudnn_lib / name}")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current.split(":")[0] == str(cudnn_lib):
        return
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(cudnn_lib) + (f":{current}" if current else "")
    # glibc reads this search path at process startup; changing os.environ in
    # the running interpreter does not update subsequent dlopen searches.
    print(f"[real-offline] restarting with virtualenv cuDNN first: {cudnn_lib}", flush=True)
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", nargs="+",
        default=[f".runs/finetune_real/{arch}/full/stage3/*/step_*" for arch in ARCHITECTURES],
        help="portable checkpoint directories or quoted globs (default: both Real runs)",
    )
    parser.add_argument(
        "--prop-root", "--prop-roots", dest="prop_root", type=Path,
        default=Path("resource/datasets/proprietary/real/cola_merged_cleaned_0608_384"),
        help="one proprietary Real cache root",
    )
    parser.add_argument("--out-dir", type=Path, default=Path(".runs/eval/real384"))
    parser.add_argument("--episode-idx", nargs="+", type=int, help="explicit zero-based indices; default: one random episode shared by all checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, choices=[1], default=1)
    parser.add_argument("--chunk-keep-steps", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true", help="read metadata only; no inference or output files")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_inference_steps < 1 or args.seed < 0:
        parser.error("batch size/inference steps must be positive and seed must be nonnegative")
    if not 1 <= args.chunk_keep_steps <= 50:
        parser.error("--chunk-keep-steps must be in [1, 50]")
    return args


def _checkpoint_info(checkpoint: Path) -> dict[str, Any]:
    for name in ("_SUCCESS", "config.json", "model.safetensors", "proprio_stats_sonic.json", "proprio_norm_config.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"required checkpoint file is missing: {checkpoint / name}")
    payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config = payload["config"]
    architecture = config["model_architecture"]
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported Real architecture: {architecture}")
    if (config["action_horizon"], config["action_dim"], config["state_dim"]) != (50, 66, 34):
        raise ValueError(f"{checkpoint}: expected horizon=50, action_dim=66, state_dim=34")
    return {
        "checkpoint": str(checkpoint),
        "architecture": architecture,
        "run": checkpoint.parent.name,
        "step": checkpoint.name,
        "proprio_stats_path": str(checkpoint / "proprio_stats_sonic.json"),
        "proprio_norm_config": json.loads((checkpoint / "proprio_norm_config.json").read_text(encoding="utf-8")),
    }


def _evaluate_checkpoint(
    offline: Any,
    job: dict[str, Any],
    dataset: Any,
    episode_indices: list[int],
    args: argparse.Namespace,
) -> None:
    import torch

    device = torch.device(args.device)
    checkpoint = Path(job["checkpoint"])
    print(f"[real-offline] loading {checkpoint}", flush=True)
    model = offline.QwenSonicFM.from_checkpoint(checkpoint, map_location=device).to(device).eval()
    collator = offline._collator_for(model, None)
    # The original helper inherits training augmentation defaults.
    collator.image_transform_mode = "eval"
    collator.image_color_jitter = False
    collator.qwen_image_size = tuple(model.config.qwen_image_size)
    collator.image_crop_fraction = model.config.image_crop_fraction
    label = f"{job['architecture']} / {job['run']} / {job['step']}"
    for episode_idx in episode_indices:
        episode_id = dataset.episodes[episode_idx].episode_id
        seed = (args.seed + int(episode_id)) % (2**63)
        generator = torch.Generator(device=device).manual_seed(seed)
        # ponytail: fixed batch size/window order gives matching per-episode
        # noise streams; explicit per-window noise is needed to regroup batches.
        seeded_model = SimpleNamespace(
            config=model.config,
            predict_action=partial(model.predict_action, generator=generator),
        )
        output = Path(job["output_dir"]) / f"episode_{episode_id}.png"
        offline._run_visualization(
            seeded_model, dataset, device=device, collator=collator,
            episode_idx=episode_idx, timesteps=None, max_plot_windows=0,
            chunk_keep_steps=args.chunk_keep_steps, batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            out_path=output, title_prefix=f"{label} / seed={seed}",
        )
        print(f"[real-offline] plot -> {output}", flush=True)


def main() -> None:
    args = _parse_args()
    if args.device.startswith("cuda"):
        _ensure_cudnn_library_path()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/huge-matplotlib")

    import offline_compare as offline
    from huge.data.prop_dataset import PropDataset
    from huge.data.state_normalizer import StateNormConfig, StateNormalizer

    checkpoints = [Path(path).resolve() for path in offline._expand_ckpts(args.ckpt)]
    jobs = [_checkpoint_info(path) for path in checkpoints]
    normalizers = [
        StateNormalizer(
            json.loads(Path(job["proprio_stats_path"]).read_text(encoding="utf-8")),
            StateNormConfig(**job["proprio_norm_config"]),
        )
        for job in jobs
    ]
    dataset = PropDataset(args.prop_root.resolve(), sample_stride=1, strict_video=True)
    if not dataset.episodes:
        raise ValueError("Real dataset contains no episodes")
    # Select once, independently of the fixed action-noise seed.
    episode_indices = (
        [random.SystemRandom().randrange(len(dataset.episodes))]
        if args.episode_idx is None else sorted(set(args.episode_idx))
    )
    if not episode_indices or any(i < 0 or i >= len(dataset.episodes) for i in episode_indices):
        raise ValueError(f"episode indices must be in [0, {len(dataset.episodes) - 1}]")
    episodes = [dataset.episodes[i] for i in episode_indices]
    output_root = args.out_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for job in jobs:
        job["output_dir"] = str(output_root / job["architecture"] / job["run"] / job["step"])
    if len({job["output_dir"] for job in jobs}) != len(jobs):
        raise ValueError("checkpoint architecture/run/step names collide; evaluate these copies separately")
    windows = sum((ep.length + args.chunk_keep_steps - 1) // args.chunk_keep_steps for ep in episodes)
    print(
        f"[real-offline] checkpoints={len(jobs)} episodes={len(episodes)} "
        f"frames_per_checkpoint={sum(ep.length for ep in episodes)} "
        f"windows_per_checkpoint={windows} plots={len(jobs) * len(episodes)}",
        flush=True,
    )
    print(f"[real-offline] selected episodes: {[(i, ep.episode_id) for i, ep in zip(episode_indices, episodes, strict=True)]}", flush=True)
    print(f"[real-offline] sample_stride=1 max_plot_windows=0 image_transform=eval seed={args.seed}", flush=True)
    print(f"[real-offline] output root: {output_root}", flush=True)
    for job in jobs:
        print(f"[real-offline] {job['checkpoint']} -> {job['output_dir']}", flush=True)
        print(f"[real-offline] stats: {job['proprio_stats_path']} config={job['proprio_norm_config']}", flush=True)
    if args.dry_run:
        print("[real-offline] dry run complete; no models loaded or output files written", flush=True)
        return

    output_root.mkdir(parents=True, exist_ok=False)
    config = {
        "dataset_root": str(args.prop_root.resolve()),
        "sample_stride": 1, "max_plot_windows": 0,
        "chunk_keep_steps": args.chunk_keep_steps,
        "batch_size": args.batch_size, "num_inference_steps": args.num_inference_steps,
        "device": args.device, "seed": args.seed,
        "noise": "dedicated generator per episode; seed=(base_seed+int(episode_id))%2**63; keep batching/window order fixed",
        "image_transform_mode": "eval", "image_color_jitter": False,
        "episode_selection": "random_one" if args.episode_idx is None else "explicit",
        "episodes": [{"index": i, "episode_id": ep.episode_id, "frames": ep.length} for i, ep in zip(episode_indices, episodes, strict=True)],
        "checkpoints": jobs,
    }
    (output_root / "evaluation_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for job, normalizer in zip(jobs, normalizers, strict=True):
        dataset.proprio_normalizer = normalizer
        _evaluate_checkpoint(offline, job, dataset, episode_indices, args)
        # All model/partial references leave scope before loading the next model.
        gc.collect()
        if args.device.startswith("cuda"):
            offline.torch.cuda.empty_cache()
    print(f"[real-offline] done: {len(jobs) * len(episodes)} plots under {output_root}", flush=True)


if __name__ == "__main__":
    main()

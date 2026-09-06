"""CPU-only preflight for the paired Real/Ego Stage 3 training path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from huge.data.collator import HugeCollator
from huge.data.proprio_stats import validate_proprio_stats_contract
from huge.data.schemas import PROPRIO_LAYOUT
from huge.data.validation_manifest import ensure_validation_manifest, load_validation_manifest
from huge.training.checkpointing import write_json_atomic
from huge.training.runner import (
    _cumulative_stats_contract,
    _default_cumulative_stats_path,
    _validation_sources,
    build_argparser,
    build_dataset_from_args,
    validate_init_checkpoint,
)


def check_data(args: Any) -> dict[str, Any]:
    if args.stage != "stage3" or args.real_num_episodes is None or args.init_ckpt is None:
        raise ValueError("preflight requires Stage 3 Real/Ego counts and --init-ckpt")
    checkpoint = validate_init_checkpoint(args)
    stage_scope, sources, task_scope = _cumulative_stats_contract(args)
    stats = args.proprio_stats_path or _default_cumulative_stats_path(args, stage_scope)
    stats_payload = validate_proprio_stats_contract(stats, stage_scope=stage_scope, sources=sources, task_scope=task_scope)
    validation = ensure_validation_manifest(
        args.dataset_root, sources=_validation_sources(args, sources),
        stage_scope=stage_scope, stats_path=stats, force=args.force_data_audit,
    )
    inventories = load_validation_manifest(validation["publication"])["sources"]
    for tag in sources:
        for field in ("n_files", "n_frames"):
            if stats_payload["_meta"]["sources"][tag][field] != inventories[tag][field]:
                raise ValueError(f"{tag}: stats/inventory {field} mismatch")
    dataset = build_dataset_from_args(args, stats, inventories)
    collator = HugeCollator(
        use_qwen_processor=True, qwen_processor_path=args.base_vlm_path,
        image_transform_mode="eval", max_text_tokens=args.max_text_tokens,
        model_max_length=args.model_max_length, char_max_length=args.char_max_length,
        model_architecture=args.model_architecture, action_horizon=args.action_horizon,
        oft_action_token=args.oft_action_token,
    )
    summaries = []
    samples = []
    for adapter in dataset.datasets:
        probes = sorted({0, len(adapter) // 2, len(adapter) - 1})
        for index in probes:
            item = adapter[index]
            episode_index, t = adapter._resolve(index)
            episode = adapter.episodes[episode_index]
            valid = min(50, episode.length - t)
            action = np.load(episode.token_path, mmap_mode="r")
            assert np.array_equal(item["normalized_actions"][:valid], action[t:t + valid])
            assert np.all(item["sonic_mask"][:valid] == 1) and not item["sonic_mask"][valid:].any()
            assert np.all(item["state_mask"] == 1)
            assert np.isfinite(item["state"]).all() and np.abs(item["state"]).max() <= 1.0001
            assert 0 <= item["video_frame_idx"] < episode.video_len_hint
            assert item["image"] is not None
            samples.append(item)
        normalizer = adapter.proprio_normalizer
        lo = np.concatenate([normalizer._params[name][0] for name, _, _ in PROPRIO_LAYOUT])
        hi = np.concatenate([normalizer._params[name][1] for name, _, _ in PROPRIO_LAYOUT])
        clipped = np.zeros(34, dtype=np.int64)
        count = 0
        for episode in adapter.episodes:
            state = np.load(episode.proprio_path, mmap_mode="r")
            clipped += ((state < lo) | (state > hi)).sum(axis=0)
            count += len(state)
        summaries.append({"source": adapter.dataset_id, "selected_frames": count,
                          "clipped_fraction_per_channel": (clipped / count).tolist(),
                          "constant_stats_channels": np.flatnonzero(hi == lo).tolist()})
    batch = collator(samples)
    return {"status": "passed", "mode": "cpu_data_only", "config": vars(args),
            "initial_checkpoint_config": checkpoint, "stats": str(Path(stats).resolve()),
            "stats_scope": stage_scope, "stats_reference_sources": {tag: str(root.resolve()) for tag, root in sources.items()},
            "validation": validation, "dataset": dataset.data_contract(), "normalization": summaries,
            "probe_samples": [{key: item[key] for key in ("dataset_id", "episode_id", "frame_idx", "video_frame_idx", "video_path")} for item in samples],
            "batch_shapes": {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")}}


def main() -> None:
    parser = build_argparser("stage3")
    parser.set_defaults(output_dir=".runs/preprocess/stage3_real_ego_preflight")
    args = parser.parse_args()
    report = check_data(args)
    output = Path(args.output_dir) / "data_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, report)
    print(f"CPU data preflight passed; report={output}")


if __name__ == "__main__":
    main()

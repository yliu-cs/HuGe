"""Offline Huge evaluation with prediction-vs-ground-truth plots."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from tqdm import tqdm

from huge.data.collator import HugeCollator
from huge.data.datamodule import build_stage_dataset
from huge.data.schemas import ACTION_DIM, MOTION_TOKEN_DIM
from huge.models.qwen_sonicfm import QwenSonicFM
from huge.training.runner import move_batch


def _natural_path_key(path: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path))


def _expand_ckpts(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        matches = sorted(glob.glob(os.path.expanduser(raw)), key=_natural_path_key) if glob.has_magic(raw) else [raw]
        if glob.has_magic(raw) and not matches:
            raise FileNotFoundError(f"no checkpoints matched {raw!r}")
        for match in matches:
            resolved = str(Path(match).expanduser().resolve())
            if resolved not in seen:
                seen.add(resolved)
                out.append(match)
    return out


def _ckpt_label(path: str, index: int) -> str:
    p = Path(path)
    label = p.parent.name if p.name == "model.safetensors" else p.name
    if label.startswith("step_") and p.parent.parent.name:
        label = f"{p.parent.parent.name}_{label}"
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._")
    return label or f"ckpt{index:02d}"


def _default_proprio_stats(stage: str, override: str | None) -> str | None:
    if override:
        return override
    stage = stage.lower()
    if stage in {"stage2", "he_wbt", "posttrain_he_wbt"}:
        return "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats_sonic.json"
    if stage in {"stage3", "prop", "proprietary", "finetune_prop", "prop_ego", "proprietary_ego"}:
        return "resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json"
    return None


def _collator_for(model: QwenSonicFM, use_qwen_processor: bool | None) -> HugeCollator:
    use_qwen = bool(model.config.use_qwen) if use_qwen_processor is None else bool(use_qwen_processor)
    return HugeCollator(
        qwen_processor_path=model.config.base_vlm_path,
        use_qwen_processor=use_qwen,
        model_max_length=model.config.model_max_length,
        max_text_tokens=model.config.max_text_tokens,
        char_max_length=model.config.char_max_length,
        model_architecture=model.config.model_architecture,
        action_horizon=model.config.action_horizon,
        oft_action_token=model.config.oft_action_token,
    )


def _set_predict_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


@torch.no_grad()
def _predict_batch(
    model: QwenSonicFM,
    batch: dict[str, Any],
    device: torch.device,
    *,
    num_inference_steps: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    batch = move_batch(batch, device)
    if seed is not None:
        _set_predict_seed(seed, device)
    return model.predict_action(
        input_ids=batch["input_ids"],
        images=batch["images"],
        state=batch["state"],
        state_mask=batch["state_mask"],
        qwen_inputs=batch.get("qwen_inputs"),
        action_token_mask=batch.get("action_token_mask"),
        num_inference_steps=num_inference_steps,
    )


def _episode_span(dataset: Any, episode_idx: int) -> tuple[int, int]:
    spans_fn = getattr(dataset, "episode_spans", None)
    if spans_fn is None:
        raise ValueError("visualization requires a dataset with episode_spans()")
    spans = np.asarray(spans_fn(), dtype=np.int64)
    if not len(spans):
        raise ValueError("dataset has no episode spans")
    if episode_idx < 0:
        episode_idx += len(spans)
    if episode_idx < 0 or episode_idx >= len(spans):
        raise IndexError(f"episode_idx {episode_idx} outside [0, {len(spans) - 1}]")
    start, length = spans[episode_idx]
    return int(start), int(length)


def _plot_starts(length: int, stride: int, explicit: list[int] | None, max_windows: int) -> list[int]:
    if explicit:
        starts = sorted(set(t for t in explicit if 0 <= t < length))
    else:
        starts = list(range(0, length, stride))
    if max_windows > 0:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError("no valid plot timesteps selected")
    return starts


def _selected_visual_starts(
    *,
    ep_len: int,
    timesteps: list[int] | None,
    fallback_timesteps: list[int] | None,
    stride: int,
    max_windows: int,
) -> list[int]:
    explicit = timesteps if timesteps is not None else fallback_timesteps
    return _plot_starts(ep_len, max(1, int(stride)), explicit, max_windows)


def _raw_item_image(collator: HugeCollator, item: dict[str, Any]) -> np.ndarray:
    # Reuse the collator's image resolution logic so deploy-frame and dataset-video
    # samples follow the same path before we override the item image.
    return np.asarray(collator._image(item).convert("RGB"), dtype=np.uint8)


def _copy_item_with_image(item: dict[str, Any], image: np.ndarray) -> dict[str, Any]:
    out = dict(item)
    out["image"] = np.asarray(image, dtype=np.uint8)
    return out


def _action_delta_metrics(ref: np.ndarray, alt: np.ndarray) -> dict[str, float]:
    diff = np.asarray(alt, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
    motion = diff[..., :MOTION_TOKEN_DIM]
    hand = diff[..., MOTION_TOKEN_DIM:ACTION_DIM]
    motion_l2 = np.linalg.norm(motion, axis=-1)
    return {
        "motion_l2_mean_per_step": float(motion_l2.mean()),
        "motion_l2_p95_per_step": float(np.quantile(motion_l2, 0.95)),
        "motion_l2_max_per_step": float(motion_l2.max(initial=0.0)),
        "motion_abs_mean": float(np.abs(motion).mean()),
        "hand_abs_mean": float(np.abs(hand).mean()) if hand.size else 0.0,
        "right_hand_abs_mean": float(np.abs(diff[..., MOTION_TOKEN_DIM + 1]).mean()),
        "all_abs_mean": float(np.abs(diff).mean()),
    }


def _print_metric_row(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix}: "
        f"motion_l2_mean={metrics['motion_l2_mean_per_step']:.5f} "
        f"motion_l2_p95={metrics['motion_l2_p95_per_step']:.5f} "
        f"motion_abs={metrics['motion_abs_mean']:.5f} "
        f"hand_abs={metrics['hand_abs_mean']:.5f} "
        f"right_hand_abs={metrics['right_hand_abs_mean']:.5f}"
    )


@torch.no_grad()
def _run_visual_ablation(
    model: QwenSonicFM,
    dataset: Any,
    *,
    device: torch.device,
    collator: HugeCollator,
    episode_idx: int,
    timesteps: list[int] | None,
    fallback_timesteps: list[int] | None,
    stride: int,
    max_windows: int,
    batch_size: int,
    num_inference_steps: int | None,
    seed: int,
) -> dict[str, Any]:
    ep_start, ep_len = _episode_span(dataset, episode_idx)
    starts = _selected_visual_starts(
        ep_len=ep_len,
        timesteps=timesteps,
        fallback_timesteps=fallback_timesteps,
        stride=stride,
        max_windows=max_windows,
    )
    base_items = [dataset[ep_start + local] for local in starts]
    raw_images = [_raw_item_image(collator, item) for item in base_items]
    base_preds: list[np.ndarray] = []
    for offset in tqdm(range(0, len(base_items), batch_size), desc="visual ablation original"):
        batch = collator(base_items[offset : offset + batch_size])
        pred = _predict_batch(
            model,
            batch,
            device,
            num_inference_steps=num_inference_steps,
            seed=seed + offset,
        )
        base_preds.append(pred.detach().cpu().numpy().astype(np.float32))
    base_pred = np.concatenate(base_preds, axis=0)

    black_items = [
        _copy_item_with_image(item, np.zeros_like(image, dtype=np.uint8))
        for item, image in zip(base_items, raw_images, strict=True)
    ]
    first_items = [
        _copy_item_with_image(item, raw_images[0])
        for item in base_items
    ]
    next_items = [
        _copy_item_with_image(item, raw_images[(idx + 1) % len(raw_images)])
        for idx, item in enumerate(base_items)
    ]
    variants = {
        "black_image": black_items,
        "first_image_for_all": first_items,
        "next_image_rotated": next_items,
    }
    report: dict[str, Any] = {
        "type": "visual_ablation",
        "episode_idx": episode_idx,
        "starts": starts,
        "seed": seed,
        "variants": {},
    }
    print(f"[visual-ablation] episode={episode_idx} starts={starts}")
    for name, items in variants.items():
        preds: list[np.ndarray] = []
        for offset in tqdm(range(0, len(items), batch_size), desc=f"visual ablation {name}"):
            batch = collator(items[offset : offset + batch_size])
            pred = _predict_batch(
                model,
                batch,
                device,
                num_inference_steps=num_inference_steps,
                seed=seed + offset,
            )
            preds.append(pred.detach().cpu().numpy().astype(np.float32))
        pred_arr = np.concatenate(preds, axis=0)
        metrics = _action_delta_metrics(base_pred, pred_arr)
        per_sample = [
            {"start": int(start), **_action_delta_metrics(base_pred[i : i + 1], pred_arr[i : i + 1])}
            for i, start in enumerate(starts)
        ]
        report["variants"][name] = {"summary": metrics, "per_sample": per_sample}
        _print_metric_row(f"[visual-ablation] {name}", metrics)
    return report


@torch.no_grad()
def _run_same_state_image_swap(
    model: QwenSonicFM,
    dataset: Any,
    *,
    device: torch.device,
    collator: HugeCollator,
    episode_idx: int,
    timesteps: list[int] | None,
    fallback_timesteps: list[int] | None,
    stride: int,
    max_windows: int,
    num_inference_steps: int | None,
    seed: int,
) -> dict[str, Any]:
    ep_start, ep_len = _episode_span(dataset, episode_idx)
    starts = _selected_visual_starts(
        ep_len=ep_len,
        timesteps=timesteps,
        fallback_timesteps=fallback_timesteps,
        stride=stride,
        max_windows=max_windows,
    )
    source_items = [dataset[ep_start + local] for local in starts]
    source_images = [_raw_item_image(collator, item) for item in source_items]
    anchor = source_items[0]
    anchor_item = _copy_item_with_image(anchor, source_images[0])
    anchor_pred = _predict_batch(
        model,
        collator([anchor_item]),
        device,
        num_inference_steps=num_inference_steps,
        seed=seed,
    ).detach().cpu().numpy().astype(np.float32)

    variants: list[tuple[str, int | None, np.ndarray]] = [
        ("anchor_original", int(starts[0]), source_images[0]),
        ("black_image", None, np.zeros_like(source_images[0], dtype=np.uint8)),
    ]
    variants.extend((f"image_from_t{start}", int(start), image) for start, image in zip(starts[1:], source_images[1:], strict=True))
    report: dict[str, Any] = {
        "type": "same_state_different_images",
        "episode_idx": episode_idx,
        "anchor_start": int(starts[0]),
        "image_starts": starts,
        "seed": seed,
        "variants": {},
    }
    print(f"[visual-swap] episode={episode_idx} anchor_start={starts[0]} image_starts={starts}")
    for name, source_start, image in variants:
        item = _copy_item_with_image(anchor, image)
        pred = _predict_batch(
            model,
            collator([item]),
            device,
            num_inference_steps=num_inference_steps,
            seed=seed,
        ).detach().cpu().numpy().astype(np.float32)
        metrics = _action_delta_metrics(anchor_pred, pred)
        report["variants"][name] = {
            "source_start": source_start,
            "summary": metrics,
        }
        _print_metric_row(f"[visual-swap] {name}", metrics)
    return report


def _plot_episode(
    out_path: Path,
    *,
    pred_full: np.ndarray,
    gt_full: np.ndarray,
    starts: list[int],
    motion_dims: list[int],
    title: str,
) -> None:
    n_rows = len(motion_dims) + 1
    fig = plt.figure(figsize=(14, 1.45 * n_rows + 0.8), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, 1)
    xs = np.arange(gt_full.shape[0])

    for row, dim in enumerate(motion_dims):
        ax = fig.add_subplot(gs[row, 0])
        ax.plot(xs, gt_full[:, dim], color="tab:blue", lw=1.0, label="gt")
        ax.plot(xs, pred_full[:, dim], color="tab:red", lw=1.0, label="pred")
        for s in starts:
            ax.axvline(s, color="gray", lw=0.4, alpha=0.35)
        ax.set_ylabel(f"dim {dim}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(title, fontsize=10)
        if row < len(motion_dims) - 1:
            ax.set_xticklabels([])

    row = len(motion_dims)
    ax = fig.add_subplot(gs[row, 0])
    for hand_idx, color in enumerate(("tab:orange", "tab:green")):
        dim = MOTION_TOKEN_DIM + hand_idx
        ax.step(xs, gt_full[:, dim], where="post", lw=1.0, color=color, alpha=0.65, label=f"gt hand{hand_idx}")
        ax.plot(xs, pred_full[:, dim], lw=1.0, color=color, linestyle="--", label=f"pred hand{hand_idx}")
    for s in starts:
        ax.axvline(s, color="gray", lw=0.4, alpha=0.35)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("hand bits", fontsize=9)
    ax.set_xlabel("episode sample tick")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def _run_visualization(
    model: QwenSonicFM,
    dataset: Any,
    *,
    device: torch.device,
    collator: HugeCollator,
    episode_idx: int,
    timesteps: list[int] | None,
    max_plot_windows: int,
    chunk_keep_steps: int | None,
    batch_size: int,
    num_inference_steps: int | None,
    out_path: Path,
    title_prefix: str,
) -> Path:
    ep_start, ep_len = _episode_span(dataset, episode_idx)
    horizon = int(model.config.action_horizon)
    keep_steps = horizon if chunk_keep_steps is None else int(chunk_keep_steps)
    if keep_steps <= 0 or keep_steps > horizon:
        raise ValueError(f"chunk_keep_steps must be in [1, {horizon}], got {keep_steps}")
    starts = _plot_starts(ep_len, keep_steps, timesteps, max_plot_windows)
    pred_full = np.full((ep_len, ACTION_DIM), np.nan, dtype=np.float32)
    gt_full = np.full((ep_len, ACTION_DIM), np.nan, dtype=np.float32)

    for offset in tqdm(range(0, len(starts), batch_size), desc="visual windows"):
        chunk_starts = starts[offset : offset + batch_size]
        items = [dataset[ep_start + local] for local in chunk_starts]
        batch = collator(items)
        pred = _predict_batch(
            model,
            batch,
            device,
            num_inference_steps=num_inference_steps,
        ).detach().cpu().numpy().astype(np.float32)
        target = batch["normalized_actions"].numpy().astype(np.float32)
        mask = batch["sonic_mask"].numpy().astype(np.float32)

        for local, pred_i, target_i, mask_i in zip(chunk_starts, pred, target, mask, strict=True):
            valid_rows = np.flatnonzero(mask_i.mean(axis=-1) > 0)
            if len(valid_rows) == 0:
                continue
            valid = min(int(valid_rows[-1]) + 1, keep_steps, ep_len - local)
            pred_full[local : local + valid] = pred_i[:valid]
            gt_full[local : local + valid] = target_i[:valid]

    dims = list(range(MOTION_TOKEN_DIM))
    _plot_episode(
        out_path,
        pred_full=pred_full,
        gt_full=gt_full,
        starts=starts,
        motion_dims=dims,
        title=(
            f"{title_prefix} episode={episode_idx} windows={len(starts)} "
            f"keep={keep_steps}/{horizon} dims=all({MOTION_TOKEN_DIM})"
        ),
    )
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", "--ckpt-path", dest="ckpt", nargs="+", required=True)
    parser.add_argument("--stage", default="stage2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--proprio-stats-path", default=None)
    parser.add_argument("--prop-roots", nargs="*", default=None)
    parser.add_argument("--prop-ego-roots", nargs="*", default=None)
    parser.add_argument("--out-dir", default=".runs/eval/offline_compare", help="directory for visualization PNGs")
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--timesteps", nargs="*", type=int, default=None, help="episode-local sample starts for plots")
    parser.add_argument("--max-plot-windows", type=int, default=64, help="<=0 means no limit")
    parser.add_argument(
        "--chunk-keep-steps",
        type=int,
        default=None,
        help="keep only the first N actions from each predicted chunk when stitching plots; default keeps full horizon",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--use-qwen-processor", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--visual-ablation", action="store_true", help="compare original images against black/stale/swapped images")
    parser.add_argument("--visual-swap-test", action="store_true", help="hold proprio/prompt fixed and swap only the image")
    parser.add_argument("--visual-test-timesteps", nargs="*", type=int, default=None, help="episode-local image starts for visual sensitivity tests")
    parser.add_argument("--visual-test-stride", type=int, default=100)
    parser.add_argument("--visual-test-windows", type=int, default=8)
    parser.add_argument("--visual-test-seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    args = _parse_args()

    stats_path = _default_proprio_stats(args.stage, args.proprio_stats_path)
    dataset = build_stage_dataset(
        args.stage,
        sample_stride=args.sample_stride,
        max_episodes=args.max_episodes,
        proprio_stats_path=stats_path,
        prop_roots=args.prop_roots,
        prop_ego_roots=args.prop_ego_roots,
    )
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, ckpt in enumerate(_expand_ckpts(args.ckpt), start=1):
        label = _ckpt_label(ckpt, idx)
        print(f"[offline] loading checkpoint {ckpt}")
        model = QwenSonicFM.from_checkpoint(ckpt, map_location=device).to(device).eval()
        collator = _collator_for(model, args.use_qwen_processor)
        png_path = out_dir / f"{run_stamp}_{label}_ep{args.episode_idx:06d}.png"
        _run_visualization(
            model,
            dataset,
            device=device,
            collator=collator,
            episode_idx=args.episode_idx,
            timesteps=args.timesteps,
            max_plot_windows=args.max_plot_windows,
            chunk_keep_steps=args.chunk_keep_steps,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            out_path=png_path,
            title_prefix=label,
        )
        print(f"[offline] plot -> {png_path}")
        diagnostics: list[dict[str, Any]] = []
        if args.visual_ablation:
            diagnostics.append(
                _run_visual_ablation(
                    model,
                    dataset,
                    device=device,
                    collator=collator,
                    episode_idx=args.episode_idx,
                    timesteps=args.visual_test_timesteps,
                    fallback_timesteps=args.timesteps,
                    stride=args.visual_test_stride,
                    max_windows=args.visual_test_windows,
                    batch_size=args.batch_size,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.visual_test_seed,
                )
            )
        if args.visual_swap_test:
            diagnostics.append(
                _run_same_state_image_swap(
                    model,
                    dataset,
                    device=device,
                    collator=collator,
                    episode_idx=args.episode_idx,
                    timesteps=args.visual_test_timesteps,
                    fallback_timesteps=args.timesteps,
                    stride=args.visual_test_stride,
                    max_windows=args.visual_test_windows,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.visual_test_seed,
                )
            )
        if diagnostics:
            json_path = out_dir / f"{run_stamp}_{label}_ep{args.episode_idx:06d}_visual_sensitivity.json"
            json_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[offline] visual sensitivity -> {json_path}")


if __name__ == "__main__":
    main()

"""Dataset builders for stage training/evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset, Subset

from huge.data.egodex_dataset import EgoDexDataset
from huge.data.he_dataset import HEDataset
from huge.data.hiw_dataset import HIWDataset
from huge.data.humanoid_arena_dataset import HumanoidArenaDataset
from huge.data.prop_dataset import PropDataset
from huge.data.prop_ego_dataset import PropEgoDataset
from huge.data.sonic_dataset import MixtureDataset
from huge.data.stage1_cache_dataset import Behavior1KDataset, EgoVerseDataset, XperienceDataset
from huge.data.state_normalizer import StateNormalizer
from huge.data.wbt_dataset import WBTDataset

@dataclass(frozen=True, slots=True)
class DatasetSpec:
    kind: str
    root: Path


STAGE1_SOURCE_ORDER = ("egodex", "egoverse", "behavior1k", "xperience")
STAGE2_SOURCE_ORDER = ("he", "wbt", "hiw")


def normalize_dataset_specs(
    values: Sequence[Sequence[str] | DatasetSpec],
    *,
    stage: str,
) -> list[DatasetSpec]:
    order = STAGE1_SOURCE_ORDER if stage == "stage1" else STAGE2_SOURCE_ORDER
    specs = [
        value
        if isinstance(value, DatasetSpec)
        else DatasetSpec(kind=str(value[0]).lower(), root=Path(value[1]))
        for value in values
    ]
    kinds = [spec.kind for spec in specs]
    unknown = sorted(set(kinds) - set(order))
    if unknown:
        raise ValueError(f"unsupported {stage} dataset source kinds: {unknown}; expected {list(order)}")
    duplicates = sorted({kind for kind in kinds if kinds.count(kind) > 1})
    if duplicates:
        raise ValueError(f"duplicate {stage} dataset source kinds: {duplicates}")
    return sorted(specs, key=lambda spec: order.index(spec.kind))


def _build_egodex(spec: DatasetSpec, kwargs: dict[str, Any]) -> Dataset:
    return EgoDexDataset(spec.root, proprio_root=spec.root.with_name(f"{spec.root.name}-cache") / "g1_proprio", **kwargs)


def _build_egoverse(spec: DatasetSpec, kwargs: dict[str, Any]) -> Dataset:
    return EgoVerseDataset(spec.root, **kwargs)


def _build_behavior1k(spec: DatasetSpec, kwargs: dict[str, Any]) -> Dataset:
    return Behavior1KDataset(spec.root, **kwargs)


def _build_xperience(spec: DatasetSpec, kwargs: dict[str, Any]) -> Dataset:
    return XperienceDataset(spec.root, **kwargs)


STAGE1_DATASET_BUILDERS: dict[str, Callable[[DatasetSpec, dict[str, Any]], Dataset]] = {
    "egodex": _build_egodex,
    "egoverse": _build_egoverse,
    "behavior1k": _build_behavior1k,
    "xperience": _build_xperience,
}

def _build_he(spec: DatasetSpec, kwargs: dict[str, Any]) -> Dataset:
    return HEDataset(
        spec.root,
        cache_root=spec.root.with_name(f"{spec.root.name}-cache"),
        **kwargs,
    )


def _build_wbt(spec: DatasetSpec, kwargs: dict[str, Any]) -> Dataset:
    return WBTDataset(spec.root, **kwargs)


def _build_hiw(spec: DatasetSpec, kwargs: dict[str, Any]) -> Dataset:
    return HIWDataset(
        spec.root,
        sample_stride=kwargs["sample_stride"],
        proprio_normalizer=kwargs["proprio_normalizer"],
        max_episodes=kwargs["max_episodes"],
        validated_inventory=kwargs.get("validated_inventory"),
    )


STAGE2_DATASET_BUILDERS: dict[str, Callable[[DatasetSpec, dict[str, Any]], Dataset]] = {
    "he": _build_he,
    "wbt": _build_wbt,
    "hiw": _build_hiw,
}





class EpisodeSubset(Subset):
    """Subset that preserves episode spans for block-shuffle sampling."""

    def __init__(self, dataset: Dataset, indices: list[int], episode_lengths: list[int]) -> None:
        super().__init__(dataset, indices)
        self._episode_lengths = np.asarray(episode_lengths, dtype=np.int64)

    def episode_spans(self) -> np.ndarray:
        spans = np.empty((len(self._episode_lengths), 2), dtype=np.int64)
        offset = 0
        for i, length in enumerate(self._episode_lengths):
            spans[i] = (offset, int(length))
            offset += int(length)
        return spans


def _truncate_episode_indices(indices: list[int], lengths: list[int], limit: int | None) -> tuple[list[int], list[int]]:
    if limit is None or len(indices) <= limit:
        return indices, lengths
    kept = indices[:limit]
    out_lengths: list[int] = []
    cursor = 0
    for length in lengths:
        if cursor >= limit:
            break
        n = min(int(length), limit - cursor)
        out_lengths.append(n)
        cursor += n
    return kept, out_lengths


def _split_by_episode(
    dataset: Dataset,
    spans: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(spans))
    rng.shuffle(order)
    n_val_ep = max(1, int(round(len(spans) * val_fraction))) if len(spans) > 1 else 0
    val_eps = set(int(x) for x in order[:n_val_ep])

    train_indices: list[int] = []
    train_lengths: list[int] = []
    val_indices: list[int] = []
    val_lengths: list[int] = []
    for ep_idx, (start, length) in enumerate(spans):
        start_i = int(start)
        length_i = int(length)
        target_indices = val_indices if ep_idx in val_eps else train_indices
        target_lengths = val_lengths if ep_idx in val_eps else train_lengths
        target_indices.extend(range(start_i, start_i + length_i))
        target_lengths.append(length_i)

    train_indices, train_lengths = _truncate_episode_indices(train_indices, train_lengths, max_train_samples)
    val_indices, val_lengths = _truncate_episode_indices(val_indices, val_lengths, max_val_samples)
    return EpisodeSubset(dataset, train_indices, train_lengths), EpisodeSubset(dataset, val_indices, val_lengths)


def split_dataset(dataset: Dataset, val_fraction: float = 0.02, seed: int = 7, max_train_samples: int | None = None, max_val_samples: int | None = None) -> tuple[Dataset, Dataset]:
    spans_fn = getattr(dataset, "episode_spans", None)
    if spans_fn is not None:
        spans = np.asarray(spans_fn(), dtype=np.int64)
        if spans.size:
            return _split_by_episode(
                dataset,
                spans,
                val_fraction=val_fraction,
                seed=seed,
                max_train_samples=max_train_samples,
                max_val_samples=max_val_samples,
            )
    n = len(dataset)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if max_train_samples is not None:
        train_idx = train_idx[:max_train_samples]
    if max_val_samples is not None:
        val_idx = val_idx[:max_val_samples]
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def limit_dataset(dataset: Dataset, max_samples: int | None = None) -> Dataset:
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    spans_fn = getattr(dataset, "episode_spans", None)
    if spans_fn is not None:
        spans = np.asarray(spans_fn(), dtype=np.int64)
        if spans.size:
            indices = list(range(max_samples))
            lengths: list[int] = []
            cursor = 0
            for _, length in spans:
                if cursor >= max_samples:
                    break
                n = min(int(length), max_samples - cursor)
                lengths.append(n)
                cursor += n
            return EpisodeSubset(dataset, indices, lengths)
    return Subset(dataset, list(range(max_samples)))


def build_stage_dataset(
    stage: str,
    *,
    repo_root: str | Path = ".",
    dataset_root: str | Path | None = None,
    sample_stride: int = 1,
    max_episodes: int | None = None,
    proprio_stats_path: str | Path | None = None,
    egodex_root: str | Path | None = None,
    stage1_sources: Sequence[Sequence[str] | DatasetSpec] | None = None,
    stage2_sources: Sequence[Sequence[str] | DatasetSpec] | None = None,
    episode_fraction: float | None = None,
    episode_selection_seed: int = 7,
    prop_roots: list[str] | None = None,
    prop_ego_roots: list[str] | None = None,
    humanoid_arena_roots: list[str] | None = None,
    humanoid_arena_task: str = "all",
    validated_inventories: Mapping[str, dict[str, Any]] | None = None,
) -> Dataset:
    root = Path(repo_root)
    data = Path(dataset_root) if dataset_root is not None else root / "resource" / "dataset"
    stage = stage.lower()
    normalizer = StateNormalizer.from_json(proprio_stats_path) if proprio_stats_path else None
    if stage in {"stage1", "egodex", "pretrain_egodex"}:
        if normalizer is None:
            raise ValueError("Stage 1 requires --proprio-stats-path")
        if stage1_sources:
            specs = normalize_dataset_specs(stage1_sources, stage="stage1")
        else:
            specs = [DatasetSpec("egodex", Path(egodex_root or data / "EgoDex"))]
        kwargs = {
            "sample_stride": sample_stride,
            "max_episodes": max_episodes,
            "episode_fraction": episode_fraction,
            "episode_selection_seed": episode_selection_seed,
            "proprio_normalizer": normalizer,
        }
        datasets = [STAGE1_DATASET_BUILDERS[spec.kind](spec, kwargs) for spec in specs]
        return datasets[0] if len(datasets) == 1 else MixtureDataset(datasets)
    if stage in {"stage2", "he_wbt", "posttrain_he_wbt"}:
        specs = (
            normalize_dataset_specs(stage2_sources, stage="stage2")
            if stage2_sources
            else [
                DatasetSpec("he", data / "humanoid-everyday"),
                DatasetSpec("wbt", data / "G1_WBT"),
            ]
        )
        kwargs = {
            "sample_stride": sample_stride,
            "proprio_normalizer": normalizer,
            "max_episodes": max_episodes,
        }
        validation_tags = {"he": "HE", "wbt": "WBT", "hiw": "HIW"}
        datasets = [
            STAGE2_DATASET_BUILDERS[spec.kind](
                spec,
                {
                    **kwargs,
                    "validated_inventory": (validated_inventories or {}).get(validation_tags[spec.kind]),
                },
            )
            for spec in specs
        ]
        return datasets[0] if len(datasets) == 1 else MixtureDataset(datasets)
    if stage in {"stage3", "prop", "proprietary", "finetune_prop", "prop_ego", "proprietary_ego"}:
        datasets: list[Dataset] = []
        if stage in {"prop_ego", "proprietary_ego"}:
            roots = prop_ego_roots or [str(data / "proprietary" / "ego" / "2026-5-23")]
            datasets.extend(PropEgoDataset(path, sample_stride=sample_stride, max_episodes=max_episodes) for path in roots)
            return MixtureDataset(datasets)

        roots = [str(data / "proprietary" / "real" / "cola_merged_cleaned_0530")] if prop_roots is None else prop_roots
        datasets.extend(
            PropDataset(
                path,
                sample_stride=sample_stride,
                proprio_normalizer=normalizer,
                max_episodes=max_episodes,
            )
            for path in roots
        )
        datasets.extend(
            HumanoidArenaDataset(
                path,
                task=humanoid_arena_task,
                sample_stride=sample_stride,
                proprio_normalizer=normalizer,
                max_episodes=max_episodes,
                validated_inventory=(validated_inventories or {}).get("HUMANOID_ARENA"),
            )
            for path in (humanoid_arena_roots or [])
        )
        return datasets[0] if len(datasets) == 1 else MixtureDataset(datasets)
    raise ValueError(f"unknown stage {stage!r}")


def write_split_manifest(path: str | Path, dataset: Dataset, train_len: int, val_len: int | None = None, extra: dict[str, Any] | None = None) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_type": type(dataset).__name__, "train_len": train_len, **(extra or {})}
    if val_len is not None:
        payload["val_len"] = val_len
    dst.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

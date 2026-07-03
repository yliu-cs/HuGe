"""Dataset builders for stage training/evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Subset

from kappa.data.egodex_dataset import EgoDexDataset
from kappa.data.he_dataset import HEDataset
from kappa.data.prop_dataset import PropDataset
from kappa.data.prop_ego_dataset import PropEgoDataset
from kappa.data.sonic_dataset import MixtureDataset
from kappa.data.state_normalizer import StateNormalizer
from kappa.data.wbt_dataset import WBTDataset


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
    sample_stride: int = 1,
    max_episodes: int | None = None,
    proprio_stats_path: str | Path | None = None,
    egodex_root: str | Path | None = None,
    prop_roots: list[str] | None = None,
    prop_ego_roots: list[str] | None = None,
) -> Dataset:
    root = Path(repo_root)
    data = root / "resource" / "dataset"
    stage = stage.lower()
    normalizer = StateNormalizer.from_json(proprio_stats_path) if proprio_stats_path else None
    if stage in {"stage1", "egodex", "pretrain_egodex"}:
        return EgoDexDataset(egodex_root or data / "EgoDex", sample_stride=sample_stride, max_episodes=max_episodes)
    if stage in {"stage2", "he_wbt", "posttrain_he_wbt"}:
        he = HEDataset(
            data / "humanoid-everyday",
            cache_root=data / "humanoid-everyday-cache",
            sample_stride=sample_stride,
            proprio_normalizer=normalizer,
            max_episodes=max_episodes,
        )
        wbt = WBTDataset(
            data / "G1_WBT",
            sample_stride=sample_stride,
            proprio_normalizer=normalizer,
            max_episodes=max_episodes,
        )
        return MixtureDataset([he, wbt])
    if stage in {"stage3", "prop", "proprietary", "finetune_prop", "prop_ego", "proprietary_ego"}:
        datasets: list[Dataset] = []
        if stage in {"prop_ego", "proprietary_ego"}:
            roots = prop_ego_roots or [str(data / "proprietary" / "ego" / "2026-5-23")]
            datasets.extend(PropEgoDataset(path, sample_stride=sample_stride, max_episodes=max_episodes) for path in roots)
            return MixtureDataset(datasets)

        roots = prop_roots or [str(data / "proprietary" / "real" / "cola_merged_cleaned_0530")]
        datasets.extend(
            PropDataset(
                path,
                sample_stride=sample_stride,
                proprio_normalizer=normalizer,
                max_episodes=max_episodes,
            )
            for path in roots
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

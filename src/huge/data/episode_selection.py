"""Deterministic, nested episode selection for training ablations."""

from __future__ import annotations

import hashlib
import math
import operator
from collections import Counter
from collections.abc import Sequence

import numpy as np

SELECTION_NAMESPACE = "huge-stage1-episode-selection-v1"
SELECTION_ALGORITHM = "per_source_pcg64_permutation_v1"


def _validated_seed(base_seed: int) -> int:
    try:
        seed = operator.index(base_seed)
    except TypeError as exc:
        raise ValueError(f"base_seed must be an integer, got {base_seed!r}") from exc
    if seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {seed}")
    return seed


def _validated_episode_ids(episode_ids: Sequence[str]) -> list[str]:
    values = [str(episode_id) for episode_id in episode_ids]
    if any(not value for value in values):
        raise ValueError("episode ids must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError("episode ids must be unique")
    return values


def derive_source_seed(base_seed: int, source_id: str) -> int:
    seed = _validated_seed(base_seed)
    source = str(source_id)
    if not source:
        raise ValueError("source_id must be non-empty")
    payload = f"{SELECTION_NAMESPACE}\0{seed}\0{source}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def select_episode_indices(
    episode_count: int,
    *,
    source_id: str,
    base_seed: int,
    fraction: float | None = None,
    max_episodes: int | None = None,
) -> np.ndarray:
    """Return original-order indices from a deterministic random prefix.

    Using one source-specific permutation makes smaller fractions strict
    subsets of larger fractions for a fixed source and seed.
    """
    count = int(episode_count)
    if count < 0:
        raise ValueError(f"episode_count must be non-negative, got {count}")
    _validated_seed(base_seed)
    if not str(source_id):
        raise ValueError("source_id must be non-empty")
    if fraction is not None and max_episodes is not None:
        raise ValueError("fraction and max_episodes are mutually exclusive")
    if fraction is not None:
        value = float(fraction)
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"fraction must satisfy 0 < fraction <= 1, got {fraction}")
        keep = count if value == 1.0 or count == 0 else max(1, math.floor(count * value))
    elif max_episodes is not None:
        limit = int(max_episodes)
        if limit <= 0:
            raise ValueError(f"max_episodes must be positive, got {limit}")
        return np.arange(min(count, limit), dtype=np.int64)
    else:
        keep = count
    if keep == count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(derive_source_seed(base_seed, source_id)))
    return np.sort(rng.permutation(count)[:keep]).astype(np.int64, copy=False)


def selected_episode_digest(episode_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for episode_id in _validated_episode_ids(episode_ids):
        digest.update(episode_id.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def select_episode_ids(
    episode_ids: Sequence[str], count: int, *, source_id: str, base_seed: int = 7,
) -> list[str]:
    """Select an exact random prefix, returning IDs in canonical order."""
    ids = sorted(_validated_episode_ids(episode_ids))
    count = operator.index(count)
    if not 0 <= count <= len(ids):
        raise ValueError(f"{source_id}: requested {count} episodes from a pool of {len(ids)}")
    rng = np.random.Generator(np.random.PCG64(derive_source_seed(base_seed, source_id)))
    return [ids[i] for i in sorted(rng.permutation(len(ids))[:count])]


def task_histogram(episode_ids: Sequence[str], task_ids: Sequence[str]) -> dict[str, int]:
    episodes = _validated_episode_ids(episode_ids)
    tasks = [str(task_id) for task_id in task_ids]
    if len(episodes) != len(tasks):
        raise ValueError(f"episode/task length mismatch: {len(episodes)} != {len(tasks)}")
    if any(not task for task in tasks):
        raise ValueError("task ids must be non-empty")
    return dict(sorted(Counter(tasks).items()))


def global_selected_episode_digest(
    selections: Sequence[tuple[str, Sequence[str]]],
) -> str:
    digest = hashlib.sha256()
    seen_sources: set[str] = set()
    for source_id, episode_ids in selections:
        source = str(source_id)
        if not source or source in seen_sources:
            raise ValueError(f"source ids must be non-empty and unique, got {source!r}")
        seen_sources.add(source)
        for episode_id in _validated_episode_ids(episode_ids):
            digest.update(f"{source}\t{episode_id}\n".encode())
    return digest.hexdigest()

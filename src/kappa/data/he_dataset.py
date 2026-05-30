"""Humanoid-Everyday token dataset with task text routing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kappa.data.sonic_dataset import SonicTokenDataset


def _load_task_map(data_root: Path) -> dict[int, str]:
    path = data_root / "meta" / "tasks.jsonl"
    out: dict[int, str] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out[int(row["task_index"])] = str(row.get("description") or row.get("task") or "perform the task")
    return out


def _load_episode_language(data_root: Path, cache_root: Path) -> dict[str, str]:
    task_map = _load_task_map(data_root)
    index_path = cache_root / "meta" / "episode_index_50hz_g1.npz"
    out: dict[str, str] = {}
    if not index_path.exists():
        return out
    with np.load(index_path) as idx:
        ep_ids = idx["ep_episode_idx"]
        ep_tasks = idx["ep_task"]
        for ep, task in zip(ep_ids, ep_tasks, strict=False):
            out[str(int(ep)).zfill(6)] = task_map.get(int(task), "perform the task")
    return out


class HEDataset(SonicTokenDataset):
    def __init__(self, data_root: str | Path, cache_root: str | Path | None = None, **kwargs) -> None:
        data = Path(data_root)
        cache = Path(cache_root) if cache_root is not None else data.parent / f"{data.name}-cache"
        super().__init__(
            cache,
            dataset_id="humanoid_everyday",
            language_map=_load_episode_language(data, cache),
            **kwargs,
        )

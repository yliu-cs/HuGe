"""HumanoidArena preprocessed token dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huge.data.sonic_dataset import EpisodeRecord, SonicTokenDataset


class HumanoidArenaDataset(SonicTokenDataset):
    def __init__(self, data_root: str | Path, *, task: str = "all", **kwargs: Any) -> None:
        root = Path(data_root)
        manifest = json.loads((root / "meta" / "conversion_manifest.json").read_text(encoding="utf-8"))
        rows = manifest.get("episodes")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"invalid or empty HumanoidArena conversion manifest under {root}")
        available_tasks = sorted({str(row["task"]) for row in rows})
        if task != "all" and task not in available_tasks:
            raise ValueError(
                f"unknown HumanoidArena task {task!r}; expected 'all' or one of {available_tasks}"
            )
        self.task = task
        self._allowed_episode_ids = {
            str(row["episode_id"])
            for row in rows
            if task == "all" or row["task"] == task
        }
        self._filtered_max_episodes = kwargs.pop("max_episodes", None)
        kwargs["video_len_hint_mode"] = "token_1to1"
        kwargs.setdefault("task_id", task)
        language_map = json.loads((root / "meta" / "language_map.json").read_text(encoding="utf-8"))
        super().__init__(
            root,
            dataset_id="humanoid_arena",
            language_map=language_map,
            max_episodes=None,
            **kwargs,
        )

    def _discover_episodes(self) -> list[EpisodeRecord]:
        records = [
            record
            for record in super()._discover_episodes()
            if record.episode_id in self._allowed_episode_ids
        ]
        if self._filtered_max_episodes is not None:
            records = records[: self._filtered_max_episodes]
        if not records:
            raise FileNotFoundError(f"no HumanoidArena cache episodes found for task {self.task!r}")
        return records

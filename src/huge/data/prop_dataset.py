"""Proprietary real robot token dataset."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from huge.data.sonic_dataset import SonicTokenDataset


def _task_map(root: Path) -> dict[int, str]:
    path = root / "meta" / "tasks.jsonl"
    out: dict[int, str] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                out[int(row["task_index"])] = str(row.get("task") or row.get("description") or "perform the task")
    return out


def _language_map(root: Path) -> dict[str, str]:
    tasks = _task_map(root)
    out: dict[str, str] = {}
    for parquet in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pl.read_parquet(parquet, columns=["episode_index", "task_index"])
        if df.is_empty():
            continue
        ep = int(df["episode_index"][0])
        task = int(df["task_index"][0])
        out[str(ep).zfill(6)] = tasks.get(task, "perform the task")
    return out


class PropDataset(SonicTokenDataset):
    def __init__(self, data_root: str | Path, **kwargs) -> None:
        root = Path(data_root)
        kwargs["video_len_hint_mode"] = "token_1to1"
        super().__init__(
            root,
            dataset_id="proprietary_real",
            language_map=_language_map(root),
            **kwargs,
        )

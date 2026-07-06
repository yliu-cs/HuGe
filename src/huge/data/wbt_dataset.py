"""G1-WBT token dataset with subset task text from episodes_meta.parquet."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from huge.data.sonic_dataset import SonicTokenDataset


def _load_language_map(root: Path) -> dict[str, str]:
    meta = root / "meta" / "episodes_meta.parquet"
    out: dict[str, str] = {}
    if not meta.exists():
        return out
    df = pl.read_parquet(meta)
    for row in df.iter_rows(named=True):
        out[str(int(row["episode_index"])).zfill(6)] = str(row.get("task") or "perform the task")
    return out


class WBTDataset(SonicTokenDataset):
    def __init__(self, data_root: str | Path = "resource/dataset/G1_WBT", **kwargs) -> None:
        root = Path(data_root)
        super().__init__(
            root,
            dataset_id="g1_wbt",
            language_map=_load_language_map(root),
            **kwargs,
        )

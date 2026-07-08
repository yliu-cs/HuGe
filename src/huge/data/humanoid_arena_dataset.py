"""HumanoidArena preprocessed token dataset."""

from __future__ import annotations

import json
from pathlib import Path

from huge.data.sonic_dataset import SonicTokenDataset


class HumanoidArenaDataset(SonicTokenDataset):
    def __init__(self, data_root: str | Path, **kwargs) -> None:
        root = Path(data_root)
        kwargs["video_len_hint_mode"] = "token_1to1"
        language_map = json.loads((root / "meta" / "language_map.json").read_text(encoding="utf-8"))
        super().__init__(
            root,
            dataset_id="humanoid_arena",
            language_map=language_map,
            **kwargs,
        )

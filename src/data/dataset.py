from __future__ import annotations
from typing import Any
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset class."""

    def __init__(self, config: dict[str, Any], split: str = "train"):
        self.config = config
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raise NotImplementedError

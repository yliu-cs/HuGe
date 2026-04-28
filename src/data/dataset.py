"""Dataset base class and map-style transform wrapper.

``Dataset`` applies a per-sample transform pipeline (``repack -> field -> model``)
on top of a raw torch ``Dataset`` (HE / EgoDex). Mirrors
``psi.data.dataset.Dataset``: the raw dataset emits
``{image, text, sonic_tokens, sonic_mask}``, the transform standardizes the
shape and applies image augmentation, and ``KappaCollator`` then builds
Qwen3-VL inputs at batch level.
"""
from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """Wraps a raw dataset and applies the configured DataTransform per sample."""

    def __init__(
        self,
        raw_dataset: TorchDataset,
        transform: Callable,
        transform_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.transform = transform
        self.transform_kwargs = transform_kwargs or {}

    def __len__(self) -> int:
        return len(self.raw_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self.raw_dataset[idx]
        return self.transform(raw, **self.transform_kwargs)

from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class DataTransform(BaseModel):
    """Base data transform configuration."""

    def __call__(self, batch: dict[str, Any], split: str = "train") -> dict[str, Any]:
        return batch


class RepackTransform(DataTransform):
    """Repacks raw data into standard format."""

    pass


class FieldTransform(DataTransform):
    """Performs normalization/denormalization on specific fields."""

    pass


class ModelTransform(DataTransform):
    """Model-specific processing (e.g., tokenization, encoding)."""

    pass

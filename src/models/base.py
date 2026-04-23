from __future__ import annotations
import torch
import torch.nn as nn
from typing import Any


class BaseModel(nn.Module):
    """Base model class for all architectures."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        self.eval()
        return self.forward(*args, **kwargs)

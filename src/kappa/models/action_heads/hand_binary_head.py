"""Binary closed/open hand prediction head."""

from __future__ import annotations

import torch
from torch import nn

from kappa.data.schemas import HAND_DIM


class HandBinaryHead(nn.Module):
    def __init__(self, hidden_size: int, hand_dim: int = HAND_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hand_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)

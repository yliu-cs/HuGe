"""StarVLA-style OFT action head for Kappa."""

from __future__ import annotations

import torch
from torch import nn

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class MLPResNet(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([MLPResNetBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.fc1(self.layer_norm1(x))))
        for block in self.blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class StarVLAOFTActionHead(nn.Module):
    """Parallel continuous action regression from action placeholder hidden states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        action_dim: int = ACTION_DIM,
        action_horizon: int = ACTION_HORIZON,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.model = MLPResNet(
            input_dim=hidden_size,
            hidden_dim=hidden_size * 2,
            output_dim=action_dim,
            num_blocks=2,
            dropout=dropout,
        )

    def predict(self, action_queries: torch.Tensor) -> torch.Tensor:
        if action_queries.ndim != 3:
            raise ValueError(f"action_queries must be [B, T, H], got {tuple(action_queries.shape)}")
        batch_size, horizon, hidden_size = action_queries.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"hidden size {hidden_size} != {self.hidden_size}")
        flat = action_queries.reshape(batch_size * horizon, hidden_size)
        actions = self.model(flat)
        return actions.reshape(batch_size, horizon, self.action_dim)

    def forward(self, *, action_queries: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_actions = self.predict(action_queries)
        horizon = pred_actions.shape[1]
        if horizon > self.action_horizon:
            raise ValueError(f"horizon {horizon} exceeds configured {self.action_horizon}")
        target_actions = actions[:, :horizon].to(dtype=pred_actions.dtype, device=pred_actions.device)
        return {
            "pred_actions": pred_actions,
            "target_actions": target_actions,
            "hidden": action_queries,
        }

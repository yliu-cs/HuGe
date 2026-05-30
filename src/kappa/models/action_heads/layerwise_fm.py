"""Small Layerwise Flow-Matching action head.

This is the minimal trainable head used for smoke tests and first-pass
integration. It keeps the public ``denoise_step`` and ``predict_action``
interfaces stable so the internal implementation can later be swapped for the
full starVLA-style layerwise cross-DiT.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON, HAND_DIM, MOTION_TOKEN_DIM
from kappa.models.action_heads.hand_binary_head import HandBinaryHead


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 128) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        flat = t.reshape(-1).float()
        args = flat[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.freq_dim:
            emb = torch.nn.functional.pad(emb, (0, self.freq_dim - emb.shape[-1]))
        return self.mlp(emb.to(self.mlp[0].weight.dtype)).reshape(*t.shape, -1)


class LayerwiseFMActionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 512,
        condition_size: int = 512,
        action_horizon: int = ACTION_HORIZON,
        motion_dim: int = MOTION_TOKEN_DIM,
        hand_dim: int = HAND_DIM,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.action_horizon = action_horizon
        self.motion_dim = motion_dim
        self.hand_dim = hand_dim
        self.action_embed = nn.Linear(motion_dim, hidden_size)
        self.condition_proj = nn.Linear(condition_size, hidden_size)
        self.timestep = TimestepEmbedder(hidden_size)
        self.pos = nn.Parameter(torch.zeros(1, action_horizon, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, motion_dim),
        )
        self.hand_head = HandBinaryHead(hidden_size, hand_dim=hand_dim)
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def forward(
        self,
        noisy_motion: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if noisy_motion.shape[-1] != self.motion_dim:
            raise ValueError(f"noisy_motion last dim must be {self.motion_dim}, got {noisy_motion.shape}")
        batch, horizon, _ = noisy_motion.shape
        if horizon > self.action_horizon:
            raise ValueError(f"horizon {horizon} exceeds configured {self.action_horizon}")
        if condition.ndim == 3:
            condition = condition.mean(dim=1)
        if condition.shape[-1] != self.condition_size:
            raise ValueError(f"condition dim {condition.shape[-1]} != {self.condition_size}")
        if t.ndim == 0:
            t = t.expand(batch)
        if t.ndim == 1:
            t_emb = self.timestep(t).unsqueeze(1)
        else:
            t_emb = self.timestep(t)
        cond = self.condition_proj(condition).unsqueeze(1)
        hidden = self.action_embed(noisy_motion) + cond + t_emb + self.pos[:, :horizon]
        hidden = self.blocks(hidden)
        velocity = self.velocity_head(hidden)
        hand_logits = self.hand_head(hidden)
        return {"velocity": velocity, "hand_logits": hand_logits, "hidden": hidden}

    @torch.no_grad()
    def predict(
        self,
        condition: torch.Tensor,
        *,
        horizon: int | None = None,
        num_inference_steps: int = 4,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        horizon = horizon or self.action_horizon
        batch = condition.shape[0]
        device = condition.device
        dtype = condition.dtype
        x = torch.randn(batch, horizon, self.motion_dim, device=device, dtype=dtype, generator=generator)
        steps = max(int(num_inference_steps), 1)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch,), 1.0 - i * dt, device=device, dtype=dtype)
            pred = self.forward(x, t, condition)
            x = x - dt * pred["velocity"]
        hand_logits = self.forward(x, torch.zeros((batch,), device=device, dtype=dtype), condition)["hand_logits"]
        hand = (torch.sigmoid(hand_logits) >= 0.5).to(x.dtype)
        return torch.cat([x, hand], dim=-1)

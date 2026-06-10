"""StarVLA-style GR00T action head for Kappa."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Beta

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM
from kappa.models.action_heads.starvla_pi_head import (
    ActionEncoder,
    BasicTransformerBlock,
    MLP,
    SinusoidalTimestepEncoder,
)


class SingleContextDiT(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        depth: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        timestep_emb: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_emb=timestep_emb,
                encoder_attention_mask=encoder_attention_mask,
            )
        return self.final_norm(hidden_states)


class StarVLAGR00TActionHead(nn.Module):
    """Single-context flow-matching action head using the last VLM hidden layer."""

    def __init__(
        self,
        *,
        hidden_size: int = 1024,
        action_dim: int = ACTION_DIM,
        state_dim: int = PROPRIO_DIM,
        action_horizon: int = ACTION_HORIZON,
        num_layers: int = 16,
        num_heads: int = 16,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        num_future_tokens: int = 32,
        num_inference_steps: int = 4,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
        state_dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps
        self.noise_s = noise_s
        self.state_dropout_prob = float(state_dropout_prob)

        self.action_encoder = ActionEncoder(action_dim, hidden_size=hidden_size)
        self.timestep_encoder = SinusoidalTimestepEncoder(hidden_size)
        self.state_encoder = MLP(state_dim, hidden_size, hidden_size, dropout=dropout) if state_dim > 0 else None
        self.future_tokens = nn.Embedding(num_future_tokens, hidden_size)
        self.position_embedding = nn.Embedding(action_horizon, hidden_size)
        self.dit = SingleContextDiT(
            hidden_size=hidden_size,
            depth=num_layers,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )
        self.action_decoder = MLP(input_dim=hidden_size, hidden_dim=1024, output_dim=action_dim, dropout=dropout)
        self.decoder = self.action_decoder
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def sample_time(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample((batch_size,)).to(device=device, dtype=dtype).clamp(max=self.noise_s)
        return (self.noise_s - sample) / self.noise_s

    def _state_tokens(
        self,
        state: torch.Tensor | None,
        state_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.state_encoder is None:
            return None
        if state is None:
            state = torch.zeros((batch_size, self.state_dim), device=device, dtype=dtype)
            state_mask = torch.zeros((batch_size, 1), device=device, dtype=dtype)
        state = state.to(device=device, dtype=dtype).reshape(batch_size, -1)[:, : self.state_dim]
        state_feat = self.state_encoder(state).unsqueeze(1)
        if state_mask is not None:
            state_feat = state_feat * state_mask.reshape(batch_size, -1).to(device=device, dtype=dtype)[:, :1].unsqueeze(-1)
        if self.training and self.state_dropout_prob > 0.0:
            keep = (torch.rand((batch_size, 1, 1), device=device) >= self.state_dropout_prob).to(dtype)
            state_feat = state_feat * keep
        return state_feat

    def _build_sequence(
        self,
        actions: torch.Tensor,
        timestep_emb: torch.Tensor,
        state: torch.Tensor | None,
        state_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, horizon, _ = actions.shape
        device = actions.device
        dtype = actions.dtype
        action_features = self.action_encoder(actions, timestep_emb)
        pos_ids = torch.arange(horizon, dtype=torch.long, device=device)
        action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0).to(dtype)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).to(dtype)
        state_tokens = self._state_tokens(
            state,
            state_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if state_tokens is None:
            return torch.cat([future_tokens, action_features], dim=1)
        return torch.cat([state_tokens, future_tokens, action_features], dim=1)

    def forward(
        self,
        *,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"action dim {actions.shape[-1]} != {self.action_dim}")
        batch_size, horizon, _ = actions.shape
        if horizon > self.action_horizon:
            raise ValueError(f"horizon {horizon} exceeds configured {self.action_horizon}")
        if noise is None:
            noise = torch.randn_like(actions)
        if t is None:
            t = self.sample_time(batch_size, device=actions.device, dtype=actions.dtype)
        else:
            t = t.to(device=actions.device, dtype=actions.dtype)
        timestep_emb = self.timestep_encoder(t)
        t_view = t.reshape(batch_size, 1, 1)
        noisy = (1.0 - t_view) * noise + t_view * actions
        target_velocity = actions - noise
        hidden_states = self._build_sequence(noisy, timestep_emb, state, state_mask)
        hidden_states = self.dit(
            hidden_states,
            vl_embs,
            timestep_emb=timestep_emb,
            encoder_attention_mask=encoder_attention_mask,
        )
        action_hidden = hidden_states[:, -horizon:]
        velocity = self.action_decoder(action_hidden)
        return {
            "velocity": velocity,
            "target_velocity": target_velocity,
            "hidden": action_hidden,
        }

    @torch.no_grad()
    def predict(
        self,
        *,
        vl_embs: torch.Tensor,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        horizon: int | None = None,
        num_inference_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        horizon = horizon or self.action_horizon
        if horizon > self.action_horizon:
            raise ValueError(f"horizon {horizon} exceeds configured {self.action_horizon}")
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        dtype = vl_embs.dtype
        actions = torch.randn(batch_size, horizon, self.action_dim, device=device, dtype=dtype, generator=generator)
        steps = max(int(num_inference_steps or self.num_inference_steps), 1)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((batch_size,), step / float(steps), device=device, dtype=dtype)
            timestep_emb = self.timestep_encoder(t)
            hidden_states = self._build_sequence(actions, timestep_emb, state, state_mask)
            hidden_states = self.dit(
                hidden_states,
                vl_embs,
                timestep_emb=timestep_emb,
                encoder_attention_mask=encoder_attention_mask,
            )
            velocity = self.action_decoder(hidden_states[:, -horizon:])
            actions = actions + dt * velocity
        return actions

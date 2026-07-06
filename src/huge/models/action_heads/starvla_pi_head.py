"""StarVLA-style PI action head for Huge."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Beta

from huge.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.time_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mix = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, actions: torch.Tensor, timestep_emb: torch.Tensor) -> torch.Tensor:
        action_feat = self.action_proj(actions)
        time_feat = self.time_proj(timestep_emb).unsqueeze(1).expand(-1, actions.shape[1], -1)
        return self.mix(torch.cat([action_feat, time_feat.to(action_feat.dtype)], dim=-1))


class SinusoidalTimestepEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        max_period: float = 10000.0,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError(f"hidden_size must be even for sinusoidal embedding, got {hidden_size}")
        self.hidden_size = hidden_size
        self.max_period = float(max_period)
        self.time_scale = float(time_scale)
        self.mlp = MLP(hidden_size, hidden_size, hidden_size)
        half = hidden_size // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1).to(torch.float32) * self.time_scale
        args = t[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        mlp_dtype = self.mlp.net[0].weight.dtype
        return self.mlp(emb.to(mlp_dtype))


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_norm = AdaLayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = AdaLayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = AdaLayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ffn_multiplier, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        timestep_emb: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        x = self.self_norm(hidden_states, timestep_emb)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        hidden_states = residual + x

        residual = hidden_states
        x = self.cross_norm(hidden_states, timestep_emb)
        key_padding_mask = None
        if encoder_attention_mask is not None:
            key_padding_mask = ~encoder_attention_mask.to(torch.bool)
        x, _ = self.cross_attn(
            x,
            encoder_hidden_states.to(hidden_states.dtype),
            encoder_hidden_states.to(hidden_states.dtype),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + x

        residual = hidden_states
        x = self.ffn_norm(hidden_states, timestep_emb)
        hidden_states = residual + self.ffn(x)
        return hidden_states


class DiT(nn.Module):
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
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
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
        vl_embs_list: list[torch.Tensor],
        *,
        timestep_emb: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(vl_embs_list) != self.depth:
            raise ValueError(f"expected {self.depth} VLM layers, got {len(vl_embs_list)}")
        for block, encoder_hidden in zip(self.blocks, vl_embs_list, strict=True):
            hidden_states = block(
                hidden_states,
                encoder_hidden,
                timestep_emb=timestep_emb,
                encoder_attention_mask=encoder_attention_mask,
            )
        return self.final_norm(hidden_states)


class StarVLAPIActionHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int = 1024,
        action_dim: int = ACTION_DIM,
        state_dim: int = PROPRIO_DIM,
        action_horizon: int = ACTION_HORIZON,
        num_layers: int = 28,
        num_heads: int = 16,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
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
        self.input_embedding_dim = hidden_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_inference_steps = num_inference_steps
        self.noise_s = noise_s
        self.state_dropout_prob = float(state_dropout_prob)

        self.action_encoder = ActionEncoder(action_dim, hidden_size=self.input_embedding_dim)
        self.timestep_encoder = SinusoidalTimestepEncoder(hidden_size)
        self.state_encoder = MLP(state_dim, hidden_size, hidden_size, dropout=dropout) if state_dim > 0 else None
        self.position_embedding = nn.Embedding(action_horizon, hidden_size)
        self.dit = DiT(
            hidden_size=hidden_size,
            depth=num_layers,
            num_heads=num_heads,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )
        self.action_decoder = MLP(input_dim=hidden_size, hidden_dim=1024, output_dim=action_dim, dropout=dropout)
        self.decoder = self.action_decoder
        self.beta_dist = Beta(noise_beta_alpha, noise_beta_beta)
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
        state_feat = self.state_encoder(state.to(device=device, dtype=dtype)).unsqueeze(1)
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
        state_tokens = self._state_tokens(
            state,
            state_mask,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if state_tokens is None:
            return action_features
        return torch.cat([state_tokens, action_features], dim=1)

    def _run_dit(
        self,
        hidden_states: torch.Tensor,
        vl_embs_list: list[torch.Tensor],
        timestep_emb: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.dit(
            hidden_states,
            vl_embs_list,
            timestep_emb=timestep_emb,
            encoder_attention_mask=encoder_attention_mask,
        )

    def forward(
        self,
        *,
        vl_embs_list: list[torch.Tensor],
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
        hidden_states = self._run_dit(hidden_states, vl_embs_list, timestep_emb, encoder_attention_mask)
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
        vl_embs_list: list[torch.Tensor],
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
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype
        actions = torch.randn(batch_size, horizon, self.action_dim, device=device, dtype=dtype, generator=generator)
        steps = max(int(num_inference_steps or self.num_inference_steps), 1)
        dt = 1.0 / steps
        for step in range(steps):
            t = torch.full((batch_size,), step / float(steps), device=device, dtype=dtype)
            timestep_emb = self.timestep_encoder(t)
            hidden_states = self._build_sequence(actions, timestep_emb, state, state_mask)
            hidden_states = self._run_dit(hidden_states, vl_embs_list, timestep_emb, encoder_attention_mask)
            velocity = self.action_decoder(hidden_states[:, -horizon:])
            actions = actions + dt * velocity
        return actions


LayerwiseCrossDiTFMActionHead = StarVLAPIActionHead

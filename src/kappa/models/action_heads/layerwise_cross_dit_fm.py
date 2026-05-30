"""Layer-wise cross-attention flow-matching action head for Kappa PI."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Beta

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
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


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_size)
        self.mix = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, actions: torch.Tensor, timestep_emb: torch.Tensor) -> torch.Tensor:
        action_feat = self.action_proj(actions)
        time_feat = timestep_emb.unsqueeze(1).expand(-1, actions.shape[1], -1)
        return self.mix(torch.cat([action_feat, time_feat.to(action_feat.dtype)], dim=-1))


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


class CrossDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_proj = nn.Linear(hidden_size, hidden_size)
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
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
        hidden_states = hidden_states + self.time_proj(timestep_emb).unsqueeze(1).to(hidden_states.dtype)
        residual = hidden_states
        x = self.self_norm(hidden_states)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        hidden_states = residual + x

        residual = hidden_states
        x = self.cross_norm(hidden_states)
        key_padding_mask = None
        if encoder_attention_mask is not None:
            key_padding_mask = ~encoder_attention_mask.to(torch.bool)
        x, _ = self.cross_attn(
            x,
            encoder_hidden_states,
            encoder_hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden_states = residual + x

        residual = hidden_states
        hidden_states = residual + self.ffn(self.ffn_norm(hidden_states))
        return hidden_states


class LayerwiseCrossDiTFMActionHead(nn.Module):
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
        num_future_tokens: int = 32,
        max_seq_len: int = 1024,
        num_inference_steps: int = 4,
        noise_beta_alpha: float = 1.5,
        noise_beta_beta: float = 1.0,
        noise_s: float = 0.999,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.num_layers = num_layers
        self.num_inference_steps = num_inference_steps
        self.noise_s = noise_s

        self.timestep = TimestepEmbedder(hidden_size)
        self.action_encoder = ActionEncoder(action_dim, hidden_size)
        self.state_encoder = MLP(state_dim, hidden_size, hidden_size, dropout=dropout) if state_dim > 0 else None
        self.future_tokens = nn.Embedding(num_future_tokens, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList(
            [
                CrossDiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = MLP(hidden_size, hidden_size, action_dim, dropout=dropout)
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
        state_feat = self.state_encoder(state.to(dtype=dtype)).unsqueeze(1)
        if state_mask is not None:
            state_feat = state_feat * state_mask.reshape(batch_size, -1).to(dtype=dtype)[:, :1].unsqueeze(-1)
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
        future = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1).to(dtype)
        state_tokens = self._state_tokens(state, state_mask, batch_size=batch_size, device=device, dtype=dtype)
        pieces = [future, action_features] if state_tokens is None else [state_tokens, future, action_features]
        return torch.cat(pieces, dim=1)

    def _run_blocks(
        self,
        hidden_states: torch.Tensor,
        vl_embs_list: list[torch.Tensor],
        timestep_emb: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if len(vl_embs_list) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} VLM layers, got {len(vl_embs_list)}")
        for idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                vl_embs_list[idx].to(hidden_states.dtype),
                timestep_emb=timestep_emb,
                encoder_attention_mask=encoder_attention_mask,
            )
        return hidden_states

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
        t_view = t.reshape(batch_size, 1, 1)
        noisy = (1.0 - t_view) * noise + t_view * actions
        target_velocity = actions - noise
        timestep_emb = self.timestep(t)
        hidden_states = self._build_sequence(noisy, timestep_emb, state, state_mask)
        hidden_states = self._run_blocks(hidden_states, vl_embs_list, timestep_emb, encoder_attention_mask)
        action_hidden = hidden_states[:, -horizon:]
        velocity = self.decoder(action_hidden)
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
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype
        actions = torch.randn(batch_size, horizon, self.action_dim, device=device, dtype=dtype, generator=generator)
        steps = max(int(num_inference_steps or self.num_inference_steps), 1)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((batch_size,), i / float(steps), device=device, dtype=dtype)
            timestep_emb = self.timestep(t)
            hidden_states = self._build_sequence(actions, timestep_emb, state, state_mask)
            hidden_states = self._run_blocks(hidden_states, vl_embs_list, timestep_emb, encoder_attention_mask)
            velocity = self.decoder(hidden_states[:, -horizon:])
            actions = actions + dt * velocity
        return actions

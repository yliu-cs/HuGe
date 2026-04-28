"""Mixture-of-Transformers Kappa model with flow matching action expert."""
from __future__ import annotations

import math
from typing import Optional
import torch
import torch.nn as nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by MLP, following last0/DiT."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_dim = freq_dim

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal_embedding(t, self.freq_dim)
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class VelocityHead(nn.Module):
    """MLP head for velocity prediction from action hidden states."""

    def __init__(self, hidden_size: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoTAttention(nn.Module):
    """Shared attention with independent Q/K/V/O projections for action tokens."""

    def __init__(self, original_attn: Qwen3VLTextAttention):
        super().__init__()
        self.original = original_attn
        config = original_attn.config
        head_dim = original_attn.head_dim

        self.q_proj_action = nn.Linear(
            config.hidden_size, config.num_attention_heads * head_dim, bias=config.attention_bias
        )
        self.k_proj_action = nn.Linear(
            config.hidden_size, config.num_key_value_heads * head_dim, bias=config.attention_bias
        )
        self.v_proj_action = nn.Linear(
            config.hidden_size, config.num_key_value_heads * head_dim, bias=config.attention_bias
        )
        self.o_proj_action = nn.Linear(
            config.num_attention_heads * head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm_action = Qwen3VLTextRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm_action = Qwen3VLTextRMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        n_latent: Optional[int] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # ``n_latent`` is the count of language/vision tokens; action tokens
        # are the contiguous tail. Caller guarantees this layout, so we slice
        # directly instead of fancy-indexing through arange tensors.
        attn = self.original
        input_shape = hidden_states.shape[:-1]
        hidden_shape_fn = lambda x: (*x.shape[:-1], -1, attn.head_dim)

        latent_h = hidden_states[:, :n_latent]
        action_h = hidden_states[:, n_latent:]

        q_latent = attn.q_norm(attn.q_proj(latent_h).view(hidden_shape_fn(latent_h)))
        k_latent = attn.k_norm(attn.k_proj(latent_h).view(hidden_shape_fn(latent_h)))
        v_latent = attn.v_proj(latent_h).view(hidden_shape_fn(latent_h))

        q_action = self.q_norm_action(self.q_proj_action(action_h).view(hidden_shape_fn(action_h)))
        k_action = self.k_norm_action(self.k_proj_action(action_h).view(hidden_shape_fn(action_h)))
        v_action = self.v_proj_action(action_h).view(hidden_shape_fn(action_h))

        query_states = torch.cat([q_latent, q_action], dim=1).transpose(1, 2)
        key_states = torch.cat([k_latent, k_action], dim=1).transpose(1, 2)
        value_states = torch.cat([v_latent, v_action], dim=1).transpose(1, 2)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
        # Force SDPA: the action expert relies on a 4D float attention mask
        # (bidirectional among action tokens, causal among text), which
        # flash-attn-2 does not support. The vision tower / non-MoT paths
        # still use whatever ``_attn_implementation`` the model was loaded
        # with (typically flash_attention_2).
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            "sdpa", eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else attn.attention_dropout,
            scaling=attn.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        latent_out = attn.o_proj(attn_output[:, :n_latent])
        action_out = self.o_proj_action(attn_output[:, n_latent:])
        return torch.cat([latent_out, action_out], dim=1), attn_weights


class MoTDecoderLayer(nn.Module):
    """Decoder layer with shared attention and independent MLP for action tokens."""

    def __init__(self, original_layer: Qwen3VLTextDecoderLayer):
        super().__init__()
        config = original_layer.self_attn.config
        self.hidden_size = config.hidden_size

        self.self_attn = MoTAttention(original_layer.self_attn)
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm

        self.mlp_action = Qwen3VLTextMLP(config)
        self.input_layernorm_action = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_action = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        n_latent: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        # latent | action are kept contiguous along dim=1, so we slice
        # directly. The previous ``empty_like + index_put_`` reorder was a
        # no-op (the indexes were ``arange``s) and burned an allocation per
        # layer × 28 layers per step.
        residual = hidden_states

        latent_h = hidden_states[:, :n_latent]
        action_h = hidden_states[:, n_latent:]
        normed = torch.cat([
            self.input_layernorm(latent_h),
            self.input_layernorm_action(action_h),
        ], dim=1)

        attn_out, _ = self.self_attn(
            hidden_states=normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            n_latent=n_latent,
            **kwargs,
        )
        hidden_states = residual + attn_out.to(hidden_states.dtype)

        residual = hidden_states
        latent_h = hidden_states[:, :n_latent]
        action_h = hidden_states[:, n_latent:]
        mlp_latent = self.mlp(self.post_attention_layernorm(latent_h)).to(hidden_states.dtype)
        mlp_action = self.mlp_action(self.post_attention_layernorm_action(action_h)).to(hidden_states.dtype)
        hidden_states = residual + torch.cat([mlp_latent, mlp_action], dim=1)

        return hidden_states


class MoTKappaModel(nn.Module):
    """Kappa MoT model with flow matching action expert.

    Input: image + text + timestep token + noisy SONIC tokens (no GT).
    Output: predicted velocity for denoising.
    """

    def __init__(self, base_model, sonic_dim: int = 66, action_loss_weight: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.sonic_dim = sonic_dim
        self.action_loss_weight = action_loss_weight

        hidden_size = base_model.config.hidden_size
        self.action_embedder = nn.Linear(sonic_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.velocity_head = VelocityHead(hidden_size, sonic_dim)
        self.norm_action = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)

        text_model = base_model.model.language_model
        new_layers = nn.ModuleList()
        for layer in text_model.layers:
            new_layers.append(MoTDecoderLayer(layer))
        text_model.layers = new_layers

    @property
    def config(self):
        return self.base_model.config

    @property
    def dtype(self):
        return self.base_model.dtype

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def named_parameters(self, *args, **kwargs):
        yield from super().named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        yield from super().parameters(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        noisy_sonic_tokens: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        velocity_target: Optional[torch.Tensor] = None,
        sonic_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        if noisy_sonic_tokens is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )

        qwen_model = self.base_model.model
        language_model = qwen_model.language_model

        inputs_embeds = language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_outputs = qwen_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = qwen_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        B, T_text, D = inputs_embeds.shape
        T_action = noisy_sonic_tokens.shape[1]

        # Timestep token: [B, 1, D]
        t_emb = self.t_embedder(timesteps).unsqueeze(1)
        # Noisy action embeddings: [B, T_action, D]
        action_embeds = self.action_embedder(noisy_sonic_tokens.to(inputs_embeds.dtype))
        # Concatenate: [text+image, timestep, noisy_actions]
        all_embeds = torch.cat([inputs_embeds, t_emb, action_embeds], dim=1)

        T_total = T_text + 1 + T_action
        # Timestep token + action tokens share the action expert path; they
        # are the contiguous tail starting at index ``T_text``.
        n_latent = T_text

        full_attn_mask = torch.zeros(B, 1, T_total, T_total, device=input_ids.device, dtype=all_embeds.dtype)
        causal_mask = torch.triu(torch.ones(T_text, T_text, device=input_ids.device, dtype=torch.bool), diagonal=1)
        full_attn_mask[:, 0, :T_text, :T_text].masked_fill_(causal_mask, float("-inf"))
        if attention_mask is not None:
            pad_positions = attention_mask == 0
            full_attn_mask[:, 0, :, :T_text].masked_fill_(pad_positions.unsqueeze(1), float("-inf"))

        position_ids = torch.arange(T_total, device=input_ids.device).unsqueeze(0).expand(B, -1)
        position_embeddings = language_model.rotary_emb(all_embeds, position_ids)

        hidden_states = all_embeds
        for layer in language_model.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=full_attn_mask,
                n_latent=n_latent,
            )

        # Extract action hidden states (skip timestep token at index T_text)
        action_hidden = self.norm_action(hidden_states[:, T_text + 1:])
        velocity_pred = self.velocity_head(action_hidden)

        if velocity_target is not None:
            target = velocity_target.to(velocity_pred.dtype)
            if sonic_mask is not None:
                # `sonic_mask` may be [B, T] (time-only, HE) or [B, T, D]
                # (per-dim, e.g. EgoDex masking out unused hand-binary dims).
                if sonic_mask.dim() == 2:
                    mask = sonic_mask.unsqueeze(-1).to(velocity_pred.dtype)
                    denom = (mask.sum() * self.sonic_dim).clamp(min=1)
                else:
                    mask = sonic_mask.to(velocity_pred.dtype)
                    denom = mask.sum().clamp(min=1)
                mse = nn.functional.mse_loss(velocity_pred, target, reduction="none")
                action_loss = (mse * mask).sum() / denom
            else:
                action_loss = nn.functional.mse_loss(velocity_pred, target)
            return CausalLMOutputWithPast(loss=self.action_loss_weight * action_loss)

        # Inference path: no target, return predicted velocity via `logits` slot.
        return CausalLMOutputWithPast(
            loss=torch.zeros((), device=velocity_pred.device),
            logits=velocity_pred,
        )

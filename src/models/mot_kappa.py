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
        kv_cache_out: Optional[dict] = None,
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

        # Latent-prefix KV caching: post-RoPE K and V for the latent positions
        # are cross-step invariant under the new mask (which forbids latent →
        # action attention), so callers in the inference loop can populate the
        # cache once and reuse it. ``v_latent`` would also be invariant pre-
        # RoPE, but we keep the slicing symmetric here.
        if kv_cache_out is not None:
            kv_cache_out["k"] = key_states[:, :, :n_latent, :].detach().contiguous()
            kv_cache_out["v"] = value_states[:, :, :n_latent, :].detach().contiguous()

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

    def forward_cached(
        self,
        action_h: torch.Tensor,
        position_embeddings_action: tuple[torch.Tensor, torch.Tensor],
        attention_mask_action: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
        """Action-only attention reusing cached latent K/V.

        ``action_h`` is the timestep+action tail (length ``T_action+1``).
        ``position_embeddings_action`` is the rotary (cos, sin) sliced to that
        tail. ``attention_mask_action`` is the row-sliced mask of shape
        ``[B, 1, T_action+1, T_total]``. ``kv_cache`` holds post-RoPE
        ``k``/``v`` of shape ``[B, num_kv_heads, T_text, head_dim]`` produced
        by a prior full forward.

        Returns ``[B, T_action+1, hidden_size]`` after ``o_proj_action``.
        """
        attn = self.original
        head_dim = attn.head_dim
        B, T_tail, _ = action_h.shape
        hidden_shape_fn = lambda x: (*x.shape[:-1], -1, head_dim)

        q_action = self.q_norm_action(
            self.q_proj_action(action_h).view(hidden_shape_fn(action_h))
        ).transpose(1, 2)
        k_action = self.k_norm_action(
            self.k_proj_action(action_h).view(hidden_shape_fn(action_h))
        ).transpose(1, 2)
        v_action = self.v_proj_action(action_h).view(hidden_shape_fn(action_h)).transpose(1, 2)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        cos_a, sin_a = position_embeddings_action
        q_action, k_action = apply_rotary_pos_emb(q_action, k_action, cos_a, sin_a)

        # Stitch cached latent K/V (already post-RoPE) with fresh action K/V.
        key_states = torch.cat([kv_cache["k"], k_action], dim=2)
        value_states = torch.cat([kv_cache["v"], v_action], dim=2)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            "sdpa", eager_attention_forward
        )

        attn_output, _ = attention_interface(
            attn,
            q_action,
            key_states,
            value_states,
            attention_mask_action,
            dropout=0.0,
            scaling=attn.scaling,
        )
        attn_output = attn_output.reshape(B, T_tail, -1).contiguous()
        return self.o_proj_action(attn_output)


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
        kv_cache_out: Optional[dict] = None,
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
            kv_cache_out=kv_cache_out,
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

    def forward_cached(
        self,
        action_h: torch.Tensor,
        position_embeddings_action: tuple[torch.Tensor, torch.Tensor],
        attention_mask_action: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
        """Action-only layer step. Skips the latent path entirely — its
        attention/MLP outputs are unchanged across denoising steps under the
        latent→action mask, so the caller has cached the latent K/V from a
        prior full forward and discarded the latent hidden states.
        """
        residual = action_h
        normed = self.input_layernorm_action(action_h)
        attn_out = self.self_attn.forward_cached(
            action_h=normed,
            position_embeddings_action=position_embeddings_action,
            attention_mask_action=attention_mask_action,
            kv_cache=kv_cache,
        )
        action_h = residual + attn_out.to(action_h.dtype)

        residual = action_h
        mlp_out = self.mlp_action(self.post_attention_layernorm_action(action_h)).to(action_h.dtype)
        return residual + mlp_out


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
        # Block latent → (timestep + action): the latent prefix's hidden states
        # depend only on (image, text) and are constant across denoising steps,
        # which lets inference cache K_latent/V_latent once per observation
        # (see ``encode_latent`` / ``denoise_step``). π₀ / SmolVLA use the same
        # one-way attention between the VLM prefix and the action expert.
        full_attn_mask[:, 0, :T_text, T_text:] = float("-inf")
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

    @torch.no_grad()
    def encode_latent(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        T_action: int = 50,
    ) -> dict:
        """Run the latent prefix once and return a cache for ``denoise_step``.

        Computes inputs_embeds (with image fusion), the full attention mask,
        rotary position embeddings, and per-layer ``(K_latent, V_latent)``
        post-RoPE — every quantity that the denoising loop would otherwise
        recompute despite being constant across steps.

        Run with a dummy zero tail in the action expert path: the latent →
        action mask block ensures latent hidden states (and thus K_latent /
        V_latent) are independent of timestep / noisy_sonic_tokens, so the
        zeros do not affect what is cached. The action segment of the encode
        pass is computed but discarded.
        """
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
        T_total = T_text + 1 + T_action

        # Dummy tail (timestep + action positions). The latent→action mask
        # block makes latent_h independent of these zeros, so K_latent/V_latent
        # are correct regardless of what we put here.
        tail = torch.zeros(B, 1 + T_action, D, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        all_embeds = torch.cat([inputs_embeds, tail], dim=1)

        full_attn_mask = torch.zeros(B, 1, T_total, T_total, device=input_ids.device, dtype=all_embeds.dtype)
        causal_mask = torch.triu(
            torch.ones(T_text, T_text, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        full_attn_mask[:, 0, :T_text, :T_text].masked_fill_(causal_mask, float("-inf"))
        full_attn_mask[:, 0, :T_text, T_text:] = float("-inf")
        if attention_mask is not None:
            pad_positions = attention_mask == 0
            full_attn_mask[:, 0, :, :T_text].masked_fill_(pad_positions.unsqueeze(1), float("-inf"))

        position_ids = torch.arange(T_total, device=input_ids.device).unsqueeze(0).expand(B, -1)
        position_embeddings = language_model.rotary_emb(all_embeds, position_ids)

        layer_caches: list[dict] = []
        hidden_states = all_embeds
        for layer in language_model.layers:
            cache: dict = {}
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=full_attn_mask,
                n_latent=T_text,
                kv_cache_out=cache,
            )
            layer_caches.append(cache)

        # Pre-slice everything ``denoise_step`` needs for the action tail so
        # the per-step path does no Python-side slicing. Cos/sin on the action
        # rows; mask rows are the action queries.
        cos_full, sin_full = position_embeddings
        cos_action = cos_full[:, T_text:, :].contiguous()
        sin_action = sin_full[:, T_text:, :].contiguous()
        attention_mask_action = full_attn_mask[:, :, T_text:, :].contiguous()

        return {
            "T_text": T_text,
            "T_action": T_action,
            "T_total": T_total,
            "B": B,
            "device": input_ids.device,
            "dtype": inputs_embeds.dtype,
            "position_embeddings_action": (cos_action, sin_action),
            "attention_mask_action": attention_mask_action,
            "layer_caches": layer_caches,
        }

    @torch.no_grad()
    def denoise_step(
        self,
        cache: dict,
        noisy_sonic_tokens: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Single flow-matching forward reusing the cached latent prefix.

        Only the action expert (timestep token + ``T_action`` action tokens) is
        re-evaluated each call; the language/vision prefix is read from the
        cached per-layer K/V. Returns velocity prediction with shape
        ``[B, T_action, sonic_dim]``.
        """
        language_model = self.base_model.model.language_model

        t_emb = self.t_embedder(timesteps).unsqueeze(1)
        action_embeds = self.action_embedder(noisy_sonic_tokens.to(cache["dtype"]))
        action_h = torch.cat([t_emb, action_embeds], dim=1)

        for layer, kv_cache in zip(language_model.layers, cache["layer_caches"]):
            action_h = layer.forward_cached(
                action_h=action_h,
                position_embeddings_action=cache["position_embeddings_action"],
                attention_mask_action=cache["attention_mask_action"],
                kv_cache=kv_cache,
            )

        action_hidden = self.norm_action(action_h[:, 1:])  # drop timestep token
        return self.velocity_head(action_hidden)

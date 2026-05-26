"""Mixture-of-Transformers Kappa model with SONIC flow-matching action expert.

Two-expert layout: latent (text+vision[+proprio]) | sonic-action. Sequence
layout when proprio is on:

    ``[image+text (T_text) | proprio (T_p=1) | t_emb_s + sonic (T_s)]``

When proprio is off: ``[image+text | t_emb_s (1) | sonic (T_s)]``.

Proprio joins the **latent prefix** (Qwen base expert) so its K/V are cached
once and reused during denoise. The 64-D proprio is projected via the
top-level ``proprio_embedder`` Linear.
"""
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
    """Sinusoidal timestep embedding followed by MLP."""

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
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        flat = t.reshape(-1).float()
        args = flat[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb.view(*t.shape, dim)

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


def _attn_proj(in_features: int, out_features: int, bias: bool) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=bias)


class MoTAttention(nn.Module):
    """Shared joint attention with one extra SONIC-expert path.

    ``forward`` accepts ``hidden_states`` already laid out as
    ``[latent (T_text + T_p) | sonic (T_s)]``. The caller passes the segment
    boundary as ``n_latent``; tokens before ``n_latent`` (text + image +
    proprio) go through the base Qwen projections, and tokens at
    ``[n_latent:]`` go through the ``*_action`` projections.
    """

    def __init__(self, original_attn: Qwen3VLTextAttention):
        super().__init__()
        self.original = original_attn
        config = original_attn.config
        head_dim = original_attn.head_dim

        self.q_proj_action = _attn_proj(
            config.hidden_size, config.num_attention_heads * head_dim, config.attention_bias,
        )
        self.k_proj_action = _attn_proj(
            config.hidden_size, config.num_key_value_heads * head_dim, config.attention_bias,
        )
        self.v_proj_action = _attn_proj(
            config.hidden_size, config.num_key_value_heads * head_dim, config.attention_bias,
        )
        self.o_proj_action = _attn_proj(
            config.num_attention_heads * head_dim, config.hidden_size, config.attention_bias,
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
        attn = self.original
        input_shape = hidden_states.shape[:-1]
        head_dim = attn.head_dim

        def _shape(x: torch.Tensor) -> tuple[int, ...]:
            return (*x.shape[:-1], -1, head_dim)

        latent_h = hidden_states[:, :n_latent]
        sonic_h = hidden_states[:, n_latent:]

        q_l = attn.q_norm(attn.q_proj(latent_h).view(_shape(latent_h)))
        k_l = attn.k_norm(attn.k_proj(latent_h).view(_shape(latent_h)))
        v_l = attn.v_proj(latent_h).view(_shape(latent_h))

        q_s = self.q_norm_action(self.q_proj_action(sonic_h).view(_shape(sonic_h)))
        k_s = self.k_norm_action(self.k_proj_action(sonic_h).view(_shape(sonic_h)))
        v_s = self.v_proj_action(sonic_h).view(_shape(sonic_h))

        query_states = torch.cat([q_l, q_s], dim=1).transpose(1, 2)
        key_states = torch.cat([k_l, k_s], dim=1).transpose(1, 2)
        value_states = torch.cat([v_l, v_s], dim=1).transpose(1, 2)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Latent-prefix KV caching: latent → tail is always blocked, so
        # K_latent / V_latent are constant across denoising steps. Cache them
        # once and reuse during ``denoise_step``.
        if kv_cache_out is not None:
            kv_cache_out["k"] = key_states[:, :, :n_latent, :].detach().contiguous()
            kv_cache_out["v"] = value_states[:, :, :n_latent, :].detach().contiguous()

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            "sdpa", eager_attention_forward,
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
        sonic_out = self.o_proj_action(attn_output[:, n_latent:])
        return torch.cat([latent_out, sonic_out], dim=1), attn_weights

    def forward_cached(
        self,
        action_h: torch.Tensor,
        position_embeddings_action: tuple[torch.Tensor, torch.Tensor],
        attention_mask_action: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
        """Action-only attention reusing cached latent K/V.

        ``action_h`` is the timestep+sonic tail.
        """
        attn = self.original
        head_dim = attn.head_dim
        B, T_tail, _ = action_h.shape

        def _shape(x: torch.Tensor) -> tuple[int, ...]:
            return (*x.shape[:-1], -1, head_dim)

        q_action = self.q_norm_action(self.q_proj_action(action_h).view(_shape(action_h))).transpose(1, 2)
        k_action = self.k_norm_action(self.k_proj_action(action_h).view(_shape(action_h))).transpose(1, 2)
        v_action = self.v_proj_action(action_h).view(_shape(action_h)).transpose(1, 2)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        cos_a, sin_a = position_embeddings_action
        q_action, k_action = apply_rotary_pos_emb(q_action, k_action, cos_a, sin_a)

        key_states = torch.cat([kv_cache["k"], k_action], dim=2)
        value_states = torch.cat([kv_cache["v"], v_action], dim=2)

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_vl.modeling_qwen3_vl import eager_attention_forward
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            "sdpa", eager_attention_forward,
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
    """Decoder layer with shared attention and per-expert MLP/norm副本."""

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
        self.post_attention_layernorm_action = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        n_latent: Optional[int] = None,
        kv_cache_out: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        latent_h = hidden_states[:, :n_latent]
        sonic_h = hidden_states[:, n_latent:]
        normed = torch.cat(
            [self.input_layernorm(latent_h), self.input_layernorm_action(sonic_h)],
            dim=1,
        )

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
        sonic_h = hidden_states[:, n_latent:]
        mlp_l = self.mlp(self.post_attention_layernorm(latent_h)).to(hidden_states.dtype)
        mlp_s = self.mlp_action(self.post_attention_layernorm_action(sonic_h)).to(hidden_states.dtype)
        return residual + torch.cat([mlp_l, mlp_s], dim=1)

    def forward_cached(
        self,
        action_h: torch.Tensor,
        position_embeddings_action: tuple[torch.Tensor, torch.Tensor],
        attention_mask_action: torch.Tensor,
        kv_cache: dict,
    ) -> torch.Tensor:
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
        mlp_s = self.mlp_action(self.post_attention_layernorm_action(action_h)).to(action_h.dtype)
        return residual + mlp_s


def _causal_mask(T: int, device, dtype) -> torch.Tensor:
    return torch.triu(
        torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1,
    )


class MoTKappaModel(nn.Module):
    """Kappa MoT model with SONIC flow-matching action expert.

    See module docstring for sequence layouts. ``use_proprio`` adds a single
    64-D proprio token at the end of the latent prefix (between text/image and
    the SONIC segment).
    """

    def __init__(
        self,
        base_model,
        sonic_dim: int = 66,
        action_loss_weight: float = 1.0,
        use_proprio: bool = False,
        proprio_dim: int = 64,
    ):
        super().__init__()
        self.base_model = base_model
        self.sonic_dim = sonic_dim
        self.action_loss_weight = action_loss_weight
        self.use_proprio = use_proprio
        self.proprio_dim = proprio_dim

        hidden_size = base_model.config.hidden_size
        self.action_embedder = nn.Linear(sonic_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.velocity_head = VelocityHead(hidden_size, sonic_dim)
        self.norm_action = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)

        if self.use_proprio:
            self.proprio_embedder = nn.Linear(proprio_dim, hidden_size)
            nn.init.normal_(self.proprio_embedder.weight, std=0.02)
            nn.init.zeros_(self.proprio_embedder.bias)

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

    # -------------------------------------------------------------------
    # Mask construction
    # -------------------------------------------------------------------

    def _build_mask(
        self,
        B: int,
        T_text: int,
        T_s: int,
        T_p: int,
        device,
        dtype,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """3-section attention mask covering ``[text | proprio | t_emb_s + sonic]``.

        Proprio sits inside the latent prefix so its K/V are cached together
        with text/image; the SONIC expert attends to it via the per-layer
        attention.

        Constraints:

        * ``text → text`` causal; ``text → proprio/sonic`` blocked
          (KV cache invariant + protect Qwen text representations from
          proprio feedback).
        * ``proprio → text`` full; ``proprio → proprio`` self;
          ``proprio → sonic`` blocked (KV cache invariant).
        * ``sonic → text + proprio`` full; ``sonic → sonic`` bidirectional.
        """
        T_total = T_text + T_p + T_s
        mask = torch.zeros(B, 1, T_total, T_total, device=device, dtype=dtype)

        n_text_end = T_text
        n_proprio_end = n_text_end + T_p          # == n_latent

        mask[:, 0, :n_text_end, :n_text_end].masked_fill_(
            _causal_mask(T_text, device, dtype), float("-inf"),
        )
        mask[:, 0, :n_text_end, n_text_end:] = float("-inf")

        if T_p > 0:
            mask[:, 0, n_text_end:n_proprio_end, n_proprio_end:] = float("-inf")

        if attention_mask is not None:
            pad_positions = attention_mask == 0
            mask[:, 0, :, :T_text].masked_fill_(pad_positions.unsqueeze(1), float("-inf"))

        return mask

    # -------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------

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
        proprio: Optional[torch.Tensor] = None,
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
                pixel_values, image_grid_thw, return_dict=True,
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype,
            )
            image_mask, _ = qwen_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        B, T_text, D = inputs_embeds.shape

        # SONIC path: timestep token + noisy sonic tokens. ``T_s`` includes
        # the leading t_emb_s token.
        t_emb = self.t_embedder(timesteps).reshape(B, 1, D)
        sonic_embeds = self.action_embedder(noisy_sonic_tokens.to(inputs_embeds.dtype))
        sonic_segment = torch.cat([t_emb, sonic_embeds], dim=1)
        T_s = sonic_segment.shape[1]

        T_p = 0
        proprio_segment: Optional[torch.Tensor] = None

        if self.use_proprio and proprio is not None:
            proprio_segment = self.proprio_embedder(
                proprio.to(inputs_embeds.dtype)
            ).reshape(B, 1, D)
            T_p = 1

        # Layout: [text | proprio? | sonic].
        parts = [inputs_embeds]
        if T_p > 0:
            parts.append(proprio_segment)
        parts.append(sonic_segment)
        all_embeds = torch.cat(parts, dim=1)

        T_total = T_text + T_p + T_s
        n_latent = T_text + T_p

        full_attn_mask = self._build_mask(
            B=B, T_text=T_text, T_s=T_s, T_p=T_p,
            device=input_ids.device, dtype=all_embeds.dtype,
            attention_mask=attention_mask,
        )

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

        # SONIC velocity prediction (skip leading t_emb_s within sonic seg).
        sonic_hidden = self.norm_action(hidden_states[:, n_latent + 1:])
        velocity_pred = self.velocity_head(sonic_hidden)

        loss_total = torch.zeros((), device=velocity_pred.device, dtype=torch.float32)
        sonic_loss = None

        if velocity_target is not None:
            target = velocity_target.to(velocity_pred.dtype)
            if sonic_mask is not None:
                if sonic_mask.dim() == 2:
                    mask_s = sonic_mask.unsqueeze(-1).to(velocity_pred.dtype)
                    denom = (mask_s.sum() * self.sonic_dim).clamp(min=1)
                else:
                    mask_s = sonic_mask.to(velocity_pred.dtype)
                    denom = mask_s.sum().clamp(min=1)
                mse = nn.functional.mse_loss(velocity_pred, target, reduction="none")
                sonic_loss = (mse * mask_s).sum() / denom
            else:
                sonic_loss = nn.functional.mse_loss(velocity_pred, target)
            loss_total = loss_total + self.action_loss_weight * sonic_loss

        if velocity_target is not None:
            output = CausalLMOutputWithPast(loss=loss_total)
            if sonic_loss is not None:
                output.sonic_loss = sonic_loss.detach()
            return output

        return CausalLMOutputWithPast(
            loss=torch.zeros((), device=velocity_pred.device),
            logits=velocity_pred,
        )

    # -------------------------------------------------------------------
    # Inference: cached latent prefix + per-step denoise
    # -------------------------------------------------------------------

    @torch.no_grad()
    def encode_latent(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        T_action: int = 50,
        T_proprio: int = 0,
        proprio: Optional[torch.Tensor] = None,
    ) -> dict:
        qwen_model = self.base_model.model
        language_model = qwen_model.language_model

        inputs_embeds = language_model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_outputs = qwen_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True,
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype,
            )
            image_mask, _ = qwen_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        B, T_text, D = inputs_embeds.shape
        T_s = 1 + T_action  # t_emb_s token + sonic chunk
        T_p = T_proprio
        T_total = T_text + T_p + T_s

        # Embed proprio into the latent prefix so its K/V are baked into the
        # KV cache once and reused for every denoise step.
        if T_p > 0:
            assert proprio is not None, "T_proprio>0 but proprio tensor not provided"
            proprio_embed = self.proprio_embedder(
                proprio.to(inputs_embeds.dtype)
            ).reshape(B, T_p, D)
            prefix_embeds = torch.cat([inputs_embeds, proprio_embed], dim=1)
        else:
            prefix_embeds = inputs_embeds

        n_latent = T_text + T_p

        tail = torch.zeros(
            B, T_s, D,
            device=inputs_embeds.device, dtype=inputs_embeds.dtype,
        )
        all_embeds = torch.cat([prefix_embeds, tail], dim=1)

        full_attn_mask = self._build_mask(
            B=B, T_text=T_text, T_s=T_s, T_p=T_p,
            device=input_ids.device, dtype=all_embeds.dtype,
            attention_mask=attention_mask,
        )

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
                n_latent=n_latent,
                kv_cache_out=cache,
            )
            layer_caches.append(cache)

        cos_full, sin_full = position_embeddings
        cos_action = cos_full[:, n_latent:, :].contiguous()
        sin_action = sin_full[:, n_latent:, :].contiguous()
        attention_mask_action = full_attn_mask[:, :, n_latent:, :].contiguous()

        return {
            "T_text": T_text,
            "T_s": T_s,
            "T_p": T_p,
            "T_action": T_action,
            "T_total": T_total,
            "n_latent": n_latent,
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
        """SONIC denoise step."""
        language_model = self.base_model.model.language_model

        t_emb = self.t_embedder(timesteps).reshape(cache["B"], 1, -1)
        action_embeds = self.action_embedder(noisy_sonic_tokens.to(cache["dtype"]))
        action_h = torch.cat([t_emb, action_embeds], dim=1)

        for layer, kv_cache in zip(language_model.layers, cache["layer_caches"]):
            action_h = layer.forward_cached(
                action_h=action_h,
                position_embeddings_action=cache["position_embeddings_action"],
                attention_mask_action=cache["attention_mask_action"],
                kv_cache=kv_cache,
            )

        action_hidden = self.norm_action(action_h[:, 1:])
        return self.velocity_head(action_hidden)

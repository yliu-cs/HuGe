"""Mixture-of-Transformers Kappa model with flow matching action expert.

Two expert configurations are supported:

* **2-expert (legacy / pretrain)** — latent (text+vision) | sonic-action.
  Sequence layout: ``[image+text | t_emb (1) | sonic (T_s)]``.
* **3-expert (post-train)** — latent (text+vision[+proprio]) | sonic-action | g1.
  Sequence layout when proprio is on:
    ``[image+text (T_text) | proprio (T_p=1) | t_emb_s + sonic (T_s) | g1 (T_g)]``
  When proprio is off: ``[image+text | t_emb_s (1) | sonic (T_s) | g1 (T_g)]``.

Proprio joins the **latent prefix** (Qwen base expert), so both SONIC and G1
attend to it via the per-layer attention. The G1 expert adds per-layer
``*_g1`` attention/MLP/norm副本 plus top-level ``action_embedder_g1`` /
``t_embedder_g1`` / ``norm_action_g1`` / ``velocity_head_g1``. The 64-D
proprio is projected once via ``proprio_embedder`` (a top-level Linear) and
its K/V are cached together with text/image into the prefix KV cache.

Real-Time Chunking (RTC) is supported by giving the G1 timestep a per-token
2-D shape and substituting clean prefix actions in the noisy input.
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
    """Sinusoidal timestep embedding followed by MLP.

    Accepts ``t`` with shape ``(B,)`` (one timestep per sample, the legacy /
    SONIC path) or ``(B, T)`` (per-token timesteps for RTC).
    """

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
    """Shared joint attention with one or two extra expert paths.

    ``forward`` accepts ``hidden_states`` already laid out as
    ``[latent (T_text + T_p) | sonic (T_s) | g1 (T_g)]``. The caller passes
    the segment boundaries as ``n_latent`` and ``n_sonic_end``; tokens before
    ``n_latent`` (text + image + proprio) go through the base Qwen
    projections, and tokens at ``[n_sonic_end:]`` go through the ``*_g1``
    projections. When ``n_sonic_end == n_total`` the legacy 2-expert layout
    is used (no G1 path constructed at runtime).
    """

    def __init__(self, original_attn: Qwen3VLTextAttention, has_g1: bool = False):
        super().__init__()
        self.original = original_attn
        config = original_attn.config
        head_dim = original_attn.head_dim
        self.has_g1 = has_g1

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

        if has_g1:
            self.q_proj_g1 = _attn_proj(
                config.hidden_size, config.num_attention_heads * head_dim, config.attention_bias,
            )
            self.k_proj_g1 = _attn_proj(
                config.hidden_size, config.num_key_value_heads * head_dim, config.attention_bias,
            )
            self.v_proj_g1 = _attn_proj(
                config.hidden_size, config.num_key_value_heads * head_dim, config.attention_bias,
            )
            self.o_proj_g1 = _attn_proj(
                config.num_attention_heads * head_dim, config.hidden_size, config.attention_bias,
            )
            self.q_norm_g1 = Qwen3VLTextRMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm_g1 = Qwen3VLTextRMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        n_latent: Optional[int] = None,
        n_sonic_end: Optional[int] = None,
        kv_cache_out: Optional[dict] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn = self.original
        input_shape = hidden_states.shape[:-1]
        T_total = hidden_states.shape[1]
        head_dim = attn.head_dim

        if n_sonic_end is None or n_sonic_end == T_total:
            # 2-expert (latent | sonic). Backwards compatible with pre-train.
            n_sonic_end = T_total
            has_g1_runtime = False
        else:
            has_g1_runtime = True

        def _shape(x: torch.Tensor) -> tuple[int, ...]:
            return (*x.shape[:-1], -1, head_dim)

        latent_h = hidden_states[:, :n_latent]
        sonic_h = hidden_states[:, n_latent:n_sonic_end]

        q_l = attn.q_norm(attn.q_proj(latent_h).view(_shape(latent_h)))
        k_l = attn.k_norm(attn.k_proj(latent_h).view(_shape(latent_h)))
        v_l = attn.v_proj(latent_h).view(_shape(latent_h))

        q_s = self.q_norm_action(self.q_proj_action(sonic_h).view(_shape(sonic_h)))
        k_s = self.k_norm_action(self.k_proj_action(sonic_h).view(_shape(sonic_h)))
        v_s = self.v_proj_action(sonic_h).view(_shape(sonic_h))

        if has_g1_runtime:
            g1_h = hidden_states[:, n_sonic_end:]
            q_g = self.q_norm_g1(self.q_proj_g1(g1_h).view(_shape(g1_h)))
            k_g = self.k_norm_g1(self.k_proj_g1(g1_h).view(_shape(g1_h)))
            v_g = self.v_proj_g1(g1_h).view(_shape(g1_h))
            query_states = torch.cat([q_l, q_s, q_g], dim=1).transpose(1, 2)
            key_states = torch.cat([k_l, k_s, k_g], dim=1).transpose(1, 2)
            value_states = torch.cat([v_l, v_s, v_g], dim=1).transpose(1, 2)
        else:
            query_states = torch.cat([q_l, q_s], dim=1).transpose(1, 2)
            key_states = torch.cat([k_l, k_s], dim=1).transpose(1, 2)
            value_states = torch.cat([v_l, v_s], dim=1).transpose(1, 2)

        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Latent-prefix KV caching: latent → tail is always blocked, so
        # K_latent / V_latent are constant across denoising steps. Cache them
        # once and reuse during ``denoise_step`` / ``denoise_step_rtc``.
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
        sonic_out = self.o_proj_action(attn_output[:, n_latent:n_sonic_end])
        if has_g1_runtime:
            g1_out = self.o_proj_g1(attn_output[:, n_sonic_end:])
            return torch.cat([latent_out, sonic_out, g1_out], dim=1), attn_weights
        return torch.cat([latent_out, sonic_out], dim=1), attn_weights

    def forward_cached(
        self,
        action_h: torch.Tensor,
        position_embeddings_action: tuple[torch.Tensor, torch.Tensor],
        attention_mask_action: torch.Tensor,
        kv_cache: dict,
        n_sonic_in_tail: Optional[int] = None,
    ) -> torch.Tensor:
        """Action-only attention reusing cached latent K/V.

        ``action_h`` is the timestep+sonic[+proprio+g1] tail. When
        ``n_sonic_in_tail`` is given, tokens at index ``[n_sonic_in_tail:]``
        in the tail go through the ``*_g1`` projections.
        """
        attn = self.original
        head_dim = attn.head_dim
        B, T_tail, _ = action_h.shape

        def _shape(x: torch.Tensor) -> tuple[int, ...]:
            return (*x.shape[:-1], -1, head_dim)

        if n_sonic_in_tail is None or n_sonic_in_tail == T_tail:
            n_sonic_in_tail = T_tail
            has_g1_runtime = False
        else:
            has_g1_runtime = True

        sonic_h = action_h[:, :n_sonic_in_tail]
        q_s = self.q_norm_action(self.q_proj_action(sonic_h).view(_shape(sonic_h))).transpose(1, 2)
        k_s = self.k_norm_action(self.k_proj_action(sonic_h).view(_shape(sonic_h))).transpose(1, 2)
        v_s = self.v_proj_action(sonic_h).view(_shape(sonic_h)).transpose(1, 2)

        if has_g1_runtime:
            g1_h = action_h[:, n_sonic_in_tail:]
            q_g = self.q_norm_g1(self.q_proj_g1(g1_h).view(_shape(g1_h))).transpose(1, 2)
            k_g = self.k_norm_g1(self.k_proj_g1(g1_h).view(_shape(g1_h))).transpose(1, 2)
            v_g = self.v_proj_g1(g1_h).view(_shape(g1_h)).transpose(1, 2)
            q_action = torch.cat([q_s, q_g], dim=2)
            k_action = torch.cat([k_s, k_g], dim=2)
            v_action = torch.cat([v_s, v_g], dim=2)
        else:
            q_action, k_action, v_action = q_s, k_s, v_s

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
        sonic_out = self.o_proj_action(attn_output[:, :n_sonic_in_tail])
        if has_g1_runtime:
            g1_out = self.o_proj_g1(attn_output[:, n_sonic_in_tail:])
            return torch.cat([sonic_out, g1_out], dim=1)
        return sonic_out


class MoTDecoderLayer(nn.Module):
    """Decoder layer with shared attention and per-expert MLP/norm副本."""

    def __init__(self, original_layer: Qwen3VLTextDecoderLayer, has_g1: bool = False):
        super().__init__()
        config = original_layer.self_attn.config
        self.hidden_size = config.hidden_size
        self.has_g1 = has_g1

        self.self_attn = MoTAttention(original_layer.self_attn, has_g1=has_g1)
        self.mlp = original_layer.mlp
        self.input_layernorm = original_layer.input_layernorm
        self.post_attention_layernorm = original_layer.post_attention_layernorm

        self.mlp_action = Qwen3VLTextMLP(config)
        self.input_layernorm_action = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_action = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
        )

        if has_g1:
            self.mlp_g1 = Qwen3VLTextMLP(config)
            self.input_layernorm_g1 = Qwen3VLTextRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm_g1 = Qwen3VLTextRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        n_latent: Optional[int] = None,
        n_sonic_end: Optional[int] = None,
        kv_cache_out: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        T_total = hidden_states.shape[1]
        if n_sonic_end is None:
            n_sonic_end = T_total
        has_g1_runtime = self.has_g1 and (n_sonic_end < T_total)

        residual = hidden_states
        latent_h = hidden_states[:, :n_latent]
        sonic_h = hidden_states[:, n_latent:n_sonic_end]
        normed_parts = [
            self.input_layernorm(latent_h),
            self.input_layernorm_action(sonic_h),
        ]
        if has_g1_runtime:
            g1_h = hidden_states[:, n_sonic_end:]
            normed_parts.append(self.input_layernorm_g1(g1_h))
        normed = torch.cat(normed_parts, dim=1)

        attn_out, _ = self.self_attn(
            hidden_states=normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            n_latent=n_latent,
            n_sonic_end=n_sonic_end,
            kv_cache_out=kv_cache_out,
            **kwargs,
        )
        hidden_states = residual + attn_out.to(hidden_states.dtype)

        residual = hidden_states
        latent_h = hidden_states[:, :n_latent]
        sonic_h = hidden_states[:, n_latent:n_sonic_end]
        mlp_l = self.mlp(self.post_attention_layernorm(latent_h)).to(hidden_states.dtype)
        mlp_s = self.mlp_action(self.post_attention_layernorm_action(sonic_h)).to(hidden_states.dtype)
        if has_g1_runtime:
            g1_h = hidden_states[:, n_sonic_end:]
            mlp_g = self.mlp_g1(self.post_attention_layernorm_g1(g1_h)).to(hidden_states.dtype)
            hidden_states = residual + torch.cat([mlp_l, mlp_s, mlp_g], dim=1)
        else:
            hidden_states = residual + torch.cat([mlp_l, mlp_s], dim=1)
        return hidden_states

    def forward_cached(
        self,
        action_h: torch.Tensor,
        position_embeddings_action: tuple[torch.Tensor, torch.Tensor],
        attention_mask_action: torch.Tensor,
        kv_cache: dict,
        n_sonic_in_tail: Optional[int] = None,
    ) -> torch.Tensor:
        T_tail = action_h.shape[1]
        if n_sonic_in_tail is None:
            n_sonic_in_tail = T_tail
        has_g1_runtime = self.has_g1 and (n_sonic_in_tail < T_tail)

        residual = action_h
        sonic_h = action_h[:, :n_sonic_in_tail]
        normed_parts = [self.input_layernorm_action(sonic_h)]
        if has_g1_runtime:
            g1_h = action_h[:, n_sonic_in_tail:]
            normed_parts.append(self.input_layernorm_g1(g1_h))
        normed = torch.cat(normed_parts, dim=1)

        attn_out = self.self_attn.forward_cached(
            action_h=normed,
            position_embeddings_action=position_embeddings_action,
            attention_mask_action=attention_mask_action,
            kv_cache=kv_cache,
            n_sonic_in_tail=n_sonic_in_tail,
        )
        action_h = residual + attn_out.to(action_h.dtype)

        residual = action_h
        sonic_h = action_h[:, :n_sonic_in_tail]
        mlp_s = self.mlp_action(self.post_attention_layernorm_action(sonic_h)).to(action_h.dtype)
        if has_g1_runtime:
            g1_h = action_h[:, n_sonic_in_tail:]
            mlp_g = self.mlp_g1(self.post_attention_layernorm_g1(g1_h)).to(action_h.dtype)
            return residual + torch.cat([mlp_s, mlp_g], dim=1)
        return residual + mlp_s


def _causal_mask(T: int, device, dtype) -> torch.Tensor:
    return torch.triu(
        torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1,
    )


class MoTKappaModel(nn.Module):
    """Kappa MoT model with flow matching action expert(s).

    See module docstring for sequence layouts. ``use_g1_expert`` activates the
    third expert; ``use_proprio`` adds a single 64-D proprio token at the end
    of the latent prefix (between text/image and the SONIC segment), so both
    SONIC and G1 attend to proprio via the per-layer attention. ``use_rtc``
    switches the G1 timestep into a per-token tensor and enables prefix
    substitution in the collator/inference paths.
    """

    def __init__(
        self,
        base_model,
        sonic_dim: int = 66,
        action_loss_weight: float = 1.0,
        use_g1_expert: bool = False,
        g1_dim: int = 31,
        g1_loss_weight: float = 0.3,
        use_proprio: bool = False,
        proprio_dim: int = 64,
        use_rtc: bool = False,
        rtc_on_sonic: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.sonic_dim = sonic_dim
        self.action_loss_weight = action_loss_weight
        self.use_g1_expert = use_g1_expert
        self.g1_dim = g1_dim
        self.g1_loss_weight = g1_loss_weight
        self.use_proprio = use_proprio and use_g1_expert
        self.proprio_dim = proprio_dim
        self.use_rtc = use_rtc
        self.rtc_on_sonic = rtc_on_sonic

        hidden_size = base_model.config.hidden_size
        self.action_embedder = nn.Linear(sonic_dim, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.velocity_head = VelocityHead(hidden_size, sonic_dim)
        self.norm_action = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)

        if use_g1_expert:
            self.action_embedder_g1 = nn.Linear(g1_dim, hidden_size)
            self.t_embedder_g1 = TimestepEmbedder(hidden_size)
            self.velocity_head_g1 = VelocityHead(hidden_size, g1_dim)
            self.norm_action_g1 = Qwen3VLTextRMSNorm(hidden_size, eps=1e-6)

        if self.use_proprio:
            self.proprio_embedder = nn.Linear(proprio_dim, hidden_size)
            nn.init.normal_(self.proprio_embedder.weight, std=0.02)
            nn.init.zeros_(self.proprio_embedder.bias)

        text_model = base_model.model.language_model
        new_layers = nn.ModuleList()
        for layer in text_model.layers:
            new_layers.append(MoTDecoderLayer(layer, has_g1=use_g1_expert))
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
        T_g: int,
        device,
        dtype,
        attention_mask: Optional[torch.Tensor],
        sonic_causal: bool,
    ) -> torch.Tensor:
        """4-section attention mask covering the full layout
        ``[text | proprio | t_emb_s + sonic | g1]``.

        Proprio sits inside the latent prefix so its K/V are cached together
        with text/image and the SONIC + G1 experts can both attend to it.

        Constraints:

        * ``text → text`` causal; ``text → proprio/sonic/g1`` blocked
          (KV cache invariant + protect Qwen text representations from
          proprio feedback).
        * ``proprio → text`` full; ``proprio → proprio`` self;
          ``proprio → sonic/g1`` blocked (KV cache invariant).
        * ``sonic → text + proprio`` full; ``sonic → sonic`` bidirectional
          (or causal if ``rtc_on_sonic``); ``sonic → g1`` blocked.
        * ``g1 → text + proprio + sonic`` full; ``g1 → g1`` causal (RTC).
        """
        T_total = T_text + T_p + T_s + T_g
        mask = torch.zeros(B, 1, T_total, T_total, device=device, dtype=dtype)

        n_text_end = T_text
        n_proprio_end = n_text_end + T_p          # == n_latent
        n_sonic_end = n_proprio_end + T_s
        # n_g1_end = n_sonic_end + T_g  (== T_total)

        # text causal among text rows.
        mask[:, 0, :n_text_end, :n_text_end].masked_fill_(
            _causal_mask(T_text, device, dtype), float("-inf"),
        )
        # text → tail (proprio + sonic + g1) blocked.
        mask[:, 0, :n_text_end, n_text_end:] = float("-inf")

        # proprio → sonic + proprio → g1 blocked.
        if T_p > 0:
            mask[:, 0, n_text_end:n_proprio_end, n_proprio_end:] = float("-inf")

        # sonic intra-segment.
        if sonic_causal and T_s > 1:
            sonic_self = mask[:, 0, n_proprio_end:n_sonic_end, n_proprio_end:n_sonic_end]
            sonic_self.masked_fill_(
                _causal_mask(T_s, device, dtype), float("-inf"),
            )
        # sonic → g1 blocked.
        if T_g > 0:
            mask[:, 0, n_proprio_end:n_sonic_end, n_sonic_end:] = float("-inf")

        # g1 segment causal.
        if T_g > 1:
            g1_self = mask[:, 0, n_sonic_end:, n_sonic_end:]
            g1_self.masked_fill_(_causal_mask(T_g, device, dtype), float("-inf"))

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
        # G1 expert inputs
        noisy_g1_tokens: Optional[torch.Tensor] = None,
        g1_timesteps: Optional[torch.Tensor] = None,
        g1_velocity_target: Optional[torch.Tensor] = None,
        g1_mask: Optional[torch.Tensor] = None,
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
        T_action = noisy_sonic_tokens.shape[1]

        # SONIC path: timestep token + noisy sonic tokens. ``T_s`` includes
        # the leading t_emb_s token because they share the SONIC expert.
        t_emb = self.t_embedder(timesteps).reshape(B, 1, D)
        sonic_embeds = self.action_embedder(noisy_sonic_tokens.to(inputs_embeds.dtype))
        sonic_segment = torch.cat([t_emb, sonic_embeds], dim=1)
        T_s = sonic_segment.shape[1]

        T_p = 0
        T_g = 0
        proprio_segment: Optional[torch.Tensor] = None
        g1_segment: Optional[torch.Tensor] = None

        if self.use_proprio and proprio is not None:
            proprio_segment = self.proprio_embedder(
                proprio.to(inputs_embeds.dtype)
            ).reshape(B, 1, D)
            T_p = 1

        if self.use_g1_expert and noisy_g1_tokens is not None:
            T_g = noisy_g1_tokens.shape[1]
            g1_action_emb = self.action_embedder_g1(noisy_g1_tokens.to(inputs_embeds.dtype))
            if g1_timesteps is None:
                g1_timesteps = timesteps
            # Per-token (RTC) or per-sample timestep both supported.
            g1_t_emb = self.t_embedder_g1(g1_timesteps)
            if g1_t_emb.dim() == 2:
                # ``(B, D)`` -> broadcast as additive bias across the chunk.
                g1_t_emb = g1_t_emb.unsqueeze(1).expand(-1, T_g, -1)
            g1_segment = (g1_action_emb + g1_t_emb).to(inputs_embeds.dtype)

        # Layout: [text | proprio? | sonic | g1?]. Proprio is part of the
        # latent prefix so SONIC and G1 both attend to it.
        parts = [inputs_embeds]
        if T_p > 0:
            parts.append(proprio_segment)
        parts.append(sonic_segment)
        if T_g > 0:
            parts.append(g1_segment)
        all_embeds = torch.cat(parts, dim=1)

        T_total = T_text + T_p + T_s + T_g
        n_latent = T_text + T_p
        n_sonic_end = n_latent + T_s

        sonic_causal = bool(self.rtc_on_sonic and self.use_rtc)
        full_attn_mask = self._build_mask(
            B=B, T_text=T_text, T_s=T_s, T_p=T_p, T_g=T_g,
            device=input_ids.device, dtype=all_embeds.dtype,
            attention_mask=attention_mask, sonic_causal=sonic_causal,
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
                n_sonic_end=n_sonic_end,
            )

        # SONIC velocity prediction (skip leading t_emb_s within sonic seg).
        sonic_hidden = self.norm_action(hidden_states[:, n_latent + 1:n_sonic_end])
        velocity_pred = self.velocity_head(sonic_hidden)

        loss_total = torch.zeros((), device=velocity_pred.device, dtype=torch.float32)
        sonic_loss = None
        g1_loss = None

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

        velocity_pred_g1 = None
        if T_g > 0:
            g1_hidden = self.norm_action_g1(hidden_states[:, n_sonic_end:])
            velocity_pred_g1 = self.velocity_head_g1(g1_hidden)
            if g1_velocity_target is not None:
                target_g = g1_velocity_target.to(velocity_pred_g1.dtype)
                if g1_mask is not None:
                    mask_g = g1_mask.to(velocity_pred_g1.dtype)
                    if mask_g.dim() == 2:
                        mask_g = mask_g.unsqueeze(-1)
                    denom_g = (mask_g.sum() * self.g1_dim).clamp(min=1)
                    mse_g = nn.functional.mse_loss(velocity_pred_g1, target_g, reduction="none")
                    g1_loss = (mse_g * mask_g).sum() / denom_g
                else:
                    g1_loss = nn.functional.mse_loss(velocity_pred_g1, target_g)
                loss_total = loss_total + self.g1_loss_weight * g1_loss

        if velocity_target is not None or g1_velocity_target is not None:
            output = CausalLMOutputWithPast(loss=loss_total)
            if sonic_loss is not None:
                output.sonic_loss = sonic_loss.detach()
            if g1_loss is not None:
                output.g1_loss = g1_loss.detach()
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
        T_g1: int = 0,
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
        T_g = T_g1
        T_total = T_text + T_p + T_s + T_g

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
        n_sonic_end = n_latent + T_s

        tail = torch.zeros(
            B, T_s + T_g, D,
            device=inputs_embeds.device, dtype=inputs_embeds.dtype,
        )
        all_embeds = torch.cat([prefix_embeds, tail], dim=1)

        full_attn_mask = self._build_mask(
            B=B, T_text=T_text, T_s=T_s, T_p=T_p, T_g=T_g,
            device=input_ids.device, dtype=all_embeds.dtype,
            attention_mask=attention_mask,
            sonic_causal=bool(self.rtc_on_sonic and self.use_rtc),
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
                n_sonic_end=n_sonic_end,
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
            "T_g": T_g,
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
        """Single-expert (SONIC only) denoise step. Used for backward compat
        and when ``use_g1_expert=False``.
        """
        language_model = self.base_model.model.language_model

        t_emb = self.t_embedder(timesteps).reshape(cache["B"], 1, -1)
        action_embeds = self.action_embedder(noisy_sonic_tokens.to(cache["dtype"]))
        action_h = torch.cat([t_emb, action_embeds], dim=1)
        T_tail = action_h.shape[1]

        for layer, kv_cache in zip(language_model.layers, cache["layer_caches"]):
            action_h = layer.forward_cached(
                action_h=action_h,
                position_embeddings_action=cache["position_embeddings_action"],
                attention_mask_action=cache["attention_mask_action"],
                kv_cache=kv_cache,
                n_sonic_in_tail=T_tail,
            )

        action_hidden = self.norm_action(action_h[:, 1:])
        return self.velocity_head(action_hidden)

    @torch.no_grad()
    def denoise_step_rtc(
        self,
        cache: dict,
        noisy_sonic_tokens: torch.Tensor,
        timesteps_sonic: torch.Tensor,
        noisy_g1_tokens: torch.Tensor,
        timesteps_g1: torch.Tensor,
        prev_g1: Optional[torch.Tensor] = None,
        delay_g1: Optional[torch.Tensor] = None,
        prev_sonic: Optional[torch.Tensor] = None,
        delay_sonic: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Three-expert RTC denoise step.

        Performs prefix substitution on ``noisy_g1_tokens`` (and optionally on
        ``noisy_sonic_tokens`` when ``rtc_on_sonic`` is enabled) using
        ``prev_*`` plus ``delay_*``. Returns ``(velocity_sonic, velocity_g1)``.

        Proprio is not handed to this step — it was embedded into the latent
        prefix during ``encode_latent`` so its K/V already live in the
        per-layer KV cache.
        """
        language_model = self.base_model.model.language_model
        B = cache["B"]
        D = self.base_model.config.hidden_size
        T_g = cache["T_g"]
        T_action = cache["T_action"]

        # SONIC path
        if prev_sonic is not None and delay_sonic is not None:
            arange_s = torch.arange(T_action, device=noisy_sonic_tokens.device)
            prefix_mask_s = arange_s[None, :] < delay_sonic[:, None]
            noisy_sonic_tokens = torch.where(
                prefix_mask_s[..., None], prev_sonic, noisy_sonic_tokens,
            )
            t_per_pos_s = torch.where(
                prefix_mask_s, torch.zeros_like(prefix_mask_s, dtype=timesteps_sonic.dtype),
                timesteps_sonic[:, None].expand(-1, T_action),
            )
            t_emb_s = self.t_embedder(t_per_pos_s.mean(dim=1))  # legacy single-token timestep
            t_emb_s = t_emb_s.reshape(B, 1, D)
        else:
            t_emb_s = self.t_embedder(timesteps_sonic).reshape(B, 1, D)

        sonic_embeds = self.action_embedder(noisy_sonic_tokens.to(cache["dtype"]))
        sonic_seg = torch.cat([t_emb_s, sonic_embeds], dim=1)
        T_s_tail = sonic_seg.shape[1]

        # G1 path with RTC prefix substitution
        if prev_g1 is not None and delay_g1 is not None:
            arange_g = torch.arange(T_g, device=noisy_g1_tokens.device)
            prefix_mask_g = arange_g[None, :] < delay_g1[:, None]
            noisy_g1_tokens = torch.where(
                prefix_mask_g[..., None], prev_g1, noisy_g1_tokens,
            )
            zero_t = torch.zeros_like(timesteps_g1).reshape(-1, 1).expand(-1, T_g)
            t_per_pos_g = torch.where(
                prefix_mask_g, zero_t, timesteps_g1.reshape(-1, 1).expand(-1, T_g),
            )
        else:
            t_per_pos_g = timesteps_g1.reshape(-1, 1).expand(-1, T_g)

        g1_action_emb = self.action_embedder_g1(noisy_g1_tokens.to(cache["dtype"]))
        g1_t_emb = self.t_embedder_g1(t_per_pos_g)
        g1_seg = (g1_action_emb + g1_t_emb).to(cache["dtype"])

        # Tail layout: [sonic | g1]. Proprio is in the cached prefix, not here.
        action_h = torch.cat([sonic_seg, g1_seg], dim=1)

        for layer, kv_cache in zip(language_model.layers, cache["layer_caches"]):
            action_h = layer.forward_cached(
                action_h=action_h,
                position_embeddings_action=cache["position_embeddings_action"],
                attention_mask_action=cache["attention_mask_action"],
                kv_cache=kv_cache,
                n_sonic_in_tail=T_s_tail,
            )

        # SONIC velocity (drop leading t_emb_s token).
        sonic_hidden = self.norm_action(action_h[:, 1:T_s_tail])
        v_sonic = self.velocity_head(sonic_hidden)

        # G1 velocity.
        g1_hidden = self.norm_action_g1(action_h[:, T_s_tail:])
        v_g1 = self.velocity_head_g1(g1_hidden)
        return v_sonic, v_g1

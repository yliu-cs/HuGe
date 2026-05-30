"""QwenSonicFM model facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON, MOTION_TOKEN_DIM, PROPRIO_DIM
from kappa.models.action_heads.layerwise_fm import LayerwiseFMActionHead


@dataclass
class QwenSonicFMConfig:
    hidden_size: int = 512
    vlm_hidden_size: int = 512
    action_horizon: int = ACTION_HORIZON
    state_dim: int = PROPRIO_DIM
    action_dim: int = ACTION_DIM
    num_inference_steps: int = 4
    base_vlm_path: str = "resource/model/Qwen3-VL-2B-Instruct"
    use_qwen: bool = True
    freeze_qwen: bool = True
    qwen_lora: bool = False
    train_qwen_lora: bool = False
    qwen_lora_r: int = 8
    qwen_lora_alpha: int = 16
    qwen_lora_dropout: float = 0.0
    qwen_lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    max_text_tokens: int | None = 48
    model_max_length: int = 8192
    char_max_length: int = 256


class QwenSonicFM(nn.Module):
    def __init__(self, config: QwenSonicFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or QwenSonicFMConfig()
        self.qwen = None
        self.text_embed = nn.Embedding(32768, self.config.vlm_hidden_size)
        self.image_proj = nn.Linear(3, self.config.vlm_hidden_size)
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.vlm_hidden_size),
            nn.GELU(),
            nn.Linear(self.config.vlm_hidden_size, self.config.vlm_hidden_size),
        )
        self.action_head = LayerwiseFMActionHead(
            hidden_size=self.config.hidden_size,
            condition_size=self.config.vlm_hidden_size,
            action_horizon=self.config.action_horizon,
        )
        self.qwen_proj = nn.Linear(self.config.vlm_hidden_size, self.config.vlm_hidden_size)
        if self.config.use_qwen:
            self._load_qwen()

    def _load_qwen(self) -> None:
        from transformers import AutoModelForImageTextToText

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        impls = ["flash_attention_2", "sdpa", None] if torch.cuda.is_available() else ["sdpa", None]
        last_err: Exception | None = None
        for impl in impls:
            try:
                kwargs: dict[str, Any] = {"local_files_only": True, "dtype": dtype}
                if impl is not None:
                    kwargs["attn_implementation"] = impl
                self.qwen = AutoModelForImageTextToText.from_pretrained(
                    Path(self.config.base_vlm_path),
                    **kwargs,
                )
                break
            except Exception as exc:
                last_err = exc
        if self.qwen is None:
            raise RuntimeError("failed to load Qwen3VL with flash-attn/sdpa fallback") from last_err
        text_config = getattr(self.qwen.config, "text_config", None)
        hidden = int(
            getattr(self.qwen.config, "hidden_size", None)
            or getattr(text_config, "hidden_size", None)
            or self.config.vlm_hidden_size
        )
        self.qwen_proj = nn.Linear(hidden, self.config.vlm_hidden_size)
        if self.config.train_qwen_lora and not self.config.qwen_lora:
            raise ValueError("train_qwen_lora requires qwen_lora")
        if self.config.qwen_lora:
            self._apply_qwen_lora()
        elif self.config.freeze_qwen:
            for p in self.qwen.parameters():
                p.requires_grad_(False)

    def _apply_qwen_lora(self) -> None:
        if self.qwen is None:
            raise RuntimeError("cannot apply LoRA before Qwen is loaded")
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError("Qwen LoRA requires the `peft` package; run `uv sync` after updating dependencies") from exc

        for p in self.qwen.parameters():
            p.requires_grad_(False)
        lora_config = LoraConfig(
            r=self.config.qwen_lora_r,
            lora_alpha=self.config.qwen_lora_alpha,
            lora_dropout=self.config.qwen_lora_dropout,
            target_modules=list(self.config.qwen_lora_target_modules),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        if not self.config.train_qwen_lora:
            for p in self.qwen.parameters():
                p.requires_grad_(False)

    def _qwen_grad_enabled(self) -> bool:
        return self.training and (self.config.train_qwen_lora or not self.config.freeze_qwen)

    def encode_latent(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        qwen_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        if self.qwen is not None and qwen_inputs is not None:
            qwen_inputs = {k: v.to(next(self.qwen.parameters()).device) if torch.is_tensor(v) else v for k, v in qwen_inputs.items()}
            with torch.set_grad_enabled(self._qwen_grad_enabled()):
                out = self.qwen(**qwen_inputs, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
            mask = qwen_inputs.get("attention_mask")
            if mask is not None:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
                pooled = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / denom
            else:
                pooled = hidden.mean(dim=1)
            pieces.append(self.qwen_proj(pooled.to(self.qwen_proj.weight.dtype)))
        elif input_ids is not None:
            pieces.append(self.text_embed(input_ids).mean(dim=1))
        if images is not None:
            img = images.float()
            if img.ndim == 5:
                img = img[:, -1]
            if img.shape[-1] == 3:
                pooled = img.reshape(img.shape[0], -1, 3).mean(dim=1)
            elif img.shape[1] == 3:
                pooled = img.flatten(2).mean(dim=-1)
            else:
                raise ValueError(f"cannot infer image channel layout from {images.shape}")
            pieces.append(self.image_proj(pooled.to(self.image_proj.weight.dtype)))
        if state is not None:
            state_feat = self.state_encoder(state.to(self.state_encoder[0].weight.dtype))
            if state_mask is not None:
                state_feat = state_feat * state_mask.reshape(state_feat.shape[0], -1).to(state_feat.dtype)[:, :1]
            pieces.append(state_feat)
        if not pieces:
            raise ValueError("encode_latent needs at least one of input_ids, images, or state")
        return torch.stack(pieces, dim=0).sum(dim=0)

    def denoise_step(
        self,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.action_head(noisy_action[..., :MOTION_TOKEN_DIM], t, condition)

    def forward(
        self,
        *,
        normalized_actions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        qwen_inputs: dict[str, torch.Tensor] | None = None,
        noise: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        condition = self.encode_latent(input_ids=input_ids, images=images, state=state, state_mask=state_mask, qwen_inputs=qwen_inputs)
        action_dtype = self.action_head.action_embed.weight.dtype
        motion = normalized_actions[..., :MOTION_TOKEN_DIM].to(action_dtype)
        if noise is None:
            noise = torch.randn_like(motion)
        else:
            noise = noise.to(dtype=motion.dtype, device=motion.device)
        if t is None:
            t = torch.rand((motion.shape[0],), device=motion.device, dtype=motion.dtype)
        else:
            t = t.to(dtype=motion.dtype, device=motion.device)
        noisy = t.reshape(-1, 1, 1) * noise + (1.0 - t.reshape(-1, 1, 1)) * motion
        pred = self.denoise_step(noisy, t, condition)
        pred["target_velocity"] = noise - motion
        pred["target_hand"] = normalized_actions[..., MOTION_TOKEN_DIM:ACTION_DIM]
        return pred

    @torch.no_grad()
    def predict_action(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        qwen_inputs: dict[str, torch.Tensor] | None = None,
        horizon: int | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        condition = self.encode_latent(input_ids=input_ids, images=images, state=state, state_mask=state_mask, qwen_inputs=qwen_inputs)
        return self.action_head.predict(
            condition,
            horizon=horizon or self.config.action_horizon,
            num_inference_steps=num_inference_steps or self.config.num_inference_steps,
        )

    def save_checkpoint(
        self,
        output_dir: str | Path,
        metadata: dict[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        import json
        from dataclasses import asdict

        from safetensors.torch import save_file, save_model

        dst = Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)
        if state_dict is None:
            save_model(self, str(dst / "model.safetensors"))
        else:
            tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items() if torch.is_tensor(v)}
            save_file(tensors, str(dst / "model.safetensors"))
        (dst / "config.json").write_text(
            json.dumps({"config": asdict(self.config), "metadata": metadata or {}}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, *, map_location: str | torch.device = "cpu") -> "QwenSonicFM":
        import json

        from safetensors.torch import load_model

        ckpt_path = Path(checkpoint_dir)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "model.safetensors"
        config_path = ckpt_path.parent / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        config = QwenSonicFMConfig(**payload.get("config", {}))
        model = cls(config)
        load_model(model, str(ckpt_path), strict=False, device=str(map_location))
        return model

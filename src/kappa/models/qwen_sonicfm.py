"""QwenSonicFM model facade."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kappa.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM
from kappa.models.action_heads.starvla_pi_head import StarVLAPIActionHead


@dataclass
class QwenSonicFMConfig:
    action_hidden_size: int = 1024
    num_action_layers: int = 16
    num_attention_heads: int | None = None
    action_ffn_multiplier: int = 4
    action_dropout: float = 0.1
    num_future_tokens: int = 32
    state_dropout_prob: float = 0.2
    flow_noise_beta_alpha: float = 1.5
    flow_noise_beta_beta: float = 1.0
    flow_noise_s: float = 0.999
    action_horizon: int = ACTION_HORIZON
    state_dim: int = PROPRIO_DIM
    action_dim: int = ACTION_DIM
    num_inference_steps: int = 4
    base_vlm_path: str = "resource/model/Qwen3-VL-2B-Instruct"
    use_qwen: bool = True
    freeze_qwen: bool = False
    qwen_lora: bool = False
    train_qwen_lora: bool = False
    qwen_lora_r: int = 8
    qwen_lora_alpha: int = 16
    qwen_lora_dropout: float = 0.0
    qwen_lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    max_text_tokens: int | None = 48
    model_max_length: int = 8192
    char_max_length: int = 256
    qwen_image_size: tuple[int, int] = (256, 256)
    image_crop_fraction: float = 0.95
    image_color_jitter: bool = True


class QwenSonicFM(nn.Module):
    def __init__(self, config: QwenSonicFMConfig | None = None) -> None:
        super().__init__()
        self.config = config or QwenSonicFMConfig()
        self.qwen = None
        if not self.config.use_qwen:
            raise ValueError("Kappa now requires use_qwen=True; char-token fallback has been removed")
        self.action_head: StarVLAPIActionHead | None = None
        self.vlm_projector = nn.Identity()
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
        text_config = getattr(self.qwen.config, "text_config", self.qwen.config)
        hidden = getattr(self.qwen.config, "hidden_size", None) or getattr(text_config, "hidden_size", None)
        if hidden is None:
            raise ValueError("loaded Qwen model does not expose hidden_size")
        hidden = int(hidden)
        num_vl_layers = getattr(text_config, "num_hidden_layers", None)
        if num_vl_layers is None:
            raise ValueError("loaded Qwen model does not expose text_config.num_hidden_layers")
        self._init_action_structure(hidden, int(num_vl_layers))
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

    def _init_action_structure(self, qwen_hidden_size: int, num_vl_layers: int) -> None:
        action_hidden_size = int(self.config.action_hidden_size or 1024)
        num_action_layers = int(self.config.num_action_layers or 16)
        if action_hidden_size % 64 != 0:
            raise ValueError(f"action_hidden_size {action_hidden_size} is not divisible by 64")
        num_heads = int(self.config.num_attention_heads or (action_hidden_size // 64))
        if num_action_layers <= 0:
            raise ValueError(f"num_action_layers must be positive, got {num_action_layers}")
        if num_action_layers > num_vl_layers:
            raise ValueError(
                f"num_action_layers={num_action_layers} exceeds available VLM layers {num_vl_layers}"
            )
        if action_hidden_size % num_heads != 0:
            raise ValueError(f"action_hidden_size {action_hidden_size} must be divisible by num_heads {num_heads}")
        self.config.action_hidden_size = action_hidden_size
        self.config.num_action_layers = num_action_layers
        self.config.num_attention_heads = num_heads
        self.vlm_projector = (
            nn.Identity()
            if qwen_hidden_size == action_hidden_size
            else nn.Sequential(
                nn.LayerNorm(qwen_hidden_size),
                nn.Linear(qwen_hidden_size, action_hidden_size),
            )
        )
        self.action_head = StarVLAPIActionHead(
            hidden_size=action_hidden_size,
            action_dim=self.config.action_dim,
            state_dim=self.config.state_dim,
            action_horizon=self.config.action_horizon,
            num_layers=num_action_layers,
            num_heads=num_heads,
            ffn_multiplier=self.config.action_ffn_multiplier,
            dropout=self.config.action_dropout,
            num_inference_steps=self.config.num_inference_steps,
            noise_beta_alpha=self.config.flow_noise_beta_alpha,
            noise_beta_beta=self.config.flow_noise_beta_beta,
            noise_s=self.config.flow_noise_s,
            state_dropout_prob=self.config.state_dropout_prob,
        )

    def _qwen_grad_enabled(self) -> bool:
        return self.training and (self.config.train_qwen_lora or not self.config.freeze_qwen)

    def encode_vlm_layers(
        self,
        *,
        qwen_inputs: dict[str, torch.Tensor] | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        if qwen_inputs is None:
            raise ValueError("Kappa PI requires qwen_inputs from the Qwen processor")
        if self.qwen is None:
            raise RuntimeError("Kappa PI requires a loaded Qwen model")
        qwen_inputs = {k: v.to(next(self.qwen.parameters()).device) if torch.is_tensor(v) else v for k, v in qwen_inputs.items()}
        with torch.set_grad_enabled(self._qwen_grad_enabled()):
            out = self.qwen(**qwen_inputs, output_hidden_states=True, return_dict=True)
        hidden_states = list(out.hidden_states)
        num_action_layers = int(self.config.num_action_layers or 0)
        if len(hidden_states) < num_action_layers:
            raise ValueError(
                f"num_action_layers={num_action_layers} exceeds available hidden states {len(hidden_states)}"
            )
        selected = hidden_states[-num_action_layers:]
        attention_mask = qwen_inputs.get("attention_mask")
        if attention_mask is not None and selected:
            attention_mask = attention_mask[:, : selected[0].shape[1]]
        return selected, attention_mask

    def _project_vlm_layers(self, vl_embs_list: list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(self.vlm_projector, nn.Identity):
            return vl_embs_list
        target_dtype = next(self.vlm_projector.parameters()).dtype
        return [self.vlm_projector(h.to(dtype=target_dtype)) for h in vl_embs_list]

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
        del input_ids, images
        vl_embs_list, attention_mask = self.encode_vlm_layers(qwen_inputs=qwen_inputs)
        vl_embs_list = self._project_vlm_layers(vl_embs_list)
        if self.action_head is None:
            raise RuntimeError("PI action head is not initialized")
        return self.action_head(
            vl_embs_list=vl_embs_list,
            actions=normalized_actions.to(vl_embs_list[0].dtype),
            state=state,
            state_mask=state_mask,
            encoder_attention_mask=attention_mask,
            noise=noise,
            t=t,
        )

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
        del input_ids, images
        vl_embs_list, attention_mask = self.encode_vlm_layers(qwen_inputs=qwen_inputs)
        vl_embs_list = self._project_vlm_layers(vl_embs_list)
        if self.action_head is None:
            raise RuntimeError("PI action head is not initialized")
        return self.action_head.predict(
            vl_embs_list=vl_embs_list,
            state=state,
            state_mask=state_mask,
            encoder_attention_mask=attention_mask,
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
            tensors: dict[str, torch.Tensor] = {}
            seen_storage_ptrs: set[int] = set()
            for k, v in state_dict.items():
                if not torch.is_tensor(v):
                    continue
                storage_ptr = v.untyped_storage().data_ptr()
                if storage_ptr in seen_storage_ptrs:
                    continue
                seen_storage_ptrs.add(storage_ptr)
                tensors[k] = v.detach().cpu().contiguous()
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
        config_raw = payload.get("config", {})
        allowed_keys = {field.name for field in fields(QwenSonicFMConfig)}
        config = QwenSonicFMConfig(**{k: v for k, v in config_raw.items() if k in allowed_keys})
        model = cls(config)
        load_model(model, str(ckpt_path), strict=False, device=str(map_location))
        return model

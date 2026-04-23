"""Kappa model wrapper."""
from __future__ import annotations

from typing import Optional, Any, Union
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen2TokenizerFast

from src.utils import initialize_overwatch
overwatch = initialize_overwatch(__name__)


class KappaModelWrapper(nn.Module):
    """Kappa model wrapper around VLM backbone for action prediction."""

    def __init__(
        self,
        model_name_or_path: str,
        tune_vision: bool = False,
        tune_llm: bool = True,
        tune_mlp: bool = True,
        lora_config: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        overwatch.info(f"Loading backbone from {model_name_or_path}")

        load_dtype = torch.float32 if (device is not None and device.type == "mps") else torch.bfloat16

        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                attn_implementation="sdpa",
                torch_dtype=load_dtype,
                local_files_only=True,
            )
            overwatch.info(f"Loaded backbone from {model_name_or_path}")
        except Exception as e:
            overwatch.warning(f"Failed to load with sdpa, trying without: {e}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=load_dtype,
                local_files_only=True,
            )

        if not hasattr(self.model.config, "hidden_size"):
            self.model.config.hidden_size = self.model.config.text_config.hidden_size

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer

        if not tune_vision:
            for param in self.model.model.visual.parameters():
                param.requires_grad = False
        if not tune_llm:
            for param in self.model.model.language_model.parameters():
                param.requires_grad = False
        if not tune_mlp:
            if hasattr(self.model.model.visual, "merger"):
                for param in self.model.model.visual.merger.parameters():
                    param.requires_grad = False

        if lora_config is not None:
            self._apply_lora(lora_config)

        if device is not None:
            self.model = self.model.to(device)

    def convert_to_mot(self, sonic_dim: int = 64, action_loss_weight: float = 1.0):
        from src.models.mot_kappa import MoTKappaModel
        overwatch.info("Converting to MoT architecture...")
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.model = MoTKappaModel(
            self.model, sonic_dim=sonic_dim, action_loss_weight=action_loss_weight,
        ).to(dtype=dtype, device=device)
        overwatch.info("MoT conversion complete")

    def _apply_lora(self, lora_config):
        """Apply LoRA to the model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                r=lora_config.get("r", 8),
                lora_alpha=lora_config.get("alpha", 16),
                dropout=lora_config.get("dropout", 0.0),
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, lora_cfg)
            overwatch.info("Applied LoRA")
        except ImportError:
            overwatch.warning("peft not installed, skipping LoRA")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass through the model."""
        from transformers.modeling_outputs import CausalLMOutputWithPast

        outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            image_grid_thw=image_grid_thw,
        )
        return outputs

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

"""Kappa model wrapper."""
from __future__ import annotations

from typing import Optional, Any, Union
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

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

        # Prefer flash-attn-2 for the vision tower and any non-MoT call path
        # (it is much faster than SDPA on the patch grid). MoTAttention forces
        # SDPA internally because its 4D float mask is not flash-attn-2
        # compatible. flash-attn-2 requires CUDA + bf16/fp16; fall back to
        # SDPA on CPU/MPS or when the kernel isn't available.
        prefer_fa2 = (
            torch.cuda.is_available()
            and load_dtype in (torch.bfloat16, torch.float16)
        )
        attn_impls = (
            ["flash_attention_2", "sdpa", None] if prefer_fa2 else ["sdpa", None]
        )

        last_err: Exception | None = None
        self.model = None
        for impl in attn_impls:
            try:
                kwargs = dict(torch_dtype=load_dtype, local_files_only=True)
                if impl is not None:
                    kwargs["attn_implementation"] = impl
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name_or_path, **kwargs
                )
                overwatch.info(
                    f"Loaded backbone from {model_name_or_path} "
                    f"(attn_implementation={impl or 'default'})"
                )
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                overwatch.warning(
                    f"Failed to load with attn_implementation={impl}: {e}"
                )
        if self.model is None:
            raise RuntimeError("Failed to load backbone with any attn impl") from last_err

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

    def convert_to_mot(
        self,
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
        from src.models.mot_kappa import MoTKappaModel
        overwatch.info(
            f"Converting to MoT (g1={use_g1_expert}, proprio={use_proprio}, "
            f"rtc={use_rtc}, rtc_on_sonic={rtc_on_sonic})..."
        )
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.model = MoTKappaModel(
            self.model,
            sonic_dim=sonic_dim,
            action_loss_weight=action_loss_weight,
            use_g1_expert=use_g1_expert,
            g1_dim=g1_dim,
            g1_loss_weight=g1_loss_weight,
            use_proprio=use_proprio,
            proprio_dim=proprio_dim,
            use_rtc=use_rtc,
            rtc_on_sonic=rtc_on_sonic,
        ).to(dtype=dtype, device=device)
        overwatch.info("MoT conversion complete")

    def init_g1_from_sonic_expert(self) -> int:
        """Copy SONIC expert weights into the G1 expert副本.

        For every layer the per-expert ``*_action`` parameters (q/k/v/o, q/k
        norms, MLP, layernorms) are copied to the matching ``*_g1``副本; the
        same applies to the top-level ``norm_action`` -> ``norm_action_g1`` and
        the timestep embedder. ``action_embedder_g1`` (sonic_dim → hidden) and
        ``velocity_head_g1`` (hidden → g1_dim) have different I/O dims and
        keep their random / zero init. Returns the number of tensors copied.
        """
        from src.models.mot_kappa import MoTKappaModel
        if not isinstance(self.model, MoTKappaModel):
            raise RuntimeError("init_g1_from_sonic_expert requires convert_to_mot first")
        if not self.model.use_g1_expert:
            raise RuntimeError("Model has no G1 expert; convert with use_g1_expert=True")

        copied = 0
        for layer in self.model.base_model.model.language_model.layers:
            attn = layer.self_attn
            for src, dst in [
                (attn.q_proj_action, attn.q_proj_g1),
                (attn.k_proj_action, attn.k_proj_g1),
                (attn.v_proj_action, attn.v_proj_g1),
                (attn.o_proj_action, attn.o_proj_g1),
                (attn.q_norm_action, attn.q_norm_g1),
                (attn.k_norm_action, attn.k_norm_g1),
            ]:
                dst.load_state_dict(src.state_dict())
                copied += 1
            layer.mlp_g1.load_state_dict(layer.mlp_action.state_dict())
            layer.input_layernorm_g1.load_state_dict(layer.input_layernorm_action.state_dict())
            layer.post_attention_layernorm_g1.load_state_dict(
                layer.post_attention_layernorm_action.state_dict(),
            )
            copied += 3
        # Top-level companions: timestep + final norm. Embedder/head have
        # different shapes so keep their original init.
        self.model.t_embedder_g1.load_state_dict(self.model.t_embedder.state_dict())
        self.model.norm_action_g1.load_state_dict(self.model.norm_action.state_dict())
        copied += 2
        overwatch.info(f"Initialized G1 expert from SONIC expert ({copied} tensors).")
        return copied

    def freeze_for_post_train(self) -> tuple[int, int]:
        """Freeze everything except the SONIC + G1 experts (and proprio inputs).

        Returns ``(n_trainable, n_total)``. The freezable list mirrors the
        post-train design: visual tower, text embeddings, latent decoder
        path, and the LM head are frozen; only the per-layer ``*_action`` /
        ``*_g1`` projections + MLPs + layernorms and the top-level action /
        G1 modules remain trainable.
        """
        from src.models.mot_kappa import MoTKappaModel
        if not isinstance(self.model, MoTKappaModel):
            raise RuntimeError("freeze_for_post_train requires convert_to_mot first")

        trainable_substrings = (
            "_action",          # per-layer SONIC副本 and final norm/head
            "_g1",              # per-layer G1副本 and top-level
            "action_embedder",  # sonic embedder (top-level)
            "t_embedder",       # both t_embedder + t_embedder_g1
            "velocity_head",    # both velocity_head + velocity_head_g1
            "norm_action",      # final norms (both experts)
            "proprio_embedder", # latent-prefix proprio Linear (no _g1 suffix)
        )
        n_total = 0
        n_trainable = 0
        for name, param in self.model.named_parameters():
            n_total += 1
            keep = any(sub in name for sub in trainable_substrings)
            param.requires_grad = bool(keep)
            if keep:
                n_trainable += 1
        overwatch.info(
            f"Post-train freeze: {n_trainable}/{n_total} tensors trainable."
        )
        return n_trainable, n_total

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

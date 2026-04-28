"""Kappa trainer for pretraining with action prediction."""
from __future__ import annotations

import os
from typing import Any, Optional, Union, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.utils import initialize_overwatch, move_to_device
from src.trainers import Trainer

overwatch = initialize_overwatch(__name__)


class KappaCollator:
    """Collator that prepares Qwen3-VL ChatML inputs and flow-matching targets.

    Uses ``apply_chat_template`` so the processor produces the canonical
    ``<|im_start|>user\\n<|vision_start|><|image_pad|>...<|vision_end|>{text}<|im_end|>\\n<|im_start|>assistant\\n``
    sequence and auto-expands ``<|image_pad|>`` to one token per merged patch.
    Labels are not produced because the MoT path supervises via
    ``velocity_target`` only; the CE fallback in the model is unused at train.
    """

    def __init__(self, processor, model_max_length: int = 8192):
        self.processor = processor
        self.model_max_length = model_max_length

    def __call__(self, instances):
        from qwen_vl_utils import process_vision_info

        messages = [[{
            'role': 'user',
            'content': [
                {'type': 'image', 'image': inst["image"]},
                {'type': 'text', 'text': inst.get("text", "perform the task")},
            ],
        }] for inst in instances]

        texts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            for conv in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][:, :self.model_max_length]
        attention_mask = inputs["attention_mask"][:, :self.model_max_length]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        if "mm_token_type_ids" in inputs:
            result["mm_token_type_ids"] = inputs["mm_token_type_ids"][:, :self.model_max_length]

        if "sonic_tokens" in instances[0]:
            sonic_tokens = torch.stack([inst["sonic_tokens"] for inst in instances])
            sonic_mask = torch.stack([inst["sonic_mask"] for inst in instances])

            B = sonic_tokens.shape[0]
            t = torch.distributions.Beta(1.5, 1.0).sample((B,)) * 0.999 + 0.001
            t_expanded = t[:, None, None]
            noise = torch.randn_like(sonic_tokens)
            noisy_sonic_tokens = t_expanded * noise + (1 - t_expanded) * sonic_tokens
            velocity_target = noise - sonic_tokens

            result["noisy_sonic_tokens"] = noisy_sonic_tokens
            result["timesteps"] = t
            result["velocity_target"] = velocity_target
            result["sonic_mask"] = sonic_mask

        return result


class KappaTrainer(Trainer):
    """Trainer for Kappa action prediction."""

    def __init__(self, cfg: Any, device: Union[torch.device, int]):
        super().__init__(cfg, device)
        self.vlm_processor = None
        self.tokenizer = None
        self.val_dataloader_drop_last = False
        self.train_dataloader_drop_last = True

    @property
    def task_run_name(self) -> str:
        return ".kappa"

    def init_models(self):
        from src.models import KappaModelWrapper

        overwatch.info("Initializing Kappa model...")
        model_cfg = self.cfg.model

        lora_config = None
        if model_cfg.lora:
            lora_config = {
                "r": model_cfg.lora_r,
                "alpha": model_cfg.lora_alpha,
                "dropout": model_cfg.lora_dropout,
            }

        model_wrapper = KappaModelWrapper(
            model_name_or_path=model_cfg.model_name_or_path,
            tune_vision=model_cfg.tune_mm_vision,
            tune_llm=model_cfg.tune_mm_llm,
            tune_mlp=model_cfg.tune_mm_mlp,
            lora_config=lora_config,
            device=self.device,
        )

        if getattr(model_cfg, "use_mot", False):
            model_wrapper.convert_to_mot(
                sonic_dim=model_cfg.sonic_dim,
                action_loss_weight=model_cfg.action_loss_weight,
            )

        self.model = model_wrapper.model
        self.vlm_processor = model_wrapper.processor
        self.tokenizer = model_wrapper.tokenizer

        if model_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def create_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        data_cfg = self.cfg.data
        name = getattr(data_cfg, "dataset_name", "humanoid_everyday")

        # Keep model.sonic_dim as the source of truth for EgoDex pad width even
        # if the data config defaults drift; sync before instantiating.
        if name == "egodex":
            data_cfg.sonic_dim = self.cfg.model.sonic_dim

        overwatch.info(f"Creating {name} dataset (transform: repack -> field -> model)...")
        self.train_dataset = data_cfg(split="train")
        self.val_dataset = None
        self.len_train_dataset = len(self.train_dataset)
        self.len_val_dataset = 0
        return self.train_dataset, self.val_dataset

    def create_dataloaders(
        self, train_dataset: Dataset, val_dataset: Optional[Dataset]
    ) -> tuple[DataLoader, Optional[DataLoader]]:
        collator = KappaCollator(
            processor=self.vlm_processor,
            model_max_length=self.cfg.model.model_max_length,
        )

        g = torch.Generator()
        g.manual_seed(self.cfg.seed or 42)

        nw = self.train_cfg.num_workers
        loader_kwargs = dict(
            collate_fn=collator,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
        )
        if nw > 0:
            loader_kwargs["prefetch_factor"] = self.train_cfg.prefetch_factor

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.train_cfg.train_batch_size,
            generator=g,
            drop_last=True,
            **loader_kwargs,
        )

        if val_dataset is not None:
            self.val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=self.train_cfg.val_batch_size,
                drop_last=False,
                **loader_kwargs,
            )
        else:
            self.val_dataloader = None

        return self.train_dataloader, self.val_dataloader

    def training_step(
        self,
        batch: dict[str, torch.Tensor | Any],
    ) -> tuple[bool, dict[str, Any]]:

        timer = getattr(self, "_step_timer", None)

        if timer:
            timer.start("data_transfer")
        batch = move_to_device(batch, self.device)
        batch = {k: v for k, v in batch.items() if v is not None}
        if timer:
            timer.stop("data_transfer")

        if timer:
            timer.start("forward")
        with self.accelerator.autocast():
            output: CausalLMOutputWithPast = self.model(**batch)
            loss = output.loss
        if timer:
            timer.stop("forward")

        if timer:
            timer.start("backward")
        self.accelerator.backward(loss)
        if timer:
            timer.stop("backward")

        if self.accelerator.sync_gradients:
            if timer:
                timer.start("optimizer")
            if not hasattr(self.accelerator.state, "deepspeed_plugin") or self.accelerator.state.deepspeed_plugin is None:
                if self.train_cfg.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.max_grad_norm
                    )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if timer:
                timer.stop("optimizer")

        return self.accelerator.sync_gradients, {
            "loss": loss.item(),
            "lr": self.lr_scheduler.get_last_lr()[0],
        }

    def create_optimizer(self):
        """Create optimizer with parameter groups for vision, projector, and LLM."""
        from torch.optim import AdamW

        model_cfg = self.cfg.model
        train_cfg = self.train_cfg

        param_groups = []

        projector_params = []
        llm_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "mlp1" in name or "projector" in name or "merger" in name:
                projector_params.append(param)
            elif "lm_head" in name or "embed_tokens" in name or "layers" in name or "norm" in name:
                llm_params.append(param)
            else:
                other_params.append(param)

        optimizer_kwargs = dict(train_cfg.optimizer_kwargs)

        if len(projector_params) > 0:
            param_groups.append({"params": projector_params, "lr": model_cfg.mm_projector_lr, "weight_decay": model_cfg.weight_decay})
        if len(llm_params) > 0:
            param_groups.append({"params": llm_params, "lr": train_cfg.learning_rate, "weight_decay": model_cfg.weight_decay})
        if len(other_params) > 0:
            param_groups.append({"params": other_params, "lr": train_cfg.learning_rate, "weight_decay": model_cfg.weight_decay})

        self.optimizer = AdamW(param_groups, **optimizer_kwargs)

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
    """Collator that uses the VLM processor for proper input preparation."""

    def __init__(self, processor, model_max_length: int = 8192, use_images: bool = True):
        self.processor = processor
        self.model_max_length = model_max_length
        self.use_images = use_images

    def __call__(self, instances):
        # For PIL images, process through the processor
        if "image" in instances[0] and self.use_images:
            from qwen_vl_utils import process_vision_info

            # Build messages format for qwen_vl_utils
            messages = [[{
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': inst["image"]},
                    {'type': 'text', 'text': inst.get("text", "perform the task")}
                ]
            }] for inst in instances]

            texts = [m[0]['content'][1]['text'] for m in messages]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # Manually insert image tokens at the beginning since processor doesn't do it
            # spatial_merge_size=2 merges 2x2 patches into 1
            image_token_id = 151655
            video_token_id = 151656
            spatial_merge_size = 2
            image_grid_thw = inputs.get("image_grid_thw", None)
            new_input_ids_list = []
            new_labels_list = []
            new_mm_token_type_ids_list = []

            for i, input_ids in enumerate(inputs["input_ids"]):
                if image_grid_thw is not None:
                    t = image_grid_thw[i]
                    # Divide by spatial_merge_size^2 because patches are merged
                    num_patches = (t[0] * t[1] * t[2]) // (spatial_merge_size ** 2)
                else:
                    num_patches = 0

                if num_patches > 0:
                    # Create image tokens
                    image_tokens = torch.full((num_patches,), image_token_id, dtype=input_ids.dtype)
                    new_input_ids = torch.cat([image_tokens, input_ids])

                    # Create mm_token_type_ids: 0=text, 1=image, 2=video
                    image_mm_types = torch.full((num_patches,), 1, dtype=torch.long)  # 1 for images
                    text_mm_types = torch.zeros((input_ids.shape[0],), dtype=torch.long)  # 0 for text
                    new_mm_type = torch.cat([image_mm_types, text_mm_types])
                else:
                    new_input_ids = input_ids
                    new_mm_type = torch.zeros((input_ids.shape[0],), dtype=torch.long)

                # Create labels (shift by 1 for causal LM, pad with -100)
                labels = new_input_ids.clone()
                labels[labels == self.processor.tokenizer.pad_token_id] = -100

                new_input_ids_list.append(new_input_ids)
                new_labels_list.append(labels)
                new_mm_token_type_ids_list.append(new_mm_type)

            # Pad sequences to same length
            from torch.nn.utils.rnn import pad_sequence
            input_ids = pad_sequence(new_input_ids_list, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id or 0)
            labels = pad_sequence(new_labels_list, batch_first=True, padding_value=-100)
            mm_token_type_ids = pad_sequence(new_mm_token_type_ids_list, batch_first=True, padding_value=0)

            # Truncate if needed
            input_ids = input_ids[:, :self.model_max_length]
            labels = labels[:, :self.model_max_length]
            mm_token_type_ids = mm_token_type_ids[:, :self.model_max_length]

            attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id or 0)

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "mm_token_type_ids": mm_token_type_ids,
            }

            if "pixel_values" in inputs:
                result["pixel_values"] = inputs["pixel_values"]
            if "image_grid_thw" in inputs:
                result["image_grid_thw"] = inputs["image_grid_thw"]

            if "sonic_tokens" in instances[0]:
                sonic_tokens = torch.stack([inst["sonic_tokens"] for inst in instances])
                sonic_mask = torch.stack([inst["sonic_mask"] for inst in instances])

                # Flow matching: sample timestep, add noise, compute velocity target
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
        else:
            # Original behavior for tensor data
            from torch.nn.utils.rnn import pad_sequence

            input_ids = [inst["input_ids"].squeeze(0) if inst["input_ids"].dim() == 2 else inst["input_ids"]
                         for inst in instances]
            labels = [inst["labels"].squeeze(0) if inst["labels"].dim() == 2 else inst["labels"]
                      for inst in instances]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id or 0)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)

            input_ids = input_ids[:, :self.model_max_length]
            labels = labels[:, :self.model_max_length]

            attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id or 0)

            pixel_values = torch.stack([inst["pixel_values"] for inst in instances])
            image_grid_thw = torch.stack([inst["image_grid_thw"].squeeze(0) for inst in instances])

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }


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
        from src.data import HEDataset

        overwatch.info("Creating Humanoid-Everyday datasets...")
        data_cfg = self.cfg.data

        self.train_dataset = HEDataset(
            data_root=data_cfg.data_root,
            action_chunk_size=data_cfg.action_chunk_size,
            split="train",
        )
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

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.train_cfg.train_batch_size,
            generator=g,
            collate_fn=collator,
            num_workers=self.train_cfg.num_workers,
            drop_last=True,
            persistent_workers=True if self.train_cfg.num_workers > 0 else False,
        )

        if val_dataset is not None:
            self.val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=self.train_cfg.val_batch_size,
                collate_fn=collator,
                num_workers=self.train_cfg.num_workers,
                drop_last=False,
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

        vision_params = []
        projector_params = []
        llm_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "visual" in name:
                vision_params.append(param)
            elif "mlp1" in name or "projector" in name or "merger" in name:
                projector_params.append(param)
            elif "lm_head" in name or "embed_tokens" in name or "layers" in name or "norm" in name:
                llm_params.append(param)
            else:
                other_params.append(param)

        optimizer_kwargs = dict(train_cfg.lr_scheduler_kwargs)

        if len(vision_params) > 0:
            param_groups.append({"params": vision_params, "lr": model_cfg.vision_tower_lr, "weight_decay": model_cfg.weight_decay})
        if len(projector_params) > 0:
            param_groups.append({"params": projector_params, "lr": model_cfg.mm_projector_lr, "weight_decay": model_cfg.weight_decay})
        if len(llm_params) > 0:
            param_groups.append({"params": llm_params, "lr": train_cfg.learning_rate, "weight_decay": model_cfg.weight_decay})
        if len(other_params) > 0:
            param_groups.append({"params": other_params, "lr": train_cfg.learning_rate, "weight_decay": model_cfg.weight_decay})

        self.optimizer = AdamW(param_groups, **optimizer_kwargs)

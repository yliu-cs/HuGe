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

    Post-train mode (``use_g1_expert=True``) additionally:

    * stacks ``g1_actions`` / ``g1_mask`` / ``proprio`` from each instance,
    * draws an independent G1 (and optionally SONIC) RTC delay,
    * builds prefix-aware noisy actions and velocity targets for both streams,
    * adds Gaussian jitter to the proprio (skipping hand-bit + quat dims).
    """

    def __init__(
        self,
        processor,
        model_max_length: int = 8192,
        use_g1_expert: bool = False,
        use_proprio: bool = False,
        use_rtc: bool = False,
        rtc_on_sonic: bool = True,
        g1_max_delay: int = 10,
        sonic_max_delay: int = 8,
        proprio_noise_std: float = 0.02,
        proprio_drop_prob: float = 0.1,
        proprio_dim: int = 64,
    ):
        self.processor = processor
        self.model_max_length = model_max_length
        self.use_g1_expert = use_g1_expert
        self.use_proprio = use_proprio and use_g1_expert
        self.use_rtc = use_rtc
        self.rtc_on_sonic = rtc_on_sonic and use_rtc
        self.g1_max_delay = g1_max_delay
        self.sonic_max_delay = sonic_max_delay
        self.proprio_noise_std = proprio_noise_std
        self.proprio_drop_prob = proprio_drop_prob
        self.proprio_dim = proprio_dim

    @staticmethod
    def _logit_normal_t(B: int) -> torch.Tensor:
        # Same logit-normal schedule as pretrain — density peaks at t=0.5.
        return torch.sigmoid(torch.randn(B)) * 0.999 + 0.001

    def _build_proprio_noise_mask(self) -> torch.Tensor:
        """Hand-bit columns (29:31) and odom_quat (60:64) are categorical /
        unit-norm; jittering them in normalized space breaks their geometry.
        Joint position (0:29) and joint velocity (31:60) are smooth scalars
        so we apply Gaussian noise there.
        """
        mask = torch.ones(self.proprio_dim)
        if self.proprio_dim == 64:
            mask[29:31] = 0.0
            mask[60:64] = 0.0
        return mask

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
            B, T_s, _ = sonic_tokens.shape

            t = self._logit_normal_t(B)
            noise_s = torch.randn_like(sonic_tokens)

            if self.rtc_on_sonic:
                # SONIC RTC: prefix region gets t=0 (clean a) and is masked
                # out of the loss. ``timesteps`` is per-sample (single t emb
                # token still); the prefix substitution happens in the
                # noisy_sonic field so the model attends to clean prefix
                # values. This matches Psi0's training-time RTC.
                delay_s = torch.randint(0, max(1, self.sonic_max_delay), (B,))
                arange_s = torch.arange(T_s)
                prefix_mask_s = arange_s[None, :] < delay_s[:, None]  # (B, T_s)
                t_per_pos_s = torch.where(
                    prefix_mask_s, torch.zeros_like(t)[:, None].expand(-1, T_s),
                    t[:, None].expand(-1, T_s),
                )
                t_for_noise = t_per_pos_s[..., None]
                noisy_sonic_tokens = t_for_noise * noise_s + (1 - t_for_noise) * sonic_tokens
                velocity_target = noise_s - sonic_tokens
                # Loss is suppressed on prefix positions.
                postfix_mask_s = (~prefix_mask_s).to(sonic_mask.dtype)
                if sonic_mask.dim() == 2:
                    sonic_mask = sonic_mask * postfix_mask_s
                else:
                    sonic_mask = sonic_mask * postfix_mask_s.unsqueeze(-1)
            else:
                t_expanded = t[:, None, None]
                noisy_sonic_tokens = t_expanded * noise_s + (1 - t_expanded) * sonic_tokens
                velocity_target = noise_s - sonic_tokens

            result["noisy_sonic_tokens"] = noisy_sonic_tokens
            result["timesteps"] = t
            result["velocity_target"] = velocity_target
            result["sonic_mask"] = sonic_mask

        if self.use_g1_expert and "g1_actions" in instances[0]:
            g1_actions = torch.stack([inst["g1_actions"] for inst in instances])
            g1_mask = torch.stack([inst["g1_mask"] for inst in instances])
            B, T_g, D_g = g1_actions.shape

            # Reuse the SONIC ``t`` so both losses sample the same global
            # noise level — keeps the two streams' training in sync as the
            # design doc describes.
            t = result.get("timesteps", self._logit_normal_t(B))
            delay_g = torch.randint(0, max(1, self.g1_max_delay), (B,))
            arange_g = torch.arange(T_g)
            prefix_mask_g = arange_g[None, :] < delay_g[:, None]  # (B, T_g)

            noise_g = torch.randn_like(g1_actions)
            t_per_pos_g = torch.where(
                prefix_mask_g, torch.zeros_like(t)[:, None].expand(-1, T_g),
                t[:, None].expand(-1, T_g),
            )
            t_for_noise = t_per_pos_g[..., None]
            noisy_g1_tokens = t_for_noise * noise_g + (1 - t_for_noise) * g1_actions
            g1_velocity_target = noise_g - g1_actions

            postfix_mask_g = (~prefix_mask_g).to(g1_mask.dtype)
            g1_mask = g1_mask * postfix_mask_g

            result["noisy_g1_tokens"] = noisy_g1_tokens
            # Per-pos timestep for the G1 head; the model handles 2-D shapes.
            result["g1_timesteps"] = t_per_pos_g
            result["g1_velocity_target"] = g1_velocity_target
            result["g1_mask"] = g1_mask

        if self.use_proprio and "proprio" in instances[0]:
            proprio = torch.stack([inst["proprio"] for inst in instances])  # (B, D_p)
            if self.proprio_noise_std > 0:
                noise_mask = self._build_proprio_noise_mask().to(proprio.dtype)
                proprio = proprio + torch.randn_like(proprio) * self.proprio_noise_std * noise_mask
            if self.proprio_drop_prob > 0:
                drop = (torch.rand(proprio.shape[0]) < self.proprio_drop_prob).to(proprio.dtype)
                proprio = proprio * (1.0 - drop)[:, None]
            result["proprio"] = proprio

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
                use_g1_expert=getattr(model_cfg, "use_g1_expert", False),
                g1_dim=getattr(model_cfg, "g1_dim", 31),
                g1_loss_weight=getattr(model_cfg, "g1_loss_weight", 0.3),
                use_proprio=getattr(model_cfg, "use_proprio", False),
                proprio_dim=getattr(model_cfg, "proprio_dim", 64),
                use_rtc=getattr(model_cfg, "use_rtc", False),
                rtc_on_sonic=getattr(model_cfg, "rtc_on_sonic", True),
            )

        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.vlm_processor = model_wrapper.processor
        self.tokenizer = model_wrapper.tokenizer

        if model_cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def load_pretrained_weights(self) -> Optional[str]:
        """Hook the post-train init/freeze sequence onto pretrained loading.

        Order: parent loads SONIC stage 1-2 weights with ``strict=False`` so
        the new G1 / proprio副本 stay at their random init. We then copy
        SONIC into G1 (``init_g1_from_sonic``) before freezing — this happens
        only when both flags are on, so the legacy pretrain path is unchanged.
        """
        path = super().load_pretrained_weights()
        model_cfg = self.cfg.model
        if path is None:
            return None
        if getattr(model_cfg, "init_g1_from_sonic", False):
            self.model_wrapper.init_g1_from_sonic_expert()
        if getattr(model_cfg, "freeze_for_post_train", False):
            self.model_wrapper.freeze_for_post_train()
        return path

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
        model_cfg = self.cfg.model
        collator = KappaCollator(
            processor=self.vlm_processor,
            model_max_length=model_cfg.model_max_length,
            use_g1_expert=getattr(model_cfg, "use_g1_expert", False),
            use_proprio=getattr(model_cfg, "use_proprio", False),
            use_rtc=getattr(model_cfg, "use_rtc", False),
            rtc_on_sonic=getattr(model_cfg, "rtc_on_sonic", True),
            g1_max_delay=getattr(model_cfg, "g1_max_delay", 10),
            sonic_max_delay=getattr(model_cfg, "sonic_max_delay", 8),
            proprio_noise_std=getattr(model_cfg, "proprio_noise_std", 0.02),
            proprio_drop_prob=getattr(model_cfg, "proprio_drop_prob", 0.1),
            proprio_dim=getattr(model_cfg, "proprio_dim", 64),
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

        # Push the per-worker frame cache size before workers are forked so
        # every worker inherits the configured value via copy-on-write.
        from src.data.video_reader import set_frame_cache_size
        set_frame_cache_size(self.train_cfg.frame_cache_size)

        block = self.train_cfg.block_shuffle_size
        sampler = None
        shuffle_flag: bool | None = True
        if block > 0:
            raw = getattr(train_dataset, "raw_dataset", train_dataset)
            spans_fn = getattr(raw, "episode_spans", None)
            if spans_fn is None:
                overwatch.warning(
                    f"block_shuffle_size={block} requested but {type(raw).__name__} "
                    "has no episode_spans(); falling back to shuffle=True."
                )
            else:
                from src.data.sampler import BlockShuffleSampler

                sampler = BlockShuffleSampler(
                    episode_spans=spans_fn(),
                    block_size=block,
                    generator=g,
                )
                shuffle_flag = None
                overwatch.info(
                    f"BlockShuffleSampler enabled: block_size={block}, "
                    f"episodes={len(spans_fn())}, frame_cache_size="
                    f"{self.train_cfg.frame_cache_size}"
                )

        train_kwargs = dict(loader_kwargs)
        if sampler is not None:
            train_kwargs["sampler"] = sampler
        else:
            train_kwargs["shuffle"] = shuffle_flag
            train_kwargs["generator"] = g

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.train_cfg.train_batch_size,
            drop_last=True,
            **train_kwargs,
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
            else:
                # DeepSpeed engine.step() (run inside accelerator.backward())
                # handles optimizer.step + grad clip + zero_grad, but only
                # advances the LR scheduler if it's in DeepSpeed's
                # VALID_LR_SCHEDULES — our HF cosine LambdaLR isn't, so
                # accelerate's `DeepSpeedSchedulerWrapper.step()` is a no-op
                # and engine.lr_scheduler is None. Step the inner LambdaLR
                # ourselves; loop world_size times to match
                # AcceleratedScheduler's non-DS path (which compensates for
                # the world_size-pre-scaling in `Trainer.create_scheduler`).
                inner_scheduler = self.lr_scheduler.scheduler
                for _ in range(self.world_size):
                    inner_scheduler.step()
            if timer:
                timer.stop("optimizer")

        log_metrics: dict[str, Any] = {
            "loss": loss.item(),
            "lr": self.lr,
        }
        # Surface per-expert losses for the post-train run.
        if hasattr(output, "sonic_loss") and output.sonic_loss is not None:
            log_metrics["sonic_loss"] = output.sonic_loss.item()
        if hasattr(output, "g1_loss") and output.g1_loss is not None:
            log_metrics["g1_loss"] = output.g1_loss.item()
        return self.accelerator.sync_gradients, log_metrics

    def create_optimizer(self):
        """Create optimizer with parameter groups for vision, projector, and LLM.

        Post-train adds a separate ``g1`` group so the design-doc's "give the
        new expert its own lr knob" stays available — by default it shares
        the LLM lr; configs can override via ``model_cfg.g1_lr``.
        """
        from torch.optim import AdamW

        model_cfg = self.cfg.model
        train_cfg = self.train_cfg

        param_groups = []

        projector_params = []
        llm_params = []
        g1_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # ``proprio_embedder`` is a latent-prefix Linear, no longer part of
            # the G1 expert — let it follow the default LR rather than g1_lr.
            if "_g1" in name:
                g1_params.append(param)
            elif "mlp1" in name or "projector" in name or "merger" in name:
                projector_params.append(param)
            elif "lm_head" in name or "embed_tokens" in name or "layers" in name or "norm" in name:
                llm_params.append(param)
            else:
                other_params.append(param)

        optimizer_kwargs = dict(train_cfg.optimizer_kwargs)
        g1_lr = getattr(model_cfg, "g1_lr", train_cfg.learning_rate)

        if len(projector_params) > 0:
            param_groups.append({"params": projector_params, "lr": model_cfg.mm_projector_lr, "weight_decay": model_cfg.weight_decay})
        if len(llm_params) > 0:
            param_groups.append({"params": llm_params, "lr": train_cfg.learning_rate, "weight_decay": model_cfg.weight_decay})
        if len(g1_params) > 0:
            param_groups.append({"params": g1_params, "lr": g1_lr, "weight_decay": model_cfg.weight_decay})
        if len(other_params) > 0:
            param_groups.append({"params": other_params, "lr": train_cfg.learning_rate, "weight_decay": model_cfg.weight_decay})

        self.optimizer = AdamW(param_groups, **optimizer_kwargs)

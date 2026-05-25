"""Synchronous Kappa policy adapter for offline validation (SONIC tokens).

Wraps the KappaModelWrapper + MoT conversion behind a single
``predict_chunk(image, text, proprio, prev_sonic, delay_sonic)`` method that
returns the predicted SONIC token chunk. Mirrors the inference path used by
:mod:`src.inference.kappa_rtc_controller` minus the threading model — for
offline rollout we step the policy synchronously inside the eval loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class KappaPolicyConfig:
    """Inputs the adapter needs to load weights and configure inference."""

    weights: str
    backbone: str = "resource/model/Qwen3-VL-2B-Instruct"
    # Optional override. When ``None``, ``_load_stats`` discovers
    # ``proprio_stats.json`` bundled next to ``weights`` (written by Trainer
    # at fine-tune save time), with the legacy HE+WBT global as last fallback.
    proprio_stats: Optional[str] = None
    # Must match the training-time ``data.normalize_mode``. ``bounds_q99`` reads
    # q01/q99 (the default Kappa training pipeline); ``bounds`` reads min/max.
    normalize_mode: str = "bounds_q99"
    use_proprio: bool = True
    use_rtc: bool = True
    num_inference_steps: int = 4
    device: str = "cuda:0"
    dtype: str = "bfloat16"


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }[name]


class KappaPolicy:
    """Self-contained synchronous Kappa MoT policy.

    The first call to ``predict_chunk`` is slower (CUDA / ORT autotune); the
    caller can warm it up by issuing one or two dummy predictions.
    """

    def __init__(self, cfg: KappaPolicyConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(cfg.dtype, self.device)

        self._build_model()
        self._build_scheduler()
        self._load_stats()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_model(self) -> None:
        from src.config.model_kappa import KappaModelConfig
        from src.models.kappa_model import KappaModelWrapper
        from safetensors.torch import load_file

        cfg = self.cfg
        mcfg = KappaModelConfig(
            model_name_or_path=cfg.backbone,
            use_proprio=cfg.use_proprio,
            use_rtc=cfg.use_rtc,
            num_inference_steps=cfg.num_inference_steps,
        )
        self.mcfg = mcfg

        wrapper = KappaModelWrapper(
            model_name_or_path=mcfg.model_name_or_path,
            tune_vision=False,
            tune_llm=False,
            tune_mlp=False,
            device=self.device,
        )
        wrapper.convert_to_mot(
            sonic_dim=mcfg.sonic_dim,
            action_loss_weight=mcfg.action_loss_weight,
            use_proprio=cfg.use_proprio,
            proprio_dim=mcfg.proprio_dim,
            use_rtc=cfg.use_rtc,
        )

        state = load_file(cfg.weights)
        result = wrapper.model.load_state_dict(state, strict=False)
        print(
            f"[kappa-policy] loaded {len(state)} tensors; "
            f"missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}"
        )
        wrapper.model.eval()
        self.wrapper = wrapper
        self.model = wrapper.model
        self.processor = wrapper.processor

    def _build_scheduler(self) -> None:
        from diffusers import FlowMatchEulerDiscreteScheduler
        n = getattr(self.mcfg, "num_train_timesteps", 1000)
        self._scheduler_sonic = FlowMatchEulerDiscreteScheduler(num_train_timesteps=n)

    def _load_stats(self) -> None:
        from src.data.tokenized_video_dataset import _load_normalize_bounds
        from src.inference.proprio_stats_loader import discover_proprio_stats

        resolved_path, parsed = discover_proprio_stats(
            self.cfg.weights, explicit_path=self.cfg.proprio_stats,
        )
        out = _load_normalize_bounds(parsed, "proprio", 64, self.cfg.normalize_mode)
        if out is None:
            raise ValueError(
                f"proprio stats JSON at {resolved_path!r} has no 'proprio' section"
            )
        self.proprio_min, hi_safe, _ = out
        self.proprio_max = hi_safe
        print(f"[kappa-policy] loaded proprio bounds ({self.cfg.normalize_mode}) from {resolved_path}")

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def normalize_proprio(self, proprio_64: np.ndarray) -> np.ndarray:
        """Map raw 64-D proprio to [-1, 1] using the cached bounds.

        Saturates at ±1 to match ``_normalize_to_pm1`` in
        :mod:`src.data.tokenized_video_dataset` — the policy never saw
        magnitudes >1 during training, so passing through OOD values just
        feeds the model a distribution it can't interpret.
        """
        rng = self.proprio_max - self.proprio_min
        out = (proprio_64.astype(np.float32) - self.proprio_min) / rng * 2.0 - 1.0
        return np.clip(out, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Tokenize
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_inputs(self, image: Image.Image, text: str) -> dict[str, torch.Tensor]:
        from qwen_vl_utils import process_vision_info

        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]]
        rendered = [
            self.processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False,
            )
            for conv in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=rendered, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
            "pixel_values": inputs["pixel_values"].to(self.device, dtype=self.dtype),
            "image_grid_thw": inputs["image_grid_thw"].to(self.device),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_chunk(
        self,
        image: Image.Image,
        text: str,
        proprio_64_raw: Optional[np.ndarray] = None,
        prev_sonic: Optional[np.ndarray] = None,
        delay_sonic: int = 0,
    ) -> np.ndarray:
        """Predict one ``(action_chunk_size, sonic_dim)`` SONIC token chunk.

        Pass back the returned chunk as ``prev_sonic`` next call to RTC-pin
        the SONIC prefix.
        """
        prompt = self._build_inputs(image, text)
        H = self.mcfg.action_chunk_size
        T_p = 1 if self.cfg.use_proprio else 0

        device = self.device
        if self.cfg.use_proprio:
            if proprio_64_raw is None:
                proprio_t = torch.zeros(1, self.mcfg.proprio_dim, device=device, dtype=self.dtype)
            else:
                p = self.normalize_proprio(proprio_64_raw)
                proprio_t = torch.from_numpy(p).to(device=device, dtype=self.dtype).unsqueeze(0)
        else:
            proprio_t = None

        cache = self.model.encode_latent(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            pixel_values=prompt["pixel_values"],
            image_grid_thw=prompt["image_grid_thw"],
            T_action=H,
            T_proprio=T_p,
            proprio=proprio_t,
        )
        B = cache["B"]
        dtype = cache["dtype"]

        sonic = torch.randn(B, H, self.mcfg.sonic_dim, device=device, dtype=dtype)

        if self.cfg.use_rtc and prev_sonic is not None:
            prev_sonic_t = (
                torch.from_numpy(prev_sonic.astype(np.float32))
                .to(device=device, dtype=dtype)
                .unsqueeze(0)
            )
            delay_sonic_t = torch.tensor([int(delay_sonic)], device=device, dtype=torch.long)
        else:
            prev_sonic_t = None
            delay_sonic_t = None

        self._scheduler_sonic.set_timesteps(self.cfg.num_inference_steps, device=device)
        n_train = self._scheduler_sonic.config.num_train_timesteps
        for timestep in self._scheduler_sonic.timesteps:
            t_scalar = (timestep / n_train).to(dtype)
            t_sonic = t_scalar.expand(B)

            if prev_sonic_t is not None and delay_sonic_t is not None:
                arange = torch.arange(H, device=device)
                prefix_mask_s = arange[None, :] < delay_sonic_t[:, None]
                sonic = torch.where(prefix_mask_s[..., None], prev_sonic_t, sonic)

            v_sonic = self.model.denoise_step_rtc(
                cache=cache,
                noisy_sonic_tokens=sonic,
                timesteps_sonic=t_sonic,
                prev_sonic=prev_sonic_t,
                delay_sonic=delay_sonic_t,
            )
            sonic = self._scheduler_sonic.step(v_sonic, timestep, sonic).prev_sample

        if prev_sonic_t is not None and delay_sonic_t is not None:
            arange = torch.arange(H, device=device)
            prefix_mask_s = arange[None, :] < delay_sonic_t[:, None]
            sonic = torch.where(prefix_mask_s[..., None], prev_sonic_t, sonic)

        return sonic[0].float().cpu().numpy()

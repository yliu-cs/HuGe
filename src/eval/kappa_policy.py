"""Synchronous Kappa post-train policy adapter for offline validation.

Wraps the KappaModelWrapper + MoT post-train conversion behind a single
``predict_chunk(image, text, proprio, prev_g1, delay)`` method that returns
the de-normalized 23-D G1 action chunk (arms[14] + hand_bits[2] +
loco_cmd/height/rpy[7]). Mirrors the inference path used by
:mod:`src.inference.kappa_rtc_controller` minus the threading model — for
offline rollout we step the policy synchronously inside the eval loop.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


@dataclass
class KappaPolicyConfig:
    """Inputs the adapter needs to load weights and configure inference."""

    weights: str
    backbone: str = "resource/model/Qwen3-VL-2B-Instruct"
    action_stats: str = "resource/dataset/G1_WBT-cache/meta/g1_action_stats.json"
    proprio_stats: str = "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json"
    # Must match the training-time ``data.normalize_mode``. ``bounds_q99`` reads
    # q01/q99 (the default Kappa training pipeline); ``bounds`` reads min/max.
    # A mismatch silently rescales the policy's denormalized action by the
    # (max-min) / (q99-q01) ratio, which on G1 is ~2x — actions blow past the
    # WBC's training range without warning.
    normalize_mode: str = "bounds_q99"
    use_proprio: bool = True
    use_rtc: bool = True
    rtc_on_sonic: bool = True
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
            use_g1_expert=True,
            use_proprio=cfg.use_proprio,
            use_rtc=cfg.use_rtc,
            rtc_on_sonic=cfg.rtc_on_sonic,
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
            use_g1_expert=True,
            g1_dim=mcfg.g1_dim,
            g1_loss_weight=mcfg.g1_loss_weight,
            use_proprio=cfg.use_proprio,
            proprio_dim=mcfg.proprio_dim,
            use_rtc=cfg.use_rtc,
            rtc_on_sonic=cfg.rtc_on_sonic,
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
        # Two scheduler instances because diffusers' Euler scheduler keeps an
        # internal ``_step_index`` counter that advances on every ``step``
        # call. Sonic and G1 are denoised in lockstep but each consume the
        # scheduler's step API once per ODE iteration, so they need separate
        # state.
        n = getattr(self.mcfg, "num_train_timesteps", 1000)
        self._scheduler_sonic = FlowMatchEulerDiscreteScheduler(num_train_timesteps=n)
        self._scheduler_g1 = FlowMatchEulerDiscreteScheduler(num_train_timesteps=n)

    def _load_stats(self) -> None:
        with open(self.cfg.action_stats) as f:
            astats = json.load(f).get("action", {})
        with open(self.cfg.proprio_stats) as f:
            pstats = json.load(f).get("proprio", {})

        mode = self.cfg.normalize_mode
        if mode == "bounds":
            lo_key, hi_key = "min", "max"
        elif mode == "bounds_q99":
            lo_key, hi_key = "q01", "q99"
        else:
            raise ValueError(f"unknown normalize_mode {mode!r}")

        def _bounds(d: dict) -> tuple[np.ndarray, np.ndarray]:
            lo = np.asarray(d[lo_key], dtype=np.float32)
            hi = np.asarray(d[hi_key], dtype=np.float32)
            ill = np.abs(hi - lo) < 1e-4 * (np.abs(hi) + np.abs(lo) + 1e-8)
            hi = np.where(ill, lo + 1.0, hi)
            return lo, hi

        self.action_min, self.action_max = _bounds(astats)
        self.proprio_min, self.proprio_max = _bounds(pstats)

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

    def denormalize_action(self, action_norm: np.ndarray) -> np.ndarray:
        """Map a 23-D normalized G1 action back to raw command space.

        Layout: arms(14, rad) + hand_bits(2, [0,1]) + loco_cmd(3, m/s & rad/s)
        + base_height(1, m) + rpy_cmd(3, rad).
        """
        rng = self.action_max - self.action_min
        return ((action_norm.astype(np.float32) + 1.0) / 2.0) * rng + self.action_min

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
        prev_g1_norm: Optional[np.ndarray] = None,
        delay: int = 0,
        prev_sonic: Optional[np.ndarray] = None,
        delay_sonic: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict one (action_chunk_size, *) action chunk.

        Returns ``(g1_norm, g1_denorm, sonic_chunk)``:

        - ``g1_norm`` (H, 23) — G1-expert action chunk in [-1, 1] space.
          Pass back as ``prev_g1_norm`` next call to enable RTC.
        - ``g1_denorm`` (H, 23) — same chunk de-normalized: arms[14] +
          hand_bits[2] + loco_cmd[3] + base_height[1] + rpy_cmd[3], in the
          raw command units the WBC step API consumes.
        - ``sonic_chunk`` (H, sonic_dim) — predicted SONIC tokens. Layout is
          ``[64-D SONIC latent | 2-D hand bits]``. Pass back as
          ``prev_sonic`` to RTC-pin the SONIC prefix.
        """
        prompt = self._build_inputs(image, text)
        H = self.mcfg.action_chunk_size
        T_p = 1 if self.cfg.use_proprio else 0

        # Build proprio first because it now feeds into ``encode_latent``
        # (proprio is part of the latent prefix and gets baked into the KV
        # cache once, instead of being re-embedded per denoise step).
        device = self.device
        # ``B`` is known to be 1 for offline rollouts; we expand later if the
        # underlying model is batched.
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
            T_g1=H,
            T_proprio=T_p,
            proprio=proprio_t,
        )
        B = cache["B"]
        dtype = cache["dtype"]

        sonic = torch.randn(B, H, self.mcfg.sonic_dim, device=device, dtype=dtype)
        g1 = torch.randn(B, H, self.mcfg.g1_dim, device=device, dtype=dtype)

        if self.cfg.use_rtc and prev_g1_norm is not None:
            prev_g1_t = (
                torch.from_numpy(prev_g1_norm.astype(np.float32))
                .to(device=device, dtype=dtype)
                .unsqueeze(0)
            )
            delay_t = torch.tensor([int(delay)], device=device, dtype=torch.long)
        else:
            prev_g1_t = None
            delay_t = None

        if self.cfg.use_rtc and self.cfg.rtc_on_sonic and prev_sonic is not None:
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
        self._scheduler_g1.set_timesteps(self.cfg.num_inference_steps, device=device)
        n_train = self._scheduler_sonic.config.num_train_timesteps
        for timestep in self._scheduler_sonic.timesteps:
            t_scalar = (timestep / n_train).to(dtype)
            t_sonic = t_scalar.expand(B)
            t_g1 = t_scalar.expand(B)

            if prev_g1_t is not None and delay_t is not None:
                # Hard-pin the prefix at the start of every step so
                # ``scheduler.step`` cannot drift it.
                arange = torch.arange(H, device=device)
                prefix_mask = arange[None, :] < delay_t[:, None]
                g1 = torch.where(prefix_mask[..., None], prev_g1_t, g1)
            if prev_sonic_t is not None and delay_sonic_t is not None:
                arange = torch.arange(H, device=device)
                prefix_mask_s = arange[None, :] < delay_sonic_t[:, None]
                sonic = torch.where(prefix_mask_s[..., None], prev_sonic_t, sonic)

            v_sonic, v_g1 = self.model.denoise_step_rtc(
                cache=cache,
                noisy_sonic_tokens=sonic,
                timesteps_sonic=t_sonic,
                noisy_g1_tokens=g1,
                timesteps_g1=t_g1,
                prev_g1=prev_g1_t,
                delay_g1=delay_t,
                prev_sonic=prev_sonic_t,
                delay_sonic=delay_sonic_t,
            )
            sonic = self._scheduler_sonic.step(v_sonic, timestep, sonic).prev_sample
            g1 = self._scheduler_g1.step(v_g1, timestep, g1).prev_sample

        if prev_g1_t is not None and delay_t is not None:
            arange = torch.arange(H, device=device)
            prefix_mask = arange[None, :] < delay_t[:, None]
            g1 = torch.where(prefix_mask[..., None], prev_g1_t, g1)
        if prev_sonic_t is not None and delay_sonic_t is not None:
            arange = torch.arange(H, device=device)
            prefix_mask_s = arange[None, :] < delay_sonic_t[:, None]
            sonic = torch.where(prefix_mask_s[..., None], prev_sonic_t, sonic)

        actions_norm = g1[0].float().cpu().numpy()  # (H, g1_dim=23)
        actions_denorm = np.stack([self.denormalize_action(a) for a in actions_norm])
        sonic_chunk = sonic[0].float().cpu().numpy()  # (H, sonic_dim)
        return actions_norm, actions_denorm, sonic_chunk

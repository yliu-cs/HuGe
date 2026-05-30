"""Policy wrapper that normalizes QwenSonicFM outputs for deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from kappa.data.schemas import ACTION_DIM, PROPRIO_DIM, split_action_dict
from kappa.inference.observation import assemble_proprio
from kappa.models.qwen_sonicfm import QwenSonicFM
from kappa.data.state_normalizer import StateNormalizer


def _truncate_with_tokenizer(tokenizer: Any, text: str, max_tokens: int | None) -> str:
    if max_tokens is None or max_tokens <= 0:
        return text
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
    return tokenizer.decode(ids, skip_special_tokens=True)


class KappaPolicy:
    def __init__(
        self,
        model: QwenSonicFM,
        *,
        device: str = "cpu",
        qwen_processor_path: str | Path | None = None,
        normalizer: StateNormalizer | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = torch.device(device)
        self.normalizer = normalizer
        self.processor = None
        self.tokenizer = None
        if self.model.config.use_qwen:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                qwen_processor_path or self.model.config.base_vlm_path,
                local_files_only=True,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def _state_tensor(self, observation: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        arr, mask = assemble_proprio(observation)
        if arr.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"state dim {arr.shape[-1]} != {PROPRIO_DIM}")
        if self.normalizer is not None and bool(mask.max(initial=0.0) > 0.0):
            arr = self.normalizer.normalize_flat(arr)
        return torch.from_numpy(arr).to(self.device), torch.from_numpy(mask).to(self.device)

    @staticmethod
    def _prompt(observation: dict[str, Any]) -> str:
        lang = observation.get("language", observation.get("task", "perform the task"))
        if isinstance(lang, dict):
            for key in ("annotation.human.task_description", "task"):
                if key not in lang:
                    continue
                value = lang[key]
                if isinstance(value, (list, tuple)) and value:
                    first = value[0]
                    if isinstance(first, (list, tuple)) and first:
                        return str(first[0])
                    return str(first)
                return str(value)
            if lang:
                value = next(iter(lang.values()))
                if isinstance(value, (list, tuple)) and value:
                    first = value[0]
                    if isinstance(first, (list, tuple)) and first:
                        return str(first[0])
                    return str(first)
                return str(value)
        return str(lang or "perform the task")

    def _input_ids(self, prompt: str) -> torch.Tensor:
        text = str(prompt)
        max_len = int(getattr(self.model.config, "char_max_length", 256) or 256)
        ids = [min(ord(ch), 32767) for ch in text[:max_len]] or [0]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @staticmethod
    def _image_array(observation: dict[str, Any]) -> np.ndarray | None:
        frame = observation.get("video.ego_view", observation.get("image"))
        if frame is None and isinstance(observation.get("video"), dict):
            video = observation["video"]
            frame = video.get("ego_view")
            if frame is None and video:
                frame = next(iter(video.values()))
        if frame is None:
            return None
        arr = np.asarray(frame)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"expected image shape (..., H, W, 3), got {arr.shape}")
        return arr

    def _image_tensor(self, observation: dict[str, Any]) -> torch.Tensor | None:
        arr_u8 = self._image_array(observation)
        if arr_u8 is None:
            return None
        arr = np.asarray(arr_u8, dtype=np.float32)
        if arr.max(initial=0.0) > 1.5:
            arr = arr / 255.0
        return torch.from_numpy(arr[None]).to(self.device)

    def _qwen_inputs(self, observation: dict[str, Any], prompt: str) -> dict[str, torch.Tensor] | None:
        if self.processor is None:
            raise ValueError("Qwen processor is required; fallback text/image path has been removed")
        arr = self._image_array(observation)
        if arr is None:
            raise ValueError("Qwen processor input requires an image")
        if self.tokenizer is not None:
            prompt = _truncate_with_tokenizer(
                self.tokenizer,
                prompt,
                getattr(self.model.config, "max_text_tokens", None),
            )
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if pil.size != (320, 224):
            pil = pil.resize((320, 224), Image.BILINEAR)
        conv = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info([conv], image_patch_size=16)
        except Exception:
            image_inputs, video_inputs = [pil], None
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_max_length = int(getattr(self.model.config, "model_max_length", 8192) or 8192)
        out = dict(inputs)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in out:
                out[key] = out[key][:, :model_max_length]
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in out.items()}

    @torch.no_grad()
    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        prompt = self._prompt(observation)
        state, state_mask = self._state_tensor(observation)
        action = self.model.predict_action(
            input_ids=self._input_ids(prompt),
            images=self._image_tensor(observation),
            state=state,
            state_mask=state_mask,
            qwen_inputs=self._qwen_inputs(observation, prompt),
        )
        arr = action.detach().cpu().numpy().astype(np.float32)
        if arr.shape[-1] != ACTION_DIM:
            raise ValueError(f"model returned action dim {arr.shape[-1]} != {ACTION_DIM}")
        parts = split_action_dict(arr)
        return {
            "action.motion_token": parts["action.motion_token"],
            "action.hand_binary": parts["action.hand_binary"],
            "motion_token": parts["action.motion_token"],
            "hand_binary_action": parts["action.hand_binary"],
        }

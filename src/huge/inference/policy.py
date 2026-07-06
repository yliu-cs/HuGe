"""Policy wrapper that normalizes QwenSonicFM outputs for deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from huge.data.schemas import ACTION_DIM, PROPRIO_DIM, split_action_dict
from huge.inference.observation import assemble_proprio
from huge.models.qwen_sonicfm import QwenSonicFM
from huge.data.state_normalizer import StateNormalizer
from huge.data.transforms import gr00t_image_transform


@dataclass(frozen=True)
class PolicyTrace:
    prompt: str
    input_ids: torch.Tensor
    image: torch.Tensor | None
    state_raw: torch.Tensor
    state: torch.Tensor
    state_mask: torch.Tensor
    qwen_inputs: dict[str, torch.Tensor] | None
    action_token_mask: torch.Tensor | None
    qwen_text: str | None
    action_array: np.ndarray


def _truncate_with_tokenizer(tokenizer: Any, text: str, max_tokens: int | None) -> str:
    if max_tokens is None or max_tokens <= 0:
        return text
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
    return tokenizer.decode(ids, skip_special_tokens=True)


class HugePolicy:
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
        self.model_architecture = str(getattr(self.model.config, "model_architecture", "qwenpi_v3") or "qwenpi_v3").lower()
        self.oft_action_token_id: int | None = None
        if self.model.config.use_qwen:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                qwen_processor_path or self.model.config.base_vlm_path,
                local_files_only=True,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = "left"
            if self.model_architecture == "qwenoft":
                token = getattr(self.model.config, "oft_action_token", "\U0001f50d")
                ids = self.tokenizer(token, add_special_tokens=False).input_ids
                if len(ids) != 1:
                    raise ValueError(f"qwenoft action token {token!r} must tokenize to one id, got {ids}")
                self.oft_action_token_id = int(ids[0])

    def _state_tensor(self, observation: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        arr, mask = assemble_proprio(observation)
        if arr.shape[-1] != PROPRIO_DIM:
            raise ValueError(f"state dim {arr.shape[-1]} != {PROPRIO_DIM}")
        raw = arr.copy()
        if self.normalizer is not None and bool(mask.max(initial=0.0) > 0.0):
            arr = self.normalizer.normalize_flat(arr)
        return torch.from_numpy(raw).to(self.device), torch.from_numpy(arr).to(self.device), torch.from_numpy(mask).to(self.device)

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

    def _append_oft_action_tokens(self, prompt: str) -> str:
        token = getattr(self.model.config, "oft_action_token", "\U0001f50d")
        horizon = int(getattr(self.model.config, "action_horizon", 50) or 50)
        return f"{prompt} Please predict the next {horizon} robot actions:{token * horizon}"

    def _qwen_inputs(self, observation: dict[str, Any], prompt: str) -> tuple[dict[str, torch.Tensor] | None, torch.Tensor | None, str | None]:
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
        if self.model_architecture == "qwenoft":
            prompt = self._append_oft_action_tokens(prompt)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        target_size = tuple(getattr(self.model.config, "qwen_image_size", (256, 256)) or (256, 256))
        crop_fraction = float(getattr(self.model.config, "image_crop_fraction", 0.95) or 0.95)
        pil = gr00t_image_transform(
            pil,
            train=False,
            target_size=target_size,  # type: ignore[arg-type]
            crop_fraction=crop_fraction,
            color_jitter=False,
        )
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
                out[key] = (
                    out[key][:, -model_max_length :]
                    if self.model_architecture == "qwenoft"
                    else out[key][:, :model_max_length]
                )
        action_token_mask = None
        if self.model_architecture == "qwenoft":
            if self.oft_action_token_id is None:
                raise ValueError("qwenoft requires a resolved action token id")
            action_token_mask = out["input_ids"] == self.oft_action_token_id
            horizon = int(getattr(self.model.config, "action_horizon", 50) or 50)
            count = int(action_token_mask.sum().item())
            if count < horizon:
                raise ValueError(f"qwenoft prompt has {count} action tokens after truncation, expected at least {horizon}")
            action_token_mask = action_token_mask.to(self.device)
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in out.items()}, action_token_mask, text

    def _prepare_model_inputs(self, observation: dict[str, Any]) -> dict[str, Any]:
        prompt = self._prompt(observation)
        state_raw, state, state_mask = self._state_tensor(observation)
        qwen_inputs, action_token_mask, qwen_text = self._qwen_inputs(observation, prompt)
        return {
            "prompt": prompt,
            "input_ids": self._input_ids(prompt),
            "image": self._image_tensor(observation),
            "state_raw": state_raw,
            "state": state,
            "state_mask": state_mask,
            "qwen_inputs": qwen_inputs,
            "action_token_mask": action_token_mask,
            "qwen_text": qwen_text,
        }

    @torch.no_grad()
    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        action, _ = self.get_action_with_trace(observation)
        return action

    @torch.no_grad()
    def get_action_with_trace(self, observation: dict[str, Any]) -> tuple[dict[str, np.ndarray], PolicyTrace]:
        inputs = self._prepare_model_inputs(observation)
        action = self.model.predict_action(
            input_ids=inputs["input_ids"],
            images=inputs["image"],
            state=inputs["state"],
            state_mask=inputs["state_mask"],
            qwen_inputs=inputs["qwen_inputs"],
            action_token_mask=inputs["action_token_mask"],
        )
        arr = action.detach().cpu().numpy().astype(np.float32)
        if arr.shape[-1] != ACTION_DIM:
            raise ValueError(f"model returned action dim {arr.shape[-1]} != {ACTION_DIM}")
        parts = split_action_dict(arr)
        action_dict = {
            "action.motion_token": parts["action.motion_token"],
            "action.hand_binary": parts["action.hand_binary"],
            "motion_token": parts["action.motion_token"],
            "hand_binary_action": parts["action.hand_binary"],
        }
        trace = PolicyTrace(
            prompt=inputs["prompt"],
            input_ids=inputs["input_ids"],
            image=inputs["image"],
            state_raw=inputs["state_raw"],
            state=inputs["state"],
            state_mask=inputs["state_mask"],
            qwen_inputs=inputs["qwen_inputs"],
            action_token_mask=inputs["action_token_mask"],
            qwen_text=inputs["qwen_text"],
            action_array=arr,
        )
        return action_dict, trace

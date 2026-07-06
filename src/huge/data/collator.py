"""Batch collation for QwenSonicFM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from huge.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM
from huge.data.transforms import gr00t_image_transform
from huge.data.video_reader import decode_video_frame, image_to_float_array


def _truncate_with_tokenizer(tokenizer: Any, text: str, max_tokens: int | None) -> str:
    if max_tokens is None or max_tokens <= 0:
        return text
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
    return tokenizer.decode(ids, skip_special_tokens=True)


class HugeCollator:
    """Collate dataset dicts into tensors.

    With ``use_qwen_processor=True`` the batch includes Qwen processor tensors
    under ``qwen_inputs``. ``False`` is only for data/loss smoke debugging.
    """

    def __init__(
        self,
        *,
        qwen_processor_path: str | Path = "resource/model/Qwen3-VL-2B-Instruct",
        use_qwen_processor: bool = False,
        image_size: tuple[int, int] = (224, 320),
        qwen_image_size: tuple[int, int] = (256, 256),
        image_transform_mode: str = "train",
        image_crop_fraction: float = 0.95,
        image_color_jitter: bool = True,
        model_max_length: int = 8192,
        max_text_tokens: int | None = None,
        char_max_length: int = 256,
        model_architecture: str = "qwenpi_v3",
        action_horizon: int = ACTION_HORIZON,
        oft_action_token: str = "\U0001f50d",
    ) -> None:
        self.use_qwen_processor = use_qwen_processor
        self.image_size = image_size
        self.qwen_image_size = qwen_image_size
        self.image_transform_mode = image_transform_mode
        self.image_crop_fraction = image_crop_fraction
        self.image_color_jitter = image_color_jitter
        self.model_max_length = model_max_length
        self.max_text_tokens = max_text_tokens
        self.char_max_length = char_max_length
        self.model_architecture = str(model_architecture or "qwenpi_v3").lower()
        self.action_horizon = int(action_horizon)
        self.oft_action_token = oft_action_token
        self.oft_action_token_id: int | None = None
        self.processor = None
        self.tokenizer = None
        if use_qwen_processor:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(qwen_processor_path, local_files_only=True)
            self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = "left"
            if self.model_architecture == "qwenoft":
                ids = self.tokenizer(self.oft_action_token, add_special_tokens=False).input_ids
                if len(ids) != 1:
                    raise ValueError(
                        f"qwenoft action token {self.oft_action_token!r} must tokenize to one id, got {ids}"
                    )
                self.oft_action_token_id = int(ids[0])

    def _image(self, item: dict[str, Any]) -> Image.Image:
        image = item.get("image")
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image.astype(np.uint8))
        return decode_video_frame(item.get("video_path"), int(item.get("video_frame_idx") or 0), output_size=self.image_size)

    def _qwen_image(self, image: Image.Image) -> Image.Image:
        mode = self.image_transform_mode.lower()
        if mode in {"none", "off", "disabled"}:
            return image.convert("RGB")
        if mode not in {"train", "eval"}:
            raise ValueError(f"unknown image_transform_mode {self.image_transform_mode!r}")
        return gr00t_image_transform(
            image,
            train=mode == "train",
            target_size=self.qwen_image_size,
            crop_fraction=self.image_crop_fraction,
            color_jitter=self.image_color_jitter,
        )

    def _append_oft_action_tokens(self, text: str) -> str:
        suffix = self.oft_action_token * self.action_horizon
        return f"{text} Please predict the next {self.action_horizon} robot actions:{suffix}"

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(x.get("language") or x.get("text") or "perform the task") for x in items]
        if self.tokenizer is not None:
            texts = [_truncate_with_tokenizer(self.tokenizer, text, self.max_text_tokens) for text in texts]
        qwen_texts = [self._append_oft_action_tokens(text) for text in texts] if self.model_architecture == "qwenoft" else texts
        images_pil = [self._qwen_image(self._image(x)) for x in items]
        images = torch.from_numpy(np.stack([image_to_float_array(im) for im in images_pil], axis=0))
        actions = torch.from_numpy(np.stack([np.asarray(x["normalized_actions"], dtype=np.float32) for x in items], axis=0))
        sonic_mask = torch.from_numpy(np.stack([np.asarray(x["sonic_mask"], dtype=np.float32) for x in items], axis=0))
        state = torch.from_numpy(np.stack([np.asarray(x.get("state", np.zeros(PROPRIO_DIM)), dtype=np.float32) for x in items], axis=0))
        state_mask = torch.from_numpy(np.stack([np.asarray(x.get("state_mask", np.zeros(1)), dtype=np.float32) for x in items], axis=0)).reshape(len(items), -1)
        if actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"action dim {actions.shape[-1]} != {ACTION_DIM}")
        if self.processor is None:
            raise ValueError("Huge training requires use_qwen_processor=True")
        messages = [
            [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": text}]}]
            for im, text in zip(images_pil, qwen_texts, strict=True)
        ]
        rendered = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        qwen_inputs = self.processor(
            text=rendered,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in qwen_inputs:
                qwen_inputs[key] = (
                    qwen_inputs[key][:, -self.model_max_length :]
                    if self.model_architecture == "qwenoft"
                    else qwen_inputs[key][:, : self.model_max_length]
                )
        batch: dict[str, Any] = {
            "input_ids": qwen_inputs["input_ids"],
            "images": images,
            "state": state,
            "state_mask": state_mask,
            "normalized_actions": actions,
            "sonic_mask": sonic_mask,
            "language": texts,
            "dataset_id": [x.get("dataset_id") for x in items],
            "episode_id": [x.get("episode_id") for x in items],
            "frame_idx": [x.get("frame_idx") for x in items],
        }
        batch["qwen_inputs"] = dict(qwen_inputs)
        if self.model_architecture == "qwenoft":
            if self.oft_action_token_id is None:
                raise ValueError("qwenoft requires a resolved action token id")
            action_token_mask = qwen_inputs["input_ids"] == self.oft_action_token_id
            counts = action_token_mask.sum(dim=1)
            if (counts < self.action_horizon).any():
                bad = (counts < self.action_horizon).nonzero(as_tuple=False).flatten().tolist()
                raise ValueError(
                    f"qwenoft batch lost action tokens after tokenization/truncation; bad samples={bad} counts={counts.tolist()}"
                )
            batch["action_token_mask"] = action_token_mask
        return batch

"""Thin local-files-only Qwen3-VL loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Qwen3VLBackbone:
    def __init__(self, model_path: str | Path, *, load_model: bool = False, **model_kwargs: Any) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_path = Path(model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = None
        if load_model:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                local_files_only=True,
                **model_kwargs,
            )

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        return self.processor.apply_chat_template(messages, **kwargs)

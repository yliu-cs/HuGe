"""Deploy-time call logging for real-robot inference."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np
from PIL import Image

m.patch()


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and callable(value.detach):
        try:
            return _to_serializable(value.detach().cpu().numpy())
        except Exception:
            return str(value)
    if isinstance(value, Image.Image):
        return np.asarray(value)
    return value


class DeployCallLogger:
    """Writes one msgpack payload per inference call plus a JSONL manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.calls_dir = self.root / "calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.jsonl"
        self._metadata_path = self.root / "metadata.json"
        if not self._metadata_path.exists():
            self._metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": "kappa_deploy_log_v1",
                        "created_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def log(self, *, call_idx: int, record: dict[str, Any]) -> Path:
        payload = _to_serializable(record)
        call_path = self.calls_dir / f"call_{call_idx:06d}.msgpack"
        call_path.write_bytes(msgpack.packb(payload, use_bin_type=True, default=m.encode))
        preview_name = self._write_preview_image(call_idx, payload)
        summary = self._summary(payload, call_path.name, preview_name)
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        return call_path

    @staticmethod
    def _summary(record: dict[str, Any], filename: str, preview_name: str | None) -> dict[str, Any]:
        model_inputs = record.get("model_inputs", {})
        model_output = record.get("model_output", {})
        observation = record.get("observation", {})
        action = model_output.get("action_array")
        return {
            "time": record.get("time"),
            "call_idx": record.get("call_idx"),
            "filename": filename,
            "prompt": model_inputs.get("prompt"),
            "action_shape": list(np.asarray(action).shape) if action is not None else None,
            "state_shape": list(np.asarray(model_inputs.get("state")).shape) if model_inputs.get("state") is not None else None,
            "state_raw_shape": list(np.asarray(model_inputs.get("state_raw")).shape) if model_inputs.get("state_raw") is not None else None,
            "image_shape": list(np.asarray(model_inputs.get("image")).shape) if model_inputs.get("image") is not None else None,
            "preview_image": preview_name,
            "observation_keys": sorted(observation.keys()) if isinstance(observation, dict) else None,
        }

    def _write_preview_image(self, call_idx: int, record: dict[str, Any]) -> str | None:
        image = record.get("model_inputs", {}).get("image")
        if image is None:
            return None
        arr = np.asarray(image)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.max(initial=0.0) <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        name = f"call_{call_idx:06d}.jpg"
        Image.fromarray(arr).save(self.calls_dir / name, quality=90)
        return name

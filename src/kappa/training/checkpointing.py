"""Checkpoint sidecar metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kappa.data.schemas import ActionSchema, PROPRIO_DIM, PROPRIO_LAYOUT_VERSION


def build_checkpoint_metadata(**overrides: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        **ActionSchema().to_dict(),
        "proprio_layout": PROPRIO_LAYOUT_VERSION,
        "proprio_dim": PROPRIO_DIM,
        "image_transform": "resize_short_224_center_crop_224x320",
        "server_protocol": "kappa_policy_server_msgpack_numpy",
    }
    meta.update({k: v for k, v in overrides.items() if v is not None})
    return meta


def write_sidecar(path: str | Path, metadata: dict[str, Any]) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

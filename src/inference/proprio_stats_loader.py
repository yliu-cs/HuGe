"""Checkpoint-bundled proprio stats discovery for inference / eval consumers.

The fine-tune trainer copies the active dataset's stats JSON into each
``ckpt_<step>/`` directory next to ``model.safetensors`` (see
``src.trainers.trainer.Trainer.save_checkpoint``). Inference-side consumers
should therefore auto-discover stats from the checkpoint directory rather
than carry a hardcoded global default.

Resolution order in :func:`discover_proprio_stats`:

  1. ``explicit_path`` if given AND the file exists — operator override
     (CLI flag, env var) always wins.
  2. ``<dirname(weights_path)>/proprio_stats.json`` — the ckpt-bundled stats
     written by the trainer at fine-tune time.
  3. ``legacy_fallback`` if it exists on disk — keeps older checkpoints
     (which don't bundle stats) working against the global HE+WBT file.

If none of the three resolve, :class:`FileNotFoundError` is raised so the
caller fails loudly instead of silently running uncalibrated.
"""
from __future__ import annotations

import json
import os
from typing import Any

LEGACY_FALLBACK = "resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json"


def discover_proprio_stats(
    weights_path: str,
    explicit_path: str | None = None,
    legacy_fallback: str = LEGACY_FALLBACK,
) -> tuple[str, dict[str, Any]]:
    """Resolve and load the proprio stats JSON for a given checkpoint.

    Returns ``(resolved_path, parsed_json)``. Raises ``FileNotFoundError``
    if no candidate file exists.
    """
    candidates: list[tuple[str, str]] = []
    if explicit_path:
        candidates.append(("explicit", explicit_path))
    if weights_path:
        ckpt_dir = os.path.dirname(os.path.abspath(weights_path))
        candidates.append(("ckpt_dir", os.path.join(ckpt_dir, "proprio_stats.json")))
    if legacy_fallback:
        candidates.append(("legacy_fallback", legacy_fallback))

    for source, path in candidates:
        if path and os.path.isfile(path):
            with open(path) as f:
                parsed = json.load(f)
            return path, parsed

    tried = ", ".join(f"{src}={p!r}" for src, p in candidates) or "<none>"
    raise FileNotFoundError(
        f"could not locate proprio stats JSON. Tried: {tried}. "
        f"Either pass an explicit path or place proprio_stats.json next to "
        f"the model weights."
    )

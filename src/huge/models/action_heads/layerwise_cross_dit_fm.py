"""Compatibility wrapper for the upgraded PI head."""

from __future__ import annotations

from huge.models.action_heads.starvla_pi_head import (
    LayerwiseCrossDiTFMActionHead,
    StarVLAPIActionHead,
)

__all__ = ["StarVLAPIActionHead", "LayerwiseCrossDiTFMActionHead"]

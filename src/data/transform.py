"""Per-sample data transform pipeline (repack -> field -> model).

Mirrors the Psi0 design (`psi.config.transform`) so config files can be tuned
the same way (`--data.transform.model.resize.size 224`, etc.). The pipeline
takes raw dataset output (HE / EgoDex) and emits a normalized dict that
``KappaCollator`` can batch-tokenize.

Stages
------
* **Repack**: source-specific shape normalization. Both HE and EgoDex already
  emit ``{image, text, sonic_tokens, sonic_mask}``, so repack just promotes
  ``image`` to a ``List[PIL.Image]`` (matching Psi0's ``observations`` key) and
  keeps the rest pass-through.
* **Field**: a slot for token-space normalization. SONIC tokens are precomputed
  features that are already statistics-stable, so the default is an identity
  transform — kept as an extension point.
* **Model**: image augmentation. ``Resize(short=224) -> CenterCrop(224, 320)``
  emits landscape 320W×224H frames. 224 and 320 are both multiples of the
  Qwen3-VL ``patch_size × merge_size = 32`` factor, so the patch grid is
  exactly 14×20 = 280 patches → 7×10 = 70 merged vision tokens (vs 64 at 256²). The
  area 224×320=71680 > processor ``shortest_edge=65536`` (an *area* threshold,
  not edge length), so smart_resize is a no-op and the dataloader output is
  consumed as-is.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field


# ---- Augmentation primitives (Psi0-compatible) -----------------------------


class ResizeImage(BaseModel):
    """Resize. ``size=int`` resizes the short side; ``(H, W)`` forces both."""

    size: Union[int, Tuple[int, int]] = 224

    def __call__(self):
        from torchvision.transforms import v2

        return v2.Resize(self.size, interpolation=v2.InterpolationMode.BICUBIC)

    @property
    def resolution(self) -> Tuple[int, int]:
        if isinstance(self.size, int):
            return (self.size, self.size)
        return tuple(self.size)  # type: ignore[return-value]


class CenterCrop(BaseModel):
    # ``(H, W)`` — torchvision CenterCrop convention.
    size: Union[int, Tuple[int, int]] = (224, 320)

    def __call__(self):
        from torchvision.transforms import v2

        return v2.CenterCrop(self.size)


class ColorJitter(BaseModel):
    brightness: Union[float, Tuple[float, float]] = 0.2
    contrast: Union[float, Tuple[float, float]] = (0.8, 1.2)
    saturation: Union[float, Tuple[float, float]] = (0.8, 1.2)
    hue: Union[float, Tuple[float, float]] = 0.05

    def __call__(self):
        from torchvision.transforms import v2

        return v2.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )


# ---- Pipeline base classes -------------------------------------------------


class RepackTransform(BaseModel):
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data


class FieldTransform(BaseModel):
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data


class ModelTransform(BaseModel):
    def __call__(self, data: dict[str, Any], no_aug: bool = False, **kwargs) -> dict[str, Any]:
        return data


class DataTransform(BaseModel):
    repack: RepackTransform = Field(default_factory=RepackTransform)
    field: FieldTransform = Field(default_factory=FieldTransform)
    model: ModelTransform = Field(default_factory=ModelTransform)

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        data = self.repack(data, **kwargs)
        data = self.field(data, **kwargs)
        data = self.model(data, **kwargs)
        return data


# ---- Repack (source-specific) ---------------------------------------------


class HERepackTransform(RepackTransform):
    """Pretrain repack for Humanoid-Everyday's precomputed SONIC tokens."""

    dataset_name: str = "humanoid_everyday"

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        img = data["image"]
        observations: List[Image.Image] = [img] if isinstance(img, Image.Image) else list(img)
        return {
            "observations": observations,
            "instruction": str(data.get("text", "perform the task")),
            "sonic_tokens": data["sonic_tokens"],
            "sonic_mask": data["sonic_mask"],
            "dataset": self.dataset_name,
        }


class EgodexRepackTransform(RepackTransform):
    """Pretrain repack for EgoDex's precomputed SONIC tokens."""

    dataset_name: str = "egodex"

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        img = data["image"]
        observations: List[Image.Image] = [img] if isinstance(img, Image.Image) else list(img)
        return {
            "observations": observations,
            "instruction": str(data.get("text", "perform the task")),
            "sonic_tokens": data["sonic_tokens"],
            "sonic_mask": data["sonic_mask"],
            "dataset": self.dataset_name,
        }


# ---- Field (token-space) ---------------------------------------------------


class IdentityFieldTransform(FieldTransform):
    """No-op for precomputed SONIC tokens (already in a stable feature space)."""

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        return data


# ---- Model (image augmentation) -------------------------------------------


class KappaModelTransform(ModelTransform):
    """Apply image augmentation; defer VLM tokenization to ``KappaCollator``.

    Output keys (consumed by the collator):

    * ``image`` -- single ``PIL.Image`` at the configured resolution
    * ``text`` -- instruction string
    * ``sonic_tokens`` -- ``[chunk, sonic_dim]`` tensor
    * ``sonic_mask`` -- ``[chunk, ...]`` mask tensor
    """

    resize: ResizeImage = Field(default_factory=lambda: ResizeImage(size=224))
    center_crop: CenterCrop = Field(default_factory=lambda: CenterCrop(size=(224, 320)))
    color_jitter: Optional[ColorJitter] = Field(default_factory=ColorJitter)
    img_aug: bool = False

    def __call__(self, data: dict[str, Any], no_aug: bool = False, **kwargs) -> dict[str, Any]:
        from torchvision.transforms import v2

        do_img_aug = (not no_aug) and self.img_aug and self.color_jitter is not None
        steps = [self.resize(), self.center_crop()]
        if do_img_aug:
            steps.append(self.color_jitter())  # type: ignore[union-attr]
        pipeline = v2.Compose(steps)

        observations = [pipeline(img) for img in data["observations"]]

        return {
            "image": observations[0],
            "text": data["instruction"],
            "sonic_tokens": data["sonic_tokens"],
            "sonic_mask": data["sonic_mask"],
            "dataset_name": data.get("dataset", "unknown"),
        }

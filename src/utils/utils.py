"""
Utility functions for kappa project.
"""
from __future__ import annotations

import os
import random
import re
from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T")


def seed_everything(seed: int) -> None:
    """Set random seed for all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def flatten(dictionary: dict, parent_key: str = '', separator: str = '/') -> dict:
    """Flatten nested dictionaries compressing keys."""
    from collections.abc import MutableMapping
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            if callable(getattr(value, 'item', None)):
                value = value.item()
            items.append((new_key, value))
    return dict(items)


def snake_to_pascal(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    parts = re.split(r"[-_]", snake_str)
    return ''.join([parts[0].capitalize()] + [p.capitalize() for p in parts[1:]])


def move_to_device(batch, device, dtype=None):
    """Recursively move batch data to device with async H2D copies.

    Uses ``non_blocking=True`` so pinned-memory transfers from DataLoader
    workers can overlap with the previous step's forward/backward.
    """
    if isinstance(batch, torch.Tensor):
        if dtype is not None:
            return batch.to(device, dtype=dtype, non_blocking=True)
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device, dtype) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(item, device, dtype) for item in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(item, device, dtype) for item in batch)
    elif isinstance(batch, np.ndarray):
        tensor = torch.from_numpy(batch)
        if dtype is not None:
            return tensor.to(device, dtype=dtype, non_blocking=True)
        return tensor.to(device, non_blocking=True)
    return batch


def tensor_to_str(t: torch.Tensor) -> str:
    """Convert byte tensor back to string."""
    return bytes(t.tolist()).decode('utf-8')


def batch_str_to_tensor(batch: T) -> T:
    """Convert string tensors in batch back to strings."""
    if isinstance(batch, torch.Tensor):
        return batch_str_to_tensor({"data": batch})["data"]
    ret = {}
    for k, v in batch.items():
        if k.endswith("_str"):
            v = [tensor_to_str(s) for s in v]
        ret[k.replace("_str", "")] = v
    return ret

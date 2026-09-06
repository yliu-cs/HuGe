"""Episode-aware, epoch-addressable block shuffle sampler."""

from __future__ import annotations

import hashlib
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


SAMPLER_STATE_SCHEMA_VERSION = 1


class SourceWeightedSampler(WeightedRandomSampler):
    """Frame-uniform source weighting with an epoch-addressable random stream."""

    def __init__(self, source_lengths: list[int], ego_probability: float, num_samples: int, *, base_seed: int) -> None:
        if len(source_lengths) not in {1, 2} or any(n <= 0 for n in source_lengths):
            raise ValueError("expected one Real source and optionally one nonempty Ego source")
        if not (0 < ego_probability < 1 if len(source_lengths) == 2 else ego_probability == 0):
            raise ValueError("invalid Ego sampling probability")
        self.source_lengths = list(source_lengths)
        self.ego_probability = float(ego_probability)
        self.base_seed = int(base_seed)
        self.epoch = 0
        probabilities = [1 - ego_probability, ego_probability]
        weights = torch.cat([torch.full((n,), probabilities[i] / n, dtype=torch.double)
                             for i, n in enumerate(source_lengths)])
        super().__init__(weights, num_samples, replacement=True, generator=torch.Generator())

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)
        self.generator.manual_seed(self.base_seed + self.epoch)

    def __iter__(self) -> Iterator[int]:
        self.set_epoch(self.epoch)
        return super().__iter__()

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "kind": "source_weighted", "base_seed": self.base_seed,
                "epoch": self.epoch, "source_lengths": self.source_lengths,
                "ego_probability": self.ego_probability, "num_samples": self.num_samples}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        if not isinstance(state, dict) or {**state, "epoch": 0} != {**expected, "epoch": 0}:
            raise ValueError("weighted sampler state does not match current data contract")
        self.set_epoch(state["epoch"])


class BlockShuffleSampler(Sampler[int]):
    """Shuffle contiguous episode blocks from a deterministic epoch seed."""

    def __init__(
        self,
        episode_spans: np.ndarray,
        block_size: int,
        generator: torch.Generator | None = None,
        *,
        base_seed: int | None = None,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        spans = np.asarray(episode_spans, dtype=np.int64)
        if spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError(f"episode_spans must be [N,2], got {spans.shape}")
        if np.any(spans < 0):
            raise ValueError("episode_spans starts and lengths must be non-negative")
        if base_seed is not None and generator is not None:
            raise ValueError("pass either base_seed or generator, not both")
        if base_seed is None:
            base_seed = int(generator.initial_seed()) if generator is not None else 0
        if base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        self.block_size = int(block_size)
        self.base_seed = int(base_seed)
        self.epoch = 0
        starts: list[int] = []
        lengths: list[int] = []
        for start, length in spans:
            for offset in range(0, int(length), self.block_size):
                starts.append(int(start + offset))
                lengths.append(min(self.block_size, int(length - offset)))
        self.starts = np.asarray(starts, dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.total = int(spans[:, 1].sum())
        digest = hashlib.sha256()
        digest.update(self.starts.tobytes())
        digest.update(self.lengths.tobytes())
        self.span_fingerprint = digest.hexdigest()

    def __len__(self) -> int:
        return self.total

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SAMPLER_STATE_SCHEMA_VERSION,
            "base_seed": self.base_seed,
            "epoch": self.epoch,
            "block_size": self.block_size,
            "total": self.total,
            "span_fingerprint": self.span_fingerprint,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected_keys = {"schema_version", "base_seed", "epoch", "block_size", "total", "span_fingerprint"}
        if (
            not isinstance(state, dict)
            or set(state) != expected_keys
            or state.get("schema_version") != SAMPLER_STATE_SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported BlockShuffleSampler state: {state!r}")
        expected = {
            "base_seed": self.base_seed,
            "block_size": self.block_size,
            "total": self.total,
            "span_fingerprint": self.span_fingerprint,
        }
        mismatches = {key: (expected_value, state.get(key)) for key, expected_value in expected.items() if state.get(key) != expected_value}
        if mismatches:
            raise ValueError(f"BlockShuffleSampler state does not match current data contract: {mismatches}")
        self.set_epoch(int(state.get("epoch", -1)))

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.base_seed + self.epoch)
        for block_id in rng.permutation(len(self.starts)):
            start = int(self.starts[block_id])
            length = int(self.lengths[block_id])
            yield from range(start, start + length)

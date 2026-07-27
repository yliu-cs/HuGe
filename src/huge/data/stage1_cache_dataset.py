"""Stage 1 adapters for published multi-source SONIC caches."""

from __future__ import annotations

import hashlib
import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr
from PIL import Image

from huge.data.episode_selection import (
    SELECTION_ALGORITHM,
    derive_source_seed,
    select_episode_indices,
    selected_episode_digest,
    task_histogram,
)
from huge.data.schemas import ACTION_DIM, ACTION_HORIZON, PROPRIO_DIM, validate_action_array
from huge.data.video_reader import decode_video_frame


@dataclass(frozen=True)
class Stage1Segment:
    stable_id: str
    source_episode: str
    source: str
    language: str
    source_start_frame: int
    source_end_frame: int
    length: int
    source_start_timestamp_ns: int | None
    target_hz: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_contract(path: Path, root: Path) -> dict[str, str | int]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _unwrap_bytes(value: Any) -> bytes:
    while isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.ndarray) and value.dtype == np.uint8:
        return value.tobytes()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"expected encoded image bytes, got {type(value).__name__}")


class Stage1CacheDataset:
    """Shared 66-D action/state/index behavior for Stage 1 cache adapters."""

    expected_dataset: str

    def __init__(
        self,
        cache_root: str | Path,
        *,
        dataset_id: str,
        action_horizon: int = ACTION_HORIZON,
        sample_stride: int = 1,
        episode_fraction: float | None = None,
        episode_selection_seed: int = 7,
        max_episodes: int | None = None,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.dataset_id = dataset_id
        self.action_horizon = int(action_horizon)
        self.sample_stride = int(sample_stride)
        if self.sample_stride <= 0:
            raise ValueError(f"sample_stride must be positive, got {self.sample_stride}")
        self._contract_files = self._validate_cache()
        self._manifest = json.loads(
            (self.cache_root / "meta" / "conversion_manifest.json").read_text(encoding="utf-8")
        )
        raw_root = Path(self._manifest["data_root"])
        self.data_root = raw_root if raw_root.is_absolute() else Path.cwd() / raw_root
        if not self.data_root.exists():
            raise FileNotFoundError(f"{dataset_id} source root from manifest does not exist: {self.data_root}")
        language_map = json.loads((self.cache_root / "meta" / "language_map.json").read_text(encoding="utf-8"))
        all_segments = self._read_segments(language_map)
        episode_ids = list(dict.fromkeys(segment.source_episode for segment in all_segments))
        episode_tasks: dict[str, str] = {}
        for segment in all_segments:
            previous = episode_tasks.setdefault(segment.source_episode, segment.language)
            if previous != segment.language:
                raise ValueError(
                    f"source episode {segment.source_episode} has inconsistent tasks: "
                    f"{previous!r} != {segment.language!r}"
                )
        selected = select_episode_indices(
            len(episode_ids),
            source_id=dataset_id,
            base_seed=episode_selection_seed,
            fraction=episode_fraction,
            max_episodes=max_episodes,
        )
        selected_ids = [episode_ids[int(i)] for i in selected]
        self._selected_episode_ids = tuple(selected_ids)
        selected_set = set(selected_ids)
        self.segments = [segment for segment in all_segments if segment.source_episode in selected_set]
        if not self.segments:
            raise ValueError(f"episode selection produced no segments for {dataset_id}")
        self._finalize_index()
        self._selection_summary = {
            "selection_schema_version": 1,
            "algorithm": SELECTION_ALGORITHM,
            "selection_mode": "prefix_debug" if max_episodes is not None else "fraction",
            "source_id": dataset_id,
            "dataset_id": dataset_id,
            "adapter_name": type(self).__name__,
            "adapter_schema_version": 1,
            "cache_root": str(self.cache_root),
            "data_root": str(self.data_root),
            "requested_fraction": episode_fraction,
            "episode_fraction": episode_fraction,
            "effective_episode_fraction": len(selected_ids) / len(episode_ids),
            "episode_selection_seed": int(episode_selection_seed),
            "base_selection_seed": int(episode_selection_seed),
            "source_seed": derive_source_seed(episode_selection_seed, dataset_id),
            "derived_source_seed": derive_source_seed(episode_selection_seed, dataset_id),
            "available_episodes": len(episode_ids),
            "source_episode_count": len(episode_ids),
            "selected_episodes": len(selected_ids),
            "selected_episode_count": len(selected_ids),
            "selected_episode_digest": selected_episode_digest(selected_ids),
            "selected_episode_ids_sha256": selected_episode_digest(selected_ids),
            "source_segments": len(all_segments),
            "source_segment_count": len(all_segments),
            "selected_segments": len(self.segments),
            "selected_segment_count": len(self.segments),
            "source_token_rows": sum(segment.length for segment in all_segments),
            "selected_token_rows": sum(segment.length for segment in self.segments),
            "selected_stride_samples": len(self),
            "task_episode_counts": task_histogram(
                selected_ids, [episode_tasks[episode_id] for episode_id in selected_ids]
            ),
        }

    def _validate_cache(self) -> dict[str, dict[str, str | int]]:
        required = [
            self.cache_root / "_SUCCESS",
            self.cache_root / "meta" / "schema.json",
            self.cache_root / "meta" / "conversion_manifest.json",
            self.cache_root / "meta" / "episode_index.jsonl",
            self.cache_root / "meta" / "language_map.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete {self.dataset_id} cache; missing: {', '.join(missing)}")
        if required[0].read_text(encoding="utf-8").strip() != "ok":
            raise ValueError(f"invalid cache publication marker: {required[0]}")
        schema = json.loads(required[1].read_text(encoding="utf-8"))
        if schema.get("format_version") != 1 or schema.get("action", {}).get("shape") != ["T", ACTION_DIM]:
            raise ValueError(f"unsupported action schema in {required[1]}")
        manifest = json.loads(required[2].read_text(encoding="utf-8"))
        manifest_dataset = manifest.get("dataset")
        if manifest_dataset is not None and manifest_dataset != self.expected_dataset:
            raise ValueError(
                f"{self.dataset_id} manifest dataset is {manifest_dataset!r}, expected {self.expected_dataset!r}"
            )
        token_root = self.cache_root / "sonic_tokens"
        if not token_root.is_dir():
            raise FileNotFoundError(f"missing token directory: {token_root}")
        return {path.relative_to(self.cache_root).as_posix(): _file_contract(path, self.cache_root) for path in required}

    def _read_segments(self, language_map: dict[str, str]) -> list[Stage1Segment]:
        rows: list[Stage1Segment] = []
        stable_ids: set[str] = set()
        index_path = self.cache_root / "meta" / "episode_index.jsonl"
        with index_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                stable_id = str(row["stable_id"])
                length = int(row.get("target_frames", row.get("token_frame_count", 0)))
                if not stable_id or stable_id in stable_ids:
                    raise ValueError(f"{index_path}:{line_number} has invalid/duplicate stable_id {stable_id!r}")
                stable_ids.add(stable_id)
                if not str(row.get("source_episode", "")):
                    raise ValueError(f"{index_path}:{line_number} has an empty source_episode")
                if length <= 0:
                    raise ValueError(f"{index_path}:{line_number} has invalid token length {length}")
                dataset = row.get("dataset")
                if dataset is not None and dataset != self.expected_dataset:
                    raise ValueError(
                        f"{index_path}:{line_number} dataset is {dataset!r}, expected {self.expected_dataset!r}"
                    )
                rows.append(
                    Stage1Segment(
                        stable_id=stable_id,
                        source_episode=str(row["source_episode"]),
                        source=str(row["source"]),
                        language=str(language_map.get(stable_id, row.get("main_task", ""))).strip() or "perform the task",
                        source_start_frame=int(row.get("source_start_frame", 0)),
                        source_end_frame=int(row.get("source_end_frame_exclusive", length)),
                        length=length,
                        source_start_timestamp_ns=(
                            int(row["source_start_timestamp_ns"])
                            if row.get("source_start_timestamp_ns") is not None
                            else None
                        ),
                        target_hz=float(row.get("target_hz", row.get("interpreted_hz", 50.0))),
                    )
                )
        if not rows:
            raise ValueError(f"empty episode index: {index_path}")
        return rows

    def _finalize_index(self) -> None:
        self._segment_lengths = np.asarray(
            [(segment.length + self.sample_stride - 1) // self.sample_stride for segment in self.segments],
            dtype=np.int64,
        )
        self._segment_offsets = np.empty(len(self._segment_lengths), dtype=np.int64)
        self._segment_offsets[0] = 0
        if len(self._segment_lengths) > 1:
            np.cumsum(self._segment_lengths[:-1], out=self._segment_offsets[1:])
        self._total = int(self._segment_lengths.sum())

    def __len__(self) -> int:
        return self._total

    def episode_spans(self) -> np.ndarray:
        """Return one sampler span per selected cache segment."""
        return np.column_stack((self._segment_offsets, self._segment_lengths))

    def _resolve(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self._total
        if idx < 0 or idx >= self._total:
            raise IndexError(idx)
        segment_idx = int(np.searchsorted(self._segment_offsets, idx, side="right")) - 1
        frame_idx = (idx - int(self._segment_offsets[segment_idx])) * self.sample_stride
        return segment_idx, int(frame_idx)

    def _load_action(self, segment: Stage1Segment, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        token_path = self.cache_root / "sonic_tokens" / f"{segment.stable_id}.npy"
        if not token_path.is_file():
            raise FileNotFoundError(token_path)
        raw = np.load(token_path, mmap_mode="r")
        if raw.ndim != 2 or raw.shape != (segment.length, ACTION_DIM):
            raise ValueError(f"{token_path} must be [{segment.length}, {ACTION_DIM}], got {raw.shape}")
        validate_action_array(np.asarray(raw[:1]))
        chunk = np.asarray(raw[frame_idx : frame_idx + self.action_horizon], dtype=np.float32)
        valid = len(chunk)
        action = np.zeros((self.action_horizon, ACTION_DIM), dtype=np.float32)
        mask = np.zeros_like(action)
        action[:valid] = chunk
        mask[:valid] = 1.0
        return action, mask

    def _load_image(self, segment: Stage1Segment, frame_idx: int) -> Image.Image:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, Any]:
        segment_idx, frame_idx = self._resolve(idx)
        segment = self.segments[segment_idx]
        action, mask = self._load_action(segment, frame_idx)
        return {
            "dataset_id": self.dataset_id,
            "task_id": segment.language,
            "episode_id": segment.stable_id,
            "frame_idx": frame_idx,
            "video_path": None,
            "video_frame_idx": None,
            "language": segment.language,
            "state": np.zeros((PROPRIO_DIM,), dtype=np.float32),
            "state_mask": np.zeros((1,), dtype=np.float32),
            "normalized_actions": action,
            "sonic_mask": mask,
            "image": self._load_image(segment, frame_idx),
        }

    def _vision_contract(self) -> tuple[str, str]:
        return {
            "egoverse": ("zarr/images.front_1", "source_start_frame_plus_local_token_frame"),
            "behavior1k": ("observation.images.rgb.head/mp4", "source_start_frame_plus_local_token_frame"),
            "xperience": ("stereo_left.mp4", "nearest_device_timestamp_tie_earlier"),
        }[self.dataset_id]

    def data_contract(self) -> dict[str, Any]:
        return {
            **self.selection_summary(),
            "samples": len(self),
            "contract_files": self._contract_files,
        }

    def selection_summary(self) -> dict[str, Any]:
        vision_source, vision_time_mapping = self._vision_contract()
        return {
            **self._selection_summary,
            "index_file": self._contract_files["meta/episode_index.jsonl"],
            "schema_file": self._contract_files["meta/schema.json"],
            "conversion_manifest": self._contract_files["meta/conversion_manifest.json"],
            "encoder_model_sha256": self._manifest.get("encoder_model_sha256"),
            "raw_inventory_fingerprint": self._manifest.get("raw_inventory_fingerprint"),
            "vision_source": vision_source,
            "vision_time_mapping": vision_time_mapping,
        }

    def selected_episode_ids(self) -> tuple[str, ...]:
        return self._selected_episode_ids


class EgoVerseDataset(Stage1CacheDataset):
    expected_dataset = "egoverse"

    def __init__(self, cache_root: str | Path, **kwargs: Any) -> None:
        self._zarr_cache: OrderedDict[str, Any] = OrderedDict()
        super().__init__(cache_root, dataset_id="egoverse", **kwargs)

    def _load_image(self, segment: Stage1Segment, frame_idx: int) -> Image.Image:
        source_path = self.data_root / "v3" / segment.source
        key = str(source_path)
        group = self._zarr_cache.get(key)
        if group is None:
            if not source_path.is_dir():
                raise FileNotFoundError(source_path)
            group = zarr.open_group(str(source_path), mode="r")
            self._zarr_cache[key] = group
            if len(self._zarr_cache) > 4:
                self._zarr_cache.popitem(last=False)
        else:
            self._zarr_cache.move_to_end(key)
        source_frame = segment.source_start_frame + frame_idx
        if source_frame >= segment.source_end_frame:
            raise IndexError(f"{segment.stable_id} image frame {source_frame} exceeds segment end")
        payload = _unwrap_bytes(group["images.front_1"][source_frame])
        with Image.open(io.BytesIO(payload)) as image:
            return image.convert("RGB")


class Behavior1KDataset(Stage1CacheDataset):
    expected_dataset = "behavior1k"

    def __init__(self, cache_root: str | Path, **kwargs: Any) -> None:
        super().__init__(cache_root, dataset_id="behavior1k", **kwargs)

    def _video_path(self, segment: Stage1Segment) -> Path:
        source = Path(segment.source)
        return (
            self.data_root
            / "v1"
            / "videos"
            / source.parent.name
            / "observation.images.rgb.head"
            / f"{source.stem}.mp4"
        )

    def _read_segments(self, language_map: dict[str, str]) -> list[Stage1Segment]:
        segments = super()._read_segments(language_map)
        available = [segment for segment in segments if self._video_path(segment).is_file()]
        missing = [segment for segment in segments if not self._video_path(segment).is_file()]
        self._missing_video_episode_count = len({segment.source_episode for segment in missing})
        self._missing_video_segment_count = len(missing)
        self._missing_video_token_rows = sum(segment.length for segment in missing)
        return available

    def selection_summary(self) -> dict[str, Any]:
        return {
            **super().selection_summary(),
            "missing_video_episode_count": self._missing_video_episode_count,
            "missing_video_segment_count": self._missing_video_segment_count,
            "missing_video_token_rows": self._missing_video_token_rows,
        }

    def _load_image(self, segment: Stage1Segment, frame_idx: int) -> Image.Image:
        return decode_video_frame(
            self._video_path(segment),
            segment.source_start_frame + frame_idx,
            missing_ok=False,
        )


class XperienceDataset(Stage1CacheDataset):
    expected_dataset = "xperience"

    def __init__(self, cache_root: str | Path, **kwargs: Any) -> None:
        self._timestamp_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        super().__init__(cache_root, dataset_id="xperience", **kwargs)

    def _timestamps(self, annotation_path: Path) -> np.ndarray:
        key = str(annotation_path)
        values = self._timestamp_cache.get(key)
        if values is None:
            if not annotation_path.is_file():
                raise FileNotFoundError(annotation_path)
            with h5py.File(annotation_path, "r") as stream:
                values = np.asarray(stream["video/device_timestamp"][:], dtype=np.int64)
            if values.ndim != 1 or len(values) == 0 or np.any(values[1:] <= values[:-1]):
                raise ValueError(f"invalid video/device_timestamp in {annotation_path}")
            self._timestamp_cache[key] = values
            if len(self._timestamp_cache) > 8:
                self._timestamp_cache.popitem(last=False)
        else:
            self._timestamp_cache.move_to_end(key)
        return values

    def _load_image(self, segment: Stage1Segment, frame_idx: int) -> Image.Image:
        if segment.source_start_timestamp_ns is None:
            raise ValueError(f"{segment.stable_id} lacks source_start_timestamp_ns")
        annotation = self.data_root / segment.source
        timestamps = self._timestamps(annotation)
        target = segment.source_start_timestamp_ns + round(frame_idx * 1_000_000_000 / segment.target_hz)
        pos = int(np.searchsorted(timestamps, target, side="left"))
        candidates = [max(0, min(pos, len(timestamps) - 1))]
        if pos > 0:
            candidates.append(pos - 1)
        source_frame = min(candidates, key=lambda value: (abs(int(timestamps[value]) - target), value))
        if source_frame < segment.source_start_frame or source_frame >= segment.source_end_frame:
            raise IndexError(
                f"{segment.stable_id} timestamp mapped outside [{segment.source_start_frame}, "
                f"{segment.source_end_frame}): {source_frame}"
            )
        return decode_video_frame(annotation.parent / "stereo_left.mp4", source_frame, missing_ok=False)

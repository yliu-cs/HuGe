"""HIW-500 Stage 2 adapter over in-place 66-D tokens and 34-D proprio."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from huge.data.schemas import (
    ACTION_DIM,
    ACTION_SCHEMA_VERSION,
    HAND_POLARITY,
    MOTION_TOKEN_DIM,
    PROPRIO_DIM,
    PROPRIO_LAYOUT_VERSION,
    hand_binary_is_valid,
)
from huge.data.sonic_dataset import SonicTokenDataset
from huge.data.state_normalizer import StateNormalizer
from huge.data.video_reader import decode_video_timestamp

HIW_VIDEO_KEY = "observation.images.head"
HIW_EPISODES_REPO_ID = "BitRobot/HIW-500-LeRobot"


@dataclass(frozen=True, slots=True)
class HIWEpisode:
    episode_id: str
    length: int
    language: str
    data_path: Path
    video_path: Path
    video_from_timestamp: float
    video_to_timestamp: float


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


def resolve_hiw_episode_metadata_root(data_root: Path) -> tuple[Path, str]:
    source_root = data_root / "meta" / "episodes"
    if source_root.is_dir() and any(source_root.rglob("*.parquet")):
        return source_root, "source:meta/episodes"

    from huggingface_hub import snapshot_download

    snapshot_root = Path(
        snapshot_download(
            repo_id=HIW_EPISODES_REPO_ID,
            repo_type="dataset",
            allow_patterns=["meta/episodes/**"],
        )
    )
    episodes_root = snapshot_root / "meta" / "episodes"
    if not episodes_root.is_dir() or not any(episodes_root.rglob("*.parquet")):
        raise FileNotFoundError(
            f"{HIW_EPISODES_REPO_ID} snapshot does not contain meta/episodes"
        )
    return episodes_root, f"huggingface:{HIW_EPISODES_REPO_ID}"


def _value(row: dict[str, Any], *keys: str, required: bool = True) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
        current: Any = row
        for part in key.split("/"):
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            if current is not None:
                return current
    if required:
        raise KeyError(f"none of the metadata fields exist: {keys}")
    return None


def _load_tasks(data_root: Path) -> tuple[dict[int, str], dict[str, str]]:
    rows = pq.read_table(data_root / "meta" / "tasks.parquet").to_pylist()
    by_index = {int(row["task_index"]): str(row["task"]) for row in rows}
    return by_index, {task: task for task in by_index.values()}

def _episode_metadata_columns(path: Path) -> list[str]:
    names = set(pq.read_schema(path).names)
    prefix = f"videos/{HIW_VIDEO_KEY}"
    groups = [
        ("episode_index",),
        ("length", "episode_length"),
        ("tasks", "task", "task_index", "task_indices"),
        ("data/chunk_index", "data_chunk_index", "data"),
        ("data/file_index", "data_file_index", "data"),
        (f"{prefix}/chunk_index", "video_chunk_index", "head_video_chunk_index", "videos"),
        (f"{prefix}/file_index", "video_file_index", "head_video_file_index", "videos"),
        (f"{prefix}/from_timestamp", "video_from_timestamp", "head_from_timestamp", "videos"),
        (f"{prefix}/to_timestamp", "video_to_timestamp", "head_to_timestamp", "videos"),
    ]
    columns: set[str] = set()
    for candidates in groups:
        match = next((candidate for candidate in candidates if candidate in names), None)
        if match is None:
            raise ValueError(f"{path} lacks required episode metadata alternatives {candidates}")
        columns.add(match)
    return sorted(columns)


def load_hiw_episode_metadata(data_root: Path, episodes_root: Path) -> dict[str, HIWEpisode]:
    info = json.loads((data_root / "meta" / "info.json").read_text(encoding="utf-8"))
    data_template = str(info["data_path"])
    video_template = str(info["video_path"])
    tasks_by_index, tasks_by_name = _load_tasks(data_root)
    metadata_files = sorted(episodes_root.rglob("*.parquet"))
    if not metadata_files:
        raise FileNotFoundError(f"no episode metadata Parquet files under {episodes_root}")
    episodes_absolute = episodes_root.absolute()
    escaped = [str(path) for path in metadata_files if not path.absolute().is_relative_to(episodes_absolute)]
    if escaped:
        raise ValueError(f"HIW episode metadata escapes its declared root: {escaped[:3]}")

    episodes: dict[str, HIWEpisode] = {}
    for metadata_file in metadata_files:
        for row in pq.read_table(metadata_file, columns=_episode_metadata_columns(metadata_file)).to_pylist():
            episode_index = int(_value(row, "episode_index"))
            episode_id = f"{episode_index:06d}"
            if episode_id in episodes:
                raise ValueError(f"duplicate HIW episode {episode_id}")
            length = int(_value(row, "length", "episode_length"))
            if length <= 0:
                raise ValueError(f"HIW episode {episode_id} has invalid length {length}")
            task_value = _value(row, "tasks", "task", "task_index", "task_indices", required=False)
            if isinstance(task_value, (list, tuple)):
                task_value = task_value[0] if task_value else None
            if isinstance(task_value, str):
                language = tasks_by_name.get(task_value, task_value)
            elif task_value is not None:
                language = tasks_by_index[int(task_value)]
            else:
                raise ValueError(f"HIW episode {episode_id} has no episode-level task")

            data_chunk = int(_value(row, "data/chunk_index", "data_chunk_index"))
            data_file = int(_value(row, "data/file_index", "data_file_index"))
            video_prefix = f"videos/{HIW_VIDEO_KEY}"
            video_chunk = int(
                _value(row, f"{video_prefix}/chunk_index", "video_chunk_index", "head_video_chunk_index")
            )
            video_file = int(
                _value(row, f"{video_prefix}/file_index", "video_file_index", "head_video_file_index")
            )
            video_from = float(
                _value(row, f"{video_prefix}/from_timestamp", "video_from_timestamp", "head_from_timestamp")
            )
            video_to = float(
                _value(row, f"{video_prefix}/to_timestamp", "video_to_timestamp", "head_to_timestamp")
            )
            if not 0.0 <= video_from < video_to:
                raise ValueError(f"HIW episode {episode_id} has invalid video interval [{video_from}, {video_to}]")
            data_path = data_root / data_template.format(chunk_index=data_chunk, file_index=data_file)
            video_path = data_root / video_template.format(
                video_key=HIW_VIDEO_KEY,
                chunk_index=video_chunk,
                file_index=video_file,
            )
            source_absolute = data_root.absolute()
            escaped_source_paths = [
                str(path) for path in (data_path, video_path) if not path.absolute().is_relative_to(source_absolute)
            ]
            if escaped_source_paths:
                raise ValueError(f"HIW metadata path escapes source root: {escaped_source_paths}")
            episodes[episode_id] = HIWEpisode(
                episode_id=episode_id,
                length=length,
                language=language,
                data_path=data_path,
                video_path=video_path,
                video_from_timestamp=video_from,
                video_to_timestamp=video_to,
            )

    expected = int(info["total_episodes"])
    expected_ids = {f"{index:06d}" for index in range(expected)}
    if set(episodes) != expected_ids:
        missing = sorted(expected_ids - set(episodes))[:5]
        extra = sorted(set(episodes) - expected_ids)[:5]
        raise ValueError(f"HIW episode metadata is not contiguous: missing={missing}, extra={extra}")
    missing_data = sorted({str(ep.data_path) for ep in episodes.values() if not ep.data_path.is_file()})
    missing_video = sorted({str(ep.video_path) for ep in episodes.values() if not ep.video_path.is_file()})
    if missing_data or missing_video:
        raise FileNotFoundError(
            f"HIW metadata references missing files: data={missing_data[:3]}, videos={missing_video[:3]}"
        )
    return episodes


class HIWDataset(SonicTokenDataset):
    """Read event-latched 66-D HIW tokens from the source token directory."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        cache_root: str | Path | None = None,
        sample_stride: int = 1,
        proprio_normalizer: StateNormalizer | None = None,
        max_episodes: int | None = None,
        validated_inventory: dict[str, Any] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.hiw_cache_root = (
            Path(cache_root) if cache_root is not None else self.data_root.with_name(f"{self.data_root.name}-cache")
        )
        self._episode_metadata_root, self._episode_metadata_origin = (
            resolve_hiw_episode_metadata_root(self.data_root)
        )
        self._manifest = self._validate_overlay(skip_inventory_validation=validated_inventory is not None)
        self._hiw_episodes = load_hiw_episode_metadata(
            self.data_root,
            self._episode_metadata_root,
        )
        if validated_inventory is None:
            self._validate_tokens()
        language_map = {episode_id: episode.language for episode_id, episode in self._hiw_episodes.items()}
        super().__init__(
            self.data_root,
            dataset_id="hiw",
            sample_stride=sample_stride,
            proprio_normalizer=proprio_normalizer,
            state_required=True,
            egodex_64d=False,
            language_map=language_map,
            max_episodes=max_episodes,
            video_len_hint_mode="token_1to1",
            validated_inventory=validated_inventory,
        )

    def _validate_overlay(self, *, skip_inventory_validation: bool = False) -> dict[str, Any]:
        required = [
            self.data_root / "meta" / "info.json",
            self.data_root / "meta" / "tasks.parquet",
            self.data_root / "sonic_tokens",
            self.data_root / "g1_proprio",
            self.hiw_cache_root / "_SUCCESS",
            self.hiw_cache_root / "meta" / "conversion_manifest.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"incomplete HIW source/cache; missing: {', '.join(missing)}")
        manifest_path = required[5]
        success_path = required[4]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_sha256 = _sha256(manifest_path)
        if success_path.read_text(encoding="utf-8").strip() != manifest_sha256:
            raise ValueError("HIW cache _SUCCESS does not match conversion manifest")
        if skip_inventory_validation:
            self._episode_metadata_files = list(manifest.get("episode_metadata_files", []))
        else:
            episode_files: list[dict[str, str | int]] = []
            episode_digest = hashlib.sha256()
            for path in sorted(item for item in self._episode_metadata_root.rglob("*") if item.is_file()):
                contract = _file_contract(path, self._episode_metadata_root)
                episode_files.append(contract)
                episode_digest.update(
                    f"{contract['path']}\t{contract['size']}\t{contract['sha256']}\n".encode()
                )
            if episode_files != manifest.get("episode_metadata_files"):
                raise ValueError("HIW source episode metadata file inventory changed")
            if episode_digest.hexdigest() != manifest.get("episode_metadata_digest"):
                raise ValueError("HIW source episode metadata digest changed")
            self._episode_metadata_files = episode_files

        if manifest.get("schema_version") != 4 or manifest.get("dataset_id") != "hiw500":
            raise ValueError(f"unsupported HIW cache manifest: {manifest_path}")
        if manifest.get("episode_metadata_origin") != self._episode_metadata_origin:
            raise ValueError("HIW episode metadata origin changed after cache preparation")
        source_resolved = self.data_root.resolve()
        if Path(manifest.get("source_resolved_root", "")) != source_resolved:
            raise ValueError("HIW cache was prepared for a different resolved source root")
        if Path(manifest.get("source_token_path", "")) != source_resolved / "sonic_tokens":
            raise ValueError("HIW cache manifest points at a different token directory")
        if Path(manifest.get("source_proprio_path", "")) != source_resolved / "g1_proprio":
            raise ValueError("HIW cache manifest points at a different proprio directory")
        if manifest.get("source_info_sha256") != _sha256(required[0]):
            raise ValueError("HIW source info.json changed after cache preparation")
        if manifest.get("source_tasks_sha256") != _sha256(required[1]):
            raise ValueError("HIW source tasks.parquet changed after cache preparation")

        expected_contract = {
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "motion_dim": MOTION_TOKEN_DIM,
            "hand_dim": ACTION_DIM - MOTION_TOKEN_DIM,
            "action_dim": ACTION_DIM,
            "hand_mode": "event_latched_v1",
            "hand_polarity": HAND_POLARITY,
            "initial_hand_state": "open",
            "simultaneous_event_policy": "open_wins",
            "token_storage": "source_sonic_tokens_in_place",
            "replacement_mode": "directory_swap_with_retained_backup",
            "source_fps": 30,
            "logical_control_hz": 50,
            "resample_mode": "none",
            "state_mode": "source_observation_state_29_plus_wbc_v1",
            "proprio_dim": PROPRIO_DIM,
            "proprio_layout_version": PROPRIO_LAYOUT_VERSION,
            "proprio_storage": "source_g1_proprio",
            "state_hand_mode": "observation_state_wbc_event_latched_v1",
            "projected_gravity_mode": "pivot_rpy_extrinsic_xyz_v1",
            "video_backend": "pyav_libdav1d",
            "video_key": HIW_VIDEO_KEY,
        }
        mismatched = sorted(
            key
            for key, value in expected_contract.items()
            if manifest.get(key) != value
        )
        if mismatched:
            raise ValueError(f"HIW cache manifest violates the Stage 2 contract: {mismatched}")

        trigger_threshold = manifest.get("trigger_close_threshold")
        squeeze_threshold = manifest.get("squeeze_open_threshold")
        if (
            type(trigger_threshold) not in (int, float)
            or not np.isfinite(trigger_threshold)
            or not 0.0 < trigger_threshold < 10.0
            or type(squeeze_threshold) not in (int, float)
            or not np.isfinite(squeeze_threshold)
            or not 0.0 < squeeze_threshold < 1.0
        ):
            raise ValueError("HIW cache manifest has invalid hand thresholds")
        for key in (
            "source_motion_token_inventory_digest",
            "source_hand_event_digest",
            "output_token_inventory_digest",
            "source_state_digest",
            "output_proprio_inventory_digest",
        ):
            value = manifest.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"HIW cache manifest has invalid {key}")
        event_counts = manifest.get("hand_event_counts")
        if not isinstance(event_counts, dict) or set(event_counts) != {"left", "right"}:
            raise ValueError("HIW cache manifest has invalid hand event counts")
        return manifest


    def _validate_tokens(self) -> None:
        token_paths = sorted((self.data_root / "sonic_tokens").glob("*.npy"))
        if len(token_paths) != len(self._hiw_episodes):
            raise ValueError(
                f"HIW token/metadata count mismatch: {len(token_paths)} != {len(self._hiw_episodes)}"
            )
        for token_path in token_paths:
            episode_id = token_path.stem.removeprefix("episode_")
            metadata = self._hiw_episodes.get(episode_id)
            if metadata is None:
                raise ValueError(f"HIW token has no episode metadata: {token_path}")
            token = np.load(token_path, mmap_mode="r")
            if (
                token.ndim != 2
                or token.shape != (metadata.length, ACTION_DIM)
                or token.dtype != np.float32
            ):
                raise ValueError(
                    f"{token_path} must be [{metadata.length}, {ACTION_DIM}] float32, "
                    f"got {token.shape} {token.dtype}"
                )
            if not hand_binary_is_valid(token):
                raise ValueError(f"{token_path} has invalid binary hand values")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)
        metadata = self._hiw_episodes[sample["episode_id"]]
        frame_idx = int(sample["frame_idx"])
        timestamp = metadata.video_from_timestamp + frame_idx / 30.0
        tolerance = 0.5 / 30.0
        if timestamp > metadata.video_to_timestamp + tolerance:
            raise IndexError(
                f"HIW episode {metadata.episode_id} frame {frame_idx} maps outside its video interval"
            )
        sample["task_id"] = metadata.language
        sample["language"] = metadata.language
        sample["video_path"] = str(metadata.video_path)
        sample["video_frame_idx"] = None
        sample["image"] = decode_video_timestamp(
            metadata.video_path,
            timestamp,
            missing_ok=False,
            backend="pyav",
        )
        return sample

    def data_contract(self) -> dict[str, Any]:
        manifest_path = self.hiw_cache_root / "meta" / "conversion_manifest.json"
        base = super().data_contract()
        return {
            **base,
            "dataset_id": "hiw",
            "data_root": str(self.data_root),
            "source_resolved_root": str(self.data_root.resolve()),
            "cache_root": str(self.hiw_cache_root),
            "episodes": len(self.episodes),
            "samples": len(self),
            "source_fps": 30,
            "logical_control_hz": 50,
            "resample_mode": "none",
            "token_video_mapping": "one_to_one_at_source_30hz",
            "action_dim": ACTION_DIM,
            "motion_dim": MOTION_TOKEN_DIM,
            "hand_dim": ACTION_DIM - MOTION_TOKEN_DIM,
            "hand_mode": "event_latched_v1",
            "hand_polarity": HAND_POLARITY,
            "initial_hand_state": "open",
            "trigger_close_threshold": self._manifest["trigger_close_threshold"],
            "squeeze_open_threshold": self._manifest["squeeze_open_threshold"],
            "simultaneous_event_policy": "open_wins",
            "state_mode": "source_observation_state_29_plus_wbc_v1",
            "proprio_dim": PROPRIO_DIM,
            "proprio_layout_version": PROPRIO_LAYOUT_VERSION,
            "proprio_storage": "source_g1_proprio",
            "state_hand_mode": "observation_state_wbc_event_latched_v1",
            "projected_gravity_mode": "pivot_rpy_extrinsic_xyz_v1",
            "video_backend": "pyav_libdav1d",
            "video_key": HIW_VIDEO_KEY,
            "source_info_file": _file_contract(self.data_root / "meta" / "info.json", self.data_root),
            "source_tasks_file": _file_contract(self.data_root / "meta" / "tasks.parquet", self.data_root),
            "cache_success_file": _file_contract(self.hiw_cache_root / "_SUCCESS", self.hiw_cache_root),
            "episode_metadata_origin": self._episode_metadata_origin,
            "manifest_file": {
                "path": str(manifest_path),
                "size": manifest_path.stat().st_size,
                "sha256": _sha256(manifest_path),
            },
            "episode_metadata_digest": self._manifest["episode_metadata_digest"],
            "episode_metadata_files": self._episode_metadata_files,
            "source_motion_token_inventory_digest": self._manifest[
                "source_motion_token_inventory_digest"
            ],
            "source_hand_event_digest": self._manifest["source_hand_event_digest"],
            "output_token_inventory_digest": self._manifest["output_token_inventory_digest"],
            "sonic_token_inventory_digest": self._manifest["output_token_inventory_digest"],
            "source_state_digest": self._manifest["source_state_digest"],
            "proprio_inventory_digest": self._manifest["output_proprio_inventory_digest"],
        }


def build_hiw_dataset(data_root: str | Path, **kwargs: Any) -> HIWDataset:
    return HIWDataset(data_root, **kwargs)

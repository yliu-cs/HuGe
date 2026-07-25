#!/usr/bin/env python3
"""Atomically publish HIW-500 66-D tokens and real 34-D proprio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import av
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from huge.data.hiw_dataset import (
    HIW_EPISODES_REPO_ID,
    HIWEpisode,
    load_hiw_episode_metadata,
    resolve_hiw_episode_metadata_root,
)
from huge.data.preprocess import atomic_save_npy, projected_gravity_from_quat_wxyz
from huge.data.schemas import (
    ACTION_DIM,
    ACTION_SCHEMA_VERSION,
    HAND_POLARITY,
    MOTION_TOKEN_DIM,
    PROPRIO_DIM,
    PROPRIO_LAYOUT_VERSION,
    hand_binary_is_valid,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_inventory(root: Path) -> tuple[list[dict[str, str | int]], str]:
    rows: list[dict[str, str | int]] = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        size = path.stat().st_size
        rows.append({"path": relative, "size": size, "sha256": file_hash})
        digest.update(f"{relative}\t{size}\t{file_hash}\n".encode())
    return rows, digest.hexdigest()



def _token_inventory(
    token_root: Path,
    episode_lengths: dict[str, int],
    expected_dim: int,
    *,
    validate_hands: bool = False,
) -> tuple[str, int]:
    expected_paths = {
        token_root / f"episode_{episode_id}.npy"
        for episode_id in episode_lengths
    }
    actual_paths = set(token_root.glob("*.npy"))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)[:3]
        extra = sorted(str(path) for path in actual_paths - expected_paths)[:3]
        raise ValueError(f"token inventory mismatch: missing={missing}, extra={extra}")

    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(actual_paths):
        episode_id = path.stem.removeprefix("episode_")
        expected_length = episode_lengths[episode_id]
        array = np.load(path, mmap_mode="r")
        if array.ndim != 2 or array.shape != (expected_length, expected_dim) or array.dtype != np.float32:
            raise ValueError(
                f"{path} must be [{expected_length}, {expected_dim}] float32, "
                f"got {array.shape} {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{path} contains NaN or Inf")
        if validate_hands and not hand_binary_is_valid(array):
            raise ValueError(f"{path} has non-binary hand values")
        size = path.stat().st_size
        digest.update(f"{path.name}\t{size}\t{_sha256(path)}\n".encode())
        total_bytes += size
    return digest.hexdigest(), total_bytes


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False



HAND_EVENT_NAMES = (
    "left_trigger",
    "left_squeeze",
    "right_trigger",
    "right_squeeze",
)

BODY_STATE_NAMES = (
    "kLeftHipPitch.q", "kLeftHipRoll.q", "kLeftHipYaw.q", "kLeftKnee.q",
    "kLeftAnklePitch.q", "kLeftAnkleRoll.q", "kRightHipPitch.q",
    "kRightHipRoll.q", "kRightHipYaw.q", "kRightKnee.q",
    "kRightAnklePitch.q", "kRightAnkleRoll.q", "kWaistYaw.q",
    "kWaistRoll.q", "kWaistPitch.q", "kLeftShoulderPitch.q",
    "kLeftShoulderRoll.q", "kLeftShoulderYaw.q", "kLeftElbow.q",
    "kLeftWristRoll.q", "kLeftWristPitch.q", "kLeftWristyaw.q",
    "kRightShoulderPitch.q", "kRightShoulderRoll.q", "kRightShoulderYaw.q",
    "kRightElbow.q", "kRightWristRoll.q", "kRightWristPitch.q",
    "kRightWristYaw.q",
)


def _feature_names(info: dict[str, Any], key: str, expected_dim: int) -> list[str]:
    feature = info.get("features", {}).get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"meta/info.json lacks the {key} feature")
    names = feature.get("names")
    if (
        feature.get("dtype") != "float32"
        or feature.get("shape") != [expected_dim]
        or not isinstance(names, list)
        or len(names) != expected_dim
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"HIW {key} feature must be unique named [{expected_dim}] float32")
    return [str(name) for name in names]


def _action_indices(info: dict[str, Any]) -> tuple[tuple[int, int, int, int], list[str]]:
    names = _feature_names(info, "action", 23)
    indices = tuple(names.index(name) for name in HAND_EVENT_NAMES)
    if indices != (19, 20, 21, 22):
        raise ValueError(f"unexpected HIW hand event layout: {indices}")
    return indices, names


def _feature_matrix(table: Any, key: str, path: Path, expected_dim: int) -> np.ndarray:
    column = table[key].combine_chunks()
    if column.null_count or not hasattr(column, "offsets"):
        raise ValueError(f"{path} has null or non-list {key} rows")
    offsets = column.offsets.to_numpy(zero_copy_only=False)
    if len(offsets) != table.num_rows + 1 or not np.all(np.diff(offsets) == expected_dim):
        raise ValueError(f"{path} {key} rows are not uniformly {expected_dim}-D")
    values = column.values.to_numpy(zero_copy_only=False)
    array = np.asarray(values, dtype=np.float32).reshape(table.num_rows, expected_dim)
    if not np.isfinite(array).all():
        raise ValueError(f"{path} {key} contains NaN or Inf")
    return array


def _iter_episode_arrays(
    episodes: dict[str, HIWEpisode],
    feature_dims: dict[str, int],
    *,
    selected_ids: set[str] | None = None,
) -> Iterator[tuple[HIWEpisode, dict[str, np.ndarray]]]:
    expected_ids = set(episodes) if selected_ids is None else set(selected_ids)
    unknown = expected_ids - set(episodes)
    if unknown:
        raise ValueError(f"unknown HIW episodes: {sorted(unknown)[:3]}")
    data_paths = sorted({episodes[episode_id].data_path for episode_id in expected_ids})
    seen: set[str] = set()

    for data_path in data_paths:
        table = pq.read_table(data_path, columns=[*feature_dims, "episode_index"])
        if table.num_rows == 0:
            raise ValueError(f"{data_path} has no rows")
        arrays = {
            key: _feature_matrix(table, key, data_path, width)
            for key, width in feature_dims.items()
        }
        episode_column = table["episode_index"].combine_chunks()
        if episode_column.null_count:
            raise ValueError(f"{data_path} has null episode_index rows")
        episode_ids = np.asarray(episode_column.to_numpy(zero_copy_only=False), dtype=np.int64)
        changes = np.flatnonzero(episode_ids[1:] != episode_ids[:-1]) + 1
        starts = np.concatenate((np.array([0]), changes))
        ends = np.concatenate((changes, np.array([len(episode_ids)])))
        for start, end in zip(starts, ends, strict=True):
            episode_id = f"{int(episode_ids[start]):06d}"
            metadata = episodes.get(episode_id)
            if metadata is None:
                raise ValueError(f"{data_path} contains unknown episode {episode_id}")
            if episode_id not in expected_ids:
                continue
            if episode_id in seen:
                raise ValueError(f"episode {episode_id} appears in multiple row runs")
            if metadata.data_path.absolute() != data_path.absolute():
                raise ValueError(f"episode {episode_id} appears in the wrong data parquet")
            if end - start != metadata.length:
                raise ValueError(
                    f"episode {episode_id} row count {end - start} != metadata {metadata.length}"
                )
            seen.add(episode_id)
            yield metadata, {
                key: np.ascontiguousarray(array[start:end], dtype=np.float32)
                for key, array in arrays.items()
            }

    if seen != expected_ids:
        raise ValueError(f"missing HIW episode rows: {sorted(expected_ids - seen)[:3]}")


def _iter_episode_events(
    episodes: dict[str, HIWEpisode],
    action_indices: tuple[int, int, int, int],
    *,
    selected_ids: set[str] | None = None,
) -> Iterator[tuple[HIWEpisode, np.ndarray]]:
    lower = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([10.0, 1.0, 10.0, 1.0], dtype=np.float32)
    for metadata, arrays in _iter_episode_arrays(
        episodes, {"action": 23}, selected_ids=selected_ids
    ):
        events = arrays["action"][:, action_indices]
        if np.any(events < lower) or np.any(events > upper):
            raise ValueError(f"episode {metadata.episode_id} hand event is outside its declared range")
        yield metadata, events


def _hand_states(
    events: np.ndarray,
    trigger_close_threshold: float,
    squeeze_open_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    if events.ndim != 2 or events.shape[1] != 4 or len(events) == 0:
        raise ValueError(f"hand events must be non-empty [T,4], got {events.shape}")
    states = np.zeros((len(events), 2), dtype=np.float32)
    stats: dict[str, int] = {}
    frame_numbers = np.arange(len(events), dtype=np.int64)

    for side, trigger_index, squeeze_index, output_index in (
        ("left", 0, 1, 0),
        ("right", 2, 3, 1),
    ):
        trigger = events[:, trigger_index]
        squeeze = events[:, squeeze_index]
        close = np.zeros(len(events), dtype=bool)
        opened = np.zeros(len(events), dtype=bool)
        close[1:] = (trigger[:-1] >= trigger_close_threshold) & (
            trigger[1:] < trigger_close_threshold
        )
        opened[1:] = (squeeze[:-1] <= squeeze_open_threshold) & (
            squeeze[1:] > squeeze_open_threshold
        )
        updates = np.zeros(len(events), dtype=np.int8)
        updates[0] = -1
        updates[1:] = np.where(opened[1:], -1, np.where(close[1:], 1, 0))
        last_update = np.maximum.accumulate(np.where(updates != 0, frame_numbers, 0))
        closed = updates[last_update] == 1
        states[:, output_index] = closed
        stats[f"{side}_close_events"] = int(close.sum())
        stats[f"{side}_open_events"] = int(opened.sum())
        stats[f"{side}_simultaneous_events"] = int((close & opened).sum())
        stats[f"{side}_closed_frames"] = int(closed.sum())
    return states, stats


def _update_event_digest(digest: Any, episode_id: str, events: np.ndarray) -> None:
    digest.update(f"{episode_id}\t{len(events)}\n".encode())
    digest.update(np.asarray(events, dtype="<f4").tobytes(order="C"))


def _add_stats(total: dict[str, int], values: dict[str, int]) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0) + value


def _proprio_from_source(
    body: np.ndarray,
    wbc: np.ndarray,
    hand_indices: tuple[int, int, int, int],
    trigger_close_threshold: float,
    squeeze_open_threshold: float,
) -> tuple[np.ndarray, dict[str, int]]:
    if body.ndim != 2 or body.shape[1] != 29 or wbc.shape != (len(body), 23):
        raise ValueError(f"invalid HIW state arrays: body={body.shape} wbc={wbc.shape}")
    events = wbc[:, hand_indices]
    lower = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    upper = np.array([10.0, 1.0, 10.0, 1.0], dtype=np.float32)
    if np.any(events < lower) or np.any(events > upper):
        raise ValueError("HIW observation.state.wbc hand event is outside its declared range")
    hands, stats = _hand_states(events, trigger_close_threshold, squeeze_open_threshold)
    quaternion = Rotation.from_euler("xyz", wbc[:, 3:6]).as_quat(scalar_first=True)
    gravity = projected_gravity_from_quat_wxyz(quaternion)
    state = np.concatenate((body, hands, gravity), axis=1).astype(np.float32, copy=False)
    if state.shape != (len(body), PROPRIO_DIM) or not np.isfinite(state).all():
        raise ValueError(f"invalid generated HIW proprio: {state.shape}")
    gravity_norm = np.linalg.norm(state[:, 31:34], axis=1)
    if not np.allclose(gravity_norm, 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("generated HIW projected gravity is not unit length")
    return state, stats


def _update_state_digest(
    digest: Any,
    episode_id: str,
    body: np.ndarray,
    wbc: np.ndarray,
) -> None:
    digest.update(f"{episode_id}\t{len(body)}\n".encode())
    digest.update(np.asarray(body, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(wbc, dtype="<f4").tobytes(order="C"))


def _convert_proprio(
    source_token_root: Path,
    output_root: Path,
    episodes: dict[str, HIWEpisode],
    hand_indices: tuple[int, int, int, int],
    trigger_close_threshold: float,
    squeeze_open_threshold: float,
) -> tuple[dict[str, int], str]:
    stats: dict[str, int] = {}
    source_digest = hashlib.sha256()
    converted = 0
    for metadata, arrays in _iter_episode_arrays(
        episodes,
        {"observation.state": 29, "observation.state.wbc": 23},
    ):
        body = arrays["observation.state"]
        wbc = arrays["observation.state.wbc"]
        state, episode_stats = _proprio_from_source(
            body, wbc, hand_indices, trigger_close_threshold, squeeze_open_threshold
        )
        token = np.load(
            source_token_root / f"episode_{metadata.episode_id}.npy",
            mmap_mode="r",
        )
        episode_stats["hand_action_disagreement_frames"] = int(
            np.any(state[:, 29:31] != token[:, MOTION_TOKEN_DIM:ACTION_DIM], axis=1).sum()
        )
        atomic_save_npy(output_root / f"episode_{metadata.episode_id}.npy", state)
        _update_state_digest(source_digest, metadata.episode_id, body, wbc)
        _add_stats(stats, episode_stats)
        converted += 1
        if converted % 1000 == 0:
            print(f"converted {converted}/{len(episodes)} HIW proprio episodes", flush=True)
    stats["total_frames"] = sum(episode.length for episode in episodes.values())
    return stats, source_digest.hexdigest()


def _validate_proprio_staging(
    source_token_root: Path,
    output_root: Path,
    episodes: dict[str, HIWEpisode],
    hand_indices: tuple[int, int, int, int],
    trigger_close_threshold: float,
    squeeze_open_threshold: float,
    expected_stats: dict[str, int],
    expected_source_digest: str,
) -> None:
    stats: dict[str, int] = {}
    source_digest = hashlib.sha256()
    validated = 0
    for metadata, arrays in _iter_episode_arrays(
        episodes,
        {"observation.state": 29, "observation.state.wbc": 23},
    ):
        body = arrays["observation.state"]
        wbc = arrays["observation.state.wbc"]
        expected, episode_stats = _proprio_from_source(
            body, wbc, hand_indices, trigger_close_threshold, squeeze_open_threshold
        )
        path = output_root / f"episode_{metadata.episode_id}.npy"
        actual = np.load(path, mmap_mode="r")
        if actual.shape != expected.shape or actual.dtype != np.float32:
            raise ValueError(f"invalid staged proprio: {path} {actual.shape} {actual.dtype}")
        if not np.array_equal(actual, expected):
            raise ValueError(f"source replay mismatch in {path}")
        token = np.load(
            source_token_root / f"episode_{metadata.episode_id}.npy",
            mmap_mode="r",
        )
        episode_stats["hand_action_disagreement_frames"] = int(
            np.any(actual[:, 29:31] != token[:, MOTION_TOKEN_DIM:ACTION_DIM], axis=1).sum()
        )
        _update_state_digest(source_digest, metadata.episode_id, body, wbc)
        _add_stats(stats, episode_stats)
        validated += 1
        if validated % 2000 == 0:
            print(f"validated {validated}/{len(episodes)} HIW proprio episodes", flush=True)
    stats["total_frames"] = sum(episode.length for episode in episodes.values())
    if stats != expected_stats:
        raise ValueError(f"HIW proprio statistics changed during validation: {stats} != {expected_stats}")
    if source_digest.hexdigest() != expected_source_digest:
        raise ValueError("HIW source state digest changed during validation")


def _convert_tokens(
    source_token_root: Path,
    output_root: Path,
    episodes: dict[str, HIWEpisode],
    action_indices: tuple[int, int, int, int],
    trigger_close_threshold: float,
    squeeze_open_threshold: float,
) -> tuple[dict[str, int], str]:
    stats: dict[str, int] = {}
    event_digest = hashlib.sha256()
    converted = 0
    for metadata, events in _iter_episode_events(episodes, action_indices):
        motion_path = source_token_root / f"episode_{metadata.episode_id}.npy"
        motion = np.load(motion_path, mmap_mode="r")
        if motion.shape != (metadata.length, MOTION_TOKEN_DIM) or motion.dtype != np.float32:
            raise ValueError(f"invalid source token during conversion: {motion_path}")
        hands, episode_stats = _hand_states(
            events,
            trigger_close_threshold,
            squeeze_open_threshold,
        )
        token = np.empty((metadata.length, ACTION_DIM), dtype=np.float32)
        token[:, :MOTION_TOKEN_DIM] = motion
        token[:, MOTION_TOKEN_DIM:] = hands
        atomic_save_npy(output_root / f"episode_{metadata.episode_id}.npy", token)
        _update_event_digest(event_digest, metadata.episode_id, events)
        _add_stats(stats, episode_stats)
        converted += 1
        if converted % 1000 == 0:
            print(f"converted {converted}/{len(episodes)} HIW episodes", flush=True)
    stats["total_frames"] = sum(episode.length for episode in episodes.values())
    return stats, event_digest.hexdigest()


def _validate_staging(
    source_token_root: Path,
    output_root: Path,
    episodes: dict[str, HIWEpisode],
    action_indices: tuple[int, int, int, int],
    trigger_close_threshold: float,
    squeeze_open_threshold: float,
    expected_stats: dict[str, int],
    expected_event_digest: str,
) -> None:
    stats: dict[str, int] = {}
    event_digest = hashlib.sha256()
    validated = 0
    for metadata, events in _iter_episode_events(episodes, action_indices):
        motion = np.load(
            source_token_root / f"episode_{metadata.episode_id}.npy",
            mmap_mode="r",
        )
        token_path = output_root / f"episode_{metadata.episode_id}.npy"
        token = np.load(token_path, mmap_mode="r")
        hands, episode_stats = _hand_states(
            events,
            trigger_close_threshold,
            squeeze_open_threshold,
        )
        if token.shape != (metadata.length, ACTION_DIM) or token.dtype != np.float32:
            raise ValueError(f"invalid staged token: {token_path} {token.shape} {token.dtype}")
        if not np.array_equal(token[:, :MOTION_TOKEN_DIM], motion):
            raise ValueError(f"motion dimensions changed in {token_path}")
        if not np.array_equal(token[:, MOTION_TOKEN_DIM:], hands):
            raise ValueError(f"hand event replay mismatch in {token_path}")
        if not hand_binary_is_valid(token) or not np.all(token[0, MOTION_TOKEN_DIM:] == 0.0):
            raise ValueError(f"invalid binary or initial-open hand state in {token_path}")
        _update_event_digest(event_digest, metadata.episode_id, events)
        _add_stats(stats, episode_stats)
        validated += 1
        if validated % 2000 == 0:
            print(f"validated {validated}/{len(episodes)} HIW episodes", flush=True)
    stats["total_frames"] = sum(episode.length for episode in episodes.values())
    if stats != expected_stats:
        raise ValueError(f"hand event statistics changed during validation: {stats} != {expected_stats}")
    if event_digest.hexdigest() != expected_event_digest:
        raise ValueError("hand event digest changed during validation")


def _load_existing_cache(cache_root: Path) -> tuple[dict[str, Any], str]:
    success_path = cache_root / "_SUCCESS"
    manifest_path = cache_root / "meta" / "conversion_manifest.json"
    expected_files = {success_path, manifest_path}
    actual_files = {path for path in cache_root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            f"unsupported existing HIW cache layout: "
            f"{sorted(str(path.relative_to(cache_root)) for path in actual_files)}"
        )
    manifest_sha256 = _sha256(manifest_path)
    if success_path.read_text(encoding="utf-8").strip() != manifest_sha256:
        raise ValueError("existing HIW cache _SUCCESS does not match its manifest")
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_sha256


def _validate_existing_manifest(
    manifest: dict[str, Any],
    source_resolved: Path,
    info_path: Path,
    tasks_path: Path,
    info: dict[str, Any],
    episode_files: list[dict[str, str | int]],
    episode_digest: str,
    source_token_digest: str,
) -> None:
    expected = {
        "schema_version": 2,
        "dataset_id": "hiw500",
        "motion_dim": MOTION_TOKEN_DIM,
        "hand_mode": "masked_absent",
        "state_mode": "masked_absent",
        "source_fps": 30,
        "logical_control_hz": 50,
        "resample_mode": "none",
        "video_backend": "pyav_libdav1d",
        "video_key": "observation.images.head",
        "source_total_episodes": int(info["total_episodes"]),
        "source_total_frames": int(info["total_frames"]),
        "source_info_sha256": _sha256(info_path),
        "source_tasks_sha256": _sha256(tasks_path),
        "episode_metadata_files": episode_files,
        "episode_metadata_digest": episode_digest,
        "sonic_token_inventory_digest": source_token_digest,
    }
    mismatched = sorted(key for key, value in expected.items() if manifest.get(key) != value)
    if Path(manifest.get("source_resolved_root", "")) != source_resolved:
        mismatched.append("source_resolved_root")
    if mismatched:
        raise ValueError(f"existing HIW cache no longer matches the 64-D source: {mismatched}")


def _validate_66d_manifest(
    manifest: dict[str, Any],
    source_resolved: Path,
    info_path: Path,
    tasks_path: Path,
    episode_files: list[dict[str, str | int]],
    episode_digest: str,
    token_digest: str,
) -> None:
    expected = {
        "schema_version": 3,
        "dataset_id": "hiw500",
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "action_dim": ACTION_DIM,
        "state_mode": "masked_absent",
        "source_info_sha256": _sha256(info_path),
        "source_tasks_sha256": _sha256(tasks_path),
        "episode_metadata_files": episode_files,
        "episode_metadata_digest": episode_digest,
        "output_token_inventory_digest": token_digest,
    }
    mismatched = sorted(key for key, value in expected.items() if manifest.get(key) != value)
    if Path(manifest.get("source_resolved_root", "")) != source_resolved:
        mismatched.append("source_resolved_root")
    if Path(manifest.get("source_token_path", "")) != source_resolved / "sonic_tokens":
        mismatched.append("source_token_path")
    if mismatched:
        raise ValueError(f"existing HIW cache no longer matches the 66-D source: {mismatched}")


def _restore_directory(
    canonical: Path,
    staging: Path,
    backup: Path,
    *,
    new_published: bool,
    old_moved: bool,
) -> list[str]:
    errors: list[str] = []
    if new_published:
        try:
            os.replace(canonical, staging)
        except OSError as exc:
            errors.append(f"{canonical} -> {staging}: {exc}")
    if old_moved:
        try:
            os.replace(backup, canonical)
        except OSError as exc:
            errors.append(f"{backup} -> {canonical}: {exc}")
    return errors


def prepare_cache(
    data_root: Path,
    cache_root: Path,
    *,
    trigger_close_threshold: float = 5.0,
    squeeze_open_threshold: float = 0.5,
    replace_source_tokens: bool = False,
) -> dict[str, Any]:
    if (
        not np.isfinite(trigger_close_threshold)
        or not 0.0 < trigger_close_threshold < 10.0
        or not np.isfinite(squeeze_open_threshold)
        or not 0.0 < squeeze_open_threshold < 1.0
    ):
        raise ValueError("thresholds must be finite and strictly inside their declared ranges")
    if not replace_source_tokens:
        raise ValueError("formal HIW conversion requires --replace-source-tokens")

    data_root = data_root.absolute()
    cache_root = cache_root.absolute()
    source_resolved = data_root.resolve(strict=True)
    source_token_root = source_resolved / "sonic_tokens"
    if not cache_root.is_dir():
        raise FileNotFoundError(f"existing HIW cache is required: {cache_root}")
    if _is_within(cache_root.resolve(strict=True), source_resolved):
        raise ValueError("cache root must not be inside the HIW source")
    required_source = [
        data_root / "meta" / "info.json",
        data_root / "meta" / "tasks.parquet",
        data_root / "data",
        data_root / "videos",
        data_root / "sonic_tokens",
    ]
    missing = [str(path) for path in required_source if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete HIW source; missing: {missing}")
    try:
        av.codec.Codec("libdav1d", "r")
    except Exception as exc:
        raise RuntimeError("PyAV does not expose the required libdav1d decoder") from exc

    source_lock = source_resolved / ".sonic_tokens.upgrade.lock"
    cache_lock = cache_root.parent / f".{cache_root.name}.upgrade.lock"
    owned_locks: list[Path] = []
    token_staging: Path | None = None
    cache_staging: Path | None = None
    swap_started = False
    try:
        for lock in (source_lock, cache_lock):
            lock.mkdir()
            owned_locks.append(lock)

        stale_staging = sorted(source_resolved.glob(".sonic_tokens.66d.staging-*"))
        stale_staging += sorted(cache_root.parent.glob(f".{cache_root.name}.staging-*"))
        if stale_staging:
            raise FileExistsError(f"pre-existing HIW staging directories: {stale_staging}")

        info_path = data_root / "meta" / "info.json"
        tasks_path = data_root / "meta" / "tasks.parquet"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        source_fps = info.get("fps")
        if source_fps != 30:
            raise ValueError(f"HIW source fps must be 30, got {source_fps}")
        action_indices, action_names = _action_indices(info)
        episodes_root, episodes_origin = resolve_hiw_episode_metadata_root(data_root)
        episodes = load_hiw_episode_metadata(data_root, episodes_root)
        episode_lengths = {
            episode_id: episode.length
            for episode_id, episode in episodes.items()
        }
        if len(episodes) != int(info["total_episodes"]) or sum(episode_lengths.values()) != int(
            info["total_frames"]
        ):
            raise ValueError("HIW episode metadata totals do not match meta/info.json")

        episode_files, episode_digest = _file_inventory(episodes_root)
        source_token_digest, source_token_bytes = _token_inventory(
            source_token_root,
            episode_lengths,
            MOTION_TOKEN_DIM,
        )
        old_manifest, old_manifest_sha256 = _load_existing_cache(cache_root)
        _validate_existing_manifest(
            old_manifest,
            source_resolved,
            info_path,
            tasks_path,
            info,
            episode_files,
            episode_digest,
            source_token_digest,
        )

        estimated_output_bytes = int(info["total_frames"]) * ACTION_DIM * 4
        required_free_bytes = int(estimated_output_bytes * 1.3)
        available_bytes = shutil.disk_usage(source_resolved).free
        if available_bytes < required_free_bytes:
            raise OSError(
                f"insufficient source filesystem space: {available_bytes} < {required_free_bytes}"
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        token_backup = source_resolved / f"sonic_tokens.64d.backup-{timestamp}"
        cache_backup = cache_root.with_name(f"{cache_root.name}.masked-backup-{timestamp}")
        if token_backup.exists() or cache_backup.exists():
            raise FileExistsError(f"backup target already exists: {token_backup}, {cache_backup}")

        token_staging = Path(
            tempfile.mkdtemp(
                prefix=".sonic_tokens.66d.staging-",
                dir=source_resolved,
            )
        )
        cache_staging = Path(
            tempfile.mkdtemp(
                prefix=f".{cache_root.name}.staging-",
                dir=cache_root.parent,
            )
        )
        started = time.monotonic()
        event_stats, event_digest = _convert_tokens(
            source_token_root,
            token_staging,
            episodes,
            action_indices,
            trigger_close_threshold,
            squeeze_open_threshold,
        )
        _validate_staging(
            source_token_root,
            token_staging,
            episodes,
            action_indices,
            trigger_close_threshold,
            squeeze_open_threshold,
            event_stats,
            event_digest,
        )
        source_token_digest_after, _ = _token_inventory(
            source_token_root,
            episode_lengths,
            MOTION_TOKEN_DIM,
        )
        if source_token_digest_after != source_token_digest:
            raise ValueError("64-D source token digest changed during conversion")
        output_token_digest, output_token_bytes = _token_inventory(
            token_staging,
            episode_lengths,
            ACTION_DIM,
            validate_hands=True,
        )

        total_frames = int(info["total_frames"])
        manifest = {
            "schema_version": 3,
            "dataset_id": "hiw500",
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "motion_dim": MOTION_TOKEN_DIM,
            "hand_dim": ACTION_DIM - MOTION_TOKEN_DIM,
            "action_dim": ACTION_DIM,
            "hand_mode": "event_latched_v1",
            "hand_polarity": HAND_POLARITY,
            "initial_hand_state": "open",
            "trigger_close_threshold": trigger_close_threshold,
            "squeeze_open_threshold": squeeze_open_threshold,
            "simultaneous_event_policy": "open_wins",
            "token_storage": "source_sonic_tokens_in_place",
            "replacement_mode": "directory_swap_with_retained_backup",
            "source_root": str(data_root),
            "source_resolved_root": str(source_resolved),
            "source_token_path": str(source_token_root),
            "source_token_backup_path": str(token_backup),
            "old_cache_backup_path": str(cache_backup),
            "source_info_sha256": _sha256(info_path),
            "source_tasks_sha256": _sha256(tasks_path),
            "source_total_episodes": int(info["total_episodes"]),
            "source_total_frames": total_frames,
            "source_action_feature_names": action_names,
            "episode_metadata_origin": episodes_origin,
            "episode_metadata_repo_id": HIW_EPISODES_REPO_ID,
            "episode_metadata_files": episode_files,
            "episode_metadata_digest": episode_digest,
            "source_motion_token_inventory_digest": source_token_digest,
            "source_hand_event_digest": event_digest,
            "output_token_inventory_digest": output_token_digest,
            "sonic_token_inventory_digest": output_token_digest,
            "source_motion_token_total_bytes": source_token_bytes,
            "output_token_total_bytes": output_token_bytes,
            "source_fps": 30,
            "logical_control_hz": 50,
            "resample_mode": "none",
            "state_mode": "masked_absent",
            "video_backend": "pyav_libdav1d",
            "video_key": "observation.images.head",
            "hand_event_counts": {
                side: {
                    "close": event_stats[f"{side}_close_events"],
                    "open": event_stats[f"{side}_open_events"],
                    "simultaneous": event_stats[f"{side}_simultaneous_events"],
                    "closed_frames": event_stats[f"{side}_closed_frames"],
                    "closed_ratio": event_stats[f"{side}_closed_frames"] / total_frames,
                }
                for side in ("left", "right")
            },
            "verified_video_episode_ids": [4, 12, 18],
            "old_cache_manifest_sha256": old_manifest_sha256,
            "conversion_seconds": round(time.monotonic() - started, 3),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = cache_staging / "meta" / "conversion_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (cache_staging / "_SUCCESS").write_text(
            _sha256(manifest_path) + "\n",
            encoding="utf-8",
        )
        staged_manifest, _ = _load_existing_cache(cache_staging)
        if staged_manifest != manifest:
            raise ValueError("staged HIW cache manifest changed after writing")

        swap_started = True
        old_token_moved = False
        new_token_published = False
        old_cache_moved = False
        new_cache_published = False
        try:
            os.replace(source_token_root, token_backup)
            old_token_moved = True
            os.replace(token_staging, source_token_root)
            new_token_published = True
            os.replace(cache_root, cache_backup)
            old_cache_moved = True
            os.replace(cache_staging, cache_root)
            new_cache_published = True

            final_digest, final_bytes = _token_inventory(
                source_token_root,
                episode_lengths,
                ACTION_DIM,
                validate_hands=True,
            )
            final_manifest, _ = _load_existing_cache(cache_root)
            if (
                final_digest != output_token_digest
                or final_bytes != output_token_bytes
                or final_manifest != manifest
            ):
                raise ValueError("published HIW token/cache validation failed")
        except BaseException as exc:
            rollback_errors = _restore_directory(
                cache_root,
                cache_staging,
                cache_backup,
                new_published=new_cache_published,
                old_moved=old_cache_moved,
            )
            rollback_errors += _restore_directory(
                source_token_root,
                token_staging,
                token_backup,
                new_published=new_token_published,
                old_moved=old_token_moved,
            )
            if rollback_errors:
                raise RuntimeError(
                    "HIW rollback failed; preserve all paths: " + "; ".join(rollback_errors)
                ) from exc
            raise

        print(f"published 66-D HIW tokens: {source_token_root}")
        print(f"retained 64-D backup: {token_backup}")
        print(f"published HIW cache: {cache_root}")
        print(f"retained old cache: {cache_backup}")
        return manifest
    except BaseException:
        if not swap_started:
            for staging in (token_staging, cache_staging):
                if staging is not None and staging.exists():
                    shutil.rmtree(staging)
        raise
    finally:
        for lock in reversed(owned_locks):
            try:
                lock.rmdir()
            except OSError as exc:
                print(f"warning: failed to release owned lock {lock}: {exc}")


def prepare_proprio(data_root: Path, cache_root: Path) -> dict[str, Any]:
    data_root = data_root.absolute()
    cache_root = cache_root.absolute()
    source_resolved = data_root.resolve(strict=True)
    token_root = source_resolved / "sonic_tokens"
    proprio_root = source_resolved / "g1_proprio"
    if proprio_root.exists():
        raise FileExistsError(f"refusing to replace existing HIW proprio: {proprio_root}")
    if not cache_root.is_dir():
        raise FileNotFoundError(f"existing HIW cache is required: {cache_root}")
    if _is_within(cache_root.resolve(strict=True), source_resolved):
        raise ValueError("cache root must not be inside the HIW source")

    required = [
        data_root / "meta" / "info.json",
        data_root / "meta" / "tasks.parquet",
        data_root / "data",
        token_root,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete HIW source; missing: {missing}")

    source_lock = source_resolved / ".g1_proprio.publish.lock"
    cache_lock = cache_root.parent / f".{cache_root.name}.upgrade.lock"
    owned_locks: list[Path] = []
    proprio_staging: Path | None = None
    cache_staging: Path | None = None
    swap_started = False
    try:
        for lock in (source_lock, cache_lock):
            lock.mkdir()
            owned_locks.append(lock)

        stale = sorted(source_resolved.glob(".g1_proprio.staging-*"))
        stale += sorted(cache_root.parent.glob(f".{cache_root.name}.staging-*"))
        if stale:
            raise FileExistsError(f"pre-existing HIW staging directories: {stale}")

        info_path = data_root / "meta" / "info.json"
        tasks_path = data_root / "meta" / "tasks.parquet"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("robot_type") != "unitree_g1" or info.get("fps") != 30:
            raise ValueError("HIW proprio requires unitree_g1 data at 30 fps")
        hand_indices, action_names = _action_indices(info)
        body_names = _feature_names(info, "observation.state", 29)
        wbc_names = _feature_names(info, "observation.state.wbc", 23)
        if tuple(body_names) != BODY_STATE_NAMES:
            raise ValueError("HIW observation.state does not match Huge's first 29 proprio dimensions")
        if wbc_names != action_names:
            raise ValueError("HIW observation.state.wbc and action layouts differ")

        episodes_root, episodes_origin = resolve_hiw_episode_metadata_root(data_root)
        episodes = load_hiw_episode_metadata(data_root, episodes_root)
        episode_lengths = {episode_id: episode.length for episode_id, episode in episodes.items()}
        if len(episodes) != int(info["total_episodes"]) or sum(episode_lengths.values()) != int(
            info["total_frames"]
        ):
            raise ValueError("HIW episode metadata totals do not match meta/info.json")
        episode_files, episode_digest = _file_inventory(episodes_root)
        token_digest, token_bytes = _token_inventory(
            token_root, episode_lengths, ACTION_DIM, validate_hands=True
        )
        old_manifest, old_manifest_sha256 = _load_existing_cache(cache_root)
        _validate_66d_manifest(
            old_manifest, source_resolved, info_path, tasks_path,
            episode_files, episode_digest, token_digest,
        )
        trigger_threshold = float(old_manifest["trigger_close_threshold"])
        squeeze_threshold = float(old_manifest["squeeze_open_threshold"])

        estimated_bytes = int(info["total_frames"]) * PROPRIO_DIM * 4
        if shutil.disk_usage(source_resolved).free < int(estimated_bytes * 1.1):
            raise OSError("insufficient source filesystem space for HIW proprio")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        cache_backup = cache_root.with_name(f"{cache_root.name}.no-proprio-backup-{timestamp}")
        if cache_backup.exists():
            raise FileExistsError(f"backup target already exists: {cache_backup}")
        proprio_staging = Path(tempfile.mkdtemp(prefix=".g1_proprio.staging-", dir=source_resolved))
        cache_staging = Path(
            tempfile.mkdtemp(prefix=f".{cache_root.name}.staging-", dir=cache_root.parent)
        )

        started = time.monotonic()
        state_stats, source_state_digest = _convert_proprio(
            token_root, proprio_staging, episodes, hand_indices,
            trigger_threshold, squeeze_threshold,
        )
        _validate_proprio_staging(
            token_root, proprio_staging, episodes, hand_indices,
            trigger_threshold, squeeze_threshold, state_stats, source_state_digest,
        )
        proprio_digest, proprio_bytes = _token_inventory(
            proprio_staging, episode_lengths, PROPRIO_DIM
        )
        token_digest_after, _ = _token_inventory(
            token_root, episode_lengths, ACTION_DIM, validate_hands=True
        )
        if token_digest_after != token_digest:
            raise ValueError("66-D HIW token digest changed during proprio conversion")

        manifest = {
            **old_manifest,
            "schema_version": 4,
            "state_mode": "source_observation_state_29_plus_wbc_v1",
            "proprio_dim": PROPRIO_DIM,
            "proprio_layout_version": PROPRIO_LAYOUT_VERSION,
            "proprio_storage": "source_g1_proprio",
            "proprio_publication_mode": "new_directory_atomic_publish",
            "source_proprio_path": str(proprio_root),
            "source_state_feature_names": body_names,
            "source_wbc_feature_names": wbc_names,
            "state_hand_mode": "observation_state_wbc_event_latched_v1",
            "projected_gravity_mode": "pivot_rpy_extrinsic_xyz_v1",
            "source_state_digest": source_state_digest,
            "output_proprio_inventory_digest": proprio_digest,
            "proprio_inventory_digest": proprio_digest,
            "output_proprio_total_bytes": proprio_bytes,
            "source_token_total_bytes": token_bytes,
            "state_hand_event_counts": {
                side: {
                    "close": state_stats[f"{side}_close_events"],
                    "open": state_stats[f"{side}_open_events"],
                    "simultaneous": state_stats[f"{side}_simultaneous_events"],
                    "closed_frames": state_stats[f"{side}_closed_frames"],
                }
                for side in ("left", "right")
            },
            "state_hand_action_disagreement_frames": state_stats[
                "hand_action_disagreement_frames"
            ],
            "episode_metadata_origin": episodes_origin,
            "pre_proprio_cache_manifest_sha256": old_manifest_sha256,
            "proprio_conversion_seconds": round(time.monotonic() - started, 3),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = cache_staging / "meta" / "conversion_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (cache_staging / "_SUCCESS").write_text(
            _sha256(manifest_path) + "\n", encoding="utf-8"
        )
        staged_manifest, _ = _load_existing_cache(cache_staging)
        if staged_manifest != manifest:
            raise ValueError("staged HIW proprio manifest changed after writing")

        swap_started = True
        new_proprio_published = False
        old_cache_moved = False
        new_cache_published = False
        try:
            os.replace(proprio_staging, proprio_root)
            new_proprio_published = True
            os.replace(cache_root, cache_backup)
            old_cache_moved = True
            os.replace(cache_staging, cache_root)
            new_cache_published = True
            final_digest, final_bytes = _token_inventory(
                proprio_root, episode_lengths, PROPRIO_DIM
            )
            final_manifest, _ = _load_existing_cache(cache_root)
            if (
                final_digest != proprio_digest
                or final_bytes != proprio_bytes
                or final_manifest != manifest
            ):
                raise ValueError("published HIW proprio/cache validation failed")
        except BaseException as exc:
            rollback_errors = _restore_directory(
                cache_root, cache_staging, cache_backup,
                new_published=new_cache_published, old_moved=old_cache_moved,
            )
            rollback_errors += _restore_directory(
                proprio_root, proprio_staging, proprio_root.with_name("unused"),
                new_published=new_proprio_published, old_moved=False,
            )
            if rollback_errors:
                raise RuntimeError(
                    "HIW proprio rollback failed; preserve all paths: "
                    + "; ".join(rollback_errors)
                ) from exc
            raise

        print(f"published 34-D HIW proprio: {proprio_root}")
        print(f"published HIW cache: {cache_root}")
        print(f"retained pre-proprio cache: {cache_backup}")
        return manifest
    except BaseException:
        if not swap_started:
            for staging in (proprio_staging, cache_staging):
                if staging is not None and staging.exists():
                    shutil.rmtree(staging)
        raise
    finally:
        for lock in reversed(owned_locks):
            try:
                lock.rmdir()
            except OSError as exc:
                print(f"warning: failed to release owned lock {lock}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("resource/datasets/HIW-500"))
    parser.add_argument("--cache-root", type=Path, default=Path("resource/datasets/HIW-500-cache"))
    parser.add_argument("--trigger-close-threshold", type=float, default=5.0)
    parser.add_argument("--squeeze-open-threshold", type=float, default=0.5)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replace-source-tokens", action="store_true")
    mode.add_argument("--publish-source-proprio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.publish_source_proprio:
        prepare_proprio(args.data_root, args.cache_root)
        return
    prepare_cache(
        args.data_root,
        args.cache_root,
        trigger_close_threshold=args.trigger_close_threshold,
        squeeze_open_threshold=args.squeeze_open_threshold,
        replace_source_tokens=args.replace_source_tokens,
    )


if __name__ == "__main__":
    main()

"""Extract 66-D SONIC teleop tokens from three-point pose datasets.

The default mode is read-only preflight. ``--pilot`` writes a small cache
under ``/tmp`` and ``--publish-full`` is required for a full atomic cache.
Source frames are never resampled: one 30 Hz source frame becomes one token
that downstream code interprets at 50 Hz.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

sys.dont_write_bytecode = True

from huge.data.preprocess import atomic_save_npy


ENCODER_INPUT_DIM = 1762
LATENT_DIM = 64
ACTION_DIM = 66
SOURCE_HZ = 30.0
INTERPRETED_HZ = 50.0
DEFAULT_ENCODER_MODEL = Path("resource/model/sonic/model_encoder_dynbatch.onnx")
# SONIC live teleop uses a 0.05 m torso offset plus a 0.35 m neck vector
# for the upright third tracking point.
DEFAULT_CANONICAL_HEAD_Z = 0.40
TORSO_LINK_OFFSET_Z = 0.05
# The G1 shoulder-to-tracking-point chain reaches about 0.73 m from the
# canonical head point. Keep a small tolerance for retargeting and head tilt.
DEFAULT_BEHAVIOR_MAX_WRIST_HEAD_REACH = 0.80
G1_WRIST_REFERENCE_POSITIONS = np.asarray(
    [
        [0.3797694914, 0.1236269337, 0.0952214277],
        [0.3797694914, -0.1236169337, 0.0952214277],
    ],
    dtype=np.float32,
)
G1_WRIST_REFERENCE_ORIENTATIONS = np.asarray(
    [
        [0.9999999946, 0.0000292745, 0.0000274716, -0.0000957958],
        [0.9999999946, -0.0000292745, 0.0000274716, 0.0000957958],
    ],
    dtype=np.float32,
)
LOWER_BODY = np.asarray(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0] * 2,
    dtype=np.float32,
)
IDENTITY_6D = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
MANO_TIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
LEGACY_ARIA_TIPS = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
LEGACY_ARIA_RIGHT_WRIST_CORRECTION_WXYZ = np.asarray(
    [0.0, 0.0, 1.0, 0.0], dtype=np.float32
)
AUTOTUNE_FRAMES = 1024
AUTOTUNE_MIN_GAIN = 0.10
AUTOTUNE_MIN_FREE_RATIO = 0.20
DEFAULT_PREFLIGHT_WORKERS = 16
PREFLIGHT_PROGRESS_INTERVAL = 100
WORK_QUEUE_SIZE = 128

DATASET_DEFAULTS = {
    "behavior1k": (
        Path("resource/datasets/behavior-1k-2025-challenge"),
        Path("resource/datasets/behavior-1k-2025-challenge-cache"),
    ),
    "egoverse": (
        Path("resource/datasets/egoverse"),
        Path("resource/datasets/egoverse-cache"),
    ),
    "agibotworld": (
        Path("resource/datasets/agibotworld-beta-rlds"),
        Path("resource/datasets/agibotworld-beta-rlds-cache"),
    ),
}


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    calibration_start: int | None = None
    calibration_end: int | None = None


@dataclass(frozen=True)
class Episode:
    dataset: str
    source: Path
    relative_source: str
    stable_base: str
    source_frames: int
    source_hz: float
    main_task: str
    segments: tuple[Segment, ...]
    valid_frames: int
    stats: dict[str, Any]


@dataclass(frozen=True)
class Preflight:
    episodes: tuple[Episode, ...]
    exclusions: tuple[dict[str, str], ...]
    scanned_sources: tuple[Path, ...]
    inventory_fingerprint: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_fingerprint(root: Path, paths: tuple[Path, ...] | list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        stat = path.stat()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _safe_id(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"unsafe stable id: {value!r}")
    return normalized


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    if values.shape[-1] != 4:
        raise ValueError(f"quaternions must end in dimension 4, got {values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("zero quaternion in a preflight-approved segment")
    values = values / norms
    flattened = values.reshape(len(values), -1, 4)
    for frame in range(1, len(flattened)):
        flip = np.sum(flattened[frame - 1] * flattened[frame], axis=-1) < 0
        flattened[frame, flip] *= -1
    return values.astype(np.float32)


def _to_wxyz(quaternions: np.ndarray, order: str) -> np.ndarray:
    values = np.asarray(quaternions)
    if order == "xyzw":
        values = values[..., [3, 0, 1, 2]]
    elif order != "wxyz":
        raise ValueError(f"unsupported quaternion order: {order}")
    return _normalize_quaternions(values)


def _quat_conjugate(quaternions: np.ndarray) -> np.ndarray:
    result = np.asarray(quaternions, dtype=np.float32).copy()
    result[..., 1:] *= -1
    return result


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = np.broadcast_arrays(left, right)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    ).astype(np.float32)


def _mean_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    values = _normalize_quaternions(quaternions)
    means = values.mean(axis=0)
    norms = np.linalg.norm(means, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("calibration quaternion mean is ambiguous")
    return (means / norms).astype(np.float32)


def _behavior_calibration_segment(
    poses: np.ndarray,
    segment: Segment,
    *,
    frames: int,
    max_position_delta: float,
    max_angle_degrees: float,
) -> Segment:
    source = poses[segment.start : segment.end]
    window = min(frames, len(source))
    quaternions = _to_wxyz(source[:, :, 3:7], "xyzw")
    for offset in range(len(source) - window + 1):
        start = segment.start + offset
        end = start + window
        positions = source[offset : offset + window, :, :3]
        position_center = np.median(positions, axis=0)
        position_delta = np.max(
            np.linalg.norm(positions - position_center, axis=-1)
        )
        window_quaternions = quaternions[offset : offset + window]
        quaternion_center = _mean_quaternions_wxyz(window_quaternions)
        dots = np.abs(
            np.sum(window_quaternions * quaternion_center[None, :, :], axis=-1)
        )
        angle_delta = np.degrees(
            np.max(2.0 * np.arccos(np.clip(dots, 0.0, 1.0)))
        )
        if position_delta <= max_position_delta and angle_delta <= max_angle_degrees:
            return Segment(segment.start, segment.end, start, end)
    raise ValueError(
        f"no stable {window}-frame calibration window in "
        f"[{segment.start},{segment.end})"
    )


def _calibrate_behavior_three_points(
    poses: np.ndarray,
    segment: Segment,
    *,
    canonical_head_z: float,
    max_wrist_head_reach: float,
) -> tuple[np.ndarray, np.ndarray]:
    if segment.calibration_start is None or segment.calibration_end is None:
        raise ValueError("BEHAVIOR segment is missing a calibration window")
    source = np.asarray(poses[segment.start : segment.end], dtype=np.float32)
    quaternions = _to_wxyz(source[:, :, 3:7], "xyzw")
    calibration = slice(
        segment.calibration_start - segment.start,
        segment.calibration_end - segment.start,
    )
    head_inverse = _quat_conjugate(
        _mean_quaternions_wxyz(quaternions[calibration, 2])
    )
    head_inverse_rotation = Rotation.from_quat(head_inverse, scalar_first=True)
    positions = np.einsum(
        "ij,tpj->tpi",
        head_inverse_rotation.as_matrix(),
        source[:, :, :3],
    ).astype(np.float32)
    orientations = _quat_multiply(head_inverse[None, None, :], quaternions)

    wrist_relative_to_head = positions[:, :2] - positions[:, 2:3]
    calibration_relative = np.median(
        wrist_relative_to_head[calibration], axis=0
    )
    source_reach = np.linalg.norm(calibration_relative, axis=-1)
    if np.any(source_reach < 1e-6):
        raise ValueError("BEHAVIOR calibration wrist-to-head reach is degenerate")
    canonical_head = np.asarray(
        [0.0, 0.0, canonical_head_z], dtype=np.float32
    )
    g1_reach = np.linalg.norm(
        G1_WRIST_REFERENCE_POSITIONS - canonical_head, axis=-1
    )
    wrist_scales = g1_reach / source_reach
    positions[:, :2] = G1_WRIST_REFERENCE_POSITIONS[None, :, :] + (
        wrist_scales[None, :, None]
        * (wrist_relative_to_head - calibration_relative[None, :, :])
    )
    wrist_initial = _mean_quaternions_wxyz(orientations[calibration, :2])
    wrist_rotation_offsets = _quat_multiply(
        G1_WRIST_REFERENCE_ORIENTATIONS,
        _quat_conjugate(wrist_initial),
    )
    orientations[:, :2] = _quat_multiply(
        wrist_rotation_offsets[None, :, :],
        orientations[:, :2],
    )

    head_rotation = Rotation.from_quat(orientations[:, 2], scalar_first=True)
    neck_length = canonical_head_z - TORSO_LINK_OFFSET_Z
    positions[:, 2] = np.asarray([0.0, 0.0, TORSO_LINK_OFFSET_Z]) + (
        neck_length * head_rotation.apply([0.0, 0.0, 1.0])
    )
    max_reach = float(
        np.linalg.norm(
            positions[:, :2] - positions[:, 2:3], axis=-1
        ).max()
    )
    if max_reach > max_wrist_head_reach:
        raise ValueError(
            "retargeted BEHAVIOR wrist-to-head reach "
            f"{max_reach:.3f} m exceeds {max_wrist_head_reach:.3f} m"
        )
    return positions, _normalize_quaternions(orientations)


def _canonicalize_three_points(
    positions: np.ndarray,
    source_quaternions: np.ndarray,
    *,
    quaternion_order: str,
    canonical_head_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (3, 3):
        raise ValueError(f"three-point positions must have shape [T,3,3], got {positions.shape}")
    quaternions = _to_wxyz(source_quaternions, quaternion_order)
    if quaternions.shape != (len(positions), 3, 4):
        raise ValueError(
            f"three-point orientations must have shape [T,3,4], got {quaternions.shape}"
        )

    head = quaternions[:, 2]
    yaw = np.arctan2(
        2.0 * (head[:, 0] * head[:, 3] + head[:, 1] * head[:, 2]),
        1.0 - 2.0 * (head[:, 2] ** 2 + head[:, 3] ** 2),
    )
    yaw_quaternions = np.stack(
        (
            np.cos(yaw / 2.0),
            np.zeros_like(yaw),
            np.zeros_like(yaw),
            np.sin(yaw / 2.0),
        ),
        axis=-1,
    ).astype(np.float32)
    yaw_rotation = Rotation.from_quat(yaw_quaternions, scalar_first=True)
    canonical_head = np.tile(
        np.asarray([0.0, 0.0, canonical_head_z], dtype=np.float32),
        (len(positions), 1),
    )
    root_position = positions[:, 2] - yaw_rotation.apply(canonical_head)
    relative = positions - root_position[:, None, :]
    root_matrix = yaw_rotation.as_matrix()
    local_positions = np.einsum("tji,tpj->tpi", root_matrix, relative)
    local_orientations = _quat_multiply(
        _quat_conjugate(yaw_quaternions)[:, None, :],
        quaternions,
    )
    local_orientations = _normalize_quaternions(local_orientations)
    if not np.isfinite(local_positions).all():
        raise ValueError("canonical three-point positions contain non-finite values")
    return local_positions.astype(np.float32), local_orientations.astype(np.float32)


def build_teleop_encoder_inputs(
    positions: np.ndarray,
    orientations_wxyz: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    orientations = _normalize_quaternions(orientations_wxyz)
    if positions.shape != (len(positions), 3, 3):
        raise ValueError(f"positions must have shape [T,3,3], got {positions.shape}")
    if orientations.shape != (len(positions), 3, 4):
        raise ValueError(f"orientations must have shape [T,3,4], got {orientations.shape}")

    result = np.zeros((len(positions), ENCODER_INPUT_DIM), dtype=np.float32)
    result[:, 0] = 1.0
    result[:, 595:601] = IDENTITY_6D
    result[:, 661:781] = np.tile(LOWER_BODY, 10)
    result[:, 901:910] = positions.reshape(len(positions), 9)
    result[:, 910:922] = orientations.reshape(len(positions), 12)
    return result


def _segment_ranges(
    valid: np.ndarray,
    timestamps: np.ndarray | None,
    *,
    min_source_frames: int,
    max_gap_factor: float,
) -> tuple[Segment, ...]:
    valid = np.asarray(valid, dtype=bool)
    gap_after = np.zeros(max(0, len(valid) - 1), dtype=bool)
    if timestamps is not None and len(timestamps) > 1:
        deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
        positive = deltas[deltas > 0]
        if positive.size:
            gap_after = (deltas <= 0) | (deltas > np.median(positive) * max_gap_factor)
        else:
            gap_after[:] = True
    segments: list[Segment] = []
    start = 0
    while start < len(valid):
        if not valid[start]:
            start += 1
            continue
        end = start + 1
        while end < len(valid) and valid[end] and not gap_after[end - 1]:
            end += 1
        if end - start >= min_source_frames:
            segments.append(Segment(start, end))
        start = end
    return tuple(segments)


def _pose_stats(
    poses: np.ndarray,
    hands: np.ndarray,
    quaternion_order: str,
) -> dict[str, Any]:
    positions = poses[:, :, :3]
    quaternions = poses[:, :, 3:7]
    norms = np.linalg.norm(quaternions, axis=-1)
    wrist_distance = np.linalg.norm(positions[:, :2] - positions[:, 2:3], axis=-1)
    return {
        "pose_shape": list(poses.shape),
        "pose_dtype": str(poses.dtype),
        "source_quaternion_order": quaternion_order,
        "quaternion_norm_min": float(np.nanmin(norms)),
        "quaternion_norm_max": float(np.nanmax(norms)),
        "position_min": np.nanmin(positions, axis=(0, 1)).tolist(),
        "position_median": np.nanmedian(positions, axis=(0, 1)).tolist(),
        "position_max": np.nanmax(positions, axis=(0, 1)).tolist(),
        "wrist_to_head_distance_median": np.nanmedian(wrist_distance, axis=0).tolist(),
        "hand_value_min": np.nanmin(hands, axis=0).tolist(),
        "hand_value_max": np.nanmax(hands, axis=0).tolist(),
    }


def _behavior_root(root: Path) -> Path:
    candidate = root / "v1"
    return candidate if candidate.is_dir() else root


def _read_behavior_arrays(path: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as parquet

    columns = (
        "timestamp",
        "episode_index",
        "index",
        "observation.cam_rel_poses",
        "action",
    )
    table = parquet.read_table(path, columns=list(columns))

    def array(name: str, dtype: np.dtype[Any]) -> np.ndarray:
        return np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype)

    timestamps = array("timestamp", np.float64).reshape(-1)
    episode_indices = array("episode_index", np.int64).reshape(-1)
    indices = array("index", np.int64).reshape(-1)
    poses = array("observation.cam_rel_poses", np.float32)
    actions = array("action", np.float32)
    frames = len(timestamps)
    if poses.shape != (frames, 21) or actions.shape != (frames, 23):
        raise ValueError(
            f"unexpected BEHAVIOR arrays: poses={poses.shape}, actions={actions.shape}"
        )
    if indices.shape != (frames,) or not np.array_equal(
        indices, np.arange(frames, dtype=np.int64)
    ):
        raise ValueError("BEHAVIOR index must be contiguous and zero-based per episode")
    if episode_indices.shape != (frames,) or np.unique(episode_indices).size != 1:
        raise ValueError("BEHAVIOR episode_index is not constant inside the Parquet file")
    return {
        "timestamps": timestamps,
        "episode_indices": episode_indices,
        "poses": poses.reshape(frames, 3, 7),
        "actions": actions,
    }


def _behavior_preflight_jobs(
    root: Path,
    *,
    min_source_frames: int,
    max_gap_factor: float,
    left_gripper_index: int,
    right_gripper_index: int,
    calibration_frames: int,
    calibration_max_position_delta: float,
    calibration_max_angle_degrees: float,
    canonical_head_z: float,
    max_wrist_head_reach: float,
) -> tuple[Path, list[dict[str, Any]], tuple[Path, ...], int]:
    version_root = _behavior_root(root)
    metadata = version_root / "meta/episodes.jsonl"
    info = version_root / "meta/info.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"BEHAVIOR metadata not found: {metadata}")
    if not info.is_file():
        raise FileNotFoundError(f"BEHAVIOR metadata not found: {info}")
    jobs: list[dict[str, Any]] = []
    inventory: list[Path] = [metadata, info]
    estimated_frames = 0
    with metadata.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            episode_index = int(row["episode_index"])
            source = (
                version_root
                / f"data/task-{episode_index // 10000:04d}"
                / f"episode_{episode_index:08d}.parquet"
            )
            if source.is_file():
                inventory.append(source)
            estimated_frames += int(row.get("length", 0))
            jobs.append(
                {
                    "row": row,
                    "source": source,
                    "version_root": version_root,
                    "min_source_frames": min_source_frames,
                    "max_gap_factor": max_gap_factor,
                    "left_gripper_index": left_gripper_index,
                    "right_gripper_index": right_gripper_index,
                    "calibration_frames": calibration_frames,
                    "calibration_max_position_delta": calibration_max_position_delta,
                    "calibration_max_angle_degrees": calibration_max_angle_degrees,
                    "canonical_head_z": canonical_head_z,
                    "max_wrist_head_reach": max_wrist_head_reach,
                }
            )
    return version_root, jobs, tuple(inventory), estimated_frames


def _preflight_behavior_job(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    source = Path(job["source"])
    version_root = Path(job["version_root"])
    relative_source = source.relative_to(version_root).as_posix()
    exclusions: list[dict[str, str]] = []
    if not source.is_file():
        return {
            "episode": None,
            "exclusions": [
                {"source": relative_source, "reason": "missing Parquet file"}
            ],
        }
    try:
        arrays = _read_behavior_arrays(source)
        poses = arrays["poses"]
        actions = arrays["actions"]
        timestamps = arrays["timestamps"]
        left_gripper_index = int(job["left_gripper_index"])
        right_gripper_index = int(job["right_gripper_index"])
        quaternion_norm = np.linalg.norm(poses[:, :, 3:7], axis=-1)
        valid = (
            np.isfinite(poses).all(axis=(1, 2))
            & np.isfinite(timestamps)
            & np.isfinite(
                actions[:, [left_gripper_index, right_gripper_index]]
            ).all(axis=1)
            & (quaternion_norm > 1e-8).all(axis=1)
        )
        segments = _segment_ranges(
            valid,
            timestamps,
            min_source_frames=int(job["min_source_frames"]),
            max_gap_factor=float(job["max_gap_factor"]),
        )
        calibrated_segments: list[Segment] = []
        retargeted_reach_maxima: list[float] = []
        for segment in segments:
            try:
                calibrated = _behavior_calibration_segment(
                    poses,
                    segment,
                    frames=int(job["calibration_frames"]),
                    max_position_delta=float(
                        job["calibration_max_position_delta"]
                    ),
                    max_angle_degrees=float(
                        job["calibration_max_angle_degrees"]
                    ),
                )
                calibrated_positions, _ = _calibrate_behavior_three_points(
                    poses,
                    calibrated,
                    canonical_head_z=float(job["canonical_head_z"]),
                    max_wrist_head_reach=float(job["max_wrist_head_reach"]),
                )
                calibrated_segments.append(calibrated)
                retargeted_reach_maxima.append(
                    float(
                        np.linalg.norm(
                            calibrated_positions[:, :2]
                            - calibrated_positions[:, 2:3],
                            axis=-1,
                        ).max()
                    )
                )
            except ValueError as exc:
                exclusions.append(
                    {"source": relative_source, "reason": str(exc)}
                )
        segments = tuple(calibrated_segments)
        if not segments:
            raise ValueError("no continuous segment has a stable calibration window")
        deltas = np.diff(timestamps)
        positive = deltas[deltas > 0]
        source_hz = (
            1.0 / float(np.median(positive)) if positive.size else SOURCE_HZ
        )
        hands = actions[:, [left_gripper_index, right_gripper_index]]
        pose_stats = _pose_stats(poses, hands, "xyzw")
        pose_stats["calibration"] = {
            "method": "head_relative_limb_ratio_retargeting",
            "window_frames": int(job["calibration_frames"]),
            "max_position_delta_m": float(
                job["calibration_max_position_delta"]
            ),
            "max_angle_delta_degrees": float(
                job["calibration_max_angle_degrees"]
            ),
            "max_wrist_head_reach_m": float(job["max_wrist_head_reach"]),
            "retargeted_wrist_head_reach_max_m": retargeted_reach_maxima,
            "windows": [
                [segment.calibration_start, segment.calibration_end]
                for segment in segments
            ],
        }
        episode_index = int(row["episode_index"])
        episode = Episode(
            dataset="behavior1k",
            source=source,
            relative_source=relative_source,
            stable_base=f"behavior1k__{episode_index:08d}",
            source_frames=len(poses),
            source_hz=source_hz,
            main_task=str((row.get("tasks") or [""])[0]),
            segments=segments,
            valid_frames=sum(segment.end - segment.start for segment in segments),
            stats=pose_stats,
        )
    except (OSError, ValueError, KeyError, IndexError) as exc:
        exclusions.append({"source": relative_source, "reason": str(exc)})
        episode = None
    return {"episode": episode, "exclusions": exclusions}


def _resolve_preflight_workers(value: str | int, job_count: int) -> int:
    requested = (
        min(DEFAULT_PREFLIGHT_WORKERS, _available_cpu_count())
        if value == "auto"
        else int(value)
    )
    return max(1, min(requested, _available_cpu_count(), job_count))


def _iter_behavior_preflight(
    jobs: list[dict[str, Any]],
    workers: int,
) -> Any:
    if workers == 1:
        for job in jobs:
            yield _preflight_behavior_job(job)
        return
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        yield from pool.imap(_preflight_behavior_job, jobs, chunksize=1)


def _preflight_behavior(
    root: Path,
    *,
    max_episodes: int | None,
    preflight_workers: str | int,
    min_source_frames: int,
    max_gap_factor: float,
    left_gripper_index: int,
    right_gripper_index: int,
    calibration_frames: int,
    calibration_max_position_delta: float,
    calibration_max_angle_degrees: float,
    canonical_head_z: float,
    max_wrist_head_reach: float,
) -> Preflight:
    version_root, jobs, inventory, _ = _behavior_preflight_jobs(
        root,
        min_source_frames=min_source_frames,
        max_gap_factor=max_gap_factor,
        left_gripper_index=left_gripper_index,
        right_gripper_index=right_gripper_index,
        calibration_frames=calibration_frames,
        calibration_max_position_delta=calibration_max_position_delta,
        calibration_max_angle_degrees=calibration_max_angle_degrees,
        canonical_head_z=canonical_head_z,
        max_wrist_head_reach=max_wrist_head_reach,
    )
    workers = _resolve_preflight_workers(preflight_workers, len(jobs))
    episodes: list[Episode] = []
    exclusions: list[dict[str, str]] = []
    for scanned, result in enumerate(
        _iter_behavior_preflight(jobs, workers),
        start=1,
    ):
        exclusions.extend(result["exclusions"])
        episode = result["episode"]
        if episode is not None:
            episodes.append(episode)
        if scanned % PREFLIGHT_PROGRESS_INTERVAL == 0:
            print(
                f"[behavior preflight] scanned={scanned}/{len(jobs)} "
                f"eligible={len(episodes)} excluded={len(exclusions)} "
                f"workers={workers}",
                flush=True,
            )
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError("preflight found no eligible BEHAVIOR episodes")
    return Preflight(
        episodes=tuple(episodes),
        exclusions=tuple(exclusions),
        scanned_sources=inventory,
        inventory_fingerprint=_inventory_fingerprint(version_root, inventory),
    )

def _import_zarr() -> Any:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError(
            "EgoVerse numeric preflight requires Zarr v3 support. "
            "No dependency was installed automatically; approve `uv add zarr` first."
        ) from exc
    return zarr


def _egoverse_root(root: Path) -> Path:
    candidate = root / "v3"
    return candidate if candidate.is_dir() else root


def _validate_egoverse_snapshot(
    attributes: dict[str, Any], *, allow_legacy: bool = False
) -> bool:
    embodiment = str(attributes.get("embodiment", ""))
    intrinsics = attributes.get("intrinsics")
    valid_intrinsics = isinstance(intrinsics, dict) and bool(intrinsics)
    if valid_intrinsics:
        valid_intrinsics = all(
            np.asarray(matrix).shape == (3, 4)
            and np.isfinite(np.asarray(matrix, dtype=np.float64)).all()
            for matrix in intrinsics.values()
        )
    if embodiment.startswith("human_") and valid_intrinsics:
        return False
    if allow_legacy and embodiment == "aria_bimanual":
        return True
    raise RuntimeError(
        "EgoVerse snapshot predates the official 2026-07-08 schema: "
        f"embodiment={embodiment!r}, valid_intrinsics={valid_intrinsics}. "
        "Re-download the reprocessed episodes with human_* embodiment, corrected "
        "Aria orientations, and per-episode 3x4 camera intrinsics; do not mix "
        "legacy and current episodes. Pass --allow-legacy-egoverse only when "
        "intentionally processing the old aria_bimanual snapshot."
    )


def _read_egoverse_arrays(
    path: Path, *, allow_padded: bool = False
) -> dict[str, Any]:
    zarr = _import_zarr()
    group = zarr.open_group(str(path), mode="r")
    fields = (
        "left.obs_wrist_pose",
        "right.obs_wrist_pose",
        "obs_head_pose",
        "left.obs_keypoints",
        "right.obs_keypoints",
    )
    arrays = {field: np.asarray(group[field][:], dtype=np.float32) for field in fields}
    timestamps = np.asarray(group["obs_rgb_timestamps_ns"][:], dtype=np.int64).reshape(-1)
    counts = {field: len(value) for field, value in arrays.items()}
    counts["obs_rgb_timestamps_ns"] = len(timestamps)
    counts["images.front_1"] = int(group["images.front_1"].shape[0])
    attributes = dict(group.attrs)
    expected_frames = int(attributes["total_frames"])
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise ValueError(f"EgoVerse fields have inconsistent frame counts: {counts}")
    frames = unique_counts.pop()
    if expected_frames > frames or (not allow_padded and expected_frames != frames):
        raise ValueError(
            f"EgoVerse frame counts do not match total_frames={expected_frames}: {counts}"
        )
    poses = np.stack(
        (
            arrays["left.obs_wrist_pose"],
            arrays["right.obs_wrist_pose"],
            arrays["obs_head_pose"],
        ),
        axis=1,
    )
    left_hand = arrays["left.obs_keypoints"].reshape(frames, 21, 3)
    right_hand = arrays["right.obs_keypoints"].reshape(frames, 21, 3)
    return {
        "poses": poses,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "timestamps_ns": timestamps,
        "total_frames": expected_frames,
        "padded_frames": frames - expected_frames,
        "fps": float(attributes.get("fps", SOURCE_HZ)),
        "main_task": str(
            attributes.get("task_description") or attributes.get("task_name") or ""
        ),
    }


def _hand_tip_distance(
    hand: np.ndarray, tip_indices: np.ndarray = MANO_TIPS
) -> np.ndarray:
    tips = np.asarray(hand, dtype=np.float32)[:, tip_indices]
    return np.min(np.linalg.norm(tips[:, :1] - tips[:, 1:], axis=-1), axis=1)


def _legacy_egoverse_orientations(source_xyzw: np.ndarray) -> np.ndarray:
    orientations = _to_wxyz(source_xyzw, "xyzw")
    orientations[:, 1] = _quat_multiply(
        orientations[:, 1], LEGACY_ARIA_RIGHT_WRIST_CORRECTION_WXYZ
    )
    return _normalize_quaternions(orientations)


def _preflight_egoverse(
    root: Path,
    *,
    max_episodes: int | None,
    min_source_frames: int,
    max_gap_factor: float,
    allow_legacy: bool,
) -> Preflight:
    version_root = _egoverse_root(root)
    if not version_root.is_dir():
        raise FileNotFoundError(f"EgoVerse v3 root is not a directory: {version_root}")
    episodes: list[Episode] = []
    exclusions: list[dict[str, str]] = []
    scanned: list[Path] = []
    legacy_mode: bool | None = None
    entries = sorted(
        (
            Path(entry.path)
            for entry in os.scandir(version_root)
            if entry.is_dir(follow_symlinks=False)
        ),
        key=lambda path: path.name,
    )
    for source in entries:
        metadata = source / "zarr.json"
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("zarr.json root must be an object")
            attributes = payload.get("attributes", {})
            if not isinstance(attributes, dict):
                raise ValueError("zarr.json attributes must be an object")
        except OSError as exc:
            scanned.append(source)
            exclusions.append(
                {
                    "source": source.relative_to(version_root).as_posix(),
                    "reason": f"unreadable zarr.json: {exc}",
                }
            )
            continue
        except ValueError as exc:
            scanned.append(metadata)
            exclusions.append(
                {
                    "source": source.relative_to(version_root).as_posix(),
                    "reason": f"invalid zarr.json: {exc}",
                }
            )
            continue
        scanned.append(metadata)
        embodiment = str(attributes.get("embodiment", ""))
        if allow_legacy and embodiment != "aria_bimanual":
            exclusions.append(
                {
                    "source": source.relative_to(version_root).as_posix(),
                    "reason": (
                        "legacy EgoVerse conversion only includes aria_bimanual; "
                        f"skipped embodiment={embodiment!r}"
                    ),
                }
            )
            continue
        legacy = _validate_egoverse_snapshot(
            attributes, allow_legacy=allow_legacy
        )
        if legacy_mode is None:
            legacy_mode = legacy
        elif legacy_mode != legacy:
            raise RuntimeError("EgoVerse root mixes legacy and current snapshots")
        try:
            if attributes.get("embodiment") not in {
                "human_bimanual",
                "aria_bimanual",
            }:
                raise ValueError(
                    "SONIC conversion requires a bimanual EgoVerse embodiment"
                )
            arrays = _read_egoverse_arrays(source, allow_padded=legacy)
            poses = arrays["poses"]
            left_hand = arrays["left_hand"]
            right_hand = arrays["right_hand"]
            timestamps_ns = arrays["timestamps_ns"]
            quaternion_norm = np.linalg.norm(poses[:, :, 3:7], axis=-1)
            hand_span = np.stack(
                (
                    np.max(np.linalg.norm(left_hand[:, 1:] - left_hand[:, :1], axis=-1), axis=1),
                    np.max(np.linalg.norm(right_hand[:, 1:] - right_hand[:, :1], axis=-1), axis=1),
                ),
                axis=1,
            )
            valid = (
                np.isfinite(poses).all(axis=(1, 2))
                & np.isfinite(left_hand).all(axis=(1, 2))
                & np.isfinite(right_hand).all(axis=(1, 2))
                & (timestamps_ns > 0)
                & (quaternion_norm > 1e-8).all(axis=1)
                & (hand_span > 1e-3).all(axis=1)
            )
            if legacy and int(valid.sum()) != int(arrays["total_frames"]):
                raise ValueError(
                    "legacy valid-frame count does not match total_frames: "
                    + str(int(valid.sum()))
                    + " != "
                    + str(arrays["total_frames"])
                )
            relative_timestamps = (
                timestamps_ns - timestamps_ns[0]
            ).astype(np.float64) * 1e-9
            segments = _segment_ranges(
                valid,
                relative_timestamps,
                min_source_frames=min_source_frames,
                max_gap_factor=max_gap_factor,
            )
            if not segments:
                raise ValueError("no valid continuous segment")
            timestamp_deltas = np.diff(timestamps_ns)
            adjacent_valid = valid[:-1] & valid[1:]
            positive_deltas = timestamp_deltas[
                adjacent_valid & (timestamp_deltas > 0)
            ]
            if not positive_deltas.size:
                raise ValueError("no positive adjacent RGB timestamp deltas")
            source_hz = float(1e9 / np.median(positive_deltas))
            tip_indices = LEGACY_ARIA_TIPS if legacy else MANO_TIPS
            hands = np.stack(
                (
                    _hand_tip_distance(left_hand, tip_indices),
                    _hand_tip_distance(right_hand, tip_indices),
                ),
                axis=1,
            )
            stats = _pose_stats(poses, hands, "xyzw")
            stats["declared_total_frames"] = int(arrays["total_frames"])
            stats["padded_frames"] = int(arrays["padded_frames"])
            stats["timestamp_source_hz"] = source_hz
            stats["keypoint_layout"] = "aria_legacy" if legacy else "mano"
            stats["legacy_right_wrist_orientation_corrected"] = legacy
            episodes.append(
                Episode(
                    dataset="egoverse",
                    source=source,
                    relative_source=source.relative_to(version_root).as_posix(),
                    stable_base=f"egoverse__{_safe_id(source.name)}",
                    source_frames=len(poses),
                    source_hz=source_hz,
                    main_task=str(arrays["main_task"]),
                    segments=segments,
                    valid_frames=sum(segment.end - segment.start for segment in segments),
                    stats=stats,
                )
            )
        except (OSError, ValueError, KeyError, IndexError) as exc:
            exclusions.append(
                {
                    "source": source.relative_to(version_root).as_posix(),
                    "reason": str(exc),
                }
            )
            continue
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError("preflight found no eligible EgoVerse episodes")
    return Preflight(
        episodes=tuple(episodes),
        exclusions=tuple(exclusions),
        scanned_sources=tuple(scanned),
        inventory_fingerprint=_inventory_fingerprint(version_root, scanned),
    )


def _import_tfds() -> tuple[Any, Any]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "AgiBotWorld numeric preflight requires tensorflow and "
            "tensorflow_datasets"
        ) from exc
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError("TensorFlow initialized a GPU before RLDS setup") from exc
    return tf, tfds


def _agibot_root(root: Path) -> Path:
    candidate = root / "gripper/1.0.0"
    return candidate if candidate.is_dir() else root


def _decode_agibot_record(builder: Any, record: Any, tfds: Any) -> dict[str, Any]:
    image_decoders = {
        name: tfds.decode.SkipDecoding()
        for name in ("main_image", "wrist_image_left", "wrist_image_right")
    }
    episode = builder.info.features.deserialize_example(
        record,
        decoders={"steps": {"observation": image_decoders}},
    )
    values: dict[str, list[Any]] = {
        name: []
        for name in (
            "end_position",
            "end_orientation",
            "gripper",
            "head",
            "waist",
            "robot_position",
            "robot_orientation",
            "is_first",
            "is_last",
            "is_dexhand",
        )
    }
    language = ""
    task = ""
    for step in episode["steps"]:
        state = step["observation"]["state"]
        for target, source_name in (
            ("end_position", "end/position"),
            ("end_orientation", "end/orientation"),
            ("gripper", "effector/position_gripper"),
            ("head", "head/position"),
            ("waist", "waist/position"),
            ("robot_position", "robot/position"),
            ("robot_orientation", "robot/orientation"),
        ):
            values[target].append(state[source_name].numpy())
        for name in ("is_first", "is_last", "is_dexhand"):
            values[name].append(bool(step[name].numpy()))
        if not language:
            language = step["language_instruction"].numpy().decode(
                "utf-8", errors="replace"
            )
        if not task:
            task = step["task_name"].numpy().decode("utf-8", errors="replace")
    if not values["end_position"]:
        raise ValueError("empty RLDS episode")
    result = {name: np.asarray(items) for name, items in values.items()}
    result["language"] = language
    result["task"] = task
    result["file_path"] = episode["episode_metadata"]["file_path"].numpy().decode(
        "utf-8", errors="replace"
    )
    return result


def _preflight_agibot(
    root: Path,
    *,
    max_episodes: int | None,
    min_source_frames: int,
) -> Preflight:
    tf, tfds = _import_tfds()
    version_root = _agibot_root(root)
    if not version_root.is_dir():
        raise FileNotFoundError(f"AgiBotWorld version root is missing: {version_root}")
    builder = tfds.builder_from_directory(str(version_root))
    shards = sorted(version_root.glob("*.tfrecord-*"))
    if not shards:
        raise FileNotFoundError(f"no AgiBotWorld TFRecord shards under {version_root}")
    episodes: list[Episode] = []
    exclusions: list[dict[str, str]] = []
    scanned: list[Path] = [
        version_root / "dataset_info.json",
        version_root / "features.json",
    ]
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    for shard in shards:
        scanned.append(shard)
        records = tf.data.TFRecordDataset(
            str(shard), num_parallel_reads=1
        ).with_options(options)
        for record_index, record in enumerate(records):
            relative = f"{shard.name}#record={record_index}"
            try:
                arrays = _decode_agibot_record(builder, record, tfds)
                frames = len(arrays["end_position"])
                if arrays["end_position"].shape != (frames, 2, 3):
                    raise ValueError(
                        f"unexpected end positions: {arrays['end_position'].shape}"
                    )
                if arrays["end_orientation"].shape != (frames, 2, 4):
                    raise ValueError(
                        f"unexpected end orientations: {arrays['end_orientation'].shape}"
                    )
                if arrays["gripper"].shape != (frames, 2):
                    raise ValueError(f"unexpected gripper shape: {arrays['gripper'].shape}")
                if np.any(arrays["is_dexhand"]):
                    raise ValueError("dexhand episode is outside the gripper-only contract")
                if not arrays["is_first"][0] or np.any(arrays["is_first"][1:]):
                    raise ValueError("invalid RLDS is_first markers")
                if not arrays["is_last"][-1] or np.any(arrays["is_last"][:-1]):
                    raise ValueError("invalid RLDS is_last markers")
                quaternion_norm = np.linalg.norm(
                    arrays["end_orientation"], axis=-1
                )
                valid = (
                    np.isfinite(arrays["end_position"]).all(axis=(1, 2))
                    & np.isfinite(arrays["end_orientation"]).all(axis=(1, 2))
                    & np.isfinite(arrays["gripper"]).all(axis=1)
                    & np.isfinite(arrays["head"]).all(axis=1)
                    & np.isfinite(arrays["waist"]).all(axis=1)
                    & (quaternion_norm > 1e-8).all(axis=1)
                )
                segments = _segment_ranges(
                    valid,
                    None,
                    min_source_frames=min_source_frames,
                    max_gap_factor=2.0,
                )
                if not segments:
                    raise ValueError("no valid continuous RLDS segment")
                stats = {
                    "record_index": record_index,
                    "episode_file_path": arrays["file_path"],
                    "source_quaternion_order": "xyzw",
                    "end_position_min": np.min(
                        arrays["end_position"], axis=(0, 1)
                    ).tolist(),
                    "end_position_max": np.max(
                        arrays["end_position"], axis=(0, 1)
                    ).tolist(),
                    "quaternion_norm_min": float(np.min(quaternion_norm)),
                    "quaternion_norm_max": float(np.max(quaternion_norm)),
                    "head_yaw_pitch_min": np.min(arrays["head"], axis=0).tolist(),
                    "head_yaw_pitch_max": np.max(arrays["head"], axis=0).tolist(),
                    "robot_position_min": np.min(
                        arrays["robot_position"], axis=0
                    ).tolist(),
                    "robot_position_max": np.max(
                        arrays["robot_position"], axis=0
                    ).tolist(),
                    "gripper_mm_min": np.min(arrays["gripper"], axis=0).tolist(),
                    "gripper_mm_max": np.max(arrays["gripper"], axis=0).tolist(),
                    "conversion_gate": (
                        "head mount, flange-to-tracking-point offset, and gripper "
                        "closed threshold require approved physical calibration"
                    ),
                }
                episodes.append(
                    Episode(
                        dataset="agibotworld",
                        source=shard,
                        relative_source=relative,
                        stable_base=(
                            f"agibotworld__{_safe_id(shard.name)}__"
                            f"record{record_index:03d}"
                        ),
                        source_frames=frames,
                        source_hz=SOURCE_HZ,
                        main_task=str(arrays["language"] or arrays["task"]),
                        segments=segments,
                        valid_frames=sum(
                            segment.end - segment.start for segment in segments
                        ),
                        stats=stats,
                    )
                )
            except (OSError, ValueError, KeyError, IndexError) as exc:
                exclusions.append({"source": relative, "reason": str(exc)})
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError("preflight found no eligible AgiBotWorld gripper episodes")
    return Preflight(
        episodes=tuple(episodes),
        exclusions=tuple(exclusions),
        scanned_sources=tuple(scanned),
        inventory_fingerprint=_inventory_fingerprint(version_root, scanned),
    )


def _metadata_gate(dataset: str, root: Path, max_episodes: int | None) -> dict[str, Any]:
    if dataset == "egoverse":
        version_root = _egoverse_root(root)
        samples = []
        for entry in sorted(os.scandir(version_root), key=lambda item: item.name):
            if not entry.is_dir(follow_symlinks=False):
                continue
            metadata_path = Path(entry.path) / "zarr.json"
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            samples.append(
                {
                    "source": entry.name,
                    "attributes": payload.get("attributes", {}),
                }
            )
            if max_episodes is not None and len(samples) >= max_episodes:
                break
        return {
            "dataset": dataset,
            "status": "dependency_blocked",
            "required_dependency": "zarr with Zarr v3 and zstd support",
            "installed": importlib.util.find_spec("zarr") is not None,
            "sample_sequences": samples,
        }

    version_root = root / "gripper/1.0.0"
    if not version_root.is_dir():
        version_root = root
    info = json.loads((version_root / "dataset_info.json").read_text(encoding="utf-8"))
    features = json.loads((version_root / "features.json").read_text(encoding="utf-8"))
    shards = sorted(version_root.glob("*.tfrecord-*"))
    return {
        "dataset": dataset,
        "status": "dependency_blocked",
        "required_dependency": "tensorflow and tensorflow_datasets",
        "installed": {
            name: importlib.util.find_spec(name) is not None
            for name in ("tensorflow", "tensorflow_datasets")
        },
        "shards": len(shards),
        "split_summary": [
            {
                "name": split.get("name"),
                "num_shards": len(split.get("shardLengths", [])),
                "num_examples": sum(int(value) for value in split.get("shardLengths", [])),
                "num_bytes": int(split.get("numBytes", 0)),
            }
            for split in info.get("splits", [])
        ],
        "features_sha256": hashlib.sha256(
            json.dumps(features, sort_keys=True).encode()
        ).hexdigest(),
        "conversion_status": (
            "numeric preflight requires dependencies; pilot/full additionally require "
            "approved physical calibration"
        ),
    }


def _episode_payload(
    episode: Episode,
    segment: Segment,
    config: dict[str, Any],
    *,
    frame_limit: int | None = None,
    arrays: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    end = min(segment.end, segment.start + frame_limit) if frame_limit else segment.end
    selection = slice(segment.start, end)
    if episode.dataset == "behavior1k":
        if arrays is None:
            arrays = _read_behavior_arrays(episode.source)
        positions, source_orientations = _calibrate_behavior_three_points(
            arrays["poses"],
            segment,
            canonical_head_z=float(config["canonical_head_z"]),
            max_wrist_head_reach=float(
                config["behavior_max_wrist_head_reach"]
            ),
        )
        frame_count = end - segment.start
        positions = positions[:frame_count]
        source_orientations = source_orientations[:frame_count]
        action = arrays["actions"][selection]
        hands = (
            action[
                :,
                [
                    int(config["behavior_left_gripper_index"]),
                    int(config["behavior_right_gripper_index"]),
                ],
            ]
            <= float(config["behavior_gripper_closed_max"])
        ).astype(np.float32)
        quaternion_order = "wxyz"
    elif episode.dataset == "egoverse":
        legacy = bool(config["allow_legacy_egoverse"])
        if arrays is None:
            arrays = _read_egoverse_arrays(episode.source, allow_padded=legacy)
        poses = arrays["poses"][selection]
        positions = poses[:, :, :3]
        if legacy:
            source_orientations = _legacy_egoverse_orientations(poses[:, :, 3:7])
            quaternion_order = "wxyz"
        else:
            source_orientations = poses[:, :, 3:7]
            quaternion_order = "xyzw"
        tip_indices = LEGACY_ARIA_TIPS if legacy else MANO_TIPS
        hands = np.stack(
            (
                _hand_tip_distance(arrays["left_hand"][selection], tip_indices),
                _hand_tip_distance(arrays["right_hand"][selection], tip_indices),
            ),
            axis=1,
        )
        hands = (hands <= float(config["egoverse_hand_distance_threshold"])).astype(
            np.float32
        )
    else:
        raise RuntimeError(
            "AgiBotWorld conversion requires approved head/flange calibration values"
        )

    positions, orientations = _canonicalize_three_points(
        positions,
        source_orientations,
        quaternion_order=quaternion_order,
        canonical_head_z=float(config["canonical_head_z"]),
    )
    if hands.shape != (len(positions), 2) or not np.isin(hands, (0.0, 1.0)).all():
        raise ValueError(f"invalid binary hand state shape/values: {hands.shape}")
    return positions, orientations, hands


def _provider_list(gpu_id: int | None) -> list[Any]:
    if gpu_id is None:
        return ["CPUExecutionProvider"]
    return [
        ("CUDAExecutionProvider", {"device_id": str(gpu_id)}),
        "CPUExecutionProvider",
    ]


def _create_onnx_session(model_path: Path, gpu_id: int | None) -> Any:
    if not model_path.is_file():
        raise FileNotFoundError(f"SONIC encoder model not found: {model_path}")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=_provider_list(gpu_id),
    )
    expected = "CUDAExecutionProvider" if gpu_id is not None else "CPUExecutionProvider"
    if expected not in session.get_providers():
        raise RuntimeError(
            f"requested {expected}, but ONNX session uses {session.get_providers()}"
        )
    return session


def _encode_inputs(
    inputs: np.ndarray,
    session: Any,
    *,
    batch_size: int,
) -> np.ndarray:
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if len(input_meta.shape) != 2:
        raise ValueError(f"SONIC input must be rank 2, got {input_meta.shape}")
    fixed_batch = input_meta.shape[0] if isinstance(input_meta.shape[0], int) else None
    effective_batch = fixed_batch or batch_size
    outputs: list[np.ndarray] = []
    for start in range(0, len(inputs), effective_batch):
        batch = inputs[start : start + effective_batch]
        if fixed_batch not in (None, 1) and len(batch) != fixed_batch:
            raise ValueError(
                f"fixed ONNX batch {fixed_batch} does not divide remaining {len(batch)}"
            )
        encoded = session.run(
            [output_meta.name],
            {input_meta.name: batch.astype(np.float32, copy=False)},
        )[0]
        outputs.append(np.asarray(encoded, dtype=np.float32).reshape(len(batch), -1))
    latent = np.concatenate(outputs, axis=0)
    if latent.shape != (len(inputs), LATENT_DIM) or not np.isfinite(latent).all():
        raise ValueError(f"invalid SONIC output: {latent.shape}/{latent.dtype}")
    return latent


def _encode_teleop(
    positions: np.ndarray,
    orientations: np.ndarray,
    session: Any,
    *,
    chunk_frames: int,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(positions), chunk_frames):
        end = min(start + chunk_frames, len(positions))
        inputs = build_teleop_encoder_inputs(positions[start:end], orientations[start:end])
        outputs.append(_encode_inputs(inputs, session, batch_size=batch_size))
    return np.concatenate(outputs, axis=0)


def _gpu_free_ratio(gpu_id: int) -> float:
    import torch

    free, total = torch.cuda.mem_get_info(gpu_id)
    return float(free / total)


def _detect_execution(provider: str) -> tuple[str, list[dict[str, Any]]]:
    import onnxruntime as ort
    import torch

    providers = ort.get_available_providers()
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if "CUDAExecutionProvider" not in providers or count == 0:
        if provider == "cuda":
            raise RuntimeError(
                "--provider=cuda requires a visible CUDA device and "
                f"CUDAExecutionProvider; devices={count}, providers={providers}"
            )
        return "cpu", []
    if provider == "cpu":
        return "cpu", []
    inventory = []
    for gpu_id in range(count):
        properties = torch.cuda.get_device_properties(gpu_id)
        free, total = torch.cuda.mem_get_info(gpu_id)
        inventory.append(
            {
                "logical_id": gpu_id,
                "name": properties.name,
                "total_bytes": int(total),
                "free_bytes": int(free),
                "free_ratio": float(free / total),
            }
        )
    return "cuda", inventory


def _available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _benchmark_job(job: dict[str, Any]) -> dict[str, Any]:
    episode: Episode = job["episode"]
    gpu_id = int(job["gpu_id"])
    config = job["config"]
    session = _create_onnx_session(Path(config["encoder_model"]), gpu_id)
    segment = episode.segments[0]
    positions, orientations, _ = _episode_payload(
        episode,
        segment,
        config,
        frame_limit=AUTOTUNE_FRAMES,
    )
    warmup_frames = min(32, len(positions))
    _encode_teleop(
        positions[:warmup_frames],
        orientations[:warmup_frames],
        session,
        chunk_frames=warmup_frames,
        batch_size=int(config["batch_size"]),
    )
    free_ratio = _gpu_free_ratio(gpu_id)
    started = time.monotonic()
    positions, orientations, _ = _episode_payload(
        episode,
        segment,
        config,
        frame_limit=AUTOTUNE_FRAMES,
    )
    _encode_teleop(
        positions,
        orientations,
        session,
        chunk_frames=min(int(config["chunk_frames"]), len(positions)),
        batch_size=int(config["batch_size"]),
    )
    elapsed = time.monotonic() - started
    return {
        "frames": len(positions),
        "elapsed_seconds": elapsed,
        "free_ratio": min(free_ratio, _gpu_free_ratio(gpu_id)),
    }


def _autotune_gpu(
    gpu_id: int,
    episodes: tuple[Episode, ...],
    *,
    upper_bound: int,
    config: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    if upper_bound <= 1 or len(episodes) < 2:
        return 1, [{"processes": 1, "status": "insufficient episodes for autotune"}]
    records: list[dict[str, Any]] = []
    selected = 1
    previous_fps: float | None = None
    context = mp.get_context("spawn")
    representative = episodes[gpu_id % len(episodes)]
    for candidate in range(1, upper_bound + 1):
        try:
            jobs = [
                {"episode": representative, "gpu_id": gpu_id, "config": config}
                for _ in range(candidate)
            ]
            with context.Pool(processes=candidate, maxtasksperchild=1) as pool:
                results = pool.map(_benchmark_job, jobs)
            elapsed = max(float(result["elapsed_seconds"]) for result in results)
            frames = sum(int(result["frames"]) for result in results)
            frames_per_second = frames / elapsed
            free_ratio = min(float(result["free_ratio"]) for result in results)
            record = {
                "processes": candidate,
                "status": "ok",
                "frames_per_second": frames_per_second,
                "free_ratio": free_ratio,
            }
            records.append(record)
            if free_ratio < AUTOTUNE_MIN_FREE_RATIO:
                record["stop_reason"] = "free GPU memory below 20%"
                break
            if (
                previous_fps is not None
                and frames_per_second < previous_fps * (1.0 + AUTOTUNE_MIN_GAIN)
            ):
                record["stop_reason"] = "throughput gain below 10%"
                break
            selected = candidate
            previous_fps = frames_per_second
        except Exception as exc:  # subprocess preserves the useful exception text
            records.append(
                {
                    "processes": candidate,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if candidate == 1:
                raise
            break
    return selected, records


def _worker_plan(
    preflight: Preflight,
    *,
    provider: str,
    workers_per_gpu: str | int,
    max_workers_per_gpu: int,
    config: dict[str, Any],
) -> tuple[list[int | None], dict[str, Any]]:
    effective_provider, inventory = _detect_execution(provider)
    if effective_provider == "cpu":
        return [None], {
            "effective_provider": "cpu",
            "gpu_inventory": [],
            "selected_workers_per_gpu": {},
            "autotune": {},
        }

    gpu_count = len(inventory)
    episode_count = len(preflight.episodes)
    per_gpu_episode_cap = max(1, math.ceil(episode_count / gpu_count))
    cpu_cap = max(1, _available_cpu_count() // gpu_count)
    upper_bound = min(max_workers_per_gpu, per_gpu_episode_cap, cpu_cap)
    selected: dict[int, int] = {}
    autotune: dict[str, Any] = {}
    for gpu in inventory:
        gpu_id = int(gpu["logical_id"])
        if workers_per_gpu == "auto":
            count, records = _autotune_gpu(
                gpu_id,
                preflight.episodes,
                upper_bound=upper_bound,
                config=config,
            )
            selected[gpu_id] = count
            autotune[str(gpu_id)] = records
        else:
            selected[gpu_id] = min(int(workers_per_gpu), upper_bound)
            autotune[str(gpu_id)] = [
                {"status": "manual override", "processes": selected[gpu_id]}
            ]

    assignments: list[int | None] = []
    for worker_index in range(max(selected.values())):
        for gpu_id in sorted(selected):
            if worker_index < selected[gpu_id] and len(assignments) < episode_count:
                assignments.append(gpu_id)
    actual_counts = {
        str(gpu_id): assignments.count(gpu_id) for gpu_id in sorted(selected)
    }
    return assignments, {
        "effective_provider": "cuda",
        "gpu_inventory": inventory,
        "selected_workers_per_gpu": actual_counts,
        "autotune": autotune,
    }


def _process_episode(
    episode: Episode,
    *,
    session: Any,
    staging: Path,
    config: dict[str, Any],
    verify_chunking: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    chunk_error: float | None = None
    arrays = (
        _read_behavior_arrays(episode.source)
        if episode.dataset == "behavior1k"
        else _read_egoverse_arrays(
            episode.source,
            allow_padded=bool(config["allow_legacy_egoverse"]),
        )
    )
    for segment_index, segment in enumerate(episode.segments):
        positions, orientations, hands = _episode_payload(
            episode, segment, config, arrays=arrays
        )
        latent = _encode_teleop(
            positions,
            orientations,
            session,
            chunk_frames=int(config["chunk_frames"]),
            batch_size=int(config["batch_size"]),
        )
        token = np.concatenate((latent, hands), axis=1).astype(np.float32, copy=False)
        stable_id = f"{episode.stable_base}__seg{segment_index:03d}"
        if token.shape != (segment.end - segment.start, ACTION_DIM):
            raise ValueError(f"unexpected token shape for {stable_id}: {token.shape}")
        atomic_save_npy(staging / "sonic_tokens" / f"{stable_id}.npy", token)
        rows.append(
            {
                "stable_id": stable_id,
                "dataset": episode.dataset,
                "source": episode.relative_source,
                "source_episode": episode.stable_base,
                "source_start_frame": segment.start,
                "source_end_frame_exclusive": segment.end,
                "source_frame_count": segment.end - segment.start,
                "token_frame_count": len(token),
                "source_hz": episode.source_hz,
                "interpreted_hz": INTERPRETED_HZ,
                "resampling": "none",
                "frame_mapping": "one_source_frame_to_one_token",
                "duration_scale": episode.source_hz / INTERPRETED_HZ,
                "speed_multiplier": INTERPRETED_HZ / episode.source_hz,
                "main_task": episode.main_task,
                "encoder_mode": "teleop",
                "lower_body_policy": "fixed_g1_standing",
                "calibration_method": (
                    "sonic_live_rigid_offset"
                    if episode.dataset == "behavior1k"
                    else "none"
                ),
                "calibration_start_frame": segment.calibration_start,
                "calibration_end_frame_exclusive": segment.calibration_end,
                "hand_state_method": (
                    "r1pro_action_threshold"
                    if episode.dataset == "behavior1k"
                    else (
                        "aria21_thumb_to_fingertips_absolute"
                        if config["allow_legacy_egoverse"]
                        else "mano21_thumb_to_fingertips_absolute"
                    )
                ),
                "left_closed_fraction": float(hands[:, 0].mean()),
                "right_closed_fraction": float(hands[:, 1].mean()),
            }
        )
        if verify_chunking and chunk_error is None:
            check_frames = min(200, len(positions))
            whole = _encode_teleop(
                positions[:check_frames],
                orientations[:check_frames],
                session,
                chunk_frames=check_frames,
                batch_size=int(config["batch_size"]),
            )
            chunked = _encode_teleop(
                positions[:check_frames],
                orientations[:check_frames],
                session,
                chunk_frames=min(37, check_frames),
                batch_size=int(config["batch_size"]),
            )
            chunk_error = float(np.max(np.abs(whole - chunked)))
            atol = 1.0 / 16.0 if "CUDAExecutionProvider" in session.get_providers() else 1e-6
            if not np.allclose(whole, chunked, rtol=1e-5, atol=atol):
                raise ValueError(
                    f"chunked encoder output mismatch: max_abs={chunk_error}, atol={atol}"
                )
    return {
        "rows": rows,
        "chunking_max_abs_error": chunk_error,
        "providers": session.get_providers(),
    }


def _worker_main(
    worker_id: int,
    gpu_id: int | None,
    tasks: Any,
    results: Any,
    staging: Path,
    config: dict[str, Any],
) -> None:
    try:
        session = _create_onnx_session(Path(config["encoder_model"]), gpu_id)
        chunking_verified = False
        while True:
            episode = tasks.get()
            if episode is None:
                return
            print(
                f"[teleop worker={worker_id} device="
                f"{'cpu' if gpu_id is None else f'cuda:{gpu_id}'}] "
                f"{episode.relative_source}",
                flush=True,
            )
            result = _process_episode(
                episode,
                session=session,
                staging=staging,
                config=config,
                verify_chunking=(
                    bool(config["verify_chunking"])
                    and worker_id == 0
                    and not chunking_verified
                ),
            )
            chunking_verified = (
                chunking_verified or result["chunking_max_abs_error"] is not None
            )
            results.put(("ok", worker_id, gpu_id, episode.stable_base, result))
    except BaseException as exc:
        results.put(
            (
                "error",
                worker_id,
                gpu_id,
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )


def _run_worker_stream(
    episodes: Any,
    assignments: list[int | None],
    *,
    staging: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    context = mp.get_context("spawn")
    tasks = context.Queue(maxsize=WORK_QUEUE_SIZE)
    results = context.Queue()
    processes = [
        context.Process(
            target=_worker_main,
            args=(worker_id, gpu_id, tasks, results, staging, config),
        )
        for worker_id, gpu_id in enumerate(assignments)
    ]
    for process in processes:
        process.start()

    episode_results: list[dict[str, Any]] = []
    worker_summaries = {
        worker_id: {
            "worker_id": worker_id,
            "gpu_logical_id": gpu_id,
            "episodes": 0,
            "segments": 0,
            "token_frames": 0,
            "onnx_providers": [],
        }
        for worker_id, gpu_id in enumerate(assignments)
    }

    def handle(message: tuple[Any, ...]) -> None:
        if message[0] == "error":
            _, worker_id, gpu_id, error_type, error, stack = message
            raise RuntimeError(
                f"worker={worker_id} gpu={gpu_id} failed: "
                f"{error_type}: {error}\n{stack}"
            )
        _, worker_id, _, _, result = message
        episode_results.append(result)
        summary = worker_summaries[int(worker_id)]
        summary["episodes"] += 1
        summary["segments"] += len(result["rows"])
        summary["token_frames"] += sum(
            int(row["token_frame_count"]) for row in result["rows"]
        )
        summary["onnx_providers"] = result["providers"]
        if len(episode_results) % PREFLIGHT_PROGRESS_INTERVAL == 0:
            print(
                f"[behavior encode] completed={len(episode_results)}",
                flush=True,
            )

    def check_processes() -> None:
        failed = [
            (index, process.exitcode)
            for index, process in enumerate(processes)
            if process.exitcode not in (None, 0)
        ]
        if failed:
            raise RuntimeError(f"worker processes exited unexpectedly: {failed}")

    def drain() -> None:
        while True:
            try:
                handle(results.get_nowait())
            except queue.Empty:
                return

    def put(item: Episode | None) -> None:
        while True:
            drain()
            check_processes()
            try:
                tasks.put(item, timeout=1.0)
                return
            except queue.Full:
                continue

    submitted = 0
    try:
        for episode in episodes:
            put(episode)
            submitted += 1
        for _ in assignments:
            put(None)
        while len(episode_results) < submitted:
            try:
                handle(results.get(timeout=2.0))
            except queue.Empty:
                check_processes()
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        tasks.close()
        results.close()
    return episode_results, list(worker_summaries.values())


def _run_workers(
    preflight: Preflight,
    assignments: list[int | None],
    *,
    staging: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weighted = sorted(
        preflight.episodes,
        key=lambda episode: -sum(
            segment.end - segment.start for segment in episode.segments
        ),
    )
    return _run_worker_stream(
        weighted,
        assignments,
        staging=staging,
        config=config,
    )


def _run_behavior_pipeline(
    *,
    args: argparse.Namespace,
    jobs: list[dict[str, Any]],
    inventory: tuple[Path, ...],
    inventory_fingerprint: str,
    staging: Path,
    config: dict[str, Any],
) -> tuple[
    Preflight,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[int | None],
]:
    limit = None if args.publish_full or args.max_episodes == 0 else args.max_episodes
    preflight_workers = _resolve_preflight_workers(
        args.preflight_workers,
        len(jobs),
    )
    effective_provider, gpu_inventory = _detect_execution(args.provider)
    requested_per_gpu = (
        args.max_workers_per_gpu
        if args.workers_per_gpu == "auto"
        else min(int(args.workers_per_gpu), args.max_workers_per_gpu)
    )
    seed_target = max(1, len(gpu_inventory) * requested_per_gpu)
    if limit is not None:
        seed_target = min(seed_target, limit)
    print(
        f"[behavior pipeline] candidates={len(jobs)} "
        f"preflight_workers={preflight_workers} "
        f"provider={effective_provider} seed_episodes={seed_target}",
        flush=True,
    )

    episodes: list[Episode] = []
    exclusions: list[dict[str, str]] = []
    scanned = 0
    iterator = _iter_behavior_preflight(jobs, preflight_workers)

    def accept(result: dict[str, Any]) -> Episode | None:
        nonlocal scanned
        scanned += 1
        exclusions.extend(result["exclusions"])
        episode = result["episode"]
        if episode is not None:
            episodes.append(episode)
        if scanned % PREFLIGHT_PROGRESS_INTERVAL == 0:
            print(
                f"[behavior preflight] scanned={scanned}/{len(jobs)} "
                f"eligible={len(episodes)} excluded={len(exclusions)} "
                f"workers={preflight_workers}",
                flush=True,
            )
        return episode

    try:
        while len(episodes) < seed_target:
            try:
                accept(next(iterator))
            except StopIteration:
                break
        if not episodes:
            raise ValueError("preflight found no eligible BEHAVIOR episodes")
        seed = Preflight(
            episodes=tuple(episodes),
            exclusions=tuple(exclusions),
            scanned_sources=inventory,
            inventory_fingerprint=inventory_fingerprint,
        )
        assignments, execution = _worker_plan(
            seed,
            provider=args.provider,
            workers_per_gpu=args.workers_per_gpu,
            max_workers_per_gpu=args.max_workers_per_gpu,
            config=config,
        )
        execution.update(
            {
                "pipeline": "parallel_preflight_streaming_encode",
                "preflight_workers": preflight_workers,
                "work_queue_size": WORK_QUEUE_SIZE,
            }
        )
        print(
            f"[behavior pipeline] gpu_workers={len(assignments)} "
            f"assignments={execution['selected_workers_per_gpu']}",
            flush=True,
        )

        def episode_stream() -> Any:
            for episode in tuple(episodes):
                yield episode
            if limit is not None and len(episodes) >= limit:
                return
            for result in iterator:
                episode = accept(result)
                if episode is not None:
                    yield episode
                if limit is not None and len(episodes) >= limit:
                    return

        episode_results, worker_summaries = _run_worker_stream(
            episode_stream(),
            assignments,
            staging=staging,
            config=config,
        )
    finally:
        iterator.close()

    preflight = Preflight(
        episodes=tuple(episodes),
        exclusions=tuple(exclusions),
        scanned_sources=inventory,
        inventory_fingerprint=inventory_fingerprint,
    )
    execution["preflight_scanned_episodes"] = scanned
    print(
        f"[behavior pipeline] preflight_complete scanned={scanned} "
        f"eligible={len(episodes)} excluded={len(exclusions)} "
        f"encoded={len(episode_results)}",
        flush=True,
    )
    return preflight, episode_results, worker_summaries, execution, assignments

def _ordered_rows(
    preflight: Preflight,
    episode_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for result in episode_results:
        for row in result["rows"]:
            stable_id = str(row["stable_id"])
            if stable_id in rows_by_id:
                raise ValueError(f"duplicate stable id: {stable_id}")
            rows_by_id[stable_id] = row
    expected = [
        f"{episode.stable_base}__seg{index:03d}"
        for episode in preflight.episodes
        for index in range(len(episode.segments))
    ]
    if set(expected) != set(rows_by_id):
        raise ValueError(
            f"worker output mismatch: expected {len(expected)}, got {len(rows_by_id)}"
        )
    return [rows_by_id[stable_id] for stable_id in expected]


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_cache(root: Path, rows: list[dict[str, Any]]) -> None:
    parsed = [
        json.loads(line)
        for line in (root / "meta/episode_index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    if parsed != rows:
        raise ValueError("episode_index.jsonl failed round-trip validation")
    for row in rows:
        path = root / "sonic_tokens" / f"{row['stable_id']}.npy"
        token = np.load(path, mmap_mode="r")
        expected_frames = int(row["source_frame_count"])
        if token.shape != (expected_frames, ACTION_DIM) or token.dtype != np.float32:
            raise ValueError(f"invalid token file {path}: {token.shape}/{token.dtype}")
        if not np.isfinite(token).all():
            raise ValueError(f"non-finite token values in {path}")
        if not np.isin(token[:, 64:66], (0.0, 1.0)).all():
            raise ValueError(f"non-binary hand state in {path}")


def _validate_roots(data_root: Path, cache_root: Path, *, pilot: bool) -> None:
    raw = data_root.resolve(strict=True)
    output = cache_root.resolve(strict=False)
    if output == raw or output.is_relative_to(raw):
        raise ValueError(f"cache root must not be the source or inside it: {cache_root}")
    if not pilot and cache_root.parent.resolve() != data_root.parent.resolve():
        raise ValueError("full cache must be a sibling of the source dataset")


def convert(
    preflight: Preflight | None,
    *,
    args: argparse.Namespace,
    cache_root: Path,
    config: dict[str, Any],
    execution: dict[str, Any] | None,
    assignments: list[int | None] | None,
) -> dict[str, Any]:
    _validate_roots(args.data_root, cache_root, pilot=args.pilot)
    if cache_root.exists():
        raise FileExistsError(f"cache already exists; refusing to overwrite: {cache_root}")
    if not args.encoder_model.is_file():
        raise FileNotFoundError(f"SONIC encoder model not found: {args.encoder_model}")
    behavior_plan: tuple[Path, list[dict[str, Any]], tuple[Path, ...], int] | None = None
    if preflight is None:
        if args.dataset != "behavior1k":
            raise ValueError("streaming conversion is only supported for BEHAVIOR")
        behavior_plan = _behavior_preflight_jobs(
            args.data_root,
            min_source_frames=args.min_source_frames,
            max_gap_factor=args.max_timestamp_gap_factor,
            left_gripper_index=args.behavior_left_gripper_index,
            right_gripper_index=args.behavior_right_gripper_index,
            calibration_frames=args.behavior_calibration_frames,
            calibration_max_position_delta=(
                args.behavior_calibration_max_position_delta
            ),
            calibration_max_angle_degrees=(
                args.behavior_calibration_max_angle_degrees
            ),
            canonical_head_z=args.canonical_head_z,
            max_wrist_head_reach=args.behavior_max_wrist_head_reach,
        )
        _, jobs, inventory, estimated_frames = behavior_plan
        if not jobs or estimated_frames <= 0:
            raise ValueError("BEHAVIOR metadata contains no episodes or frames")
        estimated_token_bytes = (
            estimated_frames * ACTION_DIM * np.dtype(np.float32).itemsize
        )
    else:
        estimated_token_bytes = (
            sum(
                segment.end - segment.start
                for episode in preflight.episodes
                for segment in episode.segments
            )
            * ACTION_DIM
            * np.dtype(np.float32).itemsize
        )
    required_free_bytes = math.ceil(estimated_token_bytes * 1.3)
    available_free_bytes = shutil.disk_usage(cache_root.parent).free
    if not args.pilot and available_free_bytes < required_free_bytes:
        raise OSError(
            "insufficient cache space: "
            f"need at least {required_free_bytes} bytes, have {available_free_bytes}"
        )
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    lock = cache_root.with_name(f".{cache_root.name}.lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"another writer or stale lock exists: {lock}") from exc
    staging = Path(
        tempfile.mkdtemp(prefix=f".{cache_root.name}.staging-", dir=cache_root.parent)
    )
    started = time.monotonic()
    try:
        (staging / "sonic_tokens").mkdir()
        if preflight is None:
            assert behavior_plan is not None
            version_root, jobs, inventory, _ = behavior_plan
            initial_fingerprint = _inventory_fingerprint(version_root, inventory)
            (
                preflight,
                episode_results,
                worker_summaries,
                execution,
                assignments,
            ) = _run_behavior_pipeline(
                args=args,
                jobs=jobs,
                inventory=inventory,
                inventory_fingerprint=initial_fingerprint,
                staging=staging,
                config=config,
            )
        else:
            assert execution is not None and assignments is not None
            episode_results, worker_summaries = _run_workers(
                preflight,
                assignments,
                staging=staging,
                config=config,
            )
        estimated_token_bytes = (
            sum(episode.valid_frames for episode in preflight.episodes)
            * ACTION_DIM
            * np.dtype(np.float32).itemsize
        )
        required_free_bytes = math.ceil(estimated_token_bytes * 1.3)
        if not args.pilot and available_free_bytes < required_free_bytes:
            raise OSError(
                "insufficient cache space after preflight: "
                f"need at least {required_free_bytes} bytes, "
                f"have {available_free_bytes}"
            )
        rows = _ordered_rows(preflight, episode_results)
        current_fingerprint = _inventory_fingerprint(
            _behavior_root(args.data_root)
            if args.dataset == "behavior1k"
            else _egoverse_root(args.data_root),
            preflight.scanned_sources,
        )
        if current_fingerprint != preflight.inventory_fingerprint:
            raise RuntimeError("source inventory changed after preflight")

        schema = {
            "format_version": 1,
            "action": {"shape": ["T", ACTION_DIM], "dtype": "float32"},
            "sonic": {
                "encoder_mode_id": 1,
                "encoder_mode": "teleop",
                "encoder_input_dimension": ENCODER_INPUT_DIM,
                "latent_dimensions": [0, 64],
                "left_hand_dimension": 64,
                "right_hand_dimension": 65,
                "closed_hand_value": 1,
                "open_hand_value": 0,
            },
            "three_point_order": ["left_wrist", "right_wrist", "head"],
            "source_quaternion_order": "xyzw",
            "canonical_root_method": "head_position_and_head_yaw",
            "canonical_head_position": [0.0, 0.0, args.canonical_head_z],
            "source_calibration": {
                "behavior1k": {
                    "method": "head_relative_limb_ratio_retargeting",
                    "window_frames": args.behavior_calibration_frames,
                    "max_position_delta_m": (
                        args.behavior_calibration_max_position_delta
                    ),
                    "max_angle_delta_degrees": (
                        args.behavior_calibration_max_angle_degrees
                    ),
                    "max_wrist_head_reach_m": (
                        args.behavior_max_wrist_head_reach
                    ),
                    "position_reference": "wrist_camera_relative_to_head_camera",
                    "position_scale": (
                        "per_segment_g1_to_source_calibration_reach_ratio"
                    ),
                    "g1_wrist_reference_positions": (
                        G1_WRIST_REFERENCE_POSITIONS.tolist()
                    ),
                    "g1_wrist_reference_orientations_wxyz": (
                        G1_WRIST_REFERENCE_ORIENTATIONS.tolist()
                    ),
                    "torso_link_offset_z": TORSO_LINK_OFFSET_Z,
                }
            },
            "fixed_lower_body_joint_order": [
                "left_hip_pitch",
                "left_hip_roll",
                "left_hip_yaw",
                "left_knee",
                "left_ankle_pitch",
                "left_ankle_roll",
                "right_hip_pitch",
                "right_hip_roll",
                "right_hip_yaw",
                "right_knee",
                "right_ankle_pitch",
                "right_ankle_roll",
            ],
            "fixed_lower_body_values": LOWER_BODY.tolist(),
            "frequency": {
                "source_hz": SOURCE_HZ,
                "interpreted_hz": INTERPRETED_HZ,
                "resampling": "none",
                "frame_mapping": "one_source_frame_to_one_token",
                "duration_scale": SOURCE_HZ / INTERPRETED_HZ,
                "speed_multiplier": INTERPRETED_HZ / SOURCE_HZ,
            },
            "hand_state": {
                "behavior1k": {
                    "indices": [
                        args.behavior_left_gripper_index,
                        args.behavior_right_gripper_index,
                    ],
                    "closed_when_value_lte": args.behavior_gripper_closed_max,
                },
                "egoverse": {
                    "keypoint_layout": (
                        "aria_legacy" if args.allow_legacy_egoverse else "mano"
                    ),
                    "tip_indices": (
                        LEGACY_ARIA_TIPS if args.allow_legacy_egoverse else MANO_TIPS
                    ).tolist(),
                    "closed_distance_threshold_m": args.egoverse_hand_distance_threshold,
                    "right_wrist_orientation_correction_wxyz": (
                        LEGACY_ARIA_RIGHT_WRIST_CORRECTION_WXYZ.tolist()
                        if args.allow_legacy_egoverse
                        else None
                    ),
                },
            },
        }
        language_map = {
            str(row["stable_id"]): str(row["main_task"]) for row in rows
        }
        _atomic_write_json(staging / "meta/schema.json", schema)
        _atomic_write_json(staging / "meta/language_map.json", language_map)
        _atomic_write_jsonl(staging / "meta/episode_index.jsonl", rows)
        chunk_errors = [
            float(result["chunking_max_abs_error"])
            for result in episode_results
            if result["chunking_max_abs_error"] is not None
        ]
        manifest = {
            "format_version": 1,
            "dataset": args.dataset,
            "pilot": bool(args.pilot),
            "data_root": str(args.data_root),
            "cache_root": str(cache_root),
            "raw_inventory_fingerprint": preflight.inventory_fingerprint,
            "encoder_model": str(args.encoder_model),
            "encoder_model_sha256": _file_sha256(args.encoder_model),
            "cli_resolved_config": _jsonable(vars(args)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "execution": {
                **execution,
                "worker_assignments": [
                    {"worker_id": index, "gpu_logical_id": gpu_id}
                    for index, gpu_id in enumerate(assignments)
                ],
                "workers": worker_summaries,
            },
            "chunking_max_abs_error": max(chunk_errors) if chunk_errors else None,
            "elapsed_seconds": time.monotonic() - started,
            "output_space": {
                "estimated_token_bytes": estimated_token_bytes,
                "required_free_bytes": required_free_bytes,
                "available_free_bytes_at_start": available_free_bytes,
            },
            "counts": {
                "eligible_episodes": len(preflight.episodes),
                "excluded_scanned_episodes": len(preflight.exclusions),
                "segments": len(rows),
                "source_frames": sum(int(row["source_frame_count"]) for row in rows),
                "token_frames": sum(int(row["token_frame_count"]) for row in rows),
            },
            "exclusions": list(preflight.exclusions),
        }
        _atomic_write_json(staging / "meta/conversion_manifest.json", manifest)
        _validate_cache(staging, rows)
        marker = "_PILOT_SUCCESS" if args.pilot else "_SUCCESS"
        (staging / marker).write_text("ok\n", encoding="utf-8")
        os.replace(staging, cache_root)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        try:
            lock.rmdir()
        except FileNotFoundError:
            pass


def _ensure_cuda_library_path() -> None:
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    nvidia_root = Path(sys.prefix) / "lib" / python_version / "site-packages/nvidia"
    preferred = [
        str(path.resolve())
        for path in (nvidia_root / "cudnn/lib", nvidia_root / "cublas/lib")
        if path.is_dir()
    ]
    current = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    if not preferred or current[: len(preferred)] == preferred:
        return
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        [*preferred, *(part for part in current if part not in preferred)]
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def _self_check(model_path: Path) -> None:
    assert not _validate_egoverse_snapshot(
        {
            "embodiment": "human_bimanual",
            "intrinsics": {"front_1": np.eye(3, 4).tolist()},
        }
    )
    assert _validate_egoverse_snapshot(
        {"embodiment": "aria_bimanual", "intrinsics": {}},
        allow_legacy=True,
    )
    try:
        _validate_egoverse_snapshot(
            {"embodiment": "aria_bimanual", "intrinsics": {}}
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("legacy EgoVerse metadata passed the snapshot gate")
    legacy_xyzw = np.zeros((2, 3, 4), dtype=np.float32)
    legacy_xyzw[..., 3] = 1.0
    legacy_wxyz = _legacy_egoverse_orientations(legacy_xyzw)
    assert np.allclose(legacy_wxyz[:, (0, 2)], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(
        legacy_wxyz[:, 1], LEGACY_ARIA_RIGHT_WRIST_CORRECTION_WXYZ
    )
    timestamp_segments = _segment_ranges(
        np.ones(5, dtype=bool),
        np.asarray([0.0, 1 / 30, 2 / 30, 1.0, 1 + 1 / 30]),
        min_source_frames=2,
        max_gap_factor=2.0,
    )
    assert timestamp_segments == (Segment(0, 3), Segment(3, 5))

    source_pose = np.zeros((4, 3, 7), dtype=np.float32)
    source_pose[:, :, :3] = np.asarray(
        [[0.7, 0.4, 0.8], [0.7, -0.4, 0.8], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    source_orientation = Rotation.from_euler("z", 30, degrees=True).as_quat(
        scalar_first=True
    )
    source_pose[:, :, 3:7] = source_orientation[[1, 2, 3, 0]]
    source_pose[-1, :, :3] += [0.2, -0.1, 0.3]
    bad_prefixed_pose = np.concatenate((np.zeros_like(source_pose[:1]), source_pose))
    calibration_segment = _behavior_calibration_segment(
        bad_prefixed_pose,
        Segment(1, 5),
        frames=3,
        max_position_delta=0.02,
        max_angle_degrees=5.0,
    )
    assert calibration_segment == Segment(1, 5, 1, 4)
    calibrated_positions, calibrated_orientations = (
        _calibrate_behavior_three_points(
            bad_prefixed_pose,
            calibration_segment,
            canonical_head_z=DEFAULT_CANONICAL_HEAD_Z,
            max_wrist_head_reach=DEFAULT_BEHAVIOR_MAX_WRIST_HEAD_REACH,
        )
    )
    assert np.allclose(
        np.median(calibrated_positions[:, :2], axis=0),
        G1_WRIST_REFERENCE_POSITIONS,
        atol=1e-6,
    )
    assert np.allclose(
        _mean_quaternions_wxyz(calibrated_orientations[:, :2]),
        G1_WRIST_REFERENCE_ORIENTATIONS,
        atol=1e-6,
    )
    assert np.allclose(
        calibrated_positions[:, 2], [0.0, 0.0, DEFAULT_CANONICAL_HEAD_Z]
    )
    assert np.allclose(
        calibrated_positions[:, :2],
        G1_WRIST_REFERENCE_POSITIONS,
        atol=1e-6,
    )

    positions = np.asarray(
        [
            [[0.3, 0.2, 0.5], [0.3, -0.2, 0.5], [0.0, 0.0, 0.72]],
            [[0.4, 0.2, 0.6], [0.4, -0.2, 0.6], [0.0, 0.0, 0.72]],
        ],
        dtype=np.float32,
    )
    orientations = np.zeros((2, 3, 4), dtype=np.float32)
    orientations[..., 0] = 1.0
    inputs = build_teleop_encoder_inputs(positions, orientations)
    assert inputs.shape == (2, ENCODER_INPUT_DIM)
    assert inputs.dtype == np.float32
    assert np.all(inputs[:, 0] == 1)
    assert np.all(inputs[:, 1:4] == 0)
    assert np.all(inputs[:, 4:595] == 0)
    assert np.all(inputs[:, 595:601] == IDENTITY_6D)
    assert np.all(inputs[:, 601:661] == 0)
    assert np.all(inputs[:, 661:781] == np.tile(LOWER_BODY, 10))
    assert np.all(inputs[:, 781:901] == 0)
    assert np.all(inputs[:, 901:910] == positions.reshape(2, 9))
    assert np.all(inputs[:, 910:922] == orientations.reshape(2, 12))
    assert np.all(inputs[:, 922:] == 0)
    session = _create_onnx_session(model_path, None)
    whole = _encode_teleop(
        positions,
        orientations,
        session,
        chunk_frames=2,
        batch_size=2,
    )
    chunked = _encode_teleop(
        positions,
        orientations,
        session,
        chunk_frames=1,
        batch_size=2,
    )
    assert whole.shape == (2, LATENT_DIM)
    assert np.allclose(whole, chunked, rtol=1e-5, atol=1e-6)
    print(
        "self-check passed: EgoVerse snapshot/timestamp gates, head-relative retargeting, "
        "teleop layout [2,1762], encoder output [2,64], chunk equality"
    )


def _workers_argument(value: str) -> str | int:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return parsed


def _add_dataset_parser(subparsers: Any, dataset: str) -> None:
    data_root, cache_root = DATASET_DEFAULTS[dataset]
    parser = subparsers.add_parser(dataset)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--publish-full", action="store_true")
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--cache-root", type=Path, default=cache_root)
    parser.add_argument("--encoder-model", type=Path, default=DEFAULT_ENCODER_MODEL)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--min-source-frames", type=int, default=20)
    parser.add_argument("--max-timestamp-gap-factor", type=float, default=2.0)
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--preflight-workers",
        type=_workers_argument,
        default="auto",
        metavar="{auto,N}",
        help="BEHAVIOR preflight processes; auto caps at 16",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=_workers_argument,
        default="auto",
        metavar="{auto,N}",
    )
    parser.add_argument("--max-workers-per-gpu", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-frames", type=int, default=4096)
    parser.add_argument("--canonical-head-z", type=float, default=DEFAULT_CANONICAL_HEAD_Z)
    parser.add_argument(
        "--verify-chunking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--confirm-source-schema",
        action="store_true",
        help="required for pilot/full after pose, axes, quaternion, and hand mapping review",
    )
    parser.add_argument("--behavior-left-gripper-index", type=int, default=14)
    parser.add_argument("--behavior-right-gripper-index", type=int, default=22)
    parser.add_argument("--behavior-gripper-closed-max", type=float, default=0.0)
    parser.add_argument("--behavior-calibration-frames", type=int, default=30)
    parser.add_argument(
        "--behavior-calibration-max-position-delta", type=float, default=0.02
    )
    parser.add_argument(
        "--behavior-calibration-max-angle-degrees", type=float, default=5.0
    )
    parser.add_argument(
        "--behavior-max-wrist-head-reach",
        type=float,
        default=DEFAULT_BEHAVIOR_MAX_WRIST_HEAD_REACH,
    )
    parser.add_argument("--egoverse-hand-distance-threshold", type=float, default=0.045)
    parser.add_argument("--allow-legacy-egoverse", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument(
        "--encoder-model",
        type=Path,
        default=DEFAULT_ENCODER_MODEL,
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="dataset")
    for dataset in DATASET_DEFAULTS:
        _add_dataset_parser(subparsers, dataset)
    args = parser.parse_args()
    if not args.self_check and args.dataset is None:
        parser.error("choose a dataset subcommand or --self-check")
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be >= 0")
    if args.min_source_frames < 2:
        raise ValueError("--min-source-frames must be >= 2")
    if args.max_timestamp_gap_factor <= 1:
        raise ValueError("--max-timestamp-gap-factor must be > 1")
    if args.max_workers_per_gpu <= 0:
        raise ValueError("--max-workers-per-gpu must be positive")
    if args.batch_size <= 0 or args.chunk_frames <= 0:
        raise ValueError("--batch-size and --chunk-frames must be positive")
    if args.canonical_head_z <= TORSO_LINK_OFFSET_Z:
        raise ValueError(
            f"--canonical-head-z must exceed torso offset {TORSO_LINK_OFFSET_Z}"
        )
    if not (0 <= args.behavior_left_gripper_index < 23):
        raise ValueError("--behavior-left-gripper-index must be in [0,23)")
    if not (0 <= args.behavior_right_gripper_index < 23):
        raise ValueError("--behavior-right-gripper-index must be in [0,23)")
    if args.behavior_left_gripper_index == args.behavior_right_gripper_index:
        raise ValueError("left/right gripper indices must differ")
    if args.behavior_calibration_frames < 2:
        raise ValueError("--behavior-calibration-frames must be >= 2")
    if args.behavior_calibration_max_position_delta <= 0:
        raise ValueError("--behavior-calibration-max-position-delta must be positive")
    if args.behavior_calibration_max_angle_degrees <= 0:
        raise ValueError("--behavior-calibration-max-angle-degrees must be positive")
    if args.behavior_max_wrist_head_reach <= 0:
        raise ValueError("--behavior-max-wrist-head-reach must be positive")
    if args.egoverse_hand_distance_threshold <= 0:
        raise ValueError("--egoverse-hand-distance-threshold must be positive")
    if (args.pilot or args.publish_full) and not args.confirm_source_schema:
        raise ValueError(
            "pilot/full conversion requires --confirm-source-schema after visual schema review"
        )


def _conversion_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "encoder_model": str(args.encoder_model),
        "batch_size": args.batch_size,
        "chunk_frames": args.chunk_frames,
        "canonical_head_z": args.canonical_head_z,
        "verify_chunking": args.verify_chunking,
        "behavior_left_gripper_index": args.behavior_left_gripper_index,
        "behavior_right_gripper_index": args.behavior_right_gripper_index,
        "behavior_gripper_closed_max": args.behavior_gripper_closed_max,
        "behavior_calibration_frames": args.behavior_calibration_frames,
        "behavior_calibration_max_position_delta": (
            args.behavior_calibration_max_position_delta
        ),
        "behavior_calibration_max_angle_degrees": (
            args.behavior_calibration_max_angle_degrees
        ),
        "behavior_max_wrist_head_reach": args.behavior_max_wrist_head_reach,
        "egoverse_hand_distance_threshold": args.egoverse_hand_distance_threshold,
        "allow_legacy_egoverse": args.allow_legacy_egoverse,
    }


def _preflight_summary(
    result: Preflight,
    *,
    execution: dict[str, Any],
) -> dict[str, Any]:
    return {
        "eligible_episodes": len(result.episodes),
        "excluded_scanned_episodes": len(result.exclusions),
        "segments": sum(len(episode.segments) for episode in result.episodes),
        "valid_source_frames": sum(episode.valid_frames for episode in result.episodes),
        "estimated_token_bytes": sum(episode.valid_frames for episode in result.episodes)
        * ACTION_DIM
        * np.dtype(np.float32).itemsize,
        "inventory_fingerprint": result.inventory_fingerprint,
        "frequency_contract": {
            "source_hz": SOURCE_HZ,
            "interpreted_hz": INTERPRETED_HZ,
            "resampling": "none",
            "frame_mapping": "one_source_frame_to_one_token",
            "duration_scale": SOURCE_HZ / INTERPRETED_HZ,
            "speed_multiplier": INTERPRETED_HZ / SOURCE_HZ,
        },
        "execution": execution,
        "episodes": [
            {
                "source": episode.relative_source,
                "source_frames": episode.source_frames,
                "source_hz": episode.source_hz,
                "valid_frames": episode.valid_frames,
                "segments": [[segment.start, segment.end] for segment in episode.segments],
                "main_task": episode.main_task,
                "stats": episode.stats,
            }
            for episode in result.episodes
        ],
        "exclusions": list(result.exclusions),
    }


def main() -> None:
    args = parse_args()
    if args.self_check:
        _self_check(args.encoder_model)
        return
    _validate_args(args)
    if args.provider != "cpu":
        _ensure_cuda_library_path()

    limit = None if args.publish_full or args.max_episodes == 0 else args.max_episodes
    if args.dataset == "agibotworld" and any(
        importlib.util.find_spec(name) is None
        for name in ("tensorflow", "tensorflow_datasets")
    ):
        report = _metadata_gate(args.dataset, args.data_root, limit)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.pilot or args.publish_full:
            raise RuntimeError(report["conversion_status"])
        return
    if args.dataset == "egoverse" and importlib.util.find_spec("zarr") is None:
        report = _metadata_gate(args.dataset, args.data_root, limit)
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.pilot or args.publish_full:
            raise RuntimeError(report["required_dependency"])
        return

    if args.dataset == "behavior1k" and (args.pilot or args.publish_full):
        cache_root = args.cache_root
        if args.pilot and cache_root == DATASET_DEFAULTS[args.dataset][1]:
            cache_root = Path(tempfile.gettempdir()) / (
                f"huge-{args.dataset}-sonic-pilot-{os.getpid()}"
            )
        manifest = convert(
            None,
            args=args,
            cache_root=cache_root,
            config=_conversion_config(args),
            execution=None,
            assignments=None,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    if args.dataset == "behavior1k":
        preflight = _preflight_behavior(
            args.data_root,
            max_episodes=limit,
            preflight_workers=args.preflight_workers,
            min_source_frames=args.min_source_frames,
            max_gap_factor=args.max_timestamp_gap_factor,
            left_gripper_index=args.behavior_left_gripper_index,
            right_gripper_index=args.behavior_right_gripper_index,
            calibration_frames=args.behavior_calibration_frames,
            calibration_max_position_delta=args.behavior_calibration_max_position_delta,
            calibration_max_angle_degrees=args.behavior_calibration_max_angle_degrees,
            canonical_head_z=args.canonical_head_z,
            max_wrist_head_reach=args.behavior_max_wrist_head_reach,
        )
    elif args.dataset == "egoverse":
        preflight = _preflight_egoverse(
            args.data_root,
            max_episodes=limit,
            min_source_frames=args.min_source_frames,
            max_gap_factor=args.max_timestamp_gap_factor,
            allow_legacy=args.allow_legacy_egoverse,
        )
    else:
        preflight = _preflight_agibot(
            args.data_root,
            max_episodes=limit,
            min_source_frames=args.min_source_frames,
        )
        execution = {
            "effective_provider": "not_started",
            "conversion_gate": (
                "head mount, flange-to-tracking-point offset, and gripper closed "
                "threshold require approved physical calibration"
            ),
        }
        print(
            json.dumps(
                _preflight_summary(preflight, execution=execution),
                indent=2,
                sort_keys=True,
            )
        )
        if args.pilot or args.publish_full:
            raise RuntimeError(execution["conversion_gate"])
        return

    config = _conversion_config(args)
    assignments, execution = _worker_plan(
        preflight,
        provider=args.provider,
        workers_per_gpu=args.workers_per_gpu,
        max_workers_per_gpu=args.max_workers_per_gpu,
        config=config,
    )
    print(
        json.dumps(
            _preflight_summary(preflight, execution=execution),
            indent=2,
            sort_keys=True,
        )
    )
    if not args.pilot and not args.publish_full:
        return

    cache_root = args.cache_root
    if args.pilot and cache_root == DATASET_DEFAULTS[args.dataset][1]:
        cache_root = (
            Path(tempfile.gettempdir())
            / f"huge-{args.dataset}-sonic-pilot-{os.getpid()}"
        )
    manifest = convert(
        preflight,
        args=args,
        cache_root=cache_root,
        config=config,
        execution=execution,
        assignments=assignments,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build AgiBotWorld gripper SONIC planner trajectories and paired tokens.

The default mode is read-only preflight. ``--pilot`` writes only dataset-neutral
PlannerTrajectory ``.npz`` files and validation reports. ``--publish-full`` runs
the live SONIC planner replay and writes Behavior-cache-style 66-D action tokens,
34-D pseudo proprio, and an optional LeRobot v2.1 symlink view.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import signal
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.spatial.transform import Rotation

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
CALC_PROPRIO_DIR = REPO_ROOT / "scripts/preprocess/calc_proprio"
for path in (str(SRC_DIR), str(CALC_PROPRIO_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import calc_sonic_tokens as teleop
import run as replay
from huge.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT, PROPRIO_LAYOUT_VERSION


DATA_ROOT = Path("resource/datasets/agibotworld-beta-rlds")
CACHE_ROOT = Path("resource/datasets/agibotworld-beta-rlds-planner-trajectory-cache")
TOKEN_CACHE_ROOT = Path("resource/datasets/agibotworld-beta-rlds-planner-cache")
WORK_ROOT = Path(".runs/calc_proprio/agibotworld-planner")
SOURCE_HZ = teleop.SOURCE_HZ
CONTROL_HZ = 50.0
DEFAULT_SOURCE_MINIMUM_WAIST_LIFT = 0.02
DEFAULT_SOURCE_STANDING_WAIST_LIFT = 0.37
DEFAULT_G1_STANDING_HEIGHT = 0.78
DEFAULT_MINIMUM_HEIGHT = 0.40
DEFAULT_MAXIMUM_HEIGHT = 0.80
DEFAULT_GRIPPER_CLOSED_MM_THRESHOLD = 55.0


@dataclass(frozen=True)
class AgibotJob:
    episode: teleop.Episode
    segment_index: int

    @property
    def segment(self) -> teleop.Segment:
        return self.episode.segments[self.segment_index]

    @property
    def stable_id(self) -> str:
        return f"{self.episode.stable_base}__seg{self.segment_index:03d}"


def _resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else REPO_ROOT / expanded).resolve()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        teleop._jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(path: Path, relative: str) -> str:
    stat = path.stat()
    return hashlib.sha256(
        f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    ).hexdigest()


def _lowpass(values: np.ndarray, alpha: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    result = np.empty_like(source)
    result[0] = source[0]
    for index in range(1, len(source)):
        result[index] = alpha * source[index] + (1.0 - alpha) * result[index - 1]
    return result


def _parse_relative_source(relative: str) -> tuple[str, int]:
    shard_name, marker, record = relative.partition("#record=")
    if not marker:
        raise ValueError(f"unexpected AgiBot source id: {relative}")
    return shard_name, int(record)


def _read_agibot_arrays(data_root: Path, episode: teleop.Episode) -> dict[str, Any]:
    tf, tfds = teleop._import_tfds()
    version_root = teleop._agibot_root(data_root)
    shard_name, record_index = _parse_relative_source(episode.relative_source)
    shard = version_root / shard_name
    if not shard.is_file():
        raise FileNotFoundError(f"AgiBot shard is missing: {shard}")
    builder = tfds.builder_from_directory(str(version_root))
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    records = tf.data.TFRecordDataset(
        str(shard), num_parallel_reads=1
    ).with_options(options)
    selected = records.skip(record_index).take(1)
    for record in selected:
        return teleop._decode_agibot_record(builder, record, tfds)
    raise IndexError(f"record {record_index} is absent from {shard}")




def _preflight_agibot_sharded(
    root: Path,
    *,
    max_episodes: int | None,
    min_source_frames: int,
    shard_count: int,
    shard_index: int,
) -> teleop.Preflight:
    if shard_count == 1 and shard_index == 0:
        return teleop._preflight_agibot(
            root,
            max_episodes=max_episodes,
            min_source_frames=min_source_frames,
        )
    tf, tfds = teleop._import_tfds()
    version_root = teleop._agibot_root(root)
    if not version_root.is_dir():
        raise FileNotFoundError(f"AgiBotWorld version root is missing: {version_root}")
    builder = tfds.builder_from_directory(str(version_root))
    all_shards = sorted(version_root.glob("*.tfrecord-*"))
    if not all_shards:
        raise FileNotFoundError(f"no AgiBotWorld TFRecord shards under {version_root}")
    shards = [
        shard
        for absolute_index, shard in enumerate(all_shards)
        if absolute_index % shard_count == shard_index
    ]
    if not shards:
        raise ValueError(
            f"AgiBot shard partition {shard_index}/{shard_count} selected no shards"
        )
    episodes: list[teleop.Episode] = []
    exclusions: list[dict[str, str]] = []
    scanned: list[Path] = [
        version_root / "dataset_info.json",
        version_root / "features.json",
    ]
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    for selected_shard_index, shard in enumerate(shards):
        absolute_shard_index = all_shards.index(shard)
        scanned.append(shard)
        records = tf.data.TFRecordDataset(
            str(shard), num_parallel_reads=1
        ).with_options(options)
        for record_index, record in enumerate(records):
            relative = f"{shard.name}#record={record_index}"
            try:
                arrays = teleop._decode_agibot_record(builder, record, tfds)
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
                quaternion_norm = np.linalg.norm(arrays["end_orientation"], axis=-1)
                valid = (
                    np.isfinite(arrays["end_position"]).all(axis=(1, 2))
                    & np.isfinite(arrays["end_orientation"]).all(axis=(1, 2))
                    & np.isfinite(arrays["gripper"]).all(axis=1)
                    & np.isfinite(arrays["head"]).all(axis=1)
                    & np.isfinite(arrays["waist"]).all(axis=1)
                    & (quaternion_norm > 1e-8).all(axis=1)
                )
                segments = teleop._segment_ranges(
                    valid,
                    None,
                    min_source_frames=min_source_frames,
                    max_gap_factor=2.0,
                )
                if not segments:
                    raise ValueError("no valid continuous RLDS segment")
                stats = {
                    "record_index": record_index,
                    "absolute_shard_index": absolute_shard_index,
                    "shard_count": shard_count,
                    "shard_index": shard_index,
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
                    teleop.Episode(
                        dataset="agibotworld",
                        source=shard,
                        relative_source=relative,
                        stable_base=(
                            f"agibotworld__{teleop._safe_id(shard.name)}__"
                            f"record{record_index:03d}"
                        ),
                        source_frames=frames,
                        source_hz=SOURCE_HZ,
                        main_task=str(arrays["language"] or arrays["task"]),
                        segments=segments,
                        valid_frames=sum(segment.end - segment.start for segment in segments),
                        stats=stats,
                    )
                )
            except (OSError, ValueError, KeyError, IndexError) as exc:
                exclusions.append({"source": relative, "reason": str(exc)})
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
        if (selected_shard_index + 1) % 50 == 0:
            print(
                f"[agibot shard preflight] part={shard_index}/{shard_count} "
                f"scanned_shards={selected_shard_index + 1}/{len(shards)} "
                f"eligible={len(episodes)} excluded={len(exclusions)}",
                flush=True,
            )
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError(
            f"preflight found no eligible AgiBotWorld gripper episodes in shard "
            f"partition {shard_index}/{shard_count}"
        )
    return teleop.Preflight(
        episodes=tuple(episodes),
        exclusions=tuple(exclusions),
        scanned_sources=tuple(scanned),
        inventory_fingerprint=teleop._inventory_fingerprint(version_root, scanned),
    )


PREFLIGHT_CACHE_FORMAT_VERSION = 1


def _preflight_cache_params(args: argparse.Namespace, max_episodes: int | None) -> dict[str, Any]:
    return {
        "data_root": str(args.data_root),
        "version_root": str(teleop._agibot_root(args.data_root)),
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
        "min_source_frames": int(args.min_source_frames),
        "max_episodes": max_episodes,
    }


def _segment_payload(segment: teleop.Segment) -> dict[str, int | None]:
    return {
        "start": int(segment.start),
        "end": int(segment.end),
        "calibration_start": segment.calibration_start,
        "calibration_end": segment.calibration_end,
    }


def _segment_from_payload(payload: dict[str, Any]) -> teleop.Segment:
    return teleop.Segment(
        start=int(payload["start"]),
        end=int(payload["end"]),
        calibration_start=payload.get("calibration_start"),
        calibration_end=payload.get("calibration_end"),
    )


def _episode_payload(episode: teleop.Episode) -> dict[str, Any]:
    return {
        "dataset": episode.dataset,
        "source": str(episode.source),
        "relative_source": episode.relative_source,
        "stable_base": episode.stable_base,
        "source_frames": int(episode.source_frames),
        "source_hz": float(episode.source_hz),
        "main_task": episode.main_task,
        "segments": [_segment_payload(segment) for segment in episode.segments],
        "valid_frames": int(episode.valid_frames),
        "stats": teleop._jsonable(episode.stats),
    }


def _episode_from_payload(payload: dict[str, Any]) -> teleop.Episode:
    return teleop.Episode(
        dataset=str(payload["dataset"]),
        source=Path(payload["source"]),
        relative_source=str(payload["relative_source"]),
        stable_base=str(payload["stable_base"]),
        source_frames=int(payload["source_frames"]),
        source_hz=float(payload["source_hz"]),
        main_task=str(payload["main_task"]),
        segments=tuple(_segment_from_payload(item) for item in payload["segments"]),
        valid_frames=int(payload["valid_frames"]),
        stats=dict(payload.get("stats", {})),
    )


def _load_preflight_cache(
    cache_path: Path,
    args: argparse.Namespace,
    max_episodes: int | None,
) -> teleop.Preflight | None:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        params = _preflight_cache_params(args, max_episodes)
        if payload.get("format_version") != PREFLIGHT_CACHE_FORMAT_VERSION:
            print(f"[agibot preflight cache] ignoring version mismatch: {cache_path}", flush=True)
            return None
        if payload.get("params") != params:
            print(f"[agibot preflight cache] ignoring parameter mismatch: {cache_path}", flush=True)
            return None
        scanned = tuple(Path(item) for item in payload["scanned_sources"])
        fingerprint = teleop._inventory_fingerprint(teleop._agibot_root(args.data_root), list(scanned))
        if payload.get("inventory_fingerprint") != fingerprint:
            print(f"[agibot preflight cache] ignoring stale source fingerprint: {cache_path}", flush=True)
            return None
        preflight = teleop.Preflight(
            episodes=tuple(_episode_from_payload(item) for item in payload["episodes"]),
            exclusions=tuple(dict(item) for item in payload.get("exclusions", ())),
            scanned_sources=scanned,
            inventory_fingerprint=fingerprint,
        )
        print(
            f"[agibot preflight cache] loaded {len(preflight.episodes)} episode(s) "
            f"from {cache_path}",
            flush=True,
        )
        return preflight
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[agibot preflight cache] ignoring unreadable cache {cache_path}: {error}", flush=True)
        return None


def _write_preflight_cache(
    cache_path: Path,
    preflight: teleop.Preflight,
    args: argparse.Namespace,
    max_episodes: int | None,
) -> None:
    payload = {
        "format_version": PREFLIGHT_CACHE_FORMAT_VERSION,
        "kind": "agibotworld_sharded_preflight",
        "created_at": replay.now_utc(),
        "params": _preflight_cache_params(args, max_episodes),
        "inventory_fingerprint": preflight.inventory_fingerprint,
        "scanned_sources": [str(path) for path in preflight.scanned_sources],
        "episodes": [_episode_payload(episode) for episode in preflight.episodes],
        "exclusions": list(preflight.exclusions),
    }
    teleop._atomic_write_json(cache_path, payload)
    print(
        f"[agibot preflight cache] wrote {len(preflight.episodes)} episode(s) to {cache_path}",
        flush=True,
    )

def _yaw_from_robot_orientation_xyzw(quaternions: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    valid = norms >= 1e-8
    yaw = np.zeros(len(values), dtype=np.float64)
    if valid.any():
        q = values[valid] / norms[valid, None]
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw[valid] = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    for index in range(1, len(yaw)):
        if not valid[index]:
            yaw[index] = yaw[index - 1]
    yaw = np.unwrap(yaw).astype(np.float32)
    return yaw, {
        "valid_rows": int(valid.sum()),
        "zero_quaternion_rows": int((~valid).sum()),
        "yaw_min": float(yaw.min()),
        "yaw_max": float(yaw.max()),
    }


def _yaw_pitch_quat_wxyz(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float32)
    pitch = np.asarray(pitch, dtype=np.float32)
    yaw_q = np.stack(
        (
            np.cos(yaw / 2.0),
            np.zeros_like(yaw),
            np.zeros_like(yaw),
            np.sin(yaw / 2.0),
        ),
        axis=1,
    ).astype(np.float32)
    pitch_q = np.stack(
        (
            np.cos(pitch / 2.0),
            np.zeros_like(pitch),
            np.sin(pitch / 2.0),
            np.zeros_like(pitch),
        ),
        axis=1,
    ).astype(np.float32)
    return teleop._normalize_quaternions(teleop._quat_multiply(yaw_q, pitch_q))


def _apply_flange_offsets(
    positions: np.ndarray,
    orientations_wxyz: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(
        [args.left_flange_position_offset, args.right_flange_position_offset],
        dtype=np.float32,
    )
    rotations = teleop._normalize_quaternions(
        np.asarray(
            [args.left_flange_rotation_wxyz, args.right_flange_rotation_wxyz],
            dtype=np.float32,
        )
    )
    out_pos = np.asarray(positions, dtype=np.float32).copy()
    out_ori = np.asarray(orientations_wxyz, dtype=np.float32).copy()
    for hand in range(2):
        rotation = Rotation.from_quat(out_ori[:, hand], scalar_first=True)
        out_pos[:, hand] += rotation.apply(offsets[hand]).astype(np.float32)
        out_ori[:, hand] = teleop._quat_multiply(out_ori[:, hand], rotations[hand])
    return out_pos, teleop._normalize_quaternions(out_ori)


def _three_point_targets(
    arrays: dict[str, Any],
    segment: teleop.Segment,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    selection = slice(segment.start, segment.end)
    source_pos = np.asarray(arrays["end_position"][selection], dtype=np.float32)
    source_ori = teleop._to_wxyz(arrays["end_orientation"][selection], "xyzw")
    tracking_pos, tracking_ori = _apply_flange_offsets(source_pos, source_ori, args)

    calibration_frames = min(int(args.calibration_frames), len(tracking_pos))
    if calibration_frames < 2:
        raise ValueError("AgiBot planner calibration requires at least two frames")
    calibration = slice(0, calibration_frames)
    calibration_pos = np.median(tracking_pos[calibration], axis=0).astype(np.float32)
    calibration_ori = teleop._mean_quaternions_wxyz(tracking_ori[calibration])
    axis_scale = np.asarray(args.arm_axis_scale, dtype=np.float32)

    positions = np.zeros((len(tracking_pos), 3, 3), dtype=np.float32)
    orientations = np.zeros((len(tracking_pos), 3, 4), dtype=np.float32)
    positions[:, :2] = teleop.G1_WRIST_REFERENCE_POSITIONS[None, :, :] + (
        (tracking_pos - calibration_pos[None, :, :]) * axis_scale[None, None, :]
    )
    wrist_rotation_offsets = teleop._quat_multiply(
        teleop.G1_WRIST_REFERENCE_ORIENTATIONS,
        teleop._quat_conjugate(calibration_ori),
    )
    orientations[:, :2] = teleop._quat_multiply(
        wrist_rotation_offsets[None, :, :], tracking_ori
    )

    base_yaw, base_yaw_report = _yaw_from_robot_orientation_xyzw(
        arrays["robot_orientation"][selection]
    )
    head_angles = np.asarray(arrays["head"][selection], dtype=np.float32)
    head_yaw = (
        float(args.head_yaw_sign) * head_angles[:, 0]
        + float(args.head_yaw_offset)
        + float(args.base_yaw_scale) * base_yaw
    )
    head_pitch = (
        float(args.head_pitch_sign) * head_angles[:, 1]
        + float(args.head_pitch_offset)
    )
    head_quat = _yaw_pitch_quat_wxyz(head_yaw, head_pitch)
    head_rotation = Rotation.from_quat(head_quat, scalar_first=True)
    neck_length = float(args.canonical_head_z) - teleop.TORSO_LINK_OFFSET_Z
    if neck_length <= 0.0:
        raise ValueError("--canonical-head-z must exceed the torso link offset")
    positions[:, 2] = np.asarray(
        [0.0, 0.0, teleop.TORSO_LINK_OFFSET_Z], dtype=np.float32
    ) + neck_length * head_rotation.apply([0.0, 0.0, 1.0]).astype(np.float32)
    orientations[:, 2] = head_quat

    wrist_head = np.linalg.norm(positions[:, :2] - positions[:, 2:3], axis=2)
    max_reach = float(wrist_head.max())
    if max_reach > float(args.max_wrist_head_reach):
        raise ValueError(
            "retargeted wrist-to-head reach "
            f"{max_reach:.3f} m exceeds {args.max_wrist_head_reach:.3f} m"
        )
    return positions, teleop._normalize_quaternions(orientations), {
        "calibration_frames": calibration_frames,
        "calibration_start_frame": segment.start,
        "calibration_end_frame_exclusive": segment.start + calibration_frames,
        "source_flange_position_min": source_pos.min(axis=(0, 1)).tolist(),
        "source_flange_position_max": source_pos.max(axis=(0, 1)).tolist(),
        "tracking_position_min": tracking_pos.min(axis=(0, 1)).tolist(),
        "tracking_position_max": tracking_pos.max(axis=(0, 1)).tolist(),
        "retargeted_wrist_to_head_min": np.min(wrist_head, axis=0).tolist(),
        "retargeted_wrist_to_head_max": np.max(wrist_head, axis=0).tolist(),
        "base_yaw": base_yaw_report,
        "head_yaw_min": float(head_yaw.min()),
        "head_yaw_max": float(head_yaw.max()),
        "head_pitch_min": float(head_pitch.min()),
        "head_pitch_max": float(head_pitch.max()),
    }


def _base_command(
    arrays: dict[str, Any],
    segment: teleop.Segment,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    selection = slice(segment.start, segment.end)
    robot_position = np.asarray(arrays["robot_position"][selection], dtype=np.float32)
    base_yaw, yaw_report = _yaw_from_robot_orientation_xyzw(
        arrays["robot_orientation"][selection]
    )
    filtered_xy = _lowpass(robot_position[:, :2], args.base_filter_alpha)
    filtered_yaw = _lowpass(base_yaw[:, None], args.base_filter_alpha)[:, 0]
    linear_velocity = np.zeros((len(filtered_xy), 2), dtype=np.float32)
    yaw_rate = np.zeros(len(filtered_xy), dtype=np.float32)
    if len(filtered_xy) > 1:
        linear_velocity[1:] = np.diff(filtered_xy, axis=0) * CONTROL_HZ
        yaw_rate[1:] = np.diff(filtered_yaw) * CONTROL_HZ
    command = np.concatenate(
        (linear_velocity, yaw_rate[:, None]), axis=1
    ).astype(np.float32)
    command *= np.asarray(args.base_axis_scale, dtype=np.float32)

    linear_speed = np.linalg.norm(command[:, :2], axis=1)
    linear_active = linear_speed >= float(args.linear_deadband)
    yaw_active = np.abs(command[:, 2]) >= float(args.yaw_deadband)
    command[~linear_active, :2] = 0.0
    command[~yaw_active, 2] = 0.0
    linear_speed = np.linalg.norm(command[:, :2], axis=1)

    movement = np.zeros((len(command), 3), dtype=np.float32)
    moving = linear_speed > 0.0
    movement[moving, :2] = command[moving, :2] / linear_speed[moving, None]
    facing = np.stack(
        (
            np.cos(filtered_yaw),
            np.sin(filtered_yaw),
            np.zeros_like(filtered_yaw),
        ),
        axis=1,
    ).astype(np.float32)
    speed = np.full(len(command), -1.0, dtype=np.float32)
    speed[moving] = linear_speed[moving]
    return movement, facing, speed, {
        "robot_position_min": robot_position.min(axis=0).tolist(),
        "robot_position_max": robot_position.max(axis=0).tolist(),
        "base_yaw": yaw_report,
        "base_command_abs_max": np.max(np.abs(command), axis=0).tolist(),
        "linear_active_rows": int(linear_active.sum()),
        "yaw_active_rows": int(yaw_active.sum()),
    }


def _height_from_waist(
    arrays: dict[str, Any],
    segment: teleop.Segment,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    selection = slice(segment.start, segment.end)
    waist = np.asarray(arrays["waist"][selection], dtype=np.float32)
    source_lift = waist[:, 1]
    denom = float(args.source_standing_waist_lift) - float(
        args.source_minimum_waist_lift
    )
    if denom <= 1e-6:
        raise ValueError("source standing waist lift must exceed source minimum waist lift")
    ratio = (source_lift - float(args.source_minimum_waist_lift)) / denom
    height = float(args.minimum_height) + ratio * (
        float(args.g1_standing_height) - float(args.minimum_height)
    )
    height = _lowpass(height[:, None], args.height_filter_alpha)[:, 0]
    height = np.clip(height, args.minimum_height, args.maximum_height).astype(np.float32)
    return height, {
        "source_waist_pitch_min": float(waist[:, 0].min()),
        "source_waist_pitch_max": float(waist[:, 0].max()),
        "source_waist_lift_min": float(source_lift.min()),
        "source_waist_lift_max": float(source_lift.max()),
        "height_min": float(height.min()),
        "height_max": float(height.max()),
    }


def _hand_state(
    arrays: dict[str, Any],
    segment: teleop.Segment,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    gripper = np.asarray(
        arrays["gripper"][segment.start : segment.end], dtype=np.float32
    )
    hands = (gripper <= float(args.gripper_closed_mm_threshold)).astype(np.float32)
    return hands, {
        "method": "observation_effector_position_gripper_mm_threshold",
        "threshold_mm": float(args.gripper_closed_mm_threshold),
        "source_open_mm_min": gripper.min(axis=0).tolist(),
        "source_open_mm_max": gripper.max(axis=0).tolist(),
        "left_closed_fraction": float(hands[:, 0].mean()),
        "right_closed_fraction": float(hands[:, 1].mean()),
    }


def _trajectory_for_job(
    job: AgibotJob,
    args: argparse.Namespace,
    *,
    arrays: dict[str, Any] | None = None,
) -> tuple[replay.PlannerTrajectory, dict[str, Any]]:
    if arrays is None:
        arrays = _read_agibot_arrays(args.data_root, job.episode)
    positions, orientations, pose_report = _three_point_targets(
        arrays, job.segment, args
    )
    movement, facing, speed, base_report = _base_command(arrays, job.segment, args)
    height, height_report = _height_from_waist(arrays, job.segment, args)
    hands, hand_report = _hand_state(arrays, job.segment, args)

    linear_active = speed >= 0.0
    facing_yaw = np.unwrap(np.arctan2(facing[:, 1], facing[:, 0]))
    yaw_active = np.zeros(len(facing), dtype=bool)
    if len(facing) > 1:
        yaw_active[1:] = np.abs(np.diff(facing_yaw)) * CONTROL_HZ >= args.yaw_deadband
    locomotion = linear_active | yaw_active
    squat = (~locomotion) & (height < float(args.squat_height_threshold))
    mode = np.zeros(len(height), dtype=np.float32)
    mode[locomotion] = 2.0
    mode[squat] = 4.0

    trajectory = replay.PlannerTrajectory(
        mode=mode,
        movement=movement,
        facing=facing,
        speed=speed,
        height=height,
        vr_position=positions.reshape(len(positions), 9).astype(np.float32),
        vr_orientation=orientations.reshape(len(orientations), 12).astype(np.float32),
        hand_state=hands.astype(np.float32),
    )
    check = replay.validate_planner_trajectory(trajectory)
    check.update(
        {
            "idle_rows": int((mode == 0).sum()),
            "walk_rows": int((mode == 2).sum()),
            "squat_rows": int((mode == 4).sum()),
            "turn_only_rows": int((yaw_active & ~linear_active).sum()),
            "locomotion_height_conflict_rows": int(
                (locomotion & (height < args.squat_height_threshold)).sum()
            ),
            "pose": pose_report,
            "base": base_report,
            "height": height_report,
            "hand": hand_report,
        }
    )
    return trajectory, check


def _trajectory_arrays(trajectory: replay.PlannerTrajectory) -> dict[str, np.ndarray]:
    return {
        "mode": trajectory.mode.astype(np.float32),
        "movement": trajectory.movement.astype(np.float32),
        "facing": trajectory.facing.astype(np.float32),
        "speed": trajectory.speed.astype(np.float32),
        "height": trajectory.height.astype(np.float32),
        "vr_position": trajectory.vr_position.astype(np.float32),
        "vr_orientation": trajectory.vr_orientation.astype(np.float32),
        "hand_state": trajectory.hand_state.astype(np.float32),
    }


def _mapping_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_hz": SOURCE_HZ,
        "control_hz": CONTROL_HZ,
        "resampling": "none",
        "frame_mapping": "one_source_frame_to_one_control_tick",
        "duration_scale": SOURCE_HZ / CONTROL_HZ,
        "base": {
            "method": "finite_difference_robot_position_and_yaw",
            "axis_scale": list(args.base_axis_scale),
            "filter_alpha": args.base_filter_alpha,
            "linear_deadband": args.linear_deadband,
            "yaw_deadband": args.yaw_deadband,
            "mode_policy": {
                "idle": 0,
                "walk_or_turn": 2,
                "stationary_below_squat_threshold": 4,
            },
        },
        "height": {
            "source": "observation/state/waist/position[:,1]",
            "source_minimum_waist_lift": args.source_minimum_waist_lift,
            "source_standing_waist_lift": args.source_standing_waist_lift,
            "g1_standing_height": args.g1_standing_height,
            "minimum": args.minimum_height,
            "maximum": args.maximum_height,
            "filter_alpha": args.height_filter_alpha,
            "squat_threshold": args.squat_height_threshold,
        },
        "three_point": {
            "method": "flange_relative_to_first_calibration_window_plus_synthesized_head",
            "source_flange_position": "observation/state/end/position",
            "source_flange_orientation": "observation/state/end/orientation",
            "source_quaternion_order": "xyzw",
            "target_quaternion_order": "wxyz",
            "canonical_head_z": args.canonical_head_z,
            "calibration_frames": args.calibration_frames,
            "arm_axis_scale": list(args.arm_axis_scale),
            "left_flange_position_offset": list(args.left_flange_position_offset),
            "right_flange_position_offset": list(args.right_flange_position_offset),
            "left_flange_rotation_wxyz": list(args.left_flange_rotation_wxyz),
            "right_flange_rotation_wxyz": list(args.right_flange_rotation_wxyz),
            "max_wrist_head_reach": args.max_wrist_head_reach,
            "head": {
                "method": "base_yaw_plus_head_yaw_pitch",
                "base_yaw_scale": args.base_yaw_scale,
                "head_yaw_sign": args.head_yaw_sign,
                "head_pitch_sign": args.head_pitch_sign,
                "head_yaw_offset": args.head_yaw_offset,
                "head_pitch_offset": args.head_pitch_offset,
            },
        },
        "hand": {
            "source": "observation/state/effector/position_gripper",
            "closed_mm_threshold": args.gripper_closed_mm_threshold,
            "closed": 1,
            "open": 0,
        },
        "calibration_status": (
            "approved" if args.confirm_calibration else "provisional"
        ),
        "calibration_label": args.calibration_label,
    }


def _schema(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "planner_trajectory": {
            "arrays": {
                "mode": ["T"],
                "movement": ["T", 3],
                "facing": ["T", 3],
                "speed": ["T"],
                "height": ["T"],
                "vr_position": ["T", 9],
                "vr_orientation": ["T", 12],
                "hand_state": ["T", 2],
            },
            "dtype": "float32",
            "three_point_order": ["left_wrist", "right_wrist", "head"],
            "quaternion_order": "wxyz",
            "movement_frame": "mujoco_world",
            "facing_frame": "mujoco_world",
            "vr_target_frame": "planner_local",
            "hand_convention": {"closed": 1, "open": 0},
        },
        "downstream_replay": {
            "script": "scripts/preprocess/calc_proprio/run.py",
            "token_output_shape": ["T", 66],
            "proprio_output_shape": ["T", 34],
            "requires_gpu": True,
            "status": "not_run_by_this_script",
        },
        "mapping": mapping,
    }


def _asset_hashes() -> dict[str, str | None]:
    assets = {
        "encoder": replay.ENCODER_MODEL,
        "decoder": replay.DECODER_MODEL,
        "planner": replay.PLANNER_MODEL,
        "observation_config": replay.OBS_CONFIG,
    }
    return {name: _file_hash(path) for name, path in assets.items()}


def _jobs(preflight: teleop.Preflight) -> list[AgibotJob]:
    return [
        AgibotJob(episode, segment_index)
        for episode in preflight.episodes
        for segment_index in range(len(episode.segments))
    ]


def _row_and_report(
    job: AgibotJob,
    trajectory: replay.PlannerTrajectory,
    trajectory_check: dict[str, Any],
    mapping_hash: str,
    output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode = job.episode
    segment = job.segment
    source_fingerprint = _source_fingerprint(episode.source, episode.relative_source)
    row = {
        "stable_id": job.stable_id,
        "dataset": "agibotworld",
        "source": episode.relative_source,
        "source_episode": episode.stable_base,
        "source_start_frame": segment.start,
        "source_end_frame_exclusive": segment.end,
        "source_frame_count": trajectory.rows,
        "trajectory_frame_count": trajectory.rows,
        "source_hz": episode.source_hz,
        "control_hz": CONTROL_HZ,
        "resampling": "none",
        "frame_mapping": "one_source_frame_to_one_control_tick",
        "duration_scale": SOURCE_HZ / CONTROL_HZ,
        "main_task": episode.main_task,
        "planner_trajectory": str(output_path),
        "encoder_mode": "teleop_planner",
        "lower_body_policy": "sonic_planner",
        "calibration_method": "provisional_agibot_flange_relative_synth_head",
        "calibration_start_frame": trajectory_check["pose"]["calibration_start_frame"],
        "calibration_end_frame_exclusive": trajectory_check["pose"][
            "calibration_end_frame_exclusive"
        ],
        "hand_state_method": "gripper_mm_threshold",
        "left_closed_fraction": float(trajectory.hand_state[:, 0].mean()),
        "right_closed_fraction": float(trajectory.hand_state[:, 1].mean()),
        "trajectory_check": trajectory_check,
        "source_path": episode.relative_source,
        "source_fingerprint": source_fingerprint,
        "mapping_config_sha256": mapping_hash,
    }
    report = {
        "status": "complete",
        "stable_id": job.stable_id,
        "source": episode.relative_source,
        "planner_trajectory": str(output_path),
        "rows": trajectory.rows,
        "mapping_config_sha256": mapping_hash,
        "source_fingerprint": source_fingerprint,
        "trajectory_check": trajectory_check,
        "downstream_replay_status": "not_run",
    }
    return row, report


def _live_schema(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "action": {
            "shape": ["T", 66],
            "dtype": "float32",
            "motion_token_dimensions": [0, 64],
            "left_hand_dimension": 64,
            "right_hand_dimension": 65,
            "closed_hand_value": 1,
            "open_hand_value": 0,
        },
        "proprio": {
            "shape": ["T", PROPRIO_DIM],
            "dtype": "float32",
            "layout": PROPRIO_LAYOUT_VERSION,
        },
        "planner_trajectory": {
            "three_point_order": ["left_wrist", "right_wrist", "head"],
            "quaternion_order": "wxyz",
            "movement_frame": "mujoco_world",
            "facing_frame": "mujoco_world",
            "vr_target_frame": "planner_local",
            "hand_convention": {"closed": 1, "open": 0},
        },
        "mapping": mapping,
    }


def _row_and_provenance(
    job: AgibotJob,
    trajectory: replay.PlannerTrajectory,
    trajectory_check: dict[str, Any],
    mapping_hash: str,
    asset_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode = job.episode
    segment = job.segment
    source_fingerprint = _source_fingerprint(episode.source, episode.relative_source)
    episode_file_path = str(episode.stats.get("episode_file_path") or "")
    shard_name, record_index = _parse_relative_source(episode.relative_source)
    provenance = {
        "source_path": episode.relative_source,
        "source_fingerprint": source_fingerprint,
        "source_episode_file_path": episode_file_path,
        "source_start_frame": segment.start,
        "source_end_frame_exclusive": segment.end,
        "mapping_config_sha256": mapping_hash,
        "runtime_asset_set_sha256": asset_hash,
    }
    row = {
        "stable_id": job.stable_id,
        "dataset": "agibotworld",
        "source": episode.relative_source,
        "source_episode": episode.stable_base,
        "source_start_frame": segment.start,
        "source_end_frame_exclusive": segment.end,
        "source_frame_count": trajectory.rows,
        "token_frame_count": trajectory.rows,
        "proprio_frame_count": trajectory.rows,
        "source_hz": episode.source_hz,
        "control_hz": CONTROL_HZ,
        "resampling": "none",
        "frame_mapping": "one_source_frame_to_one_control_tick",
        "duration_scale": SOURCE_HZ / CONTROL_HZ,
        "main_task": episode.main_task,
        "encoder_mode": "teleop_planner",
        "lower_body_policy": "sonic_planner",
        "calibration_method": "provisional_agibot_flange_relative_synth_head",
        "calibration_start_frame": trajectory_check["pose"]["calibration_start_frame"],
        "calibration_end_frame_exclusive": trajectory_check["pose"][
            "calibration_end_frame_exclusive"
        ],
        "hand_state_method": "gripper_mm_threshold",
        "left_closed_fraction": float(trajectory.hand_state[:, 0].mean()),
        "right_closed_fraction": float(trajectory.hand_state[:, 1].mean()),
        "trajectory_check": trajectory_check,
        "source_path": episode.relative_source,
        "source_tfrecord": shard_name,
        "source_record_index": record_index,
        "source_episode_file_path": episode_file_path,
        "mapping_config_sha256": mapping_hash,
        "runtime_asset_set_sha256": asset_hash,
        "source_fingerprint": source_fingerprint,
    }
    return row, provenance


def _write_stats(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = [root / "g1_proprio" / f"{row['stable_id']}.npy" for row in rows]
    total = sum(int(row["proprio_frame_count"]) for row in rows)
    sample_limit = 1_000_000
    stride = max(1, math.ceil(total / sample_limit))
    minimum = np.full(PROPRIO_DIM, np.inf, dtype=np.float64)
    maximum = np.full(PROPRIO_DIM, -np.inf, dtype=np.float64)
    sums = np.zeros(PROPRIO_DIM, dtype=np.float64)
    squares = np.zeros(PROPRIO_DIM, dtype=np.float64)
    samples: list[np.ndarray] = []
    seen = 0
    for path in paths:
        array = np.load(path, mmap_mode="r")
        values = np.asarray(array, dtype=np.float64)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
        sums += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
        first = (-seen) % stride
        samples.append(np.asarray(array[first::stride], dtype=np.float32))
        seen += len(array)
    sample = np.concatenate(samples, axis=0)
    mean = sums / total
    std = np.sqrt(np.maximum(0.0, squares / total - np.square(mean)))
    q01, q99 = np.quantile(sample, (0.01, 0.99), axis=0)
    grouped: dict[str, dict[str, list[float]]] = {}
    for name, start, end in PROPRIO_LAYOUT:
        grouped[name] = {
            "min": minimum[start:end].tolist(),
            "max": maximum[start:end].tolist(),
            "mean": mean[start:end].tolist(),
            "std": std[start:end].tolist(),
            "q01": q01[start:end].tolist(),
            "q99": q99[start:end].tolist(),
        }
    payload = {
        "unitree_g1_sonic": {"state": grouped},
        "_meta": {
            "sources": {
                "agibotworld_planner": {
                    "n_episodes": len(paths),
                    "n_frames": total,
                }
            },
            "n_frames": total,
            "layout": PROPRIO_LAYOUT_VERSION,
            "g1_proprio_dim": PROPRIO_DIM,
            "quantile_method": "deterministic_global_stride_sample",
            "quantile_sample_rows": int(len(sample)),
            "quantile_stride": stride,
        },
    }
    replay.atomic_json(root / "meta/proprio_stats_sonic.json", payload)
    return payload["_meta"]


def _validate_live_cache(root: Path, rows: list[dict[str, Any]]) -> None:
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
        count = int(row["source_frame_count"])
        token_path = root / "sonic_tokens" / f"{row['stable_id']}.npy"
        proprio_path = root / "g1_proprio" / f"{row['stable_id']}.npy"
        report_path = proprio_path.with_suffix(".report.json")
        token = np.load(token_path, mmap_mode="r")
        proprio = np.load(proprio_path, mmap_mode="r")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if token.shape != (count, 66) or token.dtype != np.float32:
            raise ValueError(f"invalid token output: {token_path}: {token.shape}/{token.dtype}")
        if proprio.shape != (count, PROPRIO_DIM) or proprio.dtype != np.float32:
            raise ValueError(
                f"invalid proprio output: {proprio_path}: {proprio.shape}/{proprio.dtype}"
            )
        if not np.isfinite(token).all() or not np.isfinite(proprio).all():
            raise ValueError(f"non-finite paired output for {row['stable_id']}")
        if not np.array_equal(token[:, 64:66], proprio[:, 29:31]):
            raise ValueError(f"token/proprio hands disagree for {row['stable_id']}")
        if report.get("status") != "complete" or report.get("token_rows") != count:
            raise ValueError(f"invalid planner report for {row['stable_id']}")


def _build_worker_specs(
    devices: list[str],
    args: argparse.Namespace,
    job_count: int,
) -> list[replay.WorkerSpec]:
    backend = replay.inference_backend(args)
    if not devices:
        if backend == "onnxruntime":
            devices = ["cpu"]
        else:
            raise ValueError("at least one visible GPU is required")
    requested = args.workers if args.workers is not None else len(devices)
    if backend == "tensorrt" and requested > len(devices) and not args.allow_gpu_oversubscription:
        raise ValueError(
            f"--workers={requested} exceeds {len(devices)} visible GPU(s); "
            "pass --allow-gpu-oversubscription to run multiple replay workers per GPU"
        )
    count = min(requested, job_count)
    if args.zmq_port_base + 2 * count - 1 > 65535:
        raise ValueError("worker ZMQ ports exceed 65535")
    if args.dds_domain_base + count - 1 > replay.MAX_DDS_DOMAIN:
        raise ValueError(f"worker DDS domains exceed {replay.MAX_DDS_DOMAIN}")
    specs = [
        replay.WorkerSpec(
            index=index,
            gpu_selector=devices[index % len(devices)],
            gpu_index=index % len(devices),
            gpu_replica=index // len(devices),
            dds_domain=args.dds_domain_base + index,
            input_port=args.zmq_port_base + 2 * index,
            output_port=args.zmq_port_base + 2 * index + 1,
        )
        for index in range(count)
    ]
    busy = sorted(
        port
        for spec in specs
        for port in (spec.input_port, spec.output_port)
        if not replay.port_free(port)
    )
    if busy:
        raise RuntimeError(f"replay port(s) already in use: {busy}")
    return specs


@contextmanager
def _cache_lock(cache_root: Path) -> Iterator[None]:
    lock_path = cache_root.with_name(f".{cache_root.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another AgiBot planner writer holds {cache_root}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _task_name(text: str, index: int) -> str:
    words = []
    for token in text.lower().split():
        token = "".join(ch for ch in token if ch.isalnum())
        if token:
            words.append(token)
        if len(words) >= 8:
            break
    return "_".join(words) if words else f"task_{index:06d}"


def _ensure_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    desired = os.path.relpath(target, link.parent)
    if link.is_symlink():
        existing = os.readlink(link)
        if existing == desired or (link.parent / existing).resolve() == target.resolve():
            return
        raise FileExistsError(f"stale symlink {link} -> {existing}, expected {desired}")
    if link.exists():
        raise FileExistsError(f"refusing to replace non-symlink path: {link}")
    os.symlink(desired, link)


def _write_lerobot_v2_1(root: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    view = root / "lerobot_v2_1"
    chunk_size = int(args.lerobot_chunk_size)
    video_key = str(args.lerobot_video_key)
    video_file = str(args.lerobot_video_file)
    task_to_index: dict[str, int] = {}
    task_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    skipped_video: list[dict[str, Any]] = []

    def task_index(task: str) -> int:
        if task not in task_to_index:
            index = len(task_to_index)
            task_to_index[task] = index
            task_rows.append(
                {"task": task, "task_index": index, "task_name": _task_name(task, index)}
            )
        return task_to_index[task]

    for row in rows:
        source_dir = Path(str(row.get("source_episode_file_path") or ""))
        original_video = source_dir / "videos" / video_file
        if not original_video.is_file():
            skipped_video.append(
                {
                    "stable_id": row["stable_id"],
                    "source_episode_file_path": str(source_dir),
                    "missing_video": str(original_video),
                }
            )
            continue
        episode_index = len(episode_rows)
        chunk = episode_index // chunk_size
        token_rel = Path("sonic_tokens") / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.npy"
        proprio_rel = Path("g1_proprio") / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.npy"
        video_rel = Path("videos") / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
        data_rel = Path("data") / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.json"
        _ensure_symlink(root / "sonic_tokens" / f"{row['stable_id']}.npy", view / token_rel)
        _ensure_symlink(root / "g1_proprio" / f"{row['stable_id']}.npy", view / proprio_rel)
        _ensure_symlink(original_video, view / video_rel)
        teleop._atomic_write_json(
            view / data_rel,
            {
                "stable_id": row["stable_id"],
                "source": row["source"],
                "source_tfrecord": row.get("source_tfrecord"),
                "source_record_index": row.get("source_record_index"),
                "source_episode_file_path": row.get("source_episode_file_path"),
                "source_start_frame": row["source_start_frame"],
                "source_end_frame_exclusive": row["source_end_frame_exclusive"],
            },
        )
        task = str(row.get("main_task") or "perform the task")
        t_index = task_index(task)
        item = {
            "data_path": data_rel.as_posix(),
            "episode_index": episode_index,
            "g1_proprio_path": proprio_rel.as_posix(),
            "length": int(row["token_frame_count"]),
            "original_video_path": str(original_video),
            "proprio_frame_count": int(row["proprio_frame_count"]),
            "sonic_token_path": token_rel.as_posix(),
            "source_end_frame_exclusive": int(row["source_end_frame_exclusive"]),
            "source_episode": row["source_episode"],
            "source_episode_file_path": row.get("source_episode_file_path"),
            "source_frame_count": int(row["source_frame_count"]),
            "source_path": row["source"],
            "source_record_index": row.get("source_record_index"),
            "source_start_frame": int(row["source_start_frame"]),
            "source_tfrecord": row.get("source_tfrecord"),
            "stable_id": row["stable_id"],
            "task": task,
            "task_index": t_index,
            "tasks": [task],
            "token_frame_count": int(row["token_frame_count"]),
            "video_path": video_rel.as_posix(),
        }
        episode_rows.append(item)
        source_rows.append({k: v for k, v in item.items() if k != "tasks"})
        stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "num_frames": int(row["token_frame_count"]),
                    "proprio_frame_count": int(row["proprio_frame_count"]),
                    "source_frame_count": int(row["source_frame_count"]),
                    "token_frame_count": int(row["token_frame_count"]),
                },
            }
        )

    meta = view / "meta"
    teleop._atomic_write_jsonl(meta / "tasks.jsonl", task_rows)
    teleop._atomic_write_jsonl(meta / "episodes.jsonl", episode_rows)
    teleop._atomic_write_jsonl(meta / "episodes_stats.jsonl", stats_rows)
    teleop._atomic_write_jsonl(meta / "source_id_map.jsonl", source_rows)
    teleop._atomic_write_jsonl(
        meta / "skipped_missing_head_rgb_video.jsonl", skipped_video
    )
    total_frames = sum(int(row["length"]) for row in episode_rows)
    info = {
        "codebase_version": "v2.1",
        "view_type": "symlinked_agibotworld_sonic_head_rgb",
        "robot_type": "G1",
        "source_robot_type": "AgiBotWorld gripper",
        "total_episodes": len(episode_rows),
        "source_total_episodes": len(rows),
        "skipped_missing_head_rgb_video": len(skipped_video),
        "total_frames": total_frames,
        "total_tasks": len(task_rows),
        "total_videos": len(episode_rows),
        "chunks_size": chunk_size,
        "fps": 30,
        "control_hz": CONTROL_HZ,
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.json",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "sonic_token_path": "sonic_tokens/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npy",
        "g1_proprio_path": "g1_proprio/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npy",
        "source_id_map_path": "meta/source_id_map.jsonl",
        "video_keys": [video_key],
        "frame_mapping": "one_source_frame_to_one_control_tick",
        "resampling": "none",
        "features": {
            video_key: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 640,
                    "video.channels": 3,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "action.sonic_token": {
                "dtype": "float32",
                "shape": ["T", 66],
                "names": None,
                "path": "sonic_tokens/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npy",
            },
            "observation.g1_proprio": {
                "dtype": "float32",
                "shape": ["T", PROPRIO_DIM],
                "names": None,
                "path": "g1_proprio/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npy",
            },
            "source.agibotworld_rlds": {
                "dtype": "json",
                "path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.json",
            },
        },
    }
    teleop._atomic_write_json(meta / "info.json", info)
    return {
        "path": str(view),
        "episodes": len(episode_rows),
        "skipped_missing_head_rgb_video": len(skipped_video),
        "total_frames": total_frames,
    }


def _run_live(
    preflight: teleop.Preflight,
    args: argparse.Namespace,
    cache_root: Path,
) -> dict[str, Any]:
    teleop._validate_roots(
        args.data_root,
        cache_root,
        pilot=bool(args.allow_cross_root_cache),
    )
    mapping = _mapping_config(args)
    mapping_hash = _json_hash(mapping)
    asset_hashes = _asset_hashes()
    asset_hash = _json_hash(asset_hashes)
    jobs = _jobs(preflight)
    jobs.sort(key=lambda item: (-(item.segment.end - item.segment.start), item.stable_id))
    if cache_root.exists():
        if not args.resume:
            raise FileExistsError(f"cache already exists; refusing to overwrite: {cache_root}")
        success_path = cache_root / "_SUCCESS"
        manifest_path = cache_root / "meta/conversion_manifest.json"
        index_path = cache_root / "meta/episode_index.jsonl"
        if not success_path.is_file() or not manifest_path.is_file() or not index_path.is_file():
            raise FileExistsError(
                f"cache already exists but is not a complete resumable part: {cache_root}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks = {
            "raw_inventory_fingerprint": preflight.inventory_fingerprint,
            "mapping_config_sha256": mapping_hash,
            "runtime_asset_set_sha256": asset_hash,
        }
        mismatches = [
            key for key, expected in checks.items() if manifest.get(key) != expected
        ]
        counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
        if counts.get("segments") != len(jobs):
            mismatches.append("counts.segments")
        if mismatches:
            raise ValueError(
                f"existing cache metadata does not match current inputs: {', '.join(mismatches)}"
            )
        rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        expected_ids = sorted(job.stable_id for job in jobs)
        row_ids = sorted(str(row.get("stable_id")) for row in rows)
        if row_ids != expected_ids:
            raise ValueError("existing cache episode_index does not match current preflight jobs")
        _validate_live_cache(cache_root, rows)
        print(f"[agibot resume] complete cache already exists; skipped {len(rows)} episode(s): {cache_root}", flush=True)
        return manifest
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = sum(
        (job.segment.end - job.segment.start) * (66 + PROPRIO_DIM) * 4
        for job in jobs
    )
    required_bytes = math.ceil(estimated_bytes * 1.3)
    available_bytes = (
        os.statvfs(cache_root.parent).f_bavail
        * os.statvfs(cache_root.parent).f_frsize
    )
    if available_bytes < required_bytes:
        raise OSError(f"insufficient cache space: need {required_bytes}, have {available_bytes}")

    replay.validate_runtime_assets()
    if replay.inference_backend(args) == "onnxruntime":
        devices, gpu_source = ["cpu"], "cpu"
    else:
        devices, gpu_source = replay.discover_visible_gpus()
    specs = _build_worker_specs(devices, args, len(jobs))
    args.work_root = (args.work_root / mapping_hash[:12]).resolve()
    staging = cache_root.with_name(f".{cache_root.name}.staging")
    started = time.monotonic()
    stop_event = threading.Event()
    with _cache_lock(cache_root):
        if staging.exists() and not args.resume:
            raise FileExistsError(f"staging exists; pass --resume after inspection: {staging}")
        if not staging.exists():
            (staging / "sonic_tokens").mkdir(parents=True)
            (staging / "g1_proprio").mkdir()
            (staging / "meta").mkdir()
        journal_path = staging / "meta/execution_results.jsonl"
        task_queue: queue.Queue[AgibotJob] = queue.Queue()
        result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        for job in jobs:
            task_queue.put(job)
        gate = replay.TensorRTInitGate(replay.TRT_LOCK_PATH)

        def worker(spec: replay.WorkerSpec) -> None:
            while not stop_event.is_set():
                try:
                    job = task_queue.get_nowait()
                except queue.Empty:
                    return
                try:
                    trajectory, trajectory_check = _trajectory_for_job(job, args)
                    row, provenance = _row_and_provenance(
                        job,
                        trajectory,
                        trajectory_check,
                        mapping_hash,
                        asset_hash,
                    )
                    planner_job = replay.PlannerJob(
                        trajectory_source=None,
                        token_output=staging / "sonic_tokens" / f"{job.stable_id}.npy",
                        proprio_output=staging / "g1_proprio" / f"{job.stable_id}.npy",
                        episode_id=job.stable_id,
                    )
                    existing = replay.resume_planner_report(
                        planner_job, trajectory, provenance=provenance
                    )
                    if existing is not None:
                        if not args.resume:
                            raise FileExistsError(f"output already exists for {job.stable_id}")
                        status = "skipped"
                        report = existing
                    else:
                        report = replay.run_planner_job(
                            planner_job,
                            trajectory,
                            args,
                            spec,
                            gate,
                            stop_event,
                            provenance=provenance,
                        )
                        status = "complete-with-warnings" if report["warnings"] else "complete"
                    result_queue.put(
                        {
                            "status": status,
                            "stable_id": job.stable_id,
                            "row": row,
                            "report": report,
                        }
                    )
                except BaseException as error:
                    result_queue.put(
                        {
                            "status": "failed",
                            "stable_id": job.stable_id,
                            "error": f"{type(error).__name__}: {error}",
                            "runtime_assignment": replay.runtime_assignment(spec),
                        }
                    )
                    if args.fail_fast:
                        stop_event.set()

        threads = [
            threading.Thread(
                target=worker,
                args=(spec,),
                name=f"agibot-planner-worker-{spec.index}",
            )
            for spec in specs
        ]
        results: list[dict[str, Any]] = []
        interrupted = False
        try:
            for thread in threads:
                thread.start()
            with journal_path.open("a", encoding="utf-8") as journal:
                while any(thread.is_alive() for thread in threads) or not result_queue.empty():
                    try:
                        result = result_queue.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    results.append(result)
                    journal.write(
                        json.dumps(
                            {"recorded_at": replay.now_utc(), **teleop._jsonable(result)}
                        )
                        + "\n"
                    )
                    journal.flush()
                    print(
                        f"[{len(results)}/{len(jobs)}] {result['status']} {result['stable_id']}",
                        flush=True,
                    )
                    if result["status"] == "failed":
                        print(result["error"], file=sys.stderr, flush=True)
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
        finally:
            for thread in threads:
                thread.join()
            while True:
                try:
                    results.append(result_queue.get_nowait())
                except queue.Empty:
                    break
        failures = [result for result in results if result["status"] == "failed"]
        if interrupted:
            raise KeyboardInterrupt
        if failures or len(results) != len(jobs):
            raise RuntimeError(
                f"planner conversion incomplete: results={len(results)}/{len(jobs)}, failures={len(failures)}"
            )

        rows_by_id = {result["stable_id"]: result["row"] for result in results}
        expected_ids = sorted(job.stable_id for job in jobs)
        if set(rows_by_id) != set(expected_ids):
            raise ValueError("worker results do not match expected stable IDs")
        rows = [rows_by_id[stable_id] for stable_id in expected_ids]
        language_map = {row["stable_id"]: row["main_task"] for row in rows}
        teleop._atomic_write_json(staging / "meta/schema.json", _live_schema(mapping))
        teleop._atomic_write_json(staging / "meta/language_map.json", language_map)
        teleop._atomic_write_jsonl(staging / "meta/episode_index.jsonl", rows)
        stats_meta = _write_stats(staging, rows)
        status_counts: dict[str, int] = {}
        for result in results:
            status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        lerobot_summary = None
        if not args.skip_lerobot_v2_1:
            lerobot_summary = _write_lerobot_v2_1(staging, rows, args)
        manifest = {
            "format_version": 1,
            "dataset": "agibotworld",
            "pipeline": "sonic_planner_vr3_paired_token_proprio",
            "pilot": False,
            "data_root": str(args.data_root),
            "cache_root": str(cache_root),
            "raw_inventory_fingerprint": preflight.inventory_fingerprint,
            "mapping": mapping,
            "mapping_config_sha256": mapping_hash,
            "runtime_asset_hashes": asset_hashes,
            "runtime_asset_set_sha256": asset_hash,
            "gpu_discovery_source": gpu_source,
            "inference_backend": replay.inference_backend(args),
            "workers": replay.worker_inventory(specs),
            "gpu_oversubscription": bool(args.allow_gpu_oversubscription),
            "sharding": {
                "shard_count": args.shard_count,
                "shard_index": args.shard_index,
                "method": "tfrecord_shard_index_modulo",
            },
            "elapsed_seconds": time.monotonic() - started,
            "counts": {
                "eligible_episodes": len(preflight.episodes),
                "excluded_scanned_episodes": len(preflight.exclusions),
                "segments": len(rows),
                "source_frames": sum(int(row["source_frame_count"]) for row in rows),
                "token_frames": sum(int(row["token_frame_count"]) for row in rows),
                "proprio_frames": sum(int(row["proprio_frame_count"]) for row in rows),
                "statuses": status_counts,
            },
            "output_space": {
                "estimated_bytes": estimated_bytes,
                "required_free_bytes": required_bytes,
                "available_free_bytes_at_start": available_bytes,
            },
            "stats": stats_meta,
            "lerobot_v2_1": lerobot_summary,
            "exclusions": list(preflight.exclusions),
            "cli_resolved_config": teleop._jsonable(vars(args)),
        }
        teleop._atomic_write_json(staging / "meta/conversion_manifest.json", manifest)
        teleop._atomic_write_json(
            staging / "meta/execution_report.json",
            {"status_counts": status_counts, "workers": replay.worker_inventory(specs)},
        )
        _validate_live_cache(staging, rows)
        (staging / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        os.replace(staging, cache_root)
        return manifest


def _preflight_summary(preflight: teleop.Preflight, args: argparse.Namespace) -> dict[str, Any]:
    mapping = _mapping_config(args)
    mapping_hash = _json_hash(mapping)
    checks: list[dict[str, Any]] = []
    trajectory_exclusions: list[dict[str, str]] = []
    for job in _jobs(preflight):
        try:
            _, check = _trajectory_for_job(job, args)
            checks.append(check)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            trajectory_exclusions.append(
                {"source": job.episode.relative_source, "reason": str(exc)}
            )
    trajectory_summary: dict[str, Any] = {"segments_checked": len(checks)}
    if checks:
        trajectory_summary.update(
            {
                "rows": sum(int(check["rows"]) for check in checks),
                "movement_rows": sum(int(check["movement_rows"]) for check in checks),
                "turn_only_rows": sum(int(check["turn_only_rows"]) for check in checks),
                "squat_rows": sum(int(check["squat_rows"]) for check in checks),
                "height_min": min(float(check["height_min"]) for check in checks),
                "height_max": max(float(check["height_max"]) for check in checks),
                "left_closed_fraction_mean": float(
                    np.mean([check["left_closed_fraction"] for check in checks])
                ),
                "right_closed_fraction_mean": float(
                    np.mean([check["right_closed_fraction"] for check in checks])
                ),
            }
        )
    return {
        "mode": "preflight-only",
        "eligible_episodes": len(preflight.episodes),
        "excluded_scanned_episodes": len(preflight.exclusions),
        "segments": sum(len(episode.segments) for episode in preflight.episodes),
        "valid_source_frames": sum(episode.valid_frames for episode in preflight.episodes),
        "raw_inventory_fingerprint": preflight.inventory_fingerprint,
        "sharding": {
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "method": "tfrecord_shard_index_modulo",
        },
        "mapping": mapping,
        "mapping_config_sha256": mapping_hash,
        "trajectory_summary": trajectory_summary,
        "trajectory_exclusions": trajectory_exclusions,
        "episodes": [episode.relative_source for episode in preflight.episodes],
        "exclusions": list(preflight.exclusions),
        "live_runtime_started": False,
    }


def _validate_trajectory_cache(root: Path, rows: list[dict[str, Any]]) -> None:
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
        path = root / "planner_trajectories" / f"{row['stable_id']}.npz"
        trajectory = replay.load_planner_trajectory(path)
        if trajectory.rows != int(row["trajectory_frame_count"]):
            raise ValueError(f"trajectory row mismatch for {row['stable_id']}")
        report_path = root / "meta/trajectory_reports" / f"{row['stable_id']}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("rows") != trajectory.rows:
            raise ValueError(f"invalid trajectory report for {row['stable_id']}")


def _write_pilot(preflight: teleop.Preflight, args: argparse.Namespace, cache_root: Path) -> dict[str, Any]:
    teleop._validate_roots(args.data_root, cache_root, pilot=True)
    if cache_root.exists():
        raise FileExistsError(f"cache already exists; refusing to overwrite: {cache_root}")
    mapping = _mapping_config(args)
    mapping_hash = _json_hash(mapping)
    asset_hashes = _asset_hashes()
    asset_hash = _json_hash(asset_hashes)
    jobs = _jobs(preflight)
    staging = cache_root.with_name(f".{cache_root.name}.staging")
    if staging.exists():
        raise FileExistsError(f"staging exists; inspect/remove before retrying: {staging}")
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        (staging / "planner_trajectories").mkdir(parents=True)
        (staging / "meta/trajectory_reports").mkdir(parents=True)
        (staging / "meta").mkdir(exist_ok=True)
        for index, job in enumerate(jobs, start=1):
            trajectory, check = _trajectory_for_job(job, args)
            output_path = staging / "planner_trajectories" / f"{job.stable_id}.npz"
            replay.atomic_npz(output_path, _trajectory_arrays(trajectory))
            loaded = replay.load_planner_trajectory(output_path)
            if loaded.rows != trajectory.rows:
                raise ValueError(f"saved trajectory failed row validation: {job.stable_id}")
            row, report = _row_and_report(
                job,
                trajectory,
                check,
                mapping_hash,
                output_path.relative_to(staging),
            )
            rows.append(row)
            teleop._atomic_write_json(
                staging / "meta/trajectory_reports" / f"{job.stable_id}.json",
                report,
            )
            print(
                f"[{index}/{len(jobs)}] trajectory {job.stable_id} rows={trajectory.rows}",
                flush=True,
            )
        rows.sort(key=lambda item: item["stable_id"])
        language_map = {row["stable_id"]: row["main_task"] for row in rows}
        teleop._atomic_write_json(staging / "meta/schema.json", _schema(mapping))
        teleop._atomic_write_json(staging / "meta/language_map.json", language_map)
        teleop._atomic_write_jsonl(staging / "meta/episode_index.jsonl", rows)
        manifest = {
            "format_version": 1,
            "dataset": "agibotworld",
            "pipeline": "sonic_planner_trajectory_pilot",
            "pilot": True,
            "live_runtime_started": False,
            "data_root": str(args.data_root),
            "cache_root": str(cache_root),
            "raw_inventory_fingerprint": preflight.inventory_fingerprint,
            "mapping": mapping,
            "mapping_config_sha256": mapping_hash,
            "runtime_asset_hashes": asset_hashes,
            "runtime_asset_set_sha256": asset_hash,
            "elapsed_seconds": time.monotonic() - started,
            "counts": {
                "eligible_episodes": len(preflight.episodes),
                "excluded_scanned_episodes": len(preflight.exclusions),
                "segments": len(rows),
                "trajectory_frames": sum(int(row["trajectory_frame_count"]) for row in rows),
            },
            "exclusions": list(preflight.exclusions),
            "cli_resolved_config": teleop._jsonable(vars(args)),
            "gpu_replay": {
                "status": "not_run",
                "command_template": (
                    "PYTHONPATH=src:scripts/preprocess/calc_proprio .venv/bin/python "
                    "scripts/preprocess/calc_proprio/run.py "
                    "--planner-trajectory <pilot>/planner_trajectories/<stable_id>.npz "
                    "--token-output <output>/sonic_tokens/<stable_id>.npy "
                    "--proprio-output <output>/g1_proprio/<stable_id>.npy"
                ),
            },
        }
        teleop._atomic_write_json(staging / "meta/conversion_manifest.json", manifest)
        _validate_trajectory_cache(staging, rows)
        (staging / "_PILOT_TRAJECTORY_SUCCESS").write_text("ok\n", encoding="utf-8")
        os.replace(staging, cache_root)
        return manifest
    except BaseException:
        raise


def _workers(value: str) -> str | int:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return parsed


def _quat_arg(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 4:
        raise argparse.ArgumentTypeError("quaternion requires 4 values: w x y z")
    norm = float(np.linalg.norm(values))
    if norm < 1e-8:
        raise argparse.ArgumentTypeError("quaternion must be non-zero")
    return tuple(float(value) / norm for value in values)


def _vec3_arg(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("vector requires 3 values")
    return tuple(float(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--publish-full", action="store_true")
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--confirm-calibration", action="store_true")
    calibration.add_argument("--allow-provisional-calibration", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--preflight-workers", type=_workers, default="auto")
    parser.add_argument("--preflight-cache", type=Path)
    parser.add_argument("--refresh-preflight-cache", action="store_true")
    parser.add_argument("--min-source-frames", type=int, default=20)
    parser.add_argument("--confirm-source-schema", action="store_true")
    parser.add_argument("--workers", type=int, help="default: one worker per visible GPU/CPU backend slot")
    parser.add_argument("--inference-backend", default="tensorrt", choices=("tensorrt", "onnxruntime", "ort", "cpu"))
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--zmq-port-base", type=int, default=replay.DEFAULT_ZMQ_PORT_BASE)
    parser.add_argument("--dds-domain-base", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--allow-gpu-oversubscription", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--deploy-timeout", type=float, default=150.0)
    parser.add_argument("--capture-startup-margin", type=float, default=2.0)
    parser.add_argument("--capture-tail-margin", type=float, default=15.0)
    parser.add_argument("--startup-settle-margin", type=float, default=11.0)
    parser.add_argument("--planner-mode-timeout", type=float, default=30.0)
    parser.add_argument("--planner-echo-tol", type=float, default=1e-4)
    parser.add_argument("--anchor-max-offset", type=int, default=5)
    parser.add_argument("--anchor-tol", type=float, default=1e-5)
    parser.add_argument("--max-edge-fill", type=int, default=0)
    parser.add_argument("--allow-cross-root-cache", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-lerobot-v2-1", action="store_true")
    parser.add_argument("--lerobot-chunk-size", type=int, default=1000)
    parser.add_argument("--lerobot-video-key", default="observation.images.rgb.head")
    parser.add_argument("--lerobot-video-file", default="head_color.mp4")
    parser.add_argument("--calibration-label", default="agibotworld_gripper_provisional_v1")
    parser.add_argument("--base-axis-scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--base-filter-alpha", type=float, default=0.2)
    parser.add_argument("--linear-deadband", type=float, default=0.02)
    parser.add_argument("--yaw-deadband", type=float, default=0.02)
    parser.add_argument("--source-minimum-waist-lift", type=float, default=DEFAULT_SOURCE_MINIMUM_WAIST_LIFT)
    parser.add_argument("--source-standing-waist-lift", type=float, default=DEFAULT_SOURCE_STANDING_WAIST_LIFT)
    parser.add_argument("--g1-standing-height", type=float, default=DEFAULT_G1_STANDING_HEIGHT)
    parser.add_argument("--minimum-height", type=float, default=DEFAULT_MINIMUM_HEIGHT)
    parser.add_argument("--maximum-height", type=float, default=DEFAULT_MAXIMUM_HEIGHT)
    parser.add_argument("--height-filter-alpha", type=float, default=0.2)
    parser.add_argument("--squat-height-threshold", type=float, default=0.75)
    parser.add_argument("--canonical-head-z", type=float, default=teleop.DEFAULT_CANONICAL_HEAD_Z)
    parser.add_argument("--calibration-frames", type=int, default=30)
    parser.add_argument("--arm-axis-scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--left-flange-position-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--right-flange-position-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--left-flange-rotation-wxyz", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--right-flange-rotation-wxyz", nargs=4, type=float, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument("--max-wrist-head-reach", type=float, default=0.85)
    parser.add_argument("--base-yaw-scale", type=float, default=1.0)
    parser.add_argument("--head-yaw-sign", type=float, default=1.0)
    parser.add_argument("--head-pitch-sign", type=float, default=1.0)
    parser.add_argument("--head-yaw-offset", type=float, default=0.0)
    parser.add_argument("--head-pitch-offset", type=float, default=0.0)
    parser.add_argument("--gripper-closed-mm-threshold", type=float, default=DEFAULT_GRIPPER_CLOSED_MM_THRESHOLD)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    write_mode = bool(args.pilot or args.publish_full)
    if write_mode and not args.confirm_source_schema:
        raise ValueError("pilot/full requires --confirm-source-schema")
    if write_mode and not (
        args.confirm_calibration or args.allow_provisional_calibration
    ):
        raise ValueError(
            "pilot/full requires --confirm-calibration or --allow-provisional-calibration"
        )
    if args.resume and not args.publish_full:
        raise ValueError("--resume is only valid with --publish-full")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must be in [0, shard_count)")
    if not 1 <= args.zmq_port_base <= 65534:
        raise ValueError("--zmq-port-base must be in [1,65534]")
    if not 0 <= args.dds_domain_base <= replay.MAX_DDS_DOMAIN:
        raise ValueError(f"--dds-domain-base must be in [0,{replay.MAX_DDS_DOMAIN}]")
    if args.max_edge_fill != 0:
        raise ValueError("planner mode forbids edge fill; --max-edge-fill must be 0")
    if args.lerobot_chunk_size <= 0:
        raise ValueError("--lerobot-chunk-size must be positive")
    if args.max_episodes < 0 or args.min_source_frames < 2:
        raise ValueError("episode limits are invalid")
    if args.calibration_frames < 2:
        raise ValueError("--calibration-frames must be >= 2")
    for name in ("base_filter_alpha", "height_filter_alpha"):
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0,1]")
    if args.linear_deadband < 0 or args.yaw_deadband < 0:
        raise ValueError("deadbands must be non-negative")
    if not 0.2 <= args.minimum_height < args.maximum_height <= 0.8:
        raise ValueError("height range must satisfy 0.2 <= min < max <= 0.8")
    if not args.minimum_height <= args.g1_standing_height <= args.maximum_height:
        raise ValueError("G1 standing height must be inside the height range")
    if not args.minimum_height <= args.squat_height_threshold <= args.maximum_height:
        raise ValueError("squat threshold must be inside the height range")
    if args.source_standing_waist_lift <= args.source_minimum_waist_lift:
        raise ValueError("source standing waist lift must exceed source minimum waist lift")
    if args.gripper_closed_mm_threshold <= 0.0:
        raise ValueError("--gripper-closed-mm-threshold must be positive")
    args.left_flange_position_offset = _vec3_arg(list(args.left_flange_position_offset))
    args.right_flange_position_offset = _vec3_arg(list(args.right_flange_position_offset))
    args.left_flange_rotation_wxyz = _quat_arg(list(args.left_flange_rotation_wxyz))
    args.right_flange_rotation_wxyz = _quat_arg(list(args.right_flange_rotation_wxyz))


def _self_check() -> None:
    rows = 5
    arrays = {
        "end_position": np.zeros((rows, 2, 3), dtype=np.float32),
        "end_orientation": np.zeros((rows, 2, 4), dtype=np.float32),
        "gripper": np.asarray([[40.0, 80.0]] * rows, dtype=np.float32),
        "head": np.zeros((rows, 2), dtype=np.float32),
        "waist": np.asarray([[0.0, 0.37]] * rows, dtype=np.float32),
        "robot_position": np.zeros((rows, 3), dtype=np.float32),
        "robot_orientation": np.zeros((rows, 4), dtype=np.float32),
    }
    arrays["end_position"][:, 0] = [0.6, 0.2, 0.5]
    arrays["end_position"][:, 1] = [0.6, -0.2, 0.5]
    arrays["end_position"][:, 0, 0] += np.linspace(0.0, 0.04, rows)
    arrays["end_orientation"][..., 3] = 1.0
    segment = teleop.Segment(0, rows)
    episode = teleop.Episode(
        dataset="agibotworld",
        source=Path("synthetic.tfrecord"),
        relative_source="synthetic.tfrecord#record=0",
        stable_base="synthetic",
        source_frames=rows,
        source_hz=SOURCE_HZ,
        main_task="synthetic",
        segments=(segment,),
        valid_frames=rows,
        stats={},
    )
    args = parse_args()
    args.data_root = Path(".")
    _validate_args(args)
    trajectory, check = _trajectory_for_job(
        AgibotJob(episode, 0), args, arrays=arrays
    )
    assert trajectory.rows == rows
    assert trajectory.vr_position.shape == (rows, 9)
    assert trajectory.vr_orientation.shape == (rows, 12)
    assert np.array_equal(trajectory.hand_state[:, 0], np.ones(rows, dtype=np.float32))
    assert np.array_equal(trajectory.hand_state[:, 1], np.zeros(rows, dtype=np.float32))
    assert check["vr_quaternion_norm_max_err"] < 1e-6
    print("self-check passed: AgiBot planner trajectory contract")


def main() -> int:
    args = parse_args()
    if args.self_check:
        _self_check()
        return 0
    _validate_args(args)
    args.data_root = _resolve(args.data_root)
    args.cache_root = _resolve(args.cache_root)
    args.work_root = _resolve(args.work_root)
    args.preflight_cache = _resolve(args.preflight_cache) if args.preflight_cache else None
    args.stats = _resolve(args.stats) if args.stats else None
    if args.stats is not None and not args.stats.is_file():
        raise FileNotFoundError(f"--stats does not exist: {args.stats}")
    max_episodes_explicit = any(
        token == "--max-episodes" or token.startswith("--max-episodes=")
        for token in sys.argv[1:]
    )
    limit = None if ((args.publish_full and not max_episodes_explicit) or args.max_episodes == 0) else args.max_episodes
    preflight = None
    if args.preflight_cache and not args.refresh_preflight_cache:
        preflight = _load_preflight_cache(args.preflight_cache, args, limit)
    if preflight is None:
        preflight = _preflight_agibot_sharded(
            args.data_root,
            max_episodes=limit,
            min_source_frames=args.min_source_frames,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        if args.preflight_cache:
            _write_preflight_cache(args.preflight_cache, preflight, args, limit)
    if not (args.pilot or args.publish_full):
        print(json.dumps(_preflight_summary(preflight, args), indent=2, sort_keys=True))
        return 0
    cache_root = args.cache_root
    if args.pilot:
        if cache_root == _resolve(CACHE_ROOT):
            cache_root = Path(tempfile.gettempdir()) / (
                f"huge-agibotworld-planner-trajectory-pilot-{os.getpid()}"
            )
        manifest = _write_pilot(preflight, args, cache_root)
    else:
        if cache_root == _resolve(CACHE_ROOT):
            cache_root = _resolve(TOKEN_CACHE_ROOT)
        manifest = _run_live(preflight, args, cache_root)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _termination(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _termination)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None

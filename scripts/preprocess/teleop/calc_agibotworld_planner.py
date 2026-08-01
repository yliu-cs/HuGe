#!/usr/bin/env python3
"""Build AgiBotWorld gripper SONIC planner trajectory pilots.

The default mode is read-only preflight. ``--pilot`` writes only dataset-neutral
PlannerTrajectory ``.npz`` files and validation reports. It does not start the
live SONIC planner replay and therefore does not require a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


DATA_ROOT = Path("resource/datasets/agibotworld-beta-rlds")
CACHE_ROOT = Path("resource/datasets/agibotworld-beta-rlds-planner-trajectory-cache")
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
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--confirm-calibration", action="store_true")
    calibration.add_argument("--allow-provisional-calibration", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--preflight-workers", type=_workers, default="auto")
    parser.add_argument("--min-source-frames", type=int, default=20)
    parser.add_argument("--confirm-source-schema", action="store_true")
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
    if (args.pilot and not args.confirm_source_schema):
        raise ValueError("pilot requires --confirm-source-schema")
    if args.pilot and not (
        args.confirm_calibration or args.allow_provisional_calibration
    ):
        raise ValueError(
            "pilot requires --confirm-calibration or --allow-provisional-calibration"
        )
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
    limit = None if args.max_episodes == 0 else args.max_episodes
    preflight = teleop._preflight_agibot(
        args.data_root,
        max_episodes=limit,
        min_source_frames=args.min_source_frames,
    )
    if not args.pilot:
        print(json.dumps(_preflight_summary(preflight, args), indent=2, sort_keys=True))
        return 0
    cache_root = args.cache_root
    if cache_root == _resolve(CACHE_ROOT):
        cache_root = Path(tempfile.gettempdir()) / (
            f"huge-agibotworld-planner-trajectory-pilot-{os.getpid()}"
        )
    manifest = _write_pilot(preflight, args, cache_root)
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

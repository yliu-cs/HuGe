#!/usr/bin/env python3
"""Build paired BEHAVIOR-1K SONIC planner tokens and pseudo proprio.

The default mode is read-only preflight. ``--pilot`` writes an atomic cache
under ``/tmp`` by default; ``--publish-full`` is required for the full cache.
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
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[3]
CALC_PROPRIO_DIR = REPO_ROOT / "scripts/preprocess/calc_proprio"
sys.path.insert(0, str(CALC_PROPRIO_DIR))

import calc_sonic_tokens as teleop
import run as replay
from huge.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT, PROPRIO_LAYOUT_VERSION

DATA_ROOT = Path("resource/datasets/behavior-1k-2025-challenge")
CACHE_ROOT = Path("resource/datasets/behavior-1k-2025-challenge-planner-cache")
WORK_ROOT = Path(".runs/calc_proprio/behavior1k-planner")
SOURCE_HZ = 30.0
CONTROL_HZ = 50.0
SPEED_MULTIPLIER = CONTROL_HZ / SOURCE_HZ
PILOT_EPISODES = (
    10,      # low-motion manipulation baseline
    260030,  # forward/backward and positive yaw
    101290,  # backward and negative yaw
    441510,  # positive lateral motion and large height range
    442090,  # negative lateral motion
    310380,  # bidirectional yaw
    350330,  # largest observed height range in the eligible probe
    60630,   # height motion with manipulation
    130170,  # high locomotion fraction
)


@dataclass(frozen=True)
class BehaviorJob:
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
    return (expanded if expanded.is_absolute() else REPO_ROOT / expanded).absolute()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
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


def _controller_path(version_root: Path, episode_index: int) -> Path:
    return (
        version_root
        / "meta/episodes"
        / f"task-{episode_index // 10000:04d}"
        / f"episode_{episode_index:08d}.json"
    )


def _controller_contract(path: Path) -> dict[str, Any]:
    outer = json.loads(path.read_text(encoding="utf-8"))
    config = outer.get("config")
    config = json.loads(config) if isinstance(config, str) else config
    if not isinstance(config, dict):
        raise ValueError(f"missing nested config object in {path}")
    robots = config.get("robots")
    if not isinstance(robots, list) or len(robots) != 1:
        raise ValueError(f"expected one robot config in {path}")
    robot = robots[0]
    controllers = robot.get("controller_config")
    if not isinstance(controllers, dict):
        raise ValueError(f"missing robot controller_config in {path}")
    required = {
        "base",
        "trunk",
        "arm_left",
        "arm_right",
        "gripper_left",
        "gripper_right",
    }
    if not required.issubset(controllers):
        raise ValueError(f"controller_config in {path} is missing {sorted(required - controllers.keys())}")
    return {
        "action_normalize": robot.get("action_normalize"),
        "controller_config": {name: controllers[name] for name in sorted(required)},
    }


def _scan_controller_contracts(
    version_root: Path,
    raw_jobs: list[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    paths = [
        _controller_path(version_root, int(job["row"]["episode_index"]))
        for job in raw_jobs
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing controller config(s), first: {missing[0]}")
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(paths)))) as pool:
        contracts = list(pool.map(_controller_contract, paths))
    fingerprints: dict[str, int] = {}
    for contract in contracts:
        fingerprint = _json_hash(contract)
        fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
    if len(fingerprints) != 1:
        raise ValueError(f"BEHAVIOR controller contract is not uniform: {fingerprints}")
    contract = contracts[0]
    if contract["action_normalize"] is not False:
        raise ValueError("BEHAVIOR source contract requires action_normalize=false")
    base = contract["controller_config"]["base"]
    if base.get("name") != "HolonomicBaseJointController" or base.get("motor_type") != "velocity":
        raise ValueError(f"unexpected BEHAVIOR base controller: {base}")
    return {
        "contract": contract,
        "contract_sha256": next(iter(fingerprints)),
        "scanned_configs": len(paths),
        "unique_fingerprints": fingerprints,
    }


def _discover_preflight(
    args: argparse.Namespace,
) -> tuple[teleop.Preflight, Path, list[dict[str, Any]]]:
    version_root, all_jobs, inventory, _ = teleop._behavior_preflight_jobs(
        args.data_root,
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
    requested = set(args.episode or ())
    if args.pilot and not requested:
        requested = set(PILOT_EPISODES)
    if requested:
        by_index = {int(job["row"]["episode_index"]): job for job in all_jobs}
        missing = sorted(requested - by_index.keys())
        if missing:
            raise ValueError(f"requested episode indices are absent from metadata: {missing}")
        candidate_jobs = [by_index[index] for index in sorted(requested)]
        target_eligible = None
    elif args.publish_full or args.max_episodes == 0:
        candidate_jobs = all_jobs
        target_eligible = None
    else:
        candidate_jobs = all_jobs
        target_eligible = args.max_episodes

    workers = teleop._resolve_preflight_workers(args.preflight_workers, len(candidate_jobs))
    episodes: list[teleop.Episode] = []
    exclusions: list[dict[str, str]] = []
    scanned_jobs: list[dict[str, Any]] = []
    iterator = teleop._iter_behavior_preflight(candidate_jobs, workers)
    try:
        for raw_job, result in zip(candidate_jobs, iterator):
            scanned_jobs.append(raw_job)
            exclusions.extend(result["exclusions"])
            episode = result["episode"]
            if episode is not None:
                episodes.append(episode)
            if len(scanned_jobs) % 100 == 0:
                print(
                    f"[behavior planner preflight] scanned={len(scanned_jobs)} "
                    f"eligible={len(episodes)} excluded={len(exclusions)}",
                    flush=True,
                )
            if target_eligible is not None and len(episodes) >= target_eligible:
                break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    if not episodes:
        raise ValueError("preflight found no eligible BEHAVIOR episodes")
    bad_frequency = [
        (episode.relative_source, episode.source_hz)
        for episode in episodes
        if abs(episode.source_hz - SOURCE_HZ) > 0.05
    ]
    if bad_frequency:
        raise ValueError(
            f"BEHAVIOR source frequency differs from the fixed 30 Hz contract: {bad_frequency[:5]}"
        )
    if requested:
        eligible_indices = {
            int(episode.stable_base.rsplit("__", 1)[1]) for episode in episodes
        }
        rejected = sorted(requested - eligible_indices)
        if rejected:
            raise ValueError(f"requested episodes failed pose/calibration preflight: {rejected}")
    preflight = teleop.Preflight(
        episodes=tuple(episodes),
        exclusions=tuple(exclusions),
        scanned_sources=inventory,
        inventory_fingerprint=teleop._inventory_fingerprint(version_root, inventory),
    )
    return preflight, version_root, scanned_jobs


def _lowpass(values: np.ndarray, alpha: float) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    result = np.empty_like(source)
    result[0] = source[0]
    for index in range(1, len(source)):
        result[index] = alpha * source[index] + (1.0 - alpha) * result[index - 1]
    return result


def _mapping_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_hz": SOURCE_HZ,
        "control_hz": CONTROL_HZ,
        "resampling": "none",
        "frame_mapping": "one_source_frame_to_one_control_tick",
        "duration_scale": SOURCE_HZ / CONTROL_HZ,
        "speed_multiplier": SPEED_MULTIPLIER,
        "action_layout": {
            "base": [0, 3],
            "trunk": [3, 7],
            "left_gripper": args.behavior_left_gripper_index,
            "right_gripper": args.behavior_right_gripper_index,
        },
        "base_axis_scale": list(args.base_axis_scale),
        "base_filter_alpha": args.base_filter_alpha,
        "linear_deadband": args.linear_deadband,
        "yaw_deadband": args.yaw_deadband,
        "mode_policy": {
            "idle": 0,
            "walk_or_turn": 2,
            "stationary_below_squat_threshold": 4,
            "locomotion_precedes_squat_when_both_are_requested": True,
        },
        "height": {
            "proxy": "raw_head_camera_z",
            "source_standing_head_z": args.source_standing_head_z,
            "g1_standing_height": args.g1_standing_height,
            "minimum": args.minimum_height,
            "maximum": args.maximum_height,
            "filter_alpha": args.height_filter_alpha,
            "squat_threshold": args.squat_height_threshold,
        },
        "three_point": {
            "method": "existing_behavior_head_relative_limb_ratio_retargeting",
            "canonical_head_z": args.canonical_head_z,
            "calibration_frames": args.behavior_calibration_frames,
            "calibration_max_position_delta": args.behavior_calibration_max_position_delta,
            "calibration_max_angle_degrees": args.behavior_calibration_max_angle_degrees,
            "max_wrist_head_reach": args.behavior_max_wrist_head_reach,
        },
        "hand": {
            "closed_max": args.behavior_gripper_closed_max,
            "closed": 1,
            "open": 0,
        },
    }


def _teleop_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "canonical_head_z": args.canonical_head_z,
        "behavior_max_wrist_head_reach": args.behavior_max_wrist_head_reach,
        "behavior_left_gripper_index": args.behavior_left_gripper_index,
        "behavior_right_gripper_index": args.behavior_right_gripper_index,
        "behavior_gripper_closed_max": args.behavior_gripper_closed_max,
        "allow_legacy_egoverse": False,
        "egoverse_hand_distance_threshold": 0.045,
    }


def _trajectory_for_job(
    job: BehaviorJob,
    args: argparse.Namespace,
) -> tuple[replay.PlannerTrajectory, dict[str, Any]]:
    episode = job.episode
    segment = job.segment
    arrays = teleop._read_behavior_arrays(episode.source)
    positions, orientations, hands = teleop._episode_payload(
        episode,
        segment,
        _teleop_config(args),
        arrays=arrays,
    )
    selection = slice(segment.start, segment.end)
    base = _lowpass(arrays["actions"][selection, :3], args.base_filter_alpha)
    base *= np.asarray(args.base_axis_scale, dtype=np.float32)
    command = base * SPEED_MULTIPLIER
    linear_speed = np.linalg.norm(command[:, :2], axis=1)
    linear_active = linear_speed >= args.linear_deadband
    yaw_active = np.abs(command[:, 2]) >= args.yaw_deadband
    command[~linear_active, :2] = 0.0
    command[~yaw_active, 2] = 0.0
    linear_speed = np.linalg.norm(command[:, :2], axis=1)

    yaw = np.zeros(len(command), dtype=np.float32)
    if len(command) > 1:
        yaw[1:] = np.cumsum(command[:-1, 2] / CONTROL_HZ, dtype=np.float32)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    facing = np.stack((cosine, sine, np.zeros_like(yaw)), axis=1).astype(np.float32)
    world_velocity = np.stack(
        (
            cosine * command[:, 0] - sine * command[:, 1],
            sine * command[:, 0] + cosine * command[:, 1],
        ),
        axis=1,
    )
    movement = np.zeros((len(command), 3), dtype=np.float32)
    moving = linear_speed > 0.0
    movement[moving, :2] = world_velocity[moving] / linear_speed[moving, None]

    source_head_z = arrays["poses"][selection, 2, 2]
    height = args.g1_standing_height + (
        source_head_z - args.source_standing_head_z
    )
    height = _lowpass(height[:, None], args.height_filter_alpha)[:, 0]
    height = np.clip(height, args.minimum_height, args.maximum_height).astype(np.float32)

    locomotion = moving | yaw_active
    squat = (~locomotion) & (height < args.squat_height_threshold)
    mode = np.zeros(len(command), dtype=np.float32)
    mode[locomotion] = 2.0
    mode[squat] = 4.0
    speed = np.full(len(command), -1.0, dtype=np.float32)
    speed[moving] = linear_speed[moving]
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
            "turn_only_rows": int((yaw_active & ~moving).sum()),
            "locomotion_height_conflict_rows": int(
                (locomotion & (height < args.squat_height_threshold)).sum()
            ),
            "source_head_z_min": float(source_head_z.min()),
            "source_head_z_max": float(source_head_z.max()),
            "base_command_abs_max": np.max(np.abs(command), axis=0).tolist(),
        }
    )
    return trajectory, check


def _asset_hashes() -> dict[str, str]:
    assets = {
        "encoder": replay.ENCODER_MODEL,
        "decoder": replay.DECODER_MODEL,
        "planner": replay.PLANNER_MODEL,
        "observation_config": replay.OBS_CONFIG,
    }
    return {name: _file_hash(path) for name, path in assets.items()}


def _row_and_provenance(
    job: BehaviorJob,
    trajectory: replay.PlannerTrajectory,
    trajectory_check: dict[str, Any],
    mapping_hash: str,
    source_contract_hash: str,
    asset_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode = job.episode
    segment = job.segment
    source_fingerprint = _source_fingerprint(episode.source, episode.relative_source)
    provenance = {
        "source_path": episode.relative_source,
        "source_fingerprint": source_fingerprint,
        "source_start_frame": segment.start,
        "source_end_frame_exclusive": segment.end,
        "mapping_config_sha256": mapping_hash,
        "source_contract_sha256": source_contract_hash,
        "runtime_asset_set_sha256": asset_hash,
    }
    row = {
        "stable_id": job.stable_id,
        "dataset": "behavior1k",
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
        "speed_multiplier": SPEED_MULTIPLIER,
        "main_task": episode.main_task,
        "encoder_mode": "teleop_planner",
        "lower_body_policy": "sonic_planner",
        "calibration_method": "sonic_live_rigid_offset",
        "calibration_start_frame": segment.calibration_start,
        "calibration_end_frame_exclusive": segment.calibration_end,
        "hand_state_method": "r1pro_action_threshold",
        "left_closed_fraction": float(trajectory.hand_state[:, 0].mean()),
        "right_closed_fraction": float(trajectory.hand_state[:, 1].mean()),
        "trajectory_check": trajectory_check,
        **provenance,
    }
    return row, provenance


def _schema(args: argparse.Namespace, mapping: dict[str, Any]) -> dict[str, Any]:
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
        },
        "mapping": mapping,
    }


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
            "sources": {"behavior1k_planner": {"n_episodes": len(paths), "n_frames": total}},
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


def _validate_cache(root: Path, rows: list[dict[str, Any]]) -> None:
    index_rows = [
        json.loads(line)
        for line in (root / "meta/episode_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if index_rows != rows:
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
            raise ValueError(f"invalid proprio output: {proprio_path}: {proprio.shape}/{proprio.dtype}")
        if not np.isfinite(token).all() or not np.isfinite(proprio).all():
            raise ValueError(f"non-finite paired output for {row['stable_id']}")
        if not np.array_equal(token[:, 64:66], proprio[:, 29:31]):
            raise ValueError(f"token/proprio hands disagree for {row['stable_id']}")
        if report.get("status") != "complete" or report.get("token_rows") != count:
            raise ValueError(f"invalid planner report for {row['stable_id']}")


@contextmanager
def _cache_lock(cache_root: Path) -> Iterator[None]:
    lock_path = cache_root.with_name(f".{cache_root.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another Behavior planner writer holds {cache_root}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _preflight_summary(
    preflight: teleop.Preflight,
    controller_scan: dict[str, Any],
    mapping: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    trajectory_checks: list[dict[str, Any]] = []
    for episode in preflight.episodes:
        for segment_index in range(len(episode.segments)):
            _, check = _trajectory_for_job(
                BehaviorJob(episode, segment_index), args
            )
            trajectory_checks.append(check)
    return {
        "mode": "preflight-only",
        "eligible_episodes": len(preflight.episodes),
        "excluded_scanned_episodes": len(preflight.exclusions),
        "segments": len(trajectory_checks),
        "valid_source_frames": sum(episode.valid_frames for episode in preflight.episodes),
        "raw_inventory_fingerprint": preflight.inventory_fingerprint,
        "source_contract": controller_scan,
        "mapping": mapping,
        "mapping_config_sha256": _json_hash(mapping),
        "trajectory_summary": {
            "rows": sum(int(check["rows"]) for check in trajectory_checks),
            "movement_rows": sum(int(check["movement_rows"]) for check in trajectory_checks),
            "turn_only_rows": sum(int(check["turn_only_rows"]) for check in trajectory_checks),
            "squat_rows": sum(int(check["squat_rows"]) for check in trajectory_checks),
            "locomotion_height_conflict_rows": sum(
                int(check["locomotion_height_conflict_rows"])
                for check in trajectory_checks
            ),
            "height_min": min(float(check["height_min"]) for check in trajectory_checks),
            "height_max": max(float(check["height_max"]) for check in trajectory_checks),
        },
        "episodes": [episode.relative_source for episode in preflight.episodes],
        "exclusions": list(preflight.exclusions),
        "live_runtime_started": False,
    }


def _run_live(
    preflight: teleop.Preflight,
    controller_scan: dict[str, Any],
    args: argparse.Namespace,
    cache_root: Path,
) -> dict[str, Any]:
    teleop._validate_roots(args.data_root, cache_root, pilot=args.pilot)
    if cache_root.exists():
        raise FileExistsError(f"cache already exists; refusing to overwrite: {cache_root}")
    mapping = _mapping_config(args)
    mapping_hash = _json_hash(mapping)
    asset_hashes = _asset_hashes()
    asset_hash = _json_hash(asset_hashes)
    jobs = [
        BehaviorJob(episode, segment_index)
        for episode in preflight.episodes
        for segment_index in range(len(episode.segments))
    ]
    jobs.sort(
        key=lambda item: (-(item.segment.end - item.segment.start), item.stable_id)
    )
    estimated_bytes = sum(
        (job.segment.end - job.segment.start) * (66 + PROPRIO_DIM) * 4
        for job in jobs
    )
    required_bytes = math.ceil(estimated_bytes * 1.3)
    available_bytes = os.statvfs(cache_root.parent).f_bavail * os.statvfs(cache_root.parent).f_frsize
    if not args.pilot and available_bytes < required_bytes:
        raise OSError(f"insufficient cache space: need {required_bytes}, have {available_bytes}")

    replay.validate_runtime_assets()
    devices, gpu_source = replay.discover_visible_gpus()
    specs = replay.build_worker_specs(devices, args, len(jobs))
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
        task_queue: queue.Queue[BehaviorJob] = queue.Queue()
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
                        controller_scan["contract_sha256"],
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
                        {"status": status, "stable_id": job.stable_id, "row": row, "report": report}
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
                name=f"behavior-planner-worker-{spec.index}",
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
                    journal.write(json.dumps({"recorded_at": replay.now_utc(), **teleop._jsonable(result)}) + "\n")
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
        teleop._atomic_write_json(staging / "meta/schema.json", _schema(args, mapping))
        teleop._atomic_write_json(staging / "meta/language_map.json", language_map)
        teleop._atomic_write_jsonl(staging / "meta/episode_index.jsonl", rows)
        stats_meta = _write_stats(staging, rows)
        status_counts: dict[str, int] = {}
        for result in results:
            status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        manifest = {
            "format_version": 1,
            "dataset": "behavior1k",
            "pipeline": "sonic_planner_vr3_paired_token_proprio",
            "pilot": bool(args.pilot),
            "data_root": str(args.data_root),
            "cache_root": str(cache_root),
            "raw_inventory_fingerprint": preflight.inventory_fingerprint,
            "source_contract": controller_scan,
            "mapping": mapping,
            "mapping_config_sha256": mapping_hash,
            "runtime_asset_hashes": asset_hashes,
            "runtime_asset_set_sha256": asset_hash,
            "gpu_discovery_source": gpu_source,
            "workers": replay.worker_inventory(specs),
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
            "exclusions": list(preflight.exclusions),
            "cli_resolved_config": teleop._jsonable(vars(args)),
        }
        teleop._atomic_write_json(staging / "meta/conversion_manifest.json", manifest)
        teleop._atomic_write_json(
            staging / "meta/execution_report.json",
            {"status_counts": status_counts, "workers": replay.worker_inventory(specs)},
        )
        _validate_cache(staging, rows)
        marker = "_PILOT_SUCCESS" if args.pilot else "_SUCCESS"
        (staging / marker).write_text("ok\n", encoding="utf-8")
        os.replace(staging, cache_root)
        return manifest


def _workers(value: str) -> str | int:
    if value == "auto":
        return value
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be 'auto' or a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--publish-full", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--episode", type=int, action="append")
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--preflight-workers", type=_workers, default="auto")
    parser.add_argument("--min-source-frames", type=int, default=20)
    parser.add_argument("--max-timestamp-gap-factor", type=float, default=2.0)
    parser.add_argument("--confirm-source-schema", action="store_true")
    parser.add_argument("--workers", type=int, help="default: one worker per visible GPU")
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--zmq-port-base", type=int, default=replay.DEFAULT_ZMQ_PORT_BASE)
    parser.add_argument("--dds-domain-base", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--deploy-timeout", type=float, default=150.0)
    parser.add_argument("--capture-startup-margin", type=float, default=2.0)
    parser.add_argument("--capture-tail-margin", type=float, default=15.0)
    parser.add_argument("--startup-settle-margin", type=float, default=11.0)
    parser.add_argument("--planner-mode-timeout", type=float, default=15.0)
    parser.add_argument("--planner-echo-tol", type=float, default=1e-4)
    parser.add_argument("--anchor-max-offset", type=int, default=5)
    parser.add_argument("--anchor-tol", type=float, default=1e-5)
    parser.add_argument("--max-edge-fill", type=int, default=0)
    parser.add_argument("--base-axis-scale", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--base-filter-alpha", type=float, default=0.2)
    parser.add_argument("--linear-deadband", type=float, default=0.02)
    parser.add_argument("--yaw-deadband", type=float, default=0.02)
    parser.add_argument("--source-standing-head-z", type=float, default=1.40535)
    parser.add_argument("--g1-standing-height", type=float, default=0.78)
    parser.add_argument("--minimum-height", type=float, default=0.4)
    parser.add_argument("--maximum-height", type=float, default=0.8)
    parser.add_argument("--height-filter-alpha", type=float, default=0.2)
    parser.add_argument("--squat-height-threshold", type=float, default=0.75)
    parser.add_argument("--canonical-head-z", type=float, default=teleop.DEFAULT_CANONICAL_HEAD_Z)
    parser.add_argument("--behavior-left-gripper-index", type=int, default=14)
    parser.add_argument("--behavior-right-gripper-index", type=int, default=22)
    parser.add_argument("--behavior-gripper-closed-max", type=float, default=0.0)
    parser.add_argument("--behavior-calibration-frames", type=int, default=30)
    parser.add_argument("--behavior-calibration-max-position-delta", type=float, default=0.02)
    parser.add_argument("--behavior-calibration-max-angle-degrees", type=float, default=5.0)
    parser.add_argument(
        "--behavior-max-wrist-head-reach",
        type=float,
        default=teleop.DEFAULT_BEHAVIOR_MAX_WRIST_HEAD_REACH,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if (args.pilot or args.publish_full) and not args.confirm_source_schema:
        raise ValueError("pilot/full requires --confirm-source-schema")
    if args.resume and not (args.pilot or args.publish_full):
        raise ValueError("--resume is only valid with --pilot or --publish-full")
    if args.max_episodes < 0 or args.min_source_frames < 2:
        raise ValueError("episode limits are invalid")
    if args.max_timestamp_gap_factor <= 1.0:
        raise ValueError("--max-timestamp-gap-factor must exceed 1")
    if args.workers is not None and args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not 1 <= args.zmq_port_base <= 65534:
        raise ValueError("--zmq-port-base must be in [1,65534]")
    if not 0 <= args.dds_domain_base <= replay.MAX_DDS_DOMAIN:
        raise ValueError(f"--dds-domain-base must be in [0,{replay.MAX_DDS_DOMAIN}]")
    for name in ("base_filter_alpha", "height_filter_alpha"):
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0,1]")
    if args.linear_deadband < 0 or args.yaw_deadband < 0:
        raise ValueError("deadbands must be non-negative")
    if not 0.2 <= args.minimum_height < args.maximum_height <= 0.8:
        raise ValueError("height range must satisfy 0.2 <= min < max <= 0.8")
    if not args.minimum_height <= args.squat_height_threshold <= args.maximum_height:
        raise ValueError("squat threshold must be inside the height range")
    if args.behavior_left_gripper_index != 14 or args.behavior_right_gripper_index != 22:
        raise ValueError("confirmed BEHAVIOR hand indices are fixed at 14 and 22")
    if args.max_edge_fill != 0:
        raise ValueError("planner mode forbids edge fill; --max-edge-fill must be 0")


def _self_check() -> None:
    values = _lowpass(np.asarray([[0.0], [1.0], [1.0]], dtype=np.float32), 0.5)
    assert np.allclose(values[:, 0], [0.0, 0.5, 0.75])
    rows = 3
    identity = np.zeros((rows, 12), dtype=np.float32)
    identity[:, (0, 4, 8)] = 1.0
    trajectory = replay.PlannerTrajectory(
        mode=np.asarray([0.0, 2.0, 4.0], dtype=np.float32),
        movement=np.asarray([[0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32),
        facing=np.asarray([[1, 0, 0]] * rows, dtype=np.float32),
        speed=np.asarray([-1.0, 0.5, -1.0], dtype=np.float32),
        height=np.asarray([0.78, 0.78, 0.6], dtype=np.float32),
        vr_position=np.zeros((rows, 9), dtype=np.float32),
        vr_orientation=identity,
        hand_state=np.asarray([[0, 0], [1, 0], [1, 1]], dtype=np.float32),
    )
    check = replay.validate_planner_trajectory(trajectory)
    assert check["rows"] == rows
    frames = [
        {
            "index": 100 + index,
            "_recv_monotonic": 10.0 + index / CONTROL_HZ,
            "vr_3point_orientation": trajectory.vr_orientation[index],
            "token_state": np.full(64, index, dtype=np.float32),
            "body_q": np.zeros(29, dtype=np.float32),
            "left_hand_q": np.zeros(7, dtype=np.float32),
            "right_hand_q": np.zeros(7, dtype=np.float32),
            "base_quat": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        for index in range(rows)
    ]
    aligned, alignment = replay.align_planner_frames(
        frames,
        trajectory,
        first_sent_monotonic=10.0,
        echo_tol=1e-6,
    )
    assert aligned["token_state"].shape == (rows, 64)
    assert alignment["selected_frames"] == rows
    print(
        "self-check passed: filtering, planner trajectory contract, and strict paired alignment"
    )


def main() -> int:
    args = parse_args()
    if args.self_check:
        _self_check()
        return 0
    _validate_args(args)
    args.data_root = _resolve(args.data_root)
    args.cache_root = _resolve(args.cache_root)
    args.work_root = _resolve(args.work_root)
    args.stats = _resolve(args.stats) if args.stats else None
    if args.stats is not None and not args.stats.is_file():
        raise FileNotFoundError(f"--stats does not exist: {args.stats}")

    preflight, version_root, scanned_jobs = _discover_preflight(args)
    worker_count = teleop._resolve_preflight_workers(
        args.preflight_workers, len(scanned_jobs)
    )
    controller_scan = _scan_controller_contracts(
        version_root, scanned_jobs, worker_count
    )
    mapping = _mapping_config(args)
    if not (args.pilot or args.publish_full):
        print(
            json.dumps(
                _preflight_summary(preflight, controller_scan, mapping, args),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    cache_root = args.cache_root
    if args.pilot and cache_root == _resolve(CACHE_ROOT):
        cache_root = Path(tempfile.gettempdir()) / (
            f"huge-behavior1k-planner-pilot-{os.getpid()}"
        )
    manifest = _run_live(preflight, controller_scan, args, cache_root)
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

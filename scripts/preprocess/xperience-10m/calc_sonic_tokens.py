"""Preprocess Xperience-10M full-body mocap into 66-D SONIC tokens.

The source tree is read-only. By default this command only preflights three
eligible episodes. ``--pilot`` publishes a small cache under ``/tmp``;
``--publish-full`` is required for the full atomic cache.

Conversion automatically uses one process per visible CUDA GPU; use
``CUDA_VISIBLE_DEVICES`` to restrict assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# GR00T is a read-only local dependency; never leave import bytecode in it.
sys.dont_write_bytecode = True

from gear_sonic.scripts.sonic_v3_to_latent import (
    ENCODER_INPUT_DIM,
    LATENT_DIM,
    build_smpl_encoder_inputs,
    create_onnx_session,
)
from gear_sonic.scripts.trackingdata_to_sonic_npz import _compute_wrist_joint_positions
from huge.data.preprocess import atomic_save_npy


DEFAULT_DATA_ROOT = Path("resource/datasets/xperience-10m")
DEFAULT_CACHE_ROOT = Path("resource/datasets/xperience-10m-cache")
DEFAULT_ENCODER_MODEL = Path("resource/model/sonic/model_encoder_dynbatch.onnx")
TARGET_HZ = 50.0
SOURCE_BODY_JOINTS = np.asarray([*range(22), 36, 51], dtype=np.int64)
LEFT_MANO_TIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
RIGHT_MANO_TIPS = LEFT_MANO_TIPS
MIN_HAND_SPAN_M = 1e-3
ENCODER_FUTURE_FRAMES = 10


@dataclass(frozen=True)
class Segment:
    start: int
    end: int


@dataclass(frozen=True)
class Episode:
    source: Path
    relative_source: str
    stable_base: str
    source_frames: int
    source_hz: float
    main_task: str
    segments: tuple[Segment, ...]
    valid_frames: int
    dropped_short_frames: int


@dataclass(frozen=True)
class Preflight:
    episodes: tuple[Episode, ...]
    exclusions: tuple[dict[str, str], ...]
    scanned_sources: tuple[Path, ...]
    inventory_fingerprint: str


def _discover_annotations(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Xperience data root is not a directory: {root}")
    paths: list[Path] = []
    for directory, directories, filenames in os.walk(root):
        directories.sort()
        if "annotation.hdf5" in filenames:
            paths.append(Path(directory) / "annotation.hdf5")
    if not paths:
        raise FileNotFoundError(f"No annotation.hdf5 files found under {root}")
    return sorted(paths, key=lambda path: path.as_posix())


def _inventory_fingerprint(root: Path, paths: tuple[Path, ...] | list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_base(root: Path, source: Path) -> str:
    relative = source.relative_to(root)
    if len(relative.parts) < 3:
        raise ValueError(f"Unexpected Xperience episode path: {relative}")
    value = f"{relative.parts[-3]}__{relative.parts[-2]}"
    if any(not (character.isalnum() or character in "-_") for character in value):
        raise ValueError(f"Unsafe stable episode id from {relative}: {value!r}")
    return value


def _decode_caption(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError):
        return ""
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    task = config.get("Main Task", "") if isinstance(config, dict) else ""
    return str(task).strip()


def _timestamps_ns(dataset: h5py.Dataset) -> np.ndarray:
    values: list[int] = []
    for raw in dataset[:]:
        if isinstance(raw, bytes):
            raw = raw.decode("ascii", errors="strict")
        values.append(int(raw))
    timestamps = np.asarray(values, dtype=np.int64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError(f"video/device_timestamp must contain at least two values, got {timestamps.shape}")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("video/device_timestamp is not strictly increasing")
    return timestamps


def _require_shape(handle: h5py.File, key: str, expected_tail: tuple[int, ...]) -> int:
    if key not in handle:
        raise ValueError(f"missing required HDF5 dataset {key!r}")
    shape = handle[key].shape
    if len(shape) != len(expected_tail) + 1 or tuple(shape[1:]) != expected_tail:
        raise ValueError(f"{key} expected (T, {expected_tail}), got {shape}")
    return int(shape[0])


def _read_motion_arrays(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        counts = {
            _require_shape(handle, "full_body_mocap/keypoints", (52, 3)),
            _require_shape(handle, "full_body_mocap/body_quats", (21, 4)),
            _require_shape(handle, "full_body_mocap/Ts_world_root", (7,)),
            _require_shape(handle, "hand_mocap/left_joints_3d", (21, 3)),
            _require_shape(handle, "hand_mocap/right_joints_3d", (21, 3)),
        }
        if len(counts) != 1:
            raise ValueError(f"required motion datasets have inconsistent frame counts: {sorted(counts)}")
        frames = counts.pop()
        if "full_body_mocap/frame_nums" not in handle:
            raise ValueError("missing required HDF5 dataset 'full_body_mocap/frame_nums'")
        frame_nums = np.asarray(handle["full_body_mocap/frame_nums"][:], dtype=np.int64)
        timestamps = _timestamps_ns(handle["video/device_timestamp"])
        if frame_nums.shape != (frames,) or timestamps.shape != (frames,):
            raise ValueError(
                f"time arrays must match T={frames}, got frame_nums={frame_nums.shape}, timestamps={timestamps.shape}"
            )
        if not np.array_equal(frame_nums, np.arange(frames, dtype=np.int64)):
            raise ValueError("full_body_mocap/frame_nums must be contiguous zero-based indices")
        caption = _decode_caption(handle["caption"][()]) if "caption" in handle else ""
        return {
            "keypoints": np.asarray(handle["full_body_mocap/keypoints"][:], dtype=np.float32),
            "body_quats": np.asarray(handle["full_body_mocap/body_quats"][:], dtype=np.float32),
            "root": np.asarray(handle["full_body_mocap/Ts_world_root"][:], dtype=np.float32),
            "left_hand": np.asarray(handle["hand_mocap/left_joints_3d"][:], dtype=np.float32),
            "right_hand": np.asarray(handle["hand_mocap/right_joints_3d"][:], dtype=np.float32),
            "timestamps_ns": timestamps,
            "caption": np.asarray(caption),
        }


def _hand_is_valid(hand: np.ndarray) -> np.ndarray:
    finite = np.isfinite(hand).all(axis=(1, 2))
    span = np.max(np.linalg.norm(hand[:, 1:] - hand[:, :1], axis=-1), axis=1)
    return finite & (span > MIN_HAND_SPAN_M)


def _valid_motion_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    keypoints = arrays["keypoints"]
    body_quats = arrays["body_quats"]
    root = arrays["root"]
    finite = (
        np.isfinite(keypoints[:, SOURCE_BODY_JOINTS]).all(axis=(1, 2))
        & np.isfinite(body_quats).all(axis=(1, 2))
        & np.isfinite(root).all(axis=1)
        & _hand_is_valid(arrays["left_hand"])
        & _hand_is_valid(arrays["right_hand"])
    )
    root_norm = np.linalg.norm(root[:, :4], axis=1)
    body_norm = np.linalg.norm(body_quats, axis=-1)
    normalized = (
        (np.abs(root_norm - 1.0) < 0.05)
        & (np.abs(body_norm - 1.0) < 0.05).all(axis=1)
    )
    return finite & normalized


def _segment_ranges(
    valid: np.ndarray,
    timestamps_ns: np.ndarray,
    *,
    min_source_frames: int,
    max_gap_factor: float,
) -> tuple[tuple[Segment, ...], int]:
    deltas = np.diff(timestamps_ns)
    median_delta = float(np.median(deltas))
    gap_after = deltas > median_delta * max_gap_factor
    segments: list[Segment] = []
    dropped_short_frames = 0
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
        else:
            dropped_short_frames += end - start
        start = end
    return tuple(segments), dropped_short_frames


def _preflight_episode(
    root: Path,
    source: Path,
    *,
    min_source_frames: int,
    max_gap_factor: float,
) -> Episode:
    arrays = _read_motion_arrays(source)
    timestamps = arrays["timestamps_ns"]
    valid = _valid_motion_mask(arrays)
    segments, dropped = _segment_ranges(
        valid,
        timestamps,
        min_source_frames=min_source_frames,
        max_gap_factor=max_gap_factor,
    )
    if not segments:
        raise ValueError("no continuous segment has valid body, root, and both-hand tracking")
    median_delta = float(np.median(np.diff(timestamps)))
    return Episode(
        source=source,
        relative_source=source.relative_to(root).as_posix(),
        stable_base=_stable_base(root, source),
        source_frames=len(timestamps),
        source_hz=1e9 / median_delta,
        main_task=str(arrays["caption"]),
        segments=segments,
        valid_frames=int(valid.sum()),
        dropped_short_frames=dropped,
    )


def preflight(
    root: Path,
    *,
    max_episodes: int | None,
    min_source_frames: int,
    max_gap_factor: float,
) -> Preflight:
    annotations = _discover_annotations(root)
    episodes: list[Episode] = []
    exclusions: list[dict[str, str]] = []
    scanned: list[Path] = []
    for source in annotations:
        scanned.append(source)
        try:
            episode = _preflight_episode(
                root,
                source,
                min_source_frames=min_source_frames,
                max_gap_factor=max_gap_factor,
            )
        except (OSError, ValueError, KeyError) as exc:
            exclusions.append(
                {"source": source.relative_to(root).as_posix(), "reason": str(exc)}
            )
            continue
        episodes.append(episode)
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError("preflight found no eligible Xperience episodes")
    return Preflight(
        episodes=tuple(episodes),
        exclusions=tuple(exclusions),
        scanned_sources=tuple(scanned),
        inventory_fingerprint=_inventory_fingerprint(root, scanned),
    )


def _target_seconds(source_seconds: np.ndarray, target_hz: float) -> np.ndarray:
    duration = float(source_seconds[-1])
    step = 1.0 / target_hz
    target = np.arange(0.0, duration + step * 0.5, step, dtype=np.float64)
    target[-1] = min(target[-1], duration)
    return target


def _linear_resample(
    values: np.ndarray,
    source_seconds: np.ndarray,
    target_seconds: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(len(values), -1)
    result = np.empty((len(target_seconds), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        result[:, column] = np.interp(target_seconds, source_seconds, flat[:, column])
    return result.reshape((len(target_seconds), *values.shape[1:])).astype(np.float32)


def _normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("zero quaternion in a preflight-approved segment")
    return quaternions / norms


def _slerp_resample(
    quaternions_wxyz: np.ndarray,
    source_seconds: np.ndarray,
    target_seconds: np.ndarray,
) -> np.ndarray:
    quaternions = _normalize_quaternions(quaternions_wxyz)
    if quaternions.ndim == 2:
        quaternions = quaternions[:, None, :]
        squeeze = True
    else:
        squeeze = False
    result = np.empty((len(target_seconds), quaternions.shape[1], 4), dtype=np.float32)
    for joint in range(quaternions.shape[1]):
        rotations = Rotation.from_quat(quaternions[:, joint], scalar_first=True)
        result[:, joint] = Slerp(source_seconds, rotations)(target_seconds).as_quat(
            scalar_first=True
        )
    return result[:, 0] if squeeze else result


def _hand_distance(hand: np.ndarray) -> np.ndarray:
    tips = hand[:, LEFT_MANO_TIPS]
    return np.min(np.linalg.norm(tips[:, :1] - tips[:, 1:], axis=-1), axis=1)


def _segment_payload(
    arrays: dict[str, np.ndarray],
    segment: Segment,
    *,
    target_hz: float,
    hand_distance_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    selection = slice(segment.start, segment.end)
    timestamps = arrays["timestamps_ns"][selection]
    source_seconds = (timestamps - timestamps[0]).astype(np.float64) * 1e-9
    target_seconds = _target_seconds(source_seconds, target_hz)

    root = arrays["root"][selection]
    keypoints = arrays["keypoints"][selection][:, SOURCE_BODY_JOINTS]
    centered = keypoints - root[:, None, 4:7]
    inverse_root = Rotation.from_quat(root[:, :4], scalar_first=True).inv().as_matrix()
    smpl_joints = np.einsum("tij,tkj->tki", inverse_root, centered)
    smpl_joints = _linear_resample(smpl_joints, source_seconds, target_seconds)

    body_quats = _slerp_resample(
        arrays["body_quats"][selection], source_seconds, target_seconds
    )
    smpl_pose = Rotation.from_quat(
        body_quats.reshape(-1, 4), scalar_first=True
    ).as_rotvec().reshape(len(target_seconds), 21, 3).astype(np.float32)
    joint_pos = _compute_wrist_joint_positions(smpl_pose)
    body_quat_w = _slerp_resample(root[:, :4], source_seconds, target_seconds)

    left_distance = _linear_resample(
        _hand_distance(arrays["left_hand"][selection]), source_seconds, target_seconds
    ).reshape(-1)
    right_distance = _linear_resample(
        _hand_distance(arrays["right_hand"][selection]), source_seconds, target_seconds
    ).reshape(-1)
    left_state = (left_distance <= hand_distance_threshold).astype(np.float32)
    right_state = (right_distance <= hand_distance_threshold).astype(np.float32)

    payload = {
        "smpl_joints": smpl_joints.astype(np.float32, copy=False),
        "joint_pos": joint_pos.astype(np.float32, copy=False),
        "body_quat_w": body_quat_w.astype(np.float32, copy=False),
        "left_hand_state": left_state,
        "right_hand_state": right_state,
    }
    metadata: dict[str, float | int] = {
        "source_start_frame": segment.start,
        "source_end_frame_exclusive": segment.end,
        "source_start_timestamp_ns": int(timestamps[0]),
        "source_end_timestamp_ns": int(timestamps[-1]),
        "source_duration_seconds": float(source_seconds[-1]),
        "target_frames": len(target_seconds),
        "left_closed_fraction": float(left_state.mean()),
        "right_closed_fraction": float(right_state.mean()),
    }
    return payload, metadata


def _slice_payload(payload: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    return {key: value[start:end] for key, value in payload.items()}


def _encode_payload(
    payload: dict[str, np.ndarray],
    session: Any,
    *,
    chunk_frames: int,
    batch_size: int,
) -> np.ndarray:
    frames = int(payload["smpl_joints"].shape[0])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if len(input_meta.shape) != 2:
        raise ValueError(f"SONIC encoder input must be rank 2, got {input_meta.shape}")
    if isinstance(input_meta.shape[-1], int) and input_meta.shape[-1] != ENCODER_INPUT_DIM:
        raise ValueError(f"SONIC encoder expected input dim {ENCODER_INPUT_DIM}, got {input_meta.shape}")
    fixed_batch = input_meta.shape[0] if isinstance(input_meta.shape[0], int) else None
    effective_batch = fixed_batch or batch_size
    outputs: list[np.ndarray] = []
    context = ENCODER_FUTURE_FRAMES - 1
    for start in range(0, frames, chunk_frames):
        primary_end = min(start + chunk_frames, frames)
        context_end = min(primary_end + context, frames)
        chunk_payload = _slice_payload(payload, start, context_end)
        encoder_inputs = build_smpl_encoder_inputs(chunk_payload, base_quat_source="current")
        encoder_inputs = encoder_inputs[: primary_end - start]
        for batch_start in range(0, len(encoder_inputs), effective_batch):
            batch = encoder_inputs[batch_start : batch_start + effective_batch]
            if fixed_batch is not None and len(batch) != fixed_batch:
                if fixed_batch != 1:
                    raise ValueError(
                        f"fixed ONNX batch {fixed_batch} does not divide remaining {len(batch)} frames"
                    )
            encoded = session.run(
                [output_meta.name], {input_meta.name: batch.astype(np.float32, copy=False)}
            )[0]
            outputs.append(np.asarray(encoded, dtype=np.float32).reshape(len(batch), -1))
        del encoder_inputs
    latent = np.concatenate(outputs, axis=0)
    if latent.shape != (frames, LATENT_DIM) or not np.isfinite(latent).all():
        raise ValueError(f"invalid SONIC encoder output {latent.shape}/{latent.dtype}")
    return latent


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_cache(root: Path, rows: list[dict[str, Any]]) -> None:
    parsed_rows = [
        json.loads(line)
        for line in (root / "meta/episode_index.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if parsed_rows != rows:
        raise ValueError("episode_index.jsonl does not round-trip to the frozen manifest rows")
    for row in rows:
        token_path = root / "sonic_tokens" / f"{row['stable_id']}.npy"
        token = np.load(token_path, mmap_mode="r")
        expected = int(row["target_frames"])
        if token.shape != (expected, 66) or token.dtype != np.float32:
            raise ValueError(f"invalid token file {token_path}: {token.shape}/{token.dtype}")
        if not np.isfinite(token).all():
            raise ValueError(f"non-finite token values in {token_path}")
        if not np.isin(token[:, 64:66], (0.0, 1.0)).all():
            raise ValueError(f"non-binary hand state in {token_path}")


def _provider_list(provider: str, gpu_id: int | None = None) -> list[Any] | None:
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        cuda: Any = "CUDAExecutionProvider"
        if gpu_id is not None:
            cuda = ("CUDAExecutionProvider", {"device_id": str(gpu_id)})
        return [cuda, "CPUExecutionProvider"]
    return None


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


def _visible_cuda_devices(provider: str) -> tuple[int, ...]:
    if provider == "cpu":
        return ()

    import onnxruntime as ort
    import torch

    available_providers = ort.get_available_providers()
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if "CUDAExecutionProvider" not in available_providers or count == 0:
        if provider == "cuda":
            raise RuntimeError(
                "--provider cuda requires at least one visible CUDA device and "
                f"CUDAExecutionProvider; devices={count}, providers={available_providers}"
            )
        return ()
    return tuple(range(count))


def _validate_roots(data_root: Path, cache_root: Path, *, pilot: bool) -> None:
    raw = data_root.resolve(strict=True)
    output = cache_root.resolve(strict=False)
    if output == raw or output.is_relative_to(raw):
        raise ValueError(f"cache root must not be the source or inside it: {cache_root}")
    if not pilot and cache_root.parent.resolve() != data_root.parent.resolve():
        raise ValueError("full cache must be a sibling of the Xperience source directory")


def _partition_episodes(
    episodes: tuple[Episode, ...], worker_count: int
) -> tuple[tuple[Episode, ...], ...]:
    if worker_count < 1 or worker_count > len(episodes):
        raise ValueError(f"invalid worker count {worker_count} for {len(episodes)} episodes")
    shards: list[list[Episode]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    weighted = [
        (sum(segment.end - segment.start for segment in episode.segments), index, episode)
        for index, episode in enumerate(episodes)
    ]
    for frames, _, episode in sorted(weighted, key=lambda item: (-item[0], item[1])):
        worker_id = min(range(worker_count), key=lambda index: (loads[index], index))
        shards[worker_id].append(episode)
        loads[worker_id] += frames
    return tuple(tuple(shard) for shard in shards)


def _convert_shard(
    episodes: tuple[Episode, ...],
    *,
    staging: Path,
    encoder_model: Path,
    target_hz: float,
    hand_distance_threshold: float,
    chunk_frames: int,
    batch_size: int,
    gpu_id: int | None,
    min_source_frames: int,
    max_gap_factor: float,
    verify_chunking: bool,
    worker_id: int,
    worker_count: int,
) -> dict[str, Any]:
    provider = "cuda" if gpu_id is not None else "cpu"
    session = create_onnx_session(
        encoder_model,
        providers=_provider_list(provider, gpu_id),
    )
    rows: list[dict[str, Any]] = []
    chunking_max_abs_error: float | None = None
    device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"

    for episode_index, episode in enumerate(episodes, start=1):
        arrays = _read_motion_arrays(episode.source)
        current_segments, _ = _segment_ranges(
            _valid_motion_mask(arrays),
            arrays["timestamps_ns"],
            min_source_frames=min_source_frames,
            max_gap_factor=max_gap_factor,
        )
        if current_segments != episode.segments:
            raise RuntimeError(f"source validity changed after preflight: {episode.relative_source}")
        print(
            f"[xperience worker={worker_id + 1}/{worker_count} device={device}] "
            f"episode {episode_index}/{len(episodes)} {episode.relative_source}: "
            f"{len(episode.segments)} segments",
            flush=True,
        )
        for segment_index, segment in enumerate(episode.segments):
            payload, metadata = _segment_payload(
                arrays,
                segment,
                target_hz=target_hz,
                hand_distance_threshold=hand_distance_threshold,
            )
            latent = _encode_payload(
                payload,
                session,
                chunk_frames=chunk_frames,
                batch_size=batch_size,
            )
            token = np.concatenate(
                [
                    latent,
                    payload["left_hand_state"][:, None],
                    payload["right_hand_state"][:, None],
                ],
                axis=1,
            ).astype(np.float32, copy=False)
            if token.shape != (int(metadata["target_frames"]), 66):
                raise ValueError(f"unexpected token shape {token.shape}")

            stable_id = f"{episode.stable_base}__seg{segment_index:03d}"
            atomic_save_npy(staging / "sonic_tokens" / f"{stable_id}.npy", token)
            row: dict[str, Any] = {
                "stable_id": stable_id,
                "source": episode.relative_source,
                "source_episode": episode.stable_base,
                "source_hz": episode.source_hz,
                "target_hz": target_hz,
                "sonic_encoder_mode": "smpl",
                "motion_semantics": "full_body_reference",
                "lower_body_policy": "measured",
                "hand_state_method": "mano21_thumb_to_fingertips_absolute_v1",
                "hand_distance_threshold_m": hand_distance_threshold,
                "main_task": episode.main_task,
                **metadata,
            }
            rows.append(row)

            if verify_chunking and chunking_max_abs_error is None:
                check_frames = min(200, len(payload["smpl_joints"]))
                check_payload = _slice_payload(payload, 0, check_frames)
                whole = _encode_payload(
                    check_payload,
                    session,
                    chunk_frames=check_frames,
                    batch_size=batch_size,
                )
                chunked = _encode_payload(
                    check_payload,
                    session,
                    chunk_frames=min(37, check_frames),
                    batch_size=batch_size,
                )
                chunking_max_abs_error = float(np.max(np.abs(whole - chunked)))
                chunking_atol = 1.0 / 16.0 if gpu_id is not None else 1e-6
                if not np.allclose(whole, chunked, rtol=1e-5, atol=chunking_atol):
                    raise ValueError(
                        f"chunked encoder output mismatch: max_abs={chunking_max_abs_error}, "
                        f"atol={chunking_atol}"
                    )

    return {
        "worker_id": worker_id,
        "gpu_logical_id": gpu_id,
        "onnx_providers": session.get_providers(),
        "episodes": len(episodes),
        "segments": len(rows),
        "target_frames": sum(int(row["target_frames"]) for row in rows),
        "chunking_max_abs_error": chunking_max_abs_error,
        "rows": rows,
    }


def _convert_shard_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _convert_shard(**kwargs)


def _run_gpu_workers(
    shards: tuple[tuple[Episode, ...], ...],
    gpu_ids: tuple[int, ...],
    common_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    verify_chunking = bool(common_kwargs["verify_chunking"])
    worker_count = len(shards)
    jobs = [
        {
            **common_kwargs,
            "episodes": episodes,
            "gpu_id": gpu_id,
            "worker_id": worker_id,
            "worker_count": worker_count,
            "verify_chunking": verify_chunking and worker_id == 0,
        }
        for worker_id, (episodes, gpu_id) in enumerate(zip(shards, gpu_ids, strict=True))
    ]
    context = mp.get_context("spawn")
    with context.Pool(processes=worker_count, maxtasksperchild=1) as pool:
        results = list(
            pool.imap_unordered(_convert_shard_from_kwargs, jobs, chunksize=1)
        )
    return sorted(results, key=lambda result: int(result["worker_id"]))


def _ordered_rows(
    episodes: tuple[Episode, ...], worker_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    for result in worker_results:
        for row in result["rows"]:
            stable_id = str(row["stable_id"])
            if stable_id in rows_by_id:
                raise ValueError(f"duplicate token stable_id produced: {stable_id}")
            rows_by_id[stable_id] = row
    expected = [
        f"{episode.stable_base}__seg{segment_index:03d}"
        for episode in episodes
        for segment_index in range(len(episode.segments))
    ]
    if set(rows_by_id) != set(expected):
        raise ValueError(
            f"worker outputs do not match preflight: expected={len(expected)}, actual={len(rows_by_id)}"
        )
    return [rows_by_id[stable_id] for stable_id in expected]


def convert(
    preflight_result: Preflight,
    *,
    data_root: Path,
    cache_root: Path,
    encoder_model: Path,
    pilot: bool,
    target_hz: float,
    hand_distance_threshold: float,
    chunk_frames: int,
    batch_size: int,
    provider: str,
    min_source_frames: int,
    max_gap_factor: float,
    verify_chunking: bool,
) -> dict[str, Any]:
    _validate_roots(data_root, cache_root, pilot=pilot)
    if cache_root.exists():
        raise FileExistsError(f"cache root already exists; refusing to overwrite: {cache_root}")
    if not encoder_model.is_file():
        raise FileNotFoundError(f"SONIC encoder model not found: {encoder_model}")

    visible_gpu_ids = _visible_cuda_devices(provider)
    worker_count = min(len(visible_gpu_ids), len(preflight_result.episodes)) if visible_gpu_ids else 1
    assigned_gpu_ids: tuple[int | None, ...] = (
        tuple(visible_gpu_ids[:worker_count]) if visible_gpu_ids else (None,)
    )
    shards = _partition_episodes(preflight_result.episodes, worker_count)
    effective_provider = "cuda" if visible_gpu_ids else "cpu"
    print(
        f"[xperience] provider={effective_provider} visible_cuda_devices={len(visible_gpu_ids)} "
        f"workers={worker_count}",
        flush=True,
    )

    cache_root.parent.mkdir(parents=True, exist_ok=True)
    lock = cache_root.parent / f".{cache_root.name}.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"another conversion or stale lock exists: {lock}") from exc

    staging = Path(tempfile.mkdtemp(prefix=f".{cache_root.name}.staging-", dir=cache_root.parent))
    started = time.monotonic()
    try:
        common_kwargs = {
            "staging": staging,
            "encoder_model": encoder_model,
            "target_hz": target_hz,
            "hand_distance_threshold": hand_distance_threshold,
            "chunk_frames": chunk_frames,
            "batch_size": batch_size,
            "min_source_frames": min_source_frames,
            "max_gap_factor": max_gap_factor,
            "verify_chunking": verify_chunking,
        }
        if worker_count == 1:
            worker_results = [
                _convert_shard(
                    shards[0],
                    **common_kwargs,
                    gpu_id=assigned_gpu_ids[0],
                    worker_id=0,
                    worker_count=1,
                )
            ]
        else:
            worker_results = _run_gpu_workers(
                shards,
                tuple(int(gpu_id) for gpu_id in assigned_gpu_ids if gpu_id is not None),
                common_kwargs,
            )

        rows = _ordered_rows(preflight_result.episodes, worker_results)
        if not rows:
            raise ValueError("conversion produced no token segments")
        if _inventory_fingerprint(data_root, preflight_result.scanned_sources) != preflight_result.inventory_fingerprint:
            raise RuntimeError("Xperience source inventory changed during conversion")

        language_map = {str(row["stable_id"]): str(row["main_task"]) for row in rows}
        chunking_errors = [
            float(result["chunking_max_abs_error"])
            for result in worker_results
            if result["chunking_max_abs_error"] is not None
        ]
        chunking_max_abs_error = max(chunking_errors) if chunking_errors else None
        worker_summaries = [
            {
                key: result[key]
                for key in (
                    "worker_id",
                    "gpu_logical_id",
                    "onnx_providers",
                    "episodes",
                    "segments",
                    "target_frames",
                )
            }
            for result in worker_results
        ]

        schema = {
            "format_version": 1,
            "action": {"dtype": "float32", "shape": ["T", 66], "target_hz": target_hz},
            "sonic": {
                "encoder_mode": "smpl",
                "latent_dimensions": [0, 64],
                "left_hand_dimension": 64,
                "right_hand_dimension": 65,
                "closed_hand_value": 1,
                "open_hand_value": 0,
            },
            "source": {
                "keypoints": "full_body_mocap/keypoints",
                "smpl_joint_indices": SOURCE_BODY_JOINTS.tolist(),
                "body_quats": "full_body_mocap/body_quats local wxyz joints 1..21",
                "root": "full_body_mocap/Ts_world_root [qw,qx,qy,qz,x,y,z]",
                "timestamps": "video/device_timestamp nanoseconds",
                "hand_joints": "hand_mocap/{left,right}_joints_3d MANO-21",
                "mano_tip_indices": LEFT_MANO_TIPS.tolist(),
                "missing_hand_rule": f"maximum wrist-relative span <= {MIN_HAND_SPAN_M} m is invalid",
            },
        }
        _atomic_write_json(staging / "meta/schema.json", schema)
        _atomic_write_json(staging / "meta/language_map.json", language_map)
        _atomic_write_jsonl(staging / "meta/episode_index.jsonl", rows)
        manifest: dict[str, Any] = {
            "format_version": 1,
            "pilot": pilot,
            "data_root": str(data_root),
            "cache_root": str(cache_root),
            "raw_inventory_fingerprint": preflight_result.inventory_fingerprint,
            "encoder_model": str(encoder_model),
            "encoder_model_sha256": _file_sha256(encoder_model),
            "onnx_providers": worker_results[0]["onnx_providers"],
            "execution": {
                "requested_provider": provider,
                "effective_provider": effective_provider,
                "visible_cuda_device_count": len(visible_gpu_ids),
                "worker_count": worker_count,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "workers": worker_summaries,
            },
            "chunking_max_abs_error": chunking_max_abs_error,
            "elapsed_seconds": time.monotonic() - started,
            "counts": {
                "eligible_episodes": len(preflight_result.episodes),
                "excluded_scanned_episodes": len(preflight_result.exclusions),
                "segments": len(rows),
                "target_frames": sum(int(row["target_frames"]) for row in rows),
            },
            "exclusions": list(preflight_result.exclusions),
            "segments": rows,
        }
        _atomic_write_json(staging / "meta/conversion_manifest.json", manifest)
        _validate_cache(staging, rows)
        marker = "_PILOT_SUCCESS" if pilot else "_SUCCESS"
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


def _preflight_summary(result: Preflight) -> dict[str, Any]:
    return {
        "eligible_episodes": len(result.episodes),
        "excluded_scanned_episodes": len(result.exclusions),
        "scanned_episodes": len(result.scanned_sources),
        "segments": sum(len(episode.segments) for episode in result.episodes),
        "valid_source_frames": sum(episode.valid_frames for episode in result.episodes),
        "inventory_fingerprint": result.inventory_fingerprint,
        "episodes": [
            {
                "source": episode.relative_source,
                "source_frames": episode.source_frames,
                "source_hz": episode.source_hz,
                "valid_frames": episode.valid_frames,
                "segments": [[segment.start, segment.end] for segment in episode.segments],
                "main_task": episode.main_task,
            }
            for episode in result.episodes
        ],
        "exclusions": list(result.exclusions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="only inspect source schema; default")
    mode.add_argument("--pilot", action="store_true", help="publish a small cache under /tmp")
    mode.add_argument("--publish-full", action="store_true", help="explicitly process every eligible episode")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--encoder-model", type=Path, default=DEFAULT_ENCODER_MODEL)
    parser.add_argument("--max-episodes", type=int, default=3, help="pilot/preflight eligible episode limit; 0 means all")
    parser.add_argument("--min-source-frames", type=int, default=20)
    parser.add_argument("--max-timestamp-gap-factor", type=float, default=2.0)
    parser.add_argument("--target-hz", type=float, default=TARGET_HZ)
    parser.add_argument("--hand-distance-threshold", type=float, default=0.045)
    parser.add_argument("--chunk-frames", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--provider",
        choices=("cpu", "cuda", "auto"),
        default="auto",
        help="auto/cuda use one worker per visible CUDA GPU; cpu uses one process",
    )
    parser.add_argument(
        "--verify-chunking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.pilot or args.publish_full) and args.provider != "cpu":
        _ensure_cuda_library_path()
    if args.max_episodes < 0:
        raise ValueError("--max-episodes must be >= 0")
    if args.min_source_frames < 2:
        raise ValueError("--min-source-frames must be >= 2")
    if args.target_hz <= 0 or args.max_timestamp_gap_factor <= 1:
        raise ValueError("target Hz must be positive and max gap factor must be > 1")
    if args.chunk_frames <= 0 or args.batch_size <= 0:
        raise ValueError("chunk frames and batch size must be positive")

    limit = None if args.publish_full or args.max_episodes == 0 else args.max_episodes
    result = preflight(
        args.data_root,
        max_episodes=limit,
        min_source_frames=args.min_source_frames,
        max_gap_factor=args.max_timestamp_gap_factor,
    )
    print(json.dumps(_preflight_summary(result), indent=2, sort_keys=True))
    if not args.pilot and not args.publish_full:
        return

    if args.cache_root is None:
        if args.pilot:
            args.cache_root = Path(tempfile.gettempdir()) / f"huge-xperience-sonic-pilot-{os.getpid()}"
        else:
            args.cache_root = DEFAULT_CACHE_ROOT
    manifest = convert(
        result,
        data_root=args.data_root,
        cache_root=args.cache_root,
        encoder_model=args.encoder_model,
        pilot=args.pilot,
        target_hz=args.target_hz,
        hand_distance_threshold=args.hand_distance_threshold,
        chunk_frames=args.chunk_frames,
        batch_size=args.batch_size,
        provider=args.provider,
        min_source_frames=args.min_source_frames,
        max_gap_factor=args.max_timestamp_gap_factor,
        verify_chunking=args.verify_chunking,
    )
    counts = manifest["counts"]
    print(
        f"Xperience {'pilot' if args.pilot else 'cache'} ready: "
        f"{counts['eligible_episodes']} episodes / {counts['segments']} segments / "
        f"{counts['target_frames']} frames at {args.cache_root}"
    )


if __name__ == "__main__":
    main()

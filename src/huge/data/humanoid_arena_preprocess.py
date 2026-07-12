"""Strict preprocessing for a unified HumanoidArena Stage 3 cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import cv2
import numpy as np

from huge.data.preprocess import atomic_save_npy, projected_gravity_from_quat_wxyz
from huge.data.proprio_stats import write_proprio_stats
from huge.data.state_normalizer import StateNormalizer


SUPPORTED_SCHEMAS = {"sonic_episode_v2", "sonic_episode_v3"}
TASK_PROMPTS = {
    "HOI_double_desk": "move the object from one desk to the other",
    "HOI_football_v2": "kick the football",
    "HOI_grap_cup": "pick up the cup",
    "HOI_pp_box": "pick up the box and place it",
    "HSI_boxing": "punch the boxing bag",
    "HSI_open_door": "open the door",
    "HSI_sit_sofa": "sit down on the sofa",
}

# ``robot_qpos_before_decimation`` uses HumanoidArena's interleaved SONIC order.
# The output uses Huge's grouped left-leg/right-leg/waist/left-arm/right-arm order.
CANONICAL_NAMES_29 = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
GROUP_JOINTS = {
    "left_leg": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ],
    "right_leg": [
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ],
    "waist": ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    "left_arm": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ],
}
GROUP_ORDER = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")
GROUP_IDX = {group: [CANONICAL_NAMES_29.index(name) for name in names] for group, names in GROUP_JOINTS.items()}


@dataclass(frozen=True)
class SourceEpisode:
    source: Path
    relative_source: str
    task: str
    collector: str
    episode_id: str
    source_episode_id: str
    save_timestamp_us: str
    stem: str
    prompt: str
    schema: str
    frame_count: int
    video_mode: Literal["sidecar", "embedded_rgb"]
    sidecar_video: Path | None
    fps: float


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data.files:
        return None
    value = data[key]
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got {value.shape}")
    item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _normalize_component(value: str, *, label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError(f"empty normalized {label} from {value!r}")
    return normalized


def _collector(relative: Path) -> str:
    parts = relative.parts
    try:
        sonic_index = next(index for index, part in enumerate(parts[:-1]) if part.lower() == "sonic")
    except StopIteration as exc:
        raise ValueError(f"{relative}: path does not contain sonic/SONIC") from exc
    if sonic_index + 1 >= len(parts) - 1:
        raise ValueError(f"{relative}: missing collector directory after sonic/SONIC")
    return parts[sonic_index + 1]


def _array_header(npz_path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    member = f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"{npz_path}: missing {key}")
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"{npz_path}: unsupported NPY header version {version} for {key}")
    return tuple(int(value) for value in shape), np.dtype(dtype)


def _fps(data: np.lib.npyio.NpzFile) -> float:
    if "vision_rgb_video_fps" in data.files:
        value = float(np.asarray(data["vision_rgb_video_fps"]).reshape(-1)[0])
        if np.isfinite(value) and value > 0:
            return value
    if "meta_control_dt" in data.files:
        control_dt = float(np.asarray(data["meta_control_dt"]).reshape(-1)[0])
        if np.isfinite(control_dt) and control_dt > 0:
            return float(np.clip(1.0 / control_dt, 1.0, 240.0))
    return 50.0


def _validate_array(path: Path, name: str, array: np.ndarray, shape: tuple[int, int]) -> None:
    if array.ndim != 2 or array.shape[1] != shape[1]:
        raise ValueError(f"{path}: {name} expected [T,{shape[1]}], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: nonfinite values in {name}")


def _preflight_episode(npz_path: Path, raw_root: Path) -> SourceEpisode:
    relative = npz_path.relative_to(raw_root)
    task = relative.parts[0]
    if task not in TASK_PROMPTS:
        raise ValueError(f"{npz_path}: unknown HumanoidArena task {task!r}")
    collector = _collector(relative)

    with np.load(npz_path, allow_pickle=True) as data:
        schema = _scalar(data, "schema_version")
        if schema not in SUPPORTED_SCHEMAS:
            raise ValueError(f"{npz_path}: unsupported schema {schema!r}")

        required = {
            "encoder_latent": 64,
            "robot_qpos_before_decimation": 29,
            "robot_root_orientation": 4,
        }
        arrays: dict[str, np.ndarray] = {}
        for name, width in required.items():
            if name not in data.files:
                raise ValueError(f"{npz_path}: missing {name}")
            array = np.asarray(data[name])
            _validate_array(npz_path, name, array, (0, width))
            arrays[name] = array

        frame_count = int(arrays["encoder_latent"].shape[0])
        if frame_count <= 0:
            raise ValueError(f"{npz_path}: episode has no frames")
        for name, array in arrays.items():
            if array.shape[0] != frame_count:
                raise ValueError(f"{npz_path}: length mismatch {name}={array.shape[0]} vs {frame_count}")

        for name in ("episode_step", "pico_left_grip_binary", "pico_right_grip_binary"):
            if name not in data.files:
                raise ValueError(f"{npz_path}: missing {name}")
            array = np.asarray(data[name])
            if array.ndim != 1 or array.shape[0] != frame_count:
                raise ValueError(f"{npz_path}: length mismatch {name}={array.shape} vs ({frame_count},)")
            if not np.isfinite(array).all():
                raise ValueError(f"{npz_path}: nonfinite values in {name}")
            arrays[name] = array

        steps = arrays["episode_step"]
        if frame_count > 1 and not np.all(np.diff(steps) == 1):
            raise ValueError(f"{npz_path}: episode_step not contiguous")
        for name in ("pico_left_grip_binary", "pico_right_grip_binary"):
            if not np.isin(arrays[name], (0, 1)).all():
                raise ValueError(f"{npz_path}: {name} must be binary")
        quaternion_norms = np.linalg.norm(arrays["robot_root_orientation"].astype(np.float64), axis=1)
        if not np.isfinite(quaternion_norms).all() or np.any(quaternion_norms <= 1e-8):
            raise ValueError(f"{npz_path}: invalid zero/nonfinite root quaternion")

        source_episode_id = (_scalar(data, "episode_id") or "").strip()
        save_timestamp_us = (_scalar(data, "save_timestamp_us") or "").strip()
        source_id = save_timestamp_us or source_episode_id
        if not source_id:
            raise ValueError(f"{npz_path}: missing episode_id and save_timestamp_us")
        stable_id = "__".join(
            (
                _normalize_component(task, label="task"),
                _normalize_component(collector, label="collector"),
                _normalize_component(source_id, label="source id"),
            )
        )
        fps = _fps(data)

        if schema == "sonic_episode_v3":
            video_value = (_scalar(data, "vision_rgb_video_path") or "").strip()
            if not video_value:
                raise ValueError(f"{npz_path}: missing vision_rgb_video_path")
            video_path = (npz_path.parent / video_value).resolve()
            if not video_path.is_file():
                raise ValueError(f"{npz_path}: video missing: {video_path}")
            if not video_path.is_relative_to(raw_root.resolve()):
                raise ValueError(f"{npz_path}: video escapes raw root: {video_path}")
            video_mode: Literal["sidecar", "embedded_rgb"] = "sidecar"
        else:
            rgb_shape, rgb_dtype = _array_header(npz_path, "vision_rgb")
            if len(rgb_shape) != 4 or rgb_shape[0] != frame_count or rgb_shape[-1] != 3:
                raise ValueError(f"{npz_path}: vision_rgb expected [T,H,W,3], got {rgb_shape}")
            if rgb_dtype != np.dtype(np.uint8):
                raise ValueError(f"{npz_path}: vision_rgb expected uint8, got {rgb_dtype}")
            if "vision_frame_indices" not in data.files:
                raise ValueError(f"{npz_path}: missing vision_frame_indices")
            frame_indices = np.asarray(data["vision_frame_indices"])
            if frame_indices.ndim != 1 or frame_indices.shape[0] != frame_count:
                raise ValueError(
                    f"{npz_path}: length mismatch vision_frame_indices={frame_indices.shape} vs ({frame_count},)"
                )
            video_path = None
            video_mode = "embedded_rgb"

    return SourceEpisode(
        source=npz_path,
        relative_source=relative.as_posix(),
        task=task,
        collector=collector,
        episode_id=stable_id,
        source_episode_id=source_episode_id,
        save_timestamp_us=save_timestamp_us,
        stem=f"episode_{stable_id}",
        prompt=TASK_PROMPTS[task],
        schema=schema,
        frame_count=frame_count,
        video_mode=video_mode,
        sidecar_video=video_path,
        fps=fps,
    )


def _inventory_fingerprint(raw_root: Path, episodes: list[SourceEpisode]) -> str:
    inventory: list[Path] = [episode.source for episode in episodes]
    inventory.extend(episode.sidecar_video for episode in episodes if episode.sidecar_video is not None)
    rows: list[str] = []
    resolved_root = raw_root.resolve()
    for path in sorted(set(inventory)):
        stat = path.stat()
        rows.append(f"{path.resolve().relative_to(resolved_root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def discover_and_preflight(raw_root: Path) -> tuple[list[SourceEpisode], str]:
    """Discover and strictly validate every raw HumanoidArena episode."""
    root = Path(raw_root)
    if not root.is_dir():
        raise ValueError(f"HumanoidArena raw root does not exist: {root}")
    paths = sorted(path for path in root.rglob("*.npz") if path.is_file())
    if not paths:
        raise ValueError(f"no HumanoidArena npz files under {root}")

    episodes = [_preflight_episode(path, root) for path in paths]
    seen: dict[str, Path] = {}
    for episode in episodes:
        prior = seen.get(episode.episode_id)
        if prior is not None:
            raise ValueError(
                f"duplicate stable episode id {episode.episode_id!r}: {prior} and {episode.source}"
            )
        seen[episode.episode_id] = episode.source
    return episodes, _inventory_fingerprint(root, episodes)


def reorder_qpos_grouped(qpos: np.ndarray) -> np.ndarray:
    """Convert HumanoidArena's interleaved 29 joints to Huge's grouped order."""
    array = np.asarray(qpos, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 29:
        raise ValueError(f"qpos expected [T,29], got {array.shape}")
    return np.concatenate([array[:, GROUP_IDX[group]] for group in GROUP_ORDER], axis=-1)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stats_source_fingerprint(sources: dict[str, Path]) -> str:
    rows: list[str] = []
    for tag, source_root in sources.items():
        root = Path(source_root)
        paths = sorted(root.rglob("episode_*.npy")) if root.is_dir() else []
        paths = [path for path in paths if path.is_file()]
        if not paths:
            raise ValueError(f"{tag}: no episode_*.npy files under {root}")
        resolved_root = root.resolve()
        for path in paths:
            stat = path.stat()
            rows.append(
                f"{tag}\0{path.resolve().relative_to(resolved_root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}"
            )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _write_embedded_video(npz_path: Path, destination: Path, fps: float) -> None:
    with np.load(npz_path, allow_pickle=True) as data:
        frames = np.asarray(data["vision_rgb"])
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise ValueError(f"{npz_path}: invalid embedded vision_rgb {frames.shape}/{frames.dtype}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.{os.getpid()}.mp4")
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to open MP4 writer for {destination}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(temporary))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"encoded MP4 is unreadable: {temporary}")
        encoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if encoded_frames != int(frames.shape[0]):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"encoded MP4 frame count mismatch for {npz_path}: {encoded_frames} != {frames.shape[0]}"
        )
    os.replace(temporary, destination)


def _convert_episode(episode: SourceEpisode, out_root: Path) -> None:
    with np.load(episode.source, allow_pickle=True) as data:
        encoder_latent = np.asarray(data["encoder_latent"], dtype=np.float32)
        left_grip = np.asarray(data["pico_left_grip_binary"], dtype=np.float32).reshape(-1, 1)
        right_grip = np.asarray(data["pico_right_grip_binary"], dtype=np.float32).reshape(-1, 1)
        qpos = reorder_qpos_grouped(data["robot_qpos_before_decimation"])
        gravity = projected_gravity_from_quat_wxyz(
            np.asarray(data["robot_root_orientation"], dtype=np.float32)
        )

    token = np.concatenate((encoder_latent, left_grip, right_grip), axis=-1).astype(np.float32, copy=False)
    proprio = np.concatenate((qpos, left_grip, right_grip, gravity), axis=-1).astype(np.float32, copy=False)
    atomic_save_npy(out_root / "sonic_tokens" / f"{episode.stem}.npy", token)
    atomic_save_npy(out_root / "g1_proprio" / f"{episode.stem}.npy", proprio)

    video = out_root / "videos/chunk-000/observation.images.ego_view" / f"{episode.stem}.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    if episode.video_mode == "sidecar":
        assert episode.sidecar_video is not None
        relative_target = os.path.relpath(episode.sidecar_video.resolve(), start=video.parent.resolve())
        video.symlink_to(relative_target)
    else:
        _write_embedded_video(episode.source, video, episode.fps)


def _manifest_rows(episodes: list[SourceEpisode]) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": episode.episode_id,
            "source_episode_id": episode.source_episode_id,
            "save_timestamp_us": episode.save_timestamp_us,
            "stem": episode.stem,
            "task": episode.task,
            "collector": episode.collector,
            "source": episode.relative_source,
            "schema": episode.schema,
            "n_frames": episode.frame_count,
            "video_mode": episode.video_mode,
            "prompt": episode.prompt,
        }
        for episode in episodes
    ]


def _task_counts(episodes: list[SourceEpisode]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for episode in episodes:
        counts = result.setdefault(episode.task, {"n_episodes": 0, "n_frames": 0})
        counts["n_episodes"] += 1
        counts["n_frames"] += episode.frame_count
    return result


def _validate_published_cache(root: Path, manifest: dict[str, Any]) -> None:
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"invalid or empty conversion manifest under {root}")
    language = json.loads((root / "meta/language_map.json").read_text(encoding="utf-8"))
    for row in rows:
        episode_id = row["episode_id"]
        stem = row["stem"]
        token = np.load(root / "sonic_tokens" / f"{stem}.npy", mmap_mode="r")
        proprio = np.load(root / "g1_proprio" / f"{stem}.npy", mmap_mode="r")
        expected_frames = int(row["n_frames"])
        if token.shape != (expected_frames, 66) or token.dtype != np.float32:
            raise ValueError(f"invalid token cache for {stem}: {token.shape}/{token.dtype}")
        if proprio.shape != (expected_frames, 34) or proprio.dtype != np.float32:
            raise ValueError(f"invalid proprio cache for {stem}: {proprio.shape}/{proprio.dtype}")
        if language.get(episode_id) != row["prompt"]:
            raise ValueError(f"invalid language mapping for {episode_id}")
        video = root / "videos/chunk-000/observation.images.ego_view" / f"{stem}.mp4"
        if not video.exists():
            raise ValueError(f"missing published video for {stem}")
    StateNormalizer.from_json(root / "meta/proprio_stats_union_sonic.json")


def _load_reusable_manifest(
    out_root: Path,
    *,
    raw_fingerprint: str,
    stats_fingerprint: str,
    rebuild: bool,
) -> dict[str, Any] | None:
    manifest_path = out_root / "meta/conversion_manifest.json"
    complete = (out_root / "_SUCCESS").is_file() and manifest_path.is_file()
    if not complete:
        if out_root.exists() and not rebuild:
            raise ValueError(f"incomplete output exists at {out_root}; rerun with --rebuild")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unchanged = (
        manifest.get("raw_inventory_fingerprint") == raw_fingerprint
        and manifest.get("stats_source_fingerprint") == stats_fingerprint
    )
    if unchanged and not rebuild:
        _validate_published_cache(out_root, manifest)
        return manifest
    if not unchanged and not rebuild:
        raise ValueError("source inventory changed since the published HumanoidArena cache; rerun with --rebuild")
    return None


def _publish_staging(staging: Path, out_root: Path) -> None:
    backup = out_root.with_name(f".{out_root.name}.backup.{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    moved_old = False
    try:
        if out_root.exists():
            os.replace(out_root, backup)
            moved_old = True
        os.replace(staging, out_root)
    except Exception:
        if moved_old and not out_root.exists():
            os.replace(backup, out_root)
        raise
    finally:
        if backup.exists() and out_root.exists():
            shutil.rmtree(backup)


def preprocess_humanoid_arena(
    *,
    raw_root: str | Path,
    out_root: str | Path,
    he_cache_root: str | Path,
    wbt_source_root: str | Path,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Build one atomic HumanoidArena cache and HE+WBT+HA union stats."""
    raw = Path(raw_root)
    out = Path(out_root)
    he = Path(he_cache_root)
    wbt = Path(wbt_source_root)
    raw_resolved = raw.resolve()
    out_resolved = out.resolve()
    if out_resolved == raw_resolved or out_resolved.is_relative_to(raw_resolved):
        raise ValueError(f"output root must not be the raw root or inside it: {out}")

    episodes, raw_fingerprint = discover_and_preflight(raw)
    base_stats_sources = {"HE": he / "g1_proprio", "WBT": wbt / "g1_proprio"}
    stats_fingerprint = _stats_source_fingerprint(base_stats_sources)
    reusable = _load_reusable_manifest(
        out,
        raw_fingerprint=raw_fingerprint,
        stats_fingerprint=stats_fingerprint,
        rebuild=rebuild,
    )
    if reusable is not None:
        return reusable

    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        for episode in episodes:
            _convert_episode(episode, staging)

        stats = write_proprio_stats(
            {
                **base_stats_sources,
                "HUMANOID_ARENA": staging / "g1_proprio",
            },
            staging / "meta/proprio_stats_union_sonic.json",
        )
        language_map = {episode.episode_id: episode.prompt for episode in episodes}
        _atomic_write_json(staging / "meta/language_map.json", language_map)
        manifest: dict[str, Any] = {
            "format_version": 1,
            "raw_root": str(raw),
            "out_root": str(out),
            "raw_inventory_fingerprint": raw_fingerprint,
            "stats_source_fingerprint": stats_fingerprint,
            "counts": {
                "episodes": len(episodes),
                "frames": sum(episode.frame_count for episode in episodes),
            },
            "task_counts": _task_counts(episodes),
            "stats_sources": stats["_meta"]["sources"],
            "episodes": _manifest_rows(episodes),
        }
        _atomic_write_json(staging / "meta/conversion_manifest.json", manifest)

        current_raw_fingerprint = _inventory_fingerprint(raw, episodes)
        if current_raw_fingerprint != raw_fingerprint:
            raise RuntimeError("raw HumanoidArena inventory changed during preprocessing")
        _validate_published_cache(staging, manifest)
        (staging / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        _publish_staging(staging, out)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)

#!/usr/bin/env python3
"""Huge-owned, read-only HumanoidArena SONIC evaluation orchestration.

HumanoidArena is treated as an immutable runtime dependency.  Static dry-run
mode deliberately stops before CUDA discovery, Kit map access, sockets, locks,
model loading, or SimulationApp creation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import sysconfig
import time
from typing import Any, Iterable, Mapping, Sequence
import urllib.error
import urllib.request

import yaml

from huge.inference.humanoidarena_http_server import (
    CheckpointContract,
    CheckpointValidationError,
    MAPPING_VERSION,
    PROTOCOL_VERSION,
    resolve_checkpoint_contract,
)


STATIC_PLAN_SCHEMA_VERSION = "huge_humanoidarena_static_plan_v1"
CONTRACT_SCHEMA_VERSION = "huge_humanoidarena_contract_v1"
MANIFEST_SCHEMA_VERSION = "huge_humanoidarena_attempt_v1"
COMPLETION_SCHEMA_VERSION = "huge_humanoidarena_completion_v1"
EPISODE_SCHEMA_VERSION = "huge_humanoidarena_episode_v1"
SUMMARY_SCHEMA_VERSION = "huge_humanoidarena_summary_v1"
KIT_MAP_SCHEMA_VERSION = "huge_kit_gpu_map_v1"
EPISODE_SEED_FORMULA = "sha256(task_name|group_seed|repeat_idx)_positive_int32"
UPSTREAM_SIM_SHA256 = "2c39e4eaa9024bc19020903c5cb58249f8a9a1c5048cbf29873b50db2b7047ea"
SONIC_ENCODER_SHA256 = "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"
SONIC_DECODER_SHA256 = "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"

EXIT_CONFIG = 2
EXIT_GPU = 3
EXIT_SERVER = 4
EXIT_SIM = 5
EXIT_INTERRUPTED = 130
DEFAULT_WORKERS_PER_GPU = 1
MAX_WORKERS_PER_GPU = 2

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_WORKER_RE = re.compile(r"^w[0-9]{2}$")
_SHM_RE = re.compile(r"^huge_ha_([a-z0-9]{8})_(w[0-9]{2})_([a-f0-9]{16,32})$")
_LEGAL_TASK_OUTCOMES = {"success", "fall", "timeout", "max_steps", "sim_stopped"}
_CANCEL_REQUESTED = False


class ConfigError(RuntimeError):
    pass


class GpuError(RuntimeError):
    pass


class ServerError(RuntimeError):
    pass


class SimulationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskSpec:
    cli_task: str
    checkpoint_task: str
    isaac_task: str
    yaml_stem: str
    instruction: str
    default_max_steps: int
    default_record_video_every_n: int
    memory_limit_percent: int | None = None


TASK_SPECS: dict[str, TaskSpec] = {
    "boxing": TaskSpec(
        "boxing", "HSI_boxing", "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody",
        "boxing", "punch the boxing bag", 900, 1,
    ),
    "doubledesk": TaskSpec(
        "doubledesk", "HOI_double_desk",
        "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody", "doubledesk",
        "move the object from one desk to the other", 2000, 10, 95,
    ),
    "football": TaskSpec(
        "football", "HOI_football_v2", "Isaac-Move-Football-Single-G129-Dex3-Wholebody",
        "football_single", "kick the football", 2000, 10,
    ),
    "open_door": TaskSpec(
        "open_door", "HSI_open_door", "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
        "open_door", "open the door", 1800, 1,
    ),
    "pp_box": TaskSpec(
        "pp_box", "HOI_pp_box", "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
        "pp_box", "pick up the box and place it", 1450, 1,
    ),
    "sit_sofa": TaskSpec(
        "sit_sofa", "HSI_sit_sofa", "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody",
        "sit_sofa", "sit down on the sofa", 2000, 1,
    ),
    "vision_navi": TaskSpec(
        "vision_navi", "HSI_vision_navi",
        "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
        "vision_navi", "navigate through the warehouse", 1800, 1,
    ),
}


@dataclass(frozen=True)
class EpisodeJob:
    task_cli: str
    isaac_task_name: str
    test_mode: str
    seed: int
    repeat_idx: int
    episode_seed: int
    episode_index: int
    instruction: str
    env_config_yaml_sha256: str
    result_identity: str


@dataclass(frozen=True)
class GpuIdentity:
    parent_logical_ordinal: int
    child_cuda_visible_token: str
    cuda_uuid: str
    pci_bus_id: str
    device_name: str
    total_memory_bytes: int
    compute_capability: str
    free_memory_bytes: int | None = None


@dataclass(frozen=True)
class GpuBinding:
    gpu: GpuIdentity
    renderer_ordinal: int
    worker_slot: int = 0


@dataclass
class ChildProcess:
    role: str
    worker_id: str
    process: subprocess.Popen[bytes]
    log_stream: Any


@dataclass
class OwnedShm:
    name: str
    path: Path
    uid: int
    device: int | None = None
    inode: int | None = None
    mode: int | None = None

    def observe(self) -> None:
        if self.inode is not None or not os.path.lexists(self.path):
            return
        info = self.path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_uid != self.uid:
            raise SimulationError(f"owned SHM appeared with unsafe identity: {self.path}")
        self.device, self.inode, self.mode = info.st_dev, info.st_ino, info.st_mode

    def cleanup(self) -> bool:
        if not os.path.lexists(self.path):
            return True
        if self.inode is None:
            return False
        info = self.path.lstat()
        safe = (
            not stat.S_ISLNK(info.st_mode)
            and stat.S_ISREG(info.st_mode)
            and info.st_uid == self.uid
            and info.st_dev == self.device
            and info.st_ino == self.inode
            and info.st_mode == self.mode
            and self.path.name == self.name
            and _SHM_RE.fullmatch(self.name) is not None
        )
        if not safe:
            return False
        self.path.unlink()
        return True


UPSTREAM_COMPATIBILITY_BASELINE: dict[str, str] = {
    "isaaclab_twist2_g1/script/eval_scripts/sonic/sim_eval_vla.py": UPSTREAM_SIM_SHA256,
    "isaaclab_twist2_g1/script/eval_scripts/sonic/eval_vla_suite.py": "7704b538c5eacc89cab408e4df7822faf5fee0fc99e11070f1be6a39bf2a00e4",
    "isaaclab_twist2_g1/script/eval_scripts/sonic/eval_vla_suite_parallel.py": "6e897d8f95b55c416aa1e094a9aa845ed8e2dcba803edff199928f56c434d5e4",
    "isaaclab_twist2_g1/action_provider/lerobot_vla_http_client.py": "e0232c4b17bd006a6fa0df22ba020c8231b13c94fc2ce459ce00d96996fc68c2",
    "isaaclab_twist2_g1/action_provider/action_provider_sonic.py": "701b75f0effec0c28c215a813a10d70db0a33dbf01000c7f1c142d6a8a60568c",
    "isaaclab_twist2_g1/action_provider/vla_robot_current_local_runtime_v3.py": "4037db768fd2e3c885997c1c253b3a9f04976af273827c65742783a2ecba9bf2",
    "isaaclab_twist2_g1/action_provider/vla_smpl_runtime.py": "9c592194658ac759ec662fee3d6f583be22daad6545eeb998429ab5d6f7634e7",
    "isaaclab_twist2_g1/task_runtime_profiles.py": "a9d9893c1f01f22484f1d187cb6a6541e61db85cc0b1356da546497ea241e8c4",
    "isaaclab_twist2_g1/tasks/common_env_config/loader.py": "a89df4e50c74b8c2581257fdae221264a3e0e2a53c4f4e9bd6032e81c8af834a",
    "isaaclab_twist2_g1/tasks/common_runtime/__init__.py": "320211698c445d1cf782dbb4e5f244dada1f2a576427be8cc30586ff202f3fed",
    "isaaclab_twist2_g1/tasks/common_runtime/env_runtime_hooks.py": "12e7a38f245da6ccb12fbc6f710be60eee01622b8a8b044e4b95bf8c37be244e",
    "isaaclab_twist2_g1/common_env_objects.py": "61d36badc6dfee154d91b135f662d9c8c74cb354fe37be3197d94af734f85e77",
    "external/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py": "b559d9cec89193eecb8263655335445a301996bbd52f2c45ad3bcfb41bd83c4d",
    "isaaclab_twist2_g1/script/common/runtime_paths.sh": "f41b40485c91e2ec0e48731ad778ae5963f75ebba3173a5ed94b9e280792c8dd",
    "isaaclab_twist2_g1/image_server/shared_memory_utils.py": "1e3d8cf013dc10dd79dcf6a0f5f6a65b75bb46472f3ea1dd3cab26ab4d6be37c",
}

WRAPPER_BASELINE: dict[str, str] = {
    "batch_test_scripts/task/batch_test_boxing.sh": "5c7eedc3bce45bf98cb06348679d9b06d22e20c14c48c5fe3c5ee98b1ff03116",
    "batch_test_scripts/task/batch_test_doubledesk.sh": "7660121c078b55696585234bfe9606a26adb9ab20f674b5a720ffd8e9af01b0d",
    "batch_test_scripts/task/batch_test_football.sh": "6901fe37bad5813bec4d4abc65e0209aa39170d2765e38c05923d0522f9b92b1",
    "batch_test_scripts/task/batch_test_open_door.sh": "f02c68342d66ca30d485afd2e3be9290f71e5c1724a0de456b79a1446d5985a3",
    "batch_test_scripts/task/batch_test_pp_box.sh": "cdc7d203f1ef9f2b4d04ba888b6130e9505ec62652d141d21a5316db90f60745",
    "batch_test_scripts/task/batch_test_sit_sofa.sh": "41a3d96fa614a2ebc73429549944a8944ec8c13b7c35c09d5dfd068bc82239d0",
    "batch_test_scripts/task/batch_test_vision_navi.sh": "931a7cf2a517cefcf7038b045b0d74def3fde8ce9167c2fc8ebb966c096bb60c",
    "script/eval_scripts/sonic/run_vla_eval_parallel.sh": "b8210ce2343d19f12babc66b779a49b98b87fcd91f6a0f5526c4cdf4e6c9c0f0",
    "script/eval_scripts/sonic/HSI_boxing_run_vla_eval_parallel.sh": "8f492f52b911dd6b294f7abc7bd5e2d6fd90edb14fa970bc4956b5f88df47cfb",
    "script/eval_scripts/sonic/HSI_open_door_run_vla_eval_parallel.sh": "0d017dbc81db8c587174dffe025ae2286d7e815e6416b76026392b69c45b70e5",
    "script/eval_scripts/sonic/HOI_pp_box_run_vla_eval_parallel.sh": "2f45eca762a4529ca7437f87d4de42e6cc40337dbf14629230845d47050c1e24",
    "script/eval_scripts/sonic/HSI_sit_sofa_run_vla_eval_parallel.sh": "bdf05bf672f756abec24ed80670f9fdc411d10d34265dbc089536f02505898ef",
    "script/eval_scripts/sonic/HSI_vision_navi_run_vla_eval_parallel.sh": "93341fa1fac145607b9ea12a719338b1c2d5ee49b15ffc59077c3cb3e98858e4",
    "script/eval_scripts/sonic/HOI_double_desk_run_vla_eval_parallel.sh": "9bba65ab42b46d0af82c6d4ab7ddb062c00881330f74ae06b573ed892432f08d",
    "script/eval_scripts/sonic/HOI_football_run_vla_eval_parallel.sh": "d0d2703211d8239598d503bad78b91461e913fb585d2f615bbf776637ebd2e77",
}


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("wb") as stream:
            stream.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{label} must be a JSON object: {path}")
    return payload


def _parse_bool(name: str, raw: str) -> bool:
    if raw not in {"0", "1"}:
        raise ConfigError(f"{name} must be 0 or 1")
    return raw == "1"


def _parse_int(name: str, raw: str, *, minimum: int, maximum: int = 2**31 - 1) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _parse_float(name: str, raw: str, *, minimum: float) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not value >= minimum or not value < float("inf"):
        raise ConfigError(f"{name} must be finite and >= {minimum}")
    return value


def parse_runtime_policy(task: TaskSpec, environ: Mapping[str, str]) -> dict[str, Any]:
    test_mode = environ.get("TEST_MODE", "base_test")
    if not _SAFE_COMPONENT_RE.fullmatch(test_mode) or test_mode in {".", ".."}:
        raise ConfigError("TEST_MODE is not a safe path component")
    raw_count = environ.get("HUGE_EVAL_GPU_COUNT", "")
    raw_ids = environ.get("HUGE_EVAL_GPU_LOGICAL_IDS", "")
    if raw_count:
        _parse_int("HUGE_EVAL_GPU_COUNT", raw_count, minimum=1)
    if raw_count and raw_ids:
        raise ConfigError("HUGE_EVAL_GPU_COUNT and HUGE_EVAL_GPU_LOGICAL_IDS are mutually exclusive")
    if raw_ids:
        try:
            logical_ids = [int(token.strip()) for token in raw_ids.split(",") if token.strip()]
        except ValueError as exc:
            raise ConfigError("HUGE_EVAL_GPU_LOGICAL_IDS must be comma-separated integers") from exc
        if not logical_ids or len(set(logical_ids)) != len(logical_ids) or any(value < 0 for value in logical_ids):
            raise ConfigError("HUGE_EVAL_GPU_LOGICAL_IDS must be non-empty, unique, non-negative integers")
    return {
        "test_mode": test_mode,
        "seeds_override": environ.get("SEEDS_OVERRIDE", ""),
        "repeats_override": environ.get("REPEATS_PER_SEED", ""),
        "max_steps": _parse_int("MAX_STEPS", environ.get("MAX_STEPS", str(task.default_max_steps)), minimum=1),
        "video_fps": _parse_int("VIDEO_FPS", environ.get("VIDEO_FPS", "30"), minimum=1, maximum=240),
        "post_termination_record_steps": _parse_int(
            "POST_TERMINATION_RECORD_STEPS", environ.get("POST_TERMINATION_RECORD_STEPS", "10"), minimum=0
        ),
        "record_video_every_n": _parse_int(
            "HUGE_EVAL_RECORD_VIDEO_EVERY_N",
            environ.get("HUGE_EVAL_RECORD_VIDEO_EVERY_N", str(task.default_record_video_every_n)), minimum=0,
        ),
        "step_log_every_n": _parse_int("HUGE_EVAL_STEP_LOG_EVERY_N", environ.get("HUGE_EVAL_STEP_LOG_EVERY_N", "100"), minimum=0),
        "server_timeout": _parse_float("HUGE_EVAL_SERVER_TIMEOUT", environ.get("HUGE_EVAL_SERVER_TIMEOUT", "360.0"), minimum=0.001),
        "server_ready_timeout": _parse_float(
            "HUGE_EVAL_SERVER_READY_TIMEOUT", environ.get("HUGE_EVAL_SERVER_READY_TIMEOUT", "360"), minimum=0.001
        ),
        "shutdown_grace_sec": _parse_float(
            "HUGE_EVAL_SHUTDOWN_GRACE_SEC", environ.get("HUGE_EVAL_SHUTDOWN_GRACE_SEC", "30"), minimum=0.0
        ),
        "base_port": _parse_int("HUGE_EVAL_BASE_PORT", environ.get("HUGE_EVAL_BASE_PORT", "18080"), minimum=1024, maximum=65535),
        "dry_run": _parse_bool("HUGE_EVAL_DRY_RUN", environ.get("HUGE_EVAL_DRY_RUN", "0")),
        "resume": _parse_bool("HUGE_EVAL_RESUME", environ.get("HUGE_EVAL_RESUME", "1")),
        "force_new": _parse_bool("HUGE_EVAL_FORCE_NEW_RUN", environ.get("HUGE_EVAL_FORCE_NEW_RUN", "0")),
        "allow_kit_map_bootstrap": _parse_bool(
            "HUGE_ALLOW_KIT_MAP_BOOTSTRAP", environ.get("HUGE_ALLOW_KIT_MAP_BOOTSTRAP", "0")
        ),
        "gpu_count": raw_count,
        "gpu_logical_ids": raw_ids,
        "workers_per_gpu": _parse_int(
            "HUGE_EVAL_WORKERS_PER_GPU",
            environ.get("HUGE_EVAL_WORKERS_PER_GPU", str(DEFAULT_WORKERS_PER_GPU)),
            minimum=1,
            maximum=MAX_WORKERS_PER_GPU,
        ),
        "headless": True,
        "enable_depth": False,
        "robot_type": "unitree_g1_refpose_v3_1",
        "root_rot6d_layout": "row",
        "root_max_delta_deg": 26.0,
        "heading_align": True,
        "output_delay_steps": 0,
        "memory_limit_percent": task.memory_limit_percent,
        "open_door_latch": _open_door_policy(task, environ),
    }


def _open_door_policy(task: TaskSpec, environ: Mapping[str, str]) -> dict[str, str] | None:
    if task.cli_task != "open_door":
        return None
    names = (
        "OPEN_DOOR_LATCH_DEBUG", "OPEN_DOOR_LATCH_POLL_LOG_INTERVAL",
        "OPEN_DOOR_HANDLE_UNLOCK_ANGLE_DEG", "OPEN_DOOR_JOINT_DEBUG", "OPEN_DOOR_TRANSFORM_DEBUG",
    )
    result = {name: environ[name] for name in names if name in environ}
    disabled = environ.get("OPEN_DOOR_LATCH_DISABLE", "0")
    if disabled not in {"0", "1"}:
        raise ConfigError("OPEN_DOOR_LATCH_DISABLE must be 0 or 1")
    if disabled == "1":
        result["OPEN_DOOR_LATCH_DISABLE"] = "1"
    else:
        result["OPEN_DOOR_LATCH_ENABLE"] = environ.get("OPEN_DOOR_LATCH_ENABLE", "1")
    return result


def parse_gpu_selection(visible_count: int, count_raw: str, ids_raw: str) -> list[int]:
    if visible_count < 1:
        raise GpuError("no CUDA GPUs are visible to the parent process")
    if count_raw and ids_raw:
        raise ConfigError("GPU count and logical id selection are mutually exclusive")
    if count_raw:
        count = _parse_int("HUGE_EVAL_GPU_COUNT", count_raw, minimum=1, maximum=visible_count)
        return list(range(count))
    if ids_raw:
        try:
            selected = [int(token.strip()) for token in ids_raw.split(",") if token.strip()]
        except ValueError as exc:
            raise ConfigError("HUGE_EVAL_GPU_LOGICAL_IDS must be comma-separated integers") from exc
        if not selected or len(set(selected)) != len(selected):
            raise ConfigError("GPU logical id selection must be non-empty and unique")
        if any(index < 0 or index >= visible_count for index in selected):
            raise ConfigError(f"GPU logical ids must be in [0, {visible_count - 1}]")
        return selected
    return list(range(visible_count))


def validate_checkpoint(checkpoint: str | Path, repo_root: str | Path, task: TaskSpec) -> CheckpointContract:
    repo = Path(repo_root).resolve()
    raw = Path(checkpoint).expanduser()
    candidate = (repo / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        return resolve_checkpoint_contract(
            candidate,
            allowed_root=repo / ".runs/stage3_humanoidarena",
            checkpoint_task=task.checkpoint_task,
            repo_root=repo,
        )
    except CheckpointValidationError as exc:
        raise ConfigError(str(exc)) from exc


def validate_yaml_config(path: str | Path, task: TaskSpec) -> dict[str, Any]:
    yaml_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"invalid task YAML {yaml_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("task YAML must contain an object")
    if payload.get("backend") != "sonic" or payload.get("task_name") != task.isaac_task:
        raise ConfigError("task YAML backend/task_name does not match the frozen SONIC task mapping")
    defaults = payload.get("test_defaults")
    if not isinstance(defaults, dict) or defaults.get("persistent_sim") is not True:
        raise ConfigError("task YAML requires test_defaults.persistent_sim=true")
    if defaults.get("episode_seed_formula", EPISODE_SEED_FORMULA) != EPISODE_SEED_FORMULA:
        raise ConfigError("task YAML episode seed formula is incompatible")
    seeds = defaults.get("seeds")
    repeats = defaults.get("repeats_per_seed")
    if not isinstance(seeds, list) or not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ConfigError("task YAML seeds must be a non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("task YAML seeds must be unique")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ConfigError("task YAML repeats_per_seed must be a positive integer")
    vision = defaults.get("vision_randomization")
    vision = vision if isinstance(vision, dict) and vision.get("enabled") is True else None
    return {
        "path": str(yaml_path), "sha256": sha256_file(yaml_path), "seeds": list(seeds),
        "repeats_per_seed": repeats, "persistent_sim": True,
        "episode_seed_formula": EPISODE_SEED_FORMULA, "vision_randomization": vision,
    }


def resolve_seeds_and_repeats(yaml_config: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[list[int], int]:
    if policy["seeds_override"]:
        try:
            seeds = [int(token.strip()) for token in str(policy["seeds_override"]).split(",") if token.strip()]
        except ValueError as exc:
            raise ConfigError("SEEDS_OVERRIDE must be comma-separated integers") from exc
        if not seeds or len(set(seeds)) != len(seeds):
            raise ConfigError("SEEDS_OVERRIDE must be non-empty and unique")
    else:
        seeds = [int(value) for value in yaml_config["seeds"]]
    repeats = (
        _parse_int("REPEATS_PER_SEED", str(policy["repeats_override"]), minimum=1)
        if policy["repeats_override"] else int(yaml_config["repeats_per_seed"])
    )
    return seeds, repeats


def derive_episode_seed(task_name: str, group_seed: int, repeat_idx: int) -> int:
    payload = f"{task_name}|{int(group_seed)}|{int(repeat_idx)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big", signed=False) & 0x7FFFFFFF


def build_episode_jobs(
    task: TaskSpec, test_mode: str, seeds: Sequence[int], repeats_per_seed: int, yaml_sha256: str
) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []
    episode_index = 0
    for seed in seeds:
        for repeat_idx in range(repeats_per_seed):
            episode_seed = derive_episode_seed(task.isaac_task, seed, repeat_idx)
            identity_payload = {
                "task_cli": task.cli_task, "isaac_task_name": task.isaac_task, "test_mode": test_mode,
                "seed": int(seed), "repeat_idx": repeat_idx, "episode_seed": episode_seed,
                "episode_index": episode_index, "instruction": task.instruction,
                "env_config_yaml_sha256": yaml_sha256,
            }
            jobs.append(EpisodeJob(**identity_payload, result_identity=canonical_sha256(identity_payload)))
            episode_index += 1
    return jobs


def partition_pending_jobs(jobs: Sequence[EpisodeJob], worker_count: int) -> list[list[EpisodeJob]]:
    ordered = sorted(jobs, key=lambda job: (job.seed, job.repeat_idx, job.episode_seed))
    active = min(max(0, worker_count), len(ordered))
    return [ordered[index::active] for index in range(active)] if active else []


def expand_worker_bindings(
    bindings: Sequence[GpuBinding], workers_per_gpu: int
) -> list[GpuBinding]:
    if workers_per_gpu < 1 or workers_per_gpu > MAX_WORKERS_PER_GPU:
        raise ConfigError(f"workers per GPU must be in [1, {MAX_WORKERS_PER_GPU}]")
    return [
        GpuBinding(binding.gpu, binding.renderer_ordinal, worker_slot)
        for worker_slot in range(workers_per_gpu)
        for binding in bindings
    ]


def _verify_hash(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise ConfigError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ConfigError(f"{label} hash drift: {path}: {actual}, expected {expected}")
    return actual


def _directory_inventory(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise ConfigError(f"required local model directory is missing: {path}")
    inventory: dict[str, str] = {}
    for entry in sorted(path.rglob("*")):
        if entry.is_symlink():
            resolved = entry.resolve()
            if not _is_relative_to(resolved, path.resolve()):
                raise ConfigError(f"model asset symlink escapes its root: {entry}")
            inventory[entry.relative_to(path).as_posix()] = f"symlink:{os.readlink(entry)}"
        elif entry.is_file():
            inventory[entry.relative_to(path).as_posix()] = sha256_file(entry)
    if not inventory:
        raise ConfigError(f"local model asset directory is empty: {path}")
    return inventory


def build_source_inventory(
    repo: Path, humanoidarena: Path, yaml_path: Path, checkpoint: CheckpointContract
) -> dict[str, Any]:
    upstream: dict[str, str] = {}
    for relative, expected in UPSTREAM_COMPATIBILITY_BASELINE.items():
        upstream[relative] = _verify_hash(humanoidarena / relative, expected, label="upstream compatibility file")
    isaaclab = humanoidarena / "isaaclab_twist2_g1"
    wrappers: dict[str, str] = {}
    for relative, expected in WRAPPER_BASELINE.items():
        wrappers[relative] = _verify_hash(isaaclab / relative, expected, label="wrapper semantics file")
    huge_paths = (
        "src/huge/inference/humanoidarena_http_server.py",
        "scripts/eval/humanoidarena/eval_orchestrator.py",
        "scripts/eval/humanoidarena/sim_eval_shadow.py",
        "scripts/eval/humanoidarena/run_eval.sh",
        "scripts/eval/humanoidarena/env.sh",
    )
    huge = {relative: sha256_file(repo / relative) for relative in huge_paths if (repo / relative).is_file()}
    encoder = repo / "resource/model/sonic/model_encoder.onnx"
    decoder = repo / "resource/model/sonic/model_decoder.onnx"
    _verify_hash(encoder, SONIC_ENCODER_SHA256, label="Huge SONIC encoder")
    _verify_hash(decoder, SONIC_DECODER_SHA256, label="Huge SONIC decoder")
    upstream_encoder = humanoidarena / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
    upstream_decoder = humanoidarena / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
    _verify_hash(upstream_encoder, SONIC_ENCODER_SHA256, label="HumanoidArena SONIC encoder")
    _verify_hash(upstream_decoder, SONIC_DECODER_SHA256, label="HumanoidArena SONIC decoder")
    qwen_inventory = _directory_inventory(checkpoint.base_vlm_path)
    payload = {
        "upstream_compatibility": upstream,
        "wrapper_semantics": wrappers,
        "wrapper_semantics_digest": canonical_sha256(list(wrappers.items())),
        "huge_sources": huge,
        "task_yaml": {str(yaml_path): sha256_file(yaml_path)},
        "checkpoint_files": dict(checkpoint.file_sha256),
        "sonic_assets": {
            "huge_encoder": SONIC_ENCODER_SHA256, "huge_decoder": SONIC_DECODER_SHA256,
            "humanoidarena_encoder": SONIC_ENCODER_SHA256, "humanoidarena_decoder": SONIC_DECODER_SHA256,
        },
        "qwen_assets": qwen_inventory,
        "qwen_assets_digest": canonical_sha256(qwen_inventory),
    }
    return {"files": payload, "source_inventory_digest": canonical_sha256(payload)}


def inspect_upstream_contract(
    isaaclab_python: Path, shadow: Path, humanoidarena: Path, isaaclab: Path,
    upstream_sim: Path, isaac_pythonpath: str,
) -> dict[str, Any]:
    command = [
        str(isaaclab_python), str(shadow), "--inspect-contract",
        "--humanoidarena-root", str(humanoidarena), "--isaaclab-root", str(isaaclab),
        "--upstream-sim-eval", str(upstream_sim), "--upstream-sim-sha256", UPSTREAM_SIM_SHA256,
    ]
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": "/tmp/huge-humanoidarena-pycache/inspect",
        "PYTHONPATH": isaac_pythonpath, "PROJECT_ROOT": str(isaaclab),
    })
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env, timeout=120)
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
        raise ConfigError(f"read-only upstream contract inspection failed: {exc}") from exc
    if not isinstance(report, dict) or report.get("status") != "compatible" or report.get("sha256") != UPSTREAM_SIM_SHA256:
        raise ConfigError("upstream contract inspection returned an incompatible report")
    return report


def build_static_plan(
    task: TaskSpec, checkpoint: CheckpointContract, yaml_config: Mapping[str, Any],
    jobs: Sequence[EpisodeJob], policy: Mapping[str, Any], source_inventory: Mapping[str, Any],
    upstream_contract: Mapping[str, Any], repo: Path,
) -> dict[str, Any]:
    safe_policy = dict(policy)
    for execution_only in (
        "dry_run", "allow_kit_map_bootstrap", "resume", "force_new", "gpu_count",
        "gpu_logical_ids", "base_port", "server_ready_timeout", "shutdown_grace_sec",
        "seeds_override", "repeats_override",
    ):
        safe_policy.pop(execution_only, None)
    return {
        "schema_version": STATIC_PLAN_SCHEMA_VERSION,
        "task": asdict(task),
        "checkpoint": {
            "relative_path": checkpoint.checkpoint_dir.relative_to(repo).as_posix(),
            "step": checkpoint.step, "checkpoint_task": checkpoint.checkpoint_task,
            "checkpoint_sha256": checkpoint.checkpoint_sha256, "files": dict(checkpoint.file_sha256),
            "qwen_path": str(checkpoint.base_vlm_path.relative_to(repo)),
        },
        "yaml": dict(yaml_config),
        "policy": safe_policy,
        "protocol": {"http": PROTOCOL_VERSION, "mapping": MAPPING_VERSION, "action": [50, 66], "latent": [50, 64]},
        "episode_seed_algorithm": EPISODE_SEED_FORMULA,
        "episodes": [asdict(job) for job in jobs],
        "source_inventory": dict(source_inventory),
        "upstream_contract": dict(upstream_contract),
        "software": {"python": platform.python_version(), "platform": platform.platform()},
    }


def build_executable_contract(
    static_plan: Mapping[str, Any], bindings: Sequence[Mapping[str, Any] | GpuBinding], kit_map_digest: str
) -> dict[str, Any]:
    normalized = [asdict(binding) if isinstance(binding, GpuBinding) else dict(binding) for binding in bindings]
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "static_plan": dict(static_plan),
        "static_plan_hash": canonical_sha256(static_plan),
        "gpu_bindings": normalized,
        "kit_gpu_map_sha256": kit_map_digest,
    }
    return {"contract": contract, "contract_hash": canonical_sha256(contract)}


def _normalize_uuid(value: str) -> str:
    token = value.strip()
    if not token:
        return ""
    if token.startswith("MIG-"):
        raise GpuError("MIG devices are not supported by this evaluation contract")
    if not re.fullmatch(r"GPU-[A-Za-z0-9-]+", token, re.IGNORECASE):
        raise GpuError(f"invalid CUDA GPU UUID: {value!r}")
    return token.upper()


def normalize_gpu_uuid_for_match(value: str) -> str:
    """Normalize CUDA and raw Vulkan UUID spellings for identity comparison."""
    token = str(value).strip().lower()
    token = token.removeprefix("mig-gpu-").removeprefix("gpu-")
    return re.sub(r"[^0-9a-f]", "", token)


def _normalize_pci(value: str) -> str:
    token = value.strip().upper()
    match = re.search(r"(?:[0-9A-F]{4,8}:)?([0-9A-F]{2}):([0-9A-F]{2})\.([0-7])", token)
    if not match:
        raise GpuError(f"invalid PCI bus id: {value!r}")
    domain = token.split(":", 1)[0] if token.count(":") >= 2 else "00000000"
    return f"{domain.zfill(8)}:{match.group(1)}:{match.group(2)}.{match.group(3)}"


def _nvidia_inventory() -> tuple[list[dict[str, str]], str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise GpuError("nvidia-smi is required for GPU identity cross-check")
    command = [
        executable, "--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except subprocess.SubprocessError as exc:
        raise GpuError(f"nvidia-smi inventory failed: {exc}") from exc
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        index, uuid, pci, name, memory_mib, memory_free_mib, driver = parts
        rows.append({
            "index": index, "uuid": uuid, "pci_bus_id": _normalize_pci(pci), "name": name,
            "memory_total_mib": memory_mib, "memory_free_mib": memory_free_mib,
            "driver_version": driver,
        })
    if not rows or len({row["driver_version"] for row in rows}) != 1:
        raise GpuError("nvidia-smi did not return a coherent GPU/driver inventory")
    return rows, rows[0]["driver_version"]


def discover_visible_gpus(policy: Mapping[str, Any]) -> tuple[list[GpuIdentity], str]:
    try:
        import torch
    except ImportError as exc:
        raise GpuError("PyTorch is required for CUDA runtime discovery") from exc
    if not torch.cuda.is_available():
        raise GpuError("CUDA runtime is unavailable")
    count = torch.cuda.device_count()
    selected = parse_gpu_selection(count, str(policy["gpu_count"]), str(policy["gpu_logical_ids"]))
    smi_rows, driver = _nvidia_inventory()
    by_uuid = {_normalize_uuid(row["uuid"]): row for row in smi_rows}
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    mask_tokens = [token.strip() for token in mask.split(",") if token.strip()] if mask else []
    identities: list[GpuIdentity] = []
    for ordinal in selected:
        properties = torch.cuda.get_device_properties(ordinal)
        raw_uuid = str(getattr(properties, "uuid", "") or "")
        if raw_uuid and not raw_uuid.upper().startswith(("GPU-", "MIG-")):
            raw_uuid = f"GPU-{raw_uuid}"
        if not raw_uuid and mask_tokens:
            token = mask_tokens[ordinal]
            if token.upper().startswith(("GPU-", "MIG-")):
                matches = [row["uuid"] for row in smi_rows if row["uuid"].upper().startswith(token.upper())]
                if len(matches) == 1:
                    raw_uuid = matches[0]
            elif token.isdigit():
                matches = [row["uuid"] for row in smi_rows if row["index"] == token]
                raw_uuid = matches[0] if len(matches) == 1 else ""
        if not raw_uuid and not mask_tokens:
            matches = [row["uuid"] for row in smi_rows if row["index"] == str(ordinal)]
            raw_uuid = matches[0] if len(matches) == 1 else ""
        uuid = _normalize_uuid(raw_uuid)
        row = by_uuid.get(uuid)
        if not uuid or row is None:
            raise GpuError(f"CUDA logical device {ordinal} has no unambiguous UUID/PCI identity")
        identities.append(GpuIdentity(
            ordinal, row["uuid"], row["uuid"], row["pci_bus_id"], str(properties.name),
            int(properties.total_memory), f"{properties.major}.{properties.minor}",
            int(float(row["memory_free_mib"]) * 1024 * 1024),
        ))
    return identities, driver


def host_fingerprint() -> str:
    machine_id = ""
    try:
        machine_id = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return canonical_sha256({"hostname": platform.node(), "machine_id": machine_id, "machine": platform.machine()})


def isaac_fingerprint(isaaclab_python: Path, isaaclab: Path) -> tuple[str, str]:
    code = (
        "import importlib.metadata as m,json; out={}; "
        "[(out.__setitem__(n,m.version(n))) for n in ('isaacsim','isaacsim-kernel','isaacsim-app','omni-kit') "
        "if (lambda x: x is not None)(next((d for d in m.distributions() if (d.metadata.get('Name') or '').lower()==n),None))]; "
        "print(json.dumps(out,sort_keys=True))"
    )
    try:
        completed = subprocess.run([str(isaaclab_python), "-c", code], check=True, capture_output=True, text=True, timeout=60)
        versions = json.loads(completed.stdout.strip() or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise GpuError(f"unable to fingerprint Isaac/Kit runtime: {exc}") from exc
    isaac_version = str(versions.get("isaacsim") or versions.get("isaacsim-app") or "unknown")
    if isaac_version == "unknown":
        raise GpuError("Isaac Sim package version is unavailable")
    kit_payload = {
        "versions": versions,
        "app_launcher_sha256": sha256_file(
            isaaclab.parent / "external/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py"
        ),
    }
    return isaac_version, canonical_sha256(kit_payload)


def load_kit_gpu_map(
    path: str | Path, selected_gpus: Sequence[GpuIdentity], expected_host: str,
    expected_driver: str, expected_isaac: str, expected_kit: str,
) -> tuple[list[GpuBinding], str]:
    map_path = Path(path).resolve()
    try:
        payload = _json_object(map_path, label="Kit GPU map")
    except ConfigError as exc:
        raise GpuError(str(exc)) from exc
    expected_fields = {
        "schema_version": KIT_MAP_SCHEMA_VERSION, "host_fingerprint": expected_host,
        "driver_version": expected_driver, "isaac_sim_version": expected_isaac, "kit_build": expected_kit,
    }
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise GpuError(f"Kit GPU map {key} mismatch: {payload.get(key)!r}, expected {expected!r}")
    entries = payload.get("gpus")
    if not isinstance(entries, list):
        raise GpuError("Kit GPU map requires a gpus list")
    bindings: list[GpuBinding] = []
    renderers: set[int] = set()
    for gpu in selected_gpus:
        matches = [entry for entry in entries if isinstance(entry, dict) and _normalize_uuid(str(entry.get("cuda_uuid", ""))) == _normalize_uuid(gpu.cuda_uuid)]
        if len(matches) != 1:
            raise GpuError(f"Kit GPU map must contain one entry for {gpu.cuda_uuid}")
        entry = matches[0]
        if _normalize_pci(str(entry.get("pci_bus_id", ""))) != _normalize_pci(gpu.pci_bus_id):
            raise GpuError(f"Kit GPU map PCI mismatch for {gpu.cuda_uuid}")
        renderer = entry.get("kit_renderer_ordinal")
        if isinstance(renderer, bool) or not isinstance(renderer, int) or renderer < 0 or renderer in renderers:
            raise GpuError("Kit renderer ordinals must be non-negative and unique")
        renderers.add(renderer)
        bindings.append(GpuBinding(gpu, renderer))
    return bindings, sha256_file(map_path)


def bootstrap_kit_gpu_map(
    path: Path, selected_gpus: Sequence[GpuIdentity], host: str, driver: str, isaac_version: str,
    kit_build: str, args: argparse.Namespace,
) -> None:
    report = path.with_name(f".{path.name}.bootstrap.{os.getpid()}.json")
    command = [
        str(args.isaaclab_python), str(args.shadow), "--gpu-foundation-bootstrap-only",
        "--humanoidarena-root", str(args.humanoidarena_root), "--isaaclab-root", str(args.isaaclab_root),
        "--upstream-sim-eval", str(args.upstream_sim_eval), "--upstream-sim-sha256", UPSTREAM_SIM_SHA256,
        "--renderer-gpu-ordinal", "0", "--gpu-report-json", str(report),
    ]
    env = _isaac_environment(args, run_id="bootstrap", worker_id="w00", gpu_uuid="")
    try:
        subprocess.run(command, check=True, env=env, timeout=300)
        raw = _json_object(report, label="Kit bootstrap report")
        devices = raw.get("devices")
        if not isinstance(devices, list):
            raise GpuError("Kit bootstrap returned no renderer devices")
        rows = []
        for gpu in selected_gpus:
            expected_uuid = normalize_gpu_uuid_for_match(gpu.cuda_uuid)
            matches = [
                device for device in devices
                if isinstance(device, dict)
                and normalize_gpu_uuid_for_match(str(device.get("uuid", ""))) == expected_uuid
            ]
            if len(matches) != 1:
                raise GpuError(f"Kit bootstrap could not uniquely map {gpu.cuda_uuid}")
            device = matches[0]
            if _normalize_pci(str(device.get("pci_bus_id", ""))) != _normalize_pci(gpu.pci_bus_id):
                raise GpuError(f"Kit bootstrap PCI mismatch for {gpu.cuda_uuid}")
            rows.append({
                "cuda_uuid": gpu.cuda_uuid, "pci_bus_id": gpu.pci_bus_id,
                "kit_renderer_ordinal": int(device["renderer_ordinal"]),
            })
        atomic_write_json(path, {
            "schema_version": KIT_MAP_SCHEMA_VERSION, "host_fingerprint": host,
            "driver_version": driver, "isaac_sim_version": isaac_version, "kit_build": kit_build,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpus": rows,
        })
    except subprocess.SubprocessError as exc:
        raise GpuError(f"Kit GPU map bootstrap failed: {exc}") from exc
    finally:
        report.unlink(missing_ok=True)


def select_or_create_attempt(
    attempts_root: str | Path, contract_hash: str, *, resume: bool, force_new: bool,
    now: str | None = None, nonce: str | None = None,
) -> Path:
    root = Path(attempts_root)
    root.mkdir(parents=True, exist_ok=True)
    existing: list[Path] = []
    for candidate in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _json_object(manifest_path, label="attempt manifest")
        if manifest.get("contract_hash") != contract_hash:
            raise ConfigError(f"attempt contract drift detected: {candidate}")
        existing.append(candidate)
    if resume and not force_new:
        incomplete = [path for path in existing if not (path / "_SUCCESS").exists()]
        if incomplete:
            return incomplete[0]
        if existing:
            return existing[0]
    timestamp = now or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    random_nonce = nonce or secrets.token_hex(4)
    run_id = f"{timestamp}_{random_nonce}"
    if not _SAFE_COMPONENT_RE.fullmatch(run_id):
        raise ConfigError("generated attempt id is unsafe")
    attempt = root / run_id
    attempt.mkdir()
    for relative in ("logs", "status", "shards", "raw_results", "completions", "episodes", "videos/success", "videos/failure", "recordings"):
        (attempt / relative).mkdir(parents=True, exist_ok=True)
    return attempt


def _completion_filename(job: EpisodeJob) -> str:
    return f"seed-{job.seed:010d}-repeat-{job.repeat_idx:03d}-episode-{job.episode_index:05d}.json"


def classify_existing_result(job: EpisodeJob, completion: Mapping[str, Any] | None) -> str:
    if completion is None:
        return "pending_missing"
    if completion.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        return "pending_integrity"
    if "result_identity" in completion and completion.get("result_identity") != job.result_identity:
        return "pending_integrity"
    payload = completion.get("payload", completion.get("raw_payload"))
    if not isinstance(payload, dict):
        return "pending_integrity"
    identity = completion.get("identity")
    if isinstance(identity, dict):
        expected_identity = {
            "task": job.isaac_task_name, "seed": job.seed, "repeat_idx": job.repeat_idx,
            "episode_seed": job.episode_seed, "episode_index": job.episode_index,
        }
        if any(identity.get(key) != value for key, value in expected_identity.items()):
            return "pending_integrity"
    elif any(payload.get(key) != value for key, value in (
        ("seed", job.seed), ("repeat_idx", job.repeat_idx), ("episode_seed", job.episode_seed)
    )):
        return "pending_integrity"
    failure = str(payload.get("failure_reason", ""))
    error = str(payload.get("error", "") or "")
    if not isinstance(payload.get("success"), bool):
        return "pending_integrity"
    if failure == "sim_error" or error:
        return "pending_infrastructure"
    if failure in _LEGAL_TASK_OUTCOMES and bool(payload["success"]) == (failure == "success"):
        return "task_complete"
    return "pending_integrity"


def validate_and_publish_completion(attempt: Path, job: EpisodeJob) -> dict[str, Any] | None:
    completion_path = attempt / "completions" / _completion_filename(job)
    if not completion_path.is_file():
        return None
    try:
        completion = _json_object(completion_path, label="completion envelope")
    except ConfigError:
        return None
    if classify_existing_result(job, completion) != "task_complete":
        return None
    raw_path = Path(str(completion.get("raw_result_path", ""))).resolve()
    if not _is_relative_to(raw_path, attempt) or not raw_path.is_file():
        return None
    if completion.get("raw_result_sha256") != sha256_file(raw_path):
        return None
    payload = completion.get("payload")
    if not isinstance(payload, dict) or completion.get("payload_sha256") != canonical_sha256(payload):
        return None
    artifacts: dict[str, Any] = {}
    if payload.get("video_recorded"):
        video_path = Path(str(payload.get("video_path", ""))).resolve()
        if not _is_relative_to(video_path, attempt) or not video_path.is_file():
            return None
        artifacts["video"] = {"path": str(video_path), "sha256": sha256_file(video_path)}
    canonical = {
        "schema_version": EPISODE_SCHEMA_VERSION, "result_identity": job.result_identity,
        "job": asdict(job), "payload": payload, "worker_id": completion.get("worker_id", ""),
        "artifacts": artifacts, "completion_sha256": sha256_file(completion_path),
    }
    atomic_write_json(attempt / "episodes" / _completion_filename(job), canonical)
    return canonical


def _read_existing_episode(attempt: Path, job: EpisodeJob) -> dict[str, Any] | None:
    path = attempt / "episodes" / _completion_filename(job)
    if not path.is_file():
        return None
    try:
        payload = _json_object(path, label="canonical episode")
    except ConfigError:
        return None
    if (
        payload.get("schema_version") != EPISODE_SCHEMA_VERSION
        or payload.get("result_identity") != job.result_identity
        or payload.get("job") != asdict(job)
    ):
        return None
    result = payload.get("payload")
    if classify_existing_result(job, {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "result_identity": job.result_identity,
        "raw_payload": result,
    }) != "task_complete":
        return None
    if isinstance(result, dict) and result.get("video_recorded"):
        artifact = payload.get("artifacts", {}).get("video") if isinstance(payload.get("artifacts"), dict) else None
        if not isinstance(artifact, dict):
            return None
        video_path = Path(str(artifact.get("path", ""))).resolve()
        if (
            not _is_relative_to(video_path, attempt)
            or not video_path.is_file()
            or artifact.get("sha256") != sha256_file(video_path)
        ):
            return None
    return payload


def _verify_success_attempt(
    attempt: Path, jobs: Sequence[EpisodeJob], contract_hash: str, source_digest: str
) -> None:
    summary = _json_object(attempt / "summary.json", label="successful attempt summary")
    marker = attempt / "_SUCCESS"
    try:
        marker_digest = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(f"cannot read successful attempt marker: {exc}") from exc
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("contract_hash") != contract_hash
        or summary.get("source_inventory_digest") != source_digest
        or summary.get("expected") != len(jobs)
        or summary.get("completed") != len(jobs)
        or marker_digest != canonical_sha256(summary)
    ):
        raise ConfigError("successful attempt summary/marker does not match the current contract")
    if any(_read_existing_episode(attempt, job) is None for job in jobs):
        raise ConfigError("successful attempt has missing, drifted, or invalid canonical episodes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Huge read-only HumanoidArena SONIC multi-GPU evaluator")
    parser.add_argument("--humanoidarena-root", required=True)
    parser.add_argument("--isaaclab-root", required=True)
    parser.add_argument("--isaaclab-python", required=True)
    parser.add_argument("--isaac-pythonpath", required=True)
    parser.add_argument("task")
    parser.add_argument("checkpoint")
    parser.add_argument("backend", nargs="?", default="sonic")
    return parser


def _resolve_cli(args: argparse.Namespace) -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    humanoidarena = Path(args.humanoidarena_root).resolve()
    isaaclab = Path(args.isaaclab_root).resolve()
    isaac_python = Path(args.isaaclab_python).resolve()
    if humanoidarena != (repo / "HumanoidArena").resolve():
        raise ConfigError("HumanoidArena root must be the repository-local read-only dependency")
    if isaaclab != (humanoidarena / "isaaclab_twist2_g1").resolve():
        raise ConfigError("IsaacLab root must be HumanoidArena/isaaclab_twist2_g1")
    if not isaac_python.is_file():
        raise ConfigError(f"Isaac Lab Python is missing: {isaac_python}")
    if args.task not in TASK_SPECS:
        raise ConfigError(f"unknown task {args.task!r}; expected one of {', '.join(TASK_SPECS)}")
    if args.backend != "sonic":
        raise ConfigError("backend must be sonic; all/twist2 are intentionally unsupported")
    args.repo = repo
    args.humanoidarena_root = humanoidarena
    args.isaaclab_root = isaaclab
    args.isaaclab_python = isaac_python
    args.shadow = repo / "scripts/eval/humanoidarena/sim_eval_shadow.py"
    args.upstream_sim_eval = isaaclab / "script/eval_scripts/sonic/sim_eval_vla.py"
    return args


def _output_root(repo: Path, environ: Mapping[str, str]) -> Path:
    raw = Path(environ.get("HUGE_EVAL_OUTPUT_ROOT", ".runs/eval/humanoidarena")).expanduser()
    path = (repo / raw).resolve() if not raw.is_absolute() else raw.resolve()
    allowed = (repo / ".runs").resolve()
    if not _is_relative_to(path, allowed):
        raise ConfigError(f"HUGE_EVAL_OUTPUT_ROOT must resolve inside {allowed}")
    return path


def _kit_map_path(repo: Path, output_root: Path, environ: Mapping[str, str]) -> Path:
    raw = environ.get("HUGE_KIT_GPU_MAP_FILE", "")
    if raw:
        candidate = Path(raw).expanduser()
        path = (repo / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    else:
        path = output_root / "gpu_maps" / f"{host_fingerprint()}.json"
    allowed = (repo / ".runs").resolve()
    if not _is_relative_to(path.resolve(), allowed):
        raise ConfigError(f"HUGE_KIT_GPU_MAP_FILE must resolve inside {allowed}")
    return path.resolve()


def _static_preflight(args: argparse.Namespace) -> tuple[TaskSpec, CheckpointContract, dict[str, Any], list[EpisodeJob], dict[str, Any], dict[str, Any]]:
    task = TASK_SPECS[args.task]
    policy = parse_runtime_policy(task, os.environ)
    output_root = _output_root(args.repo, os.environ)
    policy["output_root"] = str(output_root)
    policy["kit_gpu_map_file"] = str(_kit_map_path(args.repo, output_root, os.environ))
    checkpoint = validate_checkpoint(args.checkpoint, args.repo, task)
    yaml_path = args.isaaclab_root / "tasks/common_test_config" / policy["test_mode"] / f"{task.yaml_stem}_sonic_test.yaml"
    expected_yaml_root = (args.isaaclab_root / "tasks/common_test_config").resolve()
    if not _is_relative_to(yaml_path.resolve(), expected_yaml_root):
        raise ConfigError("task YAML escapes common_test_config")
    yaml_config = validate_yaml_config(yaml_path, task)
    seeds, repeats = resolve_seeds_and_repeats(yaml_config, policy)
    yaml_config = dict(yaml_config)
    yaml_config["selected_seeds"] = seeds
    yaml_config["selected_repeats_per_seed"] = repeats
    jobs = build_episode_jobs(task, policy["test_mode"], seeds, repeats, yaml_config["sha256"])
    upstream_contract = inspect_upstream_contract(
        args.isaaclab_python, args.shadow, args.humanoidarena_root, args.isaaclab_root,
        args.upstream_sim_eval, args.isaac_pythonpath,
    )
    source_inventory = build_source_inventory(args.repo, args.humanoidarena_root, yaml_path, checkpoint)
    static_plan = build_static_plan(task, checkpoint, yaml_config, jobs, policy, source_inventory, upstream_contract, args.repo)
    return task, checkpoint, policy, jobs, source_inventory, static_plan


def _isaac_environment(args: argparse.Namespace, *, run_id: str, worker_id: str, gpu_uuid: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": f"/tmp/huge-humanoidarena-pycache/{run_id}/{worker_id}",
        "PYTHONPATH": args.isaac_pythonpath, "PROJECT_ROOT": str(args.isaaclab_root),
        "OMNI_KIT_ACCEPT_EULA": "Y", "SONIC_VLA_ACTION_FORMAT": "latent64",
        "SONIC_VLA_USE_HEADING_ALIGN": "1", "SONIC_OUTPUT_DELAY_STEPS": "0",
        "ROBOT_USD_OVERRIDE": str(args.isaaclab_root / "assets/robots/g1-29dof_wholebody_dex3/g1_29dof_with_dex3_rev_1_0_m2.usd"),
    })
    if gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return env


def _server_environment(repo: Path, gpu_uuid: str) -> dict[str, str]:
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    venv_root = (repo / ".venv").resolve()
    if not _is_relative_to(purelib, venv_root):
        raise ConfigError(f"orchestrator is not running from the Huge root virtual environment: {purelib}")
    cudnn_lib = purelib / "nvidia/cudnn/lib"
    required_cudnn = (cudnn_lib / "libcudnn.so.9", cudnn_lib / "libcudnn_graph.so.9")
    if not all(path.is_file() for path in required_cudnn):
        raise ConfigError(f"Huge root virtual environment has an incomplete cuDNN installation: {cudnn_lib}")
    inherited_ld = os.environ.get("LD_LIBRARY_PATH", "")
    inherited_parts = [part for part in inherited_ld.split(":") if part and part != str(cudnn_lib)]
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": "/tmp/huge-agent-pycache",
        "PYTHONPATH": str(repo / "src"), "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "LD_LIBRARY_PATH": ":".join([str(cudnn_lib), *inherited_parts]),
    })
    return env


def _runtime_environment(task: TaskSpec, yaml_config: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    vision = yaml_config.get("vision_randomization")
    if isinstance(vision, dict):
        env["VISION_RANDOMIZATION"] = canonical_json_bytes(vision).decode("utf-8")
    latch = policy.get("open_door_latch")
    if isinstance(latch, dict):
        env.update({str(key): str(value) for key, value in latch.items()})
        if "OPEN_DOOR_LATCH_DISABLE" in env:
            env.pop("OPEN_DOOR_LATCH_ENABLE", None)
    return env


def _make_shm(run_id: str, worker_id: str) -> OwnedShm:
    run8 = re.sub(r"[^a-z0-9]", "", run_id.lower())[:8]
    if len(run8) != 8 or not _WORKER_RE.fullmatch(worker_id):
        raise SimulationError("run/worker identity cannot be represented by the SHM ownership format")
    for _ in range(100):
        nonce = secrets.token_hex(8)
        name = f"huge_ha_{run8}_{worker_id}_{nonce}"
        path = Path("/dev/shm") / name
        if not os.path.lexists(path):
            return OwnedShm(name, path, os.getuid())
    raise SimulationError("unable to allocate a unique SHM ownership name")


def _reserve_ports(base: int, count: int) -> list[socket.socket]:
    reservations: list[socket.socket] = []
    candidate = base
    try:
        while len(reservations) < count and candidate <= 65535:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", candidate))
                sock.listen(1)
                reservations.append(sock)
            except OSError:
                sock.close()
            candidate += 1
        if len(reservations) != count:
            raise ServerError(f"unable to reserve {count} loopback ports starting at {base}")
        return reservations
    except Exception:
        for sock in reservations:
            sock.close()
        raise


def _server_command(
    args: argparse.Namespace, task: TaskSpec, checkpoint: CheckpointContract, binding: GpuBinding,
    run_id: str, worker_id: str, attempt: Path, port: int,
) -> list[str]:
    return [
        str(args.repo / ".venv/bin/python"), "-m", "huge.inference.humanoidarena_http_server",
        "--repo-root", str(args.repo), "--checkpoint", str(checkpoint.checkpoint_dir),
        "--allowed-checkpoint-root", str(args.repo / ".runs/stage3_humanoidarena"),
        "--checkpoint-task", task.checkpoint_task, "--task", task.cli_task,
        "--isaac-task", task.isaac_task, "--instruction", task.instruction,
        "--host", "127.0.0.1", "--port", str(port), "--expected-cuda-uuid", binding.gpu.cuda_uuid,
        "--run-id", run_id, "--worker-id", worker_id, "--attempt-dir", str(attempt),
        "--status-file", str(attempt / "status" / f"server-{worker_id}.json"), "--device", "cuda:0",
    ]


def _write_shard(
    path: Path, jobs: Sequence[EpisodeJob], attempt: Path, checkpoint: CheckpointContract,
    policy: Mapping[str, Any], worker_id: str,
) -> None:
    episodes = []
    for job in jobs:
        episodes.append({
            "seed": job.seed, "repeat_idx": job.repeat_idx, "episode_seed": job.episode_seed,
            "episode_index": job.episode_index,
            "result_json": str(attempt / "raw_results" / _completion_filename(job)),
            "success_video_dir": str(attempt / "videos/success"),
            "failure_video_dir": str(attempt / "videos/failure"),
            "video_fps": policy["video_fps"],
            "post_termination_record_steps": policy["post_termination_record_steps"],
            "model_label": checkpoint.checkpoint_dir.name,
            "eval_model_path": str(checkpoint.checkpoint_dir),
            "recording_save_dir": str(attempt / "recordings" / worker_id),
            "max_steps": policy["max_steps"],
        })
    atomic_write_json(path, {"schema_version": "huge_humanoidarena_shard_v1", "worker_id": worker_id, "episodes": episodes})


def _shadow_command(
    args: argparse.Namespace, task: TaskSpec, checkpoint: CheckpointContract, binding: GpuBinding,
    run_id: str, worker_id: str, attempt: Path, shard_path: Path, jobs: Sequence[EpisodeJob],
    port: int, shm: OwnedShm, policy: Mapping[str, Any], yaml_config: Mapping[str, Any],
) -> list[str]:
    command = [
        str(args.isaaclab_python), str(args.shadow),
        "--humanoidarena-root", str(args.humanoidarena_root), "--isaaclab-root", str(args.isaaclab_root),
        "--upstream-sim-eval", str(args.upstream_sim_eval), "--upstream-sim-sha256", UPSTREAM_SIM_SHA256,
        "--renderer-gpu-ordinal", str(binding.renderer_ordinal), "--expected-cuda-uuid", binding.gpu.cuda_uuid,
        "--expected-pci-bus-id", binding.gpu.pci_bus_id,
        "--task", task.isaac_task, "--env_config_yaml", str(yaml_config["path"]),
        "--seed", str(jobs[0].seed), "--episode_batch_json", str(shard_path),
        "--max_steps", str(policy["max_steps"]),
        "--sonic_encoder_path", str(args.repo / "resource/model/sonic/model_encoder.onnx"),
        "--sonic_decoder_path", str(args.repo / "resource/model/sonic/model_decoder.onnx"),
        "--sonic_vla_root_rot6d_layout", "row", "--sonic_vla_root_max_delta_deg", "26.0",
        "--lerobot_server_url", f"http://127.0.0.1:{port}",
        "--lerobot_server_timeout", str(policy["server_timeout"]),
        "--robot_type", str(policy["robot_type"]), "--video_fps", str(policy["video_fps"]),
        "--post_termination_record_steps", str(policy["post_termination_record_steps"]),
        "--record_video_every_n", str(policy["record_video_every_n"]),
        "--step_log_every_n", str(policy["step_log_every_n"]),
        "--model_label", checkpoint.checkpoint_dir.name, "--eval_model_path", str(checkpoint.checkpoint_dir),
        "--recording_save_dir", str(attempt / "recordings" / worker_id), "--headless", "--device", "cuda:0",
        "--owned-shm-name", shm.name, "--run-id", run_id, "--worker-id", worker_id,
        "--attempt-dir", str(attempt), "--server-status-json", str(attempt / "status" / f"server-{worker_id}.json"),
        "--worker-status-json", str(attempt / "status" / f"worker-{worker_id}.json"),
        "--completion-dir", str(attempt / "completions"),
    ]
    if task.memory_limit_percent is not None:
        command.extend(["--memory-limit-percent", str(task.memory_limit_percent)])
    return command


def _launch_child(role: str, worker_id: str, command: Sequence[str], log_path: Path, env: Mapping[str, str], cwd: Path) -> ChildProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            list(command), cwd=cwd, env=dict(env), stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        stream.close()
        raise
    return ChildProcess(role, worker_id, process, stream)


def _health(port: int, expected: Mapping[str, Any], timeout: float, child: ChildProcess) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline and not _CANCEL_REQUESTED:
        returncode = child.process.poll()
        if returncode is not None:
            raise ServerError(f"Huge server {child.worker_id} exited during startup with {returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
                payload = json.loads(response.read())
            for key, value in expected.items():
                matches = (
                    _normalize_uuid(str(payload.get(key, ""))) == _normalize_uuid(str(value))
                    if key == "cuda_uuid" else payload.get(key) == value
                )
                if not matches:
                    raise ValueError(f"health {key}={payload.get(key)!r}, expected {value!r}")
            return
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise ServerError(f"Huge server health identity check failed: {last_error}")


def _terminate_children(children: Sequence[ChildProcess], grace: float) -> None:
    for child in children:
        if child.process.poll() is None:
            try:
                os.killpg(child.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(child.process.poll() is None for child in children):
        time.sleep(0.1)
    for child in children:
        if child.process.poll() is None:
            try:
                os.killpg(child.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for child in children:
        try:
            child.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        child.log_stream.close()


def _monitor(
    shadows: Sequence[ChildProcess], servers: Sequence[ChildProcess], shms: Sequence[OwnedShm],
    attempt: Path, run_id: str,
) -> int:
    while not _CANCEL_REQUESTED:
        for shm in shms:
            shm.observe()
        for server in servers:
            returncode = server.process.poll()
            if returncode is not None:
                raise ServerError(f"Huge server {server.worker_id} exited early with {returncode}")
            status_path = attempt / "status" / f"server-{server.worker_id}.json"
            if status_path.is_file():
                status = _json_object(status_path, label="Huge server status")
                if status.get("run_id") != run_id or status.get("worker_id") != server.worker_id:
                    raise SimulationError(f"server status identity mismatch for {server.worker_id}")
                if status.get("state") == "fatal" or status.get("fatal_error_code"):
                    raise ServerError(
                        f"Huge server {server.worker_id} entered fatal state: "
                        f"{status.get('fatal_error_code', 'unknown')}"
                    )
        for shadow in shadows:
            status_path = attempt / "status" / f"worker-{shadow.worker_id}.json"
            if not status_path.is_file():
                continue
            status = _json_object(status_path, label="shadow worker status")
            if status.get("run_id") != run_id or status.get("worker_id") != shadow.worker_id:
                raise SimulationError(f"worker status identity mismatch for {shadow.worker_id}")
            state = str(status.get("state", ""))
            if state == "server_or_infer_fatal":
                raise ServerError(f"shadow {shadow.worker_id} classified server/inference failure")
            if state in {"sim_runtime_fatal", "integrity_failure", "interrupted"}:
                raise SimulationError(f"shadow {shadow.worker_id} entered {state}")
        returncodes = [shadow.process.poll() for shadow in shadows]
        if all(code is not None for code in returncodes):
            if any(code == 40 for code in returncodes):
                raise ServerError("shadow classified a server/inference fatal error")
            if any(code != 0 for code in returncodes):
                raise SimulationError(f"shadow workers failed: {returncodes}")
            return 0
        for shadow, code in zip(shadows, returncodes):
            if code == 40:
                raise ServerError(f"shadow {shadow.worker_id} classified server/inference failure")
            if code is not None and code != 0:
                raise SimulationError(f"shadow {shadow.worker_id} exited with {code}")
        time.sleep(0.5)
    return EXIT_INTERRUPTED


class ContractLock:
    def __init__(self, path: Path, contract_hash: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.stream.seek(0)
            owner = self.stream.read().strip() or "unknown owner"
            self.stream.close()
            raise ConfigError(f"evaluation contract is already locked: {owner}") from exc
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(json.dumps({
            "host": platform.node(), "pid": os.getpid(), "contract_hash": contract_hash,
            "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()

    def __enter__(self) -> "ContractLock":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _write_manifest(attempt: Path, contract_hash: str, state: str, **runtime: Any) -> None:
    existing: dict[str, Any] = {}
    path = attempt / "manifest.json"
    if path.is_file():
        existing = _json_object(path, label="attempt manifest")
    existing.update({
        "schema_version": MANIFEST_SCHEMA_VERSION, "contract_hash": contract_hash,
        "run_id": attempt.name, "state": state, "runtime": runtime,
    })
    atomic_write_json(path, existing)


def _finalize(attempt: Path, jobs: Sequence[EpisodeJob], contract_hash: str, source_digest: str, reused: int) -> None:
    finalizing = attempt / "_FINALIZING"
    atomic_write_text(finalizing, "\n")
    episodes: list[dict[str, Any]] = []
    for job in jobs:
        episode = _read_existing_episode(attempt, job) or validate_and_publish_completion(attempt, job)
        if episode is None:
            raise SimulationError(f"missing or invalid final episode result: {job.result_identity}")
        episodes.append(episode)
    if len({episode["result_identity"] for episode in episodes}) != len(jobs):
        raise SimulationError("final episode identity set is not unique")
    successes = sum(bool(episode["payload"].get("success")) for episode in episodes)
    task_failures = len(episodes) - successes
    steps = sorted(int(episode["payload"].get("episode_steps", 0)) for episode in episodes)
    durations = sorted(float(episode["payload"].get("duration_sec", 0.0)) for episode in episodes)
    percentile = lambda values, fraction: values[min(len(values) - 1, int(round((len(values) - 1) * fraction)))]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION, "contract_hash": contract_hash,
        "source_inventory_digest": source_digest, "expected": len(jobs), "completed": len(episodes),
        "task_success": successes, "task_failure": task_failures, "infrastructure_failure": 0,
        "success_rate": {"numerator": successes, "denominator": len(episodes)}, "resume_reused": reused,
        "by_seed": {
            str(seed): {
                "completed": sum(episode["job"]["seed"] == seed for episode in episodes),
                "success": sum(episode["job"]["seed"] == seed and bool(episode["payload"].get("success")) for episode in episodes),
            }
            for seed in sorted({job.seed for job in jobs})
        },
        "by_repeat": {
            str(repeat): {
                "completed": sum(episode["job"]["repeat_idx"] == repeat for episode in episodes),
                "success": sum(episode["job"]["repeat_idx"] == repeat and bool(episode["payload"].get("success")) for episode in episodes),
            }
            for repeat in sorted({job.repeat_idx for job in jobs})
        },
        "by_worker": {
            worker: {
                "completed": sum(str(episode.get("worker_id", "")) == worker for episode in episodes),
                "success": sum(str(episode.get("worker_id", "")) == worker and bool(episode["payload"].get("success")) for episode in episodes),
            }
            for worker in sorted({str(episode.get("worker_id", "")) for episode in episodes})
        },
        "episode_steps": {
            "mean": sum(steps) / len(steps), "p50": percentile(steps, 0.5), "p95": percentile(steps, 0.95),
        },
        "duration_sec": {
            "mean": sum(durations) / len(durations), "p50": percentile(durations, 0.5),
            "p95": percentile(durations, 0.95),
        },
        "infrastructure_errors": [],
    }
    atomic_write_json(attempt / "summary.json", summary)
    rows = ["metric,value", f"expected,{len(jobs)}", f"completed,{len(episodes)}", f"task_success,{successes}", f"task_failure,{task_failures}", f"resume_reused,{reused}"]
    atomic_write_text(attempt / "summary.csv", "\n".join(rows) + "\n")
    _write_manifest(attempt, contract_hash, "success", completed=len(jobs), reused=reused)
    atomic_write_text(attempt / "_SUCCESS", canonical_sha256(summary) + "\n")
    finalizing.unlink()
    _fsync_directory(attempt)


def _execute(
    args: argparse.Namespace, task: TaskSpec, checkpoint: CheckpointContract, policy: Mapping[str, Any],
    jobs: Sequence[EpisodeJob], source_inventory: Mapping[str, Any], static_plan: Mapping[str, Any],
) -> int:
    gpus, driver = discover_visible_gpus(policy)
    host = host_fingerprint()
    isaac_version, kit_build = isaac_fingerprint(args.isaaclab_python, args.isaaclab_root)
    output_root = Path(str(policy["output_root"]))
    map_path = Path(str(policy["kit_gpu_map_file"]))
    if not map_path.is_file():
        if not policy["allow_kit_map_bootstrap"]:
            raise GpuError(f"trusted Kit GPU map is missing: {map_path}")
        bootstrap_kit_gpu_map(map_path, gpus, host, driver, isaac_version, kit_build, args)
    physical_bindings, kit_digest = load_kit_gpu_map(map_path, gpus, host, driver, isaac_version, kit_build)
    worker_bindings = expand_worker_bindings(physical_bindings, int(policy["workers_per_gpu"]))
    executable = build_executable_contract(static_plan, worker_bindings, kit_digest)
    contract_hash = executable["contract_hash"]
    checkpoint_label = f"{checkpoint.checkpoint_task}-{checkpoint.checkpoint_dir.name}"
    contract_dir = output_root / task.cli_task / checkpoint_label / contract_hash[:16]
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / "contract.json"
    contract_bytes = canonical_json_bytes(executable["contract"])
    if contract_path.is_file() and canonical_json_bytes(_json_object(contract_path, label="executable contract")) != contract_bytes:
        raise ConfigError("existing contract bytes do not match the newly computed contract")
    if not contract_path.exists():
        atomic_write_json(contract_path, executable["contract"])
        atomic_write_json(contract_dir / "source_inventory.json", source_inventory)
    else:
        stored_inventory = _json_object(contract_dir / "source_inventory.json", label="contract source inventory")
        if canonical_json_bytes(stored_inventory) != canonical_json_bytes(source_inventory):
            raise ConfigError("stored source inventory does not match the executable contract")
    with ContractLock(contract_dir / "contract.lock", contract_hash):
        attempt = select_or_create_attempt(
            contract_dir / "attempts", contract_hash, resume=bool(policy["resume"]), force_new=bool(policy["force_new"])
        )
        run_id = attempt.name
        completed = {
            job.result_identity: (
                _read_existing_episode(attempt, job) or validate_and_publish_completion(attempt, job)
            )
            for job in jobs
        }
        reused = sum(value is not None for value in completed.values())
        if (attempt / "_SUCCESS").is_file():
            _verify_success_attempt(
                attempt, jobs, contract_hash, str(source_inventory["source_inventory_digest"])
            )
            return 0
        pending = [job for job in jobs if completed[job.result_identity] is None]
        shards = partition_pending_jobs(pending, len(worker_bindings))
        if not shards:
            _finalize(attempt, jobs, contract_hash, str(source_inventory["source_inventory_digest"]), reused)
            return 0
        active_bindings = worker_bindings[: len(shards)]
        reservations = _reserve_ports(int(policy["base_port"]), len(shards))
        children: list[ChildProcess] = []
        servers: list[ChildProcess] = []
        shadows: list[ChildProcess] = []
        shms: list[OwnedShm] = []
        runtime_env = _runtime_environment(task, static_plan["yaml"], policy)
        exit_code = EXIT_SIM
        try:
            _write_manifest(attempt, contract_hash, "starting", bindings=[asdict(binding) for binding in active_bindings])
            atomic_write_json(attempt / "plan.json", executable["contract"])
            worker_rows = []
            for index, (binding, shard, reservation) in enumerate(zip(active_bindings, shards, reservations)):
                worker_id = f"w{index:02d}"
                port = reservation.getsockname()[1]
                shm = _make_shm(run_id, worker_id)
                shms.append(shm)
                shard_path = attempt / "shards" / f"worker-{worker_id}.json"
                _write_shard(shard_path, shard, attempt, checkpoint, policy, worker_id)
                server_command = _server_command(args, task, checkpoint, binding, run_id, worker_id, attempt, port)
                reservation.close()
                server = _launch_child(
                    "server", worker_id, server_command, attempt / "logs" / f"server-{worker_id}.log",
                    _server_environment(args.repo, binding.gpu.cuda_uuid), args.repo,
                )
                servers.append(server)
                children.append(server)
                expected_health = {
                    "status": "ready", "protocol_version": PROTOCOL_VERSION,
                    "checkpoint_sha256": checkpoint.checkpoint_sha256, "task": task.cli_task,
                    "cuda_uuid": binding.gpu.cuda_uuid, "state_dim": 34, "action_dim": 66,
                    "motion_token_dim": 64, "action_horizon": 50, "control_hz": 50,
                }
                _health(port, expected_health, float(policy["server_ready_timeout"]), server)
                shadow_env = _isaac_environment(args, run_id=run_id, worker_id=worker_id, gpu_uuid=binding.gpu.cuda_uuid)
                shadow_env.update(runtime_env)
                shadow_env["ISAAC_MULTI_IMAGE_SHM_NAME"] = shm.name
                shadow_command = _shadow_command(
                    args, task, checkpoint, binding, run_id, worker_id, attempt, shard_path, shard,
                    port, shm, policy, static_plan["yaml"],
                )
                shadow = _launch_child(
                    "shadow", worker_id, shadow_command, attempt / "logs" / f"sim-{worker_id}.log",
                    shadow_env, attempt,
                )
                shadows.append(shadow)
                children.append(shadow)
                worker_rows.append({"worker_id": worker_id, "port": port, "shm": shm.name, "binding": asdict(binding)})
            _write_manifest(attempt, contract_hash, "running", workers=worker_rows)
            exit_code = _monitor(shadows, servers, shms, attempt, run_id)
            if exit_code == EXIT_INTERRUPTED:
                return exit_code
        finally:
            for reservation in reservations:
                reservation.close()
            _terminate_children(children, float(policy["shutdown_grace_sec"]))
            unsafe_shm = [shm.name for shm in shms if not shm.cleanup()]
            if unsafe_shm and exit_code == 0:
                raise SimulationError(f"SHM cleanup identity failed for: {unsafe_shm}")
        for job in pending:
            if validate_and_publish_completion(attempt, job) is None:
                raise SimulationError(f"worker completed without a valid completion envelope: {job.result_identity}")
        fresh_inventory = build_source_inventory(
            args.repo, args.humanoidarena_root, Path(static_plan["yaml"]["path"]), checkpoint
        )
        if fresh_inventory["source_inventory_digest"] != source_inventory["source_inventory_digest"]:
            raise ConfigError("source inventory changed during evaluation; refusing finalization")
        _finalize(attempt, jobs, contract_hash, str(source_inventory["source_inventory_digest"]), reused)
        return 0


def _signal_handler(_signum: int, _frame: Any) -> None:
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _resolve_cli(build_parser().parse_args(argv))
        task, checkpoint, policy, jobs, source_inventory, static_plan = _static_preflight(args)
        static_hash = canonical_sha256(static_plan)
        if policy["dry_run"]:
            print(json.dumps({
                "schema_version": STATIC_PLAN_SCHEMA_VERSION, "mode": "dry_run",
                "static_plan_hash": static_hash, "static_plan": static_plan,
            }, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            return 0
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        return _execute(args, task, checkpoint, policy, jobs, source_inventory, static_plan)
    except ConfigError as exc:
        print(f"[eval_orchestrator] configuration error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except GpuError as exc:
        print(f"[eval_orchestrator] GPU binding error: {exc}", file=sys.stderr)
        return EXIT_GPU
    except ServerError as exc:
        print(f"[eval_orchestrator] Huge server error: {exc}", file=sys.stderr)
        return EXIT_SERVER
    except SimulationError as exc:
        print(f"[eval_orchestrator] simulation/result error: {exc}", file=sys.stderr)
        return EXIT_SIM
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as exc:
        print(f"[eval_orchestrator] unexpected infrastructure error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_SIM


if __name__ == "__main__":
    raise SystemExit(main())

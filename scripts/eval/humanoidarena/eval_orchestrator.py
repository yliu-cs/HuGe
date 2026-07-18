#!/usr/bin/env python3
"""Huge-owned, read-only HumanoidArena SONIC evaluation orchestration.

HumanoidArena is treated as an immutable runtime dependency.  Static dry-run
mode deliberately stops before CUDA discovery, Kit map access, sockets, locks,
model loading, or SimulationApp creation.
"""

from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures
import copy
import ctypes
from dataclasses import asdict, dataclass
import errno
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
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
RETRY_ATTEMPT_SCHEMA_VERSION = "huge_humanoidarena_retry_attempt_v1"
ROOT_SOFTWARE_SCHEMA_VERSION = "huge_root_software_v1"
ROOT_SOFTWARE_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "safetensors",
    "numpy",
    "scipy",
    "Pillow",
    "qwen-vl-utils",
    "flash-attn",
    "peft",
    "onnxruntime-gpu",
    "opencv-python",
    "psutil",
)
GPU_ADMISSION_SCHEMA_VERSION = "huge_ha_gpu_admission_v2"
KIT_MAP_SCHEMA_VERSION = "huge_kit_gpu_map_v1"
WORKER_STATUS_SCHEMA_VERSION = "huge_ha_worker_status_v1"
MODEL_IO_TRACE_SCHEMA_VERSION = "huge_humanoidarena_model_io_trace_v1"
MODEL_IO_RECEIPT_SCHEMA_VERSION = "huge_humanoidarena_model_io_receipt_v1"
MODEL_IO_TRACE_POLICY = "require_complete_v1"
CHECKPOINT_MANIFEST_VERSION = 1
PORTABLE_CHECKPOINT_KIND = "huge_portable_model"
EPISODE_SEED_FORMULA = "sha256(task_name|group_seed|repeat_idx)_positive_int32"
MODEL_NOISE_SEED_ALGORITHM = "huge_ha_model_noise_v1"
ALLOWED_TRAINING_ROOT_NAMES = (
    "stage3_humanoidarena_full_optimized",
    "stage3_humanoidarena",
)
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
_ATTEMPT_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[a-f0-9]{8}$")
_WORKER_RE = re.compile(r"^w[0-9]{2}$")
_SHM_RE = re.compile(r"^huge_ha_([a-z0-9]{8})_(w[0-9]{2})_([a-f0-9]{16,32})$")
_SHADOW_SESSION_RE = re.compile(r"^[a-f0-9]{16,32}$")
_LEGAL_TASK_OUTCOMES = {"success", "fall", "timeout", "max_steps", "sim_stopped"}
_EXECUTION_FAILED = "execution_failed"
_COMPLETED_WITH_FAILURES_MARKER = "_COMPLETED_WITH_FAILURES"
_SHADOW_TERMINAL_EXIT_CODES = {
    "complete": 0,
    "server_or_infer_fatal": 40,
    "sim_runtime_fatal": 50,
    "integrity_failure": 50,
    "interrupted": 130,
}
_SHADOW_GUARD_REMOVAL_SCHEMA_VERSION = "huge_ha_guard_removal_v1"
_GPU_ADMISSION_FILENAME_RE = re.compile(r"^admission-([0-9]{4})\.json$")
_CANCEL_REQUESTED = False


class ConfigError(RuntimeError):
    pass


class GpuError(RuntimeError):
    pass


class ServerError(RuntimeError):
    pass


class SimulationError(RuntimeError):
    pass


class UnsafeCleanupError(SimulationError):
    pass


class EpisodeAttemptError(SimulationError):
    def __init__(self, job: EpisodeJob, attempt_number: int, message: str) -> None:
        super().__init__(message)
        self.job = job
        self.attempt_number = attempt_number


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
    process_start_ticks: int | None = None
    command_sha256: str = ""


@dataclass
class WorkerCleanup:
    future: concurrent.futures.Future[None]
    shadow: ChildProcess | None
    server: ChildProcess | None
    job: EpisodeJob | None
    attempt_number: int | None
    reason: str
    error: str
    consume_attempt: bool


@dataclass
class WorkerRuntime:
    worker_id: str
    binding: GpuBinding
    server: ChildProcess | None
    port: int
    pending: deque[EpisodeJob]
    active_shadow: ChildProcess | None = None
    active_job: EpisodeJob | None = None
    active_shm: OwnedShm | None = None
    active_session_id: str | None = None
    active_status_path: Path | None = None
    active_log_path: Path | None = None
    active_started_monotonic: float | None = None
    active_binding_verified: bool = False
    active_attempt_number: int | None = None
    prepared_attempt_number: int | None = None
    server_starting_attempt_number: int | None = None
    server_started_monotonic: float | None = None
    server_started_wall_time: float | None = None
    retry_not_before_monotonic: float = 0.0
    cleanup: WorkerCleanup | None = None


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
            raise UnsafeCleanupError(f"owned SHM appeared with unsafe identity: {self.path}")
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


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without ever replacing a destination."""
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise ConfigError("Linux renameat2(RENAME_NOREPLACE) is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ConfigError(f"evaluation attempt already exists: {destination}")
    raise OSError(
        error_number,
        f"renameat2(RENAME_NOREPLACE) failed for {source} -> {destination}",
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _require_owned_attempt_path(
    attempt: Path,
    path: Path,
    *,
    label: str,
    directory: bool,
) -> Path:
    """Require a real, owned attempt child while allowing attempt itself to be a symlink."""
    try:
        attempt_root = attempt.resolve(strict=True)
        root_info = attempt_root.lstat()
        relative = path.relative_to(attempt)
        if (
            root_info.st_uid != os.getuid()
            or stat.S_ISLNK(root_info.st_mode)
            or not stat.S_ISDIR(root_info.st_mode)
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ConfigError(f"{label} has an unsafe attempt root or relative path")
        cursor = attempt
        for index, part in enumerate(relative.parts):
            cursor = cursor / part
            info = cursor.lstat()
            expect_directory = index < len(relative.parts) - 1 or directory
            if (
                info.st_uid != os.getuid()
                or stat.S_ISLNK(info.st_mode)
                or (
                    expect_directory
                    and not stat.S_ISDIR(info.st_mode)
                )
                or (
                    not expect_directory
                    and not stat.S_ISREG(info.st_mode)
                )
            ):
                raise ConfigError(f"{label} must be an owned real path: {path}")
        resolved = path.resolve(strict=True)
    except ConfigError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigError(f"cannot validate {label}: {path}: {exc}") from exc
    if not _is_relative_to(resolved, attempt_root):
        raise ConfigError(f"{label} escapes the attempt: {path}")
    return path


def _validate_canonical_attempt_entry(path: Path, attempts_root: Path) -> Path:
    """Validate one canonical attempt entry without following an unsafe link."""
    if _ATTEMPT_ID_RE.fullmatch(path.name) is None:
        raise ConfigError(f"attempt name is not canonical: {path.name}")
    try:
        entry_stat = path.lstat()
    except OSError as exc:
        raise ConfigError(f"cannot inspect attempt entry: {path}: {exc}") from exc
    if entry_stat.st_uid != os.getuid():
        raise ConfigError(f"attempt entry is not owned by the current user: {path}")
    if stat.S_ISDIR(entry_stat.st_mode):
        return path
    if not stat.S_ISLNK(entry_stat.st_mode):
        raise ConfigError(f"attempt entry must be a directory or safe symlink: {path}")
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise ConfigError(f"cannot read attempt symlink: {path}: {exc}") from exc
    target_path = Path(target)
    backing_pattern = re.compile(
        rf"^\.{re.escape(path.name)}\.object\.[0-9]+\.[0-9]+\.[a-f0-9]{{16}}$"
    )
    if (
        target_path.is_absolute()
        or len(target_path.parts) != 1
        or backing_pattern.fullmatch(target) is None
    ):
        raise ConfigError(f"attempt symlink target is unsafe: {path} -> {target!r}")
    backing = attempts_root / target
    try:
        backing_stat = backing.lstat()
        root_resolved = attempts_root.resolve(strict=True)
        backing_resolved = backing.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"attempt backing is unavailable: {path} -> {target!r}: {exc}") from exc
    if (
        backing_stat.st_uid != os.getuid()
        or stat.S_ISLNK(backing_stat.st_mode)
        or not stat.S_ISDIR(backing_stat.st_mode)
        or backing_resolved.parent != root_resolved
    ):
        raise ConfigError(f"attempt backing is unsafe: {path} -> {target!r}")
    return path


def _canonical_attempt_candidates(attempts_root: Path) -> list[Path]:
    candidates: list[tuple[int, str, Path]] = []
    for path in attempts_root.iterdir():
        if _ATTEMPT_ID_RE.fullmatch(path.name) is None:
            continue
        entry_stat = path.lstat()
        candidates.append((entry_stat.st_mtime_ns, path.name, path))
    return [entry[2] for entry in sorted(candidates)]


def _visible_attempt_targets_backing(attempt: Path, backing_name: str) -> bool:
    try:
        return stat.S_ISLNK(attempt.lstat().st_mode) and os.readlink(attempt) == backing_name
    except OSError:
        return False


def _json_object(
    path: Path,
    *,
    label: str,
    attempt: Path | None = None,
) -> dict[str, Any]:
    if attempt is not None:
        _require_owned_attempt_path(
            attempt,
            path,
            label=label,
            directory=False,
        )
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

    workers_per_gpu = _parse_int(
        "HUGE_EVAL_WORKERS_PER_GPU",
        environ.get(
            "HUGE_EVAL_WORKERS_PER_GPU",
            str(DEFAULT_WORKERS_PER_GPU),
        ),
        minimum=1,
        maximum=MAX_WORKERS_PER_GPU,
    )
    max_concurrent_shadow_startups = _parse_int(
        "HUGE_EVAL_MAX_CONCURRENT_SHADOW_STARTUPS",
        environ.get("HUGE_EVAL_MAX_CONCURRENT_SHADOW_STARTUPS", "2"),
        minimum=1,
    )
    min_free_raw = environ.get("HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB", "0")
    min_free_gpu_memory_gib = _parse_float(
        "HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB",
        min_free_raw,
        minimum=0.0,
    )
    if (
        workers_per_gpu == 2
        and (
            "HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB" not in environ
            or not str(min_free_raw).strip()
            or min_free_gpu_memory_gib <= 0.0
        )
    ):
        raise ConfigError(
            "WPG2 requires an explicit positive "
            "HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB before GPU discovery"
        )

    resume = _parse_bool(
        "HUGE_EVAL_RESUME", environ.get("HUGE_EVAL_RESUME", "1")
    )
    force_new = _parse_bool(
        "HUGE_EVAL_FORCE_NEW_RUN",
        environ.get("HUGE_EVAL_FORCE_NEW_RUN", "0"),
    )
    retry_required_contract_hash = environ.get(
        "HUGE_EVAL_RETRY_REQUIRED_CONTRACT_HASH", ""
    ).strip()
    if (
        retry_required_contract_hash
        and re.fullmatch(r"[0-9a-f]{64}", retry_required_contract_hash) is None
    ):
        raise ConfigError(
            "HUGE_EVAL_RETRY_REQUIRED_CONTRACT_HASH must be a full lowercase SHA256"
        )
    if retry_required_contract_hash and not (resume and force_new):
        raise ConfigError(
            "controlled retry requires HUGE_EVAL_RESUME=1 and "
            "HUGE_EVAL_FORCE_NEW_RUN=1"
        )
    trace_policy = environ.get(
        "HUGE_EVAL_MODEL_IO_TRACE_POLICY", MODEL_IO_TRACE_POLICY
    )
    if trace_policy != MODEL_IO_TRACE_POLICY:
        raise ConfigError(
            "HUGE_EVAL_MODEL_IO_TRACE_POLICY is frozen to "
            f"{MODEL_IO_TRACE_POLICY}"
        )
    return {
        "test_mode": test_mode,
        "effective_persistent_sim": False,
        "simulator_process_scope": "per_episode_v1",
        "model_io_schema": MODEL_IO_TRACE_SCHEMA_VERSION,
        "model_io_trace_policy": trace_policy,
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
        "shadow_startup_timeout_sec": _parse_float(
            "HUGE_EVAL_SHADOW_STARTUP_TIMEOUT_SEC",
            environ.get("HUGE_EVAL_SHADOW_STARTUP_TIMEOUT_SEC", "180"),
            minimum=0.001,
        ),
        "episode_timeout_sec": _parse_float(
            "HUGE_EVAL_EPISODE_TIMEOUT_SEC",
            environ.get("HUGE_EVAL_EPISODE_TIMEOUT_SEC", "900"),
            minimum=0.001,
        ),
        "episode_max_retries": _parse_int(
            "HUGE_EVAL_EPISODE_MAX_RETRIES",
            environ.get("HUGE_EVAL_EPISODE_MAX_RETRIES", "5"),
            minimum=0,
            maximum=100,
        ),
        "retry_backoff_sec": _parse_float(
            "HUGE_EVAL_RETRY_BACKOFF_SEC",
            environ.get("HUGE_EVAL_RETRY_BACKOFF_SEC", "5"),
            minimum=0.0,
        ),
        "min_free_gpu_memory_gib": min_free_gpu_memory_gib,
        "base_port": _parse_int("HUGE_EVAL_BASE_PORT", environ.get("HUGE_EVAL_BASE_PORT", "18080"), minimum=1024, maximum=65535),
        "dry_run": _parse_bool("HUGE_EVAL_DRY_RUN", environ.get("HUGE_EVAL_DRY_RUN", "0")),
        "resume": resume,
        "force_new": force_new,
        "retry_required_contract_hash": retry_required_contract_hash,
        "allow_kit_map_bootstrap": _parse_bool(
            "HUGE_ALLOW_KIT_MAP_BOOTSTRAP", environ.get("HUGE_ALLOW_KIT_MAP_BOOTSTRAP", "0")
        ),
        "gpu_count": raw_count,
        "gpu_logical_ids": raw_ids,
        "workers_per_gpu": workers_per_gpu,
        "max_concurrent_shadow_startups": max_concurrent_shadow_startups,
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


def _allowed_checkpoint_root(
    repo: Path,
    checkpoint: Path,
    checkpoint_task: str,
) -> Path:
    checkpoint = checkpoint.expanduser().resolve()
    matched_base: Path | None = None
    relative: Path | None = None
    for name in ALLOWED_TRAINING_ROOT_NAMES:
        base = (repo / ".runs" / name).resolve()
        if _is_relative_to(checkpoint, base):
            matched_base = base
            relative = checkpoint.relative_to(base)
            break
    if matched_base is None or relative is None:
        allowed = ", ".join(
            str((repo / ".runs" / name).resolve())
            for name in ALLOWED_TRAINING_ROOT_NAMES
        )
        raise ConfigError(
            f"checkpoint must resolve inside an allowed training root "
            f"({allowed}): {checkpoint}"
        )

    architectures = {"qwenpi_v3", "qwengr00t", "qwenoft"}
    tuning_modes = {"full", "lora", "frozen_vlm"}
    if (
        not _SAFE_COMPONENT_RE.fullmatch(checkpoint_task)
        or checkpoint_task in {".", ".."}
        or len(relative.parts) != 6
        or relative.parts[0] != checkpoint_task
        or relative.parts[1] not in architectures
        or relative.parts[2] not in tuning_modes
        or relative.parts[3] != "stage3"
        or not _SAFE_COMPONENT_RE.fullmatch(relative.parts[4])
        or relative.parts[4] in {".", "..", "_resume_states"}
        or re.fullmatch(r"step_[0-9]+", relative.parts[5]) is None
    ):
        raise ConfigError(
            "checkpoint must have the exact "
            "<task>/<architecture>/<tuning_mode>/stage3/<run_id>/step_N "
            f"shape for task {checkpoint_task}: {checkpoint}"
        )
    cursor = matched_base
    for component in relative.parts[:-1]:
        cursor = cursor / component
        try:
            info = cursor.lstat()
        except OSError as exc:
            raise ConfigError(
                f"checkpoint run path is unavailable: {cursor}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ConfigError(
                f"checkpoint run path must contain only real directories: "
                f"{cursor}"
            )
    return matched_base


def validate_checkpoint(checkpoint: str | Path, repo_root: str | Path, task: TaskSpec) -> CheckpointContract:
    repo = Path(repo_root).resolve()
    raw = Path(checkpoint).expanduser()
    candidate = (repo / raw).resolve() if not raw.is_absolute() else raw.resolve()
    allowed_root = _allowed_checkpoint_root(
        repo, candidate, task.checkpoint_task
    )
    try:
        return resolve_checkpoint_contract(
            candidate,
            allowed_root=allowed_root,
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
        "repeats_per_seed": repeats, "source_persistent_sim": True,
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


def partition_remaining_jobs(
    jobs: Sequence[EpisodeJob],
    worker_count: int,
    completed_result_identities: set[str],
) -> list[tuple[int, list[EpisodeJob]]]:
    shards = partition_pending_jobs(jobs, worker_count)
    return [
        (
            worker_index,
            [
                job
                for job in shard
                if job.result_identity not in completed_result_identities
            ],
        )
        for worker_index, shard in enumerate(shards)
        if any(
            job.result_identity not in completed_result_identities
            for job in shard
        )
    ]


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


def _portable_checkpoint_publication(
    checkpoint: Path,
    *,
    required: bool,
) -> dict[str, Any] | None:
    """Validate and bind the complete portable publication manifest."""
    root = checkpoint.resolve()
    manifest_path = checkpoint / "manifest.json"
    marker_path = checkpoint / "_SUCCESS"
    if not manifest_path.exists() and not marker_path.exists() and not required:
        return None
    for path, label in (
        (manifest_path, "portable checkpoint manifest"),
        (marker_path, "portable checkpoint success marker"),
    ):
        try:
            info = path.lstat()
        except OSError as exc:
            raise ConfigError(f"missing {label}: {path}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ConfigError(f"{label} must be a real file: {path}")
    manifest = _json_object(manifest_path, label="portable checkpoint manifest")
    marker = _json_object(marker_path, label="portable checkpoint success marker")
    manifest_sha256 = sha256_file(manifest_path)
    if (
        set(marker) != {"schema_version", "kind", "manifest_sha256"}
        or marker.get("schema_version") != CHECKPOINT_MANIFEST_VERSION
        or marker.get("kind") != PORTABLE_CHECKPOINT_KIND
        or marker.get("manifest_sha256") != manifest_sha256
        or manifest.get("schema_version") != CHECKPOINT_MANIFEST_VERSION
        or manifest.get("kind") != PORTABLE_CHECKPOINT_KIND
    ):
        raise ConfigError(
            f"portable checkpoint marker/manifest contract mismatch: {checkpoint}"
        )
    publication_metadata = manifest.get("metadata")
    publication_identity = (
        publication_metadata if isinstance(publication_metadata, dict) else manifest
    )
    step_match = re.fullmatch(r"step_([0-9]+)", checkpoint.name)
    if required and (
        step_match is None
        or publication_identity.get("global_step") != int(step_match.group(1))
        or publication_identity.get("model_architecture")
        != checkpoint.parents[3].name
        or publication_identity.get("tuning_mode") != "full"
        or publication_identity.get("run_id") != checkpoint.parent.name
    ):
        raise ConfigError(
            f"portable checkpoint manifest identity mismatch: {checkpoint}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(
            f"portable checkpoint manifest has no file inventory: {manifest_path}"
        )
    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ConfigError(f"invalid portable checkpoint manifest entry: {entry!r}")
        relative_raw = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        relative = Path(relative_raw) if isinstance(relative_raw, str) else Path()
        if (
            not isinstance(relative_raw, str)
            or not relative_raw
            or relative_raw in expected_paths
            or relative_raw in {"manifest.json", "_SUCCESS"}
            or relative.is_absolute()
            or relative.as_posix() != relative_raw
            or any(part in {"", ".", ".."} for part in relative.parts)
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ConfigError(f"invalid portable checkpoint manifest entry: {entry!r}")
        file_path = checkpoint / relative
        try:
            info = file_path.lstat()
            resolved = file_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ConfigError(
                f"portable checkpoint manifest file is unavailable: {file_path}: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or not _is_relative_to(resolved, root)
            or info.st_size != size
            or sha256_file(file_path) != digest
        ):
            raise ConfigError(
                f"portable checkpoint manifest digest/size mismatch: {file_path}"
            )
        expected_paths.add(relative_raw)
    actual_paths: set[str] = set()
    for entry in checkpoint.rglob("*"):
        relative = entry.relative_to(checkpoint).as_posix()
        if relative in {"manifest.json", "_SUCCESS"}:
            continue
        info = entry.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ConfigError(f"portable checkpoint may not contain symlinks: {entry}")
        if stat.S_ISREG(info.st_mode):
            actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise ConfigError(
            "portable checkpoint file inventory mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}, checkpoint={checkpoint}"
        )
    return {
        "schema_version": CHECKPOINT_MANIFEST_VERSION,
        "kind": PORTABLE_CHECKPOINT_KIND,
        "manifest_relative_path": "manifest.json",
        "manifest_sha256": manifest_sha256,
        "success_marker_relative_path": "_SUCCESS",
        "success_marker_sha256": sha256_file(marker_path),
        "file_count": len(entries),
    }


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
        "src/huge/inference/humanoidarena_io_trace.py",
        "src/huge/inference/humanoidarena_http_server.py",
        "src/huge/inference/policy.py",
        "src/huge/inference/observation.py",
        "src/huge/inference/sonic_adapter.py",
        "src/huge/data/state_normalizer.py",
        "src/huge/data/transforms.py",
        "src/huge/models/qwen_sonicfm.py",
        "src/huge/data/humanoid_arena_preprocess.py",
        "src/huge/data/schemas.py",
        "src/huge/models/action_heads/__init__.py",
        "src/huge/models/action_heads/hand_binary_head.py",
        "src/huge/models/action_heads/layerwise_cross_dit_fm.py",
        "src/huge/models/action_heads/starvla_gr00t_head.py",
        "src/huge/models/action_heads/starvla_oft_head.py",
        "src/huge/models/action_heads/starvla_pi_head.py",
        "scripts/eval/humanoidarena/eval_orchestrator.py",
        "scripts/eval/run_ha_eval.sh",
        "scripts/eval/humanoidarena/watch_eval_summary.py",
        "scripts/eval/humanoidarena/shard_checkpoints.py",
        "scripts/eval/humanoidarena/sim_eval_shadow.py",
        "scripts/eval/humanoidarena/run_eval.sh",
        "scripts/eval/humanoidarena/env.sh",
        "pyproject.toml",
        "uv.lock",
    )
    huge: dict[str, str] = {}
    for relative in huge_paths:
        source_path = repo / relative
        if not source_path.is_file():
            raise ConfigError(f"required Huge evaluation source is missing: {source_path}")
        huge[relative] = sha256_file(source_path)
    encoder = repo / "resource/model/sonic/model_encoder.onnx"
    decoder = repo / "resource/model/sonic/model_decoder.onnx"
    _verify_hash(encoder, SONIC_ENCODER_SHA256, label="Huge SONIC encoder")
    _verify_hash(decoder, SONIC_DECODER_SHA256, label="Huge SONIC decoder")
    upstream_encoder = humanoidarena / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
    upstream_decoder = humanoidarena / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
    _verify_hash(upstream_encoder, SONIC_ENCODER_SHA256, label="HumanoidArena SONIC encoder")
    _verify_hash(upstream_decoder, SONIC_DECODER_SHA256, label="HumanoidArena SONIC decoder")
    qwen_inventory = _directory_inventory(checkpoint.base_vlm_path)
    optimized_root = (repo / ".runs/stage3_humanoidarena_full_optimized").resolve()
    checkpoint_publication = _portable_checkpoint_publication(
        checkpoint.checkpoint_dir,
        required=_is_relative_to(checkpoint.checkpoint_dir, optimized_root),
    )
    payload = {
        "upstream_compatibility": upstream,
        "wrapper_semantics": wrappers,
        "wrapper_semantics_digest": canonical_sha256(list(wrappers.items())),
        "huge_sources": huge,
        "task_yaml": {str(yaml_path): sha256_file(yaml_path)},
        "checkpoint_files": dict(checkpoint.file_sha256),
        "checkpoint_publication": checkpoint_publication,
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


def root_software_contract(repo: Path) -> dict[str, Any]:
    """Bind the evaluator to the repository root interpreter and key packages."""
    expected_prefix = (repo / ".venv").resolve()
    try:
        current_prefix = Path(sys.prefix).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(
            f"cannot resolve the evaluator Python prefix: {exc}"
        ) from exc
    if current_prefix != expected_prefix:
        raise ConfigError(
            "evaluation software fingerprint must run from the Huge root virtual "
            f"environment: observed={current_prefix}, expected={expected_prefix}"
        )
    packages: dict[str, str | None] = {}
    for distribution in ROOT_SOFTWARE_PACKAGES:
        try:
            packages[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "schema_version": ROOT_SOFTWARE_SCHEMA_VERSION,
        "interpreter": ".venv/bin/python",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


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
        "retry_required_contract_hash",
    ):
        safe_policy.pop(execution_only, None)
    source_files = source_inventory.get("files")
    checkpoint_publication = (
        source_files.get("checkpoint_publication")
        if isinstance(source_files, Mapping)
        else None
    )
    return {
        "schema_version": STATIC_PLAN_SCHEMA_VERSION,
        "task": asdict(task),
        "checkpoint": {
            "relative_path": checkpoint.checkpoint_dir.relative_to(repo).as_posix(),
            "step": checkpoint.step, "checkpoint_task": checkpoint.checkpoint_task,
            "checkpoint_sha256": checkpoint.checkpoint_sha256, "files": dict(checkpoint.file_sha256),
            "qwen_path": str(checkpoint.base_vlm_path.relative_to(repo)),
            "model_architecture": str(
                checkpoint.config.get("model_architecture", "qwenpi_v3")
                or "qwenpi_v3"
            ),
            "action_horizon": int(checkpoint.config.get("action_horizon", 50)),
            "num_inference_steps": int(
                checkpoint.config.get("num_inference_steps", 4)
            ),
            "portable_publication": copy.deepcopy(checkpoint_publication),
        },
        "yaml": dict(yaml_config),
        "policy": safe_policy,
        "protocol": {"http": PROTOCOL_VERSION, "mapping": MAPPING_VERSION, "action": [50, 66], "latent": [50, 64]},
        "episode_seed_algorithm": EPISODE_SEED_FORMULA,
        "model_noise_seed_algorithm": MODEL_NOISE_SEED_ALGORITHM,
        "episodes": [asdict(job) for job in jobs],
        "source_inventory": dict(source_inventory),
        "upstream_contract": dict(upstream_contract),
        "software": root_software_contract(repo),
    }


def build_executable_contract(
    static_plan: Mapping[str, Any], bindings: Sequence[Mapping[str, Any] | GpuBinding], kit_map_digest: str
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for binding in bindings:
        row = copy.deepcopy(
            asdict(binding) if isinstance(binding, GpuBinding) else dict(binding)
        )
        gpu = row.get("gpu")
        if isinstance(gpu, Mapping):
            normalized_gpu = dict(gpu)
            normalized_gpu["free_memory_bytes"] = None
            row["gpu"] = normalized_gpu
        normalized.append(row)
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


def discover_visible_gpus(
    policy: Mapping[str, Any],
) -> tuple[list[GpuIdentity], str, float]:
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
        raw_free_memory = str(row.get("memory_free_mib", "")).strip()
        if raw_free_memory.upper() in {"", "N/A", "NA", "NOT SUPPORTED"}:
            free_memory_bytes = None
        else:
            try:
                free_memory_mib = float(raw_free_memory)
            except ValueError as exc:
                raise GpuError(
                    f"invalid free-memory snapshot for {row['uuid']}: {raw_free_memory!r}"
                ) from exc
            if not math.isfinite(free_memory_mib) or free_memory_mib < 0.0:
                raise GpuError(
                    f"invalid free-memory snapshot for {row['uuid']}: {raw_free_memory!r}"
                )
            free_memory_bytes = int(free_memory_mib * 1024 * 1024)
        identities.append(GpuIdentity(
            ordinal, row["uuid"], row["uuid"], row["pci_bus_id"], str(properties.name),
            int(properties.total_memory), f"{properties.major}.{properties.minor}",
            free_memory_bytes,
        ))
    captured_at = float(time.time())
    threshold_raw = policy.get("min_free_gpu_memory_gib", 0.0)
    if isinstance(threshold_raw, bool):
        raise GpuError("GPU free-memory threshold must be a non-negative finite number")
    try:
        threshold_gib = float(threshold_raw)
    except (TypeError, ValueError) as exc:
        raise GpuError("GPU free-memory threshold must be a non-negative finite number") from exc
    if not math.isfinite(threshold_gib) or threshold_gib < 0.0:
        raise GpuError("GPU free-memory threshold must be a non-negative finite number")
    required_bytes = math.ceil(threshold_gib * 1024**3)
    if required_bytes:
        for gpu in identities:
            observed = gpu.free_memory_bytes
            if observed is None or observed < required_bytes:
                observed_text = (
                    "unavailable"
                    if observed is None
                    else f"{observed / 1024**3:.6f} GiB ({observed} bytes)"
                )
                raise GpuError(
                    f"GPU {gpu.cuda_uuid} free-memory admission failed: "
                    f"observed={observed_text}, "
                    f"required={threshold_gib:.6f} GiB ({required_bytes} bytes)"
                )
    return identities, driver, captured_at


def fresh_gpu_admission_snapshot(
    selected_gpus: Sequence[GpuIdentity],
    *,
    threshold_gib: float,
    allowed_pid: int | None = None,
) -> tuple[list[GpuIdentity], float, list[dict[str, Any]]]:
    """Query fresh free memory immediately before child startup."""
    if not selected_gpus:
        raise GpuError("fresh GPU admission requires at least one selected GPU")
    if isinstance(threshold_gib, bool) or not math.isfinite(threshold_gib) or threshold_gib < 0.0:
        raise GpuError("fresh GPU admission threshold must be finite and non-negative")
    allowed_pid = os.getpid() if allowed_pid is None else allowed_pid
    if type(allowed_pid) is not int or allowed_pid < 1:
        raise GpuError("fresh GPU admission allowed PID is invalid")
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise GpuError("nvidia-smi is required for fresh GPU admission")

    selected_by_uuid: dict[str, GpuIdentity] = {}
    for gpu in selected_gpus:
        normalized = _normalize_uuid(gpu.cuda_uuid)
        if normalized in selected_by_uuid:
            raise GpuError("selected GPU UUIDs are ambiguous")
        selected_by_uuid[normalized] = gpu

    memory_command = [
        executable,
        "--query-gpu=uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        memory_result = subprocess.run(
            memory_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuError(f"fresh GPU memory query failed: {exc}") from exc
    memory_rows: dict[str, tuple[str, int, int]] = {}
    for raw_line in memory_result.stdout.splitlines():
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            raise GpuError(f"ambiguous fresh GPU memory row: {raw_line!r}")
        raw_uuid, raw_total_mib, raw_free_mib = parts
        normalized = _normalize_uuid(raw_uuid)
        if normalized in memory_rows:
            raise GpuError(f"duplicate fresh GPU memory UUID: {raw_uuid!r}")
        try:
            total_mib = float(raw_total_mib)
            free_mib = float(raw_free_mib)
        except ValueError as exc:
            raise GpuError(f"invalid fresh GPU memory row: {raw_line!r}") from exc
        if (
            not math.isfinite(total_mib)
            or not math.isfinite(free_mib)
            or total_mib <= 0.0
            or free_mib < 0.0
            or free_mib > total_mib
        ):
            raise GpuError(f"invalid fresh GPU memory row: {raw_line!r}")
        memory_rows[normalized] = (
            raw_uuid,
            int(total_mib * 1024 * 1024),
            int(free_mib * 1024 * 1024),
        )
    if set(selected_by_uuid) - set(memory_rows):
        raise GpuError(
            "fresh GPU memory query omitted selected UUIDs: "
            f"{sorted(set(selected_by_uuid) - set(memory_rows))}"
        )


    required_bytes = math.ceil(threshold_gib * 1024**3)
    refreshed: list[GpuIdentity] = []
    for gpu in selected_gpus:
        normalized = _normalize_uuid(gpu.cuda_uuid)
        raw_uuid, total_bytes, free_bytes = memory_rows[normalized]
        if free_bytes < required_bytes:
            raise GpuError(
                f"GPU {gpu.cuda_uuid} fresh free-memory admission failed: "
                f"observed={free_bytes} bytes, required={required_bytes} bytes"
            )
        refreshed.append(
            GpuIdentity(
                parent_logical_ordinal=gpu.parent_logical_ordinal,
                child_cuda_visible_token=gpu.child_cuda_visible_token,
                cuda_uuid=raw_uuid,
                pci_bus_id=gpu.pci_bus_id,
                device_name=gpu.device_name,
                total_memory_bytes=total_bytes,
                compute_capability=gpu.compute_capability,
                free_memory_bytes=free_bytes,
            )
        )
    compute_apps: list[dict[str, Any]] = []
    return refreshed, float(time.time()), compute_apps


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
    if resume and not force_new:
        try:
            existing = _canonical_attempt_candidates(root)
        except OSError as exc:
            raise ConfigError(f"cannot list evaluation attempts: {root}: {exc}") from exc
        if existing:
            candidate = _validate_canonical_attempt_entry(existing[-1], root)
            manifest_path = candidate / "manifest.json"
            if not os.path.lexists(manifest_path):
                raise ConfigError(f"latest attempt has no manifest: {candidate}")
            manifest = _json_object(
                manifest_path,
                label="latest attempt manifest",
                attempt=candidate,
            )
            if (
                manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
                or manifest.get("contract_hash") != contract_hash
                or manifest.get("run_id") != candidate.name
            ):
                raise ConfigError(f"latest attempt manifest drift detected: {candidate}")
            if manifest.get("state") == "unsafe_cleanup":
                raise ConfigError(
                    "latest attempt has unresolved cleanup failure and cannot be resumed "
                    f"safely: {candidate}"
                )
            return candidate
    timestamp = now or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    random_nonce = nonce or secrets.token_hex(4)
    run_id = f"{timestamp}_{random_nonce}"
    if _ATTEMPT_ID_RE.fullmatch(run_id) is None:
        raise ConfigError("generated attempt id is unsafe")
    attempt = root / run_id
    staging_parent = root.parent / (
        f".{root.name}.{run_id}.staging.{os.getpid()}."
        f"{time.time_ns()}.{secrets.token_hex(8)}"
    )
    staging_attempt = staging_parent / run_id
    staging_parent_created = False
    published = False
    backing = root / (
        f".{run_id}.object.{os.getpid()}.{time.time_ns()}."
        f"{secrets.token_hex(8)}"
    )
    backing_created = False
    operation_error: BaseException | None = None
    try:
        staging_parent.mkdir()
        staging_parent_created = True
        staging_attempt.mkdir()
        staging_attempt.rmdir()
        _fsync_directory(staging_parent)
        backing.mkdir()
        backing_created = True
        directories = tuple(
            backing / relative
            for relative in (
                "logs",
                "status",
                "shards",
                "raw_results",
                "completions",
                "episodes",
                "retry_ledger",
                "gpu_admission",
                ".gpu_admission_staging",
                "videos",
                "videos/success",
                "videos/failure",
                "recordings",
                "model_io",
            )
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        for directory in reversed(directories):
            _fsync_directory(directory)
        atomic_write_json(
            backing / "manifest.json",
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "contract_hash": contract_hash,
                "run_id": run_id,
                "state": "starting",
                "runtime": {"phase": "admission_pending"},
            },
        )
        _fsync_directory(backing)
        _fsync_directory(root)
        try:
            _rename_directory_no_replace(backing, attempt)
            published = True
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.EOPNOTSUPP, errno.ENOSYS}:
                raise
            try:
                os.symlink(backing.name, attempt, target_is_directory=True)
            except FileExistsError as collision:
                raise ConfigError(f"evaluation attempt already exists: {attempt}") from collision
            published = True
        _fsync_directory(root)
        _fsync_directory(staging_parent)
        _fsync_directory(root.parent)
        return attempt
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        if backing_created:
            published = published or _visible_attempt_targets_backing(attempt, backing.name)
            if not published and os.path.lexists(backing):
                try:
                    shutil.rmtree(backing)
                    _fsync_directory(root)
                except OSError:
                    if operation_error is None:
                        raise
        if staging_parent_created and os.path.lexists(staging_parent):
            try:
                shutil.rmtree(staging_parent)
                _fsync_directory(staging_parent.parent)
            except OSError:
                if operation_error is None:
                    raise


def _completion_filename(job: EpisodeJob) -> str:
    return f"seed-{job.seed:010d}-repeat-{job.repeat_idx:03d}-episode-{job.episode_index:05d}.json"


def _episode_attempt_directory(attempt: Path, job: EpisodeJob) -> Path:
    if re.fullmatch(r"[a-f0-9]{64}", job.result_identity) is None:
        raise SimulationError(f"unsafe episode result identity: {job.result_identity!r}")
    return attempt / "retry_ledger" / job.result_identity


def _episode_attempt_records(attempt: Path, job: EpisodeJob) -> list[dict[str, Any]]:
    root = _episode_attempt_directory(attempt, job)
    if not root.exists():
        return []
    if not root.is_dir():
        raise SimulationError(f"episode retry ledger is not a directory: {root}")
    records: list[dict[str, Any]] = []
    for ordinal, path in enumerate(sorted(root.glob("attempt-*.json")), start=1):
        expected_name = f"attempt-{ordinal:03d}.json"
        if path.name != expected_name or not path.is_file():
            raise SimulationError(f"episode retry ledger is not contiguous: {path}")
        record = _json_object(path, label="episode retry attempt")
        if (
            record.get("schema_version") != RETRY_ATTEMPT_SCHEMA_VERSION
            or record.get("result_identity") != job.result_identity
            or record.get("job") != asdict(job)
            or record.get("attempt_number") != ordinal
            or record.get("state") not in {
                "starting", "server_ready", "running", "terminating", "succeeded", "failed"
            }
        ):
            raise SimulationError(f"episode retry ledger record is invalid: {path}")
        records.append(record)
    unexpected = [
        path
        for path in root.iterdir()
        if path.is_file()
        and re.fullmatch(r"attempt-[0-9]{3}\.json", path.name) is None
        and re.fullmatch(
            r"\.attempt-[0-9]{3}\.json\.tmp\.[0-9]+\.[0-9]+",
            path.name,
        )
        is None
    ]
    if unexpected:
        raise SimulationError(f"episode retry ledger contains unexpected files: {unexpected}")
    return records


def _begin_episode_attempt(
    attempt: Path,
    job: EpisodeJob,
    *,
    worker_id: str,
    phase: str,
    session_id: str,
    log_path: Path,
    server: ChildProcess | None = None,
) -> int:
    if _WORKER_RE.fullmatch(worker_id) is None:
        raise SimulationError(f"invalid retry worker id: {worker_id!r}")
    if not phase or not _SAFE_COMPONENT_RE.fullmatch(phase):
        raise SimulationError(f"invalid retry phase: {phase!r}")
    records = _episode_attempt_records(attempt, job)
    if any(record["state"] not in {"succeeded", "failed"} for record in records):
        raise SimulationError(f"episode has an unfinished persisted attempt: {job.result_identity}")
    ordinal = len(records) + 1
    root = _episode_attempt_directory(attempt, job)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"attempt-{ordinal:03d}.json"
    if os.path.lexists(path):
        raise SimulationError(f"episode retry attempt already exists: {path}")
    record: dict[str, Any] = {
        "schema_version": RETRY_ATTEMPT_SCHEMA_VERSION,
        "result_identity": job.result_identity,
        "job": asdict(job),
        "attempt_number": ordinal,
        "worker_id": worker_id,
        "phase": phase,
        "session_id": session_id,
        "log_path": str(log_path),
        "owner_host": platform.node(),
        "owner_uid": os.getuid(),
        "orchestrator_pid": os.getpid(),
        "state": "starting",
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    if server is not None:
        record.update(
            {
                "server_process_pid": server.process.pid,
                "server_process_group_id": server.process.pid,
                "server_process_start_ticks": server.process_start_ticks,
                "server_process_command_sha256": server.command_sha256,
            }
        )
    atomic_write_json(path, record)
    return ordinal


def _update_episode_attempt(
    attempt: Path,
    job: EpisodeJob,
    ordinal: int,
    *,
    state: str,
    **updates: Any,
) -> None:
    records = _episode_attempt_records(attempt, job)
    if ordinal < 1 or ordinal > len(records):
        raise SimulationError(f"unknown episode attempt {ordinal}: {job.result_identity}")
    path = _episode_attempt_directory(attempt, job) / f"attempt-{ordinal:03d}.json"
    record = records[ordinal - 1]
    current = str(record["state"])
    allowed = {
        "starting": {"starting", "server_ready", "running", "terminating", "failed"},
        "server_ready": {"running", "failed"},
        "running": {"terminating", "succeeded", "failed"},
        "terminating": {"failed"},
    }
    if state not in allowed.get(current, set()):
        raise SimulationError(
            f"invalid episode attempt transition {current!r} -> {state!r}: {path}"
        )
    record.update(updates)
    record["state"] = state
    record["updated_at"] = time.time()
    if state in {"succeeded", "failed"}:
        record["finished_at"] = time.time()
    atomic_write_json(path, record)


def _mark_episode_attempt_running(
    attempt: Path,
    job: EpisodeJob,
    ordinal: int,
    *,
    pid: int,
    process_start_ticks: int | None = None,
    process_command_sha256: str = "",
    server: ChildProcess | None = None,
    session_id: str | None = None,
    log_path: Path | None = None,
) -> None:
    updates: dict[str, Any] = {
        "process_pid": pid,
        "process_group_id": pid,
        "process_role": "shadow",
        "process_start_ticks": process_start_ticks,
        "process_command_sha256": process_command_sha256,
        "phase": "shadow",
    }
    if server is not None:
        updates.update(
            {
                "server_process_pid": server.process.pid,
                "server_process_group_id": server.process.pid,
                "server_process_start_ticks": server.process_start_ticks,
                "server_process_command_sha256": server.command_sha256,
            }
        )
    if session_id is not None:
        updates["session_id"] = session_id
    if log_path is not None:
        updates["log_path"] = str(log_path)
    _update_episode_attempt(
        attempt,
        job,
        ordinal,
        state="running",
        **updates,
    )


def _finish_episode_attempt(
    attempt: Path,
    job: EpisodeJob,
    ordinal: int,
    *,
    state: str,
    reason: str = "",
    error: str = "",
    returncode: int | None = None,
) -> None:
    if state not in {"succeeded", "failed"}:
        raise SimulationError(f"invalid terminal episode attempt state: {state!r}")
    updates: dict[str, Any] = {
        "reason": reason,
        "error": error[:2000],
        "returncode": returncode,
    }
    _update_episode_attempt(attempt, job, ordinal, state=state, **updates)


def _episode_retry_exhausted(attempt: Path, job: EpisodeJob, *, max_retries: int) -> bool:
    failures = sum(record["state"] == "failed" for record in _episode_attempt_records(attempt, job))
    return failures > max_retries


def _reconcile_unfinished_episode_attempts(attempt: Path, job: EpisodeJob) -> int:
    reconciled = 0
    records = _episode_attempt_records(attempt, job)
    for record in records:
        state = str(record["state"])
        if state in {"starting", "server_ready", "running", "terminating"}:
            _finish_episode_attempt(
                attempt,
                job,
                int(record["attempt_number"]),
                state="failed",
                reason="orchestrator_restarted",
                error=f"persisted attempt was left in state {state!r}",
            )
            reconciled += 1
        elif state == "succeeded":
            raise SimulationError(
                "retry ledger records success but no valid episode completion exists: "
                f"{job.result_identity}"
            )
    return reconciled


def _cleanup_persisted_runtime(
    attempt: Path,
    jobs: Sequence[EpisodeJob],
    *,
    shutdown_grace_sec: float,
    shm_root: Path = Path("/dev/shm"),
) -> None:
    unfinished: list[dict[str, Any]] = []
    latest_by_worker: dict[str, dict[str, Any]] = {}
    latest_key_by_worker: dict[str, tuple[float, int, int]] = {}
    for job in jobs:
        for record in _episode_attempt_records(attempt, job):
            state = str(record["state"])
            if state in {"starting", "server_ready", "running", "terminating"}:
                unfinished.append(record)
            worker_id = record.get("worker_id")
            if not isinstance(worker_id, str) or _WORKER_RE.fullmatch(worker_id) is None:
                raise UnsafeCleanupError("persisted retry record has an invalid worker identity")
            timestamp = record.get("updated_at")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(float(timestamp))
            ):
                raise UnsafeCleanupError("persisted retry record has an invalid update timestamp")
            key = (
                float(timestamp),
                int(record["job"]["episode_index"]),
                int(record["attempt_number"]),
            )
            if worker_id not in latest_key_by_worker or key > latest_key_by_worker[worker_id]:
                latest_by_worker[worker_id] = record
                latest_key_by_worker[worker_id] = key

    references: dict[tuple[int, int], tuple[dict[str, Any], str, str]] = {}

    def add_reference(record: dict[str, Any], prefix: str, role: str) -> None:
        pid = record.get(f"{prefix}process_pid")
        if pid is None:
            return
        group_id = record.get(f"{prefix}process_group_id", pid)
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid < 1
            or isinstance(group_id, bool)
            or not isinstance(group_id, int)
            or group_id < 1
        ):
            raise UnsafeCleanupError(
                f"persisted {role} process identity is malformed for {record.get('worker_id')}"
            )
        key = (pid, group_id)
        existing = references.get(key)
        candidate = (record, prefix, role)
        if existing is not None:
            for suffix in ("process_start_ticks", "process_command_sha256"):
                if existing[0].get(f"{existing[1]}{suffix}") != record.get(
                    f"{prefix}{suffix}"
                ):
                    raise UnsafeCleanupError(
                        f"persisted process identity drifted for PID {pid}"
                    )
            return
        references[key] = candidate

    for record in unfinished:
        add_reference(record, "", str(record.get("process_role", "child")))
    for record in latest_by_worker.values():
        add_reference(record, "server_", "server")
        if record.get("process_role") == "server":
            add_reference(record, "", "server")

    for (pid, group_id), (record, prefix, role) in references.items():
        if not _process_group_has_live_members(group_id):
            continue
        start_ticks = record.get(f"{prefix}process_start_ticks")
        command_sha256 = record.get(f"{prefix}process_command_sha256")
        owner_uid = record.get("owner_uid")
        if (
            record.get("owner_host") != platform.node()
            or isinstance(owner_uid, bool)
            or not isinstance(owner_uid, int)
            or owner_uid != os.getuid()
            or group_id != pid
            or isinstance(start_ticks, bool)
            or not isinstance(start_ticks, int)
            or start_ticks < 1
            or not isinstance(command_sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", command_sha256) is None
        ):
            raise UnsafeCleanupError(
                f"cannot prove ownership of live persisted {role} process group {group_id}"
            )
        _terminate_persisted_process_group(
            group_id,
            shutdown_grace_sec,
            process_start_ticks=start_ticks,
            process_command_sha256=command_sha256,
            owner_uid=owner_uid,
        )

    cleaned_sessions: set[tuple[str, str]] = set()
    run8 = re.sub(r"[^a-z0-9]", "", attempt.name.lower())[:8]
    if len(run8) != 8:
        raise UnsafeCleanupError("attempt identity cannot represent persisted SHM ownership")
    for record in unfinished:
        if record.get("phase") != "shadow":
            continue
        worker_id = str(record.get("worker_id", ""))
        session_id = str(record.get("session_id", ""))
        session_key = (worker_id, session_id)
        if session_key in cleaned_sessions:
            continue
        if (
            _WORKER_RE.fullmatch(worker_id) is None
            or _SHADOW_SESSION_RE.fullmatch(session_id) is None
        ):
            raise UnsafeCleanupError("persisted shadow resource identity is malformed")
        status_path = attempt / "status" / f"shadow-{worker_id}-{session_id}.json"
        _remove_owned_shutdown_guard(
            status_path,
            run_id=attempt.name,
            worker_id=worker_id,
            session_id=session_id,
        )
        shm_name = f"huge_ha_{run8}_{worker_id}_{session_id}"
        shm = OwnedShm(shm_name, shm_root / shm_name, os.getuid())
        shm.observe()
        if not shm.cleanup():
            raise UnsafeCleanupError(f"SHM cleanup identity failed for {shm.name}")
        cleaned_sessions.add(session_key)


def _schedule_episode_failure(
    worker: WorkerRuntime,
    *,
    attempt: Path,
    job: EpisodeJob,
    attempt_number: int,
    max_retries: int,
    backoff_sec: float,
    reason: str,
    error: str,
    returncode: int | None,
    now: float,
) -> bool:
    records = _episode_attempt_records(attempt, job)
    if not records or attempt_number != len(records):
        raise SimulationError(
            f"worker {worker.worker_id} retry attempt identity drift for {job.result_identity}"
        )
    if records[-1]["state"] != "failed":
        _finish_episode_attempt(
            attempt,
            job,
            attempt_number,
            state="failed",
            reason=reason,
            error=error,
            returncode=returncode,
        )
    exhausted = _episode_retry_exhausted(attempt, job, max_retries=max_retries)
    if not exhausted:
        worker.retry_not_before_monotonic = now + backoff_sec
        return False
    if not worker.pending or worker.pending[0].result_identity != job.result_identity:
        raise SimulationError(f"worker {worker.worker_id} pending queue identity drift")
    records = _episode_attempt_records(attempt, job)
    _publish_execution_failed_completion(
        attempt,
        job,
        worker_id=worker.worker_id,
        attempt_count=len(records),
        last_reason=reason,
        last_error=error,
    )
    canonical = validate_and_publish_completion(attempt, job)
    if canonical is None or canonical.get("classification") != "execution_failed":
        raise SimulationError(
            f"failed to publish terminal execution failure: {job.result_identity}"
        )
    worker.pending.popleft()
    worker.retry_not_before_monotonic = 0.0
    return True


def _shadow_watchdog_reason(
    worker: WorkerRuntime,
    *,
    now: float,
    startup_timeout_sec: float,
    episode_timeout_sec: float,
) -> str | None:
    started = worker.active_started_monotonic
    if started is None:
        return None
    elapsed = max(0.0, now - started)
    if elapsed >= episode_timeout_sec:
        return "episode_timeout"
    if not worker.active_binding_verified and elapsed >= startup_timeout_sec:
        return "shadow_startup_timeout"
    return None


def _publish_execution_failed_completion(
    attempt: Path,
    job: EpisodeJob,
    *,
    worker_id: str,
    attempt_count: int,
    last_reason: str,
    last_error: str,
) -> dict[str, Any]:
    if attempt_count < 1:
        raise SimulationError("execution failure requires at least one attempted execution")
    raw_path = attempt / "raw_results" / _completion_filename(job)
    payload = {
        "task": job.isaac_task_name,
        "seed": job.seed,
        "repeat_idx": job.repeat_idx,
        "episode_seed": job.episode_seed,
        "episode_index": job.episode_index,
        "success": False,
        "failure_reason": _EXECUTION_FAILED,
        "error": "",
        "episode_steps": 0,
        "duration_sec": 0.0,
        "video_recorded": False,
        "video_path": "",
        "execution_attempts": attempt_count,
        "retry_count": max(0, attempt_count - 1),
        "last_execution_failure": last_reason,
        "last_execution_error": last_error[:2000],
    }
    atomic_write_json(raw_path, payload)
    identity_keys = ("task", "seed", "repeat_idx", "episode_seed", "episode_index")
    completion = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "run_id": attempt.name,
        "worker_id": worker_id,
        "result_identity": job.result_identity,
        "identity": {key: payload[key] for key in identity_keys},
        "payload_sha256": canonical_sha256(payload),
        "raw_result_path": str(raw_path.resolve()),
        "raw_result_sha256": sha256_file(raw_path),
        "payload": payload,
    }
    atomic_write_json(attempt / "completions" / _completion_filename(job), completion)
    return completion


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
    if failure == _EXECUTION_FAILED and payload["success"] is False:
        return "execution_failed"
    if failure in _LEGAL_TASK_OUTCOMES and bool(payload["success"]) == (failure == "success"):
        return "task_complete"
    return "pending_integrity"


def _model_io_contract_context(
    attempt: Path,
    job: EpisodeJob,
    worker_id: str,
) -> dict[str, Any]:
    contract_path = attempt.parent.parent / "contract.json"
    contract = _json_object(contract_path, label="model I/O executable contract")
    static_plan = contract.get("static_plan")
    checkpoint = static_plan.get("checkpoint") if isinstance(static_plan, dict) else None
    source_inventory = (
        static_plan.get("source_inventory") if isinstance(static_plan, dict) else None
    )
    task = static_plan.get("task") if isinstance(static_plan, dict) else None
    protocol = static_plan.get("protocol") if isinstance(static_plan, dict) else None
    contract_hash = canonical_sha256(contract)
    source_digest = (
        source_inventory.get("source_inventory_digest")
        if isinstance(source_inventory, dict)
        else None
    )
    if (
        not isinstance(static_plan, dict)
        or contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or contract.get("static_plan_hash") != canonical_sha256(static_plan)
        or attempt.parent.parent.name != contract_hash[:16]
        or not isinstance(checkpoint, dict)
        or not isinstance(task, dict)
        or not isinstance(protocol, dict)
        or task.get("cli_task") != job.task_cli
        or task.get("isaac_task") != job.isaac_task_name
        or not isinstance(source_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None
        or not isinstance(checkpoint.get("checkpoint_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", checkpoint["checkpoint_sha256"])
        is None
        or not isinstance(checkpoint.get("model_architecture"), str)
        or not checkpoint["model_architecture"]
        or type(checkpoint.get("action_horizon")) is not int
        or checkpoint["action_horizon"] <= 0
        or type(checkpoint.get("num_inference_steps")) is not int
        or checkpoint["num_inference_steps"] <= 0
        or not isinstance(protocol.get("mapping"), str)
        or not protocol["mapping"]
    ):
        raise ConfigError("model I/O executable contract provenance is invalid")
    if _WORKER_RE.fullmatch(worker_id) is None:
        raise ConfigError(f"invalid model I/O worker identity: {worker_id!r}")
    return {
        "run_id": attempt.name,
        "worker_id": worker_id,
        "task": job.task_cli,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "contract_hash": contract_hash,
        "source_inventory_digest": source_digest,
        "model_architecture": checkpoint["model_architecture"],
        "action_horizon": checkpoint["action_horizon"],
        "num_inference_steps": checkpoint["num_inference_steps"],
        "mapping_version": protocol["mapping"],
    }


def _validate_model_io_call_inventory(
    attempt: Path,
    session_directory: Path,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    trace_session_id: str,
    episode_seed: int,
    seeded_generation: int,
    record_count: int,
) -> int:
    """Re-validate every terminal trace call and its exact session inventory."""
    calls = manifest.get("calls")
    if not isinstance(calls, list) or len(calls) != record_count:
        raise ConfigError("model I/O manifest call inventory is invalid")
    _require_owned_attempt_path(
        attempt,
        session_directory,
        label="model I/O session directory",
        directory=True,
    )
    expected_names = {"manifest.json", "_SUCCESS"}
    expected_names.update(
        f"call-{index:06d}.{suffix}"
        for index in range(1, record_count + 1)
        for suffix in ("json", "npz")
    )
    try:
        actual_names = {entry.name for entry in session_directory.iterdir()}
    except OSError as exc:
        raise ConfigError(f"cannot enumerate model I/O session: {exc}") from exc
    if actual_names != expected_names:
        raise ConfigError(
            "model I/O session inventory mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )

    call_keys = {
        "trace_call_index",
        "http_response_sequence",
        "reset_generation",
        "json_relative_path",
        "json_bytes",
        "json_sha256",
        "npz_relative_path",
        "npz_bytes",
        "npz_sha256",
    }
    call_payload_keys = {
        "schema_version",
        "run_id",
        "worker_id",
        "task",
        "checkpoint_sha256",
        "contract_hash",
        "source_inventory_digest",
        "model_architecture",
        "action_horizon",
        "num_inference_steps",
        "mapping_version",
        "trace_session_id",
        "episode_seed",
        "seeded_reset_generation",
        "reset_generation",
        "trace_call_index",
        "http_response_sequence",
        "model_inputs",
        "model_output",
        "preprocessing_provenance",
        "wire_provenance",
        "compatibility_only_excluded",
        "npz_relative_path",
        "npz_bytes",
        "npz_sha256",
        "created_at",
    }
    total_call_bytes = 0
    previous_http_sequence = 0
    for expected_index, entry in enumerate(calls, start=1):
        expected_json_name = f"call-{expected_index:06d}.json"
        expected_npz_name = f"call-{expected_index:06d}.npz"
        if not isinstance(entry, dict) or set(entry) != call_keys:
            raise ConfigError("model I/O manifest call schema is invalid")
        http_sequence = entry.get("http_response_sequence")
        reset_generation = entry.get("reset_generation")
        json_size = entry.get("json_bytes")
        npz_size = entry.get("npz_bytes")
        json_digest = entry.get("json_sha256")
        npz_digest = entry.get("npz_sha256")
        if (
            entry.get("trace_call_index") != expected_index
            or type(http_sequence) is not int
            or http_sequence <= previous_http_sequence
            or type(reset_generation) is not int
            or reset_generation < 1
            or entry.get("json_relative_path") != expected_json_name
            or entry.get("npz_relative_path") != expected_npz_name
            or type(json_size) is not int
            or json_size <= 0
            or type(npz_size) is not int
            or npz_size <= 0
            or not isinstance(json_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", json_digest) is None
            or not isinstance(npz_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", npz_digest) is None
        ):
            raise ConfigError("model I/O manifest call values are invalid")
        previous_http_sequence = http_sequence

        json_path = session_directory / expected_json_name
        npz_path = session_directory / expected_npz_name
        _require_owned_attempt_path(
            attempt, json_path, label="model I/O call JSON", directory=False
        )
        _require_owned_attempt_path(
            attempt, npz_path, label="model I/O call NPZ", directory=False
        )
        json_info = json_path.stat()
        npz_info = npz_path.stat()
        if (
            json_info.st_size != json_size
            or npz_info.st_size != npz_size
            or sha256_file(json_path) != json_digest
            or sha256_file(npz_path) != npz_digest
        ):
            raise ConfigError("model I/O call size/digest mismatch")
        try:
            raw_json = json_path.read_bytes()
            call_payload = json.loads(raw_json)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError(f"invalid model I/O call JSON: {json_path}: {exc}") from exc
        if (
            not isinstance(call_payload, dict)
            or set(call_payload) != call_payload_keys
            or raw_json != canonical_json_bytes(call_payload) + b"\n"
            or call_payload.get("schema_version") != MODEL_IO_TRACE_SCHEMA_VERSION
            or any(call_payload.get(key) != value for key, value in context.items())
            or call_payload.get("trace_session_id") != trace_session_id
            or call_payload.get("episode_seed") != episode_seed
            or call_payload.get("seeded_reset_generation") != seeded_generation
            or call_payload.get("trace_call_index") != expected_index
            or call_payload.get("http_response_sequence") != http_sequence
            or call_payload.get("reset_generation") != reset_generation
            or call_payload.get("npz_relative_path") != expected_npz_name
            or call_payload.get("npz_bytes") != npz_size
            or call_payload.get("npz_sha256") != npz_digest
            or not isinstance(call_payload.get("created_at"), str)
            or not call_payload["created_at"]
        ):
            raise ConfigError("model I/O call provenance/schema is invalid")
        total_call_bytes += json_size + npz_size
    expected_last_sequence = previous_http_sequence if calls else 0
    if manifest.get("last_http_response_sequence") != expected_last_sequence:
        raise ConfigError("model I/O manifest last response sequence is invalid")
    return total_call_bytes


def _validate_model_io_artifact(
    attempt: Path,
    job: EpisodeJob,
    payload: Mapping[str, Any],
    worker_id: str,
    receipt: Any,
    *,
    recorded_artifact: Any = None,
) -> dict[str, Any]:
    receipt_keys = {
        "schema_version",
        "run_id",
        "worker_id",
        "trace_session_id",
        "result_identity",
        "episode_seed",
        "seeded_reset_generation",
        "outcome_sha256",
        "record_count",
        "last_trace_call_index",
        "last_http_response_sequence",
        "trace_status",
        "manifest_relative_path",
        "manifest_sha256",
        "success_marker_relative_path",
        "total_bytes",
        "error",
    }
    outcome_sha256 = canonical_sha256(payload)
    if (
        not isinstance(receipt, dict)
        or set(receipt) != receipt_keys
        or receipt.get("schema_version") != MODEL_IO_RECEIPT_SCHEMA_VERSION
        or receipt.get("run_id") != attempt.name
        or receipt.get("worker_id") != worker_id
        or receipt.get("result_identity") != job.result_identity
        or receipt.get("episode_seed") != job.episode_seed
        or receipt.get("outcome_sha256") != outcome_sha256
        or receipt.get("trace_status") != "complete"
        or receipt.get("error") is not None
    ):
        raise ConfigError("model I/O receipt identity/outcome/status is invalid")
    trace_session_id = receipt.get("trace_session_id")
    seeded_generation = receipt.get("seeded_reset_generation")
    record_count = receipt.get("record_count")
    last_trace_call_index = receipt.get("last_trace_call_index")
    last_http_response_sequence = receipt.get("last_http_response_sequence")
    total_bytes = receipt.get("total_bytes")
    if (
        not isinstance(trace_session_id, str)
        or re.fullmatch(r"[a-f0-9]{32}", trace_session_id) is None
        or type(seeded_generation) is not int
        or seeded_generation < 1
        or type(record_count) is not int
        or record_count < 0
        or type(last_trace_call_index) is not int
        or last_trace_call_index != record_count
        or type(last_http_response_sequence) is not int
        or last_http_response_sequence < 0
        or type(total_bytes) is not int
        or total_bytes <= 0
    ):
        raise ConfigError("model I/O receipt counters are invalid")
    expected_manifest_relative = (
        f"model_io/{worker_id}/session-{trace_session_id}/manifest.json"
    )
    expected_marker_relative = (
        f"model_io/{worker_id}/session-{trace_session_id}/_SUCCESS"
    )
    manifest_relative = receipt.get("manifest_relative_path")
    marker_relative = receipt.get("success_marker_relative_path")
    if (
        manifest_relative != expected_manifest_relative
        or marker_relative != expected_marker_relative
    ):
        raise ConfigError("model I/O receipt artifact paths are invalid")
    manifest_path = attempt / expected_manifest_relative
    marker_path = attempt / expected_marker_relative
    _require_owned_attempt_path(
        attempt, manifest_path, label="model I/O manifest", directory=False
    )
    _require_owned_attempt_path(
        attempt, marker_path, label="model I/O success marker", directory=False
    )
    manifest_sha256 = sha256_file(manifest_path)
    marker_sha256 = sha256_file(marker_path)
    if (
        receipt.get("manifest_sha256") != manifest_sha256
        or re.fullmatch(r"[0-9a-f]{64}", marker_sha256) is None
    ):
        raise ConfigError("model I/O receipt artifact digest is invalid")
    manifest = _json_object(
        manifest_path, label="model I/O manifest", attempt=attempt
    )
    marker = _json_object(
        marker_path, label="model I/O success marker", attempt=attempt
    )
    context = _model_io_contract_context(attempt, job, worker_id)
    expected_manifest_keys = {
        "schema_version",
        "run_id",
        "worker_id",
        "task",
        "checkpoint_sha256",
        "contract_hash",
        "source_inventory_digest",
        "model_architecture",
        "action_horizon",
        "num_inference_steps",
        "mapping_version",
        "trace_session_id",
        "episode_seed",
        "seeded_reset_generation",
        "result_identity",
        "outcome_sha256",
        "record_count",
        "last_trace_call_index",
        "last_http_response_sequence",
        "trace_status",
        "degraded_reasons",
        "calls",
        "error",
        "created_at",
    }
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != MODEL_IO_TRACE_SCHEMA_VERSION
        or any(manifest.get(key) != value for key, value in context.items())
        or manifest.get("trace_session_id") != trace_session_id
        or manifest.get("episode_seed") != job.episode_seed
        or manifest.get("seeded_reset_generation") != seeded_generation
        or manifest.get("result_identity") != job.result_identity
        or manifest.get("outcome_sha256") != outcome_sha256
        or manifest.get("record_count") != record_count
        or manifest.get("last_trace_call_index") != last_trace_call_index
        or manifest.get("last_http_response_sequence")
        != last_http_response_sequence
        or manifest.get("trace_status") != "complete"
        or manifest.get("degraded_reasons") != []
        or manifest.get("error") is not None
        or not isinstance(manifest.get("calls"), list)
        or len(manifest["calls"]) != record_count
        or not isinstance(manifest.get("created_at"), str)
        or not manifest["created_at"]
    ):
        raise ConfigError("model I/O manifest provenance is invalid")
    expected_marker = {
        "schema_version": MODEL_IO_RECEIPT_SCHEMA_VERSION,
        "trace_session_id": trace_session_id,
        "manifest_relative_path": expected_manifest_relative,
        "manifest_sha256": manifest_sha256,
    }
    if marker != expected_marker:
        raise ConfigError("model I/O success marker is invalid")
    try:
        manifest_bytes = manifest_path.read_bytes()
        marker_bytes = marker_path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read terminal model I/O artifacts: {exc}") from exc
    if (
        manifest_bytes != canonical_json_bytes(manifest) + b"\n"
        or marker_bytes != canonical_json_bytes(marker) + b"\n"
    ):
        raise ConfigError("model I/O terminal JSON is not canonical")
    call_bytes = _validate_model_io_call_inventory(
        attempt,
        manifest_path.parent,
        manifest,
        context,
        trace_session_id=trace_session_id,
        episode_seed=job.episode_seed,
        seeded_generation=seeded_generation,
        record_count=record_count,
    )
    if total_bytes != call_bytes + len(manifest_bytes) + len(marker_bytes):
        raise ConfigError("model I/O receipt total_bytes is not exact")
    artifact = {
        "receipt": dict(receipt),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
        },
        "success_marker": {
            "path": str(marker_path.resolve()),
            "sha256": marker_sha256,
        },
    }
    if recorded_artifact is not None and recorded_artifact != artifact:
        raise ConfigError("canonical model I/O artifact drifted")
    return artifact


def _episode_requires_video(attempt: Path, job: EpisodeJob) -> bool:
    contract = _json_object(
        attempt.parent.parent / "contract.json",
        label="video-policy executable contract",
    )
    static_plan = contract.get("static_plan")
    policy = static_plan.get("policy") if isinstance(static_plan, dict) else None
    episodes = static_plan.get("episodes") if isinstance(static_plan, dict) else None
    every = policy.get("record_video_every_n") if isinstance(policy, dict) else None
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or not isinstance(static_plan, dict)
        or contract.get("static_plan_hash") != canonical_sha256(static_plan)
        or not isinstance(episodes, list)
        or asdict(job) not in episodes
        or type(every) is not int
        or every < 0
    ):
        raise ConfigError("record-video policy provenance is invalid")
    return every > 0 and job.episode_index % every == 0


def validate_completion_envelope(
    attempt: Path,
    job: EpisodeJob,
    *,
    shadow_status: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    completion_path = attempt / "completions" / _completion_filename(job)
    if not os.path.lexists(completion_path):
        return None
    try:
        completion = _json_object(
            completion_path,
            label="completion envelope",
            attempt=attempt,
        )
    except ConfigError:
        return None
    classification = classify_existing_result(job, completion)
    if classification not in {"task_complete", "execution_failed"}:
        return None
    attempt_root = attempt.resolve()
    raw_path = Path(str(completion.get("raw_result_path", ""))).resolve()
    if not _is_relative_to(raw_path, attempt_root) or not raw_path.is_file():
        return None
    if completion.get("raw_result_sha256") != sha256_file(raw_path):
        return None
    payload = completion.get("payload")
    if not isinstance(payload, dict) or completion.get("payload_sha256") != canonical_sha256(payload):
        return None
    artifacts: dict[str, Any] = {}
    video_recorded = payload.get("video_recorded")
    video_path_raw = payload.get("video_path")
    if not isinstance(video_recorded, bool) or not isinstance(video_path_raw, str):
        return None
    if classification == "task_complete":
        try:
            requires_video = _episode_requires_video(attempt, job)
        except ConfigError:
            return None
        if requires_video:
            if video_recorded is not True or not video_path_raw:
                return None
            video_path = Path(video_path_raw).resolve()
            if (
                not _is_relative_to(video_path, attempt_root)
                or not video_path.is_file()
                or video_path.is_symlink()
            ):
                return None
            artifacts["video"] = {
                "path": str(video_path),
                "sha256": sha256_file(video_path),
            }
        elif video_recorded is not False or video_path_raw != "":
            return None
        if shadow_status is None:
            return None
        worker_id = completion.get("worker_id")
        try:
            artifacts["model_io"] = _validate_model_io_artifact(
                attempt,
                job,
                payload,
                str(worker_id),
                shadow_status.get("model_io"),
            )
        except (ConfigError, OSError, RuntimeError, ValueError):
            return None
    elif video_recorded is not False or video_path_raw != "" or artifacts:
        return None
    canonical = {
        "schema_version": EPISODE_SCHEMA_VERSION, "result_identity": job.result_identity,
        "job": asdict(job), "payload": payload, "worker_id": completion.get("worker_id", ""),
        "classification": classification, "artifacts": artifacts,
        "completion_sha256": sha256_file(completion_path),
    }
    return canonical


def validate_and_publish_completion(
    attempt: Path,
    job: EpisodeJob,
    *,
    shadow_status: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    canonical = validate_completion_envelope(
        attempt, job, shadow_status=shadow_status
    )
    if canonical is None:
        return None
    atomic_write_json(attempt / "episodes" / _completion_filename(job), canonical)
    return canonical


def _read_existing_episode(attempt: Path, job: EpisodeJob) -> dict[str, Any] | None:
    path = attempt / "episodes" / _completion_filename(job)
    if not os.path.lexists(path):
        return None
    try:
        payload = _json_object(
            path,
            label="canonical episode",
            attempt=attempt,
        )
    except ConfigError:
        return None
    if (
        payload.get("schema_version") != EPISODE_SCHEMA_VERSION
        or payload.get("result_identity") != job.result_identity
        or payload.get("job") != asdict(job)
    ):
        return None
    result = payload.get("payload")
    classification = classify_existing_result(job, {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "result_identity": job.result_identity,
        "raw_payload": result,
    })
    if classification not in {"task_complete", "execution_failed"}:
        return None
    stored_classification = payload.get("classification")
    if stored_classification is not None and stored_classification != classification:
        return None
    worker_id = payload.get("worker_id")
    artifacts = payload.get("artifacts")
    completion_path = attempt / "completions" / _completion_filename(job)
    if (
        not isinstance(worker_id, str)
        or _WORKER_RE.fullmatch(worker_id) is None
        or not isinstance(artifacts, dict)
        or not completion_path.is_file()
        or payload.get("completion_sha256") != sha256_file(completion_path)
    ):
        return None
    if classification == "task_complete":
        if not isinstance(result, dict):
            return None
        try:
            requires_video = _episode_requires_video(attempt, job)
        except ConfigError:
            return None
        expected_artifacts = {"model_io", "video"} if requires_video else {"model_io"}
        if set(artifacts) != expected_artifacts:
            return None
        if requires_video:
            artifact = artifacts.get("video")
            if (
                result.get("video_recorded") is not True
                or not isinstance(result.get("video_path"), str)
                or not result["video_path"]
                or not isinstance(artifact, dict)
                or set(artifact) != {"path", "sha256"}
            ):
                return None
            video_path = Path(str(artifact.get("path", ""))).resolve()
            if (
                not _is_relative_to(video_path, attempt.resolve())
                or not video_path.is_file()
                or video_path.is_symlink()
                or Path(result["video_path"]).resolve() != video_path
                or artifact.get("sha256") != sha256_file(video_path)
            ):
                return None
        elif (
            result.get("video_recorded") is not False
            or result.get("video_path") != ""
        ):
            return None
        model_io = artifacts.get("model_io")
        receipt = model_io.get("receipt") if isinstance(model_io, dict) else None
        try:
            _validate_model_io_artifact(
                attempt,
                job,
                result,
                worker_id,
                receipt,
                recorded_artifact=model_io,
            )
        except (ConfigError, OSError, RuntimeError, ValueError):
            return None
    elif (
        artifacts != {}
        or not isinstance(result, dict)
        or result.get("video_recorded") is not False
        or result.get("video_path") != ""
    ):
        return None
    return payload


def _persisted_shadow_proves_completion(
    attempt: Path,
    job: EpisodeJob,
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    worker_id = str(record.get("worker_id", ""))
    session_id = str(record.get("session_id", ""))
    process_pid = record.get("process_pid")
    process_group_id = record.get("process_group_id", process_pid)
    if (
        _WORKER_RE.fullmatch(worker_id) is None
        or _SHADOW_SESSION_RE.fullmatch(session_id) is None
        or isinstance(process_pid, bool)
        or not isinstance(process_pid, int)
        or process_pid < 1
        or isinstance(process_group_id, bool)
        or not isinstance(process_group_id, int)
        or process_group_id != process_pid
    ):
        return None
    status_path = attempt / "status" / f"shadow-{worker_id}-{session_id}.json"
    guard_path = status_path.with_name(f".{status_path.name}.shutdown_guard")
    marker_path = _shadow_guard_removal_marker_path(status_path)
    if (
        not status_path.is_file()
        or os.path.lexists(guard_path)
        or os.path.lexists(marker_path)
    ):
        return None
    try:
        _require_owned_attempt_path(
            attempt,
            status_path,
            label="persisted shadow status",
            directory=False,
        )
        status = _json_object(status_path, label="persisted shadow status")
        semantic_state, semantic_exit_code, _ = _decode_shadow_terminal_status(status)
    except (ConfigError, SimulationError):
        return None
    expected_identity = _expected_shadow_episode_identity(job)
    expected_completion = (attempt / "completions" / _completion_filename(job)).resolve()
    actual_completion = Path(str(status.get("completion", ""))).expanduser().resolve()
    process_started_at = status.get("process_started_at")
    if (
        status.get("schema_version") != WORKER_STATUS_SCHEMA_VERSION
        or status.get("run_id") != attempt.name
        or status.get("worker_id") != worker_id
        or status.get("shadow_session_id") != session_id
        or status.get("simulator_pid") != process_pid
        or isinstance(process_started_at, bool)
        or not isinstance(process_started_at, (int, float))
        or not math.isfinite(float(process_started_at))
        or float(process_started_at) <= 0.0
        or semantic_state != "complete"
        or semantic_exit_code != 0
        or status.get("cleanup_errors") != []
        or status.get("classification") != "completed"
        or status.get("completed_count") != 1
        or not _strict_episode_identity_matches(
            status.get("episode_identity"), expected_identity
        )
        or not _strict_episode_identity_matches(
            status.get("current_episode"), expected_identity
        )
        or actual_completion != expected_completion
        or _process_group_has_live_members(process_group_id)
    ):
        return None
    completion = validate_completion_envelope(
        attempt, job, shadow_status=status
    )
    if completion is None:
        return None
    envelope = _json_object(expected_completion, label="persisted completion envelope")
    expected_completion_identity = dict(expected_identity)
    expected_completion_identity.pop("result_identity")
    payload = completion.get("payload")
    if (
        envelope.get("run_id") != attempt.name
        or envelope.get("worker_id") != worker_id
        or not _strict_episode_identity_matches(
            envelope.get("identity"), expected_completion_identity
        )
        or not isinstance(payload, dict)
        or status.get("upstream_payload_sha256") != canonical_sha256(payload)
    ):
        return None
    return completion


def _load_reusable_episode_result(
    attempt: Path,
    job: EpisodeJob,
    *,
    max_retries: int,
) -> dict[str, Any] | None:
    records = _episode_attempt_records(attempt, job)
    result = _read_existing_episode(attempt, job)
    if result is None:
        result = validate_completion_envelope(attempt, job)
    if records and records[-1]["state"] == "running":
        proven_completion = _persisted_shadow_proves_completion(
            attempt, job, records[-1]
        )
        if proven_completion is not None:
            result = proven_completion
            _finish_episode_attempt(
                attempt,
                job,
                int(records[-1]["attempt_number"]),
                state="succeeded",
                reason="completion_recovered_after_orchestrator_restart",
                returncode=0,
            )
            atomic_write_json(
                attempt / "episodes" / _completion_filename(job), result
            )
            records = _episode_attempt_records(attempt, job)
    if result is None:
        return None
    if not records:
        return result
    payload = result.get("payload")
    if not isinstance(payload, dict):
        return None
    classification = classify_existing_result(
        job,
        {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "result_identity": job.result_identity,
            "raw_payload": payload,
        },
    )
    if classification == "execution_failed":
        attempts = payload.get("execution_attempts")
        retry_count = payload.get("retry_count")
        return (
            result
            if len(records) == max_retries + 1
            and all(record["state"] == "failed" for record in records)
            and type(attempts) is int
            and attempts == len(records)
            and type(retry_count) is int
            and retry_count == attempts - 1
            else None
        )
    if classification != "task_complete":
        return None
    last = records[-1]
    if last["state"] == "succeeded":
        return result
    return None


def _verify_completed_attempt(
    attempt: Path, jobs: Sequence[EpisodeJob], contract_hash: str, source_digest: str
) -> int:
    summary = _json_object(
        attempt / "summary.json",
        label="terminal attempt summary",
        attempt=attempt,
    )
    success_marker = attempt / "_SUCCESS"
    failure_marker = attempt / _COMPLETED_WITH_FAILURES_MARKER
    success_present = os.path.lexists(success_marker)
    failure_present = os.path.lexists(failure_marker)
    if success_present == failure_present:
        raise ConfigError("terminal attempt must have exactly one completion marker")
    marker = failure_marker if failure_present else success_marker
    _require_owned_attempt_path(
        attempt,
        marker,
        label="terminal attempt marker",
        directory=False,
    )
    try:
        marker_digest = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ConfigError(f"cannot read terminal attempt marker: {exc}") from exc
    raw_execution_failed = summary.get("execution_failed", 0)
    if (
        isinstance(raw_execution_failed, bool)
        or not isinstance(raw_execution_failed, int)
        or raw_execution_failed < 0
    ):
        raise ConfigError("terminal attempt has invalid execution_failed count")
    execution_failed = int(raw_execution_failed)
    expected_state = "completed_with_failures" if failure_present else "success"
    manifest = _json_object(
        attempt / "manifest.json",
        label="terminal attempt manifest",
        attempt=attempt,
    )
    runtime = manifest.get("runtime")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("contract_hash") != contract_hash
        or manifest.get("run_id") != attempt.name
        or manifest.get("state") != expected_state
        or not isinstance(runtime, dict)
        or (failure_present and execution_failed < 1)
        or (success_present and execution_failed != 0)
    ):
        raise ConfigError("terminal attempt manifest/marker state is inconsistent")
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("contract_hash") != contract_hash
        or summary.get("source_inventory_digest") != source_digest
        or summary.get("expected") != len(jobs)
        or summary.get("completed") != len(jobs)
        or marker_digest != canonical_sha256(summary)
    ):
        raise ConfigError("terminal attempt summary/marker does not match the current contract")
    episodes: list[dict[str, Any]] = []
    for job in jobs:
        episode = _read_existing_episode(attempt, job)
        if episode is None:
            raise ConfigError(
                "terminal attempt has missing, drifted, or invalid canonical episodes"
            )
        episodes.append(episode)
    successes = sum(bool(episode["payload"].get("success")) for episode in episodes)
    actual_execution_failed = sum(
        episode["payload"].get("failure_reason") == _EXECUTION_FAILED
        for episode in episodes
    )
    expected_by_seed = {
        str(seed): {
            "completed": sum(episode["job"]["seed"] == seed for episode in episodes),
            "success": sum(
                episode["job"]["seed"] == seed
                and bool(episode["payload"].get("success"))
                for episode in episodes
            ),
        }
        for seed in sorted({job.seed for job in jobs})
    }
    resume_reused = summary.get("resume_reused")
    if (
        summary.get("task_success") != successes
        or summary.get("task_failure") != len(episodes) - successes
        or summary.get("infrastructure_failure") != 0
        or summary.get("infrastructure_errors") != []
        or execution_failed != actual_execution_failed
        or summary.get("success_rate")
        != {"numerator": successes, "denominator": len(episodes)}
        or summary.get("by_seed") != expected_by_seed
        or isinstance(resume_reused, bool)
        or not isinstance(resume_reused, int)
        or not 0 <= resume_reused <= len(episodes)
        or runtime.get("completed") != len(episodes)
        or runtime.get("reused") != resume_reused
        or runtime.get("execution_failed", 0) != actual_execution_failed
    ):
        raise ConfigError(
            "terminal attempt summary/manifest counts do not match canonical episodes"
        )
    return execution_failed


def _verify_success_attempt(
    attempt: Path, jobs: Sequence[EpisodeJob], contract_hash: str, source_digest: str
) -> None:
    if _verify_completed_attempt(attempt, jobs, contract_hash, source_digest) != 0:
        raise ConfigError("successful attempt contains execution failures")


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


def _make_shm(run_id: str, worker_id: str, shadow_session_id: str) -> OwnedShm:
    run8 = re.sub(r"[^a-z0-9]", "", run_id.lower())[:8]
    if (
        len(run8) != 8 or not _WORKER_RE.fullmatch(worker_id)
        or _SHADOW_SESSION_RE.fullmatch(shadow_session_id) is None
    ):
        raise SimulationError("run/worker/session identity cannot be represented by the SHM ownership format")
    name = f"huge_ha_{run8}_{worker_id}_{shadow_session_id}"
    path = Path("/dev/shm") / name
    if os.path.lexists(path):
        raise SimulationError(f"shadow session SHM already exists: {path}")
    return OwnedShm(name, path, os.getuid())


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
    run_id: str, worker_id: str, attempt: Path, port: int, *,
    contract_hash: str, source_inventory_digest: str,
) -> list[str]:
    return [
        str(args.repo / ".venv/bin/python"), "-m", "huge.inference.humanoidarena_http_server",
        "--repo-root", str(args.repo), "--checkpoint", str(checkpoint.checkpoint_dir),
        "--allowed-checkpoint-root", str(
            _allowed_checkpoint_root(
                args.repo,
                checkpoint.checkpoint_dir,
                task.checkpoint_task,
            )
        ),
        "--checkpoint-task", task.checkpoint_task, "--task", task.cli_task,
        "--isaac-task", task.isaac_task, "--instruction", task.instruction,
        "--host", "127.0.0.1", "--port", str(port), "--expected-cuda-uuid", binding.gpu.cuda_uuid,
        "--run-id", run_id, "--worker-id", worker_id, "--attempt-dir", str(attempt),
        "--status-file", str(attempt / "status" / f"server-{worker_id}.json"),
        "--model-io-root", str(attempt / "model_io"), "--contract-hash", contract_hash,
        "--source-inventory-digest", source_inventory_digest,
        "--device", "cuda:0",
    ]


def _expected_server_health(
    task: TaskSpec,
    checkpoint: CheckpointContract,
    binding: GpuBinding,
) -> dict[str, Any]:
    return {
        "status": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "task": task.cli_task,
        "cuda_uuid": binding.gpu.cuda_uuid,
        "state_dim": 34,
        "action_dim": 66,
        "motion_token_dim": 64,
        "action_horizon": 50,
        "control_hz": 50,
        "model_noise_seed_algorithm": MODEL_NOISE_SEED_ALGORITHM,
    }


def _write_shard(
    path: Path, jobs: Sequence[EpisodeJob], attempt: Path, checkpoint: CheckpointContract,
    policy: Mapping[str, Any], worker_id: str, *, shadow_session_id: str | None = None,
) -> None:
    if len(jobs) != 1:
        raise SimulationError(f"single-episode shard requires exactly one job, got {len(jobs)}")
    if shadow_session_id is not None and _SHADOW_SESSION_RE.fullmatch(shadow_session_id) is None:
        raise SimulationError("shadow session id must contain 16..32 lowercase hex characters")
    recording_dir = attempt / "recordings" / worker_id
    if shadow_session_id is not None:
        recording_dir /= shadow_session_id
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
            "recording_save_dir": str(recording_dir),
            "max_steps": policy["max_steps"],
        })
    atomic_write_json(path, {"schema_version": "huge_humanoidarena_shard_v1", "worker_id": worker_id, "episodes": episodes})


def _shadow_command(
    args: argparse.Namespace, task: TaskSpec, checkpoint: CheckpointContract, binding: GpuBinding,
    run_id: str, worker_id: str, attempt: Path, shard_path: Path, jobs: Sequence[EpisodeJob],
    port: int, shm: OwnedShm, policy: Mapping[str, Any], yaml_config: Mapping[str, Any],
    *, shadow_session_id: str,
) -> list[str]:
    if len(jobs) != 1:
        raise SimulationError(f"single-episode shadow requires exactly one job, got {len(jobs)}")
    if _SHADOW_SESSION_RE.fullmatch(shadow_session_id) is None:
        raise SimulationError("shadow session id must contain 16..32 lowercase hex characters")
    job = jobs[0]
    recording_dir = attempt / "recordings" / worker_id / shadow_session_id
    status_path = attempt / "status" / f"shadow-{worker_id}-{shadow_session_id}.json"
    command = [
        str(args.isaaclab_python), str(args.shadow),
        "--humanoidarena-root", str(args.humanoidarena_root), "--isaaclab-root", str(args.isaaclab_root),
        "--upstream-sim-eval", str(args.upstream_sim_eval), "--upstream-sim-sha256", UPSTREAM_SIM_SHA256,
        "--renderer-gpu-ordinal", str(binding.renderer_ordinal), "--expected-cuda-uuid", binding.gpu.cuda_uuid,
        "--expected-pci-bus-id", binding.gpu.pci_bus_id,
        "--task", task.isaac_task, "--env_config_yaml", str(yaml_config["path"]),
        "--seed", str(job.seed), "--episode_batch_json", str(shard_path),
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
        "--recording_save_dir", str(recording_dir), "--headless", "--device", "cuda:0",
        "--owned-shm-name", shm.name, "--run-id", run_id, "--worker-id", worker_id,
        "--attempt-dir", str(attempt), "--server-status-json", str(attempt / "status" / f"server-{worker_id}.json"),
        "--worker-status-json", str(status_path),
        "--result-identity", job.result_identity,
        "--expected-episode-seed", str(job.episode_seed),
        "--shadow-session-id", shadow_session_id,
        "--completion-dir", str(attempt / "completions"),
    ]
    if task.memory_limit_percent is not None:
        command.extend(["--memory-limit-percent", str(task.memory_limit_percent)])
    return command


def _namespace_process_table() -> list[dict[str, Any]]:
    proc = Path("/proc")
    try:
        namespace_inode = (proc / "self" / "ns" / "pid").stat().st_ino
        self_status = (proc / "self" / "status").read_text(encoding="ascii")
        self_namespace_pid_line = next(
            line for line in self_status.splitlines() if line.startswith("NSpid:")
        )
        namespace_depth = len(self_namespace_pid_line.partition(":")[2].split())
        if namespace_depth < 1:
            raise ValueError("empty current NSpid chain")
        entries = list(proc.iterdir())
    except (OSError, UnicodeDecodeError, StopIteration, ValueError) as exc:
        raise UnsafeCleanupError(f"cannot inspect the current PID namespace: {exc}") from exc

    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            status_text = (entry / "status").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, UnicodeDecodeError) as exc:
            raise UnsafeCleanupError(
                f"cannot inspect process namespace status for {entry}: {exc}"
            ) from exc
        status: dict[str, str] = {}
        for line in status_text.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                status[key] = value.strip()
        try:
            namespace_pids = [int(value) for value in status["NSpid"].split()]
        except (KeyError, ValueError) as exc:
            raise UnsafeCleanupError(
                f"process namespace identity is incomplete for {entry}"
            ) from exc
        if len(namespace_pids) != namespace_depth:
            continue
        try:
            entry_namespace_inode = (entry / "ns" / "pid").stat().st_ino
            raw_stat = (entry / "stat").read_text(encoding="ascii")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (OSError, UnicodeDecodeError) as exc:
            raise UnsafeCleanupError(
                f"cannot inspect process identity for {entry}: {exc}"
            ) from exc
        if entry_namespace_inode != namespace_inode:
            continue
        try:
            namespace_groups = [int(value) for value in status["NSpgid"].split()]
            namespace_sessions = [int(value) for value in status["NSsid"].split()]
            real_uid = int(status["Uid"].split()[0])
            status_state = status["State"].split()[0]
            host_pid = int(entry.name)
            close_paren = raw_stat.rfind(")")
            if close_paren < 0:
                raise ValueError("missing stat command terminator")
            start_ticks = int(raw_stat[close_paren + 2 :].split()[19])
        except (KeyError, ValueError, IndexError) as exc:
            raise UnsafeCleanupError(
                f"process identity fields are incomplete for {entry}"
            ) from exc
        if (
            not namespace_pids
            or len(namespace_pids) != len(namespace_groups)
            or len(namespace_pids) != len(namespace_sessions)
            or namespace_pids[0] != host_pid
            or namespace_pids[-1] < 1
            or namespace_groups[-1] < 1
            or namespace_sessions[-1] < 1
            or real_uid < 0
            or len(status_state) != 1
            or start_ticks < 1
        ):
            raise UnsafeCleanupError(f"process identity drifted while inspecting {entry}")
        rows.append(
            {
                "proc_path": entry,
                "pid": namespace_pids[-1],
                "process_group_id": namespace_groups[-1],
                "session_id": namespace_sessions[-1],
                "start_ticks": start_ticks,
                "state": status_state,
                "uid": real_uid,
            }
        )
    return rows


def _read_process_identity(pid: int) -> dict[str, Any] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        return None

    def read_at(process_root: Path) -> dict[str, Any] | None:
        try:
            if (
                (process_root / "ns" / "pid").stat().st_ino
                != (Path("/proc") / "self" / "ns" / "pid").stat().st_ino
            ):
                return None
            status_text = (process_root / "status").read_text(encoding="ascii")
            status = {
                key: value.strip()
                for line in status_text.splitlines()
                for key, separator, value in [line.partition(":")]
                if separator
            }
            namespace_pids = [int(value) for value in status["NSpid"].split()]
            namespace_groups = [int(value) for value in status["NSpgid"].split()]
            process_uid = int(status["Uid"].split()[0])
            if (
                not namespace_pids
                or len(namespace_pids) != len(namespace_groups)
                or namespace_pids[-1] != pid
            ):
                return None
            raw_cmdline = (process_root / "cmdline").read_bytes()
            raw_stat = (process_root / "stat").read_text(encoding="ascii")
            fields = raw_stat[raw_stat.rfind(")") + 2 :].split()
            start_ticks = int(fields[19])
            command = [
                os.fsdecode(token)
                for token in raw_cmdline.split(b"\0")
                if token
            ]
        except (OSError, UnicodeDecodeError, KeyError, ValueError, IndexError):
            return None
        if start_ticks < 1 or not command:
            return None
        return {
            "process_group_id": namespace_groups[-1],
            "start_ticks": start_ticks,
            "command_sha256": canonical_sha256(command),
            "uid": process_uid,
        }

    direct = read_at(Path("/proc") / str(pid))
    if direct is not None:
        return direct
    try:
        matches = [row for row in _namespace_process_table() if row["pid"] == pid]
    except UnsafeCleanupError:
        return None
    if len(matches) != 1 or matches[0]["state"] == "Z":
        return None
    row = matches[0]
    identity = read_at(Path(row["proc_path"]))
    if identity is None or any(
        identity[key] != row[key]
        for key in ("process_group_id", "start_ticks", "uid")
    ):
        return None
    return identity


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
    identity = _read_process_identity(process.pid)
    return ChildProcess(
        role,
        worker_id,
        process,
        stream,
        process_start_ticks=(
            int(identity["start_ticks"]) if identity is not None else None
        ),
        command_sha256=canonical_sha256(list(command)),
    )


def _expected_shadow_episode_identity(job: EpisodeJob) -> dict[str, Any]:
    return {
        "result_identity": job.result_identity,
        "task": job.isaac_task_name,
        "seed": job.seed,
        "repeat_idx": job.repeat_idx,
        "episode_seed": job.episode_seed,
        "episode_index": job.episode_index,
    }


def _strict_episode_identity_matches(
    actual: Any, expected: Mapping[str, Any]
) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if type(expected_value) is int:
            if type(actual_value) is not int:
                return False
        elif type(expected_value) is str:
            if type(actual_value) is not str:
                return False
        else:
            return False
        if actual_value != expected_value:
            return False
    return True


def _decode_shadow_terminal_status(
    status: Mapping[str, Any],
) -> tuple[str, int, bool]:
    """Return semantic state, semantic exit code, and whether this is a handoff."""
    state = status.get("state")
    if state == "shutdown_ready":
        if "exit_code" in status:
            raise SimulationError("shadow shutdown_ready status contains final exit_code")
        if "planned_state" not in status or "planned_exit_code" not in status:
            raise SimulationError("shadow shutdown_ready status lacks planned terminal fields")
        semantic_state = status["planned_state"]
        semantic_exit_code = status["planned_exit_code"]
        is_shutdown_handoff = True
    else:
        if type(state) is not str or state not in _SHADOW_TERMINAL_EXIT_CODES:
            raise SimulationError(f"shadow terminal state is invalid: {state!r}")
        if "planned_state" in status or "planned_exit_code" in status:
            raise SimulationError("shadow final status contains planned terminal fields")
        if "exit_code" not in status:
            raise SimulationError("shadow final status lacks exit_code")
        semantic_state = state
        semantic_exit_code = status["exit_code"]
        is_shutdown_handoff = False

    if type(semantic_state) is not str or semantic_state not in _SHADOW_TERMINAL_EXIT_CODES:
        raise SimulationError(f"shadow semantic terminal state is invalid: {semantic_state!r}")
    expected_exit_code = _SHADOW_TERMINAL_EXIT_CODES[semantic_state]
    if type(semantic_exit_code) is not int or semantic_exit_code != expected_exit_code:
        raise SimulationError(
            "shadow semantic terminal code is invalid: "
            f"state={semantic_state!r} code={semantic_exit_code!r} "
            f"expected={expected_exit_code}"
        )
    return semantic_state, semantic_exit_code, is_shutdown_handoff


def _validate_live_server_status(
    status: Mapping[str, Any],
    *,
    worker_id: str,
    allow_infer_in_progress: bool,
) -> None:
    state = status.get("state")
    if state == "fatal" or status.get("fatal_error_code"):
        raise ServerError(
            f"Huge server {worker_id} entered fatal state: "
            f"{status.get('fatal_error_code', 'unknown')}"
        )
    if state == "ready":
        return
    if state != "infer_in_progress" or not allow_infer_in_progress:
        raise ServerError(
            f"Huge server {worker_id} entered unexpected live state: {state!r}"
        )

    sequence_fields = (
        "pending_sequence",
        "infer_sequence",
        "last_completed_sequence",
        "reset_generation",
    )
    if any(type(status.get(field)) is not int for field in sequence_fields):
        raise ServerError(
            f"Huge server {worker_id} infer_in_progress sequence fields are invalid"
        )
    pending_sequence = status["pending_sequence"]
    infer_sequence = status["infer_sequence"]
    last_completed_sequence = status["last_completed_sequence"]
    reset_generation = status["reset_generation"]
    if (
        pending_sequence != infer_sequence + 1
        or last_completed_sequence != infer_sequence
        or infer_sequence < 0
        or last_completed_sequence < 0
        or pending_sequence < 1
        or reset_generation < 1
    ):
        raise ServerError(
            f"Huge server {worker_id} infer_in_progress sequence values are invalid"
        )


def _validate_shadow_exit_status(
    worker: WorkerRuntime,
    *,
    run_id: str,
    returncode: int,
) -> dict[str, Any]:
    shadow = worker.active_shadow
    job = worker.active_job
    session_id = worker.active_session_id
    status_path = worker.active_status_path
    if shadow is None or job is None or session_id is None or status_path is None:
        raise SimulationError(f"worker {worker.worker_id} has incomplete active shadow state")
    if not status_path.is_file():
        raise SimulationError(f"shadow status is missing for {worker.worker_id}: {status_path}")
    shutdown_guard_path = status_path.with_name(f".{status_path.name}.shutdown_guard")
    if os.path.lexists(shutdown_guard_path):
        raise SimulationError(
            f"shadow shutdown guard is still present for {worker.worker_id}: {shutdown_guard_path}"
        )
    status = _json_object(
        status_path,
        label="shadow session status",
        attempt=status_path.parent.parent,
    )
    if (
        status.get("schema_version") != WORKER_STATUS_SCHEMA_VERSION
        or status.get("run_id") != run_id
        or status.get("worker_id") != worker.worker_id
        or status.get("shadow_session_id") != session_id
    ):
        raise SimulationError(f"shadow session status identity mismatch for {worker.worker_id}")
    simulator_pid = status.get("simulator_pid")
    if isinstance(simulator_pid, bool) or not isinstance(simulator_pid, int):
        raise SimulationError(f"shadow simulator_pid is invalid for {worker.worker_id}")
    if simulator_pid != shadow.process.pid:
        raise SimulationError(
            f"shadow simulator PID mismatch for {worker.worker_id}: {simulator_pid} != {shadow.process.pid}"
        )
    process_started_at = status.get("process_started_at")
    if (
        isinstance(process_started_at, bool)
        or not isinstance(process_started_at, (int, float))
        or not math.isfinite(float(process_started_at))
        or float(process_started_at) <= 0.0
    ):
        raise SimulationError(f"shadow process_started_at is invalid for {worker.worker_id}")
    expected_episode_identity = _expected_shadow_episode_identity(job)
    if not _strict_episode_identity_matches(
        status.get("episode_identity"), expected_episode_identity
    ):
        raise SimulationError(f"shadow episode identity mismatch for {worker.worker_id}")
    cleanup_errors = status.get("cleanup_errors")
    if not isinstance(cleanup_errors, list):
        raise SimulationError(f"shadow cleanup_errors is not a list for {worker.worker_id}")
    if cleanup_errors:
        raise SimulationError(f"shadow cleanup failed for {worker.worker_id}: {cleanup_errors}")
    semantic_state, semantic_exit_code, is_shutdown_handoff = (
        _decode_shadow_terminal_status(status)
    )
    terminal_detail = (
        f"semantic={semantic_state} code={semantic_exit_code} process={returncode} "
        f"error={status.get('error')!r}"
    )
    if type(returncode) is not int:
        raise SimulationError(
            f"shadow process exit code is invalid for {worker.worker_id}: {terminal_detail}"
        )
    if is_shutdown_handoff:
        if returncode not in {0, semantic_exit_code}:
            raise SimulationError(
                f"shadow handoff/process exit mismatch for {worker.worker_id}: {terminal_detail}"
            )
    elif returncode != semantic_exit_code:
        raise SimulationError(
            f"shadow final/process exit mismatch for {worker.worker_id}: {terminal_detail}"
        )
    if semantic_state == "server_or_infer_fatal":
        raise ServerError(
            f"shadow {worker.worker_id} classified server/inference failure: {terminal_detail}"
        )
    if semantic_state != "complete":
        raise SimulationError(
            f"shadow {worker.worker_id} classified terminal simulation failure: {terminal_detail}"
        )
    if status.get("classification") != "completed":
        raise SimulationError(f"shadow completion classification is invalid for {worker.worker_id}")
    completed_count = status.get("completed_count")
    if (
        isinstance(completed_count, bool)
        or not isinstance(completed_count, int)
        or completed_count != 1
    ):
        raise SimulationError(f"shadow completed_count is invalid for {worker.worker_id}")
    if not _strict_episode_identity_matches(
        status.get("current_episode"), expected_episode_identity
    ):
        raise SimulationError(f"shadow current episode identity mismatch for {worker.worker_id}")
    return status


def _finish_worker_shadow(
    worker: WorkerRuntime,
    *,
    attempt: Path,
    run_id: str,
    returncode: int,
) -> None:
    shadow = worker.active_shadow
    job = worker.active_job
    shm = worker.active_shm
    attempt_number = worker.active_attempt_number
    if shadow is None or job is None or shm is None or attempt_number is None:
        raise SimulationError(f"worker {worker.worker_id} cannot finish an incomplete active shadow")

    first_error: Exception | None = None
    try:
        waited_returncode = shadow.process.wait(timeout=0)
        if waited_returncode != returncode:
            raise SimulationError(
                f"shadow wait/poll exit mismatch for {worker.worker_id}: {waited_returncode} != {returncode}"
            )
        if _process_group_has_live_members(shadow.process.pid):
            raise SimulationError(
                f"shadow {worker.worker_id} left live process-group descendants"
            )
    except Exception as exc:
        first_error = exc
    try:
        shadow.log_stream.close()
    except Exception as exc:
        if first_error is None:
            first_error = exc

    status_valid = False
    status: dict[str, Any] | None = None
    try:
        status = _validate_shadow_exit_status(worker, run_id=run_id, returncode=returncode)
        status_valid = True
    except Exception as exc:
        if first_error is None:
            first_error = exc

    if status_valid:
        try:
            completion = validate_and_publish_completion(
                attempt, job, shadow_status=status
            )
            if completion is None:
                raise SimulationError(
                    f"shadow completed without a valid completion envelope: {job.result_identity}"
                )
            if status is not None:
                expected_episode_identity = _expected_shadow_episode_identity(job)
                expected_completion_identity = dict(expected_episode_identity)
                expected_completion_identity.pop("result_identity")
                expected_path = (attempt / "completions" / _completion_filename(job)).resolve()
                completion_path = Path(str(status.get("completion", ""))).expanduser().resolve()
                if completion_path != expected_path:
                    raise SimulationError(
                        f"shadow completion path mismatch for {worker.worker_id}: "
                        f"{completion_path} != {expected_path}"
                    )
                envelope_path = attempt / "completions" / _completion_filename(job)
                envelope = _json_object(
                    envelope_path,
                    label="completion envelope",
                    attempt=attempt,
                )
                if (
                    envelope.get("run_id") != run_id
                    or envelope.get("worker_id") != worker.worker_id
                ):
                    raise SimulationError(
                        f"shadow completion ownership mismatch for {worker.worker_id}"
                    )
                if not _strict_episode_identity_matches(
                    envelope.get("identity"), expected_completion_identity
                ):
                    raise SimulationError(
                        f"shadow completion identity mismatch for {worker.worker_id}"
                    )
                payload = completion.get("payload")
                if (
                    not isinstance(payload, dict)
                    or status.get("upstream_payload_sha256") != canonical_sha256(payload)
                ):
                    raise SimulationError(
                        f"shadow completion payload hash mismatch for {worker.worker_id}"
                    )
                payload_identity = {
                    key: payload.get(key) for key in expected_completion_identity
                }
                if not _strict_episode_identity_matches(
                    payload_identity, expected_completion_identity
                ):
                    raise SimulationError(
                        f"shadow completion payload identity mismatch for {worker.worker_id}"
                    )
        except Exception as exc:
            if first_error is None:
                first_error = exc

    try:
        shm.observe()
        if not shm.cleanup():
            raise UnsafeCleanupError(f"SHM cleanup identity failed for {shm.name}")
    except Exception as exc:
        if first_error is None or isinstance(exc, UnsafeCleanupError):
            first_error = exc

    if first_error is None:
        if not worker.pending or worker.pending[0].result_identity != job.result_identity:
            first_error = SimulationError(f"worker {worker.worker_id} pending queue identity drift")
        try:
            if first_error is None:
                _finish_episode_attempt(
                    attempt,
                    job,
                    attempt_number,
                    state="succeeded",
                    returncode=returncode,
                )
        except Exception as exc:
            first_error = exc
        if first_error is None:
            worker.pending.popleft()

    if first_error is None:
        _clear_worker_shadow_state(worker)
    if first_error is not None:
        raise first_error


def _allocate_shadow_resources(
    attempt: Path,
    *,
    run_id: str,
    worker_id: str,
) -> tuple[str, Path, Path, Path, OwnedShm]:
    for _ in range(100):
        shadow_session_id = secrets.token_hex(8)
        shard_path = attempt / "shards" / f"worker-{worker_id}-{shadow_session_id}.json"
        status_path = attempt / "status" / f"shadow-{worker_id}-{shadow_session_id}.json"
        log_path = attempt / "logs" / f"sim-{worker_id}-{shadow_session_id}.log"
        recording_path = attempt / "recordings" / worker_id / shadow_session_id
        if any(os.path.lexists(path) for path in (shard_path, status_path, log_path, recording_path)):
            continue
        shm = _make_shm(run_id, worker_id, shadow_session_id)
        return shadow_session_id, shard_path, status_path, log_path, shm
    raise SimulationError(f"unable to allocate unique shadow resources for {worker_id}")


def _launch_worker_shadow(
    worker: WorkerRuntime,
    *,
    args: argparse.Namespace,
    task: TaskSpec,
    checkpoint: CheckpointContract,
    run_id: str,
    attempt: Path,
    policy: Mapping[str, Any],
    yaml_config: Mapping[str, Any],
    runtime_env: Mapping[str, str],
) -> None:
    if worker.active_shadow is not None or worker.active_job is not None:
        raise SimulationError(f"worker {worker.worker_id} already has an active shadow")
    if not worker.pending:
        return
    if worker.server is None or worker.server_starting_attempt_number is not None:
        raise SimulationError(f"worker {worker.worker_id} server is not ready for shadow launch")
    job = worker.pending[0]
    session_id, shard_path, status_path, log_path, shm = _allocate_shadow_resources(
        attempt,
        run_id=run_id,
        worker_id=worker.worker_id,
    )
    attempt_number = worker.prepared_attempt_number
    if attempt_number is None:
        attempt_number = _begin_episode_attempt(
            attempt,
            job,
            worker_id=worker.worker_id,
            phase="shadow",
            session_id=session_id,
            log_path=log_path,
            server=worker.server,
        )
    else:
        records = _episode_attempt_records(attempt, job)
        if (
            attempt_number != len(records)
            or records[-1]["state"] != "server_ready"
        ):
            raise SimulationError(
                f"worker {worker.worker_id} prepared retry attempt is invalid"
            )
    shadow: ChildProcess | None = None
    try:
        _write_shard(
            shard_path,
            [job],
            attempt,
            checkpoint,
            policy,
            worker.worker_id,
            shadow_session_id=session_id,
        )
        shadow_env = _isaac_environment(
            args,
            run_id=run_id,
            worker_id=worker.worker_id,
            gpu_uuid=worker.binding.gpu.cuda_uuid,
        )
        shadow_env.update(runtime_env)
        shadow_env["ISAAC_MULTI_IMAGE_SHM_NAME"] = shm.name
        command = _shadow_command(
            args,
            task,
            checkpoint,
            worker.binding,
            run_id,
            worker.worker_id,
            attempt,
            shard_path,
            [job],
            worker.port,
            shm,
            policy,
            yaml_config,
            shadow_session_id=session_id,
        )
        shadow = _launch_child("shadow", worker.worker_id, command, log_path, shadow_env, attempt)
        worker.active_shadow = shadow
        worker.active_job = job
        worker.active_shm = shm
        worker.active_session_id = session_id
        worker.active_status_path = status_path
        worker.active_log_path = log_path
        worker.active_started_monotonic = time.monotonic()
        worker.active_binding_verified = False
        worker.active_attempt_number = attempt_number
        _mark_episode_attempt_running(
            attempt,
            job,
            attempt_number,
            pid=shadow.process.pid,
            process_start_ticks=shadow.process_start_ticks,
            process_command_sha256=shadow.command_sha256,
            server=worker.server,
            session_id=session_id,
            log_path=log_path,
        )
    except Exception as exc:
        cleanup_error: Exception | None = None
        if shadow is not None:
            try:
                _abort_worker_shadow(
                    worker,
                    run_id=run_id,
                    grace=float(policy["shutdown_grace_sec"]),
                )
            except Exception as cleanup_exc:
                cleanup_error = cleanup_exc
        else:
            try:
                shm.observe()
                if not shm.cleanup():
                    raise UnsafeCleanupError(f"SHM cleanup identity failed for {shm.name}")
            except Exception as cleanup_exc:
                cleanup_error = cleanup_exc
        worker.prepared_attempt_number = None
        if cleanup_error is not None:
            raise cleanup_error from exc
        raise EpisodeAttemptError(
            job,
            attempt_number,
            f"shadow launch failed for {worker.worker_id}: {type(exc).__name__}: {exc}",
        ) from exc
    worker.prepared_attempt_number = None


def _restart_worker_server(
    worker: WorkerRuntime,
    *,
    args: argparse.Namespace,
    task: TaskSpec,
    checkpoint: CheckpointContract,
    run_id: str,
    attempt: Path,
    policy: Mapping[str, Any],
    contract_hash: str,
    source_inventory_digest: str,
) -> ChildProcess:
    if (
        worker.server is not None
        or worker.active_shadow is not None
        or worker.server_starting_attempt_number is not None
    ):
        raise ServerError(f"worker {worker.worker_id} is not ready for server restart")
    if not worker.pending:
        raise ServerError(f"worker {worker.worker_id} has no episode for server recovery")
    job = worker.pending[0]
    next_ordinal = len(_episode_attempt_records(attempt, job)) + 1
    session_id = secrets.token_hex(8)
    log_path = attempt / "logs" / (
        f"server-{worker.worker_id}-retry-{next_ordinal:03d}-{session_id}.log"
    )
    attempt_number = _begin_episode_attempt(
        attempt,
        job,
        worker_id=worker.worker_id,
        phase="server_restart",
        session_id=session_id,
        log_path=log_path,
    )
    server: ChildProcess | None = None
    started_monotonic = time.monotonic()
    started_wall_time = time.time()
    try:
        command = _server_command(
            args,
            task,
            checkpoint,
            worker.binding,
            run_id,
            worker.worker_id,
            attempt,
            worker.port,
            contract_hash=contract_hash,
            source_inventory_digest=source_inventory_digest,
        )
        server = _launch_child(
            "server",
            worker.worker_id,
            command,
            log_path,
            _server_environment(args.repo, worker.binding.gpu.cuda_uuid),
            args.repo,
        )
        worker.server = server
        _update_episode_attempt(
            attempt,
            job,
            attempt_number,
            state="starting",
            process_role="server",
            process_pid=server.process.pid,
            process_group_id=server.process.pid,
            process_start_ticks=server.process_start_ticks,
            process_command_sha256=server.command_sha256,
        )
        worker.server_starting_attempt_number = attempt_number
        worker.server_started_monotonic = started_monotonic
        worker.server_started_wall_time = started_wall_time
    except Exception as exc:
        if server is not None:
            try:
                _stop_worker_server(
                    worker,
                    grace=float(policy["shutdown_grace_sec"]),
                )
            except Exception as cleanup_exc:
                raise UnsafeCleanupError(
                    f"worker {worker.worker_id} server recovery cleanup failed: {cleanup_exc}"
                ) from exc
        raise EpisodeAttemptError(
            job,
            attempt_number,
            f"server restart failed for {worker.worker_id}: {type(exc).__name__}: {exc}",
        ) from exc
    return server


def _advance_worker_server_startup(
    worker: WorkerRuntime,
    *,
    attempt: Path,
    run_id: str,
    expected_health: Mapping[str, Any],
    ready_timeout_sec: float,
    now_monotonic: float | None = None,
) -> bool:
    server = worker.server
    attempt_number = worker.server_starting_attempt_number
    started_monotonic = worker.server_started_monotonic
    started_wall_time = worker.server_started_wall_time
    if (
        server is None
        or attempt_number is None
        or started_monotonic is None
        or started_wall_time is None
        or not worker.pending
    ):
        raise SimulationError(
            f"worker {worker.worker_id} has incomplete server startup state"
        )
    job = worker.pending[0]
    now = time.monotonic() if now_monotonic is None else now_monotonic

    def pending_or_timeout(last_error: str) -> bool:
        if now - started_monotonic >= ready_timeout_sec:
            raise ServerError(
                f"Huge server {worker.worker_id} startup timed out: {last_error}"
            )
        return False

    try:
        returncode = server.process.poll()
        if returncode is not None:
            raise ServerError(
                f"Huge server {worker.worker_id} exited during startup with {returncode}"
            )
        status_path = attempt / "status" / f"server-{worker.worker_id}.json"
        if not status_path.is_file():
            return pending_or_timeout("status file is not available")
        status = _json_object(
            status_path,
            label="Huge server startup status",
            attempt=attempt,
        )
        updated_at = status.get("updated_at")
        if (
            isinstance(updated_at, bool)
            or not isinstance(updated_at, (int, float))
            or not math.isfinite(float(updated_at))
        ):
            raise SimulationError(
                f"server status updated_at is invalid for {worker.worker_id}"
            )
        if float(updated_at) < started_wall_time:
            return pending_or_timeout("only a stale status file is available")
        if status.get("run_id") != run_id or status.get("worker_id") != worker.worker_id:
            raise SimulationError(
                f"server startup status identity mismatch for {worker.worker_id}"
            )
        state = str(status.get("state", ""))
        if state == "fatal" or status.get("fatal_error_code"):
            raise ServerError(
                f"Huge server {worker.worker_id} entered fatal state during startup: "
                f"{status.get('fatal_error_code', 'unknown')}"
            )
        if state == "starting":
            return pending_or_timeout("model is still loading")
        if state != "ready":
            raise ServerError(
                f"Huge server {worker.worker_id} entered unexpected startup state {state!r}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{worker.port}/health", timeout=0.25
            ) as response:
                payload = json.loads(response.read())
        except (OSError, TimeoutError, json.JSONDecodeError, urllib.error.URLError) as exc:
            return pending_or_timeout(f"health endpoint is not ready: {exc}")
        if not isinstance(payload, dict):
            raise ServerError(
                f"Huge server {worker.worker_id} health response is not an object"
            )
        for key, expected_value in expected_health.items():
            matches = (
                _normalize_uuid(str(payload.get(key, "")))
                == _normalize_uuid(str(expected_value))
                if key == "cuda_uuid"
                else payload.get(key) == expected_value
            )
            if not matches:
                raise ServerError(
                    f"Huge server {worker.worker_id} health {key}={payload.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        _update_episode_attempt(
            attempt,
            job,
            attempt_number,
            state="server_ready",
            process_pid=server.process.pid,
            process_group_id=server.process.pid,
        )
    except EpisodeAttemptError:
        raise
    except Exception as exc:
        raise EpisodeAttemptError(
            job,
            attempt_number,
            f"server startup failed for {worker.worker_id}: {type(exc).__name__}: {exc}",
        ) from exc
    worker.prepared_attempt_number = attempt_number
    worker.server_starting_attempt_number = None
    worker.server_started_monotonic = None
    worker.server_started_wall_time = None
    return True


def _clear_worker_server_state(
    worker: WorkerRuntime, expected_server: ChildProcess
) -> None:
    if worker.server is not expected_server:
        raise SimulationError(f"worker {worker.worker_id} server ownership drifted")
    worker.server = None
    worker.prepared_attempt_number = None
    worker.server_starting_attempt_number = None
    worker.server_started_monotonic = None
    worker.server_started_wall_time = None


def _stop_worker_server(worker: WorkerRuntime, *, grace: float) -> None:
    server = worker.server
    if server is None:
        return
    _terminate_children([server], grace)
    _clear_worker_server_state(worker, server)


def _validate_live_server(worker: WorkerRuntime, *, attempt: Path, run_id: str) -> None:
    if worker.server is None:
        raise ServerError(f"Huge server {worker.worker_id} is unavailable")
    returncode = worker.server.process.poll()
    if returncode is not None:
        raise ServerError(f"Huge server {worker.worker_id} exited early with {returncode}")
    status_path = attempt / "status" / f"server-{worker.worker_id}.json"
    if not status_path.is_file():
        raise SimulationError(f"server status is missing for {worker.worker_id}")
    status = _json_object(
        status_path,
        label="Huge server status",
        attempt=attempt,
    )
    if status.get("run_id") != run_id or status.get("worker_id") != worker.worker_id:
        raise SimulationError(f"server status identity mismatch for {worker.worker_id}")
    _validate_live_server_status(
        status,
        worker_id=worker.worker_id,
        allow_infer_in_progress=worker.active_shadow is not None,
    )


def _observe_shadow_binding(worker: WorkerRuntime, *, run_id: str) -> bool:
    if worker.active_binding_verified:
        return True
    shadow = worker.active_shadow
    job = worker.active_job
    session_id = worker.active_session_id
    status_path = worker.active_status_path
    if shadow is None or job is None or session_id is None or status_path is None:
        raise SimulationError(f"worker {worker.worker_id} has incomplete active shadow state")
    if not status_path.is_file():
        return False
    status = _json_object(
        status_path,
        label="shadow session status",
        attempt=status_path.parent.parent,
    )
    if (
        status.get("schema_version") != WORKER_STATUS_SCHEMA_VERSION
        or status.get("run_id") != run_id
        or status.get("worker_id") != worker.worker_id
        or status.get("shadow_session_id") != session_id
        or status.get("simulator_pid") != shadow.process.pid
        or not _strict_episode_identity_matches(
            status.get("episode_identity"), _expected_shadow_episode_identity(job)
        )
    ):
        raise SimulationError(f"shadow startup status identity mismatch for {worker.worker_id}")
    state = str(status.get("state", ""))
    if state in {"binding_verified", "shutdown_ready", "complete"}:
        worker.active_binding_verified = True
    elif state != "starting":
        raise SimulationError(
            f"shadow {worker.worker_id} entered unexpected startup state {state!r}"
        )
    return worker.active_binding_verified


def _shadow_guard_removal_marker_path(status_path: Path) -> Path:
    return status_path.with_name(
        f".{status_path.name}.shutdown_guard.removed_by_orchestrator.json"
    )


def _read_owned_regular_bytes(
    path: Path,
    *,
    label: str,
) -> tuple[bytes, os.stat_result]:
    try:
        path_info = path.lstat()
    except OSError as exc:
        raise UnsafeCleanupError(f"cannot inspect {label} {path}: {exc}") from exc
    if (
        stat.S_ISLNK(path_info.st_mode)
        or not stat.S_ISREG(path_info.st_mode)
        or path_info.st_uid != os.getuid()
    ):
        raise UnsafeCleanupError(f"{label} ownership mismatch: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise UnsafeCleanupError(f"cannot open {label} {path}: {exc}") from exc
    try:
        opened_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_info.st_mode)
            or opened_info.st_uid != os.getuid()
            or opened_info.st_dev != path_info.st_dev
            or opened_info.st_ino != path_info.st_ino
        ):
            raise UnsafeCleanupError(f"{label} identity changed while reading: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            size += len(block)
            if size > 1024 * 1024:
                raise UnsafeCleanupError(f"{label} is unexpectedly large: {path}")
            chunks.append(block)
        return b"".join(chunks), opened_info
    except OSError as exc:
        raise UnsafeCleanupError(f"cannot read {label} {path}: {exc}") from exc
    finally:
        os.close(descriptor)


def _validate_shadow_guard_removal_marker(
    marker: Path,
    *,
    run_id: str,
    worker_id: str,
    session_id: str,
) -> dict[str, Any]:
    content, _ = _read_owned_regular_bytes(
        marker,
        label="shadow guard-removal provenance marker",
    )
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnsafeCleanupError(
            f"invalid shadow guard-removal provenance marker {marker}: {exc}"
        ) from exc
    removed_at = payload.get("removed_at") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != _SHADOW_GUARD_REMOVAL_SCHEMA_VERSION
        or payload.get("run_id") != run_id
        or payload.get("worker_id") != worker_id
        or payload.get("shadow_session_id") != session_id
        or payload.get("removed_by") != "orchestrator"
        or isinstance(removed_at, bool)
        or not isinstance(removed_at, (int, float))
        or not math.isfinite(float(removed_at))
        or float(removed_at) <= 0.0
    ):
        raise UnsafeCleanupError(
            f"shadow guard-removal provenance identity mismatch: {marker}"
        )
    return payload


def _publish_shadow_guard_removal_marker(
    marker: Path,
    *,
    run_id: str,
    worker_id: str,
    session_id: str,
) -> None:
    if os.path.lexists(marker):
        _validate_shadow_guard_removal_marker(
            marker,
            run_id=run_id,
            worker_id=worker_id,
            session_id=session_id,
        )
        return

    removed_at = float(time.time())
    if not math.isfinite(removed_at) or removed_at <= 0.0:
        raise UnsafeCleanupError("cannot create shadow guard-removal provenance timestamp")
    payload = {
        "schema_version": _SHADOW_GUARD_REMOVAL_SCHEMA_VERSION,
        "run_id": run_id,
        "worker_id": worker_id,
        "shadow_session_id": session_id,
        "removed_by": "orchestrator",
        "removed_at": removed_at,
    }
    content = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = marker.with_name(
        f".{marker.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    try:
        try:
            with temporary.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            persisted, _ = _read_owned_regular_bytes(
                temporary,
                label="temporary shadow guard-removal provenance marker",
            )
            if persisted != content:
                raise UnsafeCleanupError(
                    f"temporary shadow guard-removal provenance was truncated: {temporary}"
                )
            _validate_shadow_guard_removal_marker(
                temporary,
                run_id=run_id,
                worker_id=worker_id,
                session_id=session_id,
            )
            try:
                os.link(temporary, marker)
            except FileExistsError:
                _validate_shadow_guard_removal_marker(
                    marker,
                    run_id=run_id,
                    worker_id=worker_id,
                    session_id=session_id,
                )
            _fsync_directory(marker.parent)
            _validate_shadow_guard_removal_marker(
                marker,
                run_id=run_id,
                worker_id=worker_id,
                session_id=session_id,
            )
        finally:
            temporary.unlink(missing_ok=True)
    except UnsafeCleanupError:
        raise
    except OSError as exc:
        raise UnsafeCleanupError(
            f"cannot publish shadow guard-removal provenance marker {marker}: {exc}"
        ) from exc


def _remove_owned_shutdown_guard(
    status_path: Path,
    *,
    run_id: str,
    worker_id: str,
    session_id: str,
) -> None:
    try:
        _require_owned_attempt_path(
            status_path.parent.parent,
            status_path.parent,
            label="shadow status directory",
            directory=True,
        )
    except ConfigError as exc:
        raise UnsafeCleanupError(str(exc)) from exc
    guard = status_path.with_name(f".{status_path.name}.shutdown_guard")
    marker = _shadow_guard_removal_marker_path(status_path)
    if not os.path.lexists(guard):
        if os.path.lexists(marker):
            _validate_shadow_guard_removal_marker(
                marker,
                run_id=run_id,
                worker_id=worker_id,
                session_id=session_id,
            )
        return
    content, info = _read_owned_regular_bytes(
        guard,
        label="shadow shutdown guard",
    )
    expected = f"{run_id} {worker_id} {session_id}\n".encode("ascii")
    if content != expected:
        raise UnsafeCleanupError(f"shadow shutdown guard ownership mismatch: {guard}")
    _publish_shadow_guard_removal_marker(
        marker,
        run_id=run_id,
        worker_id=worker_id,
        session_id=session_id,
    )
    current_content, current_info = _read_owned_regular_bytes(
        guard,
        label="shadow shutdown guard",
    )
    if (
        current_content != expected
        or current_info.st_dev != info.st_dev
        or current_info.st_ino != info.st_ino
    ):
        raise UnsafeCleanupError(f"shadow shutdown guard identity changed: {guard}")
    guard.unlink()
    _fsync_directory(guard.parent)


def _clear_worker_shadow_state(worker: WorkerRuntime) -> None:
    worker.active_shadow = None
    worker.active_job = None
    worker.active_shm = None
    worker.active_session_id = None
    worker.active_status_path = None
    worker.active_log_path = None
    worker.active_started_monotonic = None
    worker.active_binding_verified = False
    worker.active_attempt_number = None


def _abort_worker_shadow(
    worker: WorkerRuntime,
    *,
    run_id: str,
    grace: float,
) -> int | None:
    shadow = worker.active_shadow
    shm = worker.active_shm
    session_id = worker.active_session_id
    status_path = worker.active_status_path
    if shadow is None or shm is None or session_id is None or status_path is None:
        raise SimulationError(f"worker {worker.worker_id} cannot abort an incomplete active shadow")
    try:
        _terminate_children([shadow], grace)
    except Exception as exc:
        raise UnsafeCleanupError(
            f"worker {worker.worker_id} shadow process group could not be reaped: {exc}"
        ) from exc
    return _release_worker_shadow_resources(worker, run_id=run_id)


def _release_worker_shadow_resources(
    worker: WorkerRuntime,
    *,
    run_id: str,
) -> int | None:
    shadow = worker.active_shadow
    shm = worker.active_shm
    session_id = worker.active_session_id
    status_path = worker.active_status_path
    if shadow is None or shm is None or session_id is None or status_path is None:
        raise SimulationError(
            f"worker {worker.worker_id} cannot release incomplete active shadow resources"
        )
    _remove_owned_shutdown_guard(
        status_path,
        run_id=run_id,
        worker_id=worker.worker_id,
        session_id=session_id,
    )
    shm.observe()
    if not shm.cleanup():
        raise UnsafeCleanupError(f"SHM cleanup identity failed for {shm.name}")
    returncode = shadow.process.poll()
    _clear_worker_shadow_state(worker)
    return returncode


def _start_worker_cleanup(
    worker: WorkerRuntime,
    executor: concurrent.futures.ThreadPoolExecutor,
    *,
    children: Sequence[ChildProcess],
    shadow: ChildProcess | None,
    server: ChildProcess | None,
    job: EpisodeJob | None,
    attempt_number: int | None,
    reason: str,
    error: str,
    consume_attempt: bool,
    shutdown_grace_sec: float,
) -> None:
    if worker.cleanup is not None:
        raise SimulationError(f"worker {worker.worker_id} already has cleanup in progress")
    owned_children = tuple(children)
    if not owned_children or any(
        child.worker_id != worker.worker_id for child in owned_children
    ):
        raise SimulationError(f"worker {worker.worker_id} cleanup children are invalid")
    if shadow is not None and shadow not in owned_children:
        raise SimulationError(f"worker {worker.worker_id} shadow cleanup ownership drifted")
    if server is not None and server not in owned_children:
        raise SimulationError(f"worker {worker.worker_id} server cleanup ownership drifted")
    if consume_attempt and (job is None or attempt_number is None):
        raise SimulationError(f"worker {worker.worker_id} cleanup lacks retry identity")
    future = executor.submit(
        _terminate_children,
        owned_children,
        shutdown_grace_sec,
    )
    worker.cleanup = WorkerCleanup(
        future=future,
        shadow=shadow,
        server=server,
        job=job,
        attempt_number=attempt_number,
        reason=reason,
        error=error,
        consume_attempt=consume_attempt,
    )


def _finish_worker_cleanup(
    worker: WorkerRuntime,
    *,
    attempt: Path,
    run_id: str,
    max_retries: int,
    backoff_sec: float,
) -> tuple[str, int | None, bool | None] | None:
    cleanup = worker.cleanup
    if cleanup is None or not cleanup.future.done():
        return None
    cleanup.future.result()
    returncode: int | None = None
    if cleanup.shadow is not None:
        if worker.active_shadow is not cleanup.shadow:
            raise SimulationError(f"worker {worker.worker_id} shadow cleanup identity drifted")
        returncode = _release_worker_shadow_resources(worker, run_id=run_id)
    if cleanup.server is not None:
        _clear_worker_server_state(worker, cleanup.server)
    worker.cleanup = None
    exhausted: bool | None = None
    if cleanup.consume_attempt:
        if cleanup.job is None or cleanup.attempt_number is None:
            raise SimulationError(f"worker {worker.worker_id} cleanup lost retry identity")
        exhausted = _schedule_episode_failure(
            worker,
            attempt=attempt,
            job=cleanup.job,
            attempt_number=cleanup.attempt_number,
            max_retries=max_retries,
            backoff_sec=backoff_sec,
            reason=cleanup.reason,
            error=cleanup.error,
            returncode=returncode,
            now=time.monotonic(),
        )
    return cleanup.reason, cleanup.attempt_number, exhausted


def _process_group_has_live_members(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        members = [
            row
            for row in _namespace_process_table()
            if row["process_group_id"] == process_group_id
        ]
    except UnsafeCleanupError:
        return True
    if members:
        return any(row["state"] != "Z" for row in members)
    # Some container runtimes expose a host /proc while Popen reports namespace-local
    # PIDs.  If killpg proves the group exists but /proc cannot identify its members,
    # fail closed and keep treating the group as live.
    return True


def _signal_process_group(process_group_id: int, signum: int) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise UnsafeCleanupError(
            f"cannot signal owned process group {process_group_id}: {exc}"
        ) from exc


def _signal_child_process_group(child: ChildProcess, signum: int) -> None:
    process_group_id = child.process.pid
    if not _process_group_has_live_members(process_group_id):
        return
    returncode = child.process.poll()
    current = _read_process_identity(process_group_id)
    has_recorded_identity = (
        child.process_start_ticks is not None
        and re.fullmatch(r"[a-f0-9]{64}", child.command_sha256) is not None
    )
    if current is not None:
        expected = {
            "process_group_id": process_group_id,
            "start_ticks": child.process_start_ticks,
            "command_sha256": child.command_sha256,
            "uid": os.getuid(),
        }
        if not has_recorded_identity or current != expected:
            raise UnsafeCleanupError(
                f"refusing to signal reused or unverified PID/PGID {process_group_id}"
            )
    elif returncode is None and has_recorded_identity:
        returncode = child.process.poll()
        if returncode is None:
            raise UnsafeCleanupError(
                f"cannot revalidate live child process identity for PID {process_group_id}"
            )
    _signal_process_group(process_group_id, signum)


def _prove_persisted_process_group(
    process_group_id: int,
    *,
    process_start_ticks: int,
    process_command_sha256: str,
    owner_uid: int,
) -> bool:
    try:
        rows = _namespace_process_table()
    except UnsafeCleanupError:
        raise
    members = [
        row for row in rows if row["process_group_id"] == process_group_id
    ]
    if not members:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise UnsafeCleanupError(
                f"cannot inspect persisted process group {process_group_id}: {exc}"
            ) from exc
        raise UnsafeCleanupError(
            f"cannot enumerate live persisted process group {process_group_id}"
        )
    for member in members:
        if (
            member["uid"] != owner_uid
            or member["session_id"] != process_group_id
            or member["pid"] < 1
            or member["start_ticks"] < 1
        ):
            raise UnsafeCleanupError(
                f"persisted process group {process_group_id} contains an unowned member"
            )

    leaders = [row for row in rows if row["pid"] == process_group_id]
    if len(leaders) > 1:
        raise UnsafeCleanupError(
            f"persisted leader PID {process_group_id} is ambiguous"
        )
    current = _read_process_identity(process_group_id)
    expected = {
        "process_group_id": process_group_id,
        "start_ticks": process_start_ticks,
        "command_sha256": process_command_sha256,
        "uid": owner_uid,
    }
    if current is not None:
        if current != expected:
            raise UnsafeCleanupError(
                f"live PID {process_group_id} no longer matches the persisted identity"
            )
    elif leaders:
        leader = leaders[0]
        if (
            leader["state"] != "Z"
            or leader["process_group_id"] != process_group_id
            or leader["session_id"] != process_group_id
            or leader["uid"] != owner_uid
            or leader["start_ticks"] != process_start_ticks
        ):
            raise UnsafeCleanupError(
                f"leader PID {process_group_id} was reused or cannot be proven"
            )
    # On Linux the numeric PID backing a process-group ID cannot be reused while
    # members of that group remain.  Therefore, once the original session leader is
    # absent, a completely enumerated same-session/same-UID residual group still
    # proves continuity with the persisted group.
    return any(member["state"] != "Z" for member in members)


def _signal_persisted_process_group(
    process_group_id: int,
    signum: int,
    *,
    process_start_ticks: int,
    process_command_sha256: str,
    owner_uid: int,
) -> None:
    if not _process_group_has_live_members(process_group_id):
        return
    if not _prove_persisted_process_group(
        process_group_id,
        process_start_ticks=process_start_ticks,
        process_command_sha256=process_command_sha256,
        owner_uid=owner_uid,
    ):
        return
    _signal_process_group(process_group_id, signum)


def _terminate_persisted_process_group(
    process_group_id: int,
    grace: float,
    *,
    process_start_ticks: int,
    process_command_sha256: str,
    owner_uid: int,
) -> None:
    if not _process_group_has_live_members(process_group_id):
        return
    _signal_persisted_process_group(
        process_group_id,
        signal.SIGTERM,
        process_start_ticks=process_start_ticks,
        process_command_sha256=process_command_sha256,
        owner_uid=owner_uid,
    )
    deadline = time.monotonic() + grace
    while (
        time.monotonic() < deadline
        and _process_group_has_live_members(process_group_id)
    ):
        time.sleep(0.1)
    if _process_group_has_live_members(process_group_id):
        _signal_persisted_process_group(
            process_group_id,
            signal.SIGKILL,
            process_start_ticks=process_start_ticks,
            process_command_sha256=process_command_sha256,
            owner_uid=owner_uid,
        )
    deadline = time.monotonic() + 5.0
    while (
        time.monotonic() < deadline
        and _process_group_has_live_members(process_group_id)
    ):
        time.sleep(0.05)
    if _process_group_has_live_members(process_group_id):
        raise UnsafeCleanupError(
            f"persisted process group {process_group_id} remained alive after shutdown"
        )


def _terminate_children(children: Sequence[ChildProcess], grace: float) -> None:
    unique = list({id(child): child for child in children}.values())
    groups = {child.process.pid: child for child in unique}
    for child in unique:
        child.process.poll()
    for process_group_id, child in groups.items():
        if _process_group_has_live_members(process_group_id):
            _signal_child_process_group(child, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while (
        time.monotonic() < deadline
        and any(_process_group_has_live_members(group) for group in groups)
    ):
        for child in unique:
            child.process.poll()
        time.sleep(0.1)
    for process_group_id, child in groups.items():
        if _process_group_has_live_members(process_group_id):
            _signal_child_process_group(child, signal.SIGKILL)
    unreaped: list[ChildProcess] = []
    for child in unique:
        try:
            child.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            unreaped.append(child)
        finally:
            child.log_stream.close()
    group_deadline = time.monotonic() + 5.0
    live_groups = {
        group for group in groups if _process_group_has_live_members(group)
    }
    while live_groups and time.monotonic() < group_deadline:
        time.sleep(0.05)
        live_groups = {
            group for group in live_groups if _process_group_has_live_members(group)
        }
    if live_groups:
        labels = sorted(live_groups)
        if any(child.role == "server" for child in unique):
            raise ServerError(f"process groups remained alive after shutdown: {labels}")
        raise SimulationError(f"process groups remained alive after shutdown: {labels}")
    if unreaped:
        labels = [f"{child.role}:{child.worker_id}:{child.process.pid}" for child in unreaped]
        if any(child.role == "server" for child in unreaped):
            raise ServerError(f"child processes remained alive after shutdown: {labels}")
        raise SimulationError(f"child processes remained alive after shutdown: {labels}")


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
    if os.path.lexists(path):
        existing = _json_object(path, label="attempt manifest", attempt=attempt)
    existing.update({
        "schema_version": MANIFEST_SCHEMA_VERSION, "contract_hash": contract_hash,
        "run_id": attempt.name, "state": state, "runtime": runtime,
    })
    atomic_write_json(path, existing)


def _require_owned_directory(path: Path, *, label: str) -> None:
    if not os.path.lexists(path):
        raise SimulationError(f"missing {label}: {path}")
    try:
        info = path.lstat()
    except OSError as exc:
        raise SimulationError(f"cannot inspect {label} {path}: {exc}") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
    ):
        raise SimulationError(f"{label} ownership mismatch: {path}")


def _validate_gpu_admission_payload(
    payload: Any,
    *,
    run_id: str,
    contract_hash: str,
    ordinal: int,
    path: Path,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "run_id",
        "contract_hash",
        "ordinal",
        "captured_at",
        "threshold_gib",
        "allowed_compute_pid",
        "compute_apps",
        "gpus",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise SimulationError(f"GPU admission payload shape is invalid: {path}")
    captured_at = payload.get("captured_at")
    threshold_gib = payload.get("threshold_gib")
    allowed_compute_pid = payload.get("allowed_compute_pid")
    compute_apps = payload.get("compute_apps")
    gpus = payload.get("gpus")
    if (
        payload.get("schema_version") != GPU_ADMISSION_SCHEMA_VERSION
        or payload.get("run_id") != run_id
        or payload.get("contract_hash") != contract_hash
        or type(payload.get("ordinal")) is not int
        or payload.get("ordinal") != ordinal
        or type(captured_at) is not float
        or not math.isfinite(captured_at)
        or captured_at <= 0.0
        or type(threshold_gib) is not float
        or not math.isfinite(threshold_gib)
        or threshold_gib < 0.0
        or type(allowed_compute_pid) is not int
        or allowed_compute_pid < 1
        or not isinstance(compute_apps, list)
        or not isinstance(gpus, list)
        or not gpus
    ):
        raise SimulationError(f"GPU admission payload values are invalid: {path}")

    expected_gpu_keys = {
        "parent_logical_ordinal",
        "cuda_uuid",
        "total_memory_bytes",
        "free_memory_bytes",
    }
    logical_ordinals: set[int] = set()
    normalized_uuids: set[str] = set()
    for gpu in gpus:
        if not isinstance(gpu, dict) or set(gpu) != expected_gpu_keys:
            raise SimulationError(f"GPU admission identity shape is invalid: {path}")
        logical_ordinal = gpu.get("parent_logical_ordinal")
        cuda_uuid = gpu.get("cuda_uuid")
        total_memory_bytes = gpu.get("total_memory_bytes")
        free_memory_bytes = gpu.get("free_memory_bytes")
        if (
            type(logical_ordinal) is not int
            or logical_ordinal < 0
            or logical_ordinal in logical_ordinals
            or type(cuda_uuid) is not str
            or type(total_memory_bytes) is not int
            or total_memory_bytes <= 0
            or type(free_memory_bytes) is not int
            or free_memory_bytes < 0
        ):
            raise SimulationError(f"GPU admission identity values are invalid: {path}")
        try:
            normalized_uuid = _normalize_uuid(cuda_uuid)
        except GpuError as exc:
            raise SimulationError(
                f"GPU admission UUID is invalid in {path}: {cuda_uuid!r}"
            ) from exc
        if not normalized_uuid or normalized_uuid in normalized_uuids:
            raise SimulationError(f"GPU admission UUIDs are not unique: {path}")
        logical_ordinals.add(logical_ordinal)
        normalized_uuids.add(normalized_uuid)
    seen_apps: set[tuple[str, int]] = set()
    for app in compute_apps:
        if not isinstance(app, dict) or set(app) != {"cuda_uuid", "pid", "allowed"}:
            raise SimulationError(f"GPU admission compute-app shape is invalid: {path}")
        app_uuid = app.get("cuda_uuid")
        pid = app.get("pid")
        try:
            normalized_app_uuid = _normalize_uuid(str(app_uuid))
        except GpuError as exc:
            raise SimulationError(
                f"GPU admission compute-app UUID is invalid: {path}"
            ) from exc
        if (
            type(app_uuid) is not str
            or normalized_app_uuid not in normalized_uuids
            or type(pid) is not int
            or pid < 1
            or (normalized_app_uuid, pid) in seen_apps
            or app.get("allowed") is not True
            or pid != allowed_compute_pid
        ):
            raise SimulationError(f"GPU admission compute-app values are invalid: {path}")
        seen_apps.add((normalized_app_uuid, pid))
    return payload


def _read_gpu_admission_records(
    attempt: Path,
    *,
    run_id: str,
    contract_hash: str,
) -> list[dict[str, Any]]:
    directory = attempt / "gpu_admission"
    _require_owned_directory(directory, label="GPU admission directory")
    try:
        entries = sorted(directory.iterdir(), key=lambda candidate: candidate.name)
    except OSError as exc:
        raise SimulationError(
            f"cannot enumerate GPU admission directory {directory}: {exc}"
        ) from exc
    records: list[dict[str, Any]] = []
    for expected_ordinal, path in enumerate(entries, start=1):
        match = _GPU_ADMISSION_FILENAME_RE.fullmatch(path.name)
        if match is None or int(match.group(1)) != expected_ordinal:
            raise SimulationError(
                f"GPU admission evidence is unexpected or non-contiguous: {path}"
            )
        content, _ = _read_owned_regular_bytes(path, label="GPU admission evidence")
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SimulationError(f"invalid GPU admission evidence {path}: {exc}") from exc
        records.append(
            _validate_gpu_admission_payload(
                payload,
                run_id=run_id,
                contract_hash=contract_hash,
                ordinal=expected_ordinal,
                path=path,
            )
        )
    return records


def _append_gpu_admission(
    attempt: Path,
    *,
    run_id: str,
    contract_hash: str,
    threshold_gib: float,
    captured_at: float,
    gpus: Sequence[GpuIdentity],
    allowed_compute_pid: int,
    compute_apps: Sequence[Mapping[str, Any]],
) -> int:
    if run_id != attempt.name:
        raise SimulationError(
            f"GPU admission run identity mismatch: {run_id!r} != {attempt.name!r}"
        )
    records = _read_gpu_admission_records(
        attempt,
        run_id=run_id,
        contract_hash=contract_hash,
    )
    ordinal = len(records) + 1
    if ordinal > 9999:
        raise SimulationError("GPU admission evidence exhausted its four-digit ordinal space")
    payload = {
        "schema_version": GPU_ADMISSION_SCHEMA_VERSION,
        "run_id": run_id,
        "contract_hash": contract_hash,
        "ordinal": ordinal,
        "captured_at": captured_at,
        "threshold_gib": threshold_gib,
        "allowed_compute_pid": allowed_compute_pid,
        "compute_apps": [dict(app) for app in compute_apps],
        "gpus": [
            {
                "parent_logical_ordinal": gpu.parent_logical_ordinal,
                "cuda_uuid": gpu.cuda_uuid,
                "total_memory_bytes": gpu.total_memory_bytes,
                "free_memory_bytes": gpu.free_memory_bytes,
            }
            for gpu in gpus
        ],
    }
    final_path = attempt / "gpu_admission" / f"admission-{ordinal:04d}.json"
    _validate_gpu_admission_payload(
        payload,
        run_id=run_id,
        contract_hash=contract_hash,
        ordinal=ordinal,
        path=final_path,
    )
    if os.path.lexists(final_path):
        raise SimulationError(f"GPU admission evidence already exists: {final_path}")

    staging = attempt / ".gpu_admission_staging"
    _require_owned_directory(staging, label="GPU admission staging directory")
    descriptor: int | None = None
    temporary: Path | None = None
    for _ in range(100):
        candidate = staging / (
            f"admission-{ordinal:04d}.{os.getpid()}.{time.time_ns()}."
            f"{secrets.token_hex(8)}.tmp"
        )
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            continue
        temporary = candidate
        break
    if descriptor is None or temporary is None:
        raise SimulationError("unable to allocate a unique GPU admission staging file")

    content = canonical_json_bytes(payload) + b"\n"
    operation_error: BaseException | None = None
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("GPU admission staging write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        persisted, _ = _read_owned_regular_bytes(
            temporary,
            label="temporary GPU admission evidence",
        )
        if persisted != content:
            raise SimulationError(
                f"temporary GPU admission evidence was truncated: {temporary}"
            )
        try:
            persisted_payload = json.loads(persisted.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SimulationError(
                f"temporary GPU admission evidence is invalid: {temporary}: {exc}"
            ) from exc
        _validate_gpu_admission_payload(
            persisted_payload,
            run_id=run_id,
            contract_hash=contract_hash,
            ordinal=ordinal,
            path=temporary,
        )
        try:
            os.link(temporary, final_path, follow_symlinks=False)
        except OSError as exc:
            raise SimulationError(
                f"cannot publish GPU admission evidence {final_path}: {exc}"
            ) from exc
        _fsync_directory(final_path.parent)
        return ordinal
    except BaseException as exc:
        operation_error = exc
        raise
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                if operation_error is None:
                    raise
        try:
            temporary.unlink(missing_ok=True)
            _fsync_directory(staging)
        except OSError:
            if operation_error is None:
                raise


def _record_gpu_admission(
    attempt: Path,
    contract_hash: str,
    *,
    threshold_gib: float,
    captured_at: float,
    gpus: Sequence[GpuIdentity],
    allowed_compute_pid: int,
    compute_apps: Sequence[Mapping[str, Any]],
) -> int:
    try:
        _write_manifest(
            attempt,
            contract_hash,
            "starting",
            phase="admission_pending",
        )
        ordinal = _append_gpu_admission(
            attempt,
            run_id=attempt.name,
            contract_hash=contract_hash,
            threshold_gib=threshold_gib,
            captured_at=captured_at,
            gpus=gpus,
            allowed_compute_pid=allowed_compute_pid,
            compute_apps=compute_apps,
        )
        _write_manifest(
            attempt,
            contract_hash,
            "starting",
            phase="admission_recorded",
            gpu_admission_ordinal=ordinal,
        )
        return ordinal
    except Exception as exc:
        try:
            _write_manifest(
                attempt,
                contract_hash,
                "failed",
                phase="admission_failed",
                error_type=type(exc).__name__,
                error=str(exc)[:2000],
            )
        except Exception:
            pass
        raise


def _finalize(
    attempt: Path,
    jobs: Sequence[EpisodeJob],
    contract_hash: str,
    source_digest: str,
    reused: int,
) -> int:
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
    execution_failed = sum(
        episode["payload"].get("failure_reason") == _EXECUTION_FAILED
        for episode in episodes
    )
    steps = sorted(int(episode["payload"].get("episode_steps", 0)) for episode in episodes)
    durations = sorted(float(episode["payload"].get("duration_sec", 0.0)) for episode in episodes)
    percentile = lambda values, fraction: values[min(len(values) - 1, int(round((len(values) - 1) * fraction)))]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION, "contract_hash": contract_hash,
        "source_inventory_digest": source_digest, "expected": len(jobs), "completed": len(episodes),
        "task_success": successes, "task_failure": task_failures, "infrastructure_failure": 0,
        "execution_failed": execution_failed,
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
    rows = [
        "metric,value",
        f"expected,{len(jobs)}",
        f"completed,{len(episodes)}",
        f"task_success,{successes}",
        f"task_failure,{task_failures}",
        f"execution_failed,{execution_failed}",
        f"resume_reused,{reused}",
    ]
    atomic_write_text(attempt / "summary.csv", "\n".join(rows) + "\n")
    marker_digest = canonical_sha256(summary) + "\n"
    if execution_failed:
        (attempt / "_SUCCESS").unlink(missing_ok=True)
        _write_manifest(
            attempt,
            contract_hash,
            "completed_with_failures",
            completed=len(jobs),
            reused=reused,
            execution_failed=execution_failed,
        )
        atomic_write_text(attempt / _COMPLETED_WITH_FAILURES_MARKER, marker_digest)
    else:
        (attempt / _COMPLETED_WITH_FAILURES_MARKER).unlink(missing_ok=True)
        _write_manifest(
            attempt,
            contract_hash,
            "success",
            completed=len(jobs),
            reused=reused,
            execution_failed=0,
        )
        atomic_write_text(attempt / "_SUCCESS", marker_digest)
    finalizing.unlink()
    _fsync_directory(attempt)
    return execution_failed


def _validate_controlled_retry_required(
    contract_dir: Path,
    jobs: Sequence[EpisodeJob],
    contract_hash: str,
    source_digest: str,
) -> Path:
    attempts_root = contract_dir / "attempts"
    if not attempts_root.is_dir():
        raise ConfigError(
            f"retry_required contract has no attempts directory: {contract_dir}"
        )
    try:
        candidates = _canonical_attempt_candidates(attempts_root)
    except OSError as exc:
        raise ConfigError(
            f"cannot inspect retry_required attempts: {attempts_root}: {exc}"
        ) from exc
    if not candidates:
        raise ConfigError(
            f"retry_required contract has no canonical attempt: {contract_dir}"
        )
    attempt = _validate_canonical_attempt_entry(
        candidates[-1], attempts_root
    )
    manifest = _json_object(
        attempt / "manifest.json",
        label="retry_required attempt manifest",
        attempt=attempt,
    )
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("contract_hash") != contract_hash
        or manifest.get("run_id") != attempt.name
        or manifest.get("state") != "completed_with_failures"
        or not (attempt / _COMPLETED_WITH_FAILURES_MARKER).is_file()
        or os.path.lexists(attempt / "_SUCCESS")
    ):
        raise ConfigError(
            f"retry_required does not reference the latest valid failed "
            f"terminal attempt: {attempt}"
        )
    execution_failed = _verify_completed_attempt(
        attempt, jobs, contract_hash, source_digest
    )
    if execution_failed <= 0:
        raise ConfigError(
            f"retry_required attempt has no verified execution failures: "
            f"{attempt}"
        )
    return attempt


def _execute(
    args: argparse.Namespace, task: TaskSpec, checkpoint: CheckpointContract, policy: Mapping[str, Any],
    jobs: Sequence[EpisodeJob], source_inventory: Mapping[str, Any], static_plan: Mapping[str, Any],
) -> int:
    gpus, driver, _ = discover_visible_gpus(policy)
    fresh_gpu_admission_snapshot(
        gpus,
        threshold_gib=float(policy["min_free_gpu_memory_gib"]),
        allowed_pid=os.getpid(),
    )
    host = host_fingerprint()
    isaac_version, kit_build = isaac_fingerprint(args.isaaclab_python, args.isaaclab_root)
    output_root = Path(str(policy["output_root"]))
    map_path = Path(str(policy["kit_gpu_map_file"]))
    if not map_path.is_file():
        if not policy["allow_kit_map_bootstrap"]:
            raise GpuError(f"trusted Kit GPU map is missing: {map_path}")
        bootstrap_kit_gpu_map(map_path, gpus, host, driver, isaac_version, kit_build, args)
    physical_bindings, kit_digest = load_kit_gpu_map(
        map_path, gpus, host, driver, isaac_version, kit_build
    )
    worker_bindings = expand_worker_bindings(
        physical_bindings, int(policy["workers_per_gpu"])
    )
    topology_rows = [
        {
            "parent_logical_ordinal": binding.gpu.parent_logical_ordinal,
            "cuda_uuid": binding.gpu.cuda_uuid,
            "renderer_ordinal": binding.renderer_ordinal,
            "worker_slots": list(range(int(policy["workers_per_gpu"]))),
        }
        for binding in physical_bindings
    ]
    print(
        "[eval_orchestrator] topology="
        + json.dumps(
            {
                "physical_gpu_count": len(physical_bindings),
                "workers_per_gpu": int(policy["workers_per_gpu"]),
                "total_workers": len(worker_bindings),
                "min_free_gpu_memory_gib": float(
                    policy["min_free_gpu_memory_gib"]
                ),
                "model_noise_seed_algorithm": MODEL_NOISE_SEED_ALGORITHM,
                "bindings": topology_rows,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stderr,
        flush=True,
    )
    executable = build_executable_contract(static_plan, worker_bindings, kit_digest)
    contract_hash = executable["contract_hash"]
    source_digest = str(source_inventory["source_inventory_digest"])
    checkpoint_label = f"{checkpoint.checkpoint_task}-{checkpoint.checkpoint_dir.name}"
    contract_dir = output_root / task.cli_task / checkpoint_label / contract_hash[:16]
    retry_required_hash = str(
        policy.get("retry_required_contract_hash", "")
    )
    if retry_required_hash and retry_required_hash != contract_hash:
        raise ConfigError(
            "retry_required previous contract hash does not equal the "
            f"current executable contract hash: previous={retry_required_hash}, "
            f"current={contract_hash}"
        )
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / "contract.json"
    contract_bytes = canonical_json_bytes(executable["contract"])
    if (
        contract_path.is_file()
        and canonical_json_bytes(_json_object(contract_path, label="executable contract"))
        != contract_bytes
    ):
        raise ConfigError("existing contract bytes do not match the newly computed contract")
    if not contract_path.exists():
        atomic_write_json(contract_path, executable["contract"])
        atomic_write_json(contract_dir / "source_inventory.json", source_inventory)
    else:
        stored_inventory = _json_object(
            contract_dir / "source_inventory.json", label="contract source inventory"
        )
        if canonical_json_bytes(stored_inventory) != canonical_json_bytes(source_inventory):
            raise ConfigError("stored source inventory does not match the executable contract")

    print(
        f"[eval_orchestrator] contract_path={contract_path} "
        f"contract_hash={contract_hash}",
        file=sys.stderr,
        flush=True,
    )
    with ContractLock(contract_dir / "contract.lock", contract_hash):
        if retry_required_hash:
            previous_attempt = _validate_controlled_retry_required(
                contract_dir,
                jobs,
                contract_hash,
                source_digest,
            )
            print(
                f"[eval_orchestrator] controlled_retry previous_attempt="
                f"{previous_attempt} contract_hash={contract_hash}",
                file=sys.stderr,
                flush=True,
            )
        attempt = select_or_create_attempt(
            contract_dir / "attempts",
            contract_hash,
            resume=bool(policy["resume"]),
            force_new=bool(policy["force_new"]),
        )
        run_id = attempt.name
        if (
            (attempt / "_SUCCESS").exists()
            or (attempt / _COMPLETED_WITH_FAILURES_MARKER).exists()
        ):
            execution_failed = _verify_completed_attempt(
                attempt, jobs, contract_hash, source_digest
            )
            return EXIT_SIM if execution_failed else 0
        try:
            _cleanup_persisted_runtime(
                attempt,
                jobs,
                shutdown_grace_sec=float(policy["shutdown_grace_sec"]),
            )
        except Exception as exc:
            _write_manifest(
                attempt,
                contract_hash,
                "unsafe_cleanup",
                cleanup_errors=[f"{type(exc).__name__}: {exc}"],
            )
            raise
        completed = {
            job.result_identity: _load_reusable_episode_result(
                attempt,
                job,
                max_retries=int(policy["episode_max_retries"]),
            )
            for job in jobs
        }
        reused = sum(value is not None for value in completed.values())
        initially_pending = [
            job for job in jobs if completed[job.result_identity] is None
        ]
        for job in initially_pending:
            _reconcile_unfinished_episode_attempts(attempt, job)
            records = _episode_attempt_records(attempt, job)
            if len(records) > int(policy["episode_max_retries"]) + 1:
                raise SimulationError(
                    f"episode retry ledger exceeds configured budget: {job.result_identity}"
                )
            if _episode_retry_exhausted(
                attempt,
                job,
                max_retries=int(policy["episode_max_retries"]),
            ):
                last = records[-1]
                _publish_execution_failed_completion(
                    attempt,
                    job,
                    worker_id=str(last["worker_id"]),
                    attempt_count=len(records),
                    last_reason=str(last.get("reason", "orchestrator_restarted")),
                    last_error=str(last.get("error", "")),
                )
                completed[job.result_identity] = validate_and_publish_completion(
                    attempt, job
                )
                if completed[job.result_identity] is None:
                    raise SimulationError(
                        f"cannot recover exhausted episode retry ledger: {job.result_identity}"
                    )

        completed_identities = {
            result_identity
            for result_identity, result in completed.items()
            if result is not None
        }
        remaining_by_worker = partition_remaining_jobs(
            jobs,
            len(worker_bindings),
            completed_identities,
        )
        if not remaining_by_worker:
            execution_failed = _finalize(
                attempt, jobs, contract_hash, source_digest, reused
            )
            return EXIT_SIM if execution_failed else 0

        worker_specs = [
            (worker_index, worker_bindings[worker_index], deque(shard))
            for worker_index, shard in remaining_by_worker
        ]
        active_bindings = [binding for _, binding, _ in worker_specs]
        refreshed_gpus, admission_captured_at, compute_apps = (
            fresh_gpu_admission_snapshot(
                gpus,
                threshold_gib=float(policy["min_free_gpu_memory_gib"]),
                allowed_pid=os.getpid(),
            )
        )
        _record_gpu_admission(
            attempt,
            contract_hash,
            threshold_gib=float(policy["min_free_gpu_memory_gib"]),
            captured_at=admission_captured_at,
            gpus=refreshed_gpus,
            allowed_compute_pid=os.getpid(),
            compute_apps=compute_apps,
        )
        reservations = _reserve_ports(
            int(policy["base_port"]), len(worker_specs)
        )
        runtime_env = _runtime_environment(task, static_plan["yaml"], policy)
        workers: list[WorkerRuntime] = []
        cleanup_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(worker_specs)),
            thread_name_prefix="huge-eval-cleanup",
        )
        frozen = False
        interrupted = False
        failure: Exception | None = None
        interrupted_failure: Exception | None = None
        try:
            _write_manifest(
                attempt,
                contract_hash,
                "starting",
                bindings=[asdict(binding) for binding in active_bindings],
            )
            atomic_write_json(attempt / "plan.json", executable["contract"])
            worker_rows: list[dict[str, Any]] = []
            for (worker_index, binding, pending), reservation in zip(
                worker_specs, reservations
            ):
                worker_id = f"w{worker_index:02d}"
                port = reservation.getsockname()[1]
                reservation.close()
                worker = WorkerRuntime(
                    worker_id=worker_id,
                    binding=binding,
                    server=None,
                    port=port,
                    pending=pending,
                )
                workers.append(worker)
                worker_rows.append(
                    {
                        "worker_id": worker_id,
                        "port": port,
                        "binding": asdict(binding),
                    }
                )
            _write_manifest(attempt, contract_hash, "running", workers=worker_rows)

            while True:
                if _CANCEL_REQUESTED:
                    frozen = True
                    interrupted = True

                for worker in workers:
                    cleanup_result = _finish_worker_cleanup(
                        worker,
                        attempt=attempt,
                        run_id=run_id,
                        max_retries=int(policy["episode_max_retries"]),
                        backoff_sec=float(policy["retry_backoff_sec"]),
                    )
                    if cleanup_result is not None:
                        reason, attempt_number, exhausted = cleanup_result
                        print(
                            f"[eval_orchestrator] {worker.worker_id} cleanup completed "
                            f"reason={reason} attempt={attempt_number} "
                            f"exhausted={'' if exhausted is None else int(exhausted)}",
                            file=sys.stderr,
                        )

                for worker in workers:
                    if worker.cleanup is not None:
                        continue
                    shadow = worker.active_shadow
                    if shadow is None:
                        continue
                    job = worker.active_job
                    attempt_number = worker.active_attempt_number
                    status_path = worker.active_status_path
                    session_id = worker.active_session_id
                    if (
                        job is None
                        or attempt_number is None
                        or status_path is None
                        or session_id is None
                        or worker.active_shm is None
                    ):
                        raise SimulationError(
                            f"worker {worker.worker_id} has incomplete active retry state"
                        )
                    worker.active_shm.observe()
                    returncode = shadow.process.poll()
                    if returncode is not None:
                        try:
                            _finish_worker_shadow(
                                worker,
                                attempt=attempt,
                                run_id=run_id,
                                returncode=returncode,
                            )
                        except UnsafeCleanupError:
                            raise
                        except Exception as exc:
                            server_to_stop = (
                                worker.server if isinstance(exc, ServerError) else None
                            )
                            cleanup_children = [shadow]
                            if server_to_stop is not None:
                                cleanup_children.append(server_to_stop)
                            reason = (
                                "server_or_inference_failure"
                                if isinstance(exc, ServerError)
                                else "shadow_abnormal_exit"
                            )
                            _start_worker_cleanup(
                                worker,
                                cleanup_executor,
                                children=cleanup_children,
                                shadow=shadow,
                                server=server_to_stop,
                                job=job,
                                attempt_number=attempt_number,
                                reason=reason,
                                error=f"{type(exc).__name__}: {exc}",
                                consume_attempt=True,
                                shutdown_grace_sec=float(
                                    policy["shutdown_grace_sec"]
                                ),
                            )
                            print(
                                f"[eval_orchestrator] {worker.worker_id} episode "
                                f"{job.episode_index} attempt {attempt_number} failed "
                                f"after shadow exit; cleanup started",
                                file=sys.stderr,
                            )
                        continue

                    watchdog_reason: str | None = None
                    watchdog_error = ""
                    try:
                        _observe_shadow_binding(worker, run_id=run_id)
                        watchdog_reason = _shadow_watchdog_reason(
                            worker,
                            now=time.monotonic(),
                            startup_timeout_sec=float(
                                policy["shadow_startup_timeout_sec"]
                            ),
                            episode_timeout_sec=float(policy["episode_timeout_sec"]),
                        )
                    except UnsafeCleanupError:
                        raise
                    except Exception as exc:
                        watchdog_reason = "shadow_status_invalid"
                        watchdog_error = f"{type(exc).__name__}: {exc}"
                    if watchdog_reason is not None:
                        _update_episode_attempt(
                            attempt,
                            job,
                            attempt_number,
                            state="terminating",
                            termination_reason=watchdog_reason,
                        )
                        server_to_stop = (
                            worker.server if watchdog_reason == "episode_timeout" else None
                        )
                        cleanup_children = [shadow]
                        if server_to_stop is not None:
                            cleanup_children.append(server_to_stop)
                        _start_worker_cleanup(
                            worker,
                            cleanup_executor,
                            children=cleanup_children,
                            shadow=shadow,
                            server=server_to_stop,
                            job=job,
                            attempt_number=attempt_number,
                            reason=watchdog_reason,
                            error=watchdog_error or watchdog_reason,
                            consume_attempt=True,
                            shutdown_grace_sec=float(policy["shutdown_grace_sec"]),
                        )
                        print(
                            f"[eval_orchestrator] {worker.worker_id} episode "
                            f"{job.episode_index} attempt {attempt_number} watchdog="
                            f"{watchdog_reason}; cleanup started",
                            file=sys.stderr,
                        )

                for worker in workers:
                    if (
                        worker.cleanup is not None
                        or worker.server is None
                        or worker.server_starting_attempt_number is not None
                        or (
                            not worker.pending and worker.active_shadow is None
                        )
                    ):
                        continue
                    try:
                        _validate_live_server(worker, attempt=attempt, run_id=run_id)
                    except (ServerError, SimulationError, ConfigError) as exc:
                        reason = (
                            "server_or_inference_failure"
                            if isinstance(exc, ServerError)
                            else "server_status_invalid"
                        )
                        server = worker.server
                        if server is None:
                            raise SimulationError(
                                f"worker {worker.worker_id} lost server cleanup ownership"
                            )
                        shadow_to_stop: ChildProcess | None = None
                        consume_attempt = False
                        if worker.active_shadow is not None:
                            job = worker.active_job
                            attempt_number = worker.active_attempt_number
                            if job is None or attempt_number is None:
                                raise SimulationError(
                                    f"worker {worker.worker_id} has incomplete server-failure state"
                                )
                            _update_episode_attempt(
                                attempt,
                                job,
                                attempt_number,
                                state="terminating",
                                termination_reason=reason,
                            )
                            shadow_to_stop = worker.active_shadow
                            consume_attempt = True
                        else:
                            if not worker.pending:
                                raise SimulationError(
                                    f"worker {worker.worker_id} lost its server without pending work"
                                )
                            job = worker.pending[0]
                            attempt_number = worker.prepared_attempt_number
                            consume_attempt = attempt_number is not None
                        cleanup_children = (
                            [shadow_to_stop, server]
                            if shadow_to_stop is not None
                            else [server]
                        )
                        _start_worker_cleanup(
                            worker,
                            cleanup_executor,
                            children=cleanup_children,
                            shadow=shadow_to_stop,
                            server=server,
                            job=job if consume_attempt else None,
                            attempt_number=attempt_number,
                            reason=reason,
                            error=f"{type(exc).__name__}: {exc}",
                            consume_attempt=consume_attempt,
                            shutdown_grace_sec=float(policy["shutdown_grace_sec"]),
                        )
                        print(
                            f"[eval_orchestrator] failed server cleanup started in fixed "
                            f"slot {worker.worker_id}; attempt={attempt_number}; "
                            f"consume_attempt={int(consume_attempt)}: {exc}",
                            file=sys.stderr,
                        )

                if not frozen:
                    shadow_startups_in_progress = sum(
                        worker.active_shadow is not None
                        and not worker.active_binding_verified
                        for worker in workers
                    )
                    for worker in workers:
                        if _CANCEL_REQUESTED:
                            frozen = True
                            interrupted = True
                            break
                        if (
                            worker.cleanup is not None
                            or worker.active_shadow is not None
                            or not worker.pending
                            or time.monotonic() < worker.retry_not_before_monotonic
                        ):
                            continue
                        if worker.server is None:
                            try:
                                _restart_worker_server(
                                    worker,
                                    args=args,
                                    task=task,
                                    checkpoint=checkpoint,
                                    run_id=run_id,
                                    attempt=attempt,
                                    policy=policy,
                                    contract_hash=contract_hash,
                                    source_inventory_digest=source_digest,
                                )
                            except EpisodeAttemptError as exc:
                                exhausted = _schedule_episode_failure(
                                    worker,
                                    attempt=attempt,
                                    job=exc.job,
                                    attempt_number=exc.attempt_number,
                                    max_retries=int(policy["episode_max_retries"]),
                                    backoff_sec=float(policy["retry_backoff_sec"]),
                                    reason="server_restart_failed",
                                    error=str(exc),
                                    returncode=None,
                                    now=time.monotonic(),
                                )
                                print(
                                    f"[eval_orchestrator] {worker.worker_id} server restart "
                                    f"attempt {exc.attempt_number} failed; "
                                    f"exhausted={int(exhausted)}",
                                    file=sys.stderr,
                                )
                                continue
                            continue
                        if worker.server_starting_attempt_number is not None:
                            try:
                                server_ready = _advance_worker_server_startup(
                                    worker,
                                    attempt=attempt,
                                    run_id=run_id,
                                    expected_health=_expected_server_health(
                                        task, checkpoint, worker.binding
                                    ),
                                    ready_timeout_sec=float(
                                        policy["server_ready_timeout"]
                                    ),
                                )
                            except EpisodeAttemptError as exc:
                                server = worker.server
                                if server is None:
                                    raise SimulationError(
                                        f"worker {worker.worker_id} lost startup cleanup ownership"
                                    )
                                _start_worker_cleanup(
                                    worker,
                                    cleanup_executor,
                                    children=[server],
                                    shadow=None,
                                    server=server,
                                    job=exc.job,
                                    attempt_number=exc.attempt_number,
                                    reason="server_restart_failed",
                                    error=str(exc),
                                    consume_attempt=True,
                                    shutdown_grace_sec=float(
                                        policy["shutdown_grace_sec"]
                                    ),
                                )
                                print(
                                    f"[eval_orchestrator] {worker.worker_id} server startup "
                                    f"attempt {exc.attempt_number} failed; "
                                    f"cleanup started",
                                    file=sys.stderr,
                                )
                                continue
                            if not server_ready:
                                continue
                        if shadow_startups_in_progress >= int(
                            policy["max_concurrent_shadow_startups"]
                        ):
                            continue
                        try:
                            _launch_worker_shadow(
                                worker,
                                args=args,
                                task=task,
                                checkpoint=checkpoint,
                                run_id=run_id,
                                attempt=attempt,
                                policy=policy,
                                yaml_config=static_plan["yaml"],
                                runtime_env=runtime_env,
                            )
                            shadow_startups_in_progress += 1
                        except EpisodeAttemptError as exc:
                            exhausted = _schedule_episode_failure(
                                worker,
                                attempt=attempt,
                                job=exc.job,
                                attempt_number=exc.attempt_number,
                                max_retries=int(policy["episode_max_retries"]),
                                backoff_sec=float(policy["retry_backoff_sec"]),
                                reason="shadow_launch_failed",
                                error=str(exc),
                                returncode=None,
                                now=time.monotonic(),
                            )
                            print(
                                f"[eval_orchestrator] {worker.worker_id} shadow launch "
                                f"attempt {exc.attempt_number} failed; "
                                f"exhausted={int(exhausted)}",
                                file=sys.stderr,
                            )

                if interrupted:
                    break
                if all(
                    not worker.pending
                    and worker.active_shadow is None
                    and worker.cleanup is None
                    for worker in workers
                ):
                    if _CANCEL_REQUESTED:
                        frozen = True
                        interrupted = True
                    break
                time.sleep(0.1)
        except Exception as exc:
            frozen = True
            if _CANCEL_REQUESTED:
                interrupted = True
                interrupted_failure = exc
            else:
                failure = exc
        finally:
            for reservation in reservations:
                reservation.close()
            cleanup_executor.shutdown(wait=True, cancel_futures=False)
            active_children = [
                worker.active_shadow
                for worker in workers
                if worker.active_shadow is not None
            ]
            server_children = list(
                {
                    id(child): child
                    for child in (
                        worker.server for worker in workers if worker.server is not None
                    )
                }.values()
            )
            cleanup_errors: list[str] = []
            cleanup_failure: Exception | None = None
            if interrupted_failure is not None:
                cleanup_failure = interrupted_failure
                cleanup_errors.append(
                    "interrupted execution failed: "
                    f"{type(interrupted_failure).__name__}: {interrupted_failure}"
                )
            try:
                _terminate_children(
                    [*active_children, *server_children],
                    float(policy["shutdown_grace_sec"]),
                )
            except Exception as exc:
                cleanup_failure = exc
                cleanup_errors.append(
                    f"child termination failed: {type(exc).__name__}: {exc}"
                )
            unsafe_shm: list[str] = []
            for worker in workers:
                if (
                    worker.active_status_path is not None
                    and worker.active_session_id is not None
                ):
                    try:
                        _remove_owned_shutdown_guard(
                            worker.active_status_path,
                            run_id=run_id,
                            worker_id=worker.worker_id,
                            session_id=worker.active_session_id,
                        )
                    except Exception as exc:
                        cleanup_errors.append(
                            f"shutdown guard cleanup failed for {worker.worker_id}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        if cleanup_failure is None:
                            cleanup_failure = exc
                shm = worker.active_shm
                if shm is not None:
                    try:
                        shm.observe()
                        if not shm.cleanup():
                            unsafe_shm.append(shm.name)
                    except Exception:
                        unsafe_shm.append(shm.name)
                _clear_worker_shadow_state(worker)
            if unsafe_shm:
                shm_failure = UnsafeCleanupError(
                    f"SHM cleanup identity failed for: {unsafe_shm}"
                )
                cleanup_errors.append(str(shm_failure))
                if cleanup_failure is None:
                    cleanup_failure = shm_failure
            if _CANCEL_REQUESTED:
                frozen = True
                interrupted = True
            if cleanup_errors:
                _write_manifest(
                    attempt,
                    contract_hash,
                    "unsafe_cleanup",
                    interrupted=interrupted,
                    cleanup_errors=cleanup_errors,
                )
                for cleanup_error in cleanup_errors:
                    print(
                        f"[eval_orchestrator] unsafe cleanup: {cleanup_error}",
                        file=sys.stderr,
                    )
                if failure is None and not interrupted:
                    failure = cleanup_failure or SimulationError(
                        "cleanup failed without a captured exception"
                    )
            elif failure is not None:
                _write_manifest(
                    attempt,
                    contract_hash,
                    "failed",
                    error=f"{type(failure).__name__}: {failure}",
                )
            elif interrupted:
                _write_manifest(attempt, contract_hash, "interrupted")

        if failure is not None:
            raise failure
        if interrupted:
            return EXIT_INTERRUPTED

        for job in initially_pending:
            if (
                _read_existing_episode(attempt, job)
                or validate_and_publish_completion(attempt, job)
            ) is None:
                raise SimulationError(
                    f"worker completed without a valid completion envelope: {job.result_identity}"
                )
        execution_failed = _finalize(
            attempt, jobs, contract_hash, source_digest, reused
        )
        return EXIT_SIM if execution_failed else 0


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

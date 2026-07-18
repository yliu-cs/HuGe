#!/usr/bin/env python3
"""Huge-owned alternate entrypoint for read-only HumanoidArena evaluation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time
import types
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


UPSTREAM_SIM_SHA256 = "2c39e4eaa9024bc19020903c5cb58249f8a9a1c5048cbf29873b50db2b7047ea"
COMPLETION_SCHEMA_VERSION = "huge_humanoidarena_completion_v1"
WORKER_STATUS_SCHEMA_VERSION = "huge_ha_worker_status_v1"
TRACE_RECEIPT_SCHEMA_VERSION = "huge_humanoidarena_model_io_receipt_v1"
SERVER_FATAL_EXIT = 40
SIM_FATAL_EXIT = 50
LEGAL_TASK_OUTCOMES = {"success", "fall", "timeout", "sim_stopped", "max_steps"}
REQUIRED_HELPER_PARAMETERS: dict[str, tuple[str, ...]] = {
    "_build_parser": (),
    "_load_simple_video_recorder": (),
    "_normalize_control_routing": ("args",),
    "_initialize_task_scene": ("env", "env_cfg", "args", "apply_optional_runtime_augments"),
    "_seed_runtime_rngs": ("seed",),
    "_configure_episode_seed_state": ("env_cfg", "episode_seed"),
    "_load_episode_specs": ("args_cli",),
    "_reset_environment_for_episode": ("env", "env_cfg", "episode_seed"),
    "_reset_lerobot_runtime_seed": ("action_provider", "episode_seed"),
    "_notify_action_provider_env_reset": ("action_provider",),
    "_run_episode_once": (
        "simulation_app",
        "env",
        "env_cfg",
        "action_provider",
        "controller",
        "args_cli",
        "spec",
        "SimpleVideoRecorder",
    ),
}
FORBIDDEN_UPSTREAM_HELPERS = (
    "main",
    "_configure_renderer_gpu",
    "_ensure_unique_multi_image_shm_name",
)
_SHM_RE = re.compile(r"^huge_ha_([a-z0-9]{8})_(w[0-9]{2})_([a-f0-9]{16,32})$")
_WORKER_RE = re.compile(r"^w[0-9]{2}$")
_SHADOW_SESSION_RE = re.compile(r"^[a-f0-9]{16,32}$")
_RESULT_IDENTITY_RE = re.compile(r"^[a-f0-9]{64}$")
_UUID_CLEAN_RE = re.compile(r"[^0-9a-f]")
_PROCESS_STARTED_AT = time.time()
_SINGLE_EPISODE_INTEGER_FIELDS = (
    "seed",
    "repeat_idx",
    "episode_seed",
    "episode_index",
)
_SINGLE_EPISODE_PATH_FIELDS = (
    "result_json",
    "success_video_dir",
    "failure_video_dir",
    "recording_save_dir",
)
_RESULT_IDENTITY_FIELDS = (
    "task_cli",
    "isaac_task_name",
    "test_mode",
    "seed",
    "repeat_idx",
    "episode_seed",
    "episode_index",
    "instruction",
    "env_config_yaml_sha256",
)


try:
    from isaaclab.app import AppLauncher as _IsaacAppLauncher
except ModuleNotFoundError:
    _IsaacAppLauncher = object  # type: ignore[assignment,misc]


class UpstreamContractError(RuntimeError):
    """Raised when a private upstream dependency no longer matches its baseline."""


@dataclass(frozen=True)
class UpstreamSimContract:
    path: Path
    allowed_root: Path
    sha256: str
    signatures: dict[str, str]


@dataclass(frozen=True)
class VulkanDevice:
    renderer_ordinal: int
    uuid: str
    device_name: str
    pci_bus_id: str = ""
    vulkan_ordinal: int | None = None


def _sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _function_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    if node.args.vararg is not None or node.args.kwarg is not None or node.args.kwonlyargs:
        raise UpstreamContractError(f"required helper {node.name} uses an unsupported variadic/keyword-only signature")
    return tuple(argument.arg for argument in (*node.args.posonlyargs, *node.args.args))


def verify_upstream_sim_contract(
    path: str | Path,
    expected_sha256: str,
    *,
    allowed_root: str | Path,
) -> UpstreamSimContract:
    root = Path(allowed_root).expanduser().resolve()
    source = Path(path).expanduser().resolve()
    if not root.is_dir() or not source.is_file() or not _is_relative_to(source, root):
        raise UpstreamContractError(f"upstream evaluator must be a regular file inside {root}: {source}")
    expected = str(expected_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise UpstreamContractError("expected upstream SHA-256 must contain 64 lowercase hex characters")
    actual = _sha256_file(source)
    if actual != expected:
        raise UpstreamContractError(f"upstream evaluator SHA-256 mismatch: expected {expected}, got {actual}")
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise UpstreamContractError(f"unable to parse upstream evaluator: {exc}") from exc
    nodes = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    signatures: dict[str, str] = {}
    for name, expected_parameters in REQUIRED_HELPER_PARAMETERS.items():
        node = nodes.get(name)
        if node is None:
            raise UpstreamContractError(f"required upstream helper is missing: {name}")
        actual_parameters = _function_parameters(node)
        if actual_parameters != expected_parameters:
            raise UpstreamContractError(
                f"upstream helper signature mismatch for {name}: {actual_parameters} != {expected_parameters}"
            )
        signatures[name] = f"({', '.join(actual_parameters)})"
    return UpstreamSimContract(path=source, allowed_root=root, sha256=actual, signatures=signatures)


def load_upstream_sim_module(contract: UpstreamSimContract) -> types.ModuleType:
    root_token = str(contract.allowed_root)
    if root_token not in sys.path:
        sys.path.insert(0, root_token)
    os.environ["PROJECT_ROOT"] = root_token
    module_name = f"huge_readonly_humanoidarena_sim_eval_{contract.sha256[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, contract.path)
    if spec is None or spec.loader is None:
        raise UpstreamContractError(f"unable to create import spec for {contract.path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, expected_parameters in REQUIRED_HELPER_PARAMETERS.items():
        helper = getattr(module, name, None)
        if not callable(helper):
            raise UpstreamContractError(f"required upstream helper is not callable after import: {name}")
        signature = inspect.signature(helper)
        if tuple(signature.parameters) != expected_parameters:
            raise UpstreamContractError(
                f"runtime signature mismatch for {name}: {tuple(signature.parameters)} != {expected_parameters}"
            )
    return module


def validate_owned_shm_name(name: str, *, run_id: str, worker_id: str) -> None:
    match = _SHM_RE.fullmatch(str(name))
    run_token = re.sub(r"[^a-z0-9]", "", str(run_id).lower())[:8]
    if match is None or len(run_token) != 8:
        raise ValueError("owned SHM name does not match the Huge evaluation format")
    if match.group(1) != run_token or match.group(2) != worker_id:
        raise ValueError("owned SHM name does not match the current run/worker identity")


def prepare_split_gpu_launcher_args(
    launcher_args: Mapping[str, Any],
    *,
    renderer_gpu_ordinal: int,
) -> dict[str, Any]:
    prepared = dict(launcher_args)
    renderer = int(renderer_gpu_ordinal)
    if renderer < 0:
        raise ValueError("renderer GPU ordinal must be non-negative")
    if str(prepared.get("device", "cuda:0")) != "cuda:0":
        raise ValueError("split-GPU launcher requires child physics device cuda:0")
    if bool(prepared.get("distributed", False)) or bool(prepared.get("multi_gpu", False)):
        raise ValueError("split-GPU launcher rejects distributed/multi_gpu mode")
    existing = shlex.split(str(prepared.get("kit_args", "") or ""))
    forbidden_prefixes = (
        "--/renderer/activeGpu=",
        "--/physics/cudaDevice=",
        "--/renderer/multiGpu/",
    )
    if any(token.startswith(forbidden_prefixes) for token in existing):
        raise ValueError("conflicting caller-supplied Kit GPU argument")
    kit_args = [
        *existing,
        f"--/renderer/activeGpu={renderer}",
        "--/physics/cudaDevice=0",
        "--/renderer/multiGpu/enabled=False",
    ]
    prepared.update(
        {
            "device": "cuda:0",
            "multi_gpu": False,
            "distributed": False,
            "physics_gpu": 0,
            "active_gpu": renderer,
            "kit_args": shlex.join(kit_args),
        }
    )
    return prepared


class SplitGpuAppLauncher(_IsaacAppLauncher):  # type: ignore[misc,valid-type]
    """AppLauncher that keeps CUDA physics ordinal 0 separate from Kit renderer ordinal."""

    def __init__(self, launcher_args: argparse.Namespace | dict[str, Any], *, renderer_gpu_ordinal: int) -> None:
        if _IsaacAppLauncher is object:
            raise RuntimeError("SplitGpuAppLauncher requires the Isaac Lab Python environment")
        raw = vars(launcher_args).copy() if isinstance(launcher_args, argparse.Namespace) else dict(launcher_args)
        self._huge_renderer_gpu_ordinal = int(renderer_gpu_ordinal)
        prepared = prepare_split_gpu_launcher_args(
            raw,
            renderer_gpu_ordinal=self._huge_renderer_gpu_ordinal,
        )
        super().__init__(prepared)

    def _resolve_device_settings(self, launcher_args: dict[str, Any]) -> None:
        super()._resolve_device_settings(launcher_args)  # type: ignore[misc]
        launcher_args["physics_gpu"] = 0
        launcher_args["active_gpu"] = self._huge_renderer_gpu_ordinal
        launcher_args["multi_gpu"] = False
        self.device_id = 0


def parse_vulkaninfo_summary(text: str) -> list[VulkanDevice]:
    devices: list[VulkanDevice] = []
    current: dict[str, Any] | None = None
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"GPU([0-9]+):", line)
        if match:
            if current is not None and current.get("uuid"):
                devices.append(VulkanDevice(**current))
            current = {
                "renderer_ordinal": int(match.group(1)),
                "uuid": "",
                "device_name": "",
                "pci_bus_id": "",
            }
            continue
        if current is None or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        normalized_key = key.lower()
        if normalized_key == "deviceuuid":
            current["uuid"] = value.lower()
        elif normalized_key == "devicename":
            current["device_name"] = value
        elif normalized_key in {"pcibusid", "pci_bus_id"}:
            current["pci_bus_id"] = value.lower()
    if current is not None and current.get("uuid"):
        devices.append(VulkanDevice(**current))
    if not devices:
        raise ValueError("vulkaninfo summary did not contain GPU devices with UUIDs")
    ordinals = [device.renderer_ordinal for device in devices]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("vulkaninfo summary contains duplicate renderer ordinals")
    return sorted(devices, key=lambda item: item.renderer_ordinal)


def _normalize_gpu_uuid(value: str) -> str:
    token = str(value).strip().lower()
    token = token.removeprefix("mig-gpu-").removeprefix("gpu-")
    return _UUID_CLEAN_RE.sub("", token)


def _normalize_pci_bus_id(value: str) -> str:
    token = str(value).strip().lower()
    match = re.search(r"(?:[0-9a-f]{4,8}:)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])", token)
    if match is None:
        return token
    domain_match = re.search(r"^([0-9a-f]{4,8}):", token)
    domain = (domain_match.group(1) if domain_match else "0000").zfill(8)
    return f"{domain}:{match.group(1)}:{match.group(2)}.{match.group(3)}"


def _nvidia_pci_by_uuid() -> dict[str, str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is required for UUID/PCI cross-check")
    completed = subprocess.run(
        [executable, "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    mapping: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line.strip() or "," not in line:
            continue
        uuid, pci = (part.strip() for part in line.split(",", 1))
        mapping[_normalize_gpu_uuid(uuid)] = _normalize_pci_bus_id(pci)
    if not mapping:
        raise RuntimeError("nvidia-smi returned no GPU UUID/PCI rows")
    return mapping


def select_kit_renderer_devices(
    devices: Sequence[VulkanDevice],
    pci_by_uuid: Mapping[str, str],
) -> list[VulkanDevice]:
    selected: list[VulkanDevice] = []
    for device in sorted(devices, key=lambda item: item.renderer_ordinal):
        key = _normalize_gpu_uuid(device.uuid)
        pci = pci_by_uuid.get(key)
        if not pci:
            continue
        selected.append(
            VulkanDevice(
                renderer_ordinal=len(selected),
                uuid=device.uuid,
                device_name=device.device_name,
                pci_bus_id=pci,
                vulkan_ordinal=device.renderer_ordinal,
            )
        )
    if not selected:
        raise RuntimeError("Vulkan inventory contains no NVIDIA device matching nvidia-smi")
    return selected


def inspect_vulkan_devices() -> list[VulkanDevice]:
    executable = shutil.which("vulkaninfo")
    if executable is None:
        raise RuntimeError("vulkaninfo is required for Kit renderer identity verification")
    completed = subprocess.run(
        [executable, "--summary"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    pci_by_uuid = _nvidia_pci_by_uuid()
    devices = parse_vulkaninfo_summary(completed.stdout)
    return select_kit_renderer_devices(devices, pci_by_uuid)


def inspect_renderer_binding(
    *,
    renderer_gpu_ordinal: int,
    expected_cuda_uuid: str,
    expected_pci_bus_id: str,
) -> dict[str, Any]:
    import carb.settings

    active_gpu = int(carb.settings.get_settings().get("/renderer/activeGpu"))
    expected_ordinal = int(renderer_gpu_ordinal)
    if active_gpu != expected_ordinal:
        raise RuntimeError(f"Kit activeGpu={active_gpu}, expected {expected_ordinal}")
    devices = inspect_vulkan_devices()
    matches = [device for device in devices if device.renderer_ordinal == expected_ordinal]
    if len(matches) != 1:
        raise RuntimeError(f"renderer ordinal {expected_ordinal} does not resolve to one Vulkan device")
    selected = matches[0]
    if _normalize_gpu_uuid(selected.uuid) != _normalize_gpu_uuid(expected_cuda_uuid):
        raise RuntimeError(
            f"renderer UUID {selected.uuid!r} does not match expected CUDA UUID {expected_cuda_uuid!r}"
        )
    if not selected.pci_bus_id or _normalize_pci_bus_id(selected.pci_bus_id) != _normalize_pci_bus_id(expected_pci_bus_id):
        raise RuntimeError(
            f"renderer PCI {selected.pci_bus_id!r} does not match expected PCI {expected_pci_bus_id!r}"
        )
    return {
        "evidence_mode": "kit_active_gpu_plus_vulkaninfo_uuid_plus_nvidia_pci",
        "active_gpu": active_gpu,
        "selected": asdict(selected),
        "devices": [asdict(device) for device in devices],
    }


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        data = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def classify_episode_result(
    payload: Mapping[str, Any],
    server_status: Mapping[str, Any],
    *,
    run_id: str,
    worker_id: str,
    episode_seed: int,
) -> str:
    if (
        server_status.get("run_id") != run_id
        or server_status.get("worker_id") != worker_id
        or server_status.get("last_seeded_episode") != int(episode_seed)
    ):
        return "integrity_failure"
    failure_reason = str(payload.get("failure_reason", ""))
    error = str(payload.get("error", "") or "")
    if not error and failure_reason in LEGAL_TASK_OUTCOMES and isinstance(payload.get("success"), bool):
        return "completed"
    if failure_reason != "sim_error" and error:
        return "integrity_failure"
    state = str(server_status.get("state", ""))
    error_lower = error.lower()
    server_evidence = (
        state in {"fatal", "infer_in_progress"}
        or bool(server_status.get("fatal_error_code"))
        or any(token in error_lower for token in ("http ", "timed out", "timeout", "connection refused", "/infer"))
    )
    if failure_reason == "sim_error" and server_evidence:
        return "server_or_infer_fatal"
    if failure_reason == "sim_error" and state == "ready":
        return "sim_runtime_fatal"
    return "integrity_failure"


def publish_completion_atomic(
    payload: Mapping[str, Any],
    *,
    raw_result_path: str | Path,
    completion_dir: str | Path,
    attempt_dir: str | Path,
    run_id: str,
    worker_id: str,
) -> Path:
    attempt = Path(attempt_dir).resolve()
    raw_path = Path(raw_result_path).resolve()
    completions = Path(completion_dir).resolve()
    if not _is_relative_to(raw_path, attempt) or not _is_relative_to(completions, attempt):
        raise ValueError("raw result and completion paths must stay inside the current attempt")
    raw = _read_json_object(raw_path, label="raw result")
    normalized_payload = dict(payload)
    if raw != normalized_payload:
        raise ValueError("raw result content does not match the upstream return payload")
    identity_keys = ("task", "seed", "repeat_idx", "episode_seed", "episode_index")
    if any(key not in normalized_payload for key in identity_keys):
        raise ValueError("episode payload is missing identity fields")
    filename = (
        f"seed-{int(normalized_payload['seed']):010d}-"
        f"repeat-{int(normalized_payload['repeat_idx']):03d}-"
        f"episode-{int(normalized_payload['episode_index']):05d}.json"
    )
    completion = completions / filename
    envelope = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "run_id": run_id,
        "worker_id": worker_id,
        "identity": {key: normalized_payload[key] for key in identity_keys},
        "payload_sha256": hashlib.sha256(_canonical_json_bytes(normalized_payload)).hexdigest(),
        "raw_result_path": str(raw_path),
        "raw_result_sha256": _sha256_file(raw_path),
        "payload": normalized_payload,
    }
    atomic_write_json(completion, envelope)
    return completion


def _validate_attempt_path(path: str | Path, attempt_dir: Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not _is_relative_to(resolved, attempt_dir):
        raise ValueError(f"{label} must resolve inside attempt directory: {resolved}")
    return resolved


def _write_worker_status(path: Path, *, run_id: str, worker_id: str, state: str, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "schema_version": WORKER_STATUS_SCHEMA_VERSION,
        "run_id": run_id,
        "worker_id": worker_id,
        "state": state,
        "updated_at": time.time(),
    }
    payload.update(extra)
    atomic_write_json(path, payload)


def _apply_memory_limit(percent: int | None) -> None:
    if percent is None:
        return
    value = int(percent)
    if value <= 0 or value > 100:
        raise ValueError("memory limit percent must be in 1..100")
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    limit = pages * page_size * value // 100
    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    if current_hard not in (-1, resource.RLIM_INFINITY):
        limit = min(limit, int(current_hard))
    resource.setrlimit(resource.RLIMIT_AS, (limit, current_hard))


def _validate_runtime_paths(args: argparse.Namespace, episode_specs: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    attempt = Path(args.attempt_dir).expanduser().resolve()
    if not attempt.is_dir():
        raise ValueError(f"attempt directory does not exist: {attempt}")
    humanoidarena = Path(args.humanoidarena_root).expanduser().resolve()
    if _is_relative_to(attempt, humanoidarena):
        raise ValueError("attempt directory cannot be inside HumanoidArena")
    worker_status = _validate_attempt_path(args.worker_status_json, attempt, label="worker status")
    server_status = _validate_attempt_path(args.server_status_json, attempt, label="server status")
    completion_dir = _validate_attempt_path(args.completion_dir, attempt, label="completion directory")
    _validate_attempt_path(args.episode_batch_json, attempt, label="episode batch")
    for spec in episode_specs:
        _validate_attempt_path(spec["result_json"], attempt, label="raw result")
        _validate_attempt_path(spec["success_video_dir"], attempt, label="success video directory")
        _validate_attempt_path(spec["failure_video_dir"], attempt, label="failure video directory")
        _validate_attempt_path(spec["recording_save_dir"], attempt, label="recording directory")
    return worker_status, server_status, completion_dir


def _strict_int_field(payload: Mapping[str, Any], key: str, *, label: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} {key} must be an integer")
    return value


def _validate_shadow_identity_args(args: argparse.Namespace) -> None:
    result_identity = str(getattr(args, "result_identity", ""))
    if _RESULT_IDENTITY_RE.fullmatch(result_identity) is None:
        raise ValueError("result identity must contain 64 lowercase hex characters")
    shadow_session_id = str(getattr(args, "shadow_session_id", ""))
    if _SHADOW_SESSION_RE.fullmatch(shadow_session_id) is None:
        raise ValueError("shadow session id must contain 16..32 lowercase hex characters")
    expected_episode_seed = getattr(args, "expected_episode_seed", None)
    if (
        isinstance(expected_episode_seed, bool)
        or not isinstance(expected_episode_seed, int)
        or expected_episode_seed < 0
        or expected_episode_seed > 0x7FFFFFFF
    ):
        raise ValueError("expected episode seed must be an integer in 0..2147483647")


def _validate_single_episode_contract(
    args: argparse.Namespace,
    episode_specs: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    _validate_shadow_identity_args(args)
    if len(episode_specs) != 1:
        raise ValueError(f"single-episode shadow requires exactly one normalized episode, got {len(episode_specs)}")
    normalized_spec = episode_specs[0]
    if not isinstance(normalized_spec, Mapping):
        raise ValueError("normalized episode spec must be an object")
    normalized = dict(normalized_spec)
    worker_status, server_status, completion_dir = _validate_runtime_paths(args, [normalized])

    attempt = Path(args.attempt_dir).expanduser().resolve()
    shard_path = _validate_attempt_path(args.episode_batch_json, attempt, label="episode batch")
    raw_shard = _read_json_object(shard_path, label="episode batch")
    if raw_shard.get("schema_version") != "huge_humanoidarena_shard_v1":
        raise ValueError("episode batch schema version is incompatible")
    if raw_shard.get("worker_id") != args.worker_id:
        raise ValueError("episode batch worker identity does not match the shadow")
    raw_episodes = raw_shard.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != 1:
        count = len(raw_episodes) if isinstance(raw_episodes, list) else 0
        raise ValueError(f"single-episode shadow requires exactly one raw episode, got {count}")
    raw_spec = raw_episodes[0]
    if not isinstance(raw_spec, dict):
        raise ValueError("raw episode spec must be an object")

    normalized_values: dict[str, int] = {}
    for key in _SINGLE_EPISODE_INTEGER_FIELDS:
        raw_value = _strict_int_field(raw_spec, key, label="raw episode")
        normalized_value = _strict_int_field(normalized, key, label="normalized episode")
        if raw_value != normalized_value:
            raise ValueError(f"raw/normalized episode {key} mismatch: {raw_value} != {normalized_value}")
        normalized_values[key] = normalized_value
    if normalized_values["episode_seed"] != int(args.expected_episode_seed):
        raise ValueError(
            "normalized episode seed does not match --expected-episode-seed: "
            f"{normalized_values['episode_seed']} != {args.expected_episode_seed}"
        )
    if normalized_values["seed"] != int(args.seed):
        raise ValueError(f"normalized group seed does not match --seed: {normalized_values['seed']} != {args.seed}")

    for key in _SINGLE_EPISODE_PATH_FIELDS:
        raw_path = _validate_attempt_path(raw_spec.get(key, ""), attempt, label=f"raw episode {key}")
        normalized_path = _validate_attempt_path(normalized.get(key, ""), attempt, label=f"normalized episode {key}")
        if raw_path != normalized_path:
            raise ValueError(f"raw/normalized episode {key} mismatch: {raw_path} != {normalized_path}")

    raw_eval_model_path = raw_spec.get("eval_model_path")
    normalized_eval_model_path = normalized.get("eval_model_path")
    if (
        not isinstance(raw_eval_model_path, str)
        or not raw_eval_model_path.strip()
        or not isinstance(normalized_eval_model_path, str)
        or not normalized_eval_model_path.strip()
    ):
        raise ValueError("raw and normalized eval_model_path must be non-empty strings")
    raw_eval_model = Path(raw_eval_model_path).expanduser().resolve()
    normalized_eval_model = Path(normalized_eval_model_path).expanduser().resolve()
    if raw_eval_model != normalized_eval_model:
        raise ValueError(
            f"raw/normalized episode eval_model_path mismatch: {raw_eval_model} != {normalized_eval_model}"
        )

    plan = _read_json_object(attempt / "plan.json", label="attempt plan")
    if plan.get("schema_version") != "huge_humanoidarena_contract_v1":
        raise ValueError("attempt plan schema version is incompatible")
    static_plan = plan.get("static_plan")
    if not isinstance(static_plan, dict):
        raise ValueError("attempt plan static_plan must be an object")
    plan_episodes = static_plan.get("episodes")
    if not isinstance(plan_episodes, list):
        raise ValueError("attempt plan episodes must be a list")

    stable_tuple = (
        str(args.task),
        normalized_values["seed"],
        normalized_values["repeat_idx"],
        normalized_values["episode_seed"],
        normalized_values["episode_index"],
    )
    matches: list[dict[str, Any]] = []
    for entry in plan_episodes:
        if not isinstance(entry, dict):
            raise ValueError("attempt plan episode must be an object")
        entry_tuple = (
            str(entry.get("isaac_task_name", "")),
            _strict_int_field(entry, "seed", label="attempt plan episode"),
            _strict_int_field(entry, "repeat_idx", label="attempt plan episode"),
            _strict_int_field(entry, "episode_seed", label="attempt plan episode"),
            _strict_int_field(entry, "episode_index", label="attempt plan episode"),
        )
        if entry_tuple == stable_tuple:
            matches.append(entry)
    if len(matches) != 1:
        raise ValueError(f"attempt plan must contain exactly one matching stable episode tuple, got {len(matches)}")

    plan_episode = matches[0]
    if any(key not in plan_episode for key in _RESULT_IDENTITY_FIELDS):
        raise ValueError("matching attempt plan episode is missing result identity fields")
    identity_payload = {key: plan_episode[key] for key in _RESULT_IDENTITY_FIELDS}
    plan_result_identity = str(plan_episode.get("result_identity", ""))
    recomputed_result_identity = hashlib.sha256(_canonical_json_bytes(identity_payload)).hexdigest()
    if plan_result_identity != recomputed_result_identity:
        raise ValueError("attempt plan episode result identity does not match its canonical stable payload")
    if plan_result_identity != args.result_identity:
        raise ValueError("required result identity does not match the attempt plan episode")

    episode_identity = {
        "result_identity": plan_result_identity,
        "task": str(args.task),
        "seed": normalized_values["seed"],
        "repeat_idx": normalized_values["repeat_idx"],
        "episode_seed": normalized_values["episode_seed"],
        "episode_index": normalized_values["episode_index"],
    }
    return worker_status, server_status, completion_dir, normalized, episode_identity


def _unavailable_model_io_result(
    *,
    result_identity: str,
    episode_seed: int,
    outcome_sha256: str,
    error: str,
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "trace_status": "incomplete",
        "result_identity": result_identity,
        "episode_seed": int(episode_seed),
        "outcome_sha256": outcome_sha256,
        "error": error,
    }


def finish_model_io_trace(
    server_url: str,
    *,
    result_identity: str,
    episode_seed: int,
    outcome_sha256: str,
    timeout_sec: float,
) -> dict[str, Any]:
    try:
        if _RESULT_IDENTITY_RE.fullmatch(str(result_identity)) is None:
            raise ValueError("result identity must contain 64 lowercase hex characters")
        if _RESULT_IDENTITY_RE.fullmatch(str(outcome_sha256)) is None:
            raise ValueError("outcome SHA-256 must contain 64 lowercase hex characters")
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise ValueError("episode seed must be an integer")
        timeout = float(timeout_sec)
        if timeout <= 0:
            raise ValueError("finish timeout must be positive")
        body = {
            "result_identity": result_identity,
            "episode_seed": episode_seed,
            "outcome_sha256": outcome_sha256,
        }
        request = urllib.request.Request(
            f"{str(server_url).rstrip('/')}/finish_episode",
            data=_canonical_json_bytes(body),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("finish response must be a JSON object")
        if payload.get("schema_version") != TRACE_RECEIPT_SCHEMA_VERSION:
            raise ValueError("finish response schema version is incompatible")
        if payload.get("result_identity") != result_identity:
            raise ValueError("finish response result identity mismatch")
        if payload.get("episode_seed") != episode_seed:
            raise ValueError("finish response episode seed mismatch")
        if payload.get("outcome_sha256") != outcome_sha256:
            raise ValueError("finish response outcome SHA-256 mismatch")
        if payload.get("trace_status") not in {"complete", "degraded", "incomplete"}:
            raise ValueError("finish response trace status is invalid")
        return payload
    except Exception as exc:
        return _unavailable_model_io_result(
            result_identity=str(result_identity),
            episode_seed=int(episode_seed) if isinstance(episode_seed, int) and not isinstance(episode_seed, bool) else -1,
            outcome_sha256=str(outcome_sha256),
            error=f"{type(exc).__name__}: {exc}",
        )


def _run_upstream_episode_with_seeded_reset_last(
    upstream: types.ModuleType,
    *,
    action_provider: Any,
    episode_seed: int,
    invoke: Callable[[], Any],
) -> Any:
    """Run one upstream episode after moving its seeded HTTP reset last."""
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
        raise TypeError("episode_seed must be an integer")
    original_seed_reset = getattr(upstream, "_reset_lerobot_runtime_seed", None)
    original_provider_reset = getattr(
        upstream, "_notify_action_provider_env_reset", None
    )
    if not callable(original_seed_reset) or not callable(original_provider_reset):
        raise UpstreamContractError(
            "upstream evaluator lacks the required LeRobot reset helpers"
        )

    seed_deferred = False
    seed_applied = False
    provider_reset_calls = 0

    def defer_seed_reset(provider: Any, seed: int) -> None:
        nonlocal seed_deferred
        if provider is not action_provider:
            raise RuntimeError("upstream seeded reset used an unexpected action provider")
        if seed_deferred:
            raise RuntimeError("upstream seeded reset was requested more than once")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed != episode_seed:
            raise RuntimeError(
                f"upstream seeded reset used {seed!r}, expected {episode_seed}"
            )
        seed_deferred = True

    def reset_provider_then_apply_seed(provider: Any) -> None:
        nonlocal provider_reset_calls, seed_applied
        if provider is not action_provider:
            raise RuntimeError("upstream provider reset used an unexpected action provider")
        provider_reset_calls += 1
        if provider_reset_calls != 1:
            raise RuntimeError("upstream provider reset was requested more than once")
        if not seed_deferred:
            raise RuntimeError("upstream provider reset ran before a seed was scheduled")
        original_provider_reset(provider)
        original_seed_reset(provider, episode_seed)
        seed_applied = True

    setattr(upstream, "_reset_lerobot_runtime_seed", defer_seed_reset)
    setattr(
        upstream,
        "_notify_action_provider_env_reset",
        reset_provider_then_apply_seed,
    )
    try:
        result = invoke()
    finally:
        setattr(upstream, "_reset_lerobot_runtime_seed", original_seed_reset)
        setattr(
            upstream,
            "_notify_action_provider_env_reset",
            original_provider_reset,
        )

    if not seed_deferred or provider_reset_calls != 1 or not seed_applied:
        raise RuntimeError(
            "upstream episode did not complete the required provider-reset/seed-reset sequence"
        )
    return result


def _run_probe(args: argparse.Namespace) -> int:
    launcher_args = {
        "headless": True,
        "enable_cameras": False,
        "device": "cuda:0",
        "multi_gpu": False,
        "distributed": False,
        "kit_args": "",
    }
    launcher = SplitGpuAppLauncher(
        launcher_args,
        renderer_gpu_ordinal=int(args.renderer_gpu_ordinal),
    )
    try:
        if args.gpu_foundation_bootstrap_only:
            import carb.settings

            report = {
                "evidence_mode": "kit_active_gpu_plus_vulkaninfo_uuid_plus_nvidia_pci",
                "active_gpu": int(carb.settings.get_settings().get("/renderer/activeGpu")),
                "devices": [asdict(device) for device in inspect_vulkan_devices()],
            }
        else:
            report = inspect_renderer_binding(
                renderer_gpu_ordinal=args.renderer_gpu_ordinal,
                expected_cuda_uuid=args.expected_cuda_uuid,
                expected_pci_bus_id=args.expected_pci_bus_id,
            )
        if args.gpu_report_json:
            report_path = Path(args.gpu_report_json).expanduser().resolve()
            humanoidarena = Path(args.humanoidarena_root).expanduser().resolve()
            if _is_relative_to(report_path, humanoidarena):
                raise ValueError("GPU report cannot be written inside HumanoidArena")
            atomic_write_json(report_path, report)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    finally:
        launcher.app.close()


def run_persistent_shard(args: argparse.Namespace, upstream: types.ModuleType) -> int:
    validate_owned_shm_name(args.owned_shm_name, run_id=args.run_id, worker_id=args.worker_id)
    if os.environ.get("ISAAC_MULTI_IMAGE_SHM_NAME", "") not in {"", args.owned_shm_name}:
        raise ValueError("ISAAC_MULTI_IMAGE_SHM_NAME conflicts with the owned SHM name")
    os.environ["ISAAC_MULTI_IMAGE_SHM_NAME"] = args.owned_shm_name
    os.environ["PROJECT_ROOT"] = str(Path(args.isaaclab_root).resolve())
    os.environ["SONIC_VLA_ACTION_FORMAT"] = "latent64"
    os.environ["SONIC_VLA_USE_HEADING_ALIGN"] = "1"
    os.environ["SONIC_OUTPUT_DELAY_STEPS"] = "0"
    _apply_memory_limit(args.memory_limit_percent)
    args.enable_cameras = True
    args.multi_gpu = False
    args.device = "cuda:0"
    upstream._normalize_control_routing(args)
    upstream.apply_task_runtime_profile(args)
    episode_specs = upstream._load_episode_specs(args)
    (
        worker_status_path,
        server_status_path,
        completion_dir,
        episode_spec,
        episode_identity,
    ) = _validate_single_episode_contract(args, episode_specs)
    model_io_result: dict[str, Any] = {
        "status": "pending",
        "trace_status": "incomplete",
        "result_identity": args.result_identity,
        "episode_seed": args.expected_episode_seed,
        "outcome_sha256": "",
        "error": None,
    }
    cleanup_errors: list[dict[str, str]] = []
    status_extra: dict[str, Any] = {"completed_count": 0}
    final_state = "sim_runtime_fatal"
    exit_code = SIM_FATAL_EXIT

    def write_status(state: str, **extra: Any) -> None:
        _write_worker_status(
            worker_status_path,
            run_id=args.run_id,
            worker_id=args.worker_id,
            state=state,
            shadow_session_id=args.shadow_session_id,
            process_started_at=_PROCESS_STARTED_AT,
            simulator_pid=os.getpid(),
            episode_identity=dict(episode_identity),
            model_io=dict(model_io_result),
            cleanup_errors=list(cleanup_errors),
            **extra,
        )

    shutdown_guard_path = worker_status_path.with_name(
        f".{worker_status_path.name}.shutdown_guard"
    )
    if os.path.lexists(shutdown_guard_path):
        raise ValueError(f"shadow shutdown guard already exists: {shutdown_guard_path}")
    with shutdown_guard_path.open("x", encoding="ascii") as stream:
        stream.write(f"{args.run_id} {args.worker_id} {args.shadow_session_id}\n")
        stream.flush()
        os.fsync(stream.fileno())

    write_status("starting", episode_count=1)

    launcher = None
    env = None
    controller = None
    action_provider = None
    try:
        launcher = SplitGpuAppLauncher(args, renderer_gpu_ordinal=args.renderer_gpu_ordinal)
        simulation_app = launcher.app
        binding_report = inspect_renderer_binding(
            renderer_gpu_ordinal=args.renderer_gpu_ordinal,
            expected_cuda_uuid=args.expected_cuda_uuid,
            expected_pci_bus_id=args.expected_pci_bus_id,
        )
        write_status("binding_verified", binding=binding_report)

        import gymnasium as gym

        import tasks  # noqa: F401
        from action_provider.create_action_provider import create_action_provider
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from layeredcontrol.robot_control_system import ControlConfig, RobotController
        from tasks.common_env_config import apply_env_config_yaml
        from tasks.common_runtime import apply_optional_runtime_augments, setup_vision_test_light

        SimpleVideoRecorder = upstream._load_simple_video_recorder()
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
        env_cfg.env_name = args.task
        apply_env_config_yaml(
            env_cfg,
            args.env_config_yaml,
            task_name=args.task,
            route_name=args.gmt_backend or args.action_source,
        )
        upstream._seed_runtime_rngs(int(episode_spec["episode_seed"]))
        upstream._configure_episode_seed_state(env_cfg, int(episode_spec["episode_seed"]))
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        upstream._initialize_task_scene(env, env_cfg, args, apply_optional_runtime_augments)
        if os.getenv("VISION_RANDOMIZATION"):
            setup_vision_test_light()
        upstream._reset_environment_for_episode(env, env_cfg, int(episode_spec["episode_seed"]))
        physics_dt = getattr(env, "physics_dt", None) or env_cfg.sim.dt
        control_hz = int(round(1.0 / (physics_dt * env_cfg.decimation)))
        control_config = ControlConfig(step_hz=control_hz, replay_mode=False, use_rl_action_mode=True)
        action_provider = create_action_provider(env, args)
        if action_provider is None:
            raise RuntimeError("failed to create HumanoidArena action provider")
        disable_recording = getattr(upstream, "_disable_action_provider_internal_recording", None)
        if callable(disable_recording):
            disable_recording(action_provider)
        env.action_provider = action_provider
        controller = RobotController(env, control_config)
        controller.set_action_provider(action_provider)

        payload = _run_upstream_episode_with_seeded_reset_last(
            upstream,
            action_provider=action_provider,
            episode_seed=int(episode_spec["episode_seed"]),
            invoke=lambda: upstream._run_episode_once(
                simulation_app,
                env,
                env_cfg,
                action_provider,
                controller,
                args,
                episode_spec,
                SimpleVideoRecorder,
            ),
        )
        if not isinstance(payload, dict):
            raise ValueError("upstream episode helper returned a non-object payload")
        outcome_sha256 = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        try:
            model_io_result = finish_model_io_trace(
                args.lerobot_server_url,
                result_identity=args.result_identity,
                episode_seed=args.expected_episode_seed,
                outcome_sha256=outcome_sha256,
                timeout_sec=args.lerobot_server_timeout,
            )
            if not isinstance(model_io_result, dict):
                raise ValueError("finish helper returned a non-object result")
        except Exception as exc:
            model_io_result = _unavailable_model_io_result(
                result_identity=args.result_identity,
                episode_seed=args.expected_episode_seed,
                outcome_sha256=outcome_sha256,
                error=f"{type(exc).__name__}: {exc}",
            )
        print(
            "[sim_eval_shadow] model_io="
            + json.dumps(model_io_result, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        )

        server_status = _read_json_object(server_status_path, label="server status")
        classification = classify_episode_result(
            payload,
            server_status,
            run_id=args.run_id,
            worker_id=args.worker_id,
            episode_seed=int(episode_spec["episode_seed"]),
        )
        status_extra = {
            "current_episode": dict(episode_identity),
            "completed_count": int(classification == "completed"),
            "classification": classification,
            "upstream_payload_sha256": outcome_sha256,
            "server_status_sha256": hashlib.sha256(_canonical_json_bytes(server_status)).hexdigest(),
        }
        if classification == "completed":
            completion = publish_completion_atomic(
                payload,
                raw_result_path=episode_spec["result_json"],
                completion_dir=completion_dir,
                attempt_dir=args.attempt_dir,
                run_id=args.run_id,
                worker_id=args.worker_id,
            )
            status_extra["completion"] = str(completion)
            final_state = "complete"
            exit_code = 0
        else:
            final_state = classification
            exit_code = SERVER_FATAL_EXIT if classification == "server_or_infer_fatal" else SIM_FATAL_EXIT
    except KeyboardInterrupt:
        final_state = "interrupted"
        exit_code = 130
        status_extra["error"] = "KeyboardInterrupt"
    except Exception as exc:
        final_state = "sim_runtime_fatal"
        exit_code = SIM_FATAL_EXIT
        status_extra["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            if controller is not None:
                controller.cleanup()
        except Exception as exc:
            cleanup_errors.append(
                {"resource": "controller", "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"[sim_eval_shadow] controller cleanup failed: {exc}", file=sys.stderr)
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            cleanup_errors.append(
                {"resource": "env", "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"[sim_eval_shadow] env cleanup failed: {exc}", file=sys.stderr)

        if cleanup_errors:
            status_extra["state_before_cleanup"] = final_state
            status_extra.setdefault("error", "one or more simulator cleanup operations failed")
            final_state = "sim_runtime_fatal"
            if exit_code == 0:
                exit_code = SIM_FATAL_EXIT

        if launcher is not None:
            try:
                write_status(
                    "shutdown_ready",
                    planned_state=final_state,
                    planned_exit_code=exit_code,
                    **status_extra,
                )
                shutdown_guard_path.unlink()
            except Exception as exc:
                print(f"[sim_eval_shadow] shutdown-ready status write failed: {exc}", file=sys.stderr)
                status_extra.setdefault("state_before_cleanup", final_state)
                status_extra.setdefault(
                    "error",
                    f"shutdown-ready status write failed: {type(exc).__name__}: {exc}",
                )
                final_state = "sim_runtime_fatal"
                exit_code = SIM_FATAL_EXIT
            try:
                launcher.app.close()
            except Exception as exc:
                cleanup_errors.append(
                    {"resource": "app", "error": f"{type(exc).__name__}: {exc}"}
                )
                print(f"[sim_eval_shadow] SimulationApp cleanup failed: {exc}", file=sys.stderr)

        if cleanup_errors:
            status_extra.setdefault("state_before_cleanup", final_state)
            status_extra.setdefault("error", "one or more simulator cleanup operations failed")
            final_state = "sim_runtime_fatal"
            if exit_code == 0:
                exit_code = SIM_FATAL_EXIT
        status_extra["exit_code"] = exit_code
        try:
            write_status(final_state, **status_extra)
        except Exception as exc:
            print(f"[sim_eval_shadow] final status write failed: {exc}", file=sys.stderr)
            if exit_code == 0:
                exit_code = SIM_FATAL_EXIT
    return exit_code


def _add_common_adapter_args(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--humanoidarena-root", required=required)
    parser.add_argument("--isaaclab-root", required=required)
    parser.add_argument("--upstream-sim-eval", required=required)
    parser.add_argument("--upstream-sim-sha256", default=UPSTREAM_SIM_SHA256)
    parser.add_argument("--renderer-gpu-ordinal", type=int, default=0)
    parser.add_argument("--expected-cuda-uuid", default="")
    parser.add_argument("--expected-pci-bus-id", default="")
    parser.add_argument("--gpu-report-json", default="")


def _build_pre_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inspect-contract", action="store_true")
    modes.add_argument("--gpu-foundation-bootstrap-only", action="store_true")
    modes.add_argument("--gpu-binding-probe-only", action="store_true")
    _add_common_adapter_args(parser, required=False)
    return parser


def build_parser(upstream: types.ModuleType) -> argparse.ArgumentParser:
    parser = upstream._build_parser()
    _add_common_adapter_args(parser, required=True)
    parser.add_argument("--owned-shm-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--server-status-json", required=True)
    parser.add_argument("--worker-status-json", required=True)
    parser.add_argument("--result-identity", required=True)
    parser.add_argument("--expected-episode-seed", required=True, type=int)
    parser.add_argument("--shadow-session-id", required=True)
    parser.add_argument("--completion-dir", required=True)
    parser.add_argument("--memory-limit-percent", type=int, default=None)
    return parser


def _resolve_roots(pre_args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if not pre_args.humanoidarena_root or not pre_args.isaaclab_root or not pre_args.upstream_sim_eval:
        raise ValueError("HumanoidArena root, IsaacLab root and upstream evaluator are required")
    humanoidarena = Path(pre_args.humanoidarena_root).expanduser().resolve()
    isaaclab = Path(pre_args.isaaclab_root).expanduser().resolve()
    upstream = Path(pre_args.upstream_sim_eval).expanduser().resolve()
    if not humanoidarena.is_dir() or isaaclab != (humanoidarena / "isaaclab_twist2_g1").resolve():
        raise ValueError("IsaacLab root must be HumanoidArena/isaaclab_twist2_g1")
    if not _is_relative_to(upstream, isaaclab):
        raise ValueError("upstream evaluator must resolve inside the IsaacLab root")
    return humanoidarena, isaaclab, upstream


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    pre_parser = _build_pre_parser()
    pre_args, unknown = pre_parser.parse_known_args(arguments)
    _humanoidarena, isaaclab, upstream_path = _resolve_roots(pre_args)
    contract = verify_upstream_sim_contract(
        upstream_path,
        pre_args.upstream_sim_sha256,
        allowed_root=isaaclab,
    )
    upstream = load_upstream_sim_module(contract)
    if pre_args.inspect_contract:
        if unknown:
            raise ValueError(f"unknown inspect-contract arguments: {unknown}")
        report = {
            "status": "compatible",
            "path": str(contract.path),
            "sha256": contract.sha256,
            "required_signatures": contract.signatures,
            "forbidden_helpers": list(FORBIDDEN_UPSTREAM_HELPERS),
        }
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    if pre_args.gpu_foundation_bootstrap_only or pre_args.gpu_binding_probe_only:
        if unknown:
            raise ValueError(f"unknown GPU probe arguments: {unknown}")
        if pre_args.gpu_binding_probe_only and (
            not pre_args.expected_cuda_uuid or not pre_args.expected_pci_bus_id
        ):
            raise ValueError("binding probe requires expected CUDA UUID and PCI bus id")
        return _run_probe(pre_args)
    parser = build_parser(upstream)
    args = parser.parse_args(arguments)
    if not _WORKER_RE.fullmatch(args.worker_id):
        parser.error("worker-id must match wNN")
    try:
        _validate_shadow_identity_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.expected_cuda_uuid or not args.expected_pci_bus_id:
        parser.error("expected CUDA UUID and PCI bus id are required")
    return run_persistent_shard(args, upstream)


if __name__ == "__main__":
    raise SystemExit(main())

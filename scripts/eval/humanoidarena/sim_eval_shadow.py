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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


UPSTREAM_SIM_SHA256 = "2c39e4eaa9024bc19020903c5cb58249f8a9a1c5048cbf29873b50db2b7047ea"
COMPLETION_SCHEMA_VERSION = "huge_humanoidarena_completion_v1"
WORKER_STATUS_SCHEMA_VERSION = "huge_ha_worker_status_v1"
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
_UUID_CLEAN_RE = re.compile(r"[^0-9a-f]")


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
    if not episode_specs:
        raise ValueError("episode batch contains no episodes")
    worker_status_path, server_status_path, completion_dir = _validate_runtime_paths(args, episode_specs)
    _write_worker_status(
        worker_status_path,
        run_id=args.run_id,
        worker_id=args.worker_id,
        state="starting",
        episode_count=len(episode_specs),
    )

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
        _write_worker_status(
            worker_status_path,
            run_id=args.run_id,
            worker_id=args.worker_id,
            state="binding_verified",
            binding=binding_report,
        )

        import gymnasium as gym

        import tasks  # noqa: F401
        from action_provider.create_action_provider import create_action_provider
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        from layeredcontrol.robot_control_system import ControlConfig, RobotController
        from tasks.common_env_config import apply_env_config_yaml
        from tasks.common_runtime import apply_optional_runtime_augments, setup_vision_test_light

        SimpleVideoRecorder = upstream._load_simple_video_recorder()
        first_spec = episode_specs[0]
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
        env_cfg.env_name = args.task
        apply_env_config_yaml(
            env_cfg,
            args.env_config_yaml,
            task_name=args.task,
            route_name=args.gmt_backend or args.action_source,
        )
        upstream._seed_runtime_rngs(int(first_spec["episode_seed"]))
        upstream._configure_episode_seed_state(env_cfg, int(first_spec["episode_seed"]))
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        upstream._initialize_task_scene(env, env_cfg, args, apply_optional_runtime_augments)
        if os.getenv("VISION_RANDOMIZATION"):
            setup_vision_test_light()
        upstream._reset_environment_for_episode(env, env_cfg, int(first_spec["episode_seed"]))
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

        for completed_count, spec in enumerate(episode_specs, start=1):
            payload = upstream._run_episode_once(
                simulation_app,
                env,
                env_cfg,
                action_provider,
                controller,
                args,
                spec,
                SimpleVideoRecorder,
            )
            if not isinstance(payload, dict):
                raise ValueError("upstream episode helper returned a non-object payload")
            server_status = _read_json_object(server_status_path, label="server status")
            classification = classify_episode_result(
                payload,
                server_status,
                run_id=args.run_id,
                worker_id=args.worker_id,
                episode_seed=int(spec["episode_seed"]),
            )
            status_extra = {
                "current_episode": {
                    "seed": int(spec["seed"]),
                    "repeat_idx": int(spec["repeat_idx"]),
                    "episode_seed": int(spec["episode_seed"]),
                    "episode_index": int(spec["episode_index"]),
                },
                "completed_count": completed_count - (classification != "completed"),
                "classification": classification,
                "upstream_payload_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
                "server_status_sha256": hashlib.sha256(_canonical_json_bytes(server_status)).hexdigest(),
            }
            if classification == "completed":
                completion = publish_completion_atomic(
                    payload,
                    raw_result_path=spec["result_json"],
                    completion_dir=completion_dir,
                    attempt_dir=args.attempt_dir,
                    run_id=args.run_id,
                    worker_id=args.worker_id,
                )
                _write_worker_status(
                    worker_status_path,
                    run_id=args.run_id,
                    worker_id=args.worker_id,
                    state="running",
                    completion=str(completion),
                    **status_extra,
                )
                continue
            _write_worker_status(
                worker_status_path,
                run_id=args.run_id,
                worker_id=args.worker_id,
                state=classification,
                **status_extra,
            )
            return SERVER_FATAL_EXIT if classification == "server_or_infer_fatal" else SIM_FATAL_EXIT
        _write_worker_status(
            worker_status_path,
            run_id=args.run_id,
            worker_id=args.worker_id,
            state="complete",
            completed_count=len(episode_specs),
        )
        return 0
    except KeyboardInterrupt:
        _write_worker_status(
            worker_status_path,
            run_id=args.run_id,
            worker_id=args.worker_id,
            state="interrupted",
        )
        return 130
    except Exception as exc:
        _write_worker_status(
            worker_status_path,
            run_id=args.run_id,
            worker_id=args.worker_id,
            state="sim_runtime_fatal",
            error=f"{type(exc).__name__}: {exc}",
        )
        return SIM_FATAL_EXIT
    finally:
        try:
            if controller is not None:
                controller.cleanup()
        except Exception as exc:
            print(f"[sim_eval_shadow] controller cleanup failed: {exc}", file=sys.stderr)
        try:
            if env is not None:
                env.close()
        except Exception as exc:
            print(f"[sim_eval_shadow] env cleanup failed: {exc}", file=sys.stderr)
        try:
            if launcher is not None:
                launcher.app.close()
        except Exception as exc:
            print(f"[sim_eval_shadow] SimulationApp cleanup failed: {exc}", file=sys.stderr)


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
    if not args.expected_cuda_uuid or not args.expected_pci_bus_id:
        parser.error("expected CUDA UUID and PCI bus id are required")
    return run_persistent_shard(args, upstream)


if __name__ == "__main__":
    raise SystemExit(main())

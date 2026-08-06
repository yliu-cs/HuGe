"""Replay SONIC token files in MuJoCo and export canonical 34-D pseudo proprio."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
import queue
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np

from align_sim_replay import ALIGN_FIELDS, align_frames_to_tokens, build_aligned_arrays, capture_g1_debug_frames
from export_pseudo_proprio import apply_hand_state, build_pseudo_proprio, self_check
from huge.data.schemas import ACTION_DIM, MOTION_TOKEN_DIM, validate_action_array

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
WBC_ROOT = REPO_ROOT / "GR00T-WholeBodyControl"
SIM_PYTHON = WBC_ROOT / ".venv_sim" / "bin" / "python"
SIM_SCRIPT = SCRIPT_DIR / "launch_headless_sim.py"
DEPLOY_ROOT = WBC_ROOT / "gear_sonic_deploy"
DEPLOY_BINARY = DEPLOY_ROOT / "target" / "release" / "g1_deploy_onnx_ref"
PLAYER_SCRIPT = WBC_ROOT / "gear_sonic" / "scripts" / "sonic_encoder_input_player.py"
SONIC_ASSETS = REPO_ROOT / "resource" / "model" / "sonic"
DECODER_MODEL = SONIC_ASSETS / "model_decoder.onnx"
ENCODER_MODEL = SONIC_ASSETS / "model_encoder.onnx"
OBS_CONFIG = SONIC_ASSETS / "observation_config.yaml"
REFERENCE_ROOT = DEPLOY_ROOT / "reference" / "example"
PLANNER_MODEL = DEPLOY_ROOT / "planner" / "target_vel" / "V2" / "planner_sonic.onnx"
G1_MESH_ROOT = WBC_ROOT / "decoupled_wbc" / "control" / "robot_model" / "model_data" / "g1" / "meshes"
RUN_ROOT = REPO_ROOT / ".runs" / "calc_proprio"
TRT_LOCK_PATH = RUN_ROOT / "tensorrt-init.lock"
OUTPUT_LOCK_ROOT = RUN_ROOT / "locks"

CONTROL_HZ = 50.0
DEFAULT_ZMQ_PORT_BASE = 5556
DEFAULT_WORKERS_PER_GPU = 10
MAX_DDS_DOMAIN = 232
DEPLOY_INIT_PATTERN = re.compile(r"Init Done")
TELEOP_MODE_PATTERN = re.compile(
    r"(?:\[Mode Filter\] Switched to mode .*\(ID=1\)|VR 3-point control enabled)"
)
SIM_STARTUP_MARGIN = 3.0
PROCESS_STOP_TIMEOUT = 10.0
CAPTURE_RECV_TIMEOUT_MS = 250
MANIFEST_INTERVAL = 30.0
NO_GPU_VALUES = {"", "-1", "nodevfiles", "none", "void"}
GPU_SELECTOR = re.compile(r"(?:\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9_./-]+)")
SUCCESS_LOGS = ("sim.log", "deploy.log", "player.log", "motion_tokens.npy")


@dataclass(frozen=True)
class Job:
    token: Path
    output: Path
    episode_id: str


@dataclass(frozen=True)
class PlannerTrajectory:
    """Dataset-neutral per-control-tick planner and VR inputs."""

    mode: np.ndarray
    movement: np.ndarray
    facing: np.ndarray
    speed: np.ndarray
    height: np.ndarray
    vr_position: np.ndarray
    vr_orientation: np.ndarray
    hand_state: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.mode.shape[0])


@dataclass(frozen=True)
class PlannerJob:
    trajectory_source: Path | None
    token_output: Path
    proprio_output: Path
    episode_id: str


@dataclass(frozen=True)
class WorkerSpec:
    index: int
    gpu_selector: str
    gpu_index: int
    gpu_replica: int
    dds_domain: int
    input_port: int
    output_port: int


class RunCancelled(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def report_path(output: Path) -> Path:
    return output.with_suffix(".report.json")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        pickle.dump(value, stream)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def planner_report_path(job: PlannerJob) -> Path:
    return report_path(job.proprio_output)


def validate_planner_trajectory(trajectory: PlannerTrajectory) -> dict[str, Any]:
    arrays = {
        "mode": (trajectory.mode, (trajectory.rows,)),
        "movement": (trajectory.movement, (trajectory.rows, 3)),
        "facing": (trajectory.facing, (trajectory.rows, 3)),
        "speed": (trajectory.speed, (trajectory.rows,)),
        "height": (trajectory.height, (trajectory.rows,)),
        "vr_position": (trajectory.vr_position, (trajectory.rows, 9)),
        "vr_orientation": (trajectory.vr_orientation, (trajectory.rows, 12)),
        "hand_state": (trajectory.hand_state, (trajectory.rows, 2)),
    }
    if trajectory.rows <= 0:
        raise ValueError("planner trajectory must contain at least one row")
    for name, (value, shape) in arrays.items():
        if value.shape != shape:
            raise ValueError(f"planner {name} shape {value.shape} != {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"planner {name} contains NaN or Inf")
    rounded_mode = np.rint(trajectory.mode)
    if not np.array_equal(trajectory.mode, rounded_mode) or np.any((rounded_mode < 0) | (rounded_mode > 26)):
        raise ValueError("planner mode must contain integer locomotion modes in [0,26]")
    movement_norm = np.linalg.norm(trajectory.movement, axis=1)
    moving = movement_norm > 1e-6
    if np.any(np.abs(movement_norm[moving] - 1.0) > 1e-3):
        raise ValueError("non-zero planner movement rows must be unit vectors")
    facing_norm = np.linalg.norm(trajectory.facing, axis=1)
    if np.any(np.abs(facing_norm - 1.0) > 1e-3):
        raise ValueError("planner facing rows must be unit vectors")
    if np.any((trajectory.speed < 0.0) & (trajectory.speed != -1.0)):
        raise ValueError("planner speed must be -1 or non-negative")
    if np.any((trajectory.height < 0.0) & (trajectory.height != -1.0)):
        raise ValueError("planner height must be -1 or non-negative")
    orientations = trajectory.vr_orientation.reshape(trajectory.rows, 3, 4)
    quaternion_error = float(np.max(np.abs(np.linalg.norm(orientations, axis=2) - 1.0)))
    if quaternion_error > 1e-3:
        raise ValueError(f"planner VR quaternion norm error {quaternion_error} exceeds 1e-3")
    if not np.isin(trajectory.hand_state, (0.0, 1.0)).all():
        raise ValueError("planner hand_state must contain only 0 or 1")
    return {
        "rows": trajectory.rows,
        "movement_rows": int(moving.sum()),
        "turning_rows": int(
            (
                np.abs(
                    np.diff(
                        np.unwrap(
                            np.arctan2(trajectory.facing[:, 1], trajectory.facing[:, 0])
                        ),
                        prepend=0.0,
                    )
                )
                > 1e-6
            ).sum()
        ),
        "height_min": float(trajectory.height.min()),
        "height_max": float(trajectory.height.max()),
        "vr_quaternion_norm_max_err": quaternion_error,
        "left_closed_fraction": float(trajectory.hand_state[:, 0].mean()),
        "right_closed_fraction": float(trajectory.hand_state[:, 1].mean()),
    }


def load_planner_trajectory(path: Path) -> PlannerTrajectory:
    if not path.is_file():
        raise FileNotFoundError(f"planner trajectory does not exist: {path}")
    required = (
        "mode",
        "movement",
        "facing",
        "speed",
        "height",
        "vr_position",
        "vr_orientation",
        "hand_state",
    )
    with np.load(path) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"planner trajectory is missing arrays: {missing}")
        trajectory = PlannerTrajectory(
            mode=np.asarray(archive["mode"], dtype=np.float32),
            movement=np.asarray(archive["movement"], dtype=np.float32),
            facing=np.asarray(archive["facing"], dtype=np.float32),
            speed=np.asarray(archive["speed"], dtype=np.float32),
            height=np.asarray(archive["height"], dtype=np.float32),
            vr_position=np.asarray(archive["vr_position"], dtype=np.float32),
            vr_orientation=np.asarray(archive["vr_orientation"], dtype=np.float32),
            hand_state=np.asarray(archive["hand_state"], dtype=np.float32),
        )
    validate_planner_trajectory(trajectory)
    return trajectory


def validate_token(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"token file does not exist: {path}")
    tokens = np.load(path, mmap_mode="r")
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError(f"expected non-empty [T,D] token array at {path}, got {tokens.shape}")
    validate_action_array(tokens, allow_64d=True)
    if tokens.shape[1] == ACTION_DIM and not np.isin(tokens[:, MOTION_TOKEN_DIM:ACTION_DIM], (0.0, 1.0)).all():
        raise ValueError(f"66D hand action at {path} must contain only 0 or 1")
    return int(tokens.shape[0]), int(tokens.shape[1])


def hand_source(width: int) -> str:
    return "action_proxy" if width == ACTION_DIM else "simulated_default"


def runtime_assets() -> dict[str, Any]:
    return {
        "sim_python": str(SIM_PYTHON),
        "sim_script": str(SIM_SCRIPT),
        "deploy_binary": str(DEPLOY_BINARY),
        "player_script": str(PLAYER_SCRIPT),
        "decoder_model": str(DECODER_MODEL),
        "encoder_model": str(ENCODER_MODEL),
        "observation_config": str(OBS_CONFIG),
        "reference_root": str(REFERENCE_ROOT),
        "planner_model": str(PLANNER_MODEL),
        "g1_mesh_root": str(G1_MESH_ROOT),
    }


def validate_runtime_assets() -> None:
    files = (
        SIM_PYTHON,
        SIM_SCRIPT,
        DEPLOY_BINARY,
        PLAYER_SCRIPT,
        DECODER_MODEL,
        ENCODER_MODEL,
        OBS_CONFIG,
        PLANNER_MODEL,
    )
    missing = [str(path) for path in files if not path.is_file()]
    if not REFERENCE_ROOT.is_dir():
        missing.append(str(REFERENCE_ROOT))
    if not G1_MESH_ROOT.is_dir():
        missing.append(str(G1_MESH_ROOT))
    if missing:
        raise FileNotFoundError(f"required replay assets are missing: {missing}")
    lfs_pointers = [
        path
        for path in G1_MESH_ROOT.glob("*")
        if path.is_file()
        and path.stat().st_size < 1024
        and path.read_bytes().startswith(b"version https://git-lfs.github.com/spec")
    ]
    if lfs_pointers:
        raise RuntimeError(
            f"WBC G1 meshes are unresolved Git LFS pointers ({len(lfs_pointers)} files; first: {lfs_pointers[0]})"
        )
    help_result = subprocess.run(
        [str(DEPLOY_BINARY), "--help"],
        cwd=DEPLOY_ROOT,
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    help_text = help_result.stdout + help_result.stderr
    if help_result.returncode or "--domain-id" not in help_text:
        raise RuntimeError(
            f"deploy binary is stale or unusable; rebuild {DEPLOY_BINARY} with --domain-id support"
        )


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            stream.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def check_ports_free(spec: WorkerSpec) -> None:
    busy = [port for port in (spec.input_port, spec.output_port) if not port_free(port)]
    if busy:
        raise RuntimeError(f"worker {spec.index} replay port(s) already in use: {busy}")


def command_text(command: list[str]) -> str:
    return shlex.join(str(part) for part in command)


def start_logged(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    print(f"+ {command_text(command)} > {log_path}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as stream:
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def stop_process(process: subprocess.Popen[bytes], name: str) -> None:
    if process.poll() is not None:
        return
    print(f"terminating {name} process group {process.pid}", flush=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=PROCESS_STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=PROCESS_STOP_TIMEOUT)


def check_cancelled(stop_event: threading.Event) -> None:
    if stop_event.is_set():
        raise RunCancelled("run cancelled")


def wait_interval(duration: float, stop_event: threading.Event) -> None:
    if duration > 0 and stop_event.wait(duration):
        raise RunCancelled("run cancelled")


def wait_process(
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
    stop_event: threading.Event,
    name: str,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        check_cancelled(stop_event)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out after {timeout}s waiting for {name}; see {log_path}")
        stop_event.wait(min(0.25, remaining))
    if process.returncode:
        raise RuntimeError(f"{name} exited with {process.returncode}; see {log_path}")


def wait_for_log_pattern(
    process: subprocess.Popen[bytes],
    log_path: Path,
    pattern: re.Pattern[str],
    timeout: float,
    stop_event: threading.Event,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_cancelled(stop_event)
        if process.poll() is not None:
            raise RuntimeError(f"process exited with {process.returncode}; see {log_path}")
        if log_path.exists() and pattern.search(log_path.read_text(errors="ignore")):
            return
        stop_event.wait(0.5)
    raise TimeoutError(f"timed out after {timeout}s waiting for {pattern.pattern!r}; see {log_path}")


def inference_backend(args: argparse.Namespace | None = None) -> str:
    value = getattr(args, "inference_backend", "tensorrt") if args is not None else "tensorrt"
    normalized = str(value).lower().replace("-", "").replace("_", "")
    if normalized in {"tensorrt", "trt", "cuda"}:
        return "tensorrt"
    if normalized in {"onnxruntime", "ort", "cpu"}:
        return "onnxruntime"
    raise ValueError(f"unsupported inference backend: {value!r}")


def deploy_command(spec: WorkerSpec, args: argparse.Namespace | None = None) -> list[str]:
    return [
        str(DEPLOY_BINARY),
        "lo",
        str(DECODER_MODEL),
        str(REFERENCE_ROOT),
        "--domain-id",
        str(spec.dds_domain),
        "--input-type",
        "zmq_manager",
        "--output-type",
        "all",
        "--hand-type",
        "dex3",
        "--disable-crc-check",
        "--obs-config",
        str(OBS_CONFIG),
        "--encoder-file",
        str(ENCODER_MODEL),
        "--planner-file",
        str(PLANNER_MODEL),
        "--zmq-port",
        str(spec.input_port),
        "--zmq-out-port",
        str(spec.output_port),
        "--inference-backend",
        inference_backend(args),
    ]


def player_command(motion_tokens: Path, spec: WorkerSpec) -> list[str]:
    return [
        str(SIM_PYTHON),
        str(PLAYER_SCRIPT),
        "--latent-input-file",
        str(motion_tokens),
        "--hz",
        str(CONTROL_HZ),
        "--source-hz",
        str(CONTROL_HZ),
        "--duration",
        "0",
        "--port",
        str(spec.input_port),
    ]


def worker_environment(spec: WorkerSpec, args: argparse.Namespace | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if inference_backend(args) == "tensorrt":
        env["CUDA_VISIBLE_DEVICES"] = spec.gpu_selector
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def discover_visible_gpus() -> tuple[list[str], str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        normalized = visible.strip()
        if normalized.lower() in NO_GPU_VALUES:
            raise RuntimeError("CUDA_VISIBLE_DEVICES explicitly exposes no GPU")
        tokens = [token.strip() for token in visible.split(",")]
        if any(not token for token in tokens):
            raise ValueError(f"invalid CUDA_VISIBLE_DEVICES={visible!r}")
        invalid = [token for token in tokens if not GPU_SELECTOR.fullmatch(token)]
        if invalid:
            raise ValueError(f"invalid CUDA_VISIBLE_DEVICES token(s): {invalid}")
        if len(set(tokens)) != len(tokens):
            raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate devices")
        return tokens, "CUDA_VISIBLE_DEVICES"

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(f"failed to run nvidia-smi: {error}") from error
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    devices: list[str] = []
    names: set[str] = set()
    for row in rows:
        if "," not in row:
            raise RuntimeError(f"unexpected nvidia-smi row: {row!r}")
        uuid, name = (part.strip() for part in row.split(",", 1))
        if not GPU_SELECTOR.fullmatch(uuid):
            raise RuntimeError(f"unexpected GPU UUID from nvidia-smi: {uuid!r}")
        devices.append(uuid)
        names.add(name)
    if not devices:
        raise RuntimeError("nvidia-smi returned no GPUs")
    if len(names) != 1:
        raise RuntimeError(f"visible GPUs must use one model for shared TensorRT caches, got {sorted(names)}")
    return devices, "nvidia-smi"


def build_worker_specs(
    devices: list[str],
    args: argparse.Namespace,
    job_count: int,
) -> list[WorkerSpec]:
    if not devices:
        if inference_backend(args) == "onnxruntime":
            devices = ["cpu"]
        else:
            raise ValueError("at least one visible GPU is required")
    workers_per_gpu = getattr(args, "workers_per_gpu", None)
    requested = (
        args.workers
        if args.workers is not None
        else len(devices) * (workers_per_gpu or DEFAULT_WORKERS_PER_GPU)
    )
    count = min(requested, job_count)
    if args.zmq_port_base + 2 * count - 1 > 65535:
        raise ValueError("worker ZMQ ports exceed 65535")
    if args.dds_domain_base + count - 1 > MAX_DDS_DOMAIN:
        raise ValueError(f"worker DDS domains exceed {MAX_DDS_DOMAIN}")
    specs = [
        WorkerSpec(
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
        if not port_free(port)
    )
    if busy:
        raise RuntimeError(f"replay port(s) already in use: {busy}")
    return specs


def worker_inventory(specs: list[WorkerSpec]) -> list[dict[str, Any]]:
    return [
        {
            "worker": spec.index,
            "gpu_selector": spec.gpu_selector,
            "gpu_index": spec.gpu_index,
            "gpu_replica": spec.gpu_replica,
            "dds_domain": spec.dds_domain,
            "zmq_input_port": spec.input_port,
            "zmq_output_port": spec.output_port,
        }
        for spec in specs
    ]


def runtime_assignment(spec: WorkerSpec) -> dict[str, Any]:
    return worker_inventory([spec])[0]


def worker_summary(specs: list[WorkerSpec]) -> dict[str, Any]:
    per_gpu: dict[str, int] = {}
    for spec in specs:
        per_gpu[spec.gpu_selector] = per_gpu.get(spec.gpu_selector, 0) + 1
    logical_cpus = os.cpu_count()
    return {
        "total_workers": len(specs),
        "workers_per_gpu": per_gpu,
        "logical_cpus": logical_cpus,
        "logical_cpus_per_worker": (
            logical_cpus / len(specs) if logical_cpus is not None and specs else None
        ),
    }


def resume_report(job: Job, rows: int, width: int) -> dict[str, Any] | None:
    output_report = report_path(job.output)
    if not job.output.exists() and not output_report.exists():
        return None
    if not job.output.is_file() or not output_report.is_file():
        raise RuntimeError(f"incomplete output/report pair for {job.episode_id}")
    report = json.loads(output_report.read_text(encoding="utf-8"))
    output = np.load(job.output, mmap_mode="r")
    expected = {
        "status": "complete",
        "input_path": str(job.token),
        "output_path": str(job.output),
        "input_dim": width,
        "token_rows": rows,
        "proprio_rows": rows,
        "hand_state_source": hand_source(width),
    }
    mismatches = {key: (report.get(key), value) for key, value in expected.items() if report.get(key) != value}
    if output.shape != (rows, 34) or output.dtype != np.float32:
        mismatches["output_array"] = ((output.shape, str(output.dtype)), ((rows, 34), "float32"))
    if not np.isfinite(output).all():
        mismatches["output_finite"] = (False, True)
    if mismatches:
        raise RuntimeError(f"resume validation failed for {job.episode_id}: {mismatches}")
    return report


def resume_planner_report(
    job: PlannerJob,
    trajectory: PlannerTrajectory,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    output_report = planner_report_path(job)
    paths = (job.token_output, job.proprio_output, output_report)
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise RuntimeError(f"incomplete planner token/proprio/report set for {job.episode_id}")
    report = json.loads(output_report.read_text(encoding="utf-8"))
    action = np.load(job.token_output, mmap_mode="r")
    proprio = np.load(job.proprio_output, mmap_mode="r")
    expected = {
        "status": "complete",
        "mode": "planner_trajectory",
        "episode_id": job.episode_id,
        "trajectory_source": str(job.trajectory_source) if job.trajectory_source else None,
        "token_output_path": str(job.token_output),
        "output_path": str(job.proprio_output),
        "token_rows": trajectory.rows,
        "proprio_rows": trajectory.rows,
        "input_dim": ACTION_DIM,
        "provenance": provenance or {},
    }
    mismatches = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if action.shape != (trajectory.rows, ACTION_DIM) or action.dtype != np.float32:
        mismatches["token_array"] = (
            (action.shape, str(action.dtype)),
            ((trajectory.rows, ACTION_DIM), "float32"),
        )
    if proprio.shape != (trajectory.rows, 34) or proprio.dtype != np.float32:
        mismatches["proprio_array"] = (
            (proprio.shape, str(proprio.dtype)),
            ((trajectory.rows, 34), "float32"),
        )
    if not np.isfinite(action).all() or not np.isfinite(proprio).all():
        mismatches["finite"] = (False, True)
    if action.shape == (trajectory.rows, ACTION_DIM) and not np.array_equal(
        action[:, MOTION_TOKEN_DIM:ACTION_DIM], trajectory.hand_state
    ):
        mismatches["hand_state"] = ("output differs", "trajectory")
    if mismatches:
        raise RuntimeError(f"planner resume validation failed for {job.episode_id}: {mismatches}")
    return report


def prepare_motion_tokens(token: Path, width: int, work_dir: Path) -> Path:
    if width == MOTION_TOKEN_DIM:
        return token
    motion_path = work_dir / "motion_tokens.npy"
    atomic_npy(motion_path, np.asarray(np.load(token)[:, :MOTION_TOKEN_DIM], dtype=np.float32))
    return motion_path


def cleanup_success_work(work_dir: Path) -> None:
    for name in SUCCESS_LOGS:
        path = work_dir / name
        try:
            if path.is_file():
                path.unlink()
        except OSError as error:
            print(f"WARNING: could not remove {path}: {error}", file=sys.stderr, flush=True)
    try:
        work_dir.rmdir()
    except OSError:
        pass


class TensorRTInitGate:
    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._local_lock = threading.Lock()
        self._ready = False

    def _launch(
        self,
        spec: WorkerSpec,
        log_path: Path,
        env: dict[str, str],
        timeout: float,
        stop_event: threading.Event,
    ) -> subprocess.Popen[bytes]:
        process = start_logged(deploy_command(spec), log_path, cwd=DEPLOY_ROOT, env=env)
        try:
            wait_for_log_pattern(process, log_path, DEPLOY_INIT_PATTERN, timeout, stop_event)
        except BaseException:
            stop_process(process, "deploy")
            raise
        return process

    def start(
        self,
        spec: WorkerSpec,
        log_path: Path,
        env: dict[str, str],
        timeout: float,
        stop_event: threading.Event,
    ) -> subprocess.Popen[bytes]:
        with self._local_lock:
            if not self._ready:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                with self._lock_path.open("a+") as lock_stream:
                    while True:
                        check_cancelled(stop_event)
                        try:
                            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            stop_event.wait(0.25)
                    try:
                        process = self._launch(spec, log_path, env, timeout, stop_event)
                        self._ready = True
                        return process
                    finally:
                        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        return self._launch(spec, log_path, env, timeout, stop_event)


def start_deploy(
    spec: WorkerSpec,
    args: argparse.Namespace,
    log_path: Path,
    env: dict[str, str],
    init_gate: TensorRTInitGate,
    stop_event: threading.Event,
) -> subprocess.Popen[bytes]:
    if inference_backend(args) == "tensorrt":
        return init_gate.start(spec, log_path, env, args.deploy_timeout, stop_event)
    process = start_logged(deploy_command(spec, args), log_path, cwd=DEPLOY_ROOT, env=env)
    try:
        wait_for_log_pattern(process, log_path, DEPLOY_INIT_PATTERN, args.deploy_timeout, stop_event)
    except BaseException:
        stop_process(process, "deploy")
        raise
    return process


def _quaternion_echo_error(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_quaternions = np.asarray(actual, dtype=np.float64).reshape(3, 4)
    expected_quaternions = np.asarray(expected, dtype=np.float64).reshape(3, 4)
    direct = np.max(np.abs(actual_quaternions - expected_quaternions), axis=1)
    flipped = np.max(np.abs(actual_quaternions + expected_quaternions), axis=1)
    return float(np.max(np.minimum(direct, flipped)))


def _warmup_orientation(orientation: np.ndarray) -> np.ndarray:
    result = np.asarray(orientation, dtype=np.float64).reshape(3, 4).copy()
    angle = 0.05
    delta = np.asarray([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    left = result[2]
    lw, lx, ly, lz = delta
    rw, rx, ry, rz = left
    result[2] = (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
    result[2] /= np.linalg.norm(result[2])
    return result.astype(np.float32).reshape(12)


def publish_planner_trajectory(
    trajectory: PlannerTrajectory,
    spec: WorkerSpec,
    release_event: threading.Event,
    cancel_event: threading.Event,
    stop_event: threading.Event,
    ready_result: queue.Queue[tuple[float | None, BaseException | None]],
    publish_result: queue.Queue[tuple[dict[str, Any] | None, BaseException | None]],
) -> None:
    """Publish warm-up and then one planner command per 50 Hz control tick."""
    context = None
    socket = None
    first_sent: float | None = None
    published = 0
    try:
        import zmq
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
            build_command_message,
            build_planner_message,
        )

        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind(f"tcp://*:{spec.input_port}")
        wait_interval(0.25, stop_event)
        warm_started = time.monotonic()
        ready_result.put((warm_started, None))

        warm_position = trajectory.vr_position[0].copy()
        warm_position[8] += 0.01
        warm_orientation = _warmup_orientation(trajectory.vr_orientation[0])
        warm_message = build_planner_message(
            mode=0,
            movement=(0.0, 0.0, 0.0),
            facing=trajectory.facing[0],
            speed=-1.0,
            height=float(trajectory.height[0]),
            vr_3pt_position=warm_position,
            vr_3pt_orientation=warm_orientation,
        )
        start_message = build_command_message(start=True, stop=False, planner=True)
        next_warm = warm_started
        while not release_event.is_set():
            if cancel_event.is_set():
                raise RunCancelled("planner publisher cancelled")
            check_cancelled(stop_event)
            now = time.monotonic()
            if now >= next_warm:
                socket.send(start_message)
                socket.send(warm_message)
                next_warm = now + 0.05
            release_event.wait(min(0.01, max(0.0, next_warm - now)))

        socket.send(start_message)
        for index in range(trajectory.rows):
            if cancel_event.is_set():
                raise RunCancelled("planner publisher cancelled")
            check_cancelled(stop_event)
            if index and index % 25 == 0:
                socket.send(start_message)
            message = build_planner_message(
                mode=int(trajectory.mode[index]),
                movement=trajectory.movement[index],
                facing=trajectory.facing[index],
                speed=float(trajectory.speed[index]),
                height=float(trajectory.height[index]),
                vr_3pt_position=trajectory.vr_position[index],
                vr_3pt_orientation=trajectory.vr_orientation[index],
            )
            sent_at = time.monotonic()
            socket.send(message)
            if first_sent is None:
                first_sent = sent_at
            published = index + 1
            deadline = first_sent + published / CONTROL_HZ
            remaining = deadline - time.monotonic()
            if remaining > 0 and cancel_event.wait(remaining):
                raise RunCancelled("planner publisher cancelled")
            check_cancelled(stop_event)
        publish_result.put(
            (
                {
                    "warm_started_monotonic": warm_started,
                    "first_frame_sent_monotonic": first_sent,
                    "last_frame_sent_monotonic": time.monotonic(),
                    "published_rows": published,
                },
                None,
            )
        )
        final_message = message
        while not cancel_event.wait(0.2):
            check_cancelled(stop_event)
            socket.send(start_message)
            socket.send(final_message)
    except BaseException as error:
        if ready_result.empty():
            ready_result.put((None, error))
        publish_result.put((None, error))
    finally:
        if socket is not None:
            try:
                from gear_sonic.utils.teleop.zmq.zmq_planner_sender import build_command_message

                socket.send(build_command_message(start=False, stop=True, planner=True))
            except Exception:
                pass
            socket.close()
        if context is not None:
            context.term()


def align_planner_frames(
    frames: list[dict[str, Any]],
    trajectory: PlannerTrajectory,
    *,
    first_sent_monotonic: float,
    echo_tol: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    expected_first = trajectory.vr_orientation[0]
    anchor = None
    for index, frame in enumerate(frames):
        received = frame.get("_recv_monotonic")
        orientation = frame.get("vr_3point_orientation")
        if received is None or received < first_sent_monotonic or orientation is None:
            continue
        actual = np.asarray(orientation)
        if actual.shape == (12,) and _quaternion_echo_error(actual, expected_first) <= echo_tol:
            anchor = index
            break
    if anchor is None:
        raise ValueError("no g1_debug frame echoed the first planner VR orientation")
    selected = frames[anchor : anchor + trajectory.rows]
    if len(selected) != trajectory.rows:
        raise ValueError(
            f"planner capture ended early: selected {len(selected)}/{trajectory.rows} rows"
        )

    indices = np.asarray([int(np.asarray(frame.get("index")).reshape(-1)[0]) for frame in selected])
    if indices.shape != (trajectory.rows,) or np.any(np.diff(indices) != 1):
        raise ValueError("planner g1_debug.index is missing, duplicated, or non-contiguous")

    echo_errors: list[float] = []
    arrays: dict[str, np.ndarray] = {}
    for row, frame in enumerate(selected):
        orientation = frame.get("vr_3point_orientation")
        if orientation is None or np.asarray(orientation).shape != (12,):
            raise ValueError(f"missing planner VR orientation echo at row {row}")
        echo_errors.append(
            _quaternion_echo_error(
                np.asarray(orientation), trajectory.vr_orientation[row]
            )
        )
    # The publisher and g1_debug producer are independent 50 Hz clocks, so an
    # occasional frame can echo an adjacent command even when debug indices are
    # complete and contiguous. Keep that phase drift as QA metadata; the hard
    # pairing gate is the continuous same-frame token/proprio sequence above.
    max_echo_error = max(echo_errors)
    echo_rows_over_tolerance = sum(error > echo_tol for error in echo_errors)

    for field in ALIGN_FIELDS:
        missing = [row for row, frame in enumerate(selected) if field not in frame]
        if missing:
            raise ValueError(f"required g1_debug field {field!r} is missing at rows {missing[:5]}")
        values = [np.asarray(frame[field], dtype=np.float64) for frame in selected]
        shape = values[0].shape
        if any(value.shape != shape for value in values):
            raise ValueError(f"g1_debug field {field!r} changed shape during planner rollout")
        arrays[field] = np.stack(values)
    if arrays["token_state"].shape != (trajectory.rows, MOTION_TOKEN_DIM):
        raise ValueError(
            f"planner token_state shape {arrays['token_state'].shape} != "
            f"({trajectory.rows},{MOTION_TOKEN_DIM})"
        )
    return arrays, {
        "anchor_idx": anchor,
        "anchor_debug_index": int(indices[0]),
        "last_debug_index": int(indices[-1]),
        "captured_frames": len(frames),
        "selected_frames": trajectory.rows,
        "missing_head": 0,
        "missing_tail": 0,
        "vr_orientation_echo_max_abs": max_echo_error,
        "vr_orientation_echo_rows_over_tolerance": echo_rows_over_tolerance,
        "command_to_anchor_seconds": float(
            selected[0]["_recv_monotonic"] - first_sent_monotonic
        ),
    }


def run_planner_job(
    job: PlannerJob,
    trajectory: PlannerTrajectory,
    args: argparse.Namespace,
    spec: WorkerSpec,
    init_gate: TensorRTInitGate,
    stop_event: threading.Event,
    *,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one dataset-neutral planner trajectory and export paired 66D/34D arrays."""
    trajectory_check = validate_planner_trajectory(trajectory)
    check_cancelled(stop_event)
    check_ports_free(spec)
    work_dir = (args.work_root / Path(job.episode_id)).resolve()
    if not work_dir.is_relative_to(args.work_root):
        raise ValueError(f"work path escapes --work-root: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    capture_duration = (
        trajectory.rows / CONTROL_HZ
        + args.startup_settle_margin
        + args.planner_mode_timeout
        + args.capture_tail_margin
        + 5.0
    )
    env = worker_environment(spec, args)
    processes: list[tuple[subprocess.Popen[bytes], str]] = []
    capture_stop = threading.Event()
    capture_result: queue.Queue[tuple[list[dict[str, Any]] | None, BaseException | None]] = queue.Queue(maxsize=1)
    release_event = threading.Event()
    publisher_cancel = threading.Event()
    ready_result: queue.Queue[tuple[float | None, BaseException | None]] = queue.Queue(maxsize=1)
    publish_result: queue.Queue[tuple[dict[str, Any] | None, BaseException | None]] = queue.Queue(maxsize=1)
    capture_thread: threading.Thread | None = None
    publisher_thread: threading.Thread | None = None

    def capture_frames() -> None:
        try:
            captured = capture_g1_debug_frames(
                capture_duration,
                port=spec.output_port,
                recv_timeout_ms=CAPTURE_RECV_TIMEOUT_MS,
                stop_event=capture_stop,
            )
            capture_result.put((captured, None))
        except BaseException as error:
            capture_result.put((None, error))

    try:
        sim_log = work_dir / "sim.log"
        sim = start_logged(
            [str(SIM_PYTHON), str(SIM_SCRIPT), "--domain-id", str(spec.dds_domain)],
            sim_log,
            cwd=WBC_ROOT,
            env=env,
        )
        processes.append((sim, "sim"))
        wait_interval(SIM_STARTUP_MARGIN, stop_event)
        if sim.poll() is not None:
            raise RuntimeError(f"sim exited with {sim.returncode}; see {sim_log}")

        deploy_log = work_dir / "deploy.log"
        deploy = start_deploy(spec, args, deploy_log, env, init_gate, stop_event)
        processes.append((deploy, "deploy"))
        capture_thread = threading.Thread(
            target=capture_frames,
            name=f"calc-proprio-planner-capture-{spec.index}",
            daemon=True,
        )
        capture_thread.start()
        wait_interval(args.capture_startup_margin, stop_event)
        if not capture_thread.is_alive():
            frames, error = capture_result.get_nowait()
            if error is not None:
                raise error
            raise RuntimeError(f"capture stopped before planner launch after {len(frames or [])} frame(s)")

        publisher_thread = threading.Thread(
            target=publish_planner_trajectory,
            args=(
                trajectory,
                spec,
                release_event,
                publisher_cancel,
                stop_event,
                ready_result,
                publish_result,
            ),
            name=f"calc-proprio-planner-publisher-{spec.index}",
            daemon=True,
        )
        publisher_thread.start()
        warm_started, ready_error = ready_result.get(timeout=5.0)
        if ready_error is not None:
            raise ready_error
        if warm_started is None:
            raise RuntimeError("planner publisher did not report warm-up start")
        wait_for_log_pattern(
            deploy,
            deploy_log,
            TELEOP_MODE_PATTERN,
            args.planner_mode_timeout,
            stop_event,
        )
        wait_interval(
            max(0.0, args.startup_settle_margin - (time.monotonic() - warm_started)),
            stop_event,
        )
        release_event.set()

        publisher_deadline = time.monotonic() + trajectory.rows / CONTROL_HZ + 10.0
        while publish_result.empty():
            check_cancelled(stop_event)
            if not publisher_thread.is_alive():
                raise RuntimeError("planner publisher stopped without a completion result")
            if time.monotonic() >= publisher_deadline:
                raise TimeoutError("planner publisher exceeded trajectory duration by 10 seconds")
            stop_event.wait(0.25)
        publish_metadata, publish_error = publish_result.get_nowait()
        if publish_error is not None:
            raise publish_error
        if publish_metadata is None:
            raise RuntimeError("planner publisher returned no metadata")
        wait_interval(args.capture_tail_margin, stop_event)
        capture_stop.set()
        capture_thread.join()
        frames, capture_error = capture_result.get_nowait()
        if capture_error is not None:
            raise capture_error
        if frames is None:
            raise RuntimeError("capture returned no frame list")

        if args.keep_work:
            atomic_pickle(work_dir / "frames.pkl", frames)
        arrays, alignment_report = align_planner_frames(
            frames,
            trajectory,
            first_sent_monotonic=float(publish_metadata["first_frame_sent_monotonic"]),
            echo_tol=args.planner_echo_tol,
        )
        if args.keep_work:
            atomic_npz(work_dir / "aligned.npz", arrays)
        del frames

        proprio, missing_head, missing_tail = build_pseudo_proprio(arrays)
        if missing_head or missing_tail:
            raise ValueError("planner alignment unexpectedly required edge fill")
        action = np.concatenate(
            (arrays["token_state"], trajectory.hand_state), axis=1
        ).astype(np.float32)
        validate_action_array(action)
        proprio, hand_report, warnings = apply_hand_state(proprio, action)
        check_report, check_warnings = self_check(
            proprio,
            token_rows=trajectory.rows,
            stats=str(args.stats) if args.stats is not None else None,
        )
        warnings.extend(check_warnings)
        report = {
            "schema_version": 3,
            "status": "complete",
            "mode": "planner_trajectory",
            "episode_id": job.episode_id,
            "trajectory_source": str(job.trajectory_source) if job.trajectory_source else None,
            "token_output_path": str(job.token_output),
            "output_path": str(job.proprio_output),
            "token_rows": trajectory.rows,
            "proprio_rows": int(proprio.shape[0]),
            "input_dim": ACTION_DIM,
            "missing_head_filled": 0,
            "missing_tail_filled": 0,
            "edge_frames_filled": 0,
            **hand_report,
            **check_report,
            **alignment_report,
            "trajectory_check": trajectory_check,
            "publisher": publish_metadata,
            "runtime_assets": runtime_assets(),
            "runtime_assignment": runtime_assignment(spec),
            "parameters": replay_parameters(args),
            "provenance": provenance or {},
            "work_retained": args.keep_work,
            "warnings": list(dict.fromkeys(warnings)),
        }
    finally:
        publisher_cancel.set()
        release_event.set()
        capture_stop.set()
        if publisher_thread is not None:
            publisher_thread.join(timeout=PROCESS_STOP_TIMEOUT)
        if capture_thread is not None:
            capture_thread.join()
        for process, name in reversed(processes):
            stop_process(process, name)

    check_cancelled(stop_event)
    atomic_npy(job.token_output, action)
    atomic_npy(job.proprio_output, proprio.astype(np.float32))
    atomic_json(planner_report_path(job), report)
    if not args.keep_work:
        cleanup_success_work(work_dir)
    return report


def run_job(
    job: Job,
    rows: int,
    width: int,
    args: argparse.Namespace,
    spec: WorkerSpec,
    init_gate: TensorRTInitGate,
    stop_event: threading.Event,
) -> dict[str, Any]:
    check_cancelled(stop_event)
    check_ports_free(spec)
    work_dir = (args.work_root / Path(job.episode_id)).resolve()
    if not work_dir.is_relative_to(args.work_root):
        raise ValueError(f"work path escapes --work-root: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    motion_tokens = prepare_motion_tokens(job.token, width, work_dir)
    capture_duration = rows / CONTROL_HZ + args.startup_settle_margin + args.capture_tail_margin
    env = worker_environment(spec, args)
    processes: list[tuple[subprocess.Popen[bytes], str]] = []
    capture_stop = threading.Event()
    capture_result: queue.Queue[tuple[list[dict[str, Any]] | None, BaseException | None]] = queue.Queue(maxsize=1)
    capture_thread: threading.Thread | None = None

    def capture_frames() -> None:
        try:
            frames = capture_g1_debug_frames(
                capture_duration,
                port=spec.output_port,
                recv_timeout_ms=CAPTURE_RECV_TIMEOUT_MS,
                stop_event=capture_stop,
            )
            capture_result.put((frames, None))
        except BaseException as error:
            capture_result.put((None, error))

    try:
        sim_log = work_dir / "sim.log"
        sim = start_logged(
            [str(SIM_PYTHON), str(SIM_SCRIPT), "--domain-id", str(spec.dds_domain)],
            sim_log,
            cwd=WBC_ROOT,
            env=env,
        )
        processes.append((sim, "sim"))
        wait_interval(SIM_STARTUP_MARGIN, stop_event)
        if sim.poll() is not None:
            raise RuntimeError(f"sim exited with {sim.returncode}; see {sim_log}")

        deploy = start_deploy(spec, args, work_dir / "deploy.log", env, init_gate, stop_event)
        processes.append((deploy, "deploy"))

        capture_thread = threading.Thread(
            target=capture_frames,
            name=f"calc-proprio-capture-{spec.index}",
            daemon=True,
        )
        capture_thread.start()
        wait_interval(args.capture_startup_margin, stop_event)
        if not capture_thread.is_alive():
            frames, error = capture_result.get_nowait()
            if error is not None:
                raise error
            raise RuntimeError(f"capture stopped before player launch after {len(frames or [])} frame(s)")

        player_log = work_dir / "player.log"
        player = start_logged(player_command(motion_tokens, spec), player_log, cwd=WBC_ROOT, env=env)
        processes.append((player, "player"))
        wait_process(
            player,
            timeout=rows / CONTROL_HZ + args.startup_settle_margin + 60.0,
            stop_event=stop_event,
            name="player",
            log_path=player_log,
        )
        while capture_thread.is_alive():
            check_cancelled(stop_event)
            capture_thread.join(timeout=0.25)
        frames, capture_error = capture_result.get_nowait()
        if capture_error is not None:
            raise capture_error
        if frames is None:
            raise RuntimeError("capture returned no frame list")

        check_cancelled(stop_event)
        motion = np.load(motion_tokens)
        if args.keep_work:
            atomic_pickle(work_dir / "frames.pkl", frames)
        alignment = align_frames_to_tokens(
            frames,
            num_tokens=rows,
            tokens=motion,
            tol=args.anchor_tol,
            max_offset=args.anchor_max_offset,
        )
        arrays = build_aligned_arrays(frames, alignment)
        if args.keep_work:
            atomic_npz(work_dir / "aligned.npz", arrays)
        del frames

        output, missing_head, missing_tail = build_pseudo_proprio(arrays)
        del arrays
        edge_frames = missing_head + missing_tail
        if edge_frames > args.max_edge_fill:
            raise ValueError(f"edge fill {edge_frames} exceeds --max-edge-fill={args.max_edge_fill}")
        tokens = np.load(job.token)
        output, hand_report, warnings = apply_hand_state(output, tokens)
        if edge_frames > 1:
            warnings.append("edge_fill_exceeds_expected_1")
        check_report, check_warnings = self_check(
            output,
            token_rows=rows,
            stats=str(args.stats) if args.stats is not None else None,
        )
        warnings.extend(check_warnings)
        alignment_report = {key: value for key, value in alignment.items() if key != "token_index_by_frame"}
        report = {
            "schema_version": 2,
            "status": "complete",
            "episode_id": job.episode_id,
            "input_path": str(job.token),
            "output_path": str(job.output),
            "input_dim": width,
            "token_rows": rows,
            "proprio_rows": int(output.shape[0]),
            "missing_head_filled": missing_head,
            "missing_tail_filled": missing_tail,
            "edge_frames_filled": edge_frames,
            **hand_report,
            **check_report,
            **alignment_report,
            "runtime_assets": runtime_assets(),
            "runtime_assignment": runtime_assignment(spec),
            "parameters": replay_parameters(args),
            "work_retained": args.keep_work,
            "warnings": list(dict.fromkeys(warnings)),
        }
    finally:
        capture_stop.set()
        if capture_thread is not None:
            capture_thread.join()
        for process, name in reversed(processes):
            stop_process(process, name)

    check_cancelled(stop_event)
    atomic_npy(job.output, output.astype(np.float32))
    atomic_json(report_path(job.output), report)
    if not args.keep_work:
        cleanup_success_work(work_dir)
    return report


def replay_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "control_hz": CONTROL_HZ,
        "workers": args.workers,
        "workers_per_gpu": getattr(args, "workers_per_gpu", None),
        "default_workers_per_gpu": DEFAULT_WORKERS_PER_GPU,
        "zmq_port_base": args.zmq_port_base,
        "dds_domain_base": args.dds_domain_base,
        "deploy_timeout": args.deploy_timeout,
        "capture_startup_margin": args.capture_startup_margin,
        "capture_tail_margin": args.capture_tail_margin,
        "startup_settle_margin": args.startup_settle_margin,
        "anchor_max_offset": args.anchor_max_offset,
        "anchor_tol": args.anchor_tol,
        "max_edge_fill": args.max_edge_fill,
        "planner_mode_timeout": args.planner_mode_timeout,
        "planner_echo_tol": args.planner_echo_tol,
        "keep_work": args.keep_work,
        "stats": str(args.stats) if args.stats is not None else None,
    }


def discover_jobs(args: argparse.Namespace) -> tuple[list[Job], dict[str, Any] | None]:
    if args.token is not None:
        token = resolve_path(args.token)
        output = resolve_path(args.output)
        if token == output:
            raise ValueError("--output must not overwrite --token")
        return [Job(token, output, token.stem)], None

    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"--input-root is not a directory: {input_root}")
    if output_root.is_relative_to(input_root) or input_root.is_relative_to(output_root):
        raise ValueError("--input-root and --output-root must not overlap")

    candidates: list[tuple[str, Path]] = []
    for candidate in input_root.glob(args.token_glob):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(input_root):
            raise ValueError(f"token symlink escapes --input-root: {candidate}")
        relative = candidate.relative_to(input_root)
        candidates.append((relative.with_suffix("").as_posix(), resolved))
    candidates.sort(key=lambda item: item[0])
    if args.episode:
        requested = set(args.episode)
        found = {episode_id for episode_id, _ in candidates}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"--episode values did not match discovered tokens: {missing}")
        candidates = [item for item in candidates if item[0] in requested]
    if not candidates:
        raise FileNotFoundError(f"no token files matched {args.token_glob!r} under {input_root}")

    jobs = [
        Job(token, output_root / token.relative_to(input_root), episode_id)
        for episode_id, token in candidates
    ]
    roots = {
        "input_root": str(Path(args.input_root)),
        "input_root_resolved": str(input_root),
        "output_root": str(Path(args.output_root)),
        "output_root_resolved": str(output_root),
    }
    return jobs, roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token")
    parser.add_argument("--output")
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--planner-trajectory", help="dataset-neutral planner trajectory .npz")
    parser.add_argument("--token-output", help="planner mode [T,66] output .npy")
    parser.add_argument("--proprio-output", help="planner mode [T,34] output .npy")
    parser.add_argument("--token-glob", default="**/*.npy")
    parser.add_argument("--episode", action="append", help="exact relative path without .npy; repeatable")
    parser.add_argument("--work-root", default=".runs/calc_proprio/work")
    worker_group = parser.add_mutually_exclusive_group()
    worker_group.add_argument(
        "--workers", type=int, help="directory live mode: explicit total worker count"
    )
    worker_group.add_argument(
        "--workers-per-gpu",
        type=int,
        help=f"directory live mode: workers per visible GPU (default: {DEFAULT_WORKERS_PER_GPU})",
    )
    parser.add_argument("--zmq-port-base", type=int, default=DEFAULT_ZMQ_PORT_BASE)
    parser.add_argument("--dds-domain-base", type=int, default=0)
    parser.add_argument(
        "--inference-backend",
        default="tensorrt",
        choices=("tensorrt", "onnxruntime", "ort", "cpu"),
        help="live deploy backend; cpu/ort aliases select ONNX Runtime CPU",
    )
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--stats")
    parser.add_argument("--deploy-timeout", type=float, default=150.0)
    parser.add_argument("--capture-startup-margin", type=float, default=2.0)
    parser.add_argument("--capture-tail-margin", type=float, default=15.0)
    parser.add_argument("--startup-settle-margin", type=float, default=11.0)
    parser.add_argument("--anchor-max-offset", type=int, default=5)
    parser.add_argument("--anchor-tol", type=float, default=1e-5)
    parser.add_argument("--max-edge-fill", type=int, default=1)
    parser.add_argument("--planner-mode-timeout", type=float, default=15.0)
    parser.add_argument("--planner-echo-tol", type=float, default=1e-4)
    args = parser.parse_args()

    token_single = args.token is not None or args.output is not None
    directory = args.input_root is not None or args.output_root is not None
    planner_single = any(
        value is not None
        for value in (args.planner_trajectory, args.token_output, args.proprio_output)
    )
    if sum((token_single, directory, planner_single)) != 1:
        parser.error(
            "choose exactly one mode: --token/--output, --input-root/--output-root, "
            "or --planner-trajectory/--token-output/--proprio-output"
        )
    if token_single and (args.token is None or args.output is None):
        parser.error("--token and --output must be supplied together")
    if directory and (args.input_root is None or args.output_root is None):
        parser.error("--input-root and --output-root must be supplied together")
    if planner_single and any(
        value is None
        for value in (args.planner_trajectory, args.token_output, args.proprio_output)
    ):
        parser.error(
            "--planner-trajectory, --token-output, and --proprio-output must be supplied together"
        )
    if not directory and args.episode:
        parser.error("--episode is only valid in directory mode")
    if not directory and (args.workers is not None or args.workers_per_gpu is not None):
        parser.error("--workers and --workers-per-gpu are only valid in directory mode")
    if args.output and Path(args.output).suffix != ".npy":
        parser.error("--output must end in .npy")
    if args.planner_trajectory and Path(args.planner_trajectory).suffix != ".npz":
        parser.error("--planner-trajectory must end in .npz")
    if args.token_output and Path(args.token_output).suffix != ".npy":
        parser.error("--token-output must end in .npy")
    if args.proprio_output and Path(args.proprio_output).suffix != ".npy":
        parser.error("--proprio-output must end in .npy")
    if args.token_output and resolve_path(args.token_output) == resolve_path(args.proprio_output):
        parser.error("--token-output and --proprio-output must differ")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.workers_per_gpu is not None and args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be at least 1")
    if not 1 <= args.zmq_port_base <= 65534:
        parser.error("--zmq-port-base must be between 1 and 65534")
    if not 0 <= args.dds_domain_base <= MAX_DDS_DOMAIN:
        parser.error(f"--dds-domain-base must be between 0 and {MAX_DDS_DOMAIN}")
    if args.deploy_timeout <= 0 or args.capture_startup_margin < 0 or args.capture_tail_margin < 0:
        parser.error("timeout and capture margins must be non-negative, with a positive deploy timeout")
    if args.startup_settle_margin < 0 or args.anchor_max_offset <= 0 or args.anchor_tol <= 0:
        parser.error("settle margin must be non-negative; anchor offset/tolerance must be positive")
    if args.max_edge_fill < 0:
        parser.error("--max-edge-fill must be non-negative")
    if args.planner_mode_timeout <= 0 or args.planner_echo_tol <= 0:
        parser.error("planner mode timeout and echo tolerance must be positive")

    args.work_root = resolve_path(args.work_root)
    args.stats = resolve_path(args.stats) if args.stats else None
    if args.stats is not None and not args.stats.is_file():
        parser.error(f"--stats does not exist: {args.stats}")
    return args

def validation_entry(job: Job, rows: int, width: int) -> dict[str, Any]:
    warnings = ["input_has_no_hand_action"] if width == MOTION_TOKEN_DIM else []
    return {
        "episode_id": job.episode_id,
        "input_path": str(job.token),
        "output_path": str(job.output),
        "input_dim": width,
        "token_rows": rows,
        "hand_state_source": hand_source(width),
        "warnings": warnings,
    }


def directory_work_root(base: Path, roots: dict[str, Any]) -> Path:
    identity = f"{roots['input_root_resolved']}\0{roots['output_root_resolved']}"
    return (base / hashlib.sha256(identity.encode()).hexdigest()[:12]).resolve()


def process_live_job(
    job: Job,
    args: argparse.Namespace,
    spec: WorkerSpec,
    init_gate: TensorRTInitGate,
    stop_event: threading.Event,
) -> dict[str, Any]:
    rows: int | None = None
    width: int | None = None
    try:
        rows, width = validate_token(job.token)
        existing = resume_report(job, rows, width)
        entry = validation_entry(job, rows, width)
        entry["runtime_assignment"] = runtime_assignment(spec)
        if existing is not None:
            if not args.resume:
                raise FileExistsError(f"output already exists: {job.output}")
            entry["warnings"] = existing.get("warnings", [])
            entry["resume_skipped"] = True
            group = "needs_review" if entry["warnings"] else "skipped"
            return {"group": group, "entry": entry, "rows": rows, "width": width, "output_rows": rows}
        report = run_job(job, rows, width, args, spec, init_gate, stop_event)
        entry["warnings"] = report["warnings"]
        group = "needs_review" if report["warnings"] else "complete"
        return {"group": group, "entry": entry, "rows": rows, "width": width, "output_rows": rows}
    except Exception as error:
        entry = {
            "episode_id": job.episode_id,
            "input_path": str(job.token),
            "runtime_assignment": runtime_assignment(spec),
            "error": f"{type(error).__name__}: {error}",
        }
        return {"group": "failed", "entry": entry, "rows": rows, "width": width, "output_rows": 0}


def process_validation_job(job: Job) -> dict[str, Any]:
    try:
        rows, width = validate_token(job.token)
        entry = validation_entry(job, rows, width)
        group = "needs_review" if entry["warnings"] else "complete"
        return {"group": group, "entry": entry, "rows": rows, "width": width, "output_rows": 0}
    except Exception as error:
        entry = {
            "episode_id": job.episode_id,
            "input_path": str(job.token),
            "error": f"{type(error).__name__}: {error}",
        }
        return {"group": "failed", "entry": entry, "rows": None, "width": None, "output_rows": 0}


def record_result(
    result: dict[str, Any],
    counts: dict[str, int],
    terminal_counts: dict[str, int],
    journal: TextIO,
) -> None:
    group = result["group"]
    terminal_counts[group] += 1
    rows = result["rows"]
    width = result["width"]
    if rows is not None and width is not None:
        counts[f"input_{width}d"] += 1
        counts["input_frames"] += rows
    counts["output_frames"] += result["output_rows"]
    journal.write(json.dumps({"recorded_at": now_utc(), "terminal_group": group, **result["entry"]}) + "\n")
    journal.flush()


def build_manifest(
    roots: dict[str, Any],
    args: argparse.Namespace,
    specs: list[WorkerSpec],
    gpu_source: str | None,
    job_count: int,
    counts: dict[str, int],
    terminal_counts: dict[str, int],
    status: str,
) -> dict[str, Any]:
    processed = sum(terminal_counts.values())
    return {
        "schema_version": 2,
        "status": status,
        "complete": status in {"complete", "complete-with-warnings"},
        "updated_at": now_utc(),
        "validate_only": args.validate_only,
        **roots,
        "token_glob": args.token_glob,
        "work_root_resolved": str(args.work_root),
        "runtime_assets": runtime_assets(),
        "parameters": replay_parameters(args),
        "gpu_discovery_source": gpu_source,
        "workers": worker_inventory(specs),
        "worker_summary": worker_summary(specs),
        "job_count": job_count,
        "processed_jobs": processed,
        "pending_jobs": job_count - processed,
        "counts": counts,
        "terminal_counts": terminal_counts,
        "result_journal": "pseudo_proprio_results.jsonl",
    }


def output_lock_path(output_root: Path) -> Path:
    digest = hashlib.sha256(str(output_root).encode()).hexdigest()[:16]
    return OUTPUT_LOCK_ROOT / f"{digest}.lock"


@contextmanager
def exclusive_output_lock(output_root: Path) -> Iterator[None]:
    lock_path = output_lock_path(output_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another calc_proprio run holds output root {output_root}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run_single(
    job: Job,
    args: argparse.Namespace,
    spec: WorkerSpec | None,
    stop_event: threading.Event,
) -> int:
    rows, width = validate_token(job.token)
    if args.validate_only:
        print(json.dumps(validation_entry(job, rows, width), indent=2))
        return 0
    if spec is None:
        raise RuntimeError("live single-file run requires a worker")
    existing = resume_report(job, rows, width)
    if existing is not None:
        if not args.resume:
            raise FileExistsError(f"output already exists: {job.output}")
        print(f"resume: validated and skipped {job.episode_id}")
        return 0
    report = run_job(job, rows, width, args, spec, TensorRTInitGate(TRT_LOCK_PATH), stop_event)
    print(json.dumps(report, indent=2))
    return 0


def run_directory(
    jobs: list[Job],
    roots: dict[str, Any],
    args: argparse.Namespace,
    specs: list[WorkerSpec],
    gpu_source: str | None,
    stop_event: threading.Event,
) -> int:
    output_root = Path(roots["output_root_resolved"])
    manifest_path = output_root / "pseudo_proprio_manifest.json"
    journal_path = output_root / "pseudo_proprio_results.jsonl"
    counts = {"input_64d": 0, "input_66d": 0, "input_frames": 0, "output_frames": 0}
    terminal_counts = {"complete": 0, "needs_review": 0, "skipped": 0, "failed": 0}
    interrupted = False

    with exclusive_output_lock(output_root):
        output_root.mkdir(parents=True, exist_ok=True)
        with journal_path.open("a", encoding="utf-8") as journal:
            atomic_json(
                manifest_path,
                build_manifest(
                    roots, args, specs, gpu_source, len(jobs), counts, terminal_counts, "running"
                ),
            )
            last_manifest = time.monotonic()
            if args.validate_only:
                try:
                    for index, job in enumerate(jobs):
                        print(f"[{index + 1}/{len(jobs)}] validate {job.episode_id}", flush=True)
                        result = process_validation_job(job)
                        record_result(result, counts, terminal_counts, journal)
                        if result["group"] == "failed":
                            print(f"FAILED {job.episode_id}: {result['entry']['error']}", file=sys.stderr, flush=True)
                            if args.fail_fast:
                                break
                        if time.monotonic() - last_manifest >= MANIFEST_INTERVAL:
                            atomic_json(
                                manifest_path,
                                build_manifest(
                                    roots, args, specs, gpu_source, len(jobs), counts, terminal_counts, "running"
                                ),
                            )
                            last_manifest = time.monotonic()
                except KeyboardInterrupt:
                    interrupted = True
            else:
                job_queue: queue.Queue[Job] = queue.Queue()
                result_queue: queue.Queue[dict[str, Any]] = queue.Queue()
                for job in jobs:
                    job_queue.put(job)
                init_gate = TensorRTInitGate(TRT_LOCK_PATH)

                def worker_loop(spec: WorkerSpec) -> None:
                    while not stop_event.is_set():
                        try:
                            job = job_queue.get_nowait()
                        except queue.Empty:
                            return
                        result = process_live_job(job, args, spec, init_gate, stop_event)
                        result_queue.put(result)
                        if result["group"] == "failed" and args.fail_fast:
                            stop_event.set()

                threads = [
                    threading.Thread(
                        target=worker_loop,
                        args=(spec,),
                        name=f"calc-proprio-worker-{spec.index}",
                    )
                    for spec in specs
                ]
                started_threads: list[threading.Thread] = []
                try:
                    for thread in threads:
                        thread.start()
                        started_threads.append(thread)
                    while any(thread.is_alive() for thread in started_threads) or not result_queue.empty():
                        try:
                            result = result_queue.get(timeout=0.25)
                        except queue.Empty:
                            result = None
                        if result is not None:
                            record_result(result, counts, terminal_counts, journal)
                            entry = result["entry"]
                            processed = sum(terminal_counts.values())
                            assignment = entry.get("runtime_assignment", {})
                            print(
                                f"[{processed}/{len(jobs)}] worker={assignment.get('worker')} "
                                f"gpu={assignment.get('gpu_selector')} {result['group']} {entry['episode_id']}",
                                flush=True,
                            )
                            if result["group"] == "failed":
                                print(f"FAILED {entry['episode_id']}: {entry['error']}", file=sys.stderr, flush=True)
                        if time.monotonic() - last_manifest >= MANIFEST_INTERVAL:
                            atomic_json(
                                manifest_path,
                                build_manifest(
                                    roots, args, specs, gpu_source, len(jobs), counts, terminal_counts, "running"
                                ),
                            )
                            last_manifest = time.monotonic()
                except KeyboardInterrupt:
                    interrupted = True
                    stop_event.set()
                finally:
                    if interrupted:
                        stop_event.set()
                    for thread in started_threads:
                        thread.join()
                    while True:
                        try:
                            result = result_queue.get_nowait()
                        except queue.Empty:
                            break
                        record_result(result, counts, terminal_counts, journal)

            processed = sum(terminal_counts.values())
            if interrupted:
                status = "interrupted"
            elif terminal_counts["failed"] or processed != len(jobs):
                status = "failed"
            elif terminal_counts["needs_review"]:
                status = "complete-with-warnings"
            else:
                status = "complete"
            manifest = build_manifest(
                roots, args, specs, gpu_source, len(jobs), counts, terminal_counts, status
            )
            atomic_json(manifest_path, manifest)

    print(f"manifest -> {manifest_path}")
    if interrupted:
        return 130
    return 0 if manifest["complete"] else 1


def run_planner_single(
    job: PlannerJob,
    trajectory: PlannerTrajectory,
    args: argparse.Namespace,
    spec: WorkerSpec | None,
    stop_event: threading.Event,
) -> int:
    check = validate_planner_trajectory(trajectory)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "mode": "planner_trajectory",
                    "trajectory_source": str(job.trajectory_source),
                    "token_output_path": str(job.token_output),
                    "proprio_output_path": str(job.proprio_output),
                    **check,
                },
                indent=2,
            )
        )
        return 0
    if spec is None:
        raise RuntimeError("live planner run requires a worker")
    existing = resume_planner_report(job, trajectory)
    if existing is not None:
        if not args.resume:
            raise FileExistsError(
                f"planner output already exists: {job.token_output} / {job.proprio_output}"
            )
        print(f"resume: validated and skipped {job.episode_id}")
        return 0
    report = run_planner_job(
        job,
        trajectory,
        args,
        spec,
        TensorRTInitGate(TRT_LOCK_PATH),
        stop_event,
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    stop_event = threading.Event()
    if args.planner_trajectory is not None:
        trajectory_source = resolve_path(args.planner_trajectory)
        token_output = resolve_path(args.token_output)
        proprio_output = resolve_path(args.proprio_output)
        if trajectory_source in {token_output, proprio_output}:
            raise ValueError("planner outputs must not overwrite the trajectory input")
        trajectory = load_planner_trajectory(trajectory_source)
        job = PlannerJob(
            trajectory_source=trajectory_source,
            token_output=token_output,
            proprio_output=proprio_output,
            episode_id=trajectory_source.stem,
        )
        specs: list[WorkerSpec] = []
        if not args.validate_only:
            validate_runtime_assets()
            devices, _ = (["cpu"], "cpu") if inference_backend(args) == "onnxruntime" else discover_visible_gpus()
            specs = build_worker_specs(devices, args, 1)
        return run_planner_single(
            job,
            trajectory,
            args,
            specs[0] if specs else None,
            stop_event,
        )

    jobs, roots = discover_jobs(args)
    if roots is not None:
        args.work_root = directory_work_root(args.work_root, roots)

    specs = []
    gpu_source: str | None = None
    if not args.validate_only:
        validate_runtime_assets()
        if inference_backend(args) == "onnxruntime":
            devices, gpu_source = ["cpu"], "cpu"
        else:
            devices, gpu_source = discover_visible_gpus()
        worker_jobs = 1 if roots is None else len(jobs)
        specs = build_worker_specs(devices, args, worker_jobs)
        print(json.dumps({"worker_plan": worker_summary(specs)}, indent=2), flush=True)

    if roots is None:
        return run_single(jobs[0], args, specs[0] if specs else None, stop_event)
    return run_directory(jobs, roots, args, specs, gpu_source, stop_event)

def handle_termination(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_termination)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None

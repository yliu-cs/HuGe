"""Loopback HTTP adapter from HumanoidArena observations to Huge actions."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import secrets
import signal
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from huge.data.humanoid_arena_preprocess import GROUP_IDX, GROUP_ORDER
from huge.data.schemas import (
    ACTION_DIM,
    ACTION_HORIZON,
    ACTION_SCHEMA_VERSION,
    CONTROL_FREQ_HZ,
    HAND_DIM,
    HAND_POLARITY,
    MOTION_TOKEN_DIM,
    PROPRIO_DIM,
    PROPRIO_LAYOUT_VERSION,
    combine_action_dict,
)
from huge.inference.humanoidarena_io_trace import (
    TRACE_RECEIPT_VERSION,
    ModelIOTraceRecorder,
    TraceFinishConflictError,
    TraceReceipt,
)
from huge.inference.sonic_adapter import HandHysteresis


PROTOCOL_VERSION = "huge_humanoidarena_http_v1"
EXPECTED_ROBOT_TYPE = "unitree_g1_refpose_v3_1"
SOURCE_STATE_DIM = 64
MAX_JSON_BODY_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_EDGE = 4096
MAPPING_VERSION = "humanoidarena_rotlocal_v31_to_huge_proprio_v1"
MODEL_NOISE_SEED_ALGORITHM = "huge_ha_model_noise_v1"
_MODEL_NOISE_SEED_MASK = (1 << 63) - 1
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "proprio_stats_sonic.json",
    "proprio_norm_config.json",
)
_STEP_RE = re.compile(r"^step_([0-9]+)$")
_GPU_UUID_RE = re.compile(r"^(?:GPU|MIG-GPU)-[A-Za-z0-9-]+$")
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")


class CheckpointValidationError(ValueError):
    """Raised before model construction when a checkpoint contract is invalid."""


class ProtocolError(ValueError):
    """A stable HTTP-facing validation error."""

    def __init__(self, http_status: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.http_status = int(http_status)
        self.error_code = str(error_code)


@dataclass(frozen=True)
class CheckpointContract:
    checkpoint_dir: Path
    step: int
    checkpoint_task: str
    base_vlm_path: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    norm_config: dict[str, bool]
    file_sha256: dict[str, str]
    checkpoint_sha256: str


@dataclass(frozen=True)
class ParsedInferRequest:
    front_rgb: np.ndarray
    state: np.ndarray


def derive_model_noise_seed(episode_seed: int) -> int:
    payload = f"{MODEL_NOISE_SEED_ALGORITHM}|{int(episode_seed)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & _MODEL_NOISE_SEED_MASK


@dataclass
class EpisodeState:
    reset_generation: int = 0
    last_seeded_episode: int | None = None
    sequence: int = 0
    last_completed_sequence: int = 0
    has_reset: bool = False
    hand_hysteresis: HandHysteresis = field(default_factory=HandHysteresis)
    model_noise_seed: int | None = None
    model_noise_generator: torch.Generator | None = field(default=None, repr=False)

    @property
    def hand_state(self) -> np.ndarray:
        return self.hand_hysteresis.state.copy()

    def reset(
        self,
        seed: int | None,
        *,
        generator_device: str | torch.device = "cpu",
    ) -> None:
        model_noise_seed = None
        model_noise_generator = None
        if seed is not None:
            model_noise_seed = derive_model_noise_seed(seed)
            model_noise_generator = torch.Generator(device=torch.device(generator_device))
            model_noise_generator.manual_seed(model_noise_seed)
        self.reset_generation += 1
        self.sequence = 0
        self.last_completed_sequence = 0
        self.has_reset = True
        self.hand_hysteresis.reset()
        self.last_seeded_episode = None
        self.model_noise_seed = model_noise_seed
        self.model_noise_generator = model_noise_generator
        if seed is not None:
            self.last_seeded_episode = int(seed)

    def require_reset(self) -> None:
        if not self.has_reset:
            raise ProtocolError(HTTPStatus.CONFLICT, "infer_before_reset", "infer requires a prior reset")

    def next_sequence(self) -> int:
        self.require_reset()
        self.sequence += 1
        return self.sequence

    def complete(self, sequence: int, action: np.ndarray) -> None:
        terminal = np.asarray(action[-1, MOTION_TOKEN_DIM:ACTION_DIM], dtype=np.float32)
        if terminal.min(initial=0.0) < 0.0 or terminal.max(initial=0.0) > 1.0:
            terminal = 1.0 / (1.0 + np.exp(-terminal))
        self.hand_hysteresis.update(terminal)
        self.last_completed_sequence = int(sequence)


@dataclass(frozen=True)
class ServerContract:
    checkpoint: CheckpointContract
    task: str
    isaac_task: str
    instruction: str
    expected_cuda_uuid: str
    run_id: str
    worker_id: str
    status_file: Path
    attempt_dir: Path
    host: str = "127.0.0.1"
    port: int = 18080
    device: str = "cuda:0"


class PolicyLike(Protocol):
    def get_action_with_trace(
        self,
        observation: dict[str, Any],
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, np.ndarray], Any]: ...


class TraceRecorderLike(Protocol):
    def begin_episode(self, *, episode_seed: int, seeded_reset_generation: int) -> None: ...

    def record_call(
        self,
        *,
        http_response_sequence: int,
        reset_generation: int,
        front_rgb: np.ndarray,
        state64: np.ndarray,
        policy_trace: Any,
    ) -> None: ...

    def finish_episode(
        self,
        *,
        result_identity: str,
        episode_seed: int,
        outcome_sha256: str,
    ) -> TraceReceipt: ...

    def abandon_episode(self, *, reason: str) -> None: ...


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
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


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"{label} must contain a JSON object: {path}")
    return payload


def _require_equal(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    actual = mapping.get(key)
    if actual != expected:
        raise CheckpointValidationError(f"{label}.{key}={actual!r}, expected {expected!r}")


def resolve_checkpoint_contract(
    checkpoint_dir: str | Path,
    *,
    allowed_root: str | Path,
    checkpoint_task: str,
    repo_root: str | Path,
) -> CheckpointContract:
    root = Path(allowed_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    if not root.is_dir():
        raise CheckpointValidationError(f"checkpoint allowed root is not a directory: {root}")
    if not checkpoint.is_dir() or not _is_relative_to(checkpoint, root):
        raise CheckpointValidationError(f"checkpoint must resolve inside {root}: {checkpoint}")
    relative = checkpoint.relative_to(root)
    if not relative.parts or relative.parts[0] != checkpoint_task:
        actual = relative.parts[0] if relative.parts else ""
        raise CheckpointValidationError(
            f"checkpoint task component {actual!r} does not match {checkpoint_task!r}"
        )
    match = _STEP_RE.fullmatch(checkpoint.name)
    if match is None:
        raise CheckpointValidationError(f"checkpoint basename must be step_<N>: {checkpoint.name}")

    hashes: dict[str, str] = {}
    for name in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint / name
        resolved = path.resolve()
        if not path.is_file() or not _is_relative_to(resolved, checkpoint):
            raise CheckpointValidationError(f"missing or escaping checkpoint file: {path}")
        hashes[name] = _sha256_file(path)

    payload = _read_json_object(checkpoint / "config.json", label="checkpoint config")
    config = payload.get("config")
    metadata = payload.get("metadata")
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise CheckpointValidationError("config.json requires object fields 'config' and 'metadata'")
    _require_equal(config, "state_dim", PROPRIO_DIM, label="config")
    _require_equal(config, "action_dim", ACTION_DIM, label="config")
    _require_equal(config, "action_horizon", ACTION_HORIZON, label="config")
    _require_equal(config, "use_qwen", True, label="config")
    expected_metadata = {
        "schema_version": ACTION_SCHEMA_VERSION,
        "action_dim": ACTION_DIM,
        "motion_token_dim": MOTION_TOKEN_DIM,
        "hand_dim": HAND_DIM,
        "hand_polarity": HAND_POLARITY,
        "action_horizon": ACTION_HORIZON,
        "control_freq_hz": CONTROL_FREQ_HZ,
        "proprio_layout": PROPRIO_LAYOUT_VERSION,
        "proprio_dim": PROPRIO_DIM,
    }
    for key, expected in expected_metadata.items():
        _require_equal(metadata, key, expected, label="metadata")
    action_schema = metadata.get("action_schema")
    if action_schema is not None:
        if not isinstance(action_schema, dict):
            raise CheckpointValidationError("metadata.action_schema must be an object")
        for key in (
            "schema_version",
            "action_dim",
            "motion_token_dim",
            "hand_dim",
            "hand_polarity",
            "action_horizon",
            "control_freq_hz",
        ):
            _require_equal(action_schema, key, expected_metadata[key], label="metadata.action_schema")

    base_raw = config.get("base_vlm_path") or metadata.get("model_name_or_path")
    if not isinstance(base_raw, str) or not base_raw.strip() or "://" in base_raw:
        raise CheckpointValidationError("checkpoint must name a local base_vlm_path")
    base_path = Path(base_raw).expanduser()
    base_path = (repo / base_path).resolve() if not base_path.is_absolute() else base_path.resolve()
    if not base_path.is_dir():
        raise CheckpointValidationError(f"local Qwen base asset not found: {base_path}")

    norm_config = _read_json_object(
        checkpoint / "proprio_norm_config.json", label="proprio normalization config"
    )
    for key in ("use_percentiles", "clip_outliers"):
        if not isinstance(norm_config.get(key), bool):
            raise CheckpointValidationError(f"proprio normalization config requires boolean {key}")
    normalized_norm = {
        "use_percentiles": bool(norm_config["use_percentiles"]),
        "clip_outliers": bool(norm_config["clip_outliers"]),
    }
    checkpoint_digest = hashlib.sha256(_canonical_json_bytes(hashes)).hexdigest()
    return CheckpointContract(
        checkpoint_dir=checkpoint,
        step=int(match.group(1)),
        checkpoint_task=checkpoint_task,
        base_vlm_path=base_path,
        config=dict(config),
        metadata=dict(metadata),
        norm_config=normalized_norm,
        file_sha256=hashes,
        checkpoint_sha256=checkpoint_digest,
    )


def _rot6d_row_to_quat_wxyz(rot6d: Sequence[float]) -> np.ndarray:
    flat = np.asarray(rot6d, dtype=np.float64).reshape(-1)
    if flat.shape != (6,) or not np.isfinite(flat).all():
        raise ValueError("root rot6d must contain 6 finite numbers")
    col0 = flat[[0, 2, 4]]
    col1 = flat[[1, 3, 5]]
    norm0 = float(np.linalg.norm(col0))
    if norm0 < 1e-8:
        raise ValueError("root rot6d first column is degenerate")
    col0 = col0 / norm0
    col1 = col1 - float(np.dot(col0, col1)) * col0
    norm1 = float(np.linalg.norm(col1))
    if norm1 < 1e-8:
        raise ValueError("root rot6d second column is degenerate")
    col1 = col1 / norm1
    col2 = np.cross(col0, col1)
    rot = np.stack((col0, col1, col2), axis=-1)
    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    return np.asarray(quat_xyzw[[3, 0, 1, 2]], dtype=np.float32)


def _projected_gravity_from_rot6d(rot6d: Sequence[float]) -> np.ndarray:
    quat = _rot6d_row_to_quat_wxyz(rot6d)
    xyzw = quat[[1, 2, 3, 0]]
    return Rotation.from_quat(xyzw).inv().apply([0.0, 0.0, -1.0]).astype(np.float32)


def _grouped_qpos_indices() -> np.ndarray:
    indices = [index for group in GROUP_ORDER for index in GROUP_IDX[group]]
    if len(indices) != 29 or sorted(indices) != list(range(29)):
        raise RuntimeError("Huge HumanoidArena joint mapping is not a 29-D permutation")
    return np.asarray(indices, dtype=np.int64)


GROUPED_QPOS_INDICES = _grouped_qpos_indices()


def map_humanoidarena_state_to_huge(
    state: Sequence[float] | np.ndarray,
    hand_state: Sequence[float] | np.ndarray,
) -> np.ndarray:
    source = np.asarray(state, dtype=np.float32).reshape(-1)
    if source.shape != (SOURCE_STATE_DIM,):
        raise ValueError(f"HumanoidArena observation state must have 64 values, got {source.shape}")
    if not np.isfinite(source).all():
        raise ValueError("HumanoidArena observation state must be finite")
    hands = np.asarray(hand_state, dtype=np.float32).reshape(-1)
    if hands.shape != (HAND_DIM,) or not np.isfinite(hands).all():
        raise ValueError("tracked hand state must contain 2 finite values")
    qpos = source[6:35][GROUPED_QPOS_INDICES]
    gravity = _projected_gravity_from_rot6d(source[:6])
    proprio = np.concatenate((qpos, hands, gravity), axis=0).astype(np.float32, copy=False)
    if proprio.shape != (PROPRIO_DIM,) or not np.isfinite(proprio).all():
        raise ValueError("mapped Huge proprio is not a finite 34-D vector")
    return proprio


def validate_action_chunk(action: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"Huge action chunk must have shape ({ACTION_HORIZON}, {ACTION_DIM}), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Huge action chunk must be finite")
    return array


def build_response(action: np.ndarray, episode: EpisodeState) -> dict[str, Any]:
    array = validate_action_chunk(action)
    sequence = episode.next_sequence()
    episode.complete(sequence, array)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "episode_seed": episode.last_seeded_episode,
        "reset_generation": episode.reset_generation,
        "sequence": sequence,
        "action_chunk": array.tolist(),
        "latent64_chunk": array[:, :MOTION_TOKEN_DIM].tolist(),
        "action_shape": [ACTION_HORIZON, ACTION_DIM],
        "latent_shape": [ACTION_HORIZON, MOTION_TOKEN_DIM],
    }


def _require_payload_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_json_object", "request body must be a JSON object")
    return payload


def parse_reset_payload(payload: Any) -> int | None:
    body = _require_payload_object(payload)
    unknown = set(body).difference({"seed"})
    if unknown:
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "unknown_reset_fields", "reset contains unknown fields")
    if "seed" not in body:
        return None
    seed = body["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_seed", "reset seed must be an integer")
    return int(seed)


def parse_finish_episode_payload(payload: Any) -> tuple[str, int, str]:
    body = _require_payload_object(payload)
    required = {"result_identity", "episode_seed", "outcome_sha256"}
    if set(body) != required:
        raise ProtocolError(
            HTTPStatus.BAD_REQUEST,
            "invalid_finish_fields",
            "finish_episode requires exactly result_identity, episode_seed, and outcome_sha256",
        )
    result_identity = body["result_identity"]
    if not isinstance(result_identity, str) or not result_identity:
        raise ProtocolError(
            HTTPStatus.BAD_REQUEST,
            "invalid_result_identity",
            "result_identity must be a non-empty string",
        )
    episode_seed = body["episode_seed"]
    if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
        raise ProtocolError(
            HTTPStatus.BAD_REQUEST, "invalid_episode_seed", "episode_seed must be an integer"
        )
    outcome_sha256 = body["outcome_sha256"]
    if not isinstance(outcome_sha256, str) or not _SHA256_HEX_RE.fullmatch(outcome_sha256):
        raise ProtocolError(
            HTTPStatus.BAD_REQUEST,
            "invalid_outcome_sha256",
            "outcome_sha256 must be 64 lowercase hexadecimal characters",
        )
    return result_identity, int(episode_seed), outcome_sha256


def parse_infer_payload(
    payload: Any,
    *,
    expected_task: str,
    expected_robot_type: str = EXPECTED_ROBOT_TYPE,
) -> ParsedInferRequest:
    body = _require_payload_object(payload)
    if body.get("task") != expected_task:
        raise ProtocolError(HTTPStatus.UNPROCESSABLE_ENTITY, "task_mismatch", "request task does not match server task")
    if body.get("robot_type") != expected_robot_type:
        raise ProtocolError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            "robot_type_mismatch",
            "request robot_type does not match server contract",
        )
    if body.get("return_chunk") is not True:
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "return_chunk_required", "return_chunk must be true")
    observation = body.get("observation")
    if not isinstance(observation, dict):
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_observation", "observation must be an object")
    images = observation.get("images")
    if not isinstance(images, dict) or "front" not in images or len(images) > 4:
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_images", "observation.images.front is required")
    front = images["front"]
    if not isinstance(front, dict):
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_front_image", "front image must be an object")
    shape = front.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or any(value <= 0 or value > MAX_IMAGE_EDGE for value in shape)
        or shape[-1] != 3
    ):
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_image_shape", "front image shape must be [H,W,3]")
    if front.get("dtype") != "uint8":
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_image_dtype", "front image dtype must be uint8")
    encoded = front.get("data_b64")
    if not isinstance(encoded, str):
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_image_data", "front image data_b64 must be a string")
    expected_bytes = math.prod(shape)
    if expected_bytes > MAX_IMAGE_BYTES or len(encoded) > (MAX_IMAGE_BYTES * 4 // 3 + 8):
        raise ProtocolError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "image_too_large", "front image exceeds size limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_image_base64", "front image base64 is invalid") from exc
    if len(raw) != expected_bytes:
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "image_length_mismatch", "front image byte length does not match shape")
    front_rgb = np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape)).copy()

    state_raw = observation.get("state")
    if not isinstance(state_raw, list) or len(state_raw) != SOURCE_STATE_DIM:
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_state_shape", "observation.state must contain 64 numbers")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in state_raw):
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_state_type", "observation.state must contain numbers")
    state = np.asarray(state_raw, dtype=np.float32)
    if not np.isfinite(state).all():
        raise ProtocolError(HTTPStatus.BAD_REQUEST, "non_finite_state", "observation.state must be finite")
    return ParsedInferRequest(front_rgb=front_rgb, state=state)


def load_safetensors_strict(model: Any, path: str | Path, *, device: str) -> None:
    from huge.models.qwen_sonicfm import load_portable_safetensors_strict

    load_portable_safetensors_strict(model, path, device=device)


def load_policy_strict(contract: CheckpointContract, *, device: str) -> PolicyLike:
    from huge.data.state_normalizer import StateNormalizer
    from huge.inference.policy import HugePolicy
    from huge.models.qwen_sonicfm import QwenSonicFM, QwenSonicFMConfig

    config_values = dict(contract.config)
    config_values["base_vlm_path"] = str(contract.base_vlm_path)
    model = QwenSonicFM(QwenSonicFMConfig.from_checkpoint_config(config_values)).to(device)
    load_safetensors_strict(model, contract.checkpoint_dir / "model.safetensors", device=device)
    model.eval()
    normalizer = StateNormalizer.from_json(
        contract.checkpoint_dir / "proprio_stats_sonic.json",
        use_percentiles=contract.norm_config["use_percentiles"],
        clip_outliers=contract.norm_config["clip_outliers"],
    )
    return HugePolicy(
        model,
        device=device,
        qwen_processor_path=contract.base_vlm_path,
        normalizer=normalizer,
    )


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


def _mapping_source_sha256() -> str:
    import huge.data.humanoid_arena_preprocess as preprocess

    return _sha256_file(Path(preprocess.__file__).resolve())


def _normalize_uuid(value: str) -> str:
    return str(value).strip().upper()


def detect_cuda_uuid(device: str = "cuda:0") -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the server child")
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, "uuid", None)
    if uuid is None:
        uuid_getter = getattr(torch.cuda, "get_device_uuid", None)
        if callable(uuid_getter):
            uuid = uuid_getter(index)
    if uuid is None:
        raise RuntimeError("PyTorch CUDA runtime did not expose the selected GPU UUID")
    token = str(uuid)
    if token and not token.upper().startswith(("GPU-", "MIG-GPU-")):
        token = f"GPU-{token}"
    return token


class InferenceRuntime:
    def __init__(
        self,
        contract: ServerContract,
        policy: PolicyLike,
        *,
        recorder: TraceRecorderLike,
        cuda_uuid: str,
    ) -> None:
        self.contract = contract
        self.policy = policy
        self.recorder = recorder
        self.cuda_uuid = cuda_uuid
        self.episode = EpisodeState()
        self.state = "starting"
        self.fatal_error_code = ""
        self._lock = threading.Lock()
        self._trace_status = "incomplete"
        self._trace_error = ""
        self._trace_session_active = False
        self._trace_recording_enabled = False
        self._trace_finished = False
        self._trace_seeded_reset_generation = 0
        self._trace_fallback_session_id = ""
        self._trace_successful_record_count = 0
        self._trace_last_http_response_sequence = 0
        self._finish_key: tuple[str, int, str] | None = None
        self._finish_receipt: dict[str, Any] | None = None
        self._write_status("starting")

    def _status_payload(self, state: str, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "state": state,
            "run_id": self.contract.run_id,
            "worker_id": self.contract.worker_id,
            "task": self.contract.task,
            "isaac_task": self.contract.isaac_task,
            "checkpoint_sha256": self.contract.checkpoint.checkpoint_sha256,
            "cuda_logical_device": 0,
            "cuda_uuid": self.cuda_uuid,
            "reset_generation": self.episode.reset_generation,
            "last_seeded_episode": self.episode.last_seeded_episode,
            "infer_sequence": self.episode.sequence,
            "last_completed_sequence": self.episode.last_completed_sequence,
            "fatal_error_code": self.fatal_error_code,
            "trace_status": self._trace_status,
            "trace_error": self._trace_error,
            "model_noise_seed_algorithm": MODEL_NOISE_SEED_ALGORITHM,
            "updated_at": time.time(),
        }
        payload.update(extra)
        return payload

    def _write_status(self, state: str, **extra: Any) -> None:
        self.state = state
        atomic_write_json(self.contract.status_file, self._status_payload(state, **extra))

    @staticmethod
    def _trace_exception_message(operation: str, exc: BaseException) -> str:
        return f"{operation}: {type(exc).__name__}: {exc}"[:1000]

    def _mark_trace_degraded(self, operation: str, exc: BaseException) -> None:
        message = self._trace_exception_message(operation, exc)
        self._trace_status = "degraded"
        self._trace_error = f"{self._trace_error}; {message}".strip("; ")

    def _abandon_trace_locked(self, reason: str) -> None:
        if not self._trace_session_active or self._trace_finished:
            return
        try:
            self.recorder.abandon_episode(reason=reason)
        except Exception as exc:
            self._mark_trace_degraded("abandon_episode", exc)
        else:
            if self._trace_status != "degraded":
                self._trace_status = "incomplete"
        finally:
            self._trace_recording_enabled = False
            self._trace_session_active = False

    def _begin_trace_locked(self, episode_seed: int) -> None:
        self._abandon_trace_locked("superseded by a new seeded reset")
        self._finish_key = None
        self._finish_receipt = None
        self._trace_status = "incomplete"
        self._trace_error = ""
        self._trace_session_active = True
        self._trace_recording_enabled = False
        self._trace_finished = False
        self._trace_seeded_reset_generation = self.episode.reset_generation
        self._trace_fallback_session_id = secrets.token_hex(16)
        self._trace_successful_record_count = 0
        self._trace_last_http_response_sequence = 0
        try:
            self.recorder.begin_episode(
                episode_seed=episode_seed,
                seeded_reset_generation=self._trace_seeded_reset_generation,
            )
        except Exception as exc:
            self._mark_trace_degraded("begin_episode", exc)
        else:
            self._trace_recording_enabled = True

    def _degraded_receipt_locked(
        self,
        *,
        result_identity: str,
        episode_seed: int,
        outcome_sha256: str,
        error: str,
    ) -> dict[str, Any]:
        if not self._trace_fallback_session_id:
            self._trace_fallback_session_id = secrets.token_hex(16)
        combined_error = f"{self._trace_error}; {error}".strip("; ")
        return {
            "schema_version": TRACE_RECEIPT_VERSION,
            "run_id": self.contract.run_id,
            "worker_id": self.contract.worker_id,
            "trace_session_id": self._trace_fallback_session_id,
            "result_identity": result_identity,
            "episode_seed": episode_seed,
            "seeded_reset_generation": self._trace_seeded_reset_generation,
            "outcome_sha256": outcome_sha256,
            "record_count": self._trace_successful_record_count,
            "last_trace_call_index": self._trace_successful_record_count,
            "last_http_response_sequence": self._trace_last_http_response_sequence,
            "trace_status": "degraded",
            "manifest_relative_path": None,
            "manifest_sha256": None,
            "success_marker_relative_path": None,
            "total_bytes": 0,
            "error": combined_error or "model I/O trace unavailable",
        }

    def mark_ready(self) -> None:
        with self._lock:
            self._write_status("ready")

    def mark_closing(self) -> None:
        with self._lock:
            self._abandon_trace_locked("server shutdown before finish_episode")
            if self.state != "fatal":
                self._write_status("closing")

    def mark_fatal(self, error_code: str, message: str) -> None:
        with self._lock:
            if self.state == "fatal":
                return
            self.fatal_error_code = str(error_code)
            self._write_status("fatal", message=str(message)[:500])

    def health(self) -> dict[str, Any]:
        with self._lock:
            if self.state != "ready":
                raise ProtocolError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "server_not_ready",
                    f"server state is {self.state}",
                )
            checkpoint = self.contract.checkpoint
            return {
                "status": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "checkpoint_dir": str(checkpoint.checkpoint_dir),
                "task": self.contract.task,
                "isaac_task": self.contract.isaac_task,
                "prompt_sha256": hashlib.sha256(self.contract.instruction.encode("utf-8")).hexdigest(),
                "cuda_logical_device": 0,
                "cuda_uuid": self.cuda_uuid,
                "state_dim": PROPRIO_DIM,
                "action_dim": ACTION_DIM,
                "motion_token_dim": MOTION_TOKEN_DIM,
                "action_horizon": ACTION_HORIZON,
                "control_hz": CONTROL_FREQ_HZ,
                "mapping_version": MAPPING_VERSION,
                "mapping_source_sha256": _mapping_source_sha256(),
                "model_noise_seed_algorithm": MODEL_NOISE_SEED_ALGORITHM,
                "trace_status": self._trace_status,
                "trace_error": self._trace_error,
            }

    def reset(self, payload: Any) -> dict[str, Any]:
        seed = parse_reset_payload(payload)
        with self._lock:
            if self.state == "fatal":
                raise ProtocolError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "server_fatal",
                    "server is in fatal state",
                )
            self.episode.reset(seed, generator_device=self.contract.device)
            if seed is not None:
                self._begin_trace_locked(seed)
            self._write_status("ready")
            return {
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "reset_generation": self.episode.reset_generation,
                "episode_seed": self.episode.last_seeded_episode,
            }

    def infer(self, payload: Any) -> dict[str, Any]:
        parsed = parse_infer_payload(
            payload,
            expected_task=self.contract.isaac_task,
            expected_robot_type=EXPECTED_ROBOT_TYPE,
        )
        with self._lock:
            if self.state == "fatal":
                raise ProtocolError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "server_fatal",
                    "server is in fatal state",
                )
            self.episode.require_reset()
            next_sequence = self.episode.sequence + 1
            self._write_status("infer_in_progress", pending_sequence=next_sequence)
            try:
                proprio = map_humanoidarena_state_to_huge(parsed.state, self.episode.hand_state)
                observation = {
                    "video.ego_view": parsed.front_rgb,
                    "state": proprio,
                    "language": self.contract.instruction,
                }
                parts, policy_trace = self.policy.get_action_with_trace(
                    observation,
                    generator=self.episode.model_noise_generator,
                )
                action = combine_action_dict(parts)
                if action.ndim == 3 and action.shape[0] == 1:
                    action = action[0]
                response = build_response(action, self.episode)
            except ProtocolError:
                self._write_status("ready")
                raise
            except Exception as exc:
                self.fatal_error_code = "model_infer_failed"
                self._write_status("fatal", message=str(exc)[:500])
                raise

            self._trace_last_http_response_sequence = int(response["sequence"])
            if self._trace_recording_enabled:
                try:
                    self.recorder.record_call(
                        http_response_sequence=int(response["sequence"]),
                        reset_generation=int(response["reset_generation"]),
                        front_rgb=parsed.front_rgb,
                        state64=parsed.state,
                        policy_trace=policy_trace,
                    )
                    self._trace_successful_record_count += 1
                except Exception as exc:
                    self._mark_trace_degraded("record_call", exc)
                    self._trace_recording_enabled = False
            self._write_status("ready")
            return response

    def finish_episode(self, payload: Any) -> dict[str, Any]:
        result_identity, episode_seed, outcome_sha256 = parse_finish_episode_payload(payload)
        finish_key = (result_identity, episode_seed, outcome_sha256)
        with self._lock:
            if self.state == "fatal":
                raise ProtocolError(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "server_fatal",
                    "server is in fatal state",
                )
            if self._finish_receipt is not None:
                if finish_key == self._finish_key:
                    return dict(self._finish_receipt)
                raise ProtocolError(
                    HTTPStatus.CONFLICT,
                    "trace_finish_conflict",
                    "finish_episode replay does not match the current trace session",
                )

            try:
                receipt = self.recorder.finish_episode(
                    result_identity=result_identity,
                    episode_seed=episode_seed,
                    outcome_sha256=outcome_sha256,
                )
                receipt_payload = asdict(receipt)
            except TraceFinishConflictError as exc:
                raise ProtocolError(
                    HTTPStatus.CONFLICT, "trace_finish_conflict", str(exc)
                ) from exc
            except Exception as exc:
                self._mark_trace_degraded("finish_episode", exc)
                receipt_payload = self._degraded_receipt_locked(
                    result_identity=result_identity,
                    episode_seed=episode_seed,
                    outcome_sha256=outcome_sha256,
                    error=self._trace_exception_message("finish_episode", exc),
                )

            trace_status = receipt_payload.get("trace_status")
            if trace_status not in {"complete", "degraded", "incomplete"}:
                invalid = ValueError(f"invalid recorder trace_status {trace_status!r}")
                self._mark_trace_degraded("finish_episode", invalid)
                receipt_payload = self._degraded_receipt_locked(
                    result_identity=result_identity,
                    episode_seed=episode_seed,
                    outcome_sha256=outcome_sha256,
                    error=str(invalid),
                )
            elif self._trace_status == "degraded":
                receipt_payload["trace_status"] = "degraded"
                receipt_payload["success_marker_relative_path"] = None
                prior_error = receipt_payload.get("error")
                receipt_payload["error"] = "; ".join(
                    item for item in (self._trace_error, str(prior_error or "")) if item
                )

            self._trace_status = str(receipt_payload["trace_status"])
            self._trace_error = str(receipt_payload.get("error") or "")
            self._trace_finished = True
            self._trace_recording_enabled = False
            self._finish_key = finish_key
            self._finish_receipt = dict(receipt_payload)
            self._write_status("ready")
            return dict(receipt_payload)


class HumanoidArenaHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        runtime: InferenceRuntime,
        *,
        max_body_bytes: int = MAX_JSON_BODY_BYTES,
    ) -> None:
        self.runtime = runtime
        self.max_body_bytes = int(max_body_bytes)
        super().__init__(address, HumanoidArenaRequestHandler)


class ServerShutdownCoordinator:
    """Request server shutdown without blocking the signal handler."""

    def __init__(self, server: HumanoidArenaHTTPServer) -> None:
        self.server = server
        self.started = threading.Event()
        self._worker_thread: threading.Thread | None = None

    @property
    def worker_thread(self) -> threading.Thread | None:
        return self._worker_thread

    def request(self) -> bool:
        if self.started.is_set():
            return False
        self.started.set()
        self._worker_thread = threading.Thread(target=self._stop_server, daemon=True)
        self._worker_thread.start()
        return True

    def _stop_server(self) -> None:
        try:
            self.server.runtime.mark_closing()
        finally:
            self.server.shutdown()


class HumanoidArenaRequestHandler(BaseHTTPRequestHandler):
    server: HumanoidArenaHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[humanoidarena_http] {self.address_string()} {format % args}")

    def _send_json(self, status: int, payload: Any) -> None:
        body = _canonical_json_bytes(payload)
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_error(self, error: ProtocolError) -> None:
        self._send_json(error.http_status, {"error_code": error.error_code, "message": str(error)})

    def _read_json(self) -> Any:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ProtocolError(HTTPStatus.BAD_REQUEST, "content_type_required", "Content-Type must be application/json")
        length_raw = self.headers.get("Content-Length")
        try:
            length = int(length_raw or "")
        except ValueError as exc:
            raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid Content-Length") from exc
        if length < 0:
            raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_content_length", "invalid Content-Length")
        if length > self.server.max_body_bytes:
            raise ProtocolError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "payload_too_large", "JSON body exceeds size limit")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ProtocolError(HTTPStatus.BAD_REQUEST, "short_request_body", "request body is incomplete")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError(HTTPStatus.BAD_REQUEST, "invalid_json", "request body is not valid UTF-8 JSON") from exc

    def do_GET(self) -> None:
        try:
            if self.path != "/health":
                raise ProtocolError(HTTPStatus.NOT_FOUND, "unknown_endpoint", "unknown endpoint")
            self._send_json(HTTPStatus.OK, self.server.runtime.health())
        except ProtocolError as exc:
            self._send_error(exc)
        except Exception as exc:
            self.server.runtime.mark_fatal("health_failed", str(exc))
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error_code": "health_failed", "message": "health failed"})

    def do_POST(self) -> None:
        try:
            if self.path not in {"/reset", "/infer", "/finish_episode"}:
                raise ProtocolError(HTTPStatus.NOT_FOUND, "unknown_endpoint", "unknown endpoint")
            payload = self._read_json()
            if self.path == "/reset":
                result = self.server.runtime.reset(payload)
            elif self.path == "/infer":
                result = self.server.runtime.infer(payload)
            else:
                result = self.server.runtime.finish_episode(payload)
            self._send_json(HTTPStatus.OK, result)
        except ProtocolError as exc:
            self._send_error(exc)
        except Exception:
            traceback.print_exc()
            self.server.runtime.mark_fatal("request_failed", "unhandled request failure")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error_code": "request_failed", "message": "unrecoverable server failure"},
            )


def create_server(
    contract: ServerContract,
    policy: PolicyLike,
    *,
    recorder: TraceRecorderLike,
    cuda_uuid: str,
    max_body_bytes: int = MAX_JSON_BODY_BYTES,
) -> HumanoidArenaHTTPServer:
    runtime = InferenceRuntime(contract, policy, recorder=recorder, cuda_uuid=cuda_uuid)
    return HumanoidArenaHTTPServer((contract.host, contract.port), runtime, max_body_bytes=max_body_bytes)


def _path_not_under_humanoidarena(path: Path, repo_root: Path) -> None:
    humanoidarena = (repo_root / "HumanoidArena").resolve()
    if _is_relative_to(path.resolve(), humanoidarena):
        raise ValueError(f"server write path cannot be inside HumanoidArena: {path}")


def _resolve_model_io_root(value: str | Path, *, attempt_dir: str | Path) -> Path:
    attempt = Path(attempt_dir).expanduser().resolve()
    model_io_root = Path(value).expanduser().resolve()
    expected = (attempt / "model_io").resolve()
    if model_io_root != expected or not _is_relative_to(model_io_root, attempt):
        raise ValueError(f"model_io_root must be the current attempt model_io directory: {expected}")
    return model_io_root


def _require_sha256_cli(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_HEX_RE.fullmatch(value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--allowed-checkpoint-root", required=True)
    parser.add_argument("--checkpoint-task", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--isaac-task", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--expected-cuda-uuid", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--model-io-root", required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--source-inventory-digest", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-body-bytes", type=int, default=MAX_JSON_BODY_BYTES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.host != "127.0.0.1":
        raise ValueError("HumanoidArena Huge server must bind 127.0.0.1")
    if not (1 <= args.port <= 65535):
        raise ValueError("port must be in 1..65535")
    if args.device != "cuda:0":
        raise ValueError("server child device must be cuda:0")
    if not _GPU_UUID_RE.fullmatch(args.expected_cuda_uuid):
        raise ValueError("expected CUDA UUID is malformed")
    attempt_dir = Path(args.attempt_dir).expanduser().resolve()
    status_file = Path(args.status_file).expanduser().resolve()
    if not attempt_dir.is_dir() or not _is_relative_to(status_file, attempt_dir):
        raise ValueError("status file must resolve inside an existing attempt directory")
    model_io_root = _resolve_model_io_root(args.model_io_root, attempt_dir=attempt_dir)
    contract_hash = _require_sha256_cli(args.contract_hash, label="contract_hash")
    source_inventory_digest = _require_sha256_cli(
        args.source_inventory_digest, label="source_inventory_digest"
    )
    _path_not_under_humanoidarena(attempt_dir, repo_root)
    _path_not_under_humanoidarena(model_io_root, repo_root)
    contract = resolve_checkpoint_contract(
        args.checkpoint,
        allowed_root=args.allowed_checkpoint_root,
        checkpoint_task=args.checkpoint_task,
        repo_root=repo_root,
    )
    server_contract = ServerContract(
        checkpoint=contract,
        task=args.task,
        isaac_task=args.isaac_task,
        instruction=args.instruction,
        expected_cuda_uuid=args.expected_cuda_uuid,
        run_id=args.run_id,
        worker_id=args.worker_id,
        status_file=status_file,
        attempt_dir=attempt_dir,
        host=args.host,
        port=args.port,
        device=args.device,
    )
    starting_status = {
        "protocol_version": PROTOCOL_VERSION,
        "state": "starting",
        "run_id": args.run_id,
        "worker_id": args.worker_id,
        "task": args.task,
        "checkpoint_sha256": contract.checkpoint_sha256,
        "cuda_uuid": args.expected_cuda_uuid,
        "trace_status": "incomplete",
        "trace_error": "",
        "model_noise_seed_algorithm": MODEL_NOISE_SEED_ALGORITHM,
        "updated_at": time.time(),
    }
    atomic_write_json(status_file, starting_status)
    try:
        recorder = ModelIOTraceRecorder(
            model_io_root=model_io_root,
            run_id=args.run_id,
            worker_id=args.worker_id,
            task=args.task,
            checkpoint_sha256=contract.checkpoint_sha256,
            contract_hash=contract_hash,
            source_inventory_digest=source_inventory_digest,
            model_architecture=str(
                contract.config.get("model_architecture", "qwenpi_v3") or "qwenpi_v3"
            ),
            action_horizon=int(contract.config.get("action_horizon", ACTION_HORIZON)),
            num_inference_steps=int(contract.config.get("num_inference_steps", 4)),
            mapping_version=MAPPING_VERSION,
        )
        policy = load_policy_strict(contract, device=args.device)
        cuda_uuid = detect_cuda_uuid(args.device)
    except Exception as exc:
        atomic_write_json(
            status_file,
            {
                **starting_status,
                "state": "fatal",
                "fatal_error_code": "model_load_failed",
                "message": str(exc)[:500],
                "updated_at": time.time(),
            },
        )
        raise
    if _normalize_uuid(cuda_uuid) != _normalize_uuid(args.expected_cuda_uuid):
        atomic_write_json(
            status_file,
            {**starting_status, "state": "fatal", "fatal_error_code": "cuda_uuid_mismatch", "actual_cuda_uuid": cuda_uuid},
        )
        raise RuntimeError(
            f"server child CUDA UUID {cuda_uuid!r} does not match expected {args.expected_cuda_uuid!r}"
        )
    server = create_server(
        server_contract,
        policy,
        recorder=recorder,
        cuda_uuid=cuda_uuid,
        max_body_bytes=args.max_body_bytes,
    )
    server.runtime.mark_ready()
    shutdown_coordinator = ServerShutdownCoordinator(server)

    def request_shutdown(signum: int, _frame: Any) -> None:
        print(f"[humanoidarena_http] received signal {signum}; shutting down")
        shutdown_coordinator.request()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    try:
        print(
            f"[humanoidarena_http] ready host={args.host} port={args.port} "
            f"task={args.task} checkpoint={contract.checkpoint_dir} cuda_uuid={cuda_uuid}"
        )
        server.serve_forever(poll_interval=0.2)
    finally:
        try:
            server.runtime.mark_closing()
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

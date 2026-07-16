"""Atomic model I/O traces for Huge HumanoidArena evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any

import numpy as np
import torch


TRACE_SCHEMA_VERSION = "huge_humanoidarena_model_io_trace_v1"
TRACE_RECEIPT_VERSION = "huge_humanoidarena_model_io_receipt_v1"

_ACTION_DIM = 66
_PROPRIO_DIM = 34
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_QWEN_KEY_RE = re.compile(r"[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TraceReceipt:
    """Episode-level trace result returned to the evaluation shadow."""

    schema_version: str
    run_id: str
    worker_id: str
    trace_session_id: str
    result_identity: str
    episode_seed: int
    seeded_reset_generation: int
    outcome_sha256: str
    record_count: int
    last_trace_call_index: int
    last_http_response_sequence: int
    trace_status: str
    manifest_relative_path: str | None
    manifest_sha256: str | None
    success_marker_relative_path: str | None
    total_bytes: int
    error: str | None


@dataclass
class _EpisodeSession:
    trace_session_id: str
    directory: Path
    episode_seed: int
    seeded_reset_generation: int
    next_trace_call_index: int = 1
    last_http_response_sequence: int = 0
    degraded_reasons: list[str] = field(default_factory=list)
    published_call_hashes: dict[int, tuple[str, str]] = field(default_factory=dict)
    finished_key: tuple[str, int, str] | None = None
    receipt: TraceReceipt | None = None
    abandoned: bool = False


class _TraceValidationError(ValueError):
    pass


class TraceFinishConflictError(RuntimeError):
    """Raised when a finalized session receives a different finish tuple."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _json_safe_scalar_or_list(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return bool(np.isfinite(value))
    if isinstance(value, list):
        return all(_json_safe_scalar_or_list(item) for item in value)
    return False


def _python_type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _wire_array_metadata(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("wire arrays with object dtype are not traceable")
    payload = array.tobytes(order="C")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _tensor_to_numpy(value: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        original_dtype = str(tensor.dtype).removeprefix("torch.")
        if tensor.is_floating_point() or tensor.is_complex():
            finite = bool(torch.isfinite(tensor).all().item())
        else:
            finite = True
        if tensor.dtype == torch.bfloat16:
            array = tensor.view(torch.uint16).numpy().copy()
            encoding = "bfloat16_uint16_bits"
        else:
            array = tensor.numpy().copy()
            encoding = "native"
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value).copy()
        original_dtype = str(array.dtype)
        if np.issubdtype(array.dtype, np.number):
            finite = bool(np.isfinite(array).all())
        else:
            finite = True
        encoding = "native"
    else:
        raise TypeError(f"expected tensor or ndarray, got {_python_type_name(value)}")

    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise TypeError("object arrays are forbidden in model I/O NPZ files")
    metadata = {
        "original_dtype": original_dtype,
        "saved_dtype": str(array.dtype),
        "shape": list(array.shape),
        "finite": finite,
        "encoding": encoding,
    }
    return array, metadata


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    np.savez(stream, **{key: arrays[key] for key in sorted(arrays)})
    return stream.getvalue()


class ModelIOTraceRecorder:
    """Write one isolated, validated model I/O trace session per episode."""

    def __init__(
        self,
        *,
        model_io_root: str | Path,
        run_id: str,
        worker_id: str | int,
        task: str,
        checkpoint_sha256: str,
        contract_hash: str,
        source_inventory_digest: str,
        model_architecture: str,
        action_horizon: int,
        num_inference_steps: int,
        mapping_version: str,
    ) -> None:
        self.model_io_root = Path(model_io_root).expanduser().resolve()
        if self.model_io_root.name != "model_io":
            raise ValueError("model_io_root must name the current attempt's model_io directory")
        self._attempt_root = self.model_io_root.parent
        self.model_io_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_id": run_id,
            "worker_id": str(worker_id),
            "task": task,
            "checkpoint_sha256": checkpoint_sha256,
            "contract_hash": contract_hash,
            "source_inventory_digest": source_inventory_digest,
            "model_architecture": model_architecture,
            "mapping_version": mapping_version,
        }
        for name, value in metadata.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
            setattr(self, name, value)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self._worker_component = self._worker_directory_component(self.worker_id)
        self._worker_root = self._confined(self.model_io_root / self._worker_component)
        self._worker_root.mkdir(parents=True, exist_ok=True)
        self._session: _EpisodeSession | None = None

    @staticmethod
    def _worker_directory_component(worker_id: str) -> str:
        if worker_id.isdecimal():
            return f"w{int(worker_id):02d}"
        match = re.fullmatch(r"w(\d+)", worker_id)
        if match:
            return f"w{int(match.group(1)):02d}"
        match = re.fullmatch(r"worker[-_]?(\d+)", worker_id)
        if match:
            return f"w{int(match.group(1)):02d}"
        raise ValueError("worker_id must be numeric or use the wNN/worker-N form")

    def _confined(self, path: Path) -> Path:
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(self.model_io_root):
            raise ValueError(f"trace path escapes model_io root: {path}")
        return resolved

    def _receipt_relative(self, path: Path) -> str:
        confined = self._confined(path)
        return confined.relative_to(self._attempt_root).as_posix()

    def _resolve_session_relative(self, session: _EpisodeSession, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_absolute():
            raise _TraceValidationError(f"absolute trace artifact path: {relative}")
        resolved = (session.directory / candidate).resolve(strict=False)
        if not resolved.is_relative_to(session.directory):
            raise _TraceValidationError(f"trace artifact escapes session: {relative}")
        self._confined(resolved)
        return resolved

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        destination = self._confined(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._confined(destination.parent)
        temporary = self._confined(
            destination.parent / f".{destination.name}.{secrets.token_hex(8)}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _atomic_write_json(self, path: Path, payload: Any) -> bytes:
        encoded = _canonical_json_bytes(payload)
        self._atomic_write_bytes(path, encoded)
        return encoded

    def _require_recordable_session(self) -> _EpisodeSession:
        session = self._session
        if session is None:
            raise RuntimeError("begin_episode must be called before record_call")
        if session.receipt is not None or session.abandoned:
            raise RuntimeError("the current trace episode is already closed")
        if session.degraded_reasons:
            raise RuntimeError(
                "trace recording is degraded; subsequent calls are disabled"
            )
        return session

    def begin_episode(self, *, episode_seed: int, seeded_reset_generation: int) -> None:
        """Start a trace session without touching model or simulator RNG state."""

        if self._session is not None and self._session.receipt is None and not self._session.abandoned:
            self.abandon_episode(reason="superseded by a new seeded reset")
        trace_session_id = secrets.token_hex(16)
        directory = self._confined(self._worker_root / f"session-{trace_session_id}")
        directory.mkdir(parents=False, exist_ok=False)
        self._fsync_directory(self._worker_root)
        self._session = _EpisodeSession(
            trace_session_id=trace_session_id,
            directory=directory,
            episode_seed=int(episode_seed),
            seeded_reset_generation=int(seeded_reset_generation),
        )

    def _provenance(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "task": self.task,
            "checkpoint_sha256": self.checkpoint_sha256,
            "contract_hash": self.contract_hash,
            "source_inventory_digest": self.source_inventory_digest,
            "model_architecture": self.model_architecture,
            "action_horizon": self.action_horizon,
            "num_inference_steps": self.num_inference_steps,
            "mapping_version": self.mapping_version,
        }

    def record_call(
        self,
        *,
        http_response_sequence: int,
        reset_generation: int,
        front_rgb: np.ndarray,
        state64: np.ndarray,
        policy_trace: Any,
    ) -> None:
        """Atomically publish the NPZ and then its JSON metadata."""

        session = self._require_recordable_session()
        trace_call_index = session.next_trace_call_index

        arrays: dict[str, np.ndarray] = {}
        tensor_metadata: dict[str, dict[str, Any]] = {}

        def add_tensor(name: str, value: Any) -> None:
            array, metadata = _tensor_to_numpy(value)
            arrays[name] = array
            tensor_metadata[name] = metadata

        try:
            add_tensor("state_raw", policy_trace.state_raw)
            add_tensor("state", policy_trace.state)
            add_tensor("state_mask", policy_trace.state_mask)
            add_tensor("action", policy_trace.action_array)
            if arrays["state_raw"].shape != (1, _PROPRIO_DIM):
                raise ValueError(
                    f"state_raw shape must be (1, {_PROPRIO_DIM}), got {arrays['state_raw'].shape}"
                )
            if arrays["state"].shape != (1, _PROPRIO_DIM):
                raise ValueError(
                    f"state shape must be (1, {_PROPRIO_DIM}), got {arrays['state'].shape}"
                )
            expected_action_shape = (1, self.action_horizon, _ACTION_DIM)
            if arrays["action"].shape != expected_action_shape:
                raise ValueError(
                    f"action shape must be {expected_action_shape}, got {arrays['action'].shape}"
                )

            action_token_mask = getattr(policy_trace, "action_token_mask", None)
            if action_token_mask is not None:
                add_tensor("action_token_mask", action_token_mask)

            qwen_values: dict[str, Any] = {}
            qwen_unknown_values: dict[str, dict[str, str]] = {}
            qwen_inputs = getattr(policy_trace, "qwen_inputs", None)
            if qwen_inputs is not None and not isinstance(qwen_inputs, dict):
                reason = (
                    "qwen_inputs has unsupported type "
                    f"{_python_type_name(qwen_inputs)}"
                )
                raise TypeError(reason)
            elif qwen_inputs:
                for key, value in sorted(qwen_inputs.items(), key=lambda item: str(item[0])):
                    if not isinstance(key, str) or not _SAFE_QWEN_KEY_RE.fullmatch(key):
                        reason = f"qwen input key is not trace-safe: {key!r}"
                        raise ValueError(reason)
                    elif torch.is_tensor(value):
                        add_tensor(f"qwen.{key}", value)
                    elif _json_safe_scalar_or_list(value):
                        qwen_values[key] = value
                    else:
                        reason = (
                            f"qwen input {key!r} has unsupported non-tensor type "
                            f"{_python_type_name(value)}"
                        )
                        raise TypeError(reason)

            npz_name = f"call-{trace_call_index:06d}.npz"
            json_name = f"call-{trace_call_index:06d}.json"
            npz_path = self._confined(session.directory / npz_name)
            json_path = self._confined(session.directory / json_name)
            npz_payload = _npz_bytes(arrays)
            self._atomic_write_bytes(npz_path, npz_payload)

            prompt = str(policy_trace.prompt)
            qwen_text_raw = getattr(policy_trace, "qwen_text", None)
            qwen_text = None if qwen_text_raw is None else str(qwen_text_raw)
            call_payload = {
                "schema_version": TRACE_SCHEMA_VERSION,
                **self._provenance(),
                "trace_session_id": session.trace_session_id,
                "episode_seed": session.episode_seed,
                "seeded_reset_generation": session.seeded_reset_generation,
                "reset_generation": int(reset_generation),
                "trace_call_index": trace_call_index,
                "http_response_sequence": int(http_response_sequence),
                "model_inputs": {
                    "tensors": {
                        key: tensor_metadata[key]
                        for key in sorted(tensor_metadata)
                        if key not in {"action", "state_raw"}
                    },
                    "qwen_json_values": qwen_values,
                    "qwen_unknown_values": qwen_unknown_values,
                },
                "model_output": {
                    "tensors": {"action": tensor_metadata["action"]},
                    "latent_slice": [0, 64],
                    "hand_slice": [64, 66],
                },
                "preprocessing_provenance": {
                    "tensors": {"state_raw": tensor_metadata["state_raw"]},
                    "prompt": prompt,
                    "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    "qwen_text": qwen_text,
                    "qwen_text_sha256": (
                        None
                        if qwen_text is None
                        else _sha256_bytes(qwen_text.encode("utf-8"))
                    ),
                },
                "wire_provenance": {
                    "front_rgb": _wire_array_metadata(front_rgb),
                    "state64": _wire_array_metadata(state64),
                },
                "compatibility_only_excluded": ["input_ids", "image"],
                "npz_relative_path": npz_name,
                "npz_bytes": len(npz_payload),
                "npz_sha256": _sha256_bytes(npz_payload),
                "created_at": _utc_now(),
            }
            json_payload = self._atomic_write_json(json_path, call_payload)
            session.published_call_hashes[trace_call_index] = (
                _sha256_bytes(json_payload),
                call_payload["npz_sha256"],
            )
            session.last_http_response_sequence = int(http_response_sequence)
            session.next_trace_call_index += 1
        except Exception as exc:
            session.degraded_reasons.append(
                f"call {trace_call_index} recording failed: {type(exc).__name__}: {exc}"
            )
            raise

    def _validate_session(self, session: _EpisodeSession) -> tuple[list[dict[str, Any]], int]:
        expected_count = session.next_trace_call_index - 1
        expected_json_names = {f"call-{index:06d}.json" for index in range(1, expected_count + 1)}
        expected_npz_names = {name.removesuffix(".json") + ".npz" for name in expected_json_names}
        actual_json_names = {path.name for path in session.directory.glob("call-*.json")}
        actual_npz_names = {path.name for path in session.directory.glob("call-*.npz")}
        if actual_json_names != expected_json_names:
            raise _TraceValidationError(
                f"JSON call index gap/duplicate/partial: expected {sorted(expected_json_names)}, "
                f"found {sorted(actual_json_names)}"
            )
        if actual_npz_names != expected_npz_names:
            raise _TraceValidationError(
                f"NPZ call index gap/partial: expected {sorted(expected_npz_names)}, "
                f"found {sorted(actual_npz_names)}"
            )

        calls: list[dict[str, Any]] = []
        total_bytes = 0
        seen_indices: set[int] = set()
        for index in range(1, expected_count + 1):
            json_path = self._confined(session.directory / f"call-{index:06d}.json")
            raw_json = json_path.read_bytes()
            json_sha256 = _sha256_bytes(raw_json)
            published_hashes = session.published_call_hashes.get(index)
            if published_hashes is None:
                raise _TraceValidationError(f"missing published hashes for {json_path.name}")
            if json_sha256 != published_hashes[0]:
                raise _TraceValidationError(f"JSON hash mismatch for {json_path.name}")
            try:
                payload = json.loads(raw_json)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise _TraceValidationError(f"invalid call JSON {json_path.name}: {exc}") from exc
            if raw_json != _canonical_json_bytes(payload):
                raise _TraceValidationError(f"non-canonical or tampered JSON: {json_path.name}")
            embedded_index = payload.get("trace_call_index")
            if embedded_index != index or embedded_index in seen_indices:
                raise _TraceValidationError(
                    f"call index mismatch/duplicate in {json_path.name}: {embedded_index}"
                )
            seen_indices.add(embedded_index)
            if payload.get("schema_version") != TRACE_SCHEMA_VERSION:
                raise _TraceValidationError(f"schema mismatch in {json_path.name}")
            if payload.get("trace_session_id") != session.trace_session_id:
                raise _TraceValidationError(f"trace session mismatch in {json_path.name}")
            if payload.get("episode_seed") != session.episode_seed:
                raise _TraceValidationError(f"episode seed mismatch in {json_path.name}")
            if payload.get("seeded_reset_generation") != session.seeded_reset_generation:
                raise _TraceValidationError(f"seeded reset generation mismatch in {json_path.name}")
            for key, expected in self._provenance().items():
                if payload.get(key) != expected:
                    raise _TraceValidationError(f"provenance {key} mismatch in {json_path.name}")

            npz_relative = payload.get("npz_relative_path")
            if not isinstance(npz_relative, str):
                raise _TraceValidationError(f"missing NPZ relative path in {json_path.name}")
            npz_path = self._resolve_session_relative(session, npz_relative)
            expected_npz_path = self._confined(session.directory / f"call-{index:06d}.npz")
            if npz_path != expected_npz_path:
                raise _TraceValidationError(f"unexpected NPZ path in {json_path.name}")
            if not npz_path.is_file():
                raise _TraceValidationError(f"missing NPZ for {json_path.name}")
            npz_sha256, npz_size = _sha256_file(npz_path)
            if npz_sha256 != payload.get("npz_sha256") or npz_size != payload.get("npz_bytes"):
                raise _TraceValidationError(f"NPZ hash/size mismatch for {json_path.name}")
            if npz_sha256 != published_hashes[1]:
                raise _TraceValidationError(f"published NPZ hash mismatch for {json_path.name}")

            expected_tensor_metadata: dict[str, Any] = {}
            model_inputs = payload.get("model_inputs")
            model_output = payload.get("model_output")
            preprocessing = payload.get("preprocessing_provenance")
            if (
                not isinstance(model_inputs, dict)
                or not isinstance(model_output, dict)
                or not isinstance(preprocessing, dict)
            ):
                raise _TraceValidationError(f"missing model I/O metadata in {json_path.name}")
            input_tensors = model_inputs.get("tensors")
            output_tensors = model_output.get("tensors")
            preprocessing_tensors = preprocessing.get("tensors")
            if (
                not isinstance(input_tensors, dict)
                or not isinstance(output_tensors, dict)
                or not isinstance(preprocessing_tensors, dict)
            ):
                raise _TraceValidationError(f"invalid tensor metadata in {json_path.name}")
            expected_tensor_metadata.update(input_tensors)
            expected_tensor_metadata.update(output_tensors)
            expected_tensor_metadata.update(preprocessing_tensors)
            try:
                with np.load(npz_path, allow_pickle=False) as arrays:
                    if set(arrays.files) != set(expected_tensor_metadata):
                        raise _TraceValidationError(
                            f"NPZ tensor keys do not match metadata in {json_path.name}"
                        )
                    for name in arrays.files:
                        array = arrays[name]
                        metadata = expected_tensor_metadata[name]
                        if list(array.shape) != metadata.get("shape"):
                            raise _TraceValidationError(
                                f"tensor shape mismatch for {name} in {json_path.name}"
                            )
                        if str(array.dtype) != metadata.get("saved_dtype"):
                            raise _TraceValidationError(
                                f"tensor dtype mismatch for {name} in {json_path.name}"
                            )
            except _TraceValidationError:
                raise
            except Exception as exc:
                raise _TraceValidationError(f"unreadable NPZ {npz_path.name}: {exc}") from exc

            total_bytes += len(raw_json) + npz_size
            calls.append(
                {
                    "trace_call_index": index,
                    "http_response_sequence": payload.get("http_response_sequence"),
                    "reset_generation": payload.get("reset_generation"),
                    "json_relative_path": json_path.name,
                    "json_bytes": len(raw_json),
                    "json_sha256": json_sha256,
                    "npz_relative_path": npz_path.name,
                    "npz_bytes": npz_size,
                    "npz_sha256": npz_sha256,
                }
            )
        return calls, total_bytes

    def _manifest_base(
        self,
        session: _EpisodeSession,
        *,
        trace_status: str,
        result_identity: str | None,
        outcome_sha256: str | None,
        record_count: int,
        last_trace_call_index: int,
        calls: list[dict[str, Any]],
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            **self._provenance(),
            "trace_session_id": session.trace_session_id,
            "episode_seed": session.episode_seed,
            "seeded_reset_generation": session.seeded_reset_generation,
            "result_identity": result_identity,
            "outcome_sha256": outcome_sha256,
            "record_count": record_count,
            "last_trace_call_index": last_trace_call_index,
            "last_http_response_sequence": (
                calls[-1]["http_response_sequence"] if calls else 0
            ),
            "trace_status": trace_status,
            "degraded_reasons": list(session.degraded_reasons),
            "calls": calls,
            "error": error,
            "created_at": _utc_now(),
        }

    def _failure_receipt(
        self,
        session: _EpisodeSession,
        *,
        result_identity: str,
        outcome_sha256: str,
        error: str,
    ) -> TraceReceipt:
        manifest_path = self._confined(session.directory / "manifest.json")
        manifest_relative_path: str | None = None
        manifest_sha256: str | None = None
        total_bytes = 0
        payload = self._manifest_base(
            session,
            trace_status="degraded",
            result_identity=result_identity,
            outcome_sha256=outcome_sha256,
            record_count=0,
            last_trace_call_index=0,
            calls=[],
            error=error,
        )
        try:
            encoded = self._atomic_write_json(manifest_path, payload)
            manifest_relative_path = self._receipt_relative(manifest_path)
            manifest_sha256 = _sha256_bytes(encoded)
            total_bytes = len(encoded)
        except Exception as manifest_exc:
            error = (
                f"{error}; failed to publish degraded manifest: "
                f"{type(manifest_exc).__name__}: {manifest_exc}"
            )
        return TraceReceipt(
            schema_version=TRACE_RECEIPT_VERSION,
            run_id=self.run_id,
            worker_id=self.worker_id,
            trace_session_id=session.trace_session_id,
            result_identity=result_identity,
            episode_seed=session.episode_seed,
            seeded_reset_generation=session.seeded_reset_generation,
            outcome_sha256=outcome_sha256,
            record_count=0,
            last_trace_call_index=0,
            last_http_response_sequence=0,
            trace_status="degraded",
            manifest_relative_path=manifest_relative_path,
            manifest_sha256=manifest_sha256,
            success_marker_relative_path=None,
            total_bytes=total_bytes,
            error=error,
        )

    def finish_episode(
        self,
        *,
        result_identity: str,
        episode_seed: int,
        outcome_sha256: str,
    ) -> TraceReceipt:
        """Validate all calls, publish manifest, then publish ``_SUCCESS`` last."""

        session = self._session
        if session is None:
            raise RuntimeError("begin_episode must be called before finish_episode")
        finish_key = (str(result_identity), int(episode_seed), str(outcome_sha256))
        if session.receipt is not None:
            if finish_key == session.finished_key:
                return session.receipt
            raise TraceFinishConflictError(
                "finish_episode replay does not match the finalized episode"
            )
        if session.abandoned:
            raise RuntimeError("cannot finish an abandoned trace episode")

        result_identity = str(result_identity)
        if not result_identity:
            raise ValueError("result_identity must be a non-empty string")
        outcome_sha256 = str(outcome_sha256)
        if not _SHA256_HEX_RE.fullmatch(outcome_sha256):
            error = "outcome_sha256 must be 64 lowercase hexadecimal characters"
            receipt = self._failure_receipt(
                session,
                result_identity=result_identity,
                outcome_sha256=outcome_sha256,
                error=error,
            )
            session.finished_key = finish_key
            session.receipt = receipt
            return receipt
        if int(episode_seed) != session.episode_seed:
            error = (
                f"finish episode seed {episode_seed} does not match active seed "
                f"{session.episode_seed}"
            )
            receipt = self._failure_receipt(
                session,
                result_identity=result_identity,
                outcome_sha256=outcome_sha256,
                error=error,
            )
            session.finished_key = finish_key
            session.receipt = receipt
            return receipt

        try:
            calls, call_bytes = self._validate_session(session)
            record_count = len(calls)
            trace_status = "degraded" if session.degraded_reasons else "complete"
            degradation_error = (
                "; ".join(session.degraded_reasons) if session.degraded_reasons else None
            )
            manifest = self._manifest_base(
                session,
                trace_status=trace_status,
                result_identity=result_identity,
                outcome_sha256=outcome_sha256,
                record_count=record_count,
                last_trace_call_index=record_count,
                calls=calls,
                error=degradation_error,
            )
            manifest_path = self._confined(session.directory / "manifest.json")
            manifest_bytes = self._atomic_write_json(manifest_path, manifest)
            manifest_sha256 = _sha256_bytes(manifest_bytes)
            marker = {
                "schema_version": TRACE_RECEIPT_VERSION,
                "trace_session_id": session.trace_session_id,
                "manifest_relative_path": self._receipt_relative(manifest_path),
                "manifest_sha256": manifest_sha256,
            }
            marker_path = self._confined(session.directory / "_SUCCESS")
            if trace_status == "complete":
                marker_bytes = self._atomic_write_json(marker_path, marker)
                marker_relative_path = self._receipt_relative(marker_path)
            else:
                marker_bytes = b""
                marker_relative_path = None
            receipt = TraceReceipt(
                schema_version=TRACE_RECEIPT_VERSION,
                run_id=self.run_id,
                worker_id=self.worker_id,
                trace_session_id=session.trace_session_id,
                result_identity=result_identity,
                episode_seed=session.episode_seed,
                seeded_reset_generation=session.seeded_reset_generation,
                outcome_sha256=outcome_sha256,
                record_count=record_count,
                last_trace_call_index=record_count,
                last_http_response_sequence=session.last_http_response_sequence,
                trace_status=trace_status,
                manifest_relative_path=self._receipt_relative(manifest_path),
                manifest_sha256=manifest_sha256,
                success_marker_relative_path=marker_relative_path,
                total_bytes=call_bytes + len(manifest_bytes) + len(marker_bytes),
                error=degradation_error,
            )
        except Exception as exc:
            receipt = self._failure_receipt(
                session,
                result_identity=result_identity,
                outcome_sha256=outcome_sha256,
                error=f"{type(exc).__name__}: {exc}",
            )
        session.finished_key = finish_key
        session.receipt = receipt
        return receipt

    def abandon_episode(self, *, reason: str) -> None:
        """Mark an unfinished session incomplete without publishing ``_SUCCESS``."""

        session = self._session
        if session is None or session.receipt is not None or session.abandoned:
            return
        reason = str(reason)
        if not reason:
            raise ValueError("reason must be a non-empty string")
        session.degraded_reasons.append(reason)
        manifest = self._manifest_base(
            session,
            trace_status="incomplete",
            result_identity=None,
            outcome_sha256=None,
            record_count=0,
            last_trace_call_index=0,
            calls=[],
            error=reason,
        )
        self._atomic_write_json(session.directory / "manifest.json", manifest)
        success_marker = self._confined(session.directory / "_SUCCESS")
        if success_marker.exists():
            raise RuntimeError("incomplete trace session unexpectedly has a success marker")
        session.abandoned = True

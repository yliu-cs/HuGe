"""Checkpoint metadata, integrity, and atomic-publication helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from huge.data.schemas import ActionSchema, PROPRIO_DIM, PROPRIO_LAYOUT_VERSION


CHECKPOINT_MANIFEST_VERSION = 1
TRAINER_STATE_SCHEMA_VERSION = 1
PORTABLE_CHECKPOINT_KIND = "huge_portable_model"
RESUME_STATE_KIND = "huge_training_resume_state"
_PUBLICATION_FILES = frozenset({"manifest.json", "_SUCCESS"})
_RNG_STATE_RE = re.compile(r"^random_states_(\d+)\.pkl$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DEEPSPEED_OPTIM_STATE_RE = re.compile(
    r"^pytorch_model/(?:bf16_)?zero_pp_rank_(\d+)_mp_rank_(\d+)_optim_states\.pt$"
)
_DEEPSPEED_MODEL_STATE_RE = re.compile(
    r"^pytorch_model/(?:(?:zero_pp_rank_(\d+)_)?mp_rank_(\d+)_model_states\.pt)$"
)


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint publication is incomplete or corrupted."""


def build_checkpoint_metadata(**overrides: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        **ActionSchema().to_dict(),
        "proprio_layout": PROPRIO_LAYOUT_VERSION,
        "proprio_dim": PROPRIO_DIM,
        "image_transform": "gr00t_explicit_train_random_eval_center_256",
        "server_protocol": "huge_policy_server_msgpack_numpy",
    }
    meta.update({k: v for k, v in overrides.items() if v is not None})
    return meta


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: str | Path, payload: Any) -> str:
    """Atomically write JSON and return the SHA256 of the published bytes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_bytes(encoded)


def write_sidecar(path: str | Path, metadata: dict[str, Any]) -> None:
    write_json_atomic(path, metadata)


def _publication_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.as_posix() in _PUBLICATION_FILES:
            continue
        if path.is_symlink():
            raise CheckpointValidationError(f"checkpoint publication may not contain symlinks: {relative}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def publish_directory_manifest(
    root: str | Path,
    *,
    kind: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Hash a staging directory and publish its manifest and marker last."""
    directory = Path(root)
    if not directory.is_dir() or directory.is_symlink():
        raise CheckpointValidationError(f"checkpoint staging directory is invalid: {directory}")
    entries = []
    for path in _publication_files(directory):
        entries.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise CheckpointValidationError(f"checkpoint staging directory contains no files: {directory}")
    reserved = {"schema_version", "kind", "files"}
    overlap = reserved.intersection(metadata)
    if overlap:
        raise CheckpointValidationError(f"checkpoint metadata uses reserved keys: {sorted(overlap)}")
    manifest = {
        "schema_version": CHECKPOINT_MANIFEST_VERSION,
        "kind": kind,
        **metadata,
        "files": entries,
    }
    manifest_sha256 = write_json_atomic(directory / "manifest.json", manifest)
    write_json_atomic(
        directory / "_SUCCESS",
        {
            "schema_version": CHECKPOINT_MANIFEST_VERSION,
            "kind": kind,
            "manifest_sha256": manifest_sha256,
        },
    )
    return manifest


def _safe_manifest_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if (
        not relative
        or "\\" in relative
        or relative_path.is_absolute()
        or not relative_path.parts
        or relative_path.as_posix() != relative
        or ".." in relative_path.parts
        or relative_path.as_posix() in _PUBLICATION_FILES
    ):
        raise CheckpointValidationError(f"unsafe checkpoint manifest path: {relative!r}")
    resolved = root / relative_path
    if resolved.is_symlink():
        raise CheckpointValidationError(f"checkpoint file may not be a symlink: {relative!r}")
    return resolved


def _validate_rank_state_inventory(
    directory: Path,
    actual_paths: set[str],
    *,
    expected_world_size: int,
) -> None:
    if type(expected_world_size) is not int or expected_world_size <= 0:
        raise CheckpointValidationError(f"invalid expected checkpoint world size: {expected_world_size!r}")
    expected_ranks = set(range(expected_world_size))
    rng_ranks = {
        int(match.group(1))
        for relative in actual_paths
        if (match := _RNG_STATE_RE.fullmatch(relative)) is not None
    }
    if rng_ranks != expected_ranks:
        raise CheckpointValidationError(
            f"resume state rank RNG inventory mismatch: expected={sorted(expected_ranks)} actual={sorted(rng_ranks)}"
        )

    deepspeed_paths = {
        relative
        for relative in actual_paths
        if relative == "latest" or relative.startswith("pytorch_model/")
    }
    if not deepspeed_paths:
        return
    if "latest" not in deepspeed_paths:
        raise CheckpointValidationError("DeepSpeed resume state is missing its latest tag file")
    try:
        latest_tag = (directory / "latest").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CheckpointValidationError("DeepSpeed resume latest tag is unreadable") from exc
    if latest_tag != "pytorch_model":
        raise CheckpointValidationError(f"unexpected DeepSpeed resume tag: {latest_tag!r}")

    optimizer_entries: dict[int, list[str]] = {}
    for relative in actual_paths:
        match = _DEEPSPEED_OPTIM_STATE_RE.fullmatch(relative)
        if match is None:
            continue
        rank = int(match.group(1))
        if match.group(2) != "00":
            raise CheckpointValidationError(f"unsupported DeepSpeed model-parallel optimizer shard: {relative}")
        optimizer_entries.setdefault(rank, []).append(relative)
    optimizer_ranks = set(optimizer_entries)
    if optimizer_ranks != expected_ranks or any(len(paths) != 1 for paths in optimizer_entries.values()):
        raise CheckpointValidationError(
            "DeepSpeed optimizer shard inventory mismatch: "
            f"expected_ranks={sorted(expected_ranks)} actual="
            f"{ {rank: sorted(paths) for rank, paths in sorted(optimizer_entries.items())} }"
        )

    unpartitioned_model_states: list[str] = []
    partitioned_model_states: dict[int, list[str]] = {}
    for relative in actual_paths:
        match = _DEEPSPEED_MODEL_STATE_RE.fullmatch(relative)
        if match is None:
            continue
        if match.group(2) != "00":
            raise CheckpointValidationError(f"unsupported DeepSpeed model-parallel model shard: {relative}")
        if match.group(1) is None:
            unpartitioned_model_states.append(relative)
        else:
            partitioned_model_states.setdefault(int(match.group(1)), []).append(relative)
    if unpartitioned_model_states and partitioned_model_states:
        raise CheckpointValidationError("DeepSpeed resume state mixes partitioned and unpartitioned model shards")
    if unpartitioned_model_states:
        if unpartitioned_model_states != ["pytorch_model/mp_rank_00_model_states.pt"]:
            raise CheckpointValidationError(
                f"DeepSpeed model state inventory mismatch: {sorted(unpartitioned_model_states)}"
            )
    elif set(partitioned_model_states) != expected_ranks or any(
        len(paths) != 1 for paths in partitioned_model_states.values()
    ):
        raise CheckpointValidationError(
            "DeepSpeed partitioned model shard inventory mismatch: "
            f"expected_ranks={sorted(expected_ranks)} actual="
            f"{ {rank: sorted(paths) for rank, paths in sorted(partitioned_model_states.items())} }"
        )


def validate_published_directory(
    root: str | Path,
    *,
    expected_kind: str | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    """Validate a complete publication, including every file digest."""
    directory = Path(root)
    if not directory.is_dir() or directory.is_symlink():
        raise CheckpointValidationError(f"checkpoint directory is invalid: {directory}")
    marker_path = directory / "_SUCCESS"
    manifest_path = directory / "manifest.json"
    if (
        not marker_path.is_file()
        or marker_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise CheckpointValidationError(f"checkpoint is incomplete (missing manifest/_SUCCESS): {directory}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(f"checkpoint publication metadata is unreadable: {directory}") from exc
    if not isinstance(marker, dict) or not isinstance(manifest, dict):
        raise CheckpointValidationError(f"checkpoint publication metadata must be JSON objects: {directory}")
    if set(marker) != {"schema_version", "kind", "manifest_sha256"}:
        raise CheckpointValidationError(f"invalid checkpoint marker schema: {sorted(marker)}")
    if type(marker.get("schema_version")) is not int or marker.get("schema_version") != CHECKPOINT_MANIFEST_VERSION:
        raise CheckpointValidationError(f"unsupported checkpoint marker schema: {marker.get('schema_version')!r}")
    if type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != CHECKPOINT_MANIFEST_VERSION:
        raise CheckpointValidationError(f"unsupported checkpoint manifest schema: {manifest.get('schema_version')!r}")
    kind = manifest.get("kind")
    if not isinstance(kind, str) or not kind:
        raise CheckpointValidationError(f"checkpoint manifest has invalid kind: {kind!r}")
    marker_kind = marker.get("kind")
    if not isinstance(marker_kind, str) or marker_kind != kind or (expected_kind is not None and kind != expected_kind):
        raise CheckpointValidationError(
            f"checkpoint kind mismatch: marker={marker_kind!r} manifest={kind!r} expected={expected_kind!r}"
        )
    manifest_sha256 = marker.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise CheckpointValidationError(f"checkpoint marker has invalid manifest digest: {manifest_sha256!r}")
    if manifest_sha256 != sha256_file(manifest_path):
        raise CheckpointValidationError(f"checkpoint manifest digest mismatch: {manifest_path}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise CheckpointValidationError(f"checkpoint manifest has no file inventory: {manifest_path}")
    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise CheckpointValidationError(f"invalid checkpoint manifest entry: {entry!r}")
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in expected_paths:
            raise CheckpointValidationError(f"invalid or duplicate checkpoint manifest path: {relative!r}")
        size = entry.get("size")
        if type(size) is not int or size < 0:
            raise CheckpointValidationError(f"invalid checkpoint manifest size for {relative!r}: {size!r}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise CheckpointValidationError(f"invalid checkpoint manifest digest for {relative!r}: {digest!r}")
        expected_paths.add(relative)
        path = _safe_manifest_path(directory, relative)
        if not path.is_file():
            raise CheckpointValidationError(f"checkpoint file is missing: {path}")
        if path.stat().st_size != size:
            raise CheckpointValidationError(f"checkpoint file size mismatch: {path}")
        if sha256_file(path) != digest:
            raise CheckpointValidationError(f"checkpoint file digest mismatch: {path}")
    actual_paths = {path.relative_to(directory).as_posix() for path in _publication_files(directory)}
    if actual_paths != expected_paths:
        raise CheckpointValidationError(
            f"checkpoint file inventory mismatch: missing={sorted(expected_paths - actual_paths)} "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    if expected_world_size is not None:
        _validate_rank_state_inventory(
            directory,
            actual_paths,
            expected_world_size=expected_world_size,
        )
    return manifest


def atomic_publish_directory(staging: str | Path, destination: str | Path) -> None:
    """Publish a completed same-filesystem staging directory without overwrite."""
    source = Path(staging)
    target = Path(destination)
    if source.parent.resolve() != target.parent.resolve():
        raise CheckpointValidationError(f"staging and destination must share a parent: {source} -> {target}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint publication: {target}")
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise CheckpointValidationError(f"staging and destination are on different filesystems: {source} -> {target}")
    os.replace(source, target)
    _fsync_directory(target.parent)

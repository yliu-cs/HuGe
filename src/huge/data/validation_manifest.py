"""Persistent action/proprio alignment validation for immutable caches."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from huge.data.proprio_stats import read_npy_header
from huge.data.schemas import PROPRIO_DIM


FORMAT_VERSION = 1
PUBLICATION_NAME = "action_proprio_validation"


@dataclass(frozen=True, slots=True)
class ValidationSource:
    tag: str
    dataset_id: str
    kind: str
    cache_root: Path
    action_root: Path
    proprio_root: Path
    action_dim: int
    version_files: tuple[Path, ...]
    keep_records: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"data version file not found: {path}")
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _directory_contract(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"validated data directory not found: {path}")
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "mtime_ns": stat.st_mtime_ns}


def _validate_success_marker(path: Path) -> None:
    value = path.read_text(encoding="utf-8").strip()
    if value == "ok":
        return
    manifest = path.parent / "meta" / "conversion_manifest.json"
    if len(value) == 64 and manifest.is_file() and value == _sha256(manifest):
        return
    raise ValueError(f"invalid dataset publication marker: {path}")

def _data_version(source: ValidationSource) -> dict[str, Any]:
    for path in source.version_files:
        if path.name == "_SUCCESS":
            _validate_success_marker(path)
    return {
        "files": [_file_contract(path) for path in source.version_files],
        "directories": [
            _directory_contract(source.action_root),
            _directory_contract(source.proprio_root),
        ],
    }


def _stage1_cache_expected(index_path: Path) -> dict[str, int]:
    expected: dict[str, int] = {}
    with index_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            stable_id = str(row["stable_id"])
            length = int(row.get("target_frames", row.get("token_frame_count", 0)))
            relative = f"{stable_id}.npy"
            if not stable_id or relative in expected or length <= 0:
                raise ValueError(f"{index_path}:{line_number} has an invalid inventory row")
            expected[relative] = length
    return expected


def _egodex_expected(index_path: Path, action_root: Path) -> dict[str, int]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    expected: dict[str, int] = {}
    if "paths" in payload:
        rows = zip(payload["paths"], payload["token_rows"], strict=True)
        for episode_path, length_value in rows:
            relative = (Path(str(episode_path)).relative_to("episodes") / "token_state.npy").as_posix()
            length = int(length_value)
            if relative in expected or length <= 0:
                raise ValueError(f"invalid EgoDex inventory row: {episode_path!r}/{length}")
            expected[relative] = length
        return expected
    if "episodes" in payload:
        for row in payload["episodes"]:
            token_path = Path(row["token_path"])
            resolved = token_path if token_path.is_absolute() else action_root.parent / token_path
            relative = resolved.relative_to(action_root).as_posix()
            length = int(row["length"])
            if relative in expected or length <= 0:
                raise ValueError(f"invalid EgoDex inventory row: {relative!r}/{length}")
            expected[relative] = length
        return expected
    raise ValueError(f"unsupported EgoDex index format: {index_path}")


def _inventory_digest(rows: list[tuple[str, str, int]]) -> str:
    digest = hashlib.sha256()
    for action_relative, proprio_relative, length in rows:
        digest.update(f"{action_relative}\t{proprio_relative}\t{length}\n".encode())
    return digest.hexdigest()


def _audit_expected(source: ValidationSource, expected: dict[str, int]) -> dict[str, Any]:
    action_files = {
        path.relative_to(source.action_root).as_posix()
        for path in source.action_root.rglob("*.npy")
        if path.is_file()
    }
    proprio_files = {
        path.relative_to(source.proprio_root).as_posix()
        for path in source.proprio_root.rglob("*.npy")
        if path.is_file()
    }
    expected_files = set(expected)
    missing = (expected_files - action_files) | (expected_files - proprio_files)
    extra = (action_files - expected_files) | (proprio_files - expected_files)
    if missing or extra:
        raise ValueError(
            f"{source.tag} action/proprio inventory mismatch: missing={len(missing)} "
            f"extra={len(extra)} examples={sorted(missing | extra)[:8]}"
        )

    def probe(relative: str) -> tuple[str, str, int]:
        action_shape, action_dtype = read_npy_header(source.action_root / relative)
        proprio_shape, proprio_dtype = read_npy_header(source.proprio_root / relative)
        length = expected[relative]
        if action_shape != (length, source.action_dim) or action_dtype != np.dtype(np.float32):
            raise ValueError(
                f"{source.tag}: invalid action {relative}: {action_shape}/{action_dtype}; "
                f"expected ({length}, {source.action_dim})/float32"
            )
        if proprio_shape != (length, PROPRIO_DIM) or proprio_dtype != np.dtype(np.float32):
            raise ValueError(
                f"{source.tag}: invalid proprio {relative}: {proprio_shape}/{proprio_dtype}; "
                f"expected ({length}, {PROPRIO_DIM})/float32"
            )
        return relative, relative, length

    with ThreadPoolExecutor(max_workers=32) as executor:
        rows = list(executor.map(probe, sorted(expected_files)))
    return _source_result(source, rows)


def _audit_sonic(source: ValidationSource) -> dict[str, Any]:
    action_files = {
        path.relative_to(source.action_root).as_posix()
        for path in source.action_root.rglob("*.npy")
        if path.is_file()
    }
    proprio_files = {
        path.relative_to(source.proprio_root).as_posix()
        for path in source.proprio_root.rglob("*.npy")
        if path.is_file()
    }
    if action_files != proprio_files:
        mismatch = action_files ^ proprio_files
        raise ValueError(
            f"{source.tag} action/proprio inventory mismatch: "
            f"action_only={len(action_files - proprio_files)} "
            f"proprio_only={len(proprio_files - action_files)} examples={sorted(mismatch)[:8]}"
        )
    if not action_files:
        raise ValueError(f"{source.tag}: no action/proprio files found")

    def probe(relative: str) -> tuple[str, str, int]:
        action_shape, action_dtype = read_npy_header(source.action_root / relative)
        proprio_shape, proprio_dtype = read_npy_header(source.proprio_root / relative)
        action_ok = (
            len(action_shape) == 2
            and action_shape[0] > 0
            and action_shape[1] == source.action_dim
            and action_dtype == np.dtype(np.float32)
        )
        proprio_ok = (
            len(proprio_shape) == 2
            and proprio_shape[0] > 0
            and proprio_shape[1] == PROPRIO_DIM
            and proprio_dtype == np.dtype(np.float32)
        )
        if not action_ok:
            raise ValueError(f"{source.tag}: invalid action {relative}: {action_shape}/{action_dtype}")
        if not proprio_ok:
            raise ValueError(f"{source.tag}: invalid proprio {relative}: {proprio_shape}/{proprio_dtype}")
        if action_shape[0] != proprio_shape[0]:
            raise ValueError(
                f"{source.tag}: T mismatch {relative}: action={action_shape[0]} proprio={proprio_shape[0]}"
            )
        return relative, relative, int(action_shape[0])

    with ThreadPoolExecutor(max_workers=32) as executor:
        rows = list(executor.map(probe, sorted(action_files)))
    return _source_result(source, rows)


def _source_result(source: ValidationSource, rows: list[tuple[str, str, int]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset_id": source.dataset_id,
        "kind": source.kind,
        "cache_root": str(source.cache_root.resolve()),
        "action_root": str(source.action_root.resolve()),
        "proprio_root": str(source.proprio_root.resolve()),
        "action_dim": source.action_dim,
        "proprio_dim": PROPRIO_DIM,
        "n_files": len(rows),
        "n_frames": sum(row[2] for row in rows),
        "inventory_sha256": _inventory_digest(rows),
        "missing": 0,
        "extra": 0,
        "header_mismatch": 0,
        "length_mismatch": 0,
    }
    if source.keep_records:
        result["records"] = [[action, proprio, length] for action, proprio, length in rows]
    return result


def audit_source(source: ValidationSource) -> dict[str, Any]:
    if source.kind == "egodex":
        expected = _egodex_expected(source.version_files[0], source.action_root)
        return _audit_expected(source, expected)
    if source.kind == "stage1_cache":
        index_path = next(path for path in source.version_files if path.name == "episode_index.jsonl")
        return _audit_expected(source, _stage1_cache_expected(index_path))
    if source.kind == "sonic":
        return _audit_sonic(source)
    raise ValueError(f"unsupported validation source kind: {source.kind}")


def build_validation_sources(
    dataset_root: str | Path,
    *,
    stage1_roots: Mapping[str, Path] | None = None,
    stage2_roots: Mapping[str, Path] | None = None,
    humanoid_arena_root: Path | None = None,
) -> dict[str, ValidationSource]:
    data = Path(dataset_root)
    s1 = {
        "egodex": data / "EgoDex",
        "egoverse": data / "egoverse-retargeted-cache",
        "behavior1k": data / "behavior-1k-2025-challenge-cache",
        "xperience": data / "xperience-10m-cache",
        **(stage1_roots or {}),
    }
    s2 = {
        "he": data / "humanoid-everyday",
        "wbt": data / "G1_WBT",
        "hiw": data / "HIW-500",
        **(stage2_roots or {}),
    }
    egodex = s1["egodex"]
    egodex_cache = egodex.with_name(f"{egodex.name}-cache")

    def stage1_cache(tag: str, dataset_id: str, root: Path) -> ValidationSource:
        return ValidationSource(
            tag=tag,
            dataset_id=dataset_id,
            kind="stage1_cache",
            cache_root=root,
            action_root=root / "sonic_tokens",
            proprio_root=root / "g1_proprio",
            action_dim=66,
            version_files=(
                root / "_SUCCESS",
                root / "meta" / "conversion_manifest.json",
                root / "meta" / "episode_index.jsonl",
            ),
        )

    he_cache = s2["he"].with_name(f"{s2['he'].name}-cache")
    hiw_cache = s2["hiw"].with_name(f"{s2['hiw'].name}-cache")
    arena = humanoid_arena_root or data / "HumanoidArena"
    return {
        "EGODEX": ValidationSource(
            tag="EGODEX",
            dataset_id="egodex",
            kind="egodex",
            cache_root=egodex,
            action_root=egodex / "episodes",
            proprio_root=egodex_cache / "g1_proprio",
            action_dim=64,
            version_files=(egodex / "meta" / "egodex_index.json",),
        ),
        "EGOVERSE": stage1_cache("EGOVERSE", "egoverse", s1["egoverse"]),
        "BEHAVIOR1K": stage1_cache("BEHAVIOR1K", "behavior1k", s1["behavior1k"]),
        "XPERIENCE": stage1_cache("XPERIENCE", "xperience", s1["xperience"]),
        "HE": ValidationSource(
            tag="HE",
            dataset_id="humanoid_everyday",
            kind="sonic",
            cache_root=he_cache,
            action_root=he_cache / "sonic_tokens",
            proprio_root=he_cache / "g1_proprio",
            action_dim=66,
            version_files=(
                he_cache / "meta" / "dataset_index_50hz_g1.npz",
                he_cache / "meta" / "episode_index_50hz_g1.npz",
            ),
            keep_records=True,
        ),
        "WBT": ValidationSource(
            tag="WBT",
            dataset_id="g1_wbt",
            kind="sonic",
            cache_root=s2["wbt"],
            action_root=s2["wbt"] / "sonic_tokens",
            proprio_root=s2["wbt"] / "g1_proprio",
            action_dim=66,
            version_files=(
                s2["wbt"] / "meta" / "episodes_meta.parquet",
            ),
            keep_records=True,
        ),
        "HIW": ValidationSource(
            tag="HIW",
            dataset_id="hiw",
            kind="sonic",
            cache_root=s2["hiw"],
            action_root=s2["hiw"] / "sonic_tokens",
            proprio_root=s2["hiw"] / "g1_proprio",
            action_dim=66,
            version_files=(
                hiw_cache / "_SUCCESS",
                hiw_cache / "meta" / "conversion_manifest.json",
                s2["hiw"] / "meta" / "info.json",
                s2["hiw"] / "meta" / "tasks.parquet",
            ),
            keep_records=True,
        ),
        "HUMANOID_ARENA": ValidationSource(
            tag="HUMANOID_ARENA",
            dataset_id="humanoid_arena",
            kind="sonic",
            cache_root=arena,
            action_root=arena / "sonic_tokens",
            proprio_root=arena / "g1_proprio",
            action_dim=66,
            version_files=(
                arena / "_SUCCESS",
                arena / "meta" / "conversion_manifest.json",
            ),
            keep_records=True,
        ),
    }


def _manifest_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _publication_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root) / "meta" / PUBLICATION_NAME


def load_validation_manifest(path_or_dataset_root: str | Path) -> dict[str, Any]:
    path = Path(path_or_dataset_root)
    publication = path if path.name == PUBLICATION_NAME else _publication_dir(path)
    manifest_path = publication / "manifest.json"
    success_path = publication / "_SUCCESS"
    expected = success_path.read_text(encoding="utf-8").strip()
    actual = _sha256(manifest_path)
    if expected != actual:
        raise ValueError(f"validation manifest _SUCCESS mismatch: {publication}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION or not isinstance(payload.get("sources"), dict):
        raise ValueError(f"unsupported action/proprio validation manifest: {manifest_path}")
    return payload


def _publish(publication: Path, payload: dict[str, Any]) -> None:
    publication.mkdir(parents=True, exist_ok=True)
    manifest_path = publication / "manifest.json"
    success_path = publication / "_SUCCESS"
    manifest_tmp = publication / f"manifest.json.tmp.{os.getpid()}"
    success_tmp = publication / f"_SUCCESS.tmp.{os.getpid()}"
    try:
        manifest_tmp.write_bytes(_manifest_bytes(payload))
        success_tmp.write_text(f"{_sha256(manifest_tmp)}\n", encoding="utf-8")
        os.replace(manifest_tmp, manifest_path)
        os.replace(success_tmp, success_path)
    finally:
        manifest_tmp.unlink(missing_ok=True)
        success_tmp.unlink(missing_ok=True)


def ensure_validation_manifest(
    dataset_root: str | Path,
    *,
    sources: Mapping[str, ValidationSource],
    stage_scope: str,
    stats_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    data = Path(dataset_root)
    publication = _publication_dir(data)
    publication.parent.mkdir(parents=True, exist_ok=True)
    lock_path = publication.parent / f".{PUBLICATION_NAME}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            payload = load_validation_manifest(publication)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            payload = {
                "format_version": FORMAT_VERSION,
                "dataset_root": str(data.resolve()),
                "sources": {},
                "stage_scopes": {},
            }
        if payload.get("dataset_root") != str(data.resolve()):
            payload = {
                "format_version": FORMAT_VERSION,
                "dataset_root": str(data.resolve()),
                "sources": {},
                "stage_scopes": {},
            }

        stats_file = _file_contract(Path(stats_path))
        stats_contract = {
            "size": stats_file["size"],
            "sha256": stats_file["sha256"],
            "source_tags": list(sources),
        }
        stored_scope = payload["stage_scopes"].get(stage_scope)
        scope_missing = stored_scope is None
        scope_changed = stored_scope is not None and stored_scope != stats_contract
        audited: list[str] = []
        cached: list[str] = []
        current_versions = {tag: _data_version(source) for tag, source in sources.items()}
        for tag, source in sources.items():
            current_version = current_versions[tag]
            existing = payload["sources"].get(tag)
            stale = force or scope_changed or not isinstance(existing, dict)
            if isinstance(existing, dict):
                stale = stale or existing.get("data_version") != current_version
            else:
                stale = True
            if not stale:
                cached.append(tag)
                continue
            before = current_version
            print(f"[data-audit] scanning {tag} action/proprio inventory", flush=True)
            entry = audit_source(source)
            print(
                f"[data-audit] passed {tag}: files={entry['n_files']} frames={entry['n_frames']}",
                flush=True,
            )
            after = _data_version(source)
            if before != after:
                raise RuntimeError(f"{tag} data version changed during full action/proprio audit")
            entry["data_version"] = after
            entry["validated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            payload["sources"][tag] = entry
            audited.append(tag)

        payload["stage_scopes"][stage_scope] = stats_contract
        if audited or scope_missing or scope_changed or not (publication / "_SUCCESS").is_file():
            payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            _publish(publication, payload)
        return {
            "publication": str(publication.resolve()),
            "audited_tags": audited,
            "cached_tags": cached,
        }

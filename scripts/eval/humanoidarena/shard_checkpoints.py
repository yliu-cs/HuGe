#!/usr/bin/env python3
"""Select pending HumanoidArena checkpoints for a stable multi-host shard."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


STEP_RE = re.compile(r"^step_([0-9]+)$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SUMMARY_SCHEMA_VERSION = "huge_humanoidarena_summary_v1"
ATTEMPT_SCHEMA_VERSION = "huge_humanoidarena_attempt_v1"
EXPECTED_SEEDS = [0, 1, 2]
EXPECTED_REPEATS_PER_SEED = 20
EXPECTED_EPISODES = len(EXPECTED_SEEDS) * EXPECTED_REPEATS_PER_SEED
EXPECTED_MAX_STEPS = 1450
EXPECTED_TEST_MODE = "base_test"
EXPECTED_CHECKPOINT_TASK = "HOI_pp_box"
EXPECTED_ISAAC_TASK = "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby"
EXPECTED_INSTRUCTION = "pick up the box and place it"
EPISODE_SEED_FORMULA = "sha256(task_name|group_seed|repeat_idx)_positive_int32"
MODEL_NOISE_SEED_ALGORITHM = "huge_ha_model_noise_v1"
MODEL_IO_TRACE_SCHEMA_VERSION = "huge_humanoidarena_model_io_trace_v1"
MODEL_IO_RECEIPT_SCHEMA_VERSION = "huge_humanoidarena_model_io_receipt_v1"
MODEL_IO_TRACE_POLICY = "require_complete_v1"
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
STATIC_PLAN_SCHEMA_VERSION = "huge_humanoidarena_static_plan_v1"
CONTRACT_SCHEMA_VERSION = "huge_humanoidarena_contract_v1"
EPISODE_SCHEMA_VERSION = "huge_humanoidarena_episode_v1"
SELECTION_SCHEMA_VERSION = "huge_humanoidarena_shard_selection_v1"
CHECKPOINT_MANIFEST_VERSION = 1
PORTABLE_CHECKPOINT_KIND = "huge_portable_model"
_PUBLICATION_FILES = frozenset({"manifest.json", "_SUCCESS"})
EXPECTED_RUN_MATRIX = (
    ("qwenpi_v3", (6000, 12000, 18000, 24000, 30000)),
    ("qwengr00t", (6000, 12000, 18000, 24000, 30000)),
    ("qwenoft", (6000, 12000, 18000, 24000, 30000)),
)
EXPECTED_CHECKPOINT_COUNT = sum(
    len(steps) for _architecture, steps in EXPECTED_RUN_MATRIX
)
ALLOWED_TRAINING_ROOT_NAMES = (
    "stage3_humanoidarena_full_optimized",
)
_WORKER_RE = re.compile(r"^w[0-9]{2}$")
_ATTEMPT_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[a-f0-9]{8}$")
_LEGAL_TASK_OUTCOMES = {"success", "fall", "timeout", "max_steps", "sim_stopped"}
_EXECUTION_FAILED = "execution_failed"
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "proprio_stats_sonic.json",
    "proprio_norm_config.json",
)
EXPECTED_HUGE_SOURCE_PATHS = (
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
TERMINAL_MARKER_STATES = {
    "_SUCCESS": "success",
    "_COMPLETED_WITH_FAILURES": "completed_with_failures",
}
_FILE_HASH_CACHE: dict[Path, tuple[tuple[int, int, int, int], str]] = {}


class ShardConfigError(RuntimeError):
    """Raised when checkpoint discovery or shard arguments are unsafe."""


@dataclass(frozen=True)
class ShardTarget:
    ordinal: int
    path: Path
    relative_path: str
    step: int


@dataclass(frozen=True)
class ContractCandidate:
    attempt_name: str
    contract_path: Path
    contract_hash: str
    attempt_path: Path
    state: str

    @property
    def sort_key(self) -> tuple[str, str, str]:
        return (self.attempt_name, self.contract_hash, str(self.contract_path))


@dataclass(frozen=True)
class CheckpointContractStatus:
    state: str
    contract_path: str | None = None
    contract_hash: str | None = None
    attempt_path: str | None = None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_owned_attempt_path(
    attempt: Path,
    path: Path,
    *,
    directory: bool,
) -> bool:
    """Return whether path is a real, owned child of a possibly symlinked attempt."""
    try:
        attempt_root = attempt.resolve(strict=True)
        relative = path.relative_to(attempt)
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            return False
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
                return False
        return _is_relative_to(path.resolve(strict=True), attempt_root)
    except (OSError, RuntimeError, ValueError):
        return False


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected)
        )
    return actual == expected


def _sha256_file(path: Path) -> str:
    resolved = path.resolve()
    info = resolved.stat()
    signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    cached = _FILE_HASH_CACHE.get(resolved)
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    _FILE_HASH_CACHE[resolved] = (signature, value)
    return value


def _validate_portable_checkpoint_publication(
    checkpoint: Path,
    *,
    architecture: str,
    step: int,
    run_id: str,
) -> dict[str, Any]:
    marker_path = checkpoint / "_SUCCESS"
    manifest_path = checkpoint / "manifest.json"
    for path in (marker_path, manifest_path):
        try:
            info = path.lstat()
        except OSError as exc:
            raise ShardConfigError(
                f"portable checkpoint publication is incomplete: "
                f"{checkpoint}: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_size <= 0
        ):
            raise ShardConfigError(
                f"portable checkpoint publication metadata is unsafe: {path}"
            )
    marker = _json_object(marker_path)
    manifest = _json_object(manifest_path)
    if marker is None or manifest is None:
        raise ShardConfigError(
            f"portable checkpoint publication metadata is invalid: {checkpoint}"
        )
    if (
        set(marker)
        != {"schema_version", "kind", "manifest_sha256"}
        or marker.get("schema_version") != CHECKPOINT_MANIFEST_VERSION
        or marker.get("kind") != PORTABLE_CHECKPOINT_KIND
        or marker.get("manifest_sha256") != _sha256_file(manifest_path)
        or manifest.get("schema_version") != CHECKPOINT_MANIFEST_VERSION
        or manifest.get("kind") != PORTABLE_CHECKPOINT_KIND
    ):
        raise ShardConfigError(
            f"portable checkpoint marker/manifest contract mismatch: "
            f"{checkpoint}"
        )

    metadata = manifest.get("metadata")
    publication = metadata if isinstance(metadata, dict) else manifest
    if (
        publication.get("global_step") != step
        or publication.get("model_architecture") != architecture
        or publication.get("tuning_mode") != "full"
        or publication.get("run_id") != run_id
    ):
        raise ShardConfigError(
            f"portable checkpoint manifest identity mismatch: {checkpoint}"
        )

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ShardConfigError(
            f"portable checkpoint manifest has no file inventory: {manifest_path}"
        )
    expected_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "size",
            "sha256",
        }:
            raise ShardConfigError(
                f"invalid portable checkpoint manifest entry: {entry!r}"
            )
        relative_raw = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        if (
            not isinstance(relative_raw, str)
            or not relative_raw
            or relative_raw in expected_paths
            or relative_raw in _PUBLICATION_FILES
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ShardConfigError(
                f"invalid portable checkpoint manifest entry: {entry!r}"
            )
        relative = Path(relative_raw)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_raw
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ShardConfigError(
                f"unsafe portable checkpoint manifest path: {relative_raw!r}"
            )
        file_path = checkpoint / relative
        try:
            info = file_path.lstat()
            resolved = file_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ShardConfigError(
                f"portable checkpoint manifest file is unavailable: "
                f"{file_path}: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or not _is_relative_to(resolved, checkpoint)
            or info.st_size != size
            or _sha256_file(file_path) != digest
        ):
            raise ShardConfigError(
                f"portable checkpoint manifest digest/size mismatch: "
                f"{file_path}"
            )
        expected_paths.add(relative_raw)

    actual_paths: set[str] = set()
    for entry in checkpoint.rglob("*"):
        relative = entry.relative_to(checkpoint).as_posix()
        if relative in _PUBLICATION_FILES:
            continue
        try:
            info = entry.lstat()
        except OSError as exc:
            raise ShardConfigError(
                f"cannot inspect portable checkpoint inventory: {entry}: {exc}"
            ) from exc
        if stat.S_ISLNK(info.st_mode):
            raise ShardConfigError(
                f"portable checkpoint may not contain symlinks: {entry}"
            )
        if stat.S_ISREG(info.st_mode):
            actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise ShardConfigError(
            f"portable checkpoint file inventory mismatch: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}, "
            f"checkpoint={checkpoint}"
        )
    if not set(REQUIRED_CHECKPOINT_FILES).issubset(actual_paths):
        raise ShardConfigError(
            f"portable checkpoint manifest lacks required evaluator files: "
            f"{checkpoint}"
        )
    return {
        "schema_version": CHECKPOINT_MANIFEST_VERSION,
        "kind": PORTABLE_CHECKPOINT_KIND,
        "manifest_relative_path": "manifest.json",
        "manifest_sha256": _sha256_file(manifest_path),
        "success_marker_relative_path": "_SUCCESS",
        "success_marker_sha256": _sha256_file(marker_path),
        "file_count": len(entries),
    }


def _directory_inventory(path: Path) -> dict[str, str] | None:
    if not path.is_dir():
        return None
    inventory: dict[str, str] = {}
    try:
        for entry in sorted(path.rglob("*")):
            if entry.is_symlink():
                if not _is_relative_to(entry.resolve(), path.resolve()):
                    return None
                inventory[entry.relative_to(path).as_posix()] = (
                    f"symlink:{os.readlink(entry)}"
                )
            elif entry.is_file():
                inventory[entry.relative_to(path).as_posix()] = _sha256_file(entry)
    except OSError:
        return None
    return inventory or None


def _matches_file_hash_mapping(
    mapping: Any,
    root: Path,
    *,
    expected_names: set[str] | None = None,
) -> bool:
    if not isinstance(mapping, dict) or not mapping:
        return False
    if expected_names is not None and set(mapping) != expected_names:
        return False
    resolved_root = root.resolve()
    for relative, expected_digest in mapping.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        ):
            return False
        path = (resolved_root / relative).resolve()
        if not path.is_file() or not _is_relative_to(path, resolved_root):
            return False
        try:
            if _sha256_file(path) != expected_digest:
                return False
        except OSError:
            return False
    return True


def validate_shard(*, index: int, count: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ShardConfigError("shard index must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ShardConfigError("shard count must be a positive integer")
    if index >= count:
        raise ShardConfigError(f"shard index must be in [0, {count - 1}]")


def _resolve_training_run_root(
    repo: Path,
    raw_root: Path,
    architecture: str,
) -> tuple[Path, Path]:
    lexical = raw_root.expanduser()
    if not lexical.is_absolute():
        lexical = repo / lexical
    try:
        entry_stat = lexical.lstat()
        root = lexical.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ShardConfigError(
            f"{architecture} run root is unavailable: {lexical}: {exc}"
        ) from exc
    if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISDIR(entry_stat.st_mode):
        raise ShardConfigError(
            f"{architecture} run root must be a real directory: {lexical}"
        )

    matched_base: Path | None = None
    relative: Path | None = None
    for base_name in ALLOWED_TRAINING_ROOT_NAMES:
        base = (repo / ".runs" / base_name).resolve()
        if _is_relative_to(root, base):
            matched_base = base
            relative = root.relative_to(base)
            break
    if matched_base is None or relative is None:
        allowed_text = ", ".join(
            str((repo / ".runs" / name).resolve())
            for name in ALLOWED_TRAINING_ROOT_NAMES
        )
        raise ShardConfigError(
            f"{architecture} run root must be inside one allowed training root: "
            f"{allowed_text}: {root}"
        )

    expected_prefix = ("HOI_pp_box", architecture, "full", "stage3")
    if (
        len(relative.parts) != 5
        or tuple(relative.parts[:4]) != expected_prefix
        or not _SAFE_COMPONENT_RE.fullmatch(relative.parts[4])
        or relative.parts[4] in {".", "..", "_resume_states"}
    ):
        raise ShardConfigError(
            f"{architecture} root must name one exact "
            f"HOI_pp_box/{architecture}/full/stage3/<run_id> run: {root}"
        )
    cursor = matched_base
    for component in relative.parts:
        cursor = cursor / component
        info = cursor.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ShardConfigError(
                f"{architecture} run path contains a non-directory or symlink: {cursor}"
            )
    return root, matched_base


def _active_training_writer(repo: Path, root: Path) -> int | None:
    root_tokens = {
        str(root).encode("utf-8"),
        root.relative_to(repo).as_posix().encode("utf-8"),
    }
    training_tokens = (
        b"train_huge.sh",
        b"run_huge.sh",
        b"train_stage",
        b"accelerate",
        b"deepspeed",
        b"torchrun",
    )
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except OSError:
            continue
        argv = tuple(token for token in raw.split(b"\0") if token)
        if (
            argv
            and any(root_token in argument for root_token in root_tokens for argument in argv)
            and any(token in raw for token in training_tokens)
        ):
            return int(proc.name)
    return None


def _validate_training_completion_barrier(
    repo: Path,
    root: Path,
    training_base: Path,
    architecture: str,
    expected_steps: Sequence[int],
) -> list[tuple[Path, str, int]]:
    trainer_lock = root / ".trainer.lock"
    if os.path.lexists(trainer_lock):
        raise ShardConfigError(
            f"training writer lock is still present for {architecture}: "
            f"{trainer_lock}"
        )
    writer_pid = _active_training_writer(repo, root)
    if writer_pid is not None:
        raise ShardConfigError(
            f"active training writer pid={writer_pid} still references {root}"
        )
    active_locks = sorted(training_base.glob("run_huge*.flag"))
    if active_locks:
        raise ShardConfigError(
            f"training matrix lock is still present for {architecture}: "
            + ", ".join(str(path) for path in active_locks)
        )
    for entry in root.rglob("*"):
        lowered = entry.name.lower()
        if "staging" in lowered or lowered in {"running", "_running", "writer.lock"}:
            raise ShardConfigError(
                f"training staging/writer artifact remains under {root}: {entry}"
            )

    observed: dict[int, Path] = {}
    for lexical in root.iterdir():
        match = STEP_RE.fullmatch(lexical.name)
        if match is None:
            continue
        try:
            info = lexical.lstat()
            resolved = lexical.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ShardConfigError(
                f"cannot inspect portable checkpoint {lexical}: {exc}"
            ) from exc
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or not _is_relative_to(resolved, root)
        ):
            raise ShardConfigError(
                f"portable checkpoint must be a real child directory: {lexical}"
            )
        step = int(match.group(1))
        if step in observed:
            raise ShardConfigError(
                f"duplicate portable checkpoint step {step} under {root}"
            )
        observed[step] = resolved

    expected = set(expected_steps)
    actual = set(observed)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ShardConfigError(
            f"{architecture} checkpoint barrier expected exactly "
            f"{list(expected_steps)}; missing={missing}, extra={extra}, root={root}"
        )

    local: list[tuple[Path, str, int]] = []
    for step in expected_steps:
        checkpoint = observed[step]
        _validate_portable_checkpoint_publication(
            checkpoint,
            architecture=architecture,
            step=step,
            run_id=root.name,
        )
        for name in REQUIRED_CHECKPOINT_FILES:
            path = checkpoint / name
            try:
                info = path.lstat()
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ShardConfigError(
                    f"incomplete checkpoint {checkpoint}: missing {name}: {exc}"
                ) from exc
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_size <= 0
                or not _is_relative_to(resolved, checkpoint)
            ):
                raise ShardConfigError(
                    f"checkpoint file must be a non-empty real file: {path}"
                )

        payload = _json_object(checkpoint / "config.json")
        config = payload.get("config") if isinstance(payload, dict) else None
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        if (
            not isinstance(config, dict)
            or not isinstance(metadata, dict)
            or config.get("model_architecture") != architecture
            or metadata.get("model_architecture") != architecture
            or metadata.get("global_step") != step
            or config.get("freeze_qwen") is not False
            or config.get("qwen_lora") is not False
            or config.get("train_qwen_lora") is not False
            or metadata.get("tuning_mode", "full") != "full"
            or metadata.get("run_id", root.name) != root.name
        ):
            raise ShardConfigError(
                f"checkpoint config does not match {architecture}/full step {step}: "
                f"{checkpoint}"
            )
        local.append((checkpoint, checkpoint.relative_to(repo).as_posix(), step))
    return local


def discover_targets(repo_root: Path, checkpoint_roots: Sequence[Path]) -> list[ShardTarget]:
    repo = repo_root.expanduser().resolve()
    if len(checkpoint_roots) != len(EXPECTED_RUN_MATRIX):
        raise ShardConfigError(
            "exactly three run-specific roots are required in "
            "qwenpi_v3, qwengr00t, qwenoft order"
        )
    ordered: list[tuple[Path, str, int]] = []
    seen_roots: set[Path] = set()
    seen_checkpoints: set[Path] = set()
    for raw_root, (architecture, expected_steps) in zip(
        checkpoint_roots, EXPECTED_RUN_MATRIX
    ):
        root, training_base = _resolve_training_run_root(
            repo, raw_root, architecture
        )
        if root in seen_roots:
            raise ShardConfigError(f"duplicate run-specific root: {root}")
        seen_roots.add(root)
        local = _validate_training_completion_barrier(
            repo, root, training_base, architecture, expected_steps
        )
        for checkpoint, _relative, _step in local:
            if checkpoint in seen_checkpoints:
                raise ShardConfigError(
                    f"duplicate portable checkpoint directory: {checkpoint}"
                )
            seen_checkpoints.add(checkpoint)
        ordered.extend(local)

    expected_total = EXPECTED_CHECKPOINT_COUNT
    if len(ordered) != expected_total:
        raise ShardConfigError(
            f"checkpoint barrier expected {expected_total} targets, got {len(ordered)}"
        )
    return [
        ShardTarget(ordinal, path, relative, step)
        for ordinal, (path, relative, step) in enumerate(ordered)
    ]

def _derive_episode_seed(group_seed: int, repeat_idx: int) -> int:
    payload = f"{EXPECTED_ISAAC_TASK}|{group_seed}|{repeat_idx}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[-4:], "big") & 0x7FFFFFFF


def _matches_episode_matrix(episodes: Any, *, yaml_sha256: str | None = None) -> bool:
    if not isinstance(episodes, list) or len(episodes) != EXPECTED_EPISODES:
        return False
    expected_pairs = {
        (seed, repeat)
        for seed in EXPECTED_SEEDS
        for repeat in range(EXPECTED_REPEATS_PER_SEED)
    }
    actual_pairs: set[tuple[int, int]] = set()
    episode_indices: set[int] = set()
    detailed_fields = {
        "task_cli",
        "isaac_task_name",
        "test_mode",
        "episode_seed",
        "instruction",
        "env_config_yaml_sha256",
        "result_identity",
    }
    has_detailed_episode = any(
        isinstance(episode, dict) and detailed_fields.intersection(episode)
        for episode in episodes
    )
    for episode in episodes:
        if not isinstance(episode, dict):
            return False
        seed = episode.get("seed")
        repeat = episode.get("repeat_idx")
        episode_index = episode.get("episode_index")
        if (
            type(seed) is not int
            or type(repeat) is not int
            or type(episode_index) is not int
            or seed not in EXPECTED_SEEDS
            or not 0 <= repeat < EXPECTED_REPEATS_PER_SEED
        ):
            return False
        actual_pairs.add((seed, repeat))
        episode_indices.add(episode_index)
        if has_detailed_episode:
            if set(episode) != detailed_fields | {
                "seed",
                "repeat_idx",
                "episode_index",
            }:
                return False
            expected_index = EXPECTED_SEEDS.index(seed) * EXPECTED_REPEATS_PER_SEED + repeat
            expected_yaml = yaml_sha256 or episode.get("env_config_yaml_sha256")
            identity = {
                "task_cli": "pp_box",
                "isaac_task_name": EXPECTED_ISAAC_TASK,
                "test_mode": EXPECTED_TEST_MODE,
                "seed": seed,
                "repeat_idx": repeat,
                "episode_seed": _derive_episode_seed(seed, repeat),
                "episode_index": expected_index,
                "instruction": EXPECTED_INSTRUCTION,
                "env_config_yaml_sha256": expected_yaml,
            }
            if (
                episode_index != expected_index
                or expected_yaml is None
                or re.fullmatch(r"[0-9a-f]{64}", str(expected_yaml)) is None
                or any(episode.get(key) != value for key, value in identity.items())
                or episode.get("result_identity") != _canonical_sha256(identity)
            ):
                return False
    return (
        actual_pairs == expected_pairs
        and episode_indices == set(range(EXPECTED_EPISODES))
    )


def _matches_current_checkpoint_and_yaml(
    contract: Mapping[str, Any],
    relative_path: str,
    repo_root: Path,
) -> bool:
    static_plan = contract.get("static_plan")
    if not isinstance(static_plan, dict):
        return False
    checkpoint_payload = static_plan.get("checkpoint")
    yaml_payload = static_plan.get("yaml")
    if not isinstance(checkpoint_payload, dict) or not isinstance(yaml_payload, dict):
        return False
    repo = repo_root.expanduser().resolve()
    checkpoint = (repo / relative_path).resolve()
    allowed_checkpoint_roots = tuple(
        (repo / ".runs" / name).resolve()
        for name in ALLOWED_TRAINING_ROOT_NAMES
    )
    if (
        not checkpoint.is_dir()
        or not any(
            _is_relative_to(checkpoint, allowed_root)
            for allowed_root in allowed_checkpoint_roots
        )
        or checkpoint_payload.get("relative_path") != relative_path
    ):
        return False
    try:
        current_files = {
            name: _sha256_file(checkpoint / name)
            for name in REQUIRED_CHECKPOINT_FILES
            if (checkpoint / name).is_file()
            and _is_relative_to((checkpoint / name).resolve(), checkpoint)
        }
        current_publication = _validate_portable_checkpoint_publication(
            checkpoint,
            architecture=checkpoint.parts[-5],
            step=int(checkpoint.name.removeprefix("step_")),
            run_id=checkpoint.parent.name,
        )
    except (OSError, ShardConfigError, ValueError):
        return False
    if (
        set(current_files) != set(REQUIRED_CHECKPOINT_FILES)
        or checkpoint_payload.get("files") != current_files
        or checkpoint_payload.get("checkpoint_sha256")
        != _canonical_sha256(current_files)
        or checkpoint_payload.get("portable_publication")
        != current_publication
    ):
        return False
    config_payload = _json_object(checkpoint / "config.json")
    config = config_payload.get("config") if isinstance(config_payload, dict) else None
    metadata = config_payload.get("metadata") if isinstance(config_payload, dict) else None
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        return False
    expected_architecture = checkpoint.parts[-5]
    current_architecture = str(
        config.get("model_architecture", "qwenpi_v3") or "qwenpi_v3"
    )
    current_action_horizon = config.get("action_horizon", 50)
    current_num_inference_steps = config.get("num_inference_steps", 4)
    if (
        checkpoint_payload.get("model_architecture")
        != current_architecture
        or checkpoint_payload.get("model_architecture") != expected_architecture
        or checkpoint_payload.get("action_horizon")
        != current_action_horizon
        or checkpoint_payload.get("action_horizon") != 50
        or checkpoint_payload.get("num_inference_steps")
        != current_num_inference_steps
        or type(current_num_inference_steps) is not int
        or current_num_inference_steps <= 0
        or type(checkpoint_payload.get("num_inference_steps")) is not int
        or checkpoint_payload["num_inference_steps"] <= 0
    ):
        return False
    qwen_raw = config.get("base_vlm_path") or metadata.get("model_name_or_path")
    if not isinstance(qwen_raw, str) or not qwen_raw or "://" in qwen_raw:
        return False
    qwen_path = Path(qwen_raw).expanduser()
    qwen_path = (
        (repo / qwen_path).resolve() if not qwen_path.is_absolute() else qwen_path.resolve()
    )
    if (
        not qwen_path.is_dir()
        or not _is_relative_to(qwen_path, repo)
        or checkpoint_payload.get("qwen_path")
        != qwen_path.relative_to(repo).as_posix()
    ):
        return False
    yaml_path = (
        repo
        / "HumanoidArena/isaaclab_twist2_g1/tasks/common_test_config/"
        "base_test/pp_box_sonic_test.yaml"
    ).resolve()
    if not yaml_path.is_file():
        return False
    try:
        yaml_digest = _sha256_file(yaml_path)
    except OSError:
        return False
    return (
        yaml_payload.get("path") == str(yaml_path)
        and yaml_payload.get("sha256") == yaml_digest
        and _matches_episode_matrix(static_plan.get("episodes"), yaml_sha256=yaml_digest)
    )


def _matches_current_source_inventory(
    contract: Mapping[str, Any],
    repo_root: Path,
) -> bool:
    static_plan = contract.get("static_plan")
    source = static_plan.get("source_inventory") if isinstance(static_plan, dict) else None
    if not isinstance(source, dict) or set(source) != {
        "files",
        "source_inventory_digest",
    }:
        return False
    files = source.get("files")
    if (
        not isinstance(files, dict)
        or set(files)
        != {
            "upstream_compatibility",
            "wrapper_semantics",
            "wrapper_semantics_digest",
            "huge_sources",
            "task_yaml",
                "checkpoint_files",
                "checkpoint_publication",
                "sonic_assets",
            "qwen_assets",
            "qwen_assets_digest",
        }
        or source.get("source_inventory_digest") != _canonical_sha256(files)
    ):
        return False
    repo = repo_root.expanduser().resolve()
    humanoidarena = (repo / "HumanoidArena").resolve()
    isaaclab = (humanoidarena / "isaaclab_twist2_g1").resolve()
    if not humanoidarena.is_dir() or not isaaclab.is_dir():
        return False
    if not _matches_file_hash_mapping(
        files.get("huge_sources"),
        repo,
        expected_names=set(EXPECTED_HUGE_SOURCE_PATHS),
    ):
        return False
    if not _matches_file_hash_mapping(
        files.get("upstream_compatibility"), humanoidarena
    ) or not _matches_file_hash_mapping(files.get("wrapper_semantics"), isaaclab):
        return False
    yaml_payload = static_plan.get("yaml")
    checkpoint_payload = static_plan.get("checkpoint")
    if not isinstance(yaml_payload, dict) or not isinstance(checkpoint_payload, dict):
        return False
    if files.get("task_yaml") != {
        yaml_payload.get("path"): yaml_payload.get("sha256")
    } or files.get("checkpoint_files") != checkpoint_payload.get("files"):
        return False
    if files.get("checkpoint_publication") != checkpoint_payload.get(
        "portable_publication"
    ):
        return False
    qwen_raw = checkpoint_payload.get("qwen_path")
    if not isinstance(qwen_raw, str):
        return False
    qwen_path = (repo / qwen_raw).resolve()
    if not _is_relative_to(qwen_path, repo):
        return False
    qwen_inventory = _directory_inventory(qwen_path)
    if (
        qwen_inventory is None
        or files.get("qwen_assets") != qwen_inventory
        or files.get("qwen_assets_digest") != _canonical_sha256(qwen_inventory)
    ):
        return False
    sonic_paths = {
        "huge_encoder": repo / "resource/model/sonic/model_encoder.onnx",
        "huge_decoder": repo / "resource/model/sonic/model_decoder.onnx",
        "humanoidarena_encoder": (
            humanoidarena
            / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
        ),
        "humanoidarena_decoder": (
            humanoidarena
            / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"
        ),
    }
    try:
        current_sonic = {
            name: _sha256_file(path)
            for name, path in sonic_paths.items()
            if path.is_file()
        }
    except OSError:
        return False
    if set(current_sonic) != set(sonic_paths) or files.get("sonic_assets") != current_sonic:
        return False
    wrapper_digest = files.get("wrapper_semantics_digest")
    return (
        isinstance(wrapper_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", wrapper_digest) is not None
    )


def _requested_gpu_topology() -> tuple[tuple[int, ...], int, float] | None:
    raw_count = os.environ.get("HUGE_EVAL_GPU_COUNT", "")
    raw_ids = os.environ.get("HUGE_EVAL_GPU_LOGICAL_IDS", "")
    if raw_count and raw_ids:
        return None
    try:
        if raw_count:
            count = int(raw_count)
            if count < 1:
                return None
            ordinals = tuple(range(count))
        elif raw_ids:
            parsed = tuple(
                int(token.strip())
                for token in raw_ids.split(",")
                if token.strip()
            )
            if (
                not parsed
                or len(set(parsed)) != len(parsed)
                or any(ordinal < 0 for ordinal in parsed)
            ):
                return None
            ordinals = parsed
        else:
            ordinals = tuple(range(8))
        workers_per_gpu = int(
            os.environ.get("HUGE_EVAL_WORKERS_PER_GPU", "1")
        )
        min_free_gib = float(
            os.environ.get("HUGE_EVAL_MIN_FREE_GPU_MEMORY_GIB", "0")
        )
    except ValueError:
        return None
    if (
        workers_per_gpu not in {1, 2}
        or not math.isfinite(min_free_gib)
        or min_free_gib < 0.0
        or (workers_per_gpu == 2 and min_free_gib <= 0.0)
    ):
        return None
    return ordinals, workers_per_gpu, min_free_gib


def _matches_requested_gpu_bindings(
    bindings: Any,
    ordinals: Sequence[int],
    workers_per_gpu: int,
) -> bool:
    if (
        not isinstance(bindings, list)
        or len(bindings) != len(ordinals) * workers_per_gpu
    ):
        return False
    expected_gpu_keys = {
        "parent_logical_ordinal",
        "child_cuda_visible_token",
        "cuda_uuid",
        "pci_bus_id",
        "device_name",
        "total_memory_bytes",
        "compute_capability",
        "free_memory_bytes",
    }
    per_parent: dict[int, list[tuple[int, int, dict[str, Any]]]] = {}
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) != {
            "gpu",
            "renderer_ordinal",
            "worker_slot",
        }:
            return False
        gpu = binding.get("gpu")
        renderer = binding.get("renderer_ordinal")
        worker_slot = binding.get("worker_slot")
        if (
            not isinstance(gpu, dict)
            or set(gpu) != expected_gpu_keys
            or type(renderer) is not int
            or renderer < 0
            or type(worker_slot) is not int
            or not 0 <= worker_slot < workers_per_gpu
        ):
            return False
        ordinal = gpu.get("parent_logical_ordinal")
        uuid = gpu.get("cuda_uuid")
        if (
            type(ordinal) is not int
            or ordinal not in ordinals
            or not isinstance(uuid, str)
            or re.fullmatch(r"GPU-[A-Za-z0-9-]+", uuid, re.IGNORECASE)
            is None
            or gpu.get("child_cuda_visible_token") != uuid
            or not isinstance(gpu.get("pci_bus_id"), str)
            or not gpu["pci_bus_id"]
            or not isinstance(gpu.get("device_name"), str)
            or not gpu["device_name"]
            or type(gpu.get("total_memory_bytes")) is not int
            or gpu["total_memory_bytes"] <= 0
            or not isinstance(gpu.get("compute_capability"), str)
            or re.fullmatch(r"[0-9]+[.][0-9]+", gpu["compute_capability"])
            is None
            or gpu.get("free_memory_bytes") is not None
        ):
            return False
        per_parent.setdefault(ordinal, []).append(
            (worker_slot, renderer, gpu)
        )

    if set(per_parent) != set(ordinals):
        return False
    physical_uuids: set[str] = set()
    physical_pcis: set[str] = set()
    physical_renderers: set[int] = set()
    for ordinal in ordinals:
        rows = per_parent[ordinal]
        if len(rows) != workers_per_gpu:
            return False
        first_gpu = rows[0][2]
        renderer_values = {renderer for _slot, renderer, _gpu in rows}
        if (
            {slot for slot, _renderer, _gpu in rows}
            != set(range(workers_per_gpu))
            or len(renderer_values) != 1
            or any(
                not _strict_json_equal(gpu, first_gpu)
                for _slot, _renderer, gpu in rows[1:]
            )
        ):
            return False
        uuid = str(first_gpu["cuda_uuid"]).upper()
        pci = str(first_gpu["pci_bus_id"]).upper()
        renderer = next(iter(renderer_values))
        if (
            uuid in physical_uuids
            or pci in physical_pcis
            or renderer in physical_renderers
        ):
            return False
        physical_uuids.add(uuid)
        physical_pcis.add(pci)
        physical_renderers.add(renderer)
    return (
        len(physical_uuids)
        == len(physical_pcis)
        == len(physical_renderers)
        == len(ordinals)
    )


def _current_host_fingerprint() -> str:
    machine_id = ""
    try:
        machine_id = Path("/etc/machine-id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        pass
    return _canonical_sha256(
        {
            "hostname": platform.node(),
            "machine_id": machine_id,
            "machine": platform.machine(),
        }
    )


def _current_root_software(repo_root: Path) -> dict[str, Any] | None:
    repo = repo_root.expanduser().resolve()
    try:
        if Path(sys.prefix).resolve(strict=True) != (repo / ".venv").resolve():
            return None
    except (OSError, RuntimeError):
        return None
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


def _contract_execution_host_is_current(
    contract: Mapping[str, Any],
) -> bool:
    static_plan = contract.get("static_plan")
    policy = (
        static_plan.get("policy")
        if isinstance(static_plan, dict)
        else None
    )
    kit_path_raw = (
        policy.get("kit_gpu_map_file")
        if isinstance(policy, dict)
        else None
    )
    if not isinstance(kit_path_raw, str):
        return False
    kit_payload = _json_object(Path(kit_path_raw).expanduser().resolve())
    return (
        isinstance(kit_payload, dict)
        and kit_payload.get("host_fingerprint")
        == _current_host_fingerprint()
    )


def _matches_current_executable_contract(
    contract: Mapping[str, Any],
    repo_root: Path,
) -> bool:
    if set(contract) != {
        "schema_version",
        "static_plan",
        "static_plan_hash",
        "gpu_bindings",
        "kit_gpu_map_sha256",
    }:
        return False
    static_plan = contract.get("static_plan")
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA_VERSION
        or not isinstance(static_plan, dict)
        or set(static_plan)
        != {
            "schema_version",
            "task",
            "checkpoint",
            "yaml",
            "policy",
            "protocol",
            "episode_seed_algorithm",
            "model_noise_seed_algorithm",
            "episodes",
            "source_inventory",
            "upstream_contract",
            "software",
        }
        or static_plan.get("schema_version") != STATIC_PLAN_SCHEMA_VERSION
        or contract.get("static_plan_hash") != _canonical_sha256(static_plan)
    ):
        return False
    expected_task = {
        "cli_task": "pp_box",
        "checkpoint_task": EXPECTED_CHECKPOINT_TASK,
        "isaac_task": EXPECTED_ISAAC_TASK,
        "yaml_stem": "pp_box",
        "instruction": EXPECTED_INSTRUCTION,
        "default_max_steps": EXPECTED_MAX_STEPS,
        "default_record_video_every_n": 1,
        "memory_limit_percent": None,
    }
    if static_plan.get("task") != expected_task:
        return False
    checkpoint = static_plan.get("checkpoint")
    yaml_config = static_plan.get("yaml")
    policy = static_plan.get("policy")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint)
        != {
            "relative_path",
            "step",
            "checkpoint_task",
            "checkpoint_sha256",
            "files",
            "qwen_path",
            "model_architecture",
            "action_horizon",
            "num_inference_steps",
            "portable_publication",
        }
        or not isinstance(yaml_config, dict)
        or set(yaml_config)
        != {
            "path",
            "sha256",
            "seeds",
            "repeats_per_seed",
            "source_persistent_sim",
            "episode_seed_formula",
            "vision_randomization",
            "selected_seeds",
            "selected_repeats_per_seed",
        }
        or yaml_config.get("seeds") != EXPECTED_SEEDS
        or yaml_config.get("repeats_per_seed") != EXPECTED_REPEATS_PER_SEED
        or yaml_config.get("vision_randomization") is not None
        or not isinstance(policy, dict)
    ):
        return False
    required_policy = {
        "test_mode": EXPECTED_TEST_MODE,
        "effective_persistent_sim": False,
        "simulator_process_scope": "per_episode_v1",
        "model_io_schema": "huge_humanoidarena_model_io_trace_v1",
        "model_io_trace_policy": MODEL_IO_TRACE_POLICY,
        "max_steps": EXPECTED_MAX_STEPS,
        "post_termination_record_steps": 10,
        "record_video_every_n": 1,
        "episode_max_retries": 1,
        "headless": True,
        "enable_depth": False,
        "robot_type": "unitree_g1_refpose_v3_1",
        "root_rot6d_layout": "row",
        "root_max_delta_deg": 26.0,
        "heading_align": True,
        "output_delay_steps": 0,
        "memory_limit_percent": None,
        "open_door_latch": None,
    }
    if any(policy.get(key) != value for key, value in required_policy.items()):
        return False
    requested_topology = _requested_gpu_topology()
    if requested_topology is None:
        return False
    requested_ordinals, requested_workers_per_gpu, requested_min_free_gib = (
        requested_topology
    )
    try:
        current_policy = {
            "video_fps": int(os.environ.get("VIDEO_FPS", "30")),
            "step_log_every_n": int(
                os.environ.get("HUGE_EVAL_STEP_LOG_EVERY_N", "100")
            ),
            "server_timeout": float(
                os.environ.get("HUGE_EVAL_SERVER_TIMEOUT", "360.0")
            ),
            "shadow_startup_timeout_sec": float(
                os.environ.get("HUGE_EVAL_SHADOW_STARTUP_TIMEOUT_SEC", "180")
            ),
            "episode_timeout_sec": float(
                os.environ.get("HUGE_EVAL_EPISODE_TIMEOUT_SEC", "900")
            ),
            "retry_backoff_sec": float(
                os.environ.get("HUGE_EVAL_RETRY_BACKOFF_SEC", "5")
            ),
            "workers_per_gpu": requested_workers_per_gpu,
            "max_concurrent_shadow_startups": int(
                os.environ.get("HUGE_EVAL_MAX_CONCURRENT_SHADOW_STARTUPS", "2")
            ),
            "min_free_gpu_memory_gib": requested_min_free_gib,
        }
    except ValueError:
        return False
    if current_policy["max_concurrent_shadow_startups"] < 1:
        return False
    if any(policy.get(key) != value for key, value in current_policy.items()):
        return False
    if (
        static_plan.get("protocol")
        != {
            "http": "huge_humanoidarena_http_v1",
            "mapping": "humanoidarena_rotlocal_v31_to_huge_proprio_v1",
            "action": [50, 66],
            "latent": [50, 64],
        }
        or static_plan.get("episode_seed_algorithm") != EPISODE_SEED_FORMULA
        or static_plan.get("model_noise_seed_algorithm")
        != MODEL_NOISE_SEED_ALGORITHM
    ):
        return False
    software = static_plan.get("software")
    if software != _current_root_software(repo_root):
        return False
    repo = repo_root.expanduser().resolve()
    upstream_path = (
        repo
        / "HumanoidArena/isaaclab_twist2_g1/script/eval_scripts/sonic/sim_eval_vla.py"
    ).resolve()
    upstream = static_plan.get("upstream_contract")
    try:
        upstream_digest = _sha256_file(upstream_path)
    except OSError:
        return False
    if (
        not isinstance(upstream, dict)
        or upstream.get("status") != "compatible"
        or upstream.get("path") != str(upstream_path)
        or upstream.get("sha256") != upstream_digest
        or not isinstance(upstream.get("required_signatures"), dict)
        or not isinstance(upstream.get("forbidden_helpers"), list)
    ):
        return False
    kit_digest = contract.get("kit_gpu_map_sha256")
    kit_path_raw = policy.get("kit_gpu_map_file")
    if (
        not isinstance(kit_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", kit_digest) is None
        or not isinstance(kit_path_raw, str)
    ):
        return False
    kit_path = Path(kit_path_raw).expanduser().resolve()
    if (
        not kit_path.is_file()
        or not _is_relative_to(kit_path, (repo / ".runs").resolve())
    ):
        return False
    try:
        if _sha256_file(kit_path) != kit_digest:
            return False
    except OSError:
        return False
    return _matches_requested_gpu_bindings(
        contract.get("gpu_bindings"),
        requested_ordinals,
        requested_workers_per_gpu,
    )


def _matches_full_pp_box_contract(
    contract: Mapping[str, Any],
    relative_path: str,
    *,
    repo_root: Path | None = None,
) -> bool:
    static_plan = contract.get("static_plan")
    if not isinstance(static_plan, dict):
        return False
    episodes = static_plan.get("episodes")
    detailed_contract = (
        isinstance(episodes, list)
        and bool(episodes)
        and isinstance(episodes[0], dict)
        and "result_identity" in episodes[0]
    )
    contract_schema = contract.get("schema_version")
    static_schema = static_plan.get("schema_version")
    static_plan_hash = contract.get("static_plan_hash")
    if (
        (
            detailed_contract
            and (
                contract_schema != CONTRACT_SCHEMA_VERSION
                or static_schema != STATIC_PLAN_SCHEMA_VERSION
                or static_plan_hash != _canonical_sha256(static_plan)
            )
        )
        or
        (contract_schema is not None and contract_schema != CONTRACT_SCHEMA_VERSION)
        or (static_schema is not None and static_schema != STATIC_PLAN_SCHEMA_VERSION)
        or (
            static_plan_hash is not None
            and static_plan_hash != _canonical_sha256(static_plan)
        )
    ):
        return False
    task = static_plan.get("task")
    checkpoint = static_plan.get("checkpoint")
    yaml_config = static_plan.get("yaml")
    policy = static_plan.get("policy")
    yaml_sha256 = yaml_config.get("sha256") if isinstance(yaml_config, dict) else None
    checkpoint_files = checkpoint.get("files") if isinstance(checkpoint, dict) else None
    checkpoint_digest = (
        checkpoint.get("checkpoint_sha256") if isinstance(checkpoint, dict) else None
    )
    detailed_checkpoint = checkpoint_digest is not None or checkpoint_files is not None
    checkpoint_details_match = True
    if detailed_checkpoint:
        checkpoint_details_match = (
            isinstance(checkpoint_files, dict)
            and bool(checkpoint_files)
            and all(
                isinstance(name, str)
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                for name, digest in checkpoint_files.items()
            )
            and checkpoint_digest == _canonical_sha256(checkpoint_files)
            and checkpoint.get("checkpoint_task") == EXPECTED_CHECKPOINT_TASK
            and checkpoint.get("step") == int(Path(relative_path).name.removeprefix("step_"))
        )
    detailed_yaml = isinstance(yaml_config, dict) and (
        "sha256" in yaml_config or "path" in yaml_config
    )
    yaml_details_match = True
    if detailed_yaml:
        yaml_path = yaml_config.get("path")
        yaml_details_match = (
            isinstance(yaml_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", yaml_sha256) is not None
            and isinstance(yaml_path, str)
            and Path(yaml_path).as_posix().endswith(
                "/tasks/common_test_config/base_test/pp_box_sonic_test.yaml"
            )
            and yaml_config.get("source_persistent_sim") is True
            and yaml_config.get("episode_seed_formula") == EPISODE_SEED_FORMULA
        )
    return (
        isinstance(task, dict)
        and task.get("cli_task") == "pp_box"
        and task.get("checkpoint_task", EXPECTED_CHECKPOINT_TASK)
        == EXPECTED_CHECKPOINT_TASK
        and task.get("isaac_task", EXPECTED_ISAAC_TASK) == EXPECTED_ISAAC_TASK
        and task.get("instruction", EXPECTED_INSTRUCTION) == EXPECTED_INSTRUCTION
        and isinstance(checkpoint, dict)
        and checkpoint.get("relative_path") == relative_path
        and checkpoint_details_match
        and _matches_episode_matrix(
            episodes,
            yaml_sha256=yaml_sha256 if isinstance(yaml_sha256, str) else None,
        )
        and isinstance(yaml_config, dict)
        and yaml_config.get("selected_seeds") == EXPECTED_SEEDS
        and yaml_config.get("selected_repeats_per_seed") == EXPECTED_REPEATS_PER_SEED
        and yaml_details_match
        and isinstance(policy, dict)
        and policy.get("test_mode", EXPECTED_TEST_MODE) == EXPECTED_TEST_MODE
        and policy.get("max_steps") == EXPECTED_MAX_STEPS
        and policy.get("effective_persistent_sim") is False
        and (
            repo_root is None
            or (
                _matches_current_executable_contract(contract, repo_root)
                and _matches_current_checkpoint_and_yaml(contract, relative_path, repo_root)
                and _matches_current_source_inventory(contract, repo_root)
            )
        )
    )


def _matches_summary_totals(summary: Mapping[str, Any]) -> bool:
    success_rate = summary.get("success_rate")
    by_seed = summary.get("by_seed")
    if not isinstance(success_rate, dict) or not isinstance(by_seed, dict):
        return False
    values = (
        success_rate.get("numerator"),
        success_rate.get("denominator"),
        summary.get("task_success"),
        summary.get("task_failure"),
    )
    if any(type(value) is not int for value in values):
        return False
    numerator, denominator, task_success, task_failure = values
    if (
        denominator != EXPECTED_EPISODES
        or not 0 <= numerator <= denominator
        or task_success != numerator
        or task_failure != EXPECTED_EPISODES - numerator
        or set(by_seed) != {str(seed) for seed in EXPECTED_SEEDS}
    ):
        return False
    seed_successes = 0
    for seed in EXPECTED_SEEDS:
        counts = by_seed.get(str(seed))
        if not isinstance(counts, dict):
            return False
        completed = counts.get("completed")
        successes = counts.get("success")
        if (
            type(completed) is not int
            or completed != EXPECTED_REPEATS_PER_SEED
            or type(successes) is not int
            or successes < 0
            or successes > completed
        ):
            return False
        seed_successes += successes
    return seed_successes == numerator


def _valid_model_io_calls(
    attempt: Path,
    session_directory: Path,
    manifest: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    trace_session_id: str,
    episode_seed: int,
    seeded_generation: int,
    record_count: int,
) -> int | None:
    calls = manifest.get("calls")
    if not isinstance(calls, list) or len(calls) != record_count:
        return None
    if not _is_owned_attempt_path(attempt, session_directory, directory=True):
        return None
    expected_names = {"manifest.json", "_SUCCESS"}
    expected_names.update(
        f"call-{index:06d}.{suffix}"
        for index in range(1, record_count + 1)
        for suffix in ("json", "npz")
    )
    try:
        if {entry.name for entry in session_directory.iterdir()} != expected_names:
            return None
    except OSError:
        return None
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
            return None
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
            return None
        previous_http_sequence = http_sequence
        json_path = session_directory / expected_json_name
        npz_path = session_directory / expected_npz_name
        if (
            not _is_owned_attempt_path(attempt, json_path, directory=False)
            or not _is_owned_attempt_path(attempt, npz_path, directory=False)
        ):
            return None
        try:
            raw_json = json_path.read_bytes()
            call_payload = json.loads(raw_json)
            if (
                json_path.stat().st_size != json_size
                or npz_path.stat().st_size != npz_size
                or _sha256_file(json_path) != json_digest
                or _sha256_file(npz_path) != npz_digest
            ):
                return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(call_payload, dict)
            or set(call_payload) != call_payload_keys
            or raw_json != _canonical_json_bytes(call_payload) + b"\n"
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
            return None
        total_call_bytes += json_size + npz_size
    expected_last_sequence = previous_http_sequence if calls else 0
    if manifest.get("last_http_response_sequence") != expected_last_sequence:
        return None
    return total_call_bytes


def _valid_model_io_artifact(
    attempt: Path,
    contract: Mapping[str, Any],
    job: Mapping[str, Any],
    payload: Mapping[str, Any],
    worker_id: str,
    artifact: Any,
) -> bool:
    if not isinstance(artifact, dict) or set(artifact) != {
        "receipt",
        "manifest",
        "success_marker",
    }:
        return False
    receipt = artifact.get("receipt")
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
    outcome_sha256 = _canonical_sha256(payload)
    if (
        not isinstance(receipt, dict)
        or set(receipt) != receipt_keys
        or receipt.get("schema_version") != MODEL_IO_RECEIPT_SCHEMA_VERSION
        or receipt.get("run_id") != attempt.name
        or receipt.get("worker_id") != worker_id
        or receipt.get("result_identity") != job.get("result_identity")
        or receipt.get("episode_seed") != job.get("episode_seed")
        or receipt.get("outcome_sha256") != outcome_sha256
        or receipt.get("trace_status") != "complete"
        or receipt.get("error") is not None
    ):
        return False
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
        return False
    manifest_relative = (
        f"model_io/{worker_id}/session-{trace_session_id}/manifest.json"
    )
    marker_relative = f"model_io/{worker_id}/session-{trace_session_id}/_SUCCESS"
    if (
        receipt.get("manifest_relative_path") != manifest_relative
        or receipt.get("success_marker_relative_path") != marker_relative
    ):
        return False
    manifest_path = attempt / manifest_relative
    marker_path = attempt / marker_relative
    if (
        not _is_owned_attempt_path(attempt, manifest_path, directory=False)
        or not _is_owned_attempt_path(attempt, marker_path, directory=False)
    ):
        return False
    try:
        manifest_sha256 = _sha256_file(manifest_path)
        marker_sha256 = _sha256_file(marker_path)
    except OSError:
        return False
    if receipt.get("manifest_sha256") != manifest_sha256:
        return False
    expected_recorded_manifest = {
        "path": str(manifest_path.resolve()),
        "sha256": manifest_sha256,
    }
    expected_recorded_marker = {
        "path": str(marker_path.resolve()),
        "sha256": marker_sha256,
    }
    if (
        artifact.get("manifest") != expected_recorded_manifest
        or artifact.get("success_marker") != expected_recorded_marker
    ):
        return False
    manifest = _json_object(manifest_path)
    marker = _json_object(marker_path)
    static_plan = contract.get("static_plan")
    checkpoint = static_plan.get("checkpoint") if isinstance(static_plan, dict) else None
    source_inventory = (
        static_plan.get("source_inventory") if isinstance(static_plan, dict) else None
    )
    task = static_plan.get("task") if isinstance(static_plan, dict) else None
    protocol = static_plan.get("protocol") if isinstance(static_plan, dict) else None
    if (
        manifest is None
        or marker is None
        or not isinstance(checkpoint, dict)
        or not isinstance(source_inventory, dict)
        or not isinstance(task, dict)
        or not isinstance(protocol, dict)
    ):
        return False
    context = {
        "run_id": attempt.name,
        "worker_id": worker_id,
        "task": task.get("cli_task"),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "contract_hash": _canonical_sha256(contract),
        "source_inventory_digest": source_inventory.get(
            "source_inventory_digest"
        ),
        "model_architecture": checkpoint.get("model_architecture"),
        "action_horizon": checkpoint.get("action_horizon"),
        "num_inference_steps": checkpoint.get("num_inference_steps"),
        "mapping_version": protocol.get("mapping"),
    }
    manifest_keys = {
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
        set(manifest) != manifest_keys
        or manifest.get("schema_version") != MODEL_IO_TRACE_SCHEMA_VERSION
        or any(manifest.get(key) != value for key, value in context.items())
        or manifest.get("trace_session_id") != trace_session_id
        or manifest.get("episode_seed") != job.get("episode_seed")
        or manifest.get("seeded_reset_generation") != seeded_generation
        or manifest.get("result_identity") != job.get("result_identity")
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
        return False
    expected_marker = {
        "schema_version": MODEL_IO_RECEIPT_SCHEMA_VERSION,
        "trace_session_id": trace_session_id,
        "manifest_relative_path": manifest_relative,
        "manifest_sha256": manifest_sha256,
    }
    if marker != expected_marker:
        return False
    try:
        manifest_bytes = manifest_path.read_bytes()
        marker_bytes = marker_path.read_bytes()
        if (
            manifest_bytes != _canonical_json_bytes(manifest) + b"\n"
            or marker_bytes != _canonical_json_bytes(marker) + b"\n"
        ):
            return False
    except (OSError, TypeError, ValueError):
        return False
    call_bytes = _valid_model_io_calls(
        attempt,
        manifest_path.parent,
        manifest,
        context,
        trace_session_id=trace_session_id,
        episode_seed=job.get("episode_seed"),
        seeded_generation=seeded_generation,
        record_count=record_count,
    )
    return (
        call_bytes is not None
        and total_bytes == call_bytes + len(manifest_bytes) + len(marker_bytes)
    )


def _canonical_episode_metrics(
    attempt: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    static_plan = contract.get("static_plan")
    jobs = static_plan.get("episodes") if isinstance(static_plan, dict) else None
    policy = static_plan.get("policy") if isinstance(static_plan, dict) else None
    record_video_every_n = (
        policy.get("record_video_every_n") if isinstance(policy, dict) else None
    )
    if (
        not isinstance(jobs, list)
        or not jobs
        or type(record_video_every_n) is not int
        or record_video_every_n < 0
    ):
        return None
    expected_job_keys = {
        "task_cli",
        "isaac_task_name",
        "test_mode",
        "seed",
        "repeat_idx",
        "episode_seed",
        "episode_index",
        "instruction",
        "env_config_yaml_sha256",
        "result_identity",
    }
    expected_by_filename: dict[str, dict[str, Any]] = {}
    result_identities: set[str] = set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != expected_job_keys:
            return None
        seed = job.get("seed")
        repeat_idx = job.get("repeat_idx")
        episode_index = job.get("episode_index")
        result_identity = job.get("result_identity")
        if (
            type(seed) is not int
            or type(repeat_idx) is not int
            or type(episode_index) is not int
            or repeat_idx < 0
            or episode_index < 0
            or not isinstance(result_identity, str)
            or re.fullmatch(r"[0-9a-f]{64}", result_identity) is None
        ):
            return None
        filename = (
            f"seed-{seed:010d}-repeat-{repeat_idx:03d}-"
            f"episode-{episode_index:05d}.json"
        )
        if filename in expected_by_filename or result_identity in result_identities:
            return None
        expected_by_filename[filename] = job
        result_identities.add(result_identity)

    episodes_dir = attempt / "episodes"
    if not episodes_dir.is_dir() or episodes_dir.is_symlink():
        return None
    try:
        entries = list(episodes_dir.iterdir())
    except OSError:
        return None
    if {entry.name for entry in entries} != set(expected_by_filename):
        return None
    if any(
        not entry.is_file() or entry.is_symlink()
        for entry in entries
    ):
        return None

    attempt_root = attempt.resolve()
    successes = 0
    execution_failed = 0
    by_seed: dict[str, dict[str, int]] = {}
    canonical_keys = {
        "schema_version",
        "result_identity",
        "job",
        "payload",
        "worker_id",
        "classification",
        "artifacts",
        "completion_sha256",
    }
    for filename, job in expected_by_filename.items():
        episode = _json_object(episodes_dir / filename)
        if (
            episode is None
            or set(episode) != canonical_keys
            or episode.get("schema_version") != EPISODE_SCHEMA_VERSION
            or episode.get("result_identity") != job["result_identity"]
            or not _strict_json_equal(episode.get("job"), job)
            or not isinstance(episode.get("worker_id"), str)
            or _WORKER_RE.fullmatch(episode["worker_id"]) is None
            or not isinstance(episode.get("completion_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", episode["completion_sha256"])
            is None
        ):
            return None
        completion_path = attempt / "completions" / filename
        if (
            not _is_owned_attempt_path(attempt, completion_path, directory=False)
            or episode.get("completion_sha256") != _sha256_file(completion_path)
        ):
            return None
        payload = episode.get("payload")
        artifacts = episode.get("artifacts")
        if not isinstance(payload, dict) or not isinstance(artifacts, dict):
            return None
        expected_identity = {
            "task": job["isaac_task_name"],
            "seed": job["seed"],
            "repeat_idx": job["repeat_idx"],
            "episode_seed": job["episode_seed"],
            "episode_index": job["episode_index"],
        }
        if any(
            not _strict_json_equal(payload.get(key), value)
            for key, value in expected_identity.items()
        ):
            return None
        success = payload.get("success")
        failure_reason = payload.get("failure_reason")
        error = payload.get("error", "")
        classification = episode.get("classification")
        if (
            not isinstance(success, bool)
            or not isinstance(failure_reason, str)
            or not isinstance(error, str)
            or error
        ):
            return None
        if failure_reason == _EXECUTION_FAILED:
            if success or classification != "execution_failed":
                return None
            execution_failed += 1
        elif failure_reason in _LEGAL_TASK_OUTCOMES:
            if success != (failure_reason == "success") or classification != "task_complete":
                return None
        else:
            return None
        episode_steps = payload.get("episode_steps")
        duration_sec = payload.get("duration_sec")
        if (
            type(episode_steps) is not int
            or episode_steps < 0
            or isinstance(duration_sec, bool)
            or not isinstance(duration_sec, (int, float))
            or not math.isfinite(float(duration_sec))
            or float(duration_sec) < 0.0
        ):
            return None
        video_recorded = payload.get("video_recorded")
        video_path_raw = payload.get("video_path")
        if not isinstance(video_recorded, bool) or not isinstance(video_path_raw, str):
            return None
        if classification == "task_complete":
            requires_video = (
                record_video_every_n > 0
                and job["episode_index"] % record_video_every_n == 0
            )
            expected_artifacts = (
                {"video", "model_io"} if requires_video else {"model_io"}
            )
            if set(artifacts) != expected_artifacts:
                return None
            if requires_video:
                video = artifacts.get("video")
                if (
                    video_recorded is not True
                    or not video_path_raw
                    or not isinstance(video, dict)
                    or set(video) != {"path", "sha256"}
                ):
                    return None
                artifact_path_raw = video.get("path")
                if not isinstance(artifact_path_raw, str):
                    return None
                try:
                    video_path = Path(video_path_raw).expanduser().resolve()
                    artifact_path = Path(artifact_path_raw).expanduser().resolve()
                except (OSError, RuntimeError):
                    return None
                if (
                    video_path != artifact_path
                    or not _is_relative_to(video_path, attempt_root)
                    or not video_path.is_file()
                    or video_path.is_symlink()
                    or video.get("sha256") != _sha256_file(video_path)
                ):
                    return None
            elif video_recorded is not False or video_path_raw != "":
                return None
            if not _valid_model_io_artifact(
                attempt,
                contract,
                job,
                payload,
                episode["worker_id"],
                artifacts.get("model_io"),
            ):
                return None
        elif video_recorded is not False or artifacts != {} or video_path_raw != "":
            return None
        if success:
            successes += 1
        seed_key = str(job["seed"])
        counts = by_seed.setdefault(seed_key, {"completed": 0, "success": 0})
        counts["completed"] += 1
        counts["success"] += int(success)

    completed = len(jobs)
    return {
        "completed": completed,
        "task_success": successes,
        "task_failure": completed - successes,
        "execution_failed": execution_failed,
        "success_rate": {"numerator": successes, "denominator": completed},
        "by_seed": by_seed,
    }


def _terminal_counts_match(
    summary: Mapping[str, Any],
    runtime: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    expected_state: str,
) -> bool:
    completed = metrics.get("completed")
    actual_execution_failed = metrics.get("execution_failed")
    reused = summary.get("resume_reused")
    if (
        type(completed) is not int
        or completed < 1
        or type(actual_execution_failed) is not int
        or not 0 <= actual_execution_failed <= completed
        or type(reused) is not int
        or not 0 <= reused <= completed
        or summary.get("expected") != completed
        or summary.get("completed") != completed
        or summary.get("task_success") != metrics.get("task_success")
        or summary.get("task_failure") != metrics.get("task_failure")
        or not _strict_json_equal(
            summary.get("success_rate"), metrics.get("success_rate")
        )
        or not _strict_json_equal(summary.get("by_seed"), metrics.get("by_seed"))
        or summary.get("infrastructure_failure") != 0
        or summary.get("infrastructure_errors") != []
        or type(runtime.get("completed")) is not int
        or runtime.get("completed") != completed
        or type(runtime.get("reused")) is not int
        or runtime.get("reused") != reused
    ):
        return False
    raw_summary_execution_failed = summary.get("execution_failed", 0)
    if (
        type(raw_summary_execution_failed) is not int
        or raw_summary_execution_failed != actual_execution_failed
    ):
        return False
    if expected_state == "success":
        if actual_execution_failed != 0 or set(runtime) not in (
            {"completed", "reused"},
            {"completed", "reused", "execution_failed"},
        ):
            return False
        runtime_execution_failed = runtime.get("execution_failed", 0)
        return (
            type(runtime_execution_failed) is int
            and runtime_execution_failed == 0
        )
    if expected_state == "completed_with_failures":
        return (
            actual_execution_failed > 0
            and "execution_failed" in summary
            and set(runtime) == {"completed", "reused", "execution_failed"}
            and type(runtime.get("execution_failed")) is int
            and runtime.get("execution_failed") == actual_execution_failed
        )
    return False


def _valid_terminal_attempt(
    contract_dir: Path,
    attempt: Path,
    marker: Path,
    contract: Mapping[str, Any],
    relative_path: str,
    *,
    repo_root: Path | None = None,
    contract_prevalidated: bool = False,
) -> bool:
    expected_state = TERMINAL_MARKER_STATES.get(marker.name)
    present_markers = [
        marker_name
        for marker_name in TERMINAL_MARKER_STATES
        if (attempt / marker_name).exists()
    ]
    if (
        expected_state is None
        or present_markers != [marker.name]
        or not _is_owned_attempt_path(attempt, marker, directory=False)
        or (
            not contract_prevalidated
            and not _matches_full_pp_box_contract(
                contract, relative_path, repo_root=repo_root
            )
        )
    ):
        return False
    if repo_root is not None and not contract_prevalidated:
        recorded_inventory = _json_object(contract_dir / "source_inventory.json")
        static_plan = contract.get("static_plan")
        expected_inventory = (
            static_plan.get("source_inventory") if isinstance(static_plan, dict) else None
        )
        if recorded_inventory is None or recorded_inventory != expected_inventory:
            return False
    summary_path = attempt / "summary.json"
    manifest_path = attempt / "manifest.json"
    if (
        not _is_owned_attempt_path(attempt, summary_path, directory=False)
        or not _is_owned_attempt_path(attempt, manifest_path, directory=False)
    ):
        return False
    summary = _json_object(summary_path)
    manifest = _json_object(manifest_path)
    if summary is None or manifest is None:
        return False
    expected = summary.get("expected")
    completed = summary.get("completed")
    infrastructure_failure = summary.get("infrastructure_failure")
    execution_failed = summary.get("execution_failed", 0)
    task_failure = summary.get("task_failure")
    contract_hash = summary.get("contract_hash")
    try:
        contract_digest = _canonical_sha256(contract)
    except (TypeError, ValueError):
        return False
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or type(expected) is not int
        or expected != EXPECTED_EPISODES
        or type(completed) is not int
        or completed != expected
        or type(infrastructure_failure) is not int
        or infrastructure_failure != 0
        or type(execution_failed) is not int
        or type(task_failure) is not int
        or execution_failed > task_failure
        or (
            execution_failed != 0
            if expected_state == "success"
            else execution_failed <= 0 or "execution_failed" not in summary
        )
        or summary.get("infrastructure_errors") != []
        or not _matches_summary_totals(summary)
        or not isinstance(contract_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", contract_hash) is None
        or contract_hash != contract_digest
        or contract_dir.name != contract_hash[:16]
    ):
        return False
    runtime = manifest.get("runtime")
    if (
        manifest.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or manifest.get("contract_hash") != contract_hash
        or manifest.get("state") != expected_state
        or manifest.get("run_id") != attempt.name
        or not isinstance(runtime, dict)
        or type(runtime.get("completed")) is not int
        or runtime.get("completed") != EXPECTED_EPISODES
    ):
        return False
    if repo_root is not None:
        static_plan = contract.get("static_plan")
        source_inventory = (
            static_plan.get("source_inventory") if isinstance(static_plan, dict) else None
        )
        source_digest = (
            source_inventory.get("source_inventory_digest")
            if isinstance(source_inventory, dict)
            else None
        )
        try:
            episode_metrics = _canonical_episode_metrics(attempt, contract)
        except (OSError, RuntimeError, TypeError, ValueError):
            return False
        if (
            set(manifest)
            != {"schema_version", "contract_hash", "run_id", "state", "runtime"}
            or summary.get("source_inventory_digest") != source_digest
            or episode_metrics is None
            or not _terminal_counts_match(
                summary,
                runtime,
                episode_metrics,
                expected_state=expected_state,
            )
        ):
            return False
    try:
        recorded_digest = marker.read_text(encoding="ascii").strip()
        summary_digest = _canonical_sha256(summary)
    except (OSError, UnicodeDecodeError, TypeError, ValueError):
        return False
    return recorded_digest == summary_digest


def _select_resumable_attempt(
    attempts_root: Path,
    contract_hash: str,
) -> Path | None:
    """Return only the latest watcher-visible attempt, failing closed on drift."""
    try:
        candidates = [
            (path.lstat().st_mtime_ns, path.name, path)
            for path in attempts_root.iterdir()
            if _ATTEMPT_ID_RE.fullmatch(path.name) is not None
        ]
        if not candidates:
            return None
        candidate = max(candidates)[2]
        entry_stat = candidate.lstat()
        if entry_stat.st_uid != os.getuid():
            return None
        if stat.S_ISLNK(entry_stat.st_mode):
            target = os.readlink(candidate)
            target_path = Path(target)
            pattern = re.compile(
                rf"^\.{re.escape(candidate.name)}\.object\.[0-9]+\.[0-9]+\.[a-f0-9]{{16}}$"
            )
            if target_path.is_absolute() or len(target_path.parts) != 1 or pattern.fullmatch(target) is None:
                return None
            backing = attempts_root / target
            backing_stat = backing.lstat()
            if (
                backing_stat.st_uid != os.getuid()
                or stat.S_ISLNK(backing_stat.st_mode)
                or not stat.S_ISDIR(backing_stat.st_mode)
                or backing.resolve(strict=True).parent != attempts_root.resolve(strict=True)
            ):
                return None
        elif not stat.S_ISDIR(entry_stat.st_mode):
            return None
    except (OSError, RuntimeError):
        return None
    manifest_path = candidate / "manifest.json"
    if not _is_owned_attempt_path(candidate, manifest_path, directory=False):
        return None
    manifest = _json_object(manifest_path)
    valid_states = {
        "starting",
        "running",
        "success",
        "completed_with_failures",
        "failed",
        "interrupted",
        "unsafe_cleanup",
    }
    if (
        manifest is None
        or manifest.get("schema_version") != ATTEMPT_SCHEMA_VERSION
        or manifest.get("contract_hash") != contract_hash
        or manifest.get("run_id") != candidate.name
        or manifest.get("state") not in valid_states
    ):
        return None
    return candidate


def checkpoint_contract_statuses(
    eval_root: Path,
    candidate_relative_paths: set[str],
    *,
    allowed_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, CheckpointContractStatus]:
    root = eval_root.expanduser().resolve()
    if allowed_root is not None:
        allowed = allowed_root.expanduser().resolve()
        if not _is_relative_to(root, allowed):
            raise ShardConfigError(f"evaluation root escapes {allowed}: {root}")
    statuses = {
        relative: CheckpointContractStatus("pending")
        for relative in candidate_relative_paths
    }
    if not root.exists():
        return statuses
    if not root.is_dir():
        raise ShardConfigError(f"evaluation root is not a directory: {root}")

    candidates: dict[str, list[ContractCandidate]] = {}
    for raw_contract_path in root.glob("**/contract.json"):
        contract_path = raw_contract_path.resolve()
        if not _is_relative_to(contract_path, root):
            raise ShardConfigError(
                f"evaluation contract escapes {root}: {raw_contract_path}"
            )
        contract = _json_object(contract_path)
        if contract is None:
            continue
        static_plan = contract.get("static_plan")
        checkpoint = (
            static_plan.get("checkpoint")
            if isinstance(static_plan, dict)
            else None
        )
        relative = (
            checkpoint.get("relative_path")
            if isinstance(checkpoint, dict)
            else None
        )
        if (
            not isinstance(relative, str)
            or relative not in candidate_relative_paths
        ):
            continue
        try:
            contract_matches = _matches_full_pp_box_contract(
                contract, relative, repo_root=repo_root
            )
        except (OSError, TypeError, ValueError):
            contract_matches = False
        if not contract_matches:
            continue
        policy = (
            static_plan.get("policy")
            if isinstance(static_plan, dict)
            else None
        )
        recorded_output_root = (
            policy.get("output_root")
            if isinstance(policy, dict)
            else None
        )
        if (
            not isinstance(recorded_output_root, str)
            or Path(recorded_output_root).expanduser().resolve() != root
        ):
            raise ShardConfigError(
                f"current contract output_root does not match its scan root: "
                f"{contract_path}"
            )

        try:
            contract_hash = _canonical_sha256(contract)
        except (TypeError, ValueError) as exc:
            raise ShardConfigError(
                f"current evaluation contract is not canonical: {contract_path}"
            ) from exc
        expected_label = f"{EXPECTED_CHECKPOINT_TASK}-{Path(relative).name}"
        if (
            contract_path.parent.name != contract_hash[:16]
            or contract_path.parent.parent.name != expected_label
            or contract_path.parent.parent.parent.name != "pp_box"
        ):
            raise ShardConfigError(
                f"current evaluation contract has an unsafe output identity: "
                f"{contract_path}"
            )
        if repo_root is not None:
            recorded_inventory = _json_object(
                contract_path.parent / "source_inventory.json"
            )
            if recorded_inventory != static_plan.get("source_inventory"):
                raise ShardConfigError(
                    f"current contract source inventory sidecar drifted: "
                    f"{contract_path}"
                )

        attempts_root = (contract_path.parent / "attempts").resolve()
        if not _is_relative_to(attempts_root, root):
            raise ShardConfigError(
                f"evaluation attempts escape {root}: {attempts_root}"
            )
        if not attempts_root.exists():
            continue
        if not attempts_root.is_dir():
            raise ShardConfigError(
                f"evaluation attempts path is not a directory: {attempts_root}"
            )
        selected_attempt = _select_resumable_attempt(
            attempts_root, contract_hash
        )
        if selected_attempt is None:
            try:
                entries = tuple(attempts_root.iterdir())
            except OSError as exc:
                raise ShardConfigError(
                    f"cannot inspect evaluation attempts: {attempts_root}"
                ) from exc
            if entries:
                raise ShardConfigError(
                    f"latest attempt is not safely resumable: {attempts_root}"
                )
            continue

        manifest = _json_object(selected_attempt / "manifest.json")
        if manifest is None:
            raise ShardConfigError(
                f"latest attempt manifest is invalid: {selected_attempt}"
            )
        manifest_state = manifest.get("state")
        if manifest_state == "unsafe_cleanup":
            raise ShardConfigError(
                f"latest attempt requires manual unsafe-cleanup audit: "
                f"{selected_attempt}"
            )
        present_markers = [
            marker_name
            for marker_name in TERMINAL_MARKER_STATES
            if os.path.lexists(selected_attempt / marker_name)
        ]
        if len(present_markers) > 1:
            raise ShardConfigError(
                f"latest attempt has conflicting terminal markers: "
                f"{selected_attempt}"
            )
        if not present_markers:
            if manifest_state in {"success", "completed_with_failures"}:
                raise ShardConfigError(
                    f"latest terminal manifest lacks its marker: "
                    f"{selected_attempt}"
                )
            candidates.setdefault(relative, []).append(
                ContractCandidate(
                    selected_attempt.name,
                    contract_path,
                    contract_hash,
                    selected_attempt.resolve(),
                    "pending",
                )
            )
            continue

        marker = selected_attempt / present_markers[0]
        if not _is_owned_attempt_path(
            selected_attempt, marker, directory=False
        ):
            raise ShardConfigError(
                f"evaluation marker is unsafe for "
                f"{selected_attempt.resolve()}: {marker}"
            )
        expected_manifest_state = TERMINAL_MARKER_STATES[marker.name]
        terminal_valid = _valid_terminal_attempt(
            contract_path.parent,
            selected_attempt,
            marker,
            contract,
            relative,
            repo_root=repo_root,
            contract_prevalidated=True,
        )
        if not terminal_valid or manifest_state != expected_manifest_state:
            raise ShardConfigError(
                f"latest terminal attempt is invalid: {selected_attempt}"
            )
        candidate_state = (
            "success"
            if marker.name == "_SUCCESS"
            else (
                "retry_required"
                if _contract_execution_host_is_current(contract)
                else "pending"
            )
        )
        candidates.setdefault(relative, []).append(
            ContractCandidate(
                selected_attempt.name,
                contract_path,
                contract_hash,
                selected_attempt.resolve(),
                candidate_state,
            )
        )

    for relative, relative_candidates in candidates.items():
        successful = [
            candidate
            for candidate in relative_candidates
            if candidate.state == "success"
        ]
        if successful:
            accepted = max(successful, key=lambda item: item.sort_key)
            statuses[relative] = CheckpointContractStatus(
                "completed",
                str(accepted.contract_path),
                accepted.contract_hash,
                str(accepted.attempt_path),
            )
            continue
        latest = max(relative_candidates, key=lambda item: item.sort_key)
        if latest.state == "retry_required":
            statuses[relative] = CheckpointContractStatus(
                "retry_required",
                str(latest.contract_path),
                latest.contract_hash,
                str(latest.attempt_path),
            )
        else:
            statuses[relative] = CheckpointContractStatus(
                "pending",
                str(latest.contract_path),
                latest.contract_hash,
                str(latest.attempt_path),
            )
    return statuses


def completed_checkpoint_paths(
    eval_root: Path,
    candidate_relative_paths: set[str],
    *,
    allowed_root: Path | None = None,
    repo_root: Path | None = None,
) -> set[str]:
    statuses = checkpoint_contract_statuses(
        eval_root,
        candidate_relative_paths,
        allowed_root=allowed_root,
        repo_root=repo_root,
    )
    return {
        relative
        for relative, status in statuses.items()
        if status.state == "completed"
    }


def select_pending_shard(
    targets: Sequence[ShardTarget],
    completed: set[str],
    index: int,
    count: int,
) -> list[ShardTarget]:
    validate_shard(index=index, count=count)
    return [
        target
        for target in targets
        if target.ordinal % count == index and target.relative_path not in completed
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument(
        "--output-format",
        choices=("paths", "actions", "jsonl"),
        default="paths",
    )
    parser.add_argument("checkpoint_roots", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_shard(index=args.shard_index, count=args.shard_count)
        targets = discover_targets(args.repo_root, args.checkpoint_roots)
        statuses = checkpoint_contract_statuses(
            args.eval_root, {target.relative_path for target in targets},
            allowed_root=args.repo_root / ".runs",
            repo_root=args.repo_root,
        )
        completed = {
            relative
            for relative, status in statuses.items()
            if status.state == "completed"
        }
        selected = select_pending_shard(
            targets, completed, args.shard_index, args.shard_count
        )
    except ShardConfigError as exc:
        print(f"[run_ha_eval] configuration error: {exc}", file=sys.stderr)
        return 2
    retry_required = {
        relative
        for relative, status in statuses.items()
        if status.state == "retry_required"
    }
    print(
        f"[run_ha_eval] shard={args.shard_index}/{args.shard_count} "
        f"total={len(targets)} completed={len(completed)} "
        f"retry_required={len(retry_required)} "
        f"pending={len(targets) - len(completed)} assigned={len(selected)}",
        file=sys.stderr,
        flush=True,
    )
    for target in targets:
        status = statuses[target.relative_path]
        if status.state == "completed":
            print(
                f"[run_ha_eval] accepted checkpoint={target.relative_path} "
                f"contract_path={status.contract_path} "
                f"contract_hash={status.contract_hash}",
                file=sys.stderr,
                flush=True,
            )

    if args.output_format == "jsonl":
        selected_ordinals = {target.ordinal for target in selected}
        for target in targets:
            status = statuses[target.relative_path]
            payload = {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "ordinal": target.ordinal,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "assigned": target.ordinal in selected_ordinals,
                "path": str(target.path),
                "relative_path": target.relative_path,
                "step": target.step,
                "status": status.state,
                "accepted_contract_path": (
                    status.contract_path
                    if status.state == "completed"
                    else None
                ),
                "accepted_contract_hash": (
                    status.contract_hash
                    if status.state == "completed"
                    else None
                ),
                "retry_required_contract_path": (
                    status.contract_path
                    if status.state == "retry_required"
                    else None
                ),
                "retry_required_contract_hash": (
                    status.contract_hash
                    if status.state == "retry_required"
                    else None
                ),
                "attempt_path": status.attempt_path,
            }
            print(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return 0

    for target in selected:
        status = statuses[target.relative_path]
        if args.output_format == "paths":
            print(target.path, flush=True)
        elif status.state == "retry_required":
            print(
                "\t".join(
                    (
                        "retry_required",
                        str(target.path),
                        str(status.contract_hash),
                        str(status.contract_path),
                    )
                ),
                flush=True,
            )
        else:
            print(f"run\t{target.path}\t\t", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

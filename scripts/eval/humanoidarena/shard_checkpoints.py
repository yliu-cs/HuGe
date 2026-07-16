#!/usr/bin/env python3
"""Select pending HumanoidArena checkpoints for a stable multi-host shard."""

from __future__ import annotations

import argparse
import hashlib
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
STATIC_PLAN_SCHEMA_VERSION = "huge_humanoidarena_static_plan_v1"
CONTRACT_SCHEMA_VERSION = "huge_humanoidarena_contract_v1"
EPISODE_SCHEMA_VERSION = "huge_humanoidarena_episode_v1"
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
    "scripts/eval/humanoidarena/watch_eval_summary.py",
    "scripts/eval/humanoidarena/shard_checkpoints.py",
    "scripts/eval/humanoidarena/sim_eval_shadow.py",
    "scripts/eval/humanoidarena/run_eval.sh",
    "scripts/eval/humanoidarena/env.sh",
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


def discover_targets(repo_root: Path, checkpoint_roots: Sequence[Path]) -> list[ShardTarget]:
    repo = repo_root.expanduser().resolve()
    allowed = (repo / ".runs/stage3_humanoidarena").resolve()
    ordered: list[tuple[Path, str, int]] = []
    seen: set[Path] = set()
    for raw_root in checkpoint_roots:
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            raise ShardConfigError(f"checkpoint root is not a directory: {root}")
        if not _is_relative_to(root, allowed):
            raise ShardConfigError(f"checkpoint root escapes {allowed}: {root}")
        local: list[tuple[Path, str, int]] = []
        for lexical_path in root.rglob("step_*"):
            match = STEP_RE.fullmatch(lexical_path.name)
            if match is None or not lexical_path.is_dir():
                continue
            resolved = lexical_path.resolve()
            if not _is_relative_to(resolved, root) or not _is_relative_to(resolved, allowed):
                raise ShardConfigError(f"checkpoint escapes its root: {lexical_path}")
            if resolved in seen:
                raise ShardConfigError(f"duplicate checkpoint directory: {resolved}")
            seen.add(resolved)
            relative = lexical_path.relative_to(repo).as_posix()
            local.append((resolved, relative, int(match.group(1))))
        local.sort(key=lambda item: (str(Path(item[1]).parent), item[2], item[1]))
        ordered.extend(local)
    if not ordered:
        raise ShardConfigError("no step_<number> checkpoint directories were discovered")
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
    allowed_checkpoint_root = (repo / ".runs/stage3_humanoidarena").resolve()
    if (
        not checkpoint.is_dir()
        or not _is_relative_to(checkpoint, allowed_checkpoint_root)
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
    except OSError:
        return False
    if (
        set(current_files) != set(REQUIRED_CHECKPOINT_FILES)
        or checkpoint_payload.get("files") != current_files
        or checkpoint_payload.get("checkpoint_sha256")
        != _canonical_sha256(current_files)
    ):
        return False
    config_payload = _json_object(checkpoint / "config.json")
    config = config_payload.get("config") if isinstance(config_payload, dict) else None
    metadata = config_payload.get("metadata") if isinstance(config_payload, dict) else None
    if not isinstance(config, dict) or not isinstance(metadata, dict):
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
        "max_steps": EXPECTED_MAX_STEPS,
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
    try:
        current_policy = {
            "video_fps": int(os.environ.get("VIDEO_FPS", "30")),
            "post_termination_record_steps": int(
                os.environ.get("POST_TERMINATION_RECORD_STEPS", "10")
            ),
            "record_video_every_n": int(
                os.environ.get("HUGE_EVAL_RECORD_VIDEO_EVERY_N", "1")
            ),
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
            "episode_max_retries": int(
                os.environ.get("HUGE_EVAL_EPISODE_MAX_RETRIES", "5")
            ),
            "retry_backoff_sec": float(
                os.environ.get("HUGE_EVAL_RETRY_BACKOFF_SEC", "5")
            ),
            "workers_per_gpu": int(
                os.environ.get("HUGE_EVAL_WORKERS_PER_GPU", "1")
            ),
        }
    except ValueError:
        return False
    if any(policy.get(key) != value for key, value in current_policy.items()):
        return False
    if static_plan.get("protocol") != {
        "http": "huge_humanoidarena_http_v1",
        "mapping": "humanoidarena_rotlocal_v31_to_huge_proprio_v1",
        "action": [50, 66],
        "latent": [50, 64],
    } or static_plan.get("episode_seed_algorithm") != EPISODE_SEED_FORMULA:
        return False
    software = static_plan.get("software")
    if software != {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }:
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
    bindings = contract.get("gpu_bindings")
    if not isinstance(bindings, list) or len(bindings) != 8:
        return False
    identities: set[tuple[int, int]] = set()
    renderers: set[int] = set()
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
            or set(gpu)
            != {
                "parent_logical_ordinal",
                "child_cuda_visible_token",
                "cuda_uuid",
                "pci_bus_id",
                "device_name",
                "total_memory_bytes",
                "compute_capability",
                "free_memory_bytes",
            }
            or type(renderer) is not int
            or renderer < 0
            or worker_slot != 0
        ):
            return False
        ordinal = gpu.get("parent_logical_ordinal")
        if (
            type(ordinal) is not int
            or ordinal < 0
            or not isinstance(gpu.get("cuda_uuid"), str)
            or re.fullmatch(r"GPU-[A-Za-z0-9-]+", gpu["cuda_uuid"], re.IGNORECASE)
            is None
            or type(gpu.get("total_memory_bytes")) is not int
            or gpu["total_memory_bytes"] <= 0
            or (
                gpu.get("free_memory_bytes") is not None
                and type(gpu.get("free_memory_bytes")) is not int
            )
        ):
            return False
        identities.add((ordinal, worker_slot))
        renderers.add(renderer)
    return identities == {(ordinal, 0) for ordinal in range(8)} and len(renderers) == 8


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


def _canonical_episode_metrics(
    attempt: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    static_plan = contract.get("static_plan")
    jobs = static_plan.get("episodes") if isinstance(static_plan, dict) else None
    if not isinstance(jobs, list) or not jobs:
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
        if video_recorded:
            video = artifacts.get("video")
            if not isinstance(video, dict) or set(artifacts) != {"video"} or set(video) != {
                "path",
                "sha256",
            }:
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
                not video_path_raw
                or video_path != artifact_path
                or not _is_relative_to(video_path, attempt_root)
                or not video_path.is_file()
                or video_path.is_symlink()
                or video.get("sha256") != _sha256_file(video_path)
            ):
                return None
        elif artifacts != {} or video_path_raw != "":
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


def completed_checkpoint_paths(
    eval_root: Path,
    candidate_relative_paths: set[str],
    *,
    allowed_root: Path | None = None,
    repo_root: Path | None = None,
) -> set[str]:
    root = eval_root.expanduser().resolve()
    if allowed_root is not None:
        allowed = allowed_root.expanduser().resolve()
        if not _is_relative_to(root, allowed):
            raise ShardConfigError(f"evaluation root escapes {allowed}: {root}")
    if not root.exists():
        return set()
    if not root.is_dir():
        raise ShardConfigError(f"evaluation root is not a directory: {root}")
    selected_outcomes: dict[str, list[tuple[str, bool]]] = {}
    blocked: set[str] = set()
    for raw_contract_path in root.glob("**/contract.json"):
        contract_path = raw_contract_path.resolve()
        if not _is_relative_to(contract_path, root):
            raise ShardConfigError(f"evaluation contract escapes {root}: {raw_contract_path}")
        contract = _json_object(contract_path)
        if contract is None:
            continue
        static_plan = contract.get("static_plan")
        checkpoint = static_plan.get("checkpoint") if isinstance(static_plan, dict) else None
        relative = checkpoint.get("relative_path") if isinstance(checkpoint, dict) else None
        if not isinstance(relative, str) or relative not in candidate_relative_paths:
            continue
        try:
            contract_matches = _matches_full_pp_box_contract(
                contract, relative, repo_root=repo_root
            )
        except (OSError, TypeError, ValueError):
            contract_matches = False
        if not contract_matches:
            continue
        if repo_root is not None:
            expected_label = f"{EXPECTED_CHECKPOINT_TASK}-{Path(relative).name}"
            if (
                contract_path.parent.parent.name != expected_label
                or contract_path.parent.parent.parent.name != "pp_box"
            ):
                blocked.add(relative)
                continue
            recorded_inventory = _json_object(
                contract_path.parent / "source_inventory.json"
            )
            if recorded_inventory != static_plan.get("source_inventory"):
                blocked.add(relative)
                continue
        attempts_root = (contract_path.parent / "attempts").resolve()
        if not _is_relative_to(attempts_root, root):
            raise ShardConfigError(f"evaluation attempts escape {root}: {attempts_root}")
        if not attempts_root.is_dir():
            blocked.add(relative)
            continue
        try:
            contract_hash = _canonical_sha256(contract)
        except (TypeError, ValueError):
            continue
        selected_attempt = _select_resumable_attempt(attempts_root, contract_hash)
        if selected_attempt is None:
            blocked.add(relative)
            continue
        present_markers = [
            marker_name
            for marker_name in TERMINAL_MARKER_STATES
            if os.path.lexists(selected_attempt / marker_name)
        ]
        if len(present_markers) != 1:
            selected_outcomes.setdefault(relative, []).append(
                (selected_attempt.name, False)
            )
            continue
        marker = selected_attempt / present_markers[0]
        if not _is_owned_attempt_path(selected_attempt, marker, directory=False):
            raise ShardConfigError(
                f"evaluation marker is unsafe for {selected_attempt.resolve()}: {marker}"
            )
        terminal_valid = _valid_terminal_attempt(
            contract_path.parent,
            selected_attempt,
            marker,
            contract,
            relative,
            repo_root=repo_root,
            contract_prevalidated=True,
        )
        selected_outcomes.setdefault(relative, []).append(
            (selected_attempt.name, terminal_valid)
        )
    completed: set[str] = set()
    for relative, outcomes in selected_outcomes.items():
        if relative in blocked:
            continue
        _attempt_name, terminal_valid = max(outcomes, key=lambda item: item[0])
        if terminal_valid:
            completed.add(relative)
    return completed


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
    parser.add_argument("checkpoint_roots", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_shard(index=args.shard_index, count=args.shard_count)
        targets = discover_targets(args.repo_root, args.checkpoint_roots)
        completed = completed_checkpoint_paths(
            args.eval_root, {target.relative_path for target in targets},
            allowed_root=args.repo_root / ".runs",
            repo_root=args.repo_root,
        )
        selected = select_pending_shard(
            targets, completed, args.shard_index, args.shard_count
        )
    except ShardConfigError as exc:
        print(f"[run_ha_eval] configuration error: {exc}", file=sys.stderr)
        return 2
    print(
        f"[run_ha_eval] shard={args.shard_index}/{args.shard_count} "
        f"total={len(targets)} completed={len(completed)} "
        f"pending={len(targets) - len(completed)} assigned={len(selected)}",
        file=sys.stderr,
        flush=True,
    )
    for target in selected:
        print(target.path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

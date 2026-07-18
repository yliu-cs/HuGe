#!/usr/bin/env python3
"""Print a watch-friendly summary of HumanoidArena evaluation results.

Run from the repository root, for example:

    watch -n 2 -d -- .venv/bin/python \
      scripts/eval/humanoidarena/watch_eval_summary.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import stat
import statistics
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any


DONE = "DONE"
DONE_WITH_FAILURES = "DONE_WITH_FAILURES"
RUNNING = "RUNNING"
PENDING = "PENDING"
ERROR = "ERROR"

SUCCESS_MARKER = "_SUCCESS"
COMPLETED_WITH_FAILURES_MARKER = "_COMPLETED_WITH_FAILURES"
MANIFEST_SCHEMA_VERSION = "huge_humanoidarena_attempt_v1"
COMPLETION_SCHEMA_VERSION = "huge_humanoidarena_completion_v1"
EPISODE_SCHEMA_VERSION = "huge_humanoidarena_episode_v1"
SUMMARY_SCHEMA_VERSION = "huge_humanoidarena_summary_v1"
SUMMARY_ROW_SCHEMA_VERSION = "huge_humanoidarena_summary_row_v1"

_LEGAL_TASK_OUTCOMES = {"success", "fall", "timeout", "max_steps", "sim_stopped"}
_EXECUTION_FAILED = "execution_failed"
_WORKER_RE = re.compile(r"^w[0-9]{2}$")
_ATTEMPT_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[a-f0-9]{8}$")


class SummaryError(ValueError):
    """Raised when an evaluation result does not match the expected schema."""


@dataclass(frozen=True)
class EvalRow:
    task: str
    architecture: str
    mode: str
    run_id: str
    step: int
    status: str
    progress: str
    success: str
    seed_rates: str
    error: str | None = None
    execution_failed: str = "0"
    checkpoint_relative_path: str | None = None
    accepted_contract_path: str | None = None
    accepted_contract_hash: str | None = None


@dataclass(frozen=True)
class ExpectedEpisode:
    filename: str
    result_identity: str
    job: dict[str, object]
    task: str
    seed: int
    repeat_idx: int
    episode_seed: int
    episode_index: int


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise SummaryError(f"{label} must contain a JSON object")
    return value


def _canonical_sha256(payload: object, *, label: str) -> str:
    try:
        content = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"cannot canonicalize {label}: {exc}") from exc
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SummaryError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


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
        relative = path.relative_to(attempt)
        if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise SummaryError(f"{label} has an unsafe relative path")
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
                raise SummaryError(f"{label} must be an owned real path: {path}")
        resolved = path.resolve(strict=True)
    except SummaryError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SummaryError(f"cannot validate {label}: {exc}") from exc
    if not _is_relative_to(resolved, attempt_root):
        raise SummaryError(f"{label} escapes the attempt: {path}")
    return path


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"{label} must be an object")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"{label} must be an integer")
    return value


def _checkpoint_identity(
    contract: dict[str, Any],
) -> tuple[str, str, str, str, int, int, str]:
    static_plan = _mapping(contract.get("static_plan"), label="static_plan")
    checkpoint = _mapping(static_plan.get("checkpoint"), label="static_plan.checkpoint")
    task_config = _mapping(static_plan.get("task"), label="static_plan.task")

    task = task_config.get("cli_task")
    relative_path = checkpoint.get("relative_path")
    if not isinstance(task, str) or not task:
        raise SummaryError("static_plan.task.cli_task must be a non-empty string")
    if not isinstance(relative_path, str) or not relative_path:
        raise SummaryError("checkpoint.relative_path must be a non-empty string")

    parts = Path(relative_path).parts
    if len(parts) < 5 or re.fullmatch(r"step_\d+", parts[-1]) is None:
        raise SummaryError(
            "checkpoint.relative_path must end with architecture/mode/stage/run_id/step_N"
        )
    architecture, mode, run_id = parts[-5], parts[-4], parts[-2]
    step = _integer(checkpoint.get("step"), label="checkpoint.step")
    if parts[-1] != f"step_{step}":
        raise SummaryError("checkpoint.step does not match checkpoint.relative_path")

    episodes = static_plan.get("episodes")
    if not isinstance(episodes, list):
        raise SummaryError("static_plan.episodes must be a list")
    return task, architecture, mode, run_id, step, len(episodes), relative_path


def _contract_episodes(
    contract: dict[str, Any],
) -> tuple[dict[str, ExpectedEpisode], dict[str, int]]:
    static_plan = _mapping(contract.get("static_plan"), label="static_plan")
    task_config = _mapping(static_plan.get("task"), label="static_plan.task")
    task_cli = task_config.get("cli_task")
    raw_episodes = static_plan.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise SummaryError("static_plan.episodes must be a non-empty list")

    expected_keys = {
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
    episodes: dict[str, ExpectedEpisode] = {}
    result_identities: set[str] = set()
    seed_counts: dict[str, int] = {}
    for position, raw_episode in enumerate(raw_episodes):
        episode = _mapping(raw_episode, label=f"static_plan.episodes[{position}]")
        if set(episode) != expected_keys:
            raise SummaryError(
                f"static_plan.episodes[{position}] fields do not match EpisodeJob"
            )
        string_fields = (
            "task_cli",
            "isaac_task_name",
            "test_mode",
            "instruction",
            "env_config_yaml_sha256",
            "result_identity",
        )
        for field in string_fields:
            if not isinstance(episode.get(field), str) or not episode[field]:
                raise SummaryError(
                    f"static_plan.episodes[{position}].{field} must be a non-empty string"
                )
        if episode["task_cli"] != task_cli:
            raise SummaryError(
                f"static_plan.episodes[{position}].task_cli does not match task"
            )

        seed = _integer(
            episode.get("seed"), label=f"static_plan.episodes[{position}].seed"
        )
        repeat_idx = _integer(
            episode.get("repeat_idx"),
            label=f"static_plan.episodes[{position}].repeat_idx",
        )
        episode_seed = _integer(
            episode.get("episode_seed"),
            label=f"static_plan.episodes[{position}].episode_seed",
        )
        episode_index = _integer(
            episode.get("episode_index"),
            label=f"static_plan.episodes[{position}].episode_index",
        )
        if repeat_idx < 0 or episode_seed < 0 or episode_index != position:
            raise SummaryError(
                f"static_plan.episodes[{position}] has invalid repeat/episode indices"
            )

        identity_payload = {
            key: episode[key]
            for key in (
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
        }
        result_identity = str(episode["result_identity"])
        if (
            re.fullmatch(r"[0-9a-f]{64}", result_identity) is None
            or result_identity
            != _canonical_sha256(
                identity_payload,
                label=f"static_plan.episodes[{position}] identity",
            )
        ):
            raise SummaryError(
                f"static_plan.episodes[{position}].result_identity is invalid"
            )
        filename = (
            f"seed-{seed:010d}-repeat-{repeat_idx:03d}-"
            f"episode-{episode_index:05d}.json"
        )
        if filename in episodes or result_identity in result_identities:
            raise SummaryError("static_plan.episodes are not uniquely identified")
        result_identities.add(result_identity)
        episodes[filename] = ExpectedEpisode(
            filename=filename,
            result_identity=result_identity,
            job=dict(episode),
            task=str(episode["isaac_task_name"]),
            seed=seed,
            repeat_idx=repeat_idx,
            episode_seed=episode_seed,
            episode_index=episode_index,
        )
        seed_key = str(seed)
        seed_counts[seed_key] = seed_counts.get(seed_key, 0) + 1
    return episodes, seed_counts


def _latest_attempt(contract_dir: Path, contract_hash: str) -> Path | None:
    attempts_dir = contract_dir / "attempts"
    if not attempts_dir.is_dir():
        return None
    try:
        candidates = [
            (path.lstat().st_mtime_ns, path.name, path)
            for path in attempts_dir.iterdir()
            if _ATTEMPT_ID_RE.fullmatch(path.name) is not None
        ]
    except OSError as exc:
        raise SummaryError(f"cannot list attempts: {exc}") from exc
    if not candidates:
        return None
    try:
        candidate = max(candidates)[2]
    except OSError as exc:
        raise SummaryError(f"cannot stat attempt directory: {exc}") from exc

    try:
        entry_stat = candidate.lstat()
        if entry_stat.st_uid != os.getuid():
            raise SummaryError(f"latest attempt is not owned by current user: {candidate.name}")
        if stat.S_ISLNK(entry_stat.st_mode):
            target = os.readlink(candidate)
            target_path = Path(target)
            pattern = re.compile(
                rf"^\.{re.escape(candidate.name)}\.object\.[0-9]+\.[0-9]+\.[a-f0-9]{{16}}$"
            )
            if target_path.is_absolute() or len(target_path.parts) != 1 or pattern.fullmatch(target) is None:
                raise SummaryError(f"latest attempt symlink target is unsafe: {candidate.name}")
            backing = attempts_dir / target
            backing_stat = backing.lstat()
            if (
                backing_stat.st_uid != os.getuid()
                or stat.S_ISLNK(backing_stat.st_mode)
                or not stat.S_ISDIR(backing_stat.st_mode)
                or backing.resolve(strict=True).parent != attempts_dir.resolve(strict=True)
            ):
                raise SummaryError(f"latest attempt backing is unsafe: {candidate.name}")
        elif not stat.S_ISDIR(entry_stat.st_mode):
            raise SummaryError(f"latest attempt must be a directory or safe symlink: {candidate.name}")
    except SummaryError:
        raise
    except (OSError, RuntimeError) as exc:
        raise SummaryError(f"latest attempt entry is invalid: {candidate.name}: {exc}") from exc

    manifest_path = candidate / "manifest.json"
    if not os.path.lexists(manifest_path):
        raise SummaryError(f"latest attempt manifest is missing: {candidate.name}")
    _require_owned_attempt_path(
        candidate,
        manifest_path,
        label="latest attempt manifest",
        directory=False,
    )
    manifest = _json_object(manifest_path, label="latest attempt manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SummaryError("attempt manifest schema_version is invalid")
    if manifest.get("contract_hash") != contract_hash:
        raise SummaryError(f"attempt contract drift detected: {candidate.name}")
    if manifest.get("run_id") != candidate.name:
        raise SummaryError("attempt manifest run_id does not match its directory")
    state = manifest.get("state")
    if state in {"unsafe_cleanup", "failed", "interrupted"}:
        raise SummaryError(f"latest attempt is not running safely: state={state!r}")
    if state not in {"starting", "running", "success", "completed_with_failures"}:
        raise SummaryError(f"latest attempt manifest state is invalid: {state!r}")
    return candidate


def _completion_identity(episode: ExpectedEpisode) -> dict[str, object]:
    return {
        "task": episode.task,
        "seed": episode.seed,
        "repeat_idx": episode.repeat_idx,
        "episode_seed": episode.episode_seed,
        "episode_index": episode.episode_index,
    }


def _validate_running_completion(
    attempt: Path,
    path: Path,
    episode: ExpectedEpisode,
) -> bool:
    completion = _json_object(path, label=f"completion {path.name}")
    if completion.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        raise SummaryError(f"completion {path.name} schema_version is invalid")
    if completion.get("run_id") != attempt.name:
        raise SummaryError(f"completion {path.name} run_id is invalid")
    worker_id = completion.get("worker_id")
    if not isinstance(worker_id, str) or _WORKER_RE.fullmatch(worker_id) is None:
        raise SummaryError(f"completion {path.name} worker_id is invalid")
    if (
        "result_identity" in completion
        and completion.get("result_identity") != episode.result_identity
    ):
        raise SummaryError(f"completion {path.name} result identity is invalid")

    expected_identity = _completion_identity(episode)
    identity = _mapping(completion.get("identity"), label=f"completion {path.name}.identity")
    if identity != expected_identity:
        raise SummaryError(f"completion {path.name} identity is invalid")
    payload = _mapping(completion.get("payload"), label=f"completion {path.name}.payload")
    if any(payload.get(key) != value for key, value in expected_identity.items()):
        raise SummaryError(f"completion {path.name} payload identity is invalid")
    if completion.get("payload_sha256") != _canonical_sha256(
        payload, label=f"completion {path.name}.payload"
    ):
        raise SummaryError(f"completion {path.name} payload_sha256 is invalid")

    success = payload.get("success")
    failure_reason = str(payload.get("failure_reason", ""))
    error = str(payload.get("error", "") or "")
    if not isinstance(success, bool):
        raise SummaryError(f"completion {path.name} outcome is invalid")
    if error:
        raise SummaryError(f"completion {path.name} contains an execution error")
    if failure_reason == _EXECUTION_FAILED:
        if success:
            raise SummaryError(f"completion {path.name} execution outcome is invalid")
        execution_failed = True
    elif (
        failure_reason not in _LEGAL_TASK_OUTCOMES
        or success != (failure_reason == "success")
    ):
        raise SummaryError(f"completion {path.name} task outcome is invalid")
    else:
        execution_failed = False

    raw_path_value = completion.get("raw_result_path")
    if not isinstance(raw_path_value, str) or not raw_path_value:
        raise SummaryError(f"completion {path.name} raw_result_path is invalid")
    try:
        raw_path = Path(raw_path_value).expanduser().resolve()
        attempt_root = attempt.resolve()
    except (OSError, RuntimeError) as exc:
        raise SummaryError(
            f"completion {path.name} raw_result_path is invalid: {exc}"
        ) from exc
    if not _is_relative_to(raw_path, attempt_root) or not raw_path.is_file():
        raise SummaryError(f"completion {path.name} raw_result_path is invalid")
    if completion.get("raw_result_sha256") != _sha256_file(raw_path):
        raise SummaryError(f"completion {path.name} raw_result_sha256 is invalid")
    raw_payload = _json_object(raw_path, label=f"completion {path.name} raw result")
    if raw_payload != payload:
        raise SummaryError(
            f"completion {path.name} raw result content does not match payload"
        )
    return execution_failed


def _running_metrics(
    attempt: Path, episodes: dict[str, ExpectedEpisode]
) -> tuple[str, str]:
    completions = attempt / "completions"
    completed = 0
    execution_failed = 0
    try:
        if os.path.lexists(completions):
            _require_owned_attempt_path(
                attempt,
                completions,
                label="completions directory",
                directory=True,
            )
            paths = completions.iterdir()
        else:
            paths = ()
        for path in sorted(paths, key=lambda candidate: candidate.name):
            if path.suffix != ".json":
                continue
            _require_owned_attempt_path(
                attempt,
                path,
                label=f"completion {path.name}",
                directory=False,
            )
            episode = episodes.get(path.name)
            if episode is None:
                raise SummaryError(f"extra completion does not match contract: {path.name}")
            completed += 1
            if _validate_running_completion(attempt, path, episode):
                execution_failed += 1
    except OSError as exc:
        raise SummaryError(f"cannot count completions: {exc}") from exc
    return f"{completed}/{len(episodes)}", str(execution_failed)


def _running_progress(
    attempt: Path, episodes: dict[str, ExpectedEpisode]
) -> str:
    return _running_metrics(attempt, episodes)[0]


def _terminal_payload_classification(
    payload: dict[str, Any], episode: ExpectedEpisode, *, label: str
) -> str:
    expected_identity = _completion_identity(episode)
    actual_identity = {key: payload.get(key) for key in expected_identity}
    if actual_identity != expected_identity or any(
        type(actual_identity[key]) is not type(expected_value)
        for key, expected_value in expected_identity.items()
    ):
        raise SummaryError(f"{label} payload identity is invalid")
    success = payload.get("success")
    failure_reason = payload.get("failure_reason")
    error = payload.get("error", "")
    if type(success) is not bool or not isinstance(failure_reason, str):
        raise SummaryError(f"{label} outcome is invalid")
    if not isinstance(error, str) or error:
        raise SummaryError(f"{label} contains an execution error")
    if failure_reason == _EXECUTION_FAILED and success is False:
        return "execution_failed"
    if (
        failure_reason in _LEGAL_TASK_OUTCOMES
        and success is (failure_reason == "success")
    ):
        return "task_complete"
    raise SummaryError(f"{label} outcome is invalid")


def _validate_terminal_episodes(
    attempt: Path,
    episodes: dict[str, ExpectedEpisode],
    summary: dict[str, Any],
) -> None:
    episodes_root = attempt / "episodes"
    if not os.path.lexists(episodes_root):
        raise SummaryError("canonical episodes directory is missing")
    _require_owned_attempt_path(
        attempt,
        episodes_root,
        label="canonical episodes directory",
        directory=True,
    )
    try:
        json_paths = {
            path.name: path
            for path in episodes_root.iterdir()
            if path.suffix == ".json"
        }
    except OSError as exc:
        raise SummaryError(f"cannot list canonical episodes: {exc}") from exc
    if set(json_paths) != set(episodes):
        missing = sorted(set(episodes) - set(json_paths))
        extra = sorted(set(json_paths) - set(episodes))
        raise SummaryError(
            f"canonical episodes do not match contract: missing={missing}, extra={extra}"
        )

    successes = 0
    execution_failed = 0
    by_seed: dict[str, dict[str, int]] = {}
    for filename, expected in episodes.items():
        path = json_paths[filename]
        _require_owned_attempt_path(
            attempt,
            path,
            label=f"canonical episode {filename}",
            directory=False,
        )
        canonical = _json_object(path, label=f"canonical episode {filename}")
        if (
            canonical.get("schema_version") != EPISODE_SCHEMA_VERSION
            or canonical.get("result_identity") != expected.result_identity
            or canonical.get("job") != expected.job
        ):
            raise SummaryError(f"canonical episode {filename} identity is invalid")
        worker_id = canonical.get("worker_id")
        if not isinstance(worker_id, str) or _WORKER_RE.fullmatch(worker_id) is None:
            raise SummaryError(f"canonical episode {filename} worker_id is invalid")
        payload = _mapping(
            canonical.get("payload"), label=f"canonical episode {filename}.payload"
        )
        classification = _terminal_payload_classification(
            payload, expected, label=f"canonical episode {filename}"
        )
        stored_classification = canonical.get("classification")
        if (
            stored_classification is not None
            and stored_classification != classification
        ):
            raise SummaryError(
                f"canonical episode {filename} classification is invalid"
            )
        artifacts = _mapping(
            canonical.get("artifacts"),
            label=f"canonical episode {filename}.artifacts",
        )
        video_recorded = payload.get("video_recorded")
        if type(video_recorded) is not bool:
            raise SummaryError(
                f"canonical episode {filename} video_recorded is invalid"
            )
        if video_recorded:
            video = _mapping(
                artifacts.get("video"),
                label=f"canonical episode {filename}.artifacts.video",
            )
            video_path_value = video.get("path")
            if not isinstance(video_path_value, str) or not video_path_value:
                raise SummaryError(
                    f"canonical episode {filename} video path is invalid"
                )
            try:
                video_path = Path(video_path_value).expanduser().resolve()
                attempt_root = attempt.resolve()
            except (OSError, RuntimeError) as exc:
                raise SummaryError(
                    f"canonical episode {filename} video path is invalid: {exc}"
                ) from exc
            if (
                not _is_relative_to(video_path, attempt_root)
                or not video_path.is_file()
                or video.get("sha256") != _sha256_file(video_path)
            ):
                raise SummaryError(
                    f"canonical episode {filename} video artifact is invalid"
                )

        succeeded = payload["success"] is True
        successes += int(succeeded)
        execution_failed += int(classification == "execution_failed")
        seed_counts = by_seed.setdefault(
            str(expected.seed), {"completed": 0, "success": 0}
        )
        seed_counts["completed"] += 1
        seed_counts["success"] += int(succeeded)

    expected_count = len(episodes)
    if (
        summary.get("task_success") != successes
        or summary.get("task_failure") != expected_count - successes
        or summary.get("execution_failed", 0) != execution_failed
        or summary.get("success_rate")
        != {"numerator": successes, "denominator": expected_count}
        or summary.get("by_seed") != by_seed
    ):
        raise SummaryError("terminal summary does not match canonical episodes")


def _seed_sort_key(item: tuple[str, Any]) -> tuple[int, int | str]:
    seed = item[0]
    try:
        return 0, int(seed)
    except ValueError:
        return 1, seed


def _done_metrics(
    summary: dict[str, Any],
    *,
    contract_hash: str,
    contract_expected: int,
    expected_seed_counts: dict[str, int] | None = None,
) -> tuple[str, str, str]:
    if summary.get("contract_hash") != contract_hash:
        raise SummaryError("summary contract_hash does not match contract.json")
    success_rate = _mapping(summary.get("success_rate"), label="success_rate")
    numerator = _integer(success_rate.get("numerator"), label="success_rate.numerator")
    denominator = _integer(
        success_rate.get("denominator"), label="success_rate.denominator"
    )
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise SummaryError("success_rate must satisfy 0 <= numerator <= denominator")

    by_seed = _mapping(summary.get("by_seed"), label="by_seed")
    if not by_seed:
        raise SummaryError("by_seed must not be empty")
    if expected_seed_counts is not None and set(by_seed) != set(expected_seed_counts):
        raise SummaryError("by_seed keys do not match contract episodes")
    rates: list[float] = []
    seed_completed = 0
    seed_successes = 0
    for seed, raw_counts in sorted(by_seed.items(), key=_seed_sort_key):
        counts = _mapping(raw_counts, label=f"by_seed[{seed!r}]")
        completed = _integer(counts.get("completed"), label=f"by_seed[{seed!r}].completed")
        successes = _integer(counts.get("success"), label=f"by_seed[{seed!r}].success")
        if (
            expected_seed_counts is not None
            and completed != expected_seed_counts[seed]
        ):
            raise SummaryError(
                f"by_seed[{seed!r}].completed does not match contract episodes"
            )
        if completed <= 0 or successes < 0 or successes > completed:
            raise SummaryError(
                f"by_seed[{seed!r}] must satisfy 0 <= success <= completed"
            )
        rates.append(successes / completed)
        seed_completed += completed
        seed_successes += successes

    completed = _integer(summary.get("completed"), label="completed")
    expected = _integer(summary.get("expected"), label="expected")
    if not (
        completed
        == expected
        == contract_expected
        == denominator
        == seed_completed
    ):
        raise SummaryError(
            "completed, expected, contract episodes, denominator, and seed totals must match"
        )
    if seed_successes != numerator:
        raise SummaryError("seed success total does not match success_rate.numerator")

    rate = numerator / denominator
    stddev = statistics.pstdev(rates)
    success = f"{rate * 100:.2f}% ± {stddev * 100:.2f}%"
    seed_rates = ",".join(f"{seed_rate * 100:.1f}%" for seed_rate in rates)
    return f"{completed}/{expected}", success, seed_rates


def _validate_terminal_outcomes(
    summary: dict[str, Any], *, contract_expected: int
) -> int:
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise SummaryError("summary schema_version is invalid")
    success_rate = _mapping(summary.get("success_rate"), label="success_rate")
    numerator = _integer(success_rate.get("numerator"), label="success_rate.numerator")
    denominator = _integer(
        success_rate.get("denominator"), label="success_rate.denominator"
    )
    task_success = _integer(summary.get("task_success"), label="task_success")
    task_failure = _integer(summary.get("task_failure"), label="task_failure")
    if task_success != numerator or task_success + task_failure != denominator:
        raise SummaryError(
            "task_success and task_failure must match success_rate totals"
        )
    infrastructure_failure = _integer(
        summary.get("infrastructure_failure"), label="infrastructure_failure"
    )
    if infrastructure_failure != 0:
        raise SummaryError("infrastructure_failure must equal 0")
    if summary.get("infrastructure_errors") != []:
        raise SummaryError("infrastructure_errors must be empty")
    reused = _integer(summary.get("resume_reused"), label="resume_reused")
    if reused < 0 or reused > contract_expected:
        raise SummaryError("resume_reused must be within contract episode totals")
    return reused


def _terminal_execution_failed(
    summary: dict[str, Any], *, terminal_status: str
) -> int:
    if "execution_failed" in summary:
        execution_failed = _integer(
            summary["execution_failed"], label="execution_failed"
        )
        if execution_failed < 0:
            raise SummaryError("execution_failed must be non-negative")
    elif terminal_status == DONE:
        execution_failed = 0
    else:
        raise SummaryError(
            "DONE_WITH_FAILURES summary must contain execution_failed"
        )

    task_failure = _integer(summary.get("task_failure"), label="task_failure")
    if execution_failed > task_failure:
        raise SummaryError("execution_failed must not exceed task_failure")
    if terminal_status == DONE and execution_failed != 0:
        raise SummaryError("DONE summary execution_failed must equal 0")
    if terminal_status == DONE_WITH_FAILURES and execution_failed == 0:
        raise SummaryError(
            "DONE_WITH_FAILURES summary execution_failed must be positive"
        )
    return execution_failed


def _validate_terminal_marker(marker_path: Path, summary: dict[str, Any]) -> None:
    marker_name = marker_path.name
    try:
        marker = marker_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise SummaryError(f"cannot read {marker_name}: {exc}") from exc
    expected = _canonical_sha256(summary, label="summary.json")
    if marker != expected:
        raise SummaryError(f"{marker_name} digest does not match summary.json")


def _validate_success_marker(marker_path: Path, summary: dict[str, Any]) -> None:
    _validate_terminal_marker(marker_path, summary)


def _validate_terminal_manifest(
    attempt: Path,
    *,
    marker_name: str,
    expected_state: str,
    contract_hash: str,
    contract_expected: int,
    summary_reused: int,
    summary_execution_failed: int,
) -> None:
    manifest_path = _require_owned_attempt_path(
        attempt,
        attempt / "manifest.json",
        label="attempt manifest",
        directory=False,
    )
    manifest = _json_object(manifest_path, label="attempt manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SummaryError("attempt manifest schema_version is invalid")
    if manifest.get("contract_hash") != contract_hash:
        raise SummaryError("attempt manifest contract_hash does not match contract")
    if manifest.get("run_id") != attempt.name:
        raise SummaryError("attempt manifest run_id does not match attempt directory")
    state = manifest.get("state")
    if state != expected_state:
        raise SummaryError(
            f"{marker_name} requires attempt manifest state {expected_state!r}, "
            f"got {state!r}"
        )
    runtime = _mapping(manifest.get("runtime"), label="attempt manifest.runtime")
    runtime_completed = _integer(
        runtime.get("completed"), label="manifest.runtime.completed"
    )
    if runtime_completed != contract_expected:
        raise SummaryError(
            "manifest.runtime.completed does not match contract episodes"
        )
    runtime_reused = _integer(
        runtime.get("reused"), label="manifest.runtime.reused"
    )
    if runtime_reused != summary_reused:
        raise SummaryError("manifest.runtime.reused does not match summary")
    if "execution_failed" in runtime:
        runtime_execution_failed = _integer(
            runtime["execution_failed"],
            label="manifest.runtime.execution_failed",
        )
        if runtime_execution_failed < 0:
            raise SummaryError(
                "manifest.runtime.execution_failed must be non-negative"
            )
    elif expected_state == "success":
        runtime_execution_failed = 0
    else:
        raise SummaryError(
            "failed terminal manifest.runtime must contain execution_failed"
        )
    if runtime_execution_failed != summary_execution_failed:
        raise SummaryError(
            "manifest.runtime.execution_failed does not match summary"
        )


def _error_identity(
    contract_path: Path, eval_root: Path
) -> tuple[str, int, str | None]:
    checkpoint_relative_path: str | None = None
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        static_plan = contract.get("static_plan") if isinstance(contract, dict) else None
        checkpoint = (
            static_plan.get("checkpoint") if isinstance(static_plan, dict) else None
        )
        candidate = (
            checkpoint.get("relative_path") if isinstance(checkpoint, dict) else None
        )
        if isinstance(candidate, str) and candidate:
            checkpoint_relative_path = candidate
    except (OSError, UnicodeError, json.JSONDecodeError):
        pass
    try:
        relative_parts = contract_path.relative_to(eval_root).parts
    except ValueError:
        relative_parts = contract_path.parts
    task = relative_parts[0] if relative_parts else "?"
    match = re.search(r"step_(\d+)", checkpoint_relative_path or str(contract_path))
    return task, int(match.group(1)) if match else -1, checkpoint_relative_path


def _row_from_contract(contract_path: Path, eval_root: Path) -> EvalRow:
    try:
        contract = _json_object(contract_path, label="contract.json")
        (
            task, architecture, mode, run_id, step, expected, checkpoint_relative_path
        ) = _checkpoint_identity(contract)
        episodes, expected_seed_counts = _contract_episodes(contract)
        if len(episodes) != expected:
            raise SummaryError("contract episode count is inconsistent")
        contract_hash = _canonical_sha256(contract, label="contract.json")
        attempt = _latest_attempt(contract_path.parent, contract_hash)
        if attempt is None:
            return EvalRow(
                task,
                architecture,
                mode,
                run_id,
                step,
                PENDING,
                f"0/{expected}",
                "-",
                "-",
                checkpoint_relative_path=checkpoint_relative_path,
            )

        summary_path = attempt / "summary.json"
        terminals = (
            (attempt / SUCCESS_MARKER, DONE, "success"),
            (
                attempt / COMPLETED_WITH_FAILURES_MARKER,
                DONE_WITH_FAILURES,
                "completed_with_failures",
            ),
        )
        present_terminals = [
            terminal for terminal in terminals if os.path.lexists(terminal[0])
        ]
        if len(present_terminals) > 1:
            raise SummaryError(
                f"{SUCCESS_MARKER} and {COMPLETED_WITH_FAILURES_MARKER} both exist"
            )
        if present_terminals:
            marker_path, terminal_status, manifest_state = present_terminals[0]
            _require_owned_attempt_path(
                attempt,
                marker_path,
                label=marker_path.name,
                directory=False,
            )
            if not os.path.lexists(summary_path):
                raise SummaryError(
                    f"{marker_path.name} exists but summary.json is missing"
                )
            _require_owned_attempt_path(
                attempt,
                summary_path,
                label="summary.json",
                directory=False,
            )
            summary = _json_object(summary_path, label="summary.json")
            _validate_terminal_marker(marker_path, summary)
            progress, success, seed_rates = _done_metrics(
                summary,
                contract_hash=contract_hash,
                contract_expected=expected,
                expected_seed_counts=expected_seed_counts,
            )
            summary_reused = _validate_terminal_outcomes(
                summary, contract_expected=expected
            )
            execution_failed = _terminal_execution_failed(
                summary, terminal_status=terminal_status
            )
            _validate_terminal_manifest(
                attempt,
                marker_name=marker_path.name,
                expected_state=manifest_state,
                contract_hash=contract_hash,
                contract_expected=expected,
                summary_reused=summary_reused,
                summary_execution_failed=execution_failed,
            )
            _validate_terminal_episodes(attempt, episodes, summary)
            return EvalRow(
                task,
                architecture,
                mode,
                run_id,
                step,
                terminal_status,
                progress,
                success,
                seed_rates,
                execution_failed=str(execution_failed),
                checkpoint_relative_path=checkpoint_relative_path,
            )
        progress, execution_failed = _running_metrics(attempt, episodes)
        return EvalRow(
            task,
            architecture,
            mode,
            run_id,
            step,
            RUNNING,
            progress,
            "-",
            "-",
            execution_failed=execution_failed,
            checkpoint_relative_path=checkpoint_relative_path,
        )
    except (SummaryError, OSError) as exc:
        task, step, checkpoint_relative_path = _error_identity(contract_path, eval_root)
        return EvalRow(
            task,
            "?",
            "?",
            "?",
            step,
            ERROR,
            "-",
            "-",
            "-",
            str(exc),
            execution_failed="-",
            checkpoint_relative_path=checkpoint_relative_path,
        )


def _selector_module() -> Any:
    module_name = (
        f"{__package__}.shard_checkpoints"
        if __package__
        else "shard_checkpoints"
    )
    return importlib.import_module(module_name)


def _strict_checkpoint_relative_path(value: object, selector: Any) -> str:
    if not isinstance(value, str) or not value:
        raise SummaryError("checkpoint.relative_path must be a non-empty string")
    path = Path(value)
    parts = path.parts
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in parts)
        or len(parts) != 8
        or parts[0] != ".runs"
        or parts[1] not in selector.ALLOWED_TRAINING_ROOT_NAMES
        or parts[2] != selector.EXPECTED_CHECKPOINT_TASK
        or parts[4] != "full"
        or parts[5] != "stage3"
        or selector._SAFE_COMPONENT_RE.fullmatch(parts[6]) is None
    ):
        raise SummaryError(
            "strict checkpoint.relative_path must be the canonical optimized "
            "HOI_pp_box/<architecture>/full/stage3/<run_id>/step_N path"
        )
    architecture = parts[3]
    expected_steps = dict(selector.EXPECTED_RUN_MATRIX)
    step_match = selector.STEP_RE.fullmatch(parts[7])
    if (
        architecture not in expected_steps
        or step_match is None
        or int(step_match.group(1)) not in expected_steps[architecture]
    ):
        raise SummaryError(
            "strict checkpoint.relative_path architecture/step is outside the "
            f"frozen {selector.EXPECTED_CHECKPOINT_COUNT}-checkpoint matrix"
        )
    return value


def _strict_identity_row(relative_path: str, *, error: str) -> EvalRow:
    parts = Path(relative_path).parts
    step_match = re.fullmatch(r"step_(\d+)", parts[-1])
    return EvalRow(
        task="pp_box",
        architecture=parts[3],
        mode=parts[4],
        run_id=parts[6],
        step=int(step_match.group(1)) if step_match else -1,
        status=ERROR,
        progress="-",
        success="-",
        seed_rates="-",
        error=error,
        execution_failed="-",
        checkpoint_relative_path=relative_path,
    )


def _strict_contract_groups(
    eval_root: Path, selector: Any
) -> tuple[dict[str, list[Path]], list[EvalRow]]:
    groups: dict[str, list[Path]] = {}
    diagnostics: dict[str, EvalRow] = {}
    seen_paths: set[Path] = set()
    for raw_path in sorted(eval_root.glob("**/contract.json")):
        safe_contract_path: Path | None = None
        try:
            contract_path = raw_path.resolve(strict=True)
            if not _is_relative_to(contract_path, eval_root):
                raise SummaryError(f"evaluation contract escapes {eval_root}: {raw_path}")
            safe_contract_path = contract_path
            if contract_path in seen_paths:
                continue
            seen_paths.add(contract_path)
            contract = _json_object(contract_path, label="contract.json")
            static_plan = _mapping(contract.get("static_plan"), label="static_plan")
            checkpoint = _mapping(
                static_plan.get("checkpoint"), label="static_plan.checkpoint"
            )
            relative_path = _strict_checkpoint_relative_path(
                checkpoint.get("relative_path"), selector
            )
        except (OSError, SummaryError) as exc:
            row = (
                _row_from_contract(safe_contract_path, eval_root)
                if safe_contract_path is not None
                else EvalRow(
                    task="?",
                    architecture="?",
                    mode="?",
                    run_id="?",
                    step=-1,
                    status=ERROR,
                    progress="-",
                    success="-",
                    seed_rates="-",
                    execution_failed="-",
                )
            )
            error = f"strict contract discovery rejected {raw_path}: {exc}"
            key = row.checkpoint_relative_path or f"@contract:{raw_path}"
            existing = diagnostics.get(key)
            if existing is not None:
                error = f"{existing.error}; {error}"
                row = existing
            diagnostics[key] = replace(
                row,
                status=ERROR,
                error=error,
                accepted_contract_path=None,
                accepted_contract_hash=None,
            )
            continue
        groups.setdefault(relative_path, []).append(contract_path)
    return groups, list(diagnostics.values())


def _strict_current_contracts(
    selector: Any,
    contract_paths: list[Path],
    *,
    eval_root: Path,
    repo_root: Path,
    relative_path: str,
) -> list[tuple[Path, str, dict[str, Any]]]:
    current: list[tuple[Path, str, dict[str, Any]]] = []
    for contract_path in contract_paths:
        contract = selector._json_object(contract_path)
        if contract is None:
            continue
        try:
            if not selector._matches_full_pp_box_contract(
                contract, relative_path, repo_root=repo_root
            ):
                continue
            static_plan = contract["static_plan"]
            policy = static_plan["policy"]
            if Path(policy["output_root"]).expanduser().resolve() != eval_root:
                continue
            contract_hash = selector._canonical_sha256(contract)
            expected_label = (
                f"{selector.EXPECTED_CHECKPOINT_TASK}-{Path(relative_path).name}"
            )
            if (
                contract_path.parent.name != contract_hash[:16]
                or contract_path.parent.parent.name != expected_label
                or contract_path.parent.parent.parent.name != "pp_box"
                or selector._json_object(
                    contract_path.parent / "source_inventory.json"
                )
                != static_plan.get("source_inventory")
            ):
                continue
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            continue
        current.append((contract_path, contract_hash, contract))
    return sorted(current, key=lambda item: (item[1], str(item[0])))


def _strict_success_candidates(
    selector: Any,
    current_contracts: list[tuple[Path, str, dict[str, Any]]],
    *,
    eval_root: Path,
    repo_root: Path,
    relative_path: str,
) -> list[tuple[Path, str, Path]]:
    successes: list[tuple[Path, str, Path]] = []
    for contract_path, contract_hash, contract in current_contracts:
        try:
            attempts_root = (contract_path.parent / "attempts").resolve()
            if (
                not attempts_root.is_dir()
                or not _is_relative_to(attempts_root, eval_root)
            ):
                continue
            attempt = selector._select_resumable_attempt(
                attempts_root, contract_hash
            )
            if attempt is None:
                continue
            manifest = selector._json_object(attempt / "manifest.json")
            present_markers = [
                marker_name
                for marker_name in selector.TERMINAL_MARKER_STATES
                if os.path.lexists(attempt / marker_name)
            ]
            marker = attempt / SUCCESS_MARKER
            if (
                manifest is None
                or manifest.get("state") != "success"
                or present_markers != [SUCCESS_MARKER]
                or not selector._is_owned_attempt_path(
                    attempt, marker, directory=False
                )
                or not selector._valid_terminal_attempt(
                    contract_path.parent,
                    attempt,
                    marker,
                    contract,
                    relative_path,
                    repo_root=repo_root,
                    contract_prevalidated=True,
                )
            ):
                continue
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        successes.append((contract_path, contract_hash, attempt.resolve()))
    return sorted(successes, key=lambda item: (item[1], str(item[0])))


def _strict_error_from_base(
    base: EvalRow, *, relative_path: str, error: str
) -> EvalRow:
    detail = error if base.error is None else f"{error}; row_error={base.error}"
    return replace(
        base,
        status=ERROR,
        error=detail,
        checkpoint_relative_path=relative_path,
        accepted_contract_path=None,
        accepted_contract_hash=None,
    )


def _collect_strict_rows(eval_root: Path, repo_root: Path) -> list[EvalRow]:
    selector = _selector_module()
    groups, rows = _strict_contract_groups(eval_root, selector)
    if not groups:
        return _sort_rows(rows)
    try:
        statuses = selector.checkpoint_contract_statuses(
            eval_root,
            set(groups),
            allowed_root=repo_root / ".runs",
            repo_root=repo_root,
        )
    except selector.ShardConfigError as exc:
        rows.extend(
            _strict_identity_row(
                relative_path,
                error=f"strict current-source validation failed: {exc}",
            )
            for relative_path in sorted(groups)
        )
        return _sort_rows(rows)

    for relative_path, contract_paths in sorted(groups.items()):
        status = statuses[relative_path]
        current_contracts = _strict_current_contracts(
            selector,
            contract_paths,
            eval_root=eval_root,
            repo_root=repo_root,
            relative_path=relative_path,
        )
        successes = _strict_success_candidates(
            selector,
            current_contracts,
            eval_root=eval_root,
            repo_root=repo_root,
            relative_path=relative_path,
        )
        if status.state == "completed":
            accepted = [
                candidate
                for candidate in successes
                if status.contract_path == str(candidate[0])
                and status.contract_hash == candidate[1]
                and status.attempt_path == str(candidate[2])
            ]
            if len(accepted) != 1:
                rows.append(
                    _strict_identity_row(
                        relative_path,
                        error=(
                            "selector accepted provenance does not uniquely match "
                            "a legal current-source `_SUCCESS` contract"
                        ),
                    )
                )
                continue
            contract_path, contract_hash, _attempt = accepted[0]
            if re.fullmatch(r"[0-9a-f]{64}", contract_hash) is None:
                rows.append(
                    _strict_identity_row(
                        relative_path,
                        error="selector accepted provenance disagrees with strict scan",
                    )
                )
                continue
            row = _row_from_contract(contract_path, eval_root)
            if row.status != DONE or row.execution_failed != "0":
                rows.append(
                    _strict_error_from_base(
                        row,
                        relative_path=relative_path,
                        error="accepted `_SUCCESS` did not produce a clean DONE row",
                    )
                )
                continue
            rows.append(
                replace(
                    row,
                    checkpoint_relative_path=relative_path,
                    accepted_contract_path=str(contract_path),
                    accepted_contract_hash=contract_hash,
                )
            )
            continue

        if successes:
            rows.append(
                _strict_identity_row(
                    relative_path,
                    error=(
                        "strict scan found `_SUCCESS` but selector did not report "
                        "the checkpoint completed"
                    ),
                )
            )
            continue

        if status.contract_path is not None:
            base = _row_from_contract(Path(status.contract_path), eval_root)
            rows.append(
                replace(
                    base,
                    checkpoint_relative_path=relative_path,
                    accepted_contract_path=None,
                    accepted_contract_hash=None,
                )
            )
            continue

        if len(current_contracts) == 1:
            contract_path, _contract_hash, _contract = current_contracts[0]
            base = _row_from_contract(contract_path, eval_root)
            if base.status == DONE:
                rows.append(
                    _strict_error_from_base(
                        base,
                        relative_path=relative_path,
                        error=(
                            "current-source row is DONE but no legal `_SUCCESS` "
                            "was accepted"
                        ),
                    )
                )
            else:
                rows.append(
                    replace(
                        base,
                        checkpoint_relative_path=relative_path,
                        accepted_contract_path=None,
                        accepted_contract_hash=None,
                    )
                )
            continue

        if len(current_contracts) > 1:
            provenance = ", ".join(
                f"{path}#{contract_hash}"
                for path, contract_hash, _contract in current_contracts
            )
            rows.append(
                _strict_identity_row(
                    relative_path,
                    error=(
                        "multiple current-source contracts exist without one "
                        f"accepted `_SUCCESS`: {provenance}"
                    ),
                )
            )
            continue

        latest = max(contract_paths, key=str)
        base = _row_from_contract(latest, eval_root)
        rows.append(
            _strict_error_from_base(
                base,
                relative_path=relative_path,
                error=(
                    "no legal current-source contract is selectable; "
                    f"discovered_contracts={len(contract_paths)}"
                ),
            )
        )

    relative_paths = [
        row.checkpoint_relative_path
        for row in rows
        if row.checkpoint_relative_path is not None
    ]
    if len(relative_paths) != len(set(relative_paths)):
        raise SummaryError(
            "strict summary produced duplicate checkpoint relative paths"
        )
    return _sort_rows(rows)


def _sort_rows(rows: list[EvalRow]) -> list[EvalRow]:
    return sorted(
        rows,
        key=lambda row: (
            row.task,
            row.architecture,
            row.mode,
            row.run_id,
            row.step,
            row.checkpoint_relative_path or "",
        ),
    )


def collect_rows(
    eval_root: Path,
    *,
    strict_current_source: bool = False,
    repo_root: Path | None = None,
) -> list[EvalRow]:
    if strict_current_source:
        resolved_repo = (
            repo_root.expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[3]
        )
        resolved_eval = eval_root.expanduser().resolve()
        return _collect_strict_rows(resolved_eval, resolved_repo)
    rows = [
        _row_from_contract(path, eval_root)
        for path in eval_root.rglob("contract.json")
    ]
    return _sort_rows(rows)


def _table(rows: list[EvalRow]) -> str:
    headers = (
        "TASK",
        "ARCH",
        "MODE",
        "RUN_ID",
        "STEP",
        "STATUS",
        "PROGRESS",
        "EXEC_FAILED",
        "SUCCESS RATE ± STD",
        "CHECKPOINT_RELATIVE_PATH",
        "ACCEPTED_CONTRACT_PATH",
        "ACCEPTED_CONTRACT_HASH",
    )
    values = [
        (
            row.task,
            row.architecture,
            row.mode,
            row.run_id,
            str(row.step) if row.step >= 0 else "?",
            row.status,
            row.progress,
            row.execution_failed,
            row.success,
            row.checkpoint_relative_path or "-",
            row.accepted_contract_path or "-",
            row.accepted_contract_hash or "-",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()

    separator = "  ".join("-" * width for width in widths)
    return "\n".join((format_row(headers), separator, *(format_row(row) for row in values)))


def render(eval_root: Path, rows: list[EvalRow]) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "HumanoidArena evaluation summary",
        f"Updated: {timestamp}",
        f"Root: {eval_root}",
        "Metric: pooled success rate ± population std across seed rates (ddof=0); execution failures count as failures",
        "",
    ]
    if not rows:
        lines.append("No evaluation contracts found.")
        return "\n".join(lines)

    lines.append(_table(rows))
    errors = [row for row in rows if row.error is not None]
    if errors:
        lines.extend(("", "Errors:"))
        for row in errors:
            lines.append(f"- {row.task} step {row.step}: {row.error}")
    return "\n".join(lines)


def render_jsonl(rows: list[EvalRow]) -> str:
    payloads = []
    for row in rows:
        payload = {
            "schema_version": SUMMARY_ROW_SCHEMA_VERSION,
            **asdict(row),
            "relative_path": row.checkpoint_relative_path,
        }
        payloads.append(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        )
    return "\n".join(payloads)


def _strict_rows_complete(rows: list[EvalRow]) -> bool:
    return bool(rows) and all(
        row.status == DONE
        and row.execution_failed == "0"
        and row.accepted_contract_path is not None
        and row.accepted_contract_hash is not None
        and re.fullmatch(r"[0-9a-f]{64}", row.accepted_contract_hash) is not None
        for row in rows
    )


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=repo_root / ".runs/eval/humanoidarena",
        help="HumanoidArena evaluation result root (default: %(default)s)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="repository root used for strict current-source validation",
    )
    parser.add_argument(
        "--strict-current-source",
        "--strict",
        action="store_true",
        help=(
            "group by checkpoint relative path and accept exactly one legal "
            "current-source _SUCCESS contract per discovered checkpoint"
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("table", "jsonl"),
        default="table",
        help="human-readable table or one canonical JSON object per row",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    eval_root = args.eval_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    try:
        rows = (
            collect_rows(
                eval_root,
                strict_current_source=args.strict_current_source,
                repo_root=repo_root,
            )
            if eval_root.is_dir()
            else []
        )
    except (OSError, SummaryError) as exc:
        print(f"[watch_eval_summary] validation error: {exc}", file=sys.stderr)
        return 2
    output = (
        render(eval_root, rows)
        if args.output_format == "table"
        else render_jsonl(rows)
    )
    if output:
        print(output)
    if args.strict_current_source and not _strict_rows_complete(rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

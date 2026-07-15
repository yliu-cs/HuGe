#!/usr/bin/env python3
"""Serial HumanoidArena evaluation over checkpoint directories owned by Huge."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, NamedTuple, Sequence


BATCH_SCHEMA_VERSION = "huge_ha_batch_summary_v1"
STEP_RE = re.compile(r"^step_([0-9]+)$")
TASKS = (
    "boxing",
    "doubledesk",
    "football",
    "open_door",
    "pp_box",
    "sit_sofa",
    "vision_navi",
)
INTERRUPTED_EXIT_CODES = {130, 143, -signal.SIGINT, -signal.SIGTERM}


class BatchConfigError(RuntimeError):
    """Raised when a batch root or discovered checkpoint is unsafe."""


class CheckpointTarget(NamedTuple):
    path: Path
    relative_path: str
    step: int


InvokeCheckpoint = Callable[[CheckpointTarget, Path], int]
_ACTIVE_CHILD: subprocess.Popen[str] | None = None
_STOP_SIGNAL: int | None = None


def _relative_to(path: Path, root: Path, *, label: str) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise BatchConfigError(f"{label} escapes allowed root {root}: {path}") from exc


def discover_checkpoints(root: Path, allowed_root: Path) -> list[CheckpointTarget]:
    """Find step directories below root and order parent paths then numeric steps."""
    allowed = allowed_root.expanduser().resolve()
    candidate_root = root.expanduser().resolve()
    if not candidate_root.is_dir():
        raise BatchConfigError(f"checkpoint batch root is not a directory: {candidate_root}")
    _relative_to(candidate_root, allowed, label="checkpoint batch root")

    targets: list[CheckpointTarget] = []
    resolved_paths: set[Path] = set()
    for lexical_path in candidate_root.rglob("step_*"):
        match = STEP_RE.fullmatch(lexical_path.name)
        if match is None or not lexical_path.is_dir():
            continue
        resolved = lexical_path.resolve()
        _relative_to(resolved, candidate_root, label="checkpoint")
        _relative_to(resolved, allowed, label="checkpoint")
        if resolved in resolved_paths:
            raise BatchConfigError(f"multiple checkpoint paths resolve to the same directory: {resolved}")
        resolved_paths.add(resolved)
        relative = lexical_path.relative_to(candidate_root).as_posix()
        targets.append(CheckpointTarget(resolved, relative, int(match.group(1))))

    targets.sort(key=lambda target: (str(Path(target.relative_path).parent), target.step, target.relative_path))
    if not targets:
        raise BatchConfigError(f"no step_<number> checkpoint directories found under {candidate_root}")
    return targets


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


def _write_summary(report_dir: Path, summary: dict[str, object]) -> None:
    _atomic_write_text(
        report_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    stream = io.StringIO(newline="")
    fieldnames = ("index", "checkpoint", "step", "status", "exit_code", "duration_sec", "log")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary["results"]:  # type: ignore[index]
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    _atomic_write_text(report_dir / "summary.csv", stream.getvalue())


def _safe_log_component(relative_path: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "__", relative_path).strip("._")
    return component or "checkpoint"


def execute_batch(
    *,
    task: str,
    backend: str,
    checkpoint_root: Path,
    targets: Sequence[CheckpointTarget],
    report_dir: Path,
    invoke: InvokeCheckpoint,
    stop_requested: Callable[[], bool] | None = None,
) -> int:
    """Execute every target serially, persisting progress after each child exits."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "logs").mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    results: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "state": "running",
        "task": task,
        "backend": backend,
        "checkpoint_root": str(checkpoint_root),
        "discovered": len(targets),
        "completed": 0,
        "succeeded": 0,
        "failed": 0,
        "started_at": started_at,
        "finished_at": None,
        "duration_sec": 0.0,
        "results": results,
    }
    _write_summary(report_dir, summary)

    interrupted = False
    for index, target in enumerate(targets):
        if stop_requested is not None and stop_requested():
            interrupted = True
            break
        log_path = report_dir / "logs" / f"{index:03d}_{_safe_log_component(target.relative_path)}.log"
        print(f"[batch_eval] [{index + 1}/{len(targets)}] start {target.relative_path}", flush=True)
        child_started = time.monotonic()
        error = ""
        try:
            exit_code = int(invoke(target, log_path))
        except KeyboardInterrupt:
            exit_code = 130
        except Exception as exc:  # Keep evaluating later checkpoints after launcher failures.
            exit_code = 127
            error = f"{type(exc).__name__}: {exc}"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[batch_eval] launcher exception: {error}\n")
        duration = time.monotonic() - child_started
        status = "success" if exit_code == 0 else "failed"
        if exit_code in INTERRUPTED_EXIT_CODES or (stop_requested is not None and stop_requested()):
            status = "interrupted"
            interrupted = True
        row: dict[str, object] = {
            "index": index,
            "checkpoint": target.relative_path,
            "step": target.step,
            "status": status,
            "exit_code": exit_code,
            "duration_sec": round(duration, 6),
            "log": str(log_path),
        }
        if error:
            row["error"] = error
        results.append(row)
        summary["completed"] = len(results)
        summary["succeeded"] = sum(row["exit_code"] == 0 for row in results)
        summary["failed"] = sum(row["exit_code"] != 0 for row in results)
        summary["duration_sec"] = round(time.time() - started_at, 6)
        _write_summary(report_dir, summary)
        print(
            f"[batch_eval] [{index + 1}/{len(targets)}] {status} "
            f"exit={exit_code} duration={duration:.1f}s {target.relative_path}",
            flush=True,
        )
        if interrupted:
            break

    summary["state"] = (
        "interrupted"
        if interrupted
        else "success"
        if int(summary["failed"]) == 0
        else "completed_with_failures"
    )
    summary["finished_at"] = time.time()
    summary["duration_sec"] = round(float(summary["finished_at"]) - started_at, 6)
    _write_summary(report_dir, summary)
    print(
        f"[batch_eval] summary state={summary['state']} completed={summary['completed']}/"
        f"{summary['discovered']} succeeded={summary['succeeded']} failed={summary['failed']} "
        f"path={report_dir / 'summary.json'}",
        flush=True,
    )
    if interrupted:
        return 130
    return 0 if int(summary["failed"]) == 0 else 1


def _request_stop(signum: int, _frame: object) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum
    child = _ACTIVE_CHILD
    if child is not None and child.poll() is None:
        try:
            child.send_signal(signum)
        except ProcessLookupError:
            pass


def _invoke_run_eval(
    target: CheckpointTarget,
    log_path: Path,
    *,
    repo_root: Path,
    run_eval: Path,
    task: str,
    backend: str,
) -> int:
    global _ACTIVE_CHILD
    command = ["bash", str(run_eval), task, str(target.path), backend]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[batch_eval] command=" + " ".join(command) + "\n")
        log.flush()
        child = subprocess.Popen(
            command,
            cwd=repo_root,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        _ACTIVE_CHILD = child
        try:
            assert child.stdout is not None
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            return child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            _ACTIVE_CHILD = None


def _report_directory(repo_root: Path, task: str) -> Path:
    root = repo_root / ".runs/eval/humanoidarena_batch" / task
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4)
        candidate = root / run_id
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise BatchConfigError(f"unable to allocate a unique batch report directory under {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("checkpoint_root")
    parser.add_argument("backend", choices=("sonic",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global _STOP_SIGNAL
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    raw_root = Path(args.checkpoint_root).expanduser()
    checkpoint_root = (repo_root / raw_root).resolve() if not raw_root.is_absolute() else raw_root.resolve()
    allowed_root = repo_root / ".runs/stage3_humanoidarena"
    try:
        targets = discover_checkpoints(checkpoint_root, allowed_root)
        report_dir = _report_directory(repo_root, args.task)
    except BatchConfigError as exc:
        print(f"[batch_eval] configuration error: {exc}", file=sys.stderr)
        return 2

    print(
        f"[batch_eval] discovered={len(targets)} root={checkpoint_root} "
        f"order=parent_path_then_numeric_step report={report_dir}",
        flush=True,
    )
    _STOP_SIGNAL = None
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    def invoke(target: CheckpointTarget, log_path: Path) -> int:
        return _invoke_run_eval(
            target,
            log_path,
            repo_root=repo_root,
            run_eval=repo_root / "scripts/eval/humanoidarena/run_eval.sh",
            task=args.task,
            backend=args.backend,
        )

    return execute_batch(
        task=args.task,
        backend=args.backend,
        checkpoint_root=checkpoint_root,
        targets=targets,
        report_dir=report_dir,
        invoke=invoke,
        stop_requested=lambda: _STOP_SIGNAL is not None,
    )


if __name__ == "__main__":
    raise SystemExit(main())

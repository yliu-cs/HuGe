#!/usr/bin/env python3
"""Incrementally download HumanoidArena's missing vision-navigation task."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path


DATASET_ID = "Twang2026/HumanoidArena_raw"
REVISION = "master"
TASK_NAME = "HSI_vision_navi"
ALLOW_PATTERNS = [f"{TASK_NAME}/**"]
SESSION_NAME = "humanoidarena-vision-navi-download"
RAW_ROOT_RELATIVE = Path("resource/dataset/HumanoidArena_raw")
LOG_RELATIVE = Path(".runs/downloads/humanoidarena_vision_navi.log")


def validate_download(task_dir: Path) -> tuple[int, int]:
    """Return recursive NPZ/MP4 counts and reject an incomplete task."""
    if not task_dir.is_dir():
        raise RuntimeError(f"downloaded task directory is missing: {task_dir}")

    npz_count = sum(1 for path in task_dir.rglob("*.npz") if path.is_file())
    mp4_count = sum(1 for path in task_dir.rglob("*.mp4") if path.is_file())
    if npz_count == 0:
        raise RuntimeError(f"downloaded task contains no NPZ files: {task_dir}")
    return npz_count, mp4_count


def download_task(
    repo_root: Path,
    download_fn: Callable[..., object] | None = None,
) -> tuple[int, int]:
    """Download only HSI_vision_navi and validate the resulting subtree."""
    repo_root = repo_root.resolve()
    raw_root = (repo_root / RAW_ROOT_RELATIVE).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)

    if download_fn is None:
        from modelscope.hub.snapshot_download import dataset_snapshot_download

        download_fn = dataset_snapshot_download

    print(
        f"[{datetime.now().astimezone().isoformat()}] "
        f"downloading {DATASET_ID}@{REVISION} with allow_patterns={ALLOW_PATTERNS}",
        flush=True,
    )
    download_fn(
        dataset_id=DATASET_ID,
        revision=REVISION,
        local_dir=str(raw_root),
        allow_patterns=ALLOW_PATTERNS,
    )

    counts = validate_download(raw_root / TASK_NAME)
    print(
        f"[{datetime.now().astimezone().isoformat()}] "
        f"download complete: task={TASK_NAME} npz={counts[0]} mp4={counts[1]}",
        flush=True,
    )
    return counts


def build_tmux_command(repo_root: Path) -> list[str]:
    """Build the detached tmux command that runs this script as a worker."""
    repo_root = repo_root.resolve()
    python_path = repo_root / ".venv/bin/python"
    script_path = repo_root / "scripts/download_humanoidarena_vision_navi.py"
    log_path = repo_root / LOG_RELATIVE

    worker_argv = [str(python_path), str(script_path), "--worker"]
    pipeline = (
        f"set -o pipefail; {shlex.join(worker_argv)} 2>&1 "
        f"| tee -a {shlex.quote(str(log_path))}"
    )
    shell_command = shlex.join(["bash", "-lc", pipeline])
    return ["tmux", "new-session", "-d", "-s", SESSION_NAME, shell_command]


def launch_tmux(repo_root: Path) -> None:
    """Validate launcher dependencies and start the detached worker."""
    repo_root = repo_root.resolve()
    python_path = repo_root / ".venv/bin/python"
    log_path = repo_root / LOG_RELATIVE

    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is not installed or not available on PATH")
    if not python_path.is_file():
        raise RuntimeError(f"project virtual-environment Python is missing: {python_path}")

    existing = subprocess.run(
        ["tmux", "has-session", "-t", SESSION_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if existing.returncode == 0:
        raise RuntimeError(f"tmux session already exists: {SESSION_NAME}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_tmux_command(repo_root), cwd=repo_root, check=True)

    print(f"started tmux session: {SESSION_NAME}")
    print(f"log: {log_path}")
    print(f"attach: tmux attach -t {SESSION_NAME}")
    print(f"follow log: tail -f {shlex.quote(str(log_path))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    try:
        if args.worker:
            download_task(repo_root)
        else:
            launch_tmux(repo_root)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a CPU-only Real holdout index by exact full-episode content exclusion."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import numpy as np
import polars as pl


MATCHING_FIELDS = ("action.motion_token", "observation.state")
CONTROL_HZ = 50


def _metadata(root: Path) -> tuple[dict[str, Any], dict[int, str]]:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    names = info["features"]["observation.state"]["names"]
    if info["fps"] != CONTROL_HZ or not isinstance(names, list) or len(names) != 43:
        raise ValueError(f"{root}: expected 50 Hz data and 43 named observation.state dimensions")
    tasks: dict[int, str] = {}
    for line in (root / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index, task = int(row["task_index"]), row["task"]
        if index in tasks or not isinstance(task, str) or not task.strip():
            raise ValueError(f"{root}/meta/tasks.jsonl: duplicate index or invalid task: {row}")
        tasks[index] = task
    return info, tasks


def _episode_paths(root: Path) -> list[Path]:
    paths = sorted(
        (root / "data").glob("chunk-*/*.parquet"),
        key=lambda path: int(path.stem.removeprefix("episode_")),
    )
    if not paths:
        raise FileNotFoundError(f"{root}: no data/chunk-*/*.parquet episodes")
    ids = [int(path.stem.removeprefix("episode_")) for path in paths]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{root}: duplicate episode IDs across Parquet files")
    return paths


def _read_episode(path: Path, tasks: dict[int, str]) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    columns = ["episode_index", "task_index", "frame_index", "timestamp", *MATCHING_FIELDS]
    try:
        frame = pl.read_parquet(path, columns=columns)
    except pl.exceptions.PolarsError as exc:
        raise ValueError(f"{path}: could not read episode columns: {exc}") from exc
    episode = int(path.stem.removeprefix("episode_"))
    length = len(frame)
    if not length or frame["episode_index"].unique().to_list() != [episode]:
        raise ValueError(f"{path}: empty episode or episode_index disagrees with filename")
    task_ids = frame["task_index"].unique().to_list()
    if len(task_ids) != 1 or task_ids[0] not in tasks:
        raise ValueError(f"{path}: expected one task_index registered in meta/tasks.jsonl")
    if not np.array_equal(frame["frame_index"].to_numpy(), np.arange(length)):
        raise ValueError(f"{path}: frame_index must start at zero and increase by one")
    timestamps = frame["timestamp"].to_numpy().astype(np.float32)
    expected = (np.arange(length) / CONTROL_HZ).astype(np.float32)
    if not np.array_equal(timestamps, expected):
        raise ValueError(f"{path}: timestamps must follow frame_index / {CONTROL_HZ}")
    arrays = []
    for field, width in zip(MATCHING_FIELDS, (64, 43), strict=True):
        try:
            array = np.asarray(frame[field].to_list(), dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid {field} array") from exc
        if array.shape != (length, width) or not np.isfinite(array).all():
            raise ValueError(f"{path}: {field} must be finite [{length},{width}], got {array.shape}")
        arrays.append(array)
    record = {"episode_id": f"{episode:06d}", "task_index": int(task_ids[0]), "length": length}
    return record, arrays[0], arrays[1]


def _resource_paths(root: Path, path: Path, record: dict[str, Any], info: dict[str, Any]) -> dict[str, str]:
    video = Path(info["video_path"].format(
        episode_chunk=int(path.parent.name.removeprefix("chunk-")),
        episode_index=int(record["episode_id"]),
        video_key="observation.images.ego_view",
    ))
    if video.is_absolute() or ".." in video.parts or video.name != f"{path.stem}.mp4":
        raise ValueError(f"{path}: video_path must select the relative standard episode MP4: {video}")
    paths = {
        "parquet_path": path.relative_to(root),
        "video_path": video,
        "token_path": Path("sonic_tokens") / f"{path.stem}.npy",
        "proprio_path": Path("g1_proprio") / f"{path.stem}.npy",
    }
    for relative in paths.values():
        if not (root / relative).is_file():
            raise FileNotFoundError(f"{path}: missing resource {root / relative}")
    for key, width in (("token_path", 66), ("proprio_path", 34)):
        cache = root / paths[key]
        shape = np.load(cache, mmap_mode="r", allow_pickle=False).shape
        if shape != (record["length"], width):
            raise ValueError(f"{cache}: expected [{record['length']},{width}], got {shape}")
    with av.open(str(root / video)) as container:
        if not container.streams.video:
            raise ValueError(f"{root / video}: no video stream")
        stream = container.streams.video[0]
        if stream.frames != record["length"] or stream.average_rate != CONTROL_HZ or stream.start_time != 0:
            raise ValueError(
                f"{root / video}: expected {record['length']} frames at 50 FPS starting at 0; "
                f"got frames={stream.frames} fps={stream.average_rate} start={stream.start_time}"
            )
    return {key: relative.as_posix() for key, relative in paths.items()}


def build_index(source_root: Path, reference_root: Path) -> dict[str, Any]:
    info, tasks = _metadata(source_root)
    reference_info, reference_tasks = _metadata(reference_root)
    if info["features"]["observation.state"]["names"] != reference_info["features"]["observation.state"]["names"]:
        raise ValueError("source and reference observation.state field ordering differs")
    source_paths = _episode_paths(source_root)
    reference_paths = _episode_paths(reference_root)
    references: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for path in reference_paths:
        record, motion, state = _read_episode(path, reference_tasks)
        references[record["length"]].append((motion, state))
    episodes = []
    for path in source_paths:
        record, motion, state = _read_episode(path, tasks)
        if any(
            np.array_equal(motion, other_motion) and np.array_equal(state, other_state)
            for other_motion, other_state in references.get(record["length"], [])
        ):
            continue
        record.update(_resource_paths(source_root, path, record, info))
        episodes.append(record)
    return {
        "schema_version": 1,
        "source_root": str(source_root),
        "reference_root": str(reference_root),
        "path_base": "source_root",
        "matching": {
            "fields": list(MATCHING_FIELDS),
            "dtype": "float32",
            "comparison": "full_episode_array_equal",
        },
        "counts": {
            "source_episodes": len(source_paths),
            "reference_episodes": len(reference_paths),
            "excluded_source_episodes": len(source_paths) - len(episodes),
            "holdout_episodes": len(episodes),
            "holdout_frames": sum(episode["length"] for episode in episodes),
        },
        "tasks": {str(index): task for index, task in sorted(tasks.items())},
        "episodes": episodes,
    }


def write_index(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        # Atomic publication without overwriting even if another writer finishes first.
        os.link(temporary, output)
    finally:
        temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, reference, output = (path.expanduser() for path in (args.source_root, args.reference_root, args.output))
    if output.resolve().is_relative_to(source.resolve()) or output.resolve().is_relative_to(reference.resolve()):
        parser.error("--output must be outside both dataset roots")
    if os.path.lexists(output):
        parser.error(f"output already exists; choose a new path: {output}")
    payload = build_index(source, reference)
    write_index(output, payload)
    print(json.dumps(payload["counts"], ensure_ascii=False), flush=True)
    print(f"wrote exact-content holdout index: {output}", flush=True)


if __name__ == "__main__":
    main()

"""Strict grouped statistics for Huge's 34-D G1 proprioception."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from huge.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT, PROPRIO_LAYOUT_VERSION


EXACT_STAGE_TAGS = {
    "stage1": ("EGODEX", "EGOVERSE", "BEHAVIOR1K", "XPERIENCE"),
    "stage12": ("EGODEX", "EGOVERSE", "BEHAVIOR1K", "XPERIENCE", "HE", "WBT", "HIW"),
    "stage123_humanoid_arena": (
        "EGODEX",
        "EGOVERSE",
        "BEHAVIOR1K",
        "XPERIENCE",
        "HE",
        "WBT",
        "HIW",
        "HUMANOID_ARENA",
    ),
}
_STAT_KEYS = ("min", "max", "mean", "std", "q01", "q99")
_EXACT_QUANTILE_METHOD = "exact_external_memmap_linear"


def read_npy_header(path: str | Path) -> tuple[tuple[int, ...], np.dtype[Any]]:
    """Read shape and dtype without mapping or loading the array body."""
    with Path(path).open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported NPY header version {version} in {path}")
    return tuple(shape), np.dtype(dtype)

def _grouped_stats(
    flat: np.ndarray,
    *,
    show_progress: bool = False,
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    groups = tqdm(
        PROPRIO_LAYOUT,
        desc="Compute stats groups",
        unit="group",
        disable=not show_progress,
    )
    for name, start, end in groups:
        part = flat[:, start:end]
        q01, q99 = np.quantile(part, (0.01, 0.99), axis=0)
        result[name] = {
            "min": part.min(axis=0).astype(float).tolist(),
            "max": part.max(axis=0).astype(float).tolist(),
            "mean": part.mean(axis=0).astype(float).tolist(),
            "std": part.std(axis=0).astype(float).tolist(),
            "q01": q01.astype(float).tolist(),
            "q99": q99.astype(float).tolist(),
        }
    return result


def compute_proprio_stats(
    sources: dict[str, Path],
    *,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Compute the legacy in-memory payload used by HumanoidArena preprocessing."""
    if not sources:
        raise ValueError("at least one proprio source is required")

    source_paths: dict[str, list[Path]] = {}
    for tag, source_root in sources.items():
        root = Path(source_root)
        paths = sorted(path for path in root.rglob("episode_*.npy") if path.is_file()) if root.is_dir() else []
        if not paths:
            raise ValueError(f"{tag}: no episode_*.npy files under {root}")
        source_paths[tag] = paths

    arrays: list[np.ndarray] = []
    counts: dict[str, dict[str, int]] = {}
    progress = tqdm(
        total=sum(len(paths) for paths in source_paths.values()),
        desc="Load union stats",
        unit="file",
        disable=not show_progress,
    )
    try:
        for tag, paths in source_paths.items():
            frame_count = 0
            for path in paths:
                array = np.load(path, mmap_mode="r")
                if array.ndim != 2 or array.shape[1] != PROPRIO_DIM:
                    raise ValueError(f"{tag}: {path} expected [T,{PROPRIO_DIM}], got {array.shape}")
                if not np.isfinite(array).all():
                    raise ValueError(f"{tag}: nonfinite values in {path}")
                arrays.append(np.asarray(array, dtype=np.float32))
                frame_count += int(array.shape[0])
                progress.update()
            counts[tag] = {"n_episodes": len(paths), "n_frames": frame_count}
    finally:
        progress.close()

    flat = np.concatenate(arrays, axis=0)
    return {
        "unitree_g1_sonic": {"state": _grouped_stats(flat, show_progress=show_progress)},
        "_meta": {
            "sources": counts,
            "n_frames": int(flat.shape[0]),
            "layout": PROPRIO_LAYOUT_VERSION,
            "g1_proprio_dim": PROPRIO_DIM,
        },
    }


def _write_payload(payload: dict[str, Any], out: Path) -> None:
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _exact_inventory(
    sources: dict[str, Path],
    stage_scope: str,
) -> tuple[dict[str, list[Path]], dict[str, dict[str, Any]], int]:
    expected_tags = EXACT_STAGE_TAGS.get(stage_scope)
    if expected_tags is None:
        raise ValueError(f"unsupported stage_scope {stage_scope!r}; expected {sorted(EXACT_STAGE_TAGS)}")
    actual_tags = set(sources)
    if actual_tags != set(expected_tags):
        raise ValueError(
            f"{stage_scope} requires source tags {list(expected_tags)}, got {sorted(actual_tags)}"
        )

    source_paths: dict[str, list[Path]] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    total_frames = 0
    for tag in expected_tags:
        root = Path(sources[tag]).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"{tag}: proprio root not found: {root}")
        paths = sorted(root.rglob("*.npy"))
        if not paths:
            raise ValueError(f"{tag}: no *.npy files under {root}")
        frames = 0
        def probe(path: Path) -> int:
            shape, dtype = read_npy_header(path)
            if len(shape) != 2 or shape[1] != PROPRIO_DIM:
                raise ValueError(f"{tag}: {path} expected [T,{PROPRIO_DIM}], got {shape}")
            if dtype != np.dtype(np.float32):
                raise ValueError(f"{tag}: {path} expected float32, got {dtype}")
            if shape[0] <= 0:
                raise ValueError(f"{tag}: {path} has no frames")
            return int(shape[0])

        with ThreadPoolExecutor(max_workers=32) as executor:
            frames = sum(executor.map(probe, paths))
        if frames <= 0:
            raise ValueError(f"{tag}: source has no frames")
        source_paths[tag] = paths
        source_meta[tag] = {
            "logical_root": str(root),
            "resolved_root": str(root.resolve()),
            "n_files": len(paths),
            "n_frames": frames,
        }
        total_frames += frames
    if total_frames <= 0:
        raise ValueError("exact proprio stats require at least one frame")
    return source_paths, source_meta, total_frames


def _merge_moments(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    chunk: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    chunk_count = int(chunk.shape[0])
    chunk_mean = chunk.mean(axis=0, dtype=np.float64)
    centered = chunk - chunk_mean
    chunk_m2 = np.einsum("ij,ij->j", centered, centered)
    if count == 0:
        return chunk_count, chunk_mean, chunk_m2
    total = count + chunk_count
    delta = chunk_mean - mean
    merged_mean = mean + delta * (chunk_count / total)
    merged_m2 = m2 + chunk_m2 + delta * delta * (count * chunk_count / total)
    return total, merged_mean, merged_m2


def _linear_quantile_positions(n_frames: int, quantile: float) -> tuple[int, int, float]:
    position = (n_frames - 1) * quantile
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    return lower, upper, position - lower


def _compute_exact_proprio_stats(
    sources: dict[str, Path],
    *,
    scratch_dir: Path,
    stage_scope: str,
    show_progress: bool,
) -> dict[str, Any]:
    source_paths, source_meta, n_frames = _exact_inventory(sources, stage_scope)
    scratch_root = Path(scratch_dir).expanduser()
    scratch_root.mkdir(parents=True, exist_ok=True)
    required_bytes = n_frames * PROPRIO_DIM * np.dtype(np.float32).itemsize
    available_bytes = shutil.disk_usage(scratch_root).free
    if available_bytes < required_bytes:
        raise OSError(
            f"scratch directory {scratch_root} needs {required_bytes} bytes, has {available_bytes}"
        )

    workspace = Path(tempfile.mkdtemp(prefix="huge-proprio-stats-", dir=scratch_root))
    try:
        matrix = np.memmap(
            workspace / "dimension_major.f32",
            mode="w+",
            dtype=np.float32,
            shape=(PROPRIO_DIM, n_frames),
        )
        count = 0
        mean = np.zeros(PROPRIO_DIM, dtype=np.float64)
        m2 = np.zeros(PROPRIO_DIM, dtype=np.float64)
        minimum = np.full(PROPRIO_DIM, np.inf, dtype=np.float64)
        maximum = np.full(PROPRIO_DIM, -np.inf, dtype=np.float64)
        offset = 0
        progress = tqdm(
            total=sum(len(paths) for paths in source_paths.values()),
            desc="Scan exact proprio stats",
            unit="file",
            disable=not show_progress,
        )
        try:
            for tag in EXACT_STAGE_TAGS[stage_scope]:
                for path in source_paths[tag]:
                    array = np.load(path, allow_pickle=False)
                    for start in range(0, len(array), 65_536):
                        chunk = np.asarray(array[start : start + 65_536])
                        if not np.isfinite(chunk).all():
                            raise ValueError(f"{tag}: nonfinite values in {path}")
                        chunk64 = chunk.astype(np.float64)
                        count, mean, m2 = _merge_moments(count, mean, m2, chunk64)
                        minimum = np.minimum(minimum, chunk64.min(axis=0))
                        maximum = np.maximum(maximum, chunk64.max(axis=0))
                        end = offset + len(chunk)
                        matrix[:, offset:end] = chunk.T
                        offset = end
                    progress.update()
        finally:
            progress.close()
        if count != n_frames or offset != n_frames:
            raise RuntimeError(f"exact stats frame count changed during scan: {count}/{offset} != {n_frames}")
        matrix.flush()

        q01 = np.empty(PROPRIO_DIM, dtype=np.float64)
        q99 = np.empty(PROPRIO_DIM, dtype=np.float64)
        q01_pos = _linear_quantile_positions(n_frames, 0.01)
        q99_pos = _linear_quantile_positions(n_frames, 0.99)
        kth = sorted({q01_pos[0], q01_pos[1], q99_pos[0], q99_pos[1]})
        for dim in tqdm(
            range(PROPRIO_DIM),
            desc="Partition exact quantiles",
            unit="dim",
            disable=not show_progress,
        ):
            values = matrix[dim]
            values.partition(kth)
            q01_lower = float(values[q01_pos[0]])
            q01_upper = float(values[q01_pos[1]])
            q99_lower = float(values[q99_pos[0]])
            q99_upper = float(values[q99_pos[1]])
            q01[dim] = q01_lower + q01_pos[2] * (q01_upper - q01_lower)
            q99[dim] = q99_lower + q99_pos[2] * (q99_upper - q99_lower)

        flat_stats = {
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": np.sqrt(m2 / n_frames),
            "q01": q01,
            "q99": q99,
        }
        grouped = {
            name: {
                key: flat_stats[key][start:end].astype(float).tolist()
                for key in _STAT_KEYS
            }
            for name, start, end in PROPRIO_LAYOUT
        }
        return {
            "unitree_g1_sonic": {"state": grouped},
            "_meta": {
                "sources": source_meta,
                "n_frames": n_frames,
                "layout": PROPRIO_LAYOUT_VERSION,
                "g1_proprio_dim": PROPRIO_DIM,
                "stage_scope": stage_scope,
                "task_scope": "all" if stage_scope == "stage123_humanoid_arena" else None,
                "frame_weighting": "global_per_frame",
                "quantile_method": _EXACT_QUANTILE_METHOD,
            },
        }
    finally:
        shutil.rmtree(workspace)


def validate_proprio_stats_contract(
    path: str | Path,
    *,
    stage_scope: str,
    sources: dict[str, Path],
    task_scope: str | None = None,
) -> dict[str, Any]:
    """Validate one exact cumulative stats file against the active roots."""
    stats_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable proprio stats: {stats_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"proprio stats must be a JSON object: {stats_path}")
    expected_tags = EXACT_STAGE_TAGS.get(stage_scope)
    if expected_tags is None or set(sources) != set(expected_tags):
        raise ValueError(f"invalid expected sources for {stage_scope}: {sorted(sources)}")
    meta = payload.get("_meta")
    if not isinstance(meta, dict):
        raise ValueError("proprio stats are missing _meta")
    expected_meta = {
        "stage_scope": stage_scope,
        "task_scope": task_scope,
        "frame_weighting": "global_per_frame",
        "quantile_method": _EXACT_QUANTILE_METHOD,
        "layout": PROPRIO_LAYOUT_VERSION,
        "g1_proprio_dim": PROPRIO_DIM,
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise ValueError(f"proprio stats {key} mismatch: {meta.get(key)!r} != {expected!r}")

    source_meta = meta.get("sources")
    if not isinstance(source_meta, dict) or set(source_meta) != set(expected_tags):
        raise ValueError(f"proprio stats source tags mismatch: {sorted(source_meta or {})}")
    total_frames = 0
    for tag in expected_tags:
        item = source_meta.get(tag)
        if not isinstance(item, dict):
            raise ValueError(f"proprio stats source metadata is invalid: {tag}")
        resolved_root = str(Path(sources[tag]).expanduser().resolve())
        if item.get("resolved_root") != resolved_root:
            raise ValueError(
                f"proprio stats {tag} root mismatch: {item.get('resolved_root')!r} != {resolved_root!r}"
            )
        n_files = item.get("n_files")
        n_source_frames = item.get("n_frames")
        if type(n_files) is not int or n_files <= 0 or type(n_source_frames) is not int or n_source_frames <= 0:
            raise ValueError(f"proprio stats {tag} requires positive n_files/n_frames")
        total_frames += n_source_frames
    if type(meta.get("n_frames")) is not int or meta["n_frames"] != total_frames:
        raise ValueError(f"proprio stats total n_frames mismatch: {meta.get('n_frames')!r} != {total_frames}")

    embodiment = payload.get("unitree_g1_sonic")
    state = embodiment.get("state") if isinstance(embodiment, dict) else None
    expected_groups = {name for name, _, _ in PROPRIO_LAYOUT}
    if not isinstance(state, dict) or set(state) != expected_groups:
        raise ValueError(f"proprio stats groups mismatch: {sorted(state or {})}")
    for name, start, end in PROPRIO_LAYOUT:
        section = state[name]
        if not isinstance(section, dict) or set(section) != set(_STAT_KEYS):
            raise ValueError(f"proprio stats group {name} has invalid fields")
        values: dict[str, np.ndarray] = {}
        for key in _STAT_KEYS:
            try:
                array = np.asarray(section[key], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"proprio stats group {name}.{key} is not numeric") from exc
            if array.shape != (end - start,) or not np.isfinite(array).all():
                raise ValueError(f"proprio stats group {name}.{key} must be finite width {end - start}")
            values[key] = array
        if np.any(values["std"] < 0):
            raise ValueError(f"proprio stats group {name}.std must be nonnegative")
        if np.any(values["min"] > values["q01"]) or np.any(values["q01"] > values["q99"]) or np.any(values["q99"] > values["max"]):
            raise ValueError(f"proprio stats group {name} violates min <= q01 <= q99 <= max")
    return payload


def write_proprio_stats(
    sources: dict[str, Path],
    out: Path,
    *,
    show_progress: bool = False,
    scratch_dir: str | Path | None = None,
    stage_scope: str | None = None,
) -> dict[str, Any]:
    """Compute and atomically write legacy or exact cumulative stats."""
    if (scratch_dir is None) != (stage_scope is None):
        raise ValueError("scratch_dir and stage_scope must be provided together")
    payload = (
        compute_proprio_stats(sources, show_progress=show_progress)
        if scratch_dir is None
        else _compute_exact_proprio_stats(
            sources,
            scratch_dir=Path(scratch_dir),
            stage_scope=str(stage_scope),
            show_progress=show_progress,
        )
    )
    _write_payload(payload, Path(out))
    return payload

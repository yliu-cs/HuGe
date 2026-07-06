"""Data audit CLI for the QwenSonicFM rewrite.

Run:
    uv run python src/huge/data/audit.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from huge.data.schemas import ACTION_DIM, MOTION_TOKEN_DIM, PROPRIO_DIM, hand_binary_is_valid
from huge.data.state_normalizer import StateNormalizer


@dataclass
class SourceSpec:
    name: str
    root: Path
    token_root: Path
    proprio_root: Path | None
    expect_state: bool
    allow_64d: bool = False
    token_glob: str = "*.npy"
    count_limit: int | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _count_files(root: Path, pattern: str, limit: int | None = None) -> int | str:
    if not root.exists():
        return 0
    count = 0
    for p in root.rglob(pattern):
        if p.is_file():
            count += 1
            if limit is not None and count >= limit:
                return f">={limit}"
    return count


def _first_npy(root: Path, pattern: str = "*.npy") -> Path | None:
    if not root.exists():
        return None
    return next((p for p in root.rglob(pattern) if p.is_file()), None)


def _sample_npy(root: Path, limit: int, pattern: str = "*.npy") -> list[Path]:
    if not root.exists():
        return []
    return [p for p in islice((p for p in root.rglob(pattern) if p.is_file()), limit)]


def _shape(path: Path | None) -> str:
    if path is None:
        return "missing"
    arr = np.load(path, mmap_mode="r")
    return "x".join(str(x) for x in arr.shape)


def _hand_stats(paths: list[Path]) -> dict[str, Any]:
    total = 0
    left_closed = 0.0
    right_closed = 0.0
    invalid = 0
    for path in paths:
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
            continue
        hand = np.asarray(arr[:, MOTION_TOKEN_DIM:ACTION_DIM])
        total += hand.shape[0]
        left_closed += float(hand[:, 0].sum())
        right_closed += float(hand[:, 1].sum())
        invalid += 0 if hand_binary_is_valid(np.asarray(arr)) else 1
    if total == 0:
        return {"frames": 0, "left_closed_ratio": None, "right_closed_ratio": None, "invalid_episode_count": invalid}
    return {
        "frames": total,
        "left_closed_ratio": left_closed / total,
        "right_closed_ratio": right_closed / total,
        "invalid_episode_count": invalid,
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _audit_source(spec: SourceSpec, sample_limit: int) -> dict[str, Any]:
    first_token = _first_npy(spec.token_root, spec.token_glob)
    first_proprio = _first_npy(spec.proprio_root) if spec.proprio_root else None
    out: dict[str, Any] = {
        "name": spec.name,
        "root": str(spec.root),
        "root_exists": spec.root.exists(),
        "root_is_symlink": spec.root.is_symlink(),
        "token_root": str(spec.token_root),
        "token_files": _count_files(spec.token_root, spec.token_glob, spec.count_limit),
        "proprio_root": str(spec.proprio_root) if spec.proprio_root else None,
        "proprio_files": _count_files(spec.proprio_root, "*.npy") if spec.proprio_root else 0,
        "first_token": str(first_token) if first_token else None,
        "first_token_shape": _shape(first_token),
        "first_proprio": str(first_proprio) if first_proprio else None,
        "first_proprio_shape": _shape(first_proprio),
        "errors": [],
    }
    if first_token is None:
        out["errors"].append("missing sonic token npy files")
    else:
        arr = np.load(first_token, mmap_mode="r")
        if arr.ndim != 2:
            out["errors"].append(f"first token rank is {arr.ndim}, expected 2")
        elif arr.shape[1] not in ((64, ACTION_DIM) if spec.allow_64d else (ACTION_DIM,)):
            out["errors"].append(f"first token dim is {arr.shape[1]}, expected {ACTION_DIM}")
        finite = bool(np.isfinite(np.asarray(arr[: min(len(arr), 16), : min(arr.shape[1], MOTION_TOKEN_DIM)])).all())
        out["motion_token_finite_sample"] = finite
    if spec.expect_state:
        if first_proprio is None:
            out["errors"].append("missing g1 proprio npy files")
        else:
            prop = np.load(first_proprio, mmap_mode="r")
            if prop.ndim != 2 or prop.shape[1] != PROPRIO_DIM:
                out["errors"].append(f"first proprio shape is {prop.shape}, expected [T, {PROPRIO_DIM}]")
    out["hand_stats"] = _hand_stats(_sample_npy(spec.token_root, sample_limit, spec.token_glob))
    return out


def _check_normalizer(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        norm = StateNormalizer.from_json(path)
        zeros = np.zeros((PROPRIO_DIM,), dtype=np.float32)
        normalized = norm.normalize_flat(norm.denormalize_flat(zeros))
        return {
            "path": str(path),
            "exists": True,
            "roundtrip_max_abs_error": float(np.max(np.abs(normalized - zeros))),
            "proprio_dim": PROPRIO_DIM,
        }
    except Exception as exc:  # pragma: no cover - audit should report, not crash.
        return {"path": str(path), "exists": True, "error": repr(exc)}


def _format_report(results: dict[str, Any]) -> str:
    lines = [
        "# Huge Data Audit Report",
        "",
        f"repo_root: `{results['repo_root']}`",
        f"sonic_encoder_exists: `{results['sonic_encoder_exists']}`",
        f"qwen_processor_path_exists: `{results['qwen_processor_path_exists']}`",
        "",
        "## Sources",
        "",
    ]
    for src in results["sources"]:
        lines.extend(
            [
                f"### {src['name']}",
                "",
                f"- root: `{src['root']}`",
                f"- root_exists: `{src['root_exists']}`",
                f"- root_is_symlink: `{src['root_is_symlink']}`",
                f"- token_files: `{src['token_files']}`",
                f"- proprio_files: `{src['proprio_files']}`",
                f"- first_token_shape: `{src['first_token_shape']}`",
                f"- first_proprio_shape: `{src['first_proprio_shape']}`",
                f"- motion_token_finite_sample: `{src.get('motion_token_finite_sample')}`",
                f"- hand_stats: `{json.dumps(src['hand_stats'], ensure_ascii=True)}`",
                f"- errors: `{json.dumps(src['errors'], ensure_ascii=True)}`",
                "",
            ]
        )
    lines.extend(["## Proprio Stats", ""])
    for item in results["proprio_stats"]:
        lines.append(f"- `{item['path']}`: `{json.dumps(item, ensure_ascii=True)}`")
    lines.append("")
    return "\n".join(lines)


def build_audit(repo_root: Path, sample_limit: int) -> dict[str, Any]:
    dataset = repo_root / "resource" / "dataset"
    specs = [
        SourceSpec(
            "egodex",
            dataset / "EgoDex",
            dataset / "EgoDex" / "egodex_repacked_simple" / "episodes",
            None,
            expect_state=False,
            allow_64d=True,
            token_glob="token_state.npy",
            count_limit=1000,
        ),
        SourceSpec(
            "humanoid_everyday",
            dataset / "humanoid-everyday",
            dataset / "humanoid-everyday-cache" / "sonic_tokens",
            dataset / "humanoid-everyday-cache" / "g1_proprio",
            expect_state=True,
        ),
        SourceSpec(
            "g1_wbt",
            dataset / "G1_WBT",
            dataset / "G1_WBT" / "sonic_tokens",
            dataset / "G1_WBT" / "g1_proprio",
            expect_state=True,
        ),
        SourceSpec(
            "proprietary_real_cola_merged_cleaned_0530",
            dataset / "proprietary" / "real" / "cola_merged_cleaned_0530",
            dataset / "proprietary" / "real" / "cola_merged_cleaned_0530" / "sonic_tokens",
            dataset / "proprietary" / "real" / "cola_merged_cleaned_0530" / "g1_proprio",
            expect_state=True,
        ),
    ]
    return {
        "repo_root": str(repo_root),
        "sonic_encoder_exists": (repo_root / "resource" / "model" / "sonic" / "model_encoder.onnx").exists(),
        "qwen_processor_path_exists": (repo_root / "resource" / "model" / "Qwen3-VL-2B-Instruct").exists(),
        "sources": [_audit_source(spec, sample_limit) for spec in specs],
        "proprio_stats": [
            _check_normalizer(dataset / "G1_WBT-cache" / "meta" / "g1_proprio_stats_sonic.json"),
            _check_normalizer(dataset / "proprietary" / "real" / "cola_merged_cleaned_0530" / "meta" / "proprio_stats_union_sonic.json"),
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sample-limit", type=int, default=8)
    args = parser.parse_args(argv)

    results = build_audit(args.repo_root.resolve(), args.sample_limit)
    output = args.output or (args.repo_root / "resource" / "info" / "data_audit_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_format_report(results), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

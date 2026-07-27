"""Export aligned G1 simulation state as Huge's canonical 34-D proprio array."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from huge.data.schemas import ACTION_DIM, MOTION_TOKEN_DIM, PROPRIO_DIM, validate_action_array
from huge.data.state_normalizer import StateNormalizer

HAND_BINARY_CLOSE_THRESHOLD = 0.5
DEX3_FULL_CLOSE_ABS = 1.5
CLIP_FRAC_WARN_THRESHOLD = 0.2


def dex3_hand_close_binary(joints: np.ndarray) -> np.ndarray:
    """Convert Dex3 joint positions to closed=1/open=0."""
    if joints.ndim != 2 or joints.shape[1] == 0:
        raise ValueError(f"expected [T,D] hand joints, got {joints.shape}")
    close_amount = np.clip(np.max(np.abs(joints), axis=-1) / DEX3_FULL_CLOSE_ABS, 0.0, 1.0)
    return (close_amount >= HAND_BINARY_CLOSE_THRESHOLD).astype(np.float64)


def projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """Convert [T,4] scalar-first quaternions to body-frame unit gravity."""
    if base_quat.ndim != 2 or base_quat.shape[1] != 4:
        raise ValueError(f"expected base_quat [T,4], got {base_quat.shape}")
    return Rot.from_quat(base_quat, scalar_first=True).inv().apply(np.array([0.0, 0.0, -1.0]))


def fill_edge_frames(array: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Fill contiguous NaN head/tail rows and reject interior gaps."""
    if array.ndim < 2 or array.shape[0] == 0:
        raise ValueError(f"expected a non-empty time-major array, got {array.shape}")
    nan_rows = np.isnan(array).any(axis=tuple(range(1, array.ndim)))
    real_rows = np.flatnonzero(~nan_rows)
    if real_rows.size == 0:
        raise ValueError("aligned field contains no real frames")
    first, last = int(real_rows[0]), int(real_rows[-1])
    if nan_rows[first : last + 1].any():
        raise ValueError("NaN frames are not confined to the aligned head/tail")

    filled = array.copy()
    if first:
        filled[:first] = array[first]
    tail = array.shape[0] - last - 1
    if tail:
        filled[last + 1 :] = array[last]
    return filled, first, tail


def build_pseudo_proprio(aligned: dict[str, np.ndarray]) -> tuple[np.ndarray, int, int]:
    """Assemble simulated body, hand and gravity fields into [T,34]."""
    required = ("body_q", "left_hand_q", "right_hand_q", "base_quat")
    missing = [field for field in required if field not in aligned]
    if missing:
        raise ValueError(f"aligned input is missing required fields: {missing}")

    filled: dict[str, np.ndarray] = {}
    gaps: set[tuple[int, int]] = set()
    for field in required:
        filled[field], head, tail = fill_edge_frames(np.asarray(aligned[field], dtype=np.float64))
        gaps.add((head, tail))
    if len(gaps) != 1:
        raise ValueError(f"aligned fields disagree on missing head/tail rows: {sorted(gaps)}")
    missing_head, missing_tail = gaps.pop()

    body_q = filled["body_q"]
    if body_q.ndim != 2 or body_q.shape[1] != 29:
        raise ValueError(f"expected body_q [T,29], got {body_q.shape}")
    lengths = {field: value.shape[0] for field, value in filled.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"aligned field lengths disagree: {lengths}")

    output = np.zeros((body_q.shape[0], PROPRIO_DIM), dtype=np.float64)
    output[:, :29] = body_q
    output[:, 29] = dex3_hand_close_binary(filled["left_hand_q"])
    output[:, 30] = dex3_hand_close_binary(filled["right_hand_q"])
    output[:, 31:34] = projected_gravity(filled["base_quat"])
    return output, missing_head, missing_tail


def apply_hand_state(output: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    """Use action hand labels for 66D input, otherwise keep simulated defaults."""
    if tokens.shape[0] != output.shape[0]:
        raise ValueError(f"token/proprio length mismatch: {tokens.shape[0]} != {output.shape[0]}")
    if tokens.shape[1] == MOTION_TOKEN_DIM:
        return output, {"hand_state_source": "simulated_default"}, ["input_has_no_hand_action"]

    action_hand = np.asarray(tokens[:, MOTION_TOKEN_DIM:ACTION_DIM], dtype=np.float64)
    if not np.isin(action_hand, (0.0, 1.0)).all():
        raise ValueError("66D hand action must contain only 0 or 1")
    changed = int(np.any(output[:, 29:31] != action_hand, axis=1).sum())
    corrected = output.copy()
    corrected[:, 29:31] = action_hand
    return corrected, {
        "hand_state_source": "action_proxy",
        "hand_frames_corrected": changed,
        "hand_frames_total": int(output.shape[0]),
    }, []


def self_check(output: np.ndarray, token_rows: int, stats: str | None) -> tuple[dict[str, Any], list[str]]:
    """Run hard schema checks and an optional distribution check."""
    if output.shape != (token_rows, PROPRIO_DIM):
        raise ValueError(f"proprio shape {output.shape} != ({token_rows}, {PROPRIO_DIM})")
    if not np.isfinite(output).all():
        raise ValueError("proprio output contains NaN or Inf")
    if not np.isin(output[:, 29:31], (0.0, 1.0)).all():
        raise ValueError("proprio hand state must contain only 0 or 1")
    gravity_norm = np.linalg.norm(output[:, 31:34], axis=1)
    gravity_error = float(np.max(np.abs(gravity_norm - 1.0)))
    if gravity_error > 1e-2:
        raise ValueError(f"projected gravity unit-norm error {gravity_error} exceeds 1e-2")

    report: dict[str, Any] = {"gravity_norm_max_err": gravity_error, "clip_frac": None}
    warnings: list[str] = []
    if stats is None:
        report["distribution_check"] = "skipped_no_stats"
    else:
        normalized = StateNormalizer.from_json(stats).normalize(output.astype(np.float32))
        report["clip_frac"] = float((np.abs(normalized) >= 0.999).mean())
        report["distribution_check"] = "complete"
        if report["clip_frac"] >= CLIP_FRAC_WARN_THRESHOLD:
            warnings.append("clip_fraction_at_or_above_0.2")
    return report, warnings


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--token", required=True, help="input [T,64] or [T,66] .npy")
    parser.add_argument("--output", required=True, help="output [T,34] .npy")
    parser.add_argument("--report", default=None, help="default: <output stem>.report.json")
    parser.add_argument("--stats", default=None, help="optional 34D proprio stats JSON")
    parser.add_argument("--max-edge-fill", type=int, default=1)
    args = parser.parse_args()

    if args.max_edge_fill < 0:
        parser.error("--max-edge-fill must be non-negative")
    output_path = Path(args.output)
    if output_path.suffix != ".npy":
        parser.error("--output must end in .npy")

    tokens = np.load(args.token)
    if tokens.ndim != 2:
        raise ValueError(f"expected [T,64] or [T,66] tokens, got {tokens.shape}")
    validate_action_array(tokens, allow_64d=True)
    if tokens.shape[0] == 0:
        raise ValueError("token input is empty")
    if tokens.shape[1] == ACTION_DIM and not np.isin(tokens[:, MOTION_TOKEN_DIM:ACTION_DIM], (0.0, 1.0)).all():
        raise ValueError("66D hand action must contain only 0 or 1")
    with np.load(args.aligned) as archive:
        aligned = {key: archive[key] for key in archive.files}

    output, missing_head, missing_tail = build_pseudo_proprio(aligned)
    edge_frames = missing_head + missing_tail
    if edge_frames > args.max_edge_fill:
        raise ValueError(f"edge fill {edge_frames} exceeds --max-edge-fill={args.max_edge_fill}")
    output, hand_report, warnings = apply_hand_state(output, tokens)
    if edge_frames > 1:
        warnings.append("edge_fill_exceeds_expected_1")
    check_report, check_warnings = self_check(output, token_rows=tokens.shape[0], stats=args.stats)
    warnings.extend(check_warnings)

    report = {
        "status": "complete",
        "input_dim": int(tokens.shape[1]),
        "token_rows": int(tokens.shape[0]),
        "proprio_rows": int(output.shape[0]),
        "missing_head_filled": missing_head,
        "missing_tail_filled": missing_tail,
        "edge_frames_filled": edge_frames,
        **hand_report,
        **check_report,
        "warnings": warnings,
    }
    _atomic_npy(output_path, output.astype(np.float32))
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")
    _atomic_json(report_path, report)
    print(f"wrote {output_path} shape={output.shape} hand={report['hand_state_source']}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

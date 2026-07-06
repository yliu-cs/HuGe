"""Assemble raw [T,34] pseudo proprio from an `align_sim_replay.py align` output
and lay it out next to the cola 66D tokens so `PropDataset`/`SonicTokenDataset`
can load it with zero changes.

Field mapping (see resource/info/ego_cola_pseudo_proprio_handoff.md section 5/7):
  Huge 34D [0:29]  (leg/leg/waist/arm/arm) <- body_q            (already raw radians,
                                                                   MuJoCo joint order == PROPRIO_LAYOUT order)
  Huge 34D [29]    (left_hand)             <- left_hand_q  -> dex3 close-binary (then corrected, see below)
  Huge 34D [30]    (right_hand)            <- right_hand_q -> dex3 close-binary (then corrected, see below)
  Huge 34D [31:34] (projected_gravity)     <- base_quat -> world gravity rotated into body frame

The close-binary and projected-gravity formulas mirror
GR00T-WholeBodyControl/gear_sonic/scripts/run_data_exporter.py's
`_dex3_hand_close_binary`/`compute_projected_gravity` (reimplemented here, not
imported, since hardcoded rule #1 forbids depending on that repo's code paths
for this pipeline -- the numbers must simply agree, not the code).

Hand-state correction (post-hoc, not a simulation-input change): the replay
pipeline's ZMQ pose stream to the WBC deploy binary only ever carries
`token_state`/`frame_index`/`body_quat_w` (see
`sonic_encoder_input_player.py`'s `publish_token_stream`) -- no hand-action
field is ever transmitted, so the simulated hand never actually moves; it's
driven by keyboard-only teleop that gets no input in headless mode. The
captured left_hand_q/right_hand_q are therefore not meaningful signal for
hand state. Since the pseudo proprio and the 66D token's hand-binary action
(dims 64:66, order left,right -- see `_hand_binary_action` in
run_data_exporter.py) are already 1:1 index-aligned by construction (same
token index == same output frame, no extra interpolation needed),
`correct_hand_state_from_action` overwrites proprio dims [29:31] with the
action's own hand-binary values as the best available proxy for
"instantaneously achieved" hand state, given the sim can't supply a real one.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from huge.data.schemas import PROPRIO_DIM, PROPRIO_LAYOUT
from huge.data.state_normalizer import StateNormalizer

DEFAULT_STATS = "resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json"
DEFAULT_OUT_ROOT = "resource/dataset/proprietary/ego/cola_pseudo_proprio"
DEFAULT_COLA_ROOT = "resource/dataset/proprietary/ego/cola"

HAND_BINARY_CLOSE_THRESHOLD = 0.5
DEX3_FULL_CLOSE_ABS = 1.5
CLIP_FRAC_WARN_THRESHOLD = 0.2  # C6: fraction of normalized dims pinned to +-1 before we flag it


def dex3_hand_close_binary(q: np.ndarray) -> np.ndarray:
    """q: [T, D] joint positions -> [T] closed=1/open=0."""
    close_amount = np.clip(np.max(np.abs(q), axis=-1) / DEX3_FULL_CLOSE_ABS, 0.0, 1.0)
    return (close_amount >= HAND_BINARY_CLOSE_THRESHOLD).astype(np.float64)


def projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """base_quat: [T,4] (w,x,y,z) -> [T,3] gravity in body frame, unit norm."""
    rot = Rot.from_quat(base_quat, scalar_first=True)
    gravity_world = np.array([0.0, 0.0, -1.0])
    return rot.inv().apply(gravity_world)


def forward_fill_tail(arr: np.ndarray, warn_threshold: int = 1) -> tuple[np.ndarray, int]:
    """Fill NaN edges (from `align_sim_replay.py`'s missing_head/missing_tail) by
    holding the nearest real frame -- back-fill a leading gap (ZMQ slow-joiner
    dropped the first token(s)), forward-fill a trailing gap (deploy exits before
    its last flush). Raises if any NaN frame isn't part of a contiguous head/tail
    run -- that would mean alignment broke somewhere in the middle."""
    nan_mask = np.isnan(arr).any(axis=tuple(range(1, arr.ndim)))
    n_missing = int(nan_mask.sum())
    if n_missing == 0:
        return arr, 0

    real_idx = np.flatnonzero(~nan_mask)
    if real_idx.size == 0:
        raise ValueError("every frame is NaN; alignment produced no matches at all")
    first_real, last_real = int(real_idx[0]), int(real_idx[-1])
    if nan_mask[first_real:last_real + 1].any():
        raise ValueError("NaN frames are not confined to the head/tail; alignment broke unexpectedly")

    filled = arr.copy()
    if first_real > 0:
        filled[:first_real] = arr[first_real]
    if last_real < arr.shape[0] - 1:
        filled[last_real + 1:] = arr[last_real]
    if n_missing > warn_threshold:
        print(f"WARNING: filled {n_missing} missing head/tail frames (> expected ~{warn_threshold})")
    return filled, n_missing


def build_pseudo_proprio(aligned: dict[str, np.ndarray]) -> np.ndarray:
    body_q, n_body = forward_fill_tail(aligned["body_q"])
    left_hand_q, _ = forward_fill_tail(aligned["left_hand_q"])
    right_hand_q, _ = forward_fill_tail(aligned["right_hand_q"])
    base_quat, _ = forward_fill_tail(aligned["base_quat"])

    if body_q.shape[1] != 29:
        raise ValueError(f"expected body_q width 29, got {body_q.shape[1]}")

    t = body_q.shape[0]
    out = np.zeros((t, PROPRIO_DIM), dtype=np.float64)
    out[:, 0:29] = body_q
    out[:, 29] = dex3_hand_close_binary(left_hand_q)
    out[:, 30] = dex3_hand_close_binary(right_hand_q)
    out[:, 31:34] = projected_gravity(base_quat)
    return out, n_body


def correct_hand_state_from_action(out: np.ndarray, tokens: np.ndarray) -> tuple[np.ndarray, dict]:
    """Overwrite proprio hand dims [29:31] with the 66D token's own hand-binary
    action (dims 64:66). See module docstring for why: the sim never actually
    drives the hand, so the sim-derived left_hand_q/right_hand_q close-binary
    is not a usable state signal -- the commanded action is the best proxy we
    have for it, and the two arrays are already frame-aligned by construction."""
    if tokens.shape[-1] < 66:
        raise ValueError(f"expected a 66D token array with a trailing hand_binary action, got dim {tokens.shape[-1]}")
    if tokens.shape[0] != out.shape[0]:
        raise ValueError(f"token/proprio frame count mismatch: {tokens.shape[0]} vs {out.shape[0]}")

    action_hand = tokens[:, 64:66].astype(np.float64)
    if not np.isin(action_hand, [0.0, 1.0]).all():
        raise ValueError("token hand_binary action is not in {0,1}; C5 polarity assumption violated")

    sim_hand = out[:, 29:31]
    n_changed = int(np.sum(np.any(sim_hand != action_hand, axis=1)))
    corrected = out.copy()
    corrected[:, 29:31] = action_hand
    return corrected, {
        "hand_state_source": "action",
        "hand_frames_corrected": n_changed,
        "hand_frames_total": int(out.shape[0]),
    }


def self_check(out: np.ndarray, token_len: int, stats_path: str) -> dict[str, float]:
    assert out.shape == (token_len, PROPRIO_DIM), f"C2 shape: {out.shape} vs [{token_len},{PROPRIO_DIM}]"  # C2
    gnorm = np.linalg.norm(out[:, 31:34], axis=1)
    assert np.allclose(gnorm, 1.0, atol=1e-2), f"C1 proj_gravity not unit norm: max err {np.max(np.abs(gnorm - 1.0))}"
    assert np.isin(out[:, 29:31], [0.0, 1.0]).all(), "C5 hand polarity not in {0,1}"  # C5

    normalizer = StateNormalizer.from_json(stats_path)
    normed = normalizer.normalize(out.astype(np.float32))
    clip_frac = float((np.abs(normed) >= 0.999).mean())
    report = {"gravity_norm_max_err": float(np.max(np.abs(gnorm - 1.0))), "clip_frac": clip_frac}
    if clip_frac >= CLIP_FRAC_WARN_THRESHOLD:
        print(f"WARNING: C6 clip_frac={clip_frac:.3f} >= {CLIP_FRAC_WARN_THRESHOLD} -- distribution may not match Real stats")
    return report


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.save(f, arr)  # np.save() would append ".npy" again if given a bare path with a non-.npy tail
    os.replace(tmp, path)


def _resolve_token66(cola_root: Path, episode_num: int, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    matches = sorted(glob.glob(str(cola_root / str(episode_num) / "*_latent_tokens_66d.npy")))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one 66d token file under {cola_root / str(episode_num)}, found {matches}")
    return Path(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aligned", required=True, help=".npz produced by align_sim_replay.py align")
    parser.add_argument("--episode-num", type=int, required=True, help="cola episode dir number, e.g. 0")
    parser.add_argument("--cola-root", default=DEFAULT_COLA_ROOT)
    parser.add_argument("--token66", default=None, help="override auto-discovered *_latent_tokens_66d.npy")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--stats", default=DEFAULT_STATS)
    args = parser.parse_args()

    cola_root = Path(args.cola_root)
    token66 = _resolve_token66(cola_root, args.episode_num, args.token66)
    tokens = np.load(token66)

    with np.load(args.aligned) as npz:
        aligned = {k: npz[k] for k in npz.files}
    out, n_missing = build_pseudo_proprio(aligned)
    out, hand_report = correct_hand_state_from_action(out, tokens)
    report = self_check(out, token_len=tokens.shape[0], stats_path=args.stats)
    report.update(hand_report)
    report["missing_tail_filled"] = n_missing
    report["episode_num"] = args.episode_num

    ep_name = f"episode_{args.episode_num:06d}.npy"
    out_root = Path(args.out_root)
    proprio_path = out_root / "g1_proprio" / ep_name
    token_path = out_root / "sonic_tokens" / ep_name

    _atomic_write_npy(proprio_path, out.astype(np.float32))
    token_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(token66, token_path)

    report_path = out_root / "g1_proprio" / f"episode_{args.episode_num:06d}.report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {proprio_path} shape={out.shape}")
    print(f"wrote {token_path} (copied from {token66})")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

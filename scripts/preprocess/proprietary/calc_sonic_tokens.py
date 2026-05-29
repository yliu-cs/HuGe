"""Materialize SONIC tokens + 34-D ``unitree_g1_sonic`` proprio for
proprietary LeRobot v2.1 dumps.

The proprietary parquets ship a 64-D ``action.motion_token`` column
(``planner_sonic.onnx`` output) so we never re-encode anything — the script
just concatenates 2-D hand bits onto the SONIC token target and assembles
the GR00T-aligned 34-D proprio block that PropDataset / Kappa consume.

Real proprietary dumps ship per-frame open/close labels under
``action.hand_binary`` (token target, leads observation by ~1 frame) and
``observation.hand_binary_state`` (proprio reading). When the latter is
missing (real/2026-05-16, simple/2026-05-21) the proprio hand bits are
derived from the dex3 joint placeholders via ``binarize_hands``.

Variant detection from ``meta/info.json``:

* ``fps`` -> source sampling rate (resample to ``--target-fps`` if different).
* ``observation.state.names`` -> joint ordering. Both layouts share
  ``[legs(12) | waist(3) | left_arm(7)]`` at the front; the real layout
  interleaves the 7-DoF left-hand block before ``right_arm`` (lhand@22,
  rhand@36), the sim layout packs both arms first (lhand@29, rhand@36).

Proprio 34-D layout (``unitree_g1_sonic``, matches Isaac-GR00T's embodiment
of the same name)::

    [left_leg(6) | right_leg(6) | waist(3) | left_arm(7) | right_arm(7) |
     left_hand(1) | right_hand(1) | projected_gravity(3)]

Output (cache_root defaults to ``<data-root>`` if writable, else
``<data-root>-cache``)::

  <cache_root>/sonic_tokens/episode_<NNNNNN>.npy   # (T_target, 66) float32
  <cache_root>/g1_proprio/episode_<NNNNNN>.npy     # (T_target, 34) float32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")


STATE_LEG_WAIST_END = 15           # state[0:15] -> legs(12) + waist(3) (fixed)


def discover_episodes(data_root: str) -> list[tuple[int, str]]:
    data_dir = os.path.join(data_root, "data")
    out: list[tuple[int, str]] = []
    for chunk_dir in sorted(os.listdir(data_dir)):
        chunk_path = os.path.join(data_dir, chunk_dir)
        if not os.path.isdir(chunk_path):
            continue
        for parquet_file in sorted(os.listdir(chunk_path)):
            if not parquet_file.endswith(".parquet"):
                continue
            stem = parquet_file[len("episode_"):-len(".parquet")]
            out.append((int(stem), os.path.join(chunk_path, parquet_file)))
    return out


def _detect_variant(data_root: str) -> dict:
    """Read ``meta/info.json`` and derive layout/fps/column flags.

    Returns a dict with:
      * ``src_fps``: float source sampling rate.
      * ``larm_start``, ``rarm_start``: int indices into observation.state.
      * ``lhand_start``, ``rhand_start``: int indices for the dex3 hand
        placeholders (real: lhand@22 rhand@36; sim: lhand@29 rhand@36).
      * ``has_hand_binary_action``, ``has_hand_binary_state``: bool.
      * ``has_projected_gravity``: bool.
    """
    info_path = os.path.join(data_root, "meta", "info.json")
    with open(info_path) as fh:
        info = json.load(fh)
    feats = info.get("features", {})
    state_feat = feats.get("observation.state")
    if state_feat is None:
        raise ValueError(f"{info_path}: missing features.observation.state")
    state_names = state_feat.get("names") or []
    try:
        larm_start = state_names.index("left_shoulder_pitch_joint")
        rarm_start = state_names.index("right_shoulder_pitch_joint")
    except ValueError as e:
        raise ValueError(
            f"{info_path}: could not locate shoulder joints in observation.state names; "
            f"is this a proprietary G1 dump?"
        ) from e
    if larm_start != STATE_LEG_WAIST_END:
        raise ValueError(
            f"{info_path}: left_shoulder_pitch_joint at index {larm_start}, "
            f"expected {STATE_LEG_WAIST_END}. New layout — update this script."
        )

    def _first_joint(prefix: str) -> int:
        for idx, name in enumerate(state_names):
            if name.startswith(prefix):
                return idx
        raise ValueError(
            f"{info_path}: no {prefix}* joint found in observation.state.names"
        )

    lhand_start = _first_joint("left_hand_")
    rhand_start = _first_joint("right_hand_")

    return {
        "src_fps": float(info.get("fps") or 50.0),
        "larm_start": int(larm_start),
        "rarm_start": int(rarm_start),
        "lhand_start": int(lhand_start),
        "rhand_start": int(rhand_start),
        "has_hand_binary_action": "action.hand_binary" in feats,
        "has_hand_binary_state": "observation.hand_binary_state" in feats,
        "has_projected_gravity": "observation.projected_gravity" in feats,
    }


def load_episode_arrays(parquet_path: str, variant: dict):
    """Read parquet -> (motion_64, state_43, pg_3, obs_bits|None, act_bits|None).

    ``action.hand_binary`` is the SONIC token target (leads observation by
    ~1 frame); ``observation.hand_binary_state`` is the proprio reading.
    ``observation.projected_gravity`` is required for the sonic proprio's
    last 3 dims — every proprietary dump under
    ``resource/dataset/proprietary`` has it (verified against
    real/2026-05-16, real/2026-05-23, simple/2026-05-21).
    """
    import numpy as np
    import polars as pl

    cols = [
        "action.motion_token",
        "observation.state",
        "observation.projected_gravity",
    ]
    if variant["has_hand_binary_state"]:
        cols.append("observation.hand_binary_state")
    if variant["has_hand_binary_action"]:
        cols.append("action.hand_binary")
    if not variant["has_projected_gravity"]:
        raise ValueError(
            f"{parquet_path}: missing observation.projected_gravity column "
            f"(required for the 34-D sonic proprio layout)."
        )

    df = pl.read_parquet(parquet_path, columns=cols)
    motion = np.asarray(df["action.motion_token"].to_list(), dtype=np.float32)
    state = np.asarray(df["observation.state"].to_list(), dtype=np.float64)
    pg = np.asarray(df["observation.projected_gravity"].to_list(), dtype=np.float32)

    n = state.shape[0]
    if motion.shape != (n, 64):
        raise ValueError(
            f"{parquet_path}: motion_token shape {motion.shape} != ({n}, 64); "
            f"this script assumes action.motion_token is the 64-D SONIC latent."
        )
    if pg.shape != (n, 3):
        raise ValueError(
            f"{parquet_path}: projected_gravity shape {pg.shape} != ({n}, 3)."
        )

    obs_bits = None
    if variant["has_hand_binary_state"]:
        obs_bits = np.asarray(
            df["observation.hand_binary_state"].to_list(), dtype=np.float32,
        )
        if obs_bits.shape != (n, 2):
            raise ValueError(
                f"{parquet_path}: observation.hand_binary_state shape {obs_bits.shape} "
                f"!= ({n}, 2)."
            )
    act_bits = None
    if variant["has_hand_binary_action"]:
        act_bits = np.asarray(
            df["action.hand_binary"].to_list(), dtype=np.float32,
        )
        if act_bits.shape != (n, 2):
            raise ValueError(
                f"{parquet_path}: action.hand_binary shape {act_bits.shape} "
                f"!= ({n}, 2)."
            )
    return motion, state, pg, obs_bits, act_bits


def _resample_to_target_fps(
    motion: "np.ndarray",
    state: "np.ndarray",
    pg: "np.ndarray",
    obs_bits: "np.ndarray | None",
    act_bits: "np.ndarray | None",
    src_fps: float,
    target_fps: float,
):
    """Linear-resample tokens / state / projected_gravity, nearest hand bits.

    Returns ``(motion_r, state_r, pg_r, obs_bits_r|None, act_bits_r|None, n_tgt)``.
    The source timeline is ``[0, (T_src-1)/src_fps]``; the target is sampled
    at ``1/target_fps`` from the same origin, so frame 0 maps to frame 0.
    """
    import numpy as np

    n_src = motion.shape[0]
    if n_src < 2:
        n_tgt = max(int(round(n_src * target_fps / src_fps)), 1)
        motion_r = np.repeat(motion, n_tgt, axis=0)
        state_r = np.repeat(state, n_tgt, axis=0)
        pg_r = np.repeat(pg, n_tgt, axis=0)
        obs_r = None if obs_bits is None else np.repeat(obs_bits, n_tgt, axis=0)
        act_r = None if act_bits is None else np.repeat(act_bits, n_tgt, axis=0)
        return motion_r, state_r, pg_r, obs_r, act_r, n_tgt

    duration = (n_src - 1) / src_fps
    n_tgt = int(round(duration * target_fps)) + 1
    t_src = np.arange(n_src, dtype=np.float64) / src_fps
    t_tgt = np.arange(n_tgt, dtype=np.float64) / target_fps
    t_tgt = np.clip(t_tgt, 0.0, t_src[-1])

    def _interp_cols(arr: "np.ndarray") -> "np.ndarray":
        out = np.empty((n_tgt, arr.shape[1]), dtype=arr.dtype)
        for d in range(arr.shape[1]):
            out[:, d] = np.interp(t_tgt, t_src, arr[:, d])
        return out

    motion_r = _interp_cols(motion)
    state_r = _interp_cols(state)
    pg_r = _interp_cols(pg)

    nn = np.clip(np.round(t_tgt * src_fps).astype(np.int64), 0, n_src - 1)
    obs_r = None if obs_bits is None else obs_bits[nn]
    act_r = None if act_bits is None else act_bits[nn]
    return motion_r, state_r, pg_r, obs_r, act_r, n_tgt


# atomic .npy write reused from the HE/WBT pipeline so an interrupted run
# never leaves a half-written cache file the resume logic would skip over.
from src.data.preprocess import atomic_save_npy  # noqa: E402


def _dump_sonic_tokens(
    motion: "np.ndarray",
    action_hand_bits: "np.ndarray | None",
    path: str,
) -> None:
    """Concat the 2-D ``action.hand_binary`` target onto the 64-D motion token."""
    import numpy as np

    n = motion.shape[0]
    if action_hand_bits is None:
        bits = np.zeros((n, 2), dtype=np.float32)
    else:
        bits = action_hand_bits.astype(np.float32, copy=False)
        if bits.shape != (n, 2):
            raise ValueError(
                f"action_hand_bits shape {bits.shape} != ({n}, 2) at {path}"
            )
    tokens = np.concatenate([motion.astype(np.float32, copy=False), bits], axis=-1)
    atomic_save_npy(tokens, path)


def _dump_proprio(
    state: "np.ndarray",
    pg: "np.ndarray",
    variant: dict,
    obs_hand_bits: "np.ndarray | None",
    hand_thresholds: dict,
    proprio_path: str,
) -> None:
    """Dump 34-D ``unitree_g1_sonic`` proprio.

    Layout: ``[left_leg(6) | right_leg(6) | waist(3) | left_arm(7) |
    right_arm(7) | left_hand(1) | right_hand(1) | projected_gravity(3)]``.

    When ``obs_hand_bits`` is None (2026-05-16 / sim) the per-side 1-D bit
    is derived from the dex3 joints via ``binarize_hands``.
    """
    import numpy as np
    from src.tokenizer.sonic_encoder import binarize_hands

    state32 = state.astype(np.float32, copy=False)
    pg32 = pg.astype(np.float32, copy=False)
    n = state32.shape[0]

    if obs_hand_bits is None:
        # binarize_hands expects a (n, >=14) array with left dex3 at [4:11] and
        # right dex3 at [11:18]. Stack the two 7-D slices to satisfy that
        # contract — the leading 4 cols are unused by binarize_hands so we
        # zero-pad them.
        lhand_start = variant["lhand_start"]
        rhand_start = variant["rhand_start"]
        left_dex = state32[:, lhand_start:lhand_start + 7]
        right_dex = state32[:, rhand_start:rhand_start + 7]
        pad = np.zeros((n, 4), dtype=np.float32)
        stacked = np.concatenate([pad, left_dex, right_dex], axis=-1)
        bits = binarize_hands(stacked, hand_thresholds).astype(np.float32, copy=False)
    else:
        bits = obs_hand_bits.astype(np.float32, copy=False)
    if bits.shape != (n, 2):
        raise ValueError(
            f"hand bits shape {bits.shape} != ({n}, 2) at {proprio_path}"
        )

    larm = variant["larm_start"]
    rarm = variant["rarm_start"]
    parts = [
        state32[:, 0:6],                    # left_leg
        state32[:, 6:12],                   # right_leg
        state32[:, 12:15],                  # waist
        state32[:, larm:larm + 7],          # left_arm
        state32[:, rarm:rarm + 7],          # right_arm
        bits[:, 0:1],                       # left_hand
        bits[:, 1:2],                       # right_hand
        pg32[:, 0:3],                       # projected_gravity
    ]
    proprio = np.concatenate(parts, axis=-1).astype(np.float32)
    if proprio.shape[1] != 34:
        raise ValueError(
            f"assembled proprio shape {proprio.shape} != (*, 34) at {proprio_path}"
        )
    atomic_save_npy(proprio, proprio_path)


def _process_root(
    data_root: str,
    cache_root_override: str | None,
    target_fps: float,
    log_failures: str | None,
    hand_threshold_path: str,
) -> None:
    from tqdm import tqdm
    from src.tokenizer.sonic_encoder import load_hand_thresholds

    cache_root = cache_root_override
    if cache_root is None:
        cache_root = data_root if os.access(data_root, os.W_OK) else f"{data_root}-cache"

    variant = _detect_variant(data_root)
    src_fps = variant["src_fps"]
    needs_resample = abs(src_fps - target_fps) > 1e-3
    layout_label = (
        "sim" if variant["rarm_start"] == 22
        else "real" if variant["rarm_start"] == 29
        else f"rarm@{variant['rarm_start']}"
    )
    print(
        f"[{data_root}] src_fps={src_fps} target_fps={target_fps} "
        f"layout={layout_label} "
        f"hand_binary_state={'real' if variant['has_hand_binary_state'] else 'derive-dex3'} "
        f"hand_binary_action={'real' if variant['has_hand_binary_action'] else 'zero'} "
        f"resample={'yes' if needs_resample else 'no'}"
    )

    if not os.path.exists(hand_threshold_path):
        raise FileNotFoundError(
            f"hand_threshold.json not found at {hand_threshold_path}. "
            f"Pass --hand-threshold-path or run "
            f"scripts/preprocess/humanoid-everyday/calc_hand_thresholds.py first."
        )
    hand_thresholds = load_hand_thresholds(hand_threshold_path)

    out_dir = os.path.join(cache_root, "sonic_tokens")
    proprio_dir = os.path.join(cache_root, "g1_proprio")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(proprio_dir, exist_ok=True)

    print(f"Discovering episodes under {data_root}/data ...")
    all_episodes = discover_episodes(data_root)
    existing_sonic = {fn for fn in os.listdir(out_dir) if fn.endswith(".npy")}
    existing_proprio = {fn for fn in os.listdir(proprio_dir) if fn.endswith(".npy")}

    pending: list[tuple[int, str, bool, bool]] = []
    for idx, path in all_episodes:
        fn = f"episode_{idx:06d}.npy"
        need_sonic = fn not in existing_sonic
        need_proprio = fn not in existing_proprio
        if need_sonic or need_proprio:
            pending.append((idx, path, need_sonic, need_proprio))
    skipped = len(all_episodes) - len(pending)
    print(f"{len(all_episodes)} episodes selected, {skipped} done, {len(pending)} pending.")
    if not pending:
        print(f"Nothing to do. Output: {out_dir}")
        return

    ok = 0
    errs: list[tuple[int, str]] = []
    for ep_idx, parquet_path, need_sonic, need_proprio in tqdm(pending, unit="ep"):
        try:
            motion, state, pg, obs_bits, act_bits = load_episode_arrays(parquet_path, variant)
            if needs_resample:
                motion, state, pg, obs_bits, act_bits, _ = _resample_to_target_fps(
                    motion, state, pg, obs_bits, act_bits, src_fps, target_fps,
                )
            if need_sonic:
                _dump_sonic_tokens(
                    motion, act_bits,
                    os.path.join(out_dir, f"episode_{ep_idx:06d}.npy"),
                )
            if need_proprio:
                _dump_proprio(
                    state, pg, variant, obs_bits, hand_thresholds,
                    os.path.join(proprio_dir, f"episode_{ep_idx:06d}.npy"),
                )
            ok += 1
        except Exception:
            errs.append((ep_idx, traceback.format_exc()))

    if errs:
        log_path = log_failures or os.path.join(out_dir, "failed_episodes.log")
        with open(log_path, "w") as fh:
            for idx, tb in errs:
                fh.write(f"--- episode {idx} ---\n{tb}\n")
        print(f"{len(errs)} failed episode(s). See {log_path}", file=sys.stderr)

    print(f"Done {data_root}. {ok} written, {skipped} already existed, {len(errs)} failed. Output: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root", default=None,
        help="Single proprietary dump root (alias for --data-roots with one entry). "
             "Defaults to resource/dataset/proprietary/real/2026-05-16 when "
             "neither --data-root nor --data-roots is given.",
    )
    parser.add_argument(
        "--data-roots", nargs="*", default=None,
        help="Process multiple proprietary roots in one invocation. Each root "
             "is detected/cached independently; pair with --cache-roots to "
             "override individual cache locations.",
    )
    parser.add_argument(
        "--cache-root", default=None,
        help="Override cache root for the single --data-root case. Defaults to "
             "<data-root> if writable, else <data-root>-cache.",
    )
    parser.add_argument(
        "--cache-roots", nargs="*", default=None,
        help="Per-root cache overrides aligned by index with --data-roots. "
             "Pass an empty string ('') to auto-resolve a single slot.",
    )
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--log-failures", default=None)
    parser.add_argument(
        "--hand-threshold-path",
        default="resource/dataset/humanoid-everyday-cache/meta/hand_threshold.json",
        help="hand_threshold.json from scripts/preprocess/humanoid-everyday/calc_hand_thresholds.py. "
             "Reused across proprietary roots since dex3 joint geometry is shared on G1. "
             "Only consulted on variants without observation.hand_binary_state (2026-05-16, simple/2026-05-21).",
    )
    args = parser.parse_args()

    if args.data_roots:
        roots = list(args.data_roots)
        if args.cache_roots is not None:
            if len(args.cache_roots) != len(roots):
                parser.error(
                    f"--cache-roots ({len(args.cache_roots)}) must match "
                    f"--data-roots ({len(roots)}) in length."
                )
            cache_overrides = [c if c else None for c in args.cache_roots]
        else:
            cache_overrides = [None] * len(roots)
    else:
        single_root = args.data_root or "resource/dataset/proprietary/real/2026-05-16"
        roots = [single_root]
        cache_overrides = [args.cache_root]

    for root, cache_override in zip(roots, cache_overrides):
        _process_root(
            root, cache_override, args.target_fps, args.log_failures,
            args.hand_threshold_path,
        )


if __name__ == "__main__":
    main()

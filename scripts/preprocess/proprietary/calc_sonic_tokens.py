"""Materialize SONIC tokens + 64-D proprio for proprietary LeRobot v2.1 dumps.

The proprietary parquets ship a 64-D ``action.motion_token`` column
(``planner_sonic.onnx`` output) so we never re-encode anything — the script
just concatenates 2-D hand bits, builds the 64-D proprio block, and writes
the two caches downstream PropDataset/PropEpisodeSource consume.

Real proprietary dumps also ship per-frame open/close labels under
``action.hand_binary`` (token target, leads observation by ~1 frame) and
``observation.hand_binary_state`` (proprio reading). Real Dex3 telemetry
lives in these dedicated columns — ``observation.state[22:29]`` /
``[36:43]`` are reserved fake-finger placeholders. Sim dumps lack both
columns and fall back to zero hand_bits.

Supports every dump under ``resource/dataset/proprietary``:

* ``real/2026-05-16``  — 50 Hz, fake fingers (zero hand_bits), real-layout state.
* ``real/2026-05-23``  — 50 Hz, **real** hand_binary columns honored
  (obs bits → proprio, action bits → SONIC token target).
* ``simple/2026-05-21`` — 30 Hz native, sim-layout state (Larm→Rarm→hands),
  resampled to 50 Hz so the cache matches the post-train timeline.

Variant detection is by inspecting ``meta/info.json``:

* ``fps`` → source sampling rate (resample to ``--target-fps`` if different).
* ``observation.state.names`` → joint ordering (Larm always starts at 15;
  Rarm starts at 22 for sim, 29 for real because real interleaves hand
  joints between arms).
* presence of ``action.hand_binary`` / ``observation.hand_binary_state`` in
  ``features`` → use real hand bits, else zeros.

Proprio is assembled into the same 64-D layout HE/WBT use so the existing
G1_WBT proprio stats apply directly:


  [jp(29) | hand_bits(2) | jv(29) | pelvis_quat_xyzw(4)] = 64.

Output (cache_root defaults to ``<data-root>`` if writable, else ``<data-root>-cache``):

  <cache_root>/sonic_tokens/episode_<NNNNNN>.npy   # (T_target, 66) float32
  <cache_root>/g1_proprio/episode_<NNNNNN>.npy     # (T_target, 64) float32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")


STATE_LEG_WAIST_END = 15           # state[0:15] -> jp[0:15] (legs + waist, fixed)
# Per-variant arm slices are resolved at runtime from observation.state.names
# because the real layout interleaves the 7-DoF hand block between Larm/Rarm
# while the sim layout packs both arms first.


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
      * ``has_hand_binary_action``, ``has_hand_binary_state``: bool.
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
        # Both real (15) and sim (15) layouts share the legs+waist block at
        # the front; bail loudly if a future dump diverges so we don't silently
        # drop joints.
        raise ValueError(
            f"{info_path}: left_shoulder_pitch_joint at index {larm_start}, "
            f"expected {STATE_LEG_WAIST_END}. New layout — update this script."
        )
    return {
        "src_fps": float(info.get("fps") or 50.0),
        "larm_start": int(larm_start),
        "rarm_start": int(rarm_start),
        "has_hand_binary_action": "action.hand_binary" in feats,
        "has_hand_binary_state": "observation.hand_binary_state" in feats,
    }


def load_episode_arrays(parquet_path: str, variant: dict):
    """Read parquet -> (motion_64, jp_29, root_quat_wxyz_4, obs_bits|None, act_bits|None).

    ``action.hand_binary`` is the SONIC token target (leads observation by
    ~1 frame); ``observation.hand_binary_state`` is the proprio reading.
    Both columns are read only when present in ``features``; missing
    columns return ``None`` and downstream falls back to zeros.
    """
    import numpy as np
    import polars as pl

    cols = [
        "action.motion_token",
        "observation.state",
        "observation.root_orientation",
    ]
    if variant["has_hand_binary_state"]:
        cols.append("observation.hand_binary_state")
    if variant["has_hand_binary_action"]:
        cols.append("action.hand_binary")

    df = pl.read_parquet(parquet_path, columns=cols)
    motion = np.asarray(df["action.motion_token"].to_list(), dtype=np.float32)
    state = np.asarray(df["observation.state"].to_list(), dtype=np.float64)
    rq = np.asarray(df["observation.root_orientation"].to_list(), dtype=np.float32)

    n = state.shape[0]
    if motion.shape != (n, 64):
        raise ValueError(
            f"{parquet_path}: motion_token shape {motion.shape} != ({n}, 64); "
            f"this script assumes action.motion_token is the 64-D SONIC latent."
        )

    larm_start = variant["larm_start"]
    rarm_start = variant["rarm_start"]
    jp = np.empty((n, 29), dtype=np.float64)
    jp[:, :STATE_LEG_WAIST_END] = state[:, :STATE_LEG_WAIST_END]
    jp[:, 15:22] = state[:, larm_start:larm_start + 7]
    jp[:, 22:29] = state[:, rarm_start:rarm_start + 7]

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
    return motion, jp, rq, obs_bits, act_bits


def _resample_to_target_fps(
    motion: "np.ndarray",
    jp: "np.ndarray",
    rq_wxyz: "np.ndarray",
    obs_bits: "np.ndarray | None",
    act_bits: "np.ndarray | None",
    src_fps: float,
    target_fps: float,
):
    """Linear-resample tokens/jp, SLERP root quat, nearest hand bits.

    Returns ``(motion_r, jp_r, rq_r_wxyz, obs_bits_r|None, act_bits_r|None, target_n)``.
    The source timeline is ``[0, (T_src-1)/src_fps]``; the target timeline
    is sampled at ``1/target_fps`` from the same origin, so frame 0 always
    maps to frame 0 and the deployment-side ``t=k/target_fps`` lookup is
    valid.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp

    n_src = motion.shape[0]
    if n_src < 2:
        # 1-frame episode: nothing to interpolate.
        n_tgt = max(int(round(n_src * target_fps / src_fps)), 1)
        motion_r = np.repeat(motion, n_tgt, axis=0)
        jp_r = np.repeat(jp, n_tgt, axis=0)
        rq_r = np.repeat(rq_wxyz, n_tgt, axis=0)
        obs_r = None if obs_bits is None else np.repeat(obs_bits, n_tgt, axis=0)
        act_r = None if act_bits is None else np.repeat(act_bits, n_tgt, axis=0)
        return motion_r, jp_r, rq_r, obs_r, act_r, n_tgt

    duration = (n_src - 1) / src_fps
    n_tgt = int(round(duration * target_fps)) + 1
    t_src = np.arange(n_src, dtype=np.float64) / src_fps
    t_tgt = np.arange(n_tgt, dtype=np.float64) / target_fps
    # Clamp the tail to t_src[-1] so SciPy's Slerp/np.interp stay in-domain
    # when float rounding pushes t_tgt[-1] a hair past duration.
    t_tgt = np.clip(t_tgt, 0.0, t_src[-1])

    def _interp_cols(arr: "np.ndarray") -> "np.ndarray":
        out = np.empty((n_tgt, arr.shape[1]), dtype=arr.dtype)
        for d in range(arr.shape[1]):
            out[:, d] = np.interp(t_tgt, t_src, arr[:, d])
        return out

    motion_r = _interp_cols(motion)
    jp_r = _interp_cols(jp)

    # SciPy quats are XYZW; convert to/from WXYZ to keep the contract identical
    # to the existing 2026-05-16 cache.
    quat_xyzw_src = rq_wxyz[:, [1, 2, 3, 0]].astype(np.float64, copy=False)
    slerp = Slerp(t_src, Rotation.from_quat(quat_xyzw_src))
    quat_xyzw_tgt = slerp(t_tgt).as_quat().astype(rq_wxyz.dtype, copy=False)
    rq_r = quat_xyzw_tgt[:, [3, 0, 1, 2]]

    nn = np.clip(np.round(t_tgt * src_fps).astype(np.int64), 0, n_src - 1)
    obs_r = None if obs_bits is None else obs_bits[nn]
    act_r = None if act_bits is None else act_bits[nn]
    return motion_r, jp_r, rq_r, obs_r, act_r, n_tgt


# atomic .npy write reused from the HE/WBT pipeline so an interrupted run
# never leaves a half-written cache file the resume logic would skip over.
from src.data.preprocess import atomic_save_npy  # noqa: E402


def _dump_sonic_tokens(
    motion: "np.ndarray",
    action_hand_bits: "np.ndarray | None",
    path: str,
) -> None:
    """Concat the 2-D ``action.hand_binary`` target onto the 64-D motion token.

    ``action_hand_bits`` is the real ``action.hand_binary`` array when the
    parquet has the column; otherwise we fall back to zeros (matches the
    2026-05-16 / sim fake-fingers contract).
    """
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
    jp: "np.ndarray",
    oq_wxyz: "np.ndarray",
    obs_hand_bits: "np.ndarray | None",
    target_fps: float,
    proprio_path: str,
) -> None:
    """Dump 64-D proprio ``[jp | hand_bits | jv | quat_xyzw]`` matching HE/WBT.

    ``obs_hand_bits`` is the real ``observation.hand_binary_state`` array
    when the parquet has the column (2026-05-23 honors it); otherwise
    hand_bits is zeroed to match HE/WBT/2026-05-16/sim semantics.
    """
    import numpy as np
    from src.tokenizer.sonic_encoder import _quat_wxyz_to_xyzw

    jp32 = jp.astype(np.float32, copy=False)
    n = jp32.shape[0]
    if obs_hand_bits is None:
        bits = np.zeros((n, 2), dtype=np.float32)
    else:
        bits = obs_hand_bits.astype(np.float32, copy=False)
        if bits.shape != (n, 2):
            raise ValueError(
                f"obs_hand_bits shape {bits.shape} != ({n}, 2) at {proprio_path}"
            )
    jv = np.zeros_like(jp32)
    if n > 1:
        jv[1:] = (jp32[1:] - jp32[:-1]) * float(target_fps)
    quat_xyzw = _quat_wxyz_to_xyzw(oq_wxyz.astype(np.float32, copy=False)).astype(np.float32)
    proprio = np.concatenate([jp32, bits, jv, quat_xyzw], axis=-1).astype(np.float32)
    atomic_save_npy(proprio, proprio_path)


def _process_root(
    data_root: str,
    cache_root_override: str | None,
    target_fps: float,
    log_failures: str | None,
) -> None:
    from tqdm import tqdm

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
        f"hand_binary={'real' if variant['has_hand_binary_action'] else 'zero'} "
        f"resample={'yes' if needs_resample else 'no'}"
    )

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
            motion, jp, oq, obs_bits, act_bits = load_episode_arrays(parquet_path, variant)
            if needs_resample:
                motion, jp, oq, obs_bits, act_bits, _ = _resample_to_target_fps(
                    motion, jp, oq, obs_bits, act_bits, src_fps, target_fps,
                )
            if need_sonic:
                _dump_sonic_tokens(
                    motion, act_bits,
                    os.path.join(out_dir, f"episode_{ep_idx:06d}.npy"),
                )
            if need_proprio:
                _dump_proprio(
                    jp, oq, obs_bits, float(target_fps),
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
        _process_root(root, cache_override, args.target_fps, args.log_failures)


if __name__ == "__main__":
    main()

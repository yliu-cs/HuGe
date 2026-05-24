"""Materialize SONIC tokens + 64-D proprio for the proprietary 2026-05-16 set.

Unlike HE/WBT, the proprietary parquet **already contains** the 64-D SONIC
latent under the ``action.motion_token`` column (planner_sonic.onnx output)
*and* the per-frame open/close labels under ``action.hand_binary`` (token
target) / ``observation.hand_binary_state`` (proprio). The dataset's
``observation.state[22:29]`` and ``[36:43]`` slots are reserved fake-finger
placeholders that stay zero — the real Dex3 telemetry lives in the dedicated
``*hand_binary*`` columns.

We concat the 2-D action hand bits onto the 64-D motion token and dump
``(T, 66)`` to ``sonic_tokens/``. No ONNX encoder pass.

Proprio is assembled from ``observation.state`` (43-D) +
``observation.root_orientation`` (WXYZ) + ``observation.hand_binary_state``
(2-D) into the same 64-D layout HE/WBT use, so the existing G1_WBT proprio
stats apply directly:

  [jp(29) | hand_bits(2) | jv(29) | pelvis_quat_xyzw(4)] = 64.

Output (cache_root defaults to ``<data-root>`` if writable, else ``<data-root>-cache``):

  <cache_root>/sonic_tokens/episode_<NNNNNN>.npy   # (T, 66) float32
  <cache_root>/g1_proprio/episode_<NNNNNN>.npy     # (T, 64) float32
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")


# Layout of the 43-D ``observation.state`` column (verified from the dataset's
# ``meta/info.json``). MuJoCo joint ordering matches SONIC's body block; the
# 7+7 fake-finger slots are dropped before building jp.
STATE_LEG_WAIST_END = 15          # state[0:15]   -> jp[0:15]   (legs + waist)
STATE_LARM_SLICE = slice(15, 22)  # state[15:22]  -> jp[15:22]  (left arm)
STATE_RARM_SLICE = slice(29, 36)  # state[29:36]  -> jp[22:29]  (right arm)


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


def load_episode_arrays(parquet_path: str):
    """Read parquet -> (sonic_latent_64, jp_29, root_quat_wxyz_4,
    obs_hand_bits_2, action_hand_bits_2).

    ``action.hand_binary`` is the token-side target (leads observation by ~1
    frame); ``observation.hand_binary_state`` is the proprio reading. Both
    columns are already 0/1 float — no thresholding needed.
    """
    import numpy as np
    import polars as pl

    df = pl.read_parquet(
        parquet_path,
        columns=[
            "action.motion_token",
            "observation.state",
            "observation.root_orientation",
            "observation.hand_binary_state",
            "action.hand_binary",
        ],
    )
    motion = np.asarray(df["action.motion_token"].to_list(), dtype=np.float32)
    state = np.asarray(df["observation.state"].to_list(), dtype=np.float64)
    rq = np.asarray(df["observation.root_orientation"].to_list(), dtype=np.float32)
    obs_bits = np.asarray(
        df["observation.hand_binary_state"].to_list(), dtype=np.float32
    )
    act_bits = np.asarray(df["action.hand_binary"].to_list(), dtype=np.float32)

    n = state.shape[0]
    if motion.shape != (n, 64):
        raise ValueError(
            f"{parquet_path}: motion_token shape {motion.shape} != ({n}, 64); "
            f"this script assumes action.motion_token is the 64-D SONIC latent."
        )
    if obs_bits.shape != (n, 2):
        raise ValueError(
            f"{parquet_path}: observation.hand_binary_state shape {obs_bits.shape} "
            f"!= ({n}, 2)"
        )
    if act_bits.shape != (n, 2):
        raise ValueError(
            f"{parquet_path}: action.hand_binary shape {act_bits.shape} "
            f"!= ({n}, 2)"
        )
    jp = np.empty((n, 29), dtype=np.float64)
    jp[:, :STATE_LEG_WAIST_END] = state[:, :STATE_LEG_WAIST_END]
    jp[:, 15:22] = state[:, STATE_LARM_SLICE]
    jp[:, 22:29] = state[:, STATE_RARM_SLICE]
    return motion, jp, rq, obs_bits, act_bits


# atomic .npy write reused from the HE/WBT pipeline so an interrupted run
# never leaves a half-written cache file the resume logic would skip over.
from src.data.preprocess import atomic_save_npy  # noqa: E402


def _dump_sonic_tokens(
    motion: "np.ndarray",
    action_hand_bits: "np.ndarray",
    path: str,
) -> None:
    """Concat the 2-D ``action.hand_binary`` target onto the 64-D motion token;
    save (n, 66)."""
    import numpy as np

    n = motion.shape[0]
    bits = action_hand_bits.astype(np.float32, copy=False)
    if bits.shape != (n, 2):
        raise ValueError(f"action_hand_bits shape {bits.shape} != ({n}, 2)")
    tokens = np.concatenate(
        [motion.astype(np.float32, copy=False), bits], axis=-1
    )
    atomic_save_npy(tokens, path)


def _dump_proprio(
    jp: "np.ndarray",
    oq_wxyz: "np.ndarray",
    obs_hand_bits: "np.ndarray",
    target_fps: float,
    proprio_path: str,
) -> None:
    """Dump 64-D proprio ``[jp | hand_bits | jv | quat_xyzw]`` matching HE/WBT,
    where ``hand_bits`` comes from ``observation.hand_binary_state``."""
    import numpy as np
    from src.tokenizer.sonic_encoder import _quat_wxyz_to_xyzw

    jp32 = jp.astype(np.float32, copy=False)
    n = jp32.shape[0]
    bits = obs_hand_bits.astype(np.float32, copy=False)
    if bits.shape != (n, 2):
        raise ValueError(f"obs_hand_bits shape {bits.shape} != ({n}, 2)")
    jv = np.zeros_like(jp32)
    if n > 1:
        jv[1:] = (jp32[1:] - jp32[:-1]) * float(target_fps)
    quat_xyzw = _quat_wxyz_to_xyzw(oq_wxyz.astype(np.float32, copy=False)).astype(np.float32)
    proprio = np.concatenate([jp32, bits, jv, quat_xyzw], axis=-1).astype(np.float32)
    atomic_save_npy(proprio, proprio_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root", default="resource/dataset/proprietary/2026-05-16",
    )
    parser.add_argument(
        "--cache-root", default=None,
        help="Defaults to <data-root> if writable, else <data-root>-cache.",
    )
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--log-failures", default=None)
    args = parser.parse_args()

    from tqdm import tqdm

    cache_root = args.cache_root
    if cache_root is None:
        cache_root = args.data_root if os.access(args.data_root, os.W_OK) else f"{args.data_root}-cache"

    src_fps = None
    info_path = os.path.join(args.data_root, "meta", "info.json")
    try:
        with open(info_path) as f:
            src_fps = float(json.load(f)["fps"])
    except Exception:
        pass
    if src_fps is not None and abs(src_fps - args.target_fps) > 1e-3:
        print(
            f"[WARN] source fps={src_fps} != target fps={args.target_fps}; "
            f"this script does not resample (planner tokens are already on the "
            f"deployment timeline)."
        )

    out_dir = os.path.join(cache_root, "sonic_tokens")
    proprio_dir = os.path.join(cache_root, "g1_proprio")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(proprio_dir, exist_ok=True)

    print(f"Discovering episodes under {args.data_root}/data ...")
    all_episodes = discover_episodes(args.data_root)
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
            motion, jp, oq, obs_bits, act_bits = load_episode_arrays(parquet_path)
            if need_sonic:
                _dump_sonic_tokens(
                    motion, act_bits,
                    os.path.join(out_dir, f"episode_{ep_idx:06d}.npy"),
                )
            if need_proprio:
                _dump_proprio(
                    jp, oq, obs_bits, float(args.target_fps),
                    os.path.join(proprio_dir, f"episode_{ep_idx:06d}.npy"),
                )
            ok += 1
        except Exception:
            errs.append((ep_idx, traceback.format_exc()))

    if errs:
        log_path = args.log_failures or os.path.join(out_dir, "failed_episodes.log")
        with open(log_path, "w") as fh:
            for idx, tb in errs:
                fh.write(f"--- episode {idx} ---\n{tb}\n")
        print(f"{len(errs)} failed episode(s). See {log_path}", file=sys.stderr)

    print(f"Done. {ok} written, {skipped} already existed, {len(errs)} failed. Output: {out_dir}")


if __name__ == "__main__":
    main()

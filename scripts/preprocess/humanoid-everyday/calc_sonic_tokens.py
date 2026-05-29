"""Precompute SONIC latent tokens for all episodes and save as .npy files.

Multi-GPU parallel: spawns one worker process per (gpu, workers_per_gpu) slot.
Each worker runs an onnxruntime-gpu session on its assigned GPU. Episodes are
distributed via a multiprocessing queue; the main process drives a single tqdm
bar off a result queue. Uses atomic `.npy.tmp` → `.npy` rename so interrupted
runs resume cleanly via the existing `os.path.exists` skip.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import traceback
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")


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


def load_robot_type_map(data_root: str) -> dict[int, str]:
    """Read meta/episodes.jsonl and return ``{episode_index: robot_type}``.

    humanoid-everyday is a G1+H1 mix; we filter on this to avoid encoding two
    incompatible joint layouts through the same SONIC encoder. The 29-dim
    ``jp`` vector with arm starting at index 15 only matches G1 (leg=15,
    arm=14); H1 (leg=13, arm=14) leaves indices 13/14 as zeros and shifts
    semantics, which silently poisons the token distribution.
    """
    import json

    path = os.path.join(data_root, "meta", "episodes.jsonl")
    out: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            out[int(entry["episode_index"])] = str(entry.get("robot_type", "")).lower()
    return out


def load_episode_arrays(parquet_path: str):
    import numpy as np
    import polars as pl

    df = pl.read_parquet(parquet_path)
    n = len(df)
    leg = df["observation.leg_joints"].to_list()
    arm = df["observation.arm_joints"].to_list()
    jp = np.zeros((n, 29), dtype=np.float64)
    for i in range(n):
        l = np.asarray(leg[i], dtype=np.float64)
        a = np.asarray(arm[i], dtype=np.float64)
        jp[i, :l.shape[0]] = l
        jp[i, 15:15 + a.shape[0]] = a
    op = np.asarray(df["observation.odometry.position"].to_list(), dtype=np.float32)
    oq = np.asarray(df["observation.odometry.quat"].to_list(), dtype=np.float32)
    ig = np.asarray(df["observation.imu.gyroscope"].to_list(), dtype=np.float32)
    hj = np.asarray(df["observation.hand_joints"].to_list(), dtype=np.float32)
    return jp, op, oq, ig, hj


# Resampling helpers live in src.data.preprocess so HE/WBT/future sources stay
# bit-identical (the SONIC token distribution is sensitive to upsampling).
from src.data.preprocess import (  # noqa: E402
    atomic_save_npy,
    projected_gravity_from_quat_wxyz as _projected_gravity_from_quat_wxyz,
    resample_linear as _resample_linear,
    resample_quat_wxyz as _resample_quat_wxyz,
)


def _dump_proprio(
    jp: "np.ndarray",
    oq: "np.ndarray",
    hj: "np.ndarray",
    hand_thresholds: tuple[float, float, float, float],
    proprio_path: str,
) -> None:
    """Dump 34-D ``unitree_g1_sonic`` proprio.

    Layout: ``[left_leg(6) | right_leg(6) | waist(3) | left_arm(7) |
    right_arm(7) | left_hand(1) | right_hand(1) | projected_gravity(3)]``.

    HE's ``jp`` is already a 29-D G1-ordered vector: ``[legs(12) | waist(3) |
    left_arm(7) | right_arm(7)]``. Hand open/close bits come from
    ``binarize_hands(observation.hand_joints)``; projected gravity is the
    body-frame ``R(odom_quat).T @ [0, 0, -1]`` (HE has no IMU-derived
    projected_gravity column, only the odometry quaternion).
    """
    import numpy as np
    from src.tokenizer.sonic_encoder import binarize_hands

    jp32 = jp.astype(np.float32, copy=False)
    hand_bits = binarize_hands(hj, hand_thresholds).astype(np.float32, copy=False)  # (n, 2)
    pg = _projected_gravity_from_quat_wxyz(oq.astype(np.float32, copy=False))         # (n, 3)

    parts = [
        jp32[:, 0:6],            # left_leg
        jp32[:, 6:12],           # right_leg
        jp32[:, 12:15],          # waist
        jp32[:, 15:22],          # left_arm
        jp32[:, 22:29],          # right_arm
        hand_bits[:, 0:1],       # left_hand
        hand_bits[:, 1:2],       # right_hand
        pg[:, 0:3],              # projected_gravity
    ]
    proprio = np.concatenate(parts, axis=-1).astype(np.float32)
    if proprio.shape[1] != 34:
        raise ValueError(f"assembled proprio shape {proprio.shape} != (*, 34) at {proprio_path}")
    atomic_save_npy(proprio, proprio_path)


def worker_main(
    gpu_id: int | None,
    work_q: mp.Queue,
    result_q: mp.Queue,
    encoder_path: str,
    out_dir: str,
    proprio_dir: str,
    hand_thresholds: tuple[float, float, float, float],
    source_fps: float,
    target_fps: float,
) -> None:
    os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import numpy as np
    import torch  # preload cuDNN so ORT CUDAExecutionProvider finds it
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    from src.tokenizer.sonic_encoder import SonicEncoder

    try:
        encoder = SonicEncoder(encoder_path, device_id=gpu_id)
    except Exception:
        result_q.put(("fatal", os.getpid(), traceback.format_exc()))
        return

    while True:
        item = work_q.get()
        if item is None:
            return
        ep_idx, parquet_path, need_sonic = item
        try:
            jp, op, oq, ig, hj = load_episode_arrays(parquet_path)
            n_src = jp.shape[0]
            n_tgt = max(1, int(round(n_src * target_fps / source_fps)))
            if n_tgt != n_src:
                jp = _resample_linear(jp, n_src, n_tgt)
                op = _resample_linear(op, n_src, n_tgt)
                oq = _resample_quat_wxyz(oq, n_src, n_tgt)
                ig = _resample_linear(ig, n_src, n_tgt)
                hj = _resample_linear(hj, n_src, n_tgt)

            if need_sonic:
                tokens = encoder.encode_episode(
                    jp, oq, ig, hj, hand_thresholds, fps=target_fps,
                )
                final = os.path.join(out_dir, f"episode_{ep_idx:06d}.npy")
                atomic_save_npy(tokens, final)
                token_len = len(tokens)
            else:
                token_len = n_tgt

            proprio_path = os.path.join(proprio_dir, f"episode_{ep_idx:06d}.npy")
            if not os.path.exists(proprio_path):
                _dump_proprio(
                    jp, oq, hj, hand_thresholds, proprio_path,
                )
            result_q.put(("ok", ep_idx, token_len))
        except Exception:
            result_q.put(("err", ep_idx, traceback.format_exc()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute SONIC tokens (multi-GPU)")
    parser.add_argument("--data-root", default="resource/dataset/humanoid-everyday")
    parser.add_argument(
        "--cache-root", default=None,
        help="Writable cache dir for sonic_tokens/ and meta/. Defaults to <data-root> if "
             "writable, else <data-root>-cache.",
    )
    parser.add_argument("--sonic-model-dir", default="resource/model/sonic")
    parser.add_argument(
        "--gpus", type=int, default=None,
        help="Number of GPUs to use (default: all visible GPUs; 0 to force CPU).",
    )
    parser.add_argument(
        "--workers-per-gpu", type=int, default=8,
        help="ORT sessions per GPU (increase to squeeze more throughput from each card).",
    )
    parser.add_argument(
        "--log-failures", default=None,
        help="Path to write failed episode indices + tracebacks (default: <out_dir>/failed_episodes.log).",
    )
    parser.add_argument(
        "--source-fps", type=float, default=None,
        help="Native sample rate of the parquet rows. Default: read from "
             "<data-root>/meta/info.json:fps.",
    )
    parser.add_argument(
        "--target-fps", type=float, default=50.0,
        help="Resample inputs to this rate before encoding so output tokens land on "
             "a 50 Hz timeline (matching EgoDex). Default 50.",
    )
    parser.add_argument(
        "--robot-type", default="g1",
        help="Filter episodes by meta/episodes.jsonl:robot_type before encoding. "
             "Default 'g1' (skip H1 — its 13D leg + 12D hand layout is incompatible "
             "with the 29D jp / shared hand_threshold pipeline). Pass 'all' to disable.",
    )
    args = parser.parse_args()

    import json
    import torch
    from tqdm import tqdm
    from src.tokenizer.sonic_encoder import load_hand_thresholds

    cache_root = args.cache_root
    if cache_root is None:
        cache_root = args.data_root if os.access(args.data_root, os.W_OK) else f"{args.data_root}-cache"

    source_fps = args.source_fps
    if source_fps is None:
        info_path = os.path.join(args.data_root, "meta", "info.json")
        try:
            with open(info_path) as f:
                source_fps = float(json.load(f)["fps"])
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise SystemExit(
                f"Could not read fps from {info_path}: {e}. Pass --source-fps explicitly."
            )
    target_fps = float(args.target_fps)
    if source_fps <= 0 or target_fps <= 0:
        raise SystemExit(f"fps must be positive, got source={source_fps} target={target_fps}")

    threshold_path = os.path.join(cache_root, "meta", "hand_threshold.json")
    if not os.path.exists(threshold_path):
        raise FileNotFoundError(
            f"Hand threshold file not found at {threshold_path}. "
            f"Run scripts/compute_hand_thresholds.py first (with the same --cache-root)."
        )
    hand_thresholds = load_hand_thresholds(threshold_path)
    print(
        f"Loaded hand thresholds: "
        f"left=({hand_thresholds[0]:.4f}, {hand_thresholds[1]:.4f}), "
        f"right=({hand_thresholds[2]:.4f}, {hand_thresholds[3]:.4f}) [low, high]"
    )
    if target_fps != source_fps:
        print(
            f"Resampling {source_fps:g} Hz -> {target_fps:g} Hz before encoding. "
            f"Skip-if-exists is keyed by filename only -- if sonic_tokens/ already "
            f"holds tokens from a different rate, clear it first."
        )

    encoder_path = os.path.join(args.sonic_model_dir, "model_encoder.onnx")
    out_dir = os.path.join(cache_root, "sonic_tokens")
    os.makedirs(out_dir, exist_ok=True)
    # ``g1_proprio/`` references the G1 *robot's* joint state cache; the
    # directory name is preserved across the G1-expert removal.
    proprio_dir = os.path.join(cache_root, "g1_proprio")
    os.makedirs(proprio_dir, exist_ok=True)

    if args.gpus is None:
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = max(0, args.gpus)

    if n_gpus == 0:
        gpu_ids: list[int | None] = [None]
        workers_per_gpu = max(1, args.workers_per_gpu)
    else:
        gpu_ids = list(range(n_gpus))
        workers_per_gpu = max(1, args.workers_per_gpu)
    n_workers = len(gpu_ids) * workers_per_gpu

    print(f"Discovering episodes under {args.data_root}/data ...")
    all_episodes = discover_episodes(args.data_root)
    discovered = len(all_episodes)

    robot_type_filter = args.robot_type.lower()
    if robot_type_filter != "all":
        rt_map = load_robot_type_map(args.data_root)
        all_episodes = [(i, p) for i, p in all_episodes if rt_map.get(i) == robot_type_filter]
        dropped = discovered - len(all_episodes)
        print(
            f"Robot-type filter='{robot_type_filter}': kept {len(all_episodes)} / "
            f"{discovered} episodes ({dropped} dropped)."
        )

    existing_sonic = {fn for fn in os.listdir(out_dir) if fn.endswith(".npy")}
    existing_proprio = {fn for fn in os.listdir(proprio_dir) if fn.endswith(".npy")}

    # An episode is pending if either of {sonic, proprio} is missing.
    # ``need_sonic=False`` lets workers skip the ONNX session for episodes
    # where only the lightweight proprio dump is missing — that can be
    # produced from parquet alone, no GPU needed.
    pending: list[tuple[int, str, bool]] = []
    for idx, path in all_episodes:
        fn = f"episode_{idx:06d}.npy"
        need_sonic = fn not in existing_sonic
        need_proprio = fn not in existing_proprio
        if need_sonic or need_proprio:
            pending.append((idx, path, need_sonic))
    skipped = len(all_episodes) - len(pending)
    print(
        f"{len(all_episodes)} episodes selected, {skipped} already done, "
        f"{len(pending)} pending. Spawning {n_workers} workers "
        f"({'CPU' if n_gpus == 0 else f'{n_gpus} GPU × {workers_per_gpu}'})."
    )
    if not pending:
        print(f"Nothing to do. Output: {out_dir}")
        return

    ctx = mp.get_context("spawn")
    work_q: mp.Queue = ctx.Queue(maxsize=max(4 * n_workers, 32))
    result_q: mp.Queue = ctx.Queue()

    workers: list[mp.Process] = []
    for w in range(n_workers):
        gpu = gpu_ids[w % len(gpu_ids)]
        p = ctx.Process(
            target=worker_main,
            args=(
                gpu, work_q, result_q, encoder_path, out_dir,
                proprio_dir, hand_thresholds,
                source_fps, target_fps,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)

    producer = ctx.Process(target=_producer_main, args=(work_q, pending, len(workers)))
    producer.start()

    stop_flag = {"stop": False}

    def _handle_sigint(*_):
        if not stop_flag["stop"]:
            stop_flag["stop"] = True
            print("\n[!] SIGINT received, draining queue and stopping workers...", file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_sigint)

    ok = 0
    errs: list[tuple[int, str]] = []
    fatal: list[tuple[int, str]] = []

    pbar = tqdm(total=len(pending), desc="Encoding episodes", unit="ep")
    done = 0
    while done < len(pending):
        if stop_flag["stop"]:
            break
        try:
            msg = result_q.get(timeout=1.0)
        except Exception:
            # check worker liveness
            if not any(p.is_alive() for p in workers):
                print("[!] All workers exited.", file=sys.stderr)
                break
            continue
        kind = msg[0]
        if kind == "ok":
            ok += 1
            done += 1
            pbar.update(1)
        elif kind == "err":
            errs.append((msg[1], msg[2]))
            done += 1
            pbar.update(1)
        elif kind == "fatal":
            fatal.append((msg[1], msg[2]))
            # worker died at startup; nothing to recover from for that slot
    pbar.close()

    if stop_flag["stop"]:
        # drain remaining items, push sentinels
        try:
            while True:
                work_q.get_nowait()
        except Exception:
            pass
        for _ in workers:
            try:
                work_q.put_nowait(None)
            except Exception:
                pass

    producer.join(timeout=5.0)
    if producer.is_alive():
        producer.terminate()
    for p in workers:
        p.join(timeout=10.0)
    for p in workers:
        if p.is_alive():
            p.terminate()

    if errs or fatal:
        log_path = args.log_failures or os.path.join(out_dir, "failed_episodes.log")
        with open(log_path, "w") as fh:
            for idx, tb in errs:
                fh.write(f"--- episode {idx} ---\n{tb}\n")
            for pid, tb in fatal:
                fh.write(f"--- worker pid {pid} (fatal) ---\n{tb}\n")
        print(f"{len(errs)} failed episode(s), {len(fatal)} fatal worker(s). See {log_path}", file=sys.stderr)

    print(
        f"Done. {ok} encoded, {skipped} already existed, {len(errs)} failed. "
        f"Output: {out_dir}"
    )


def _producer_main(work_q: mp.Queue, pending: list[tuple], n_workers: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for item in pending:
            work_q.put(item)
        for _ in range(n_workers):
            work_q.put(None)
    except Exception:
        pass


if __name__ == "__main__":
    main()

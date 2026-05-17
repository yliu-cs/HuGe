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
    build_g1_action_23,
    resample_linear as _resample_linear,
    resample_quat_wxyz as _resample_quat_wxyz,
)


def _dump_g1_action_and_proprio(
    jp: "np.ndarray",
    op: "np.ndarray",
    oq: "np.ndarray",
    hj: "np.ndarray",
    hand_thresholds: tuple[float, float, float, float],
    target_fps: float,
    g1_action_path: str,
    g1_proprio_path: str,
) -> None:
    """Dump 23-D G1 actions (arms + hand_bits + loco_cmd) and 64-D proprio
    (full-body jp + hand_bits + jv + odom_quat_xyzw) on the SONIC timeline.

    23-D action layout (see ``build_g1_action_23``):
        [0:14] arms, [14:16] hand_bits, [16:19] navigate_cmd,
        [19:20] base_height, [20:23] rpy_cmd(roll, pitch, 0).

    Legs+waist are dropped from the action target — WBC drives them at
    deployment. They remain in the 64-D proprio so the model sees full body
    state as observation; that separation matches the deployment surface.
    """
    import numpy as np
    from src.tokenizer.sonic_encoder import _quat_wxyz_to_xyzw, binarize_hands

    hand_bits = binarize_hands(hj, hand_thresholds).astype(np.float32)  # (n, 2)
    jp32 = jp.astype(np.float32, copy=False)

    g1_action = build_g1_action_23(jp32, hand_bits, op, oq, fps=target_fps)
    atomic_save_npy(g1_action, g1_action_path)

    n = jp32.shape[0]
    jv = np.zeros_like(jp32)
    if n > 1:
        jv[1:] = (jp32[1:] - jp32[:-1]) * float(target_fps)
    quat_xyzw = _quat_wxyz_to_xyzw(oq.astype(np.float32, copy=False)).astype(np.float32)
    proprio = np.concatenate(
        [jp32, hand_bits, jv, quat_xyzw], axis=-1
    ).astype(np.float32)  # (n, 29 + 2 + 29 + 4) = (n, 64)
    atomic_save_npy(proprio, g1_proprio_path)


def worker_main(
    gpu_id: int | None,
    work_q: mp.Queue,
    result_q: mp.Queue,
    encoder_path: str,
    out_dir: str,
    g1_action_dir: str,
    g1_proprio_dir: str,
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

            g1_action_path = os.path.join(g1_action_dir, f"episode_{ep_idx:06d}.npy")
            g1_proprio_path = os.path.join(g1_proprio_dir, f"episode_{ep_idx:06d}.npy")
            if not (os.path.exists(g1_action_path) and os.path.exists(g1_proprio_path)):
                _dump_g1_action_and_proprio(
                    jp, op, oq, hj, hand_thresholds, target_fps,
                    g1_action_path, g1_proprio_path,
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
    g1_action_dir = os.path.join(cache_root, "g1_actions")
    g1_proprio_dir = os.path.join(cache_root, "g1_proprio")
    os.makedirs(g1_action_dir, exist_ok=True)
    os.makedirs(g1_proprio_dir, exist_ok=True)

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
    existing_g1_action = {fn for fn in os.listdir(g1_action_dir) if fn.endswith(".npy")}
    existing_g1_proprio = {fn for fn in os.listdir(g1_proprio_dir) if fn.endswith(".npy")}

    # An episode is pending if any of {sonic, g1_action, g1_proprio} is missing.
    # ``need_sonic=False`` lets workers skip the ONNX session for episodes where
    # only the lightweight g1_action/proprio dumps are missing — those can be
    # produced from parquet alone, no GPU needed.
    pending: list[tuple[int, str, bool]] = []
    for idx, path in all_episodes:
        fn = f"episode_{idx:06d}.npy"
        need_sonic = fn not in existing_sonic
        need_g1 = (fn not in existing_g1_action) or (fn not in existing_g1_proprio)
        if need_sonic or need_g1:
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
                g1_action_dir, g1_proprio_dir, hand_thresholds,
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

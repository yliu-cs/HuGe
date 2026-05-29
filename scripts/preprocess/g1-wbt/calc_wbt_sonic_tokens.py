"""Precompute SONIC latent tokens for every G1-WBT episode.

Output:

  <cache_root>/sonic_tokens/<subset_name>/episode_<NNNNNN>.npy   # [T_50hz, 66] float32
  <cache_root>/meta/episodes_meta.parquet                        # cross-subset index

The token file format (T rows × 66 dims = 64 SONIC + 2 hand bits) matches HE
and EgoDex exactly so the dataset class produces samples that drop straight
into ``KappaCollator`` without conditional logic.

The episodes_meta.parquet table is consumed by ``WBTDataset`` to:
  * map a global ``episode_id`` -> on-disk token file + video file/offset,
  * carry the per-episode ``task`` text (LLM prompt),
  * tag the ``hand_type`` for downstream debugging / per-type analysis.

Robustness:
  * Atomic ``.npy.tmp.<pid>`` -> ``os.replace`` so interrupted runs don't leave
    half-written tokens that would silently fail mmap.
  * Skip-if-exists keyed by per-subset filename, so re-running after a partial
    failure resumes cheaply.
  * Multi-GPU: one ORT session per (gpu, worker) slot, episodes dispatched
    through a multiprocessing queue.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import traceback

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from _subsets import (  # noqa: E402
    Subset,
    compute_hand_bits,
    list_subsets,
    load_episode_rows,
    load_subset_episode_meta,
)

# Resampling + atomic .npy write live in src.data.preprocess so HE/WBT/future
# sources stay bit-identical (the SONIC token distribution is sensitive to
# upsampling), and so the dataset-side path manipulation hack is gone.
from src.data.preprocess import (  # noqa: E402
    atomic_save_npy,
    projected_gravity_from_quat_wxyz as _projected_gravity_from_quat_wxyz,
    resample_linear as _resample_linear,
    resample_quat_wxyz as _resample_quat_wxyz,
)


# ---------------------------------------------------------------------------
# Per-episode encode
# ---------------------------------------------------------------------------

def _encode_one(
    encoder,
    rows: dict,
    hand_type: str,
    source_fps: float,
    target_fps: float,
    need_sonic: bool,
):
    """Resample to the SONIC timeline, derive open/close bits straight from the
    dataset's documented hand_state range, and (optionally) run the SONIC encoder.

    Returns ``(tokens_or_None, jp, op, oq, hand_bits, n_tgt)``. Caller uses
    ``jp``, ``oq``, ``hand_bits`` to dump the 64-D proprio without needing to
    redo resampling/binarization.
    """
    import numpy as np

    jp = rows["joint_pos"]
    op = rows["odom_pos"]
    oq = rows["odom_quat"]
    hs = rows["hand_state"]
    n_src = jp.shape[0]
    n_tgt = max(1, int(round(n_src * target_fps / source_fps)))
    if n_tgt != n_src:
        jp = _resample_linear(jp, n_src, n_tgt)
        op = _resample_linear(op, n_src, n_tgt)
        oq = _resample_quat_wxyz(oq, n_src, n_tgt)
        hs = _resample_linear(hs, n_src, n_tgt)

    hand_bits = compute_hand_bits(hs, hand_type)
    tokens = None
    if need_sonic:
        imu_gyro = np.zeros((n_tgt, 3), dtype=np.float32)  # encoder ignores gyro (verified)
        tokens = encoder.encode_episode_with_bits(
            jp, oq, imu_gyro, hand_bits, fps=target_fps,
        )
    return tokens, jp, op, oq, hand_bits, n_tgt


def _dump_proprio(
    jp: "np.ndarray",
    oq: "np.ndarray",
    hand_bits: "np.ndarray",
    proprio_path: str,
) -> None:
    """Dump 34-D ``unitree_g1_sonic`` proprio.

    Layout: ``[left_leg(6) | right_leg(6) | waist(3) | left_arm(7) |
    right_arm(7) | left_hand(1) | right_hand(1) | projected_gravity(3)]``.

    WBT's ``jp`` is already a 29-D G1-ordered vector matching HE. Hand
    open/close bits come from ``compute_hand_bits(hand_state, hand_type)``
    (per-side range thresholds documented in ``_subsets.py``); projected
    gravity is the body-frame ``R(odom_quat).T @ [0, 0, -1]``.
    """
    import numpy as np

    jp32 = jp.astype(np.float32, copy=False)
    bits = hand_bits.astype(np.float32, copy=False)
    pg = _projected_gravity_from_quat_wxyz(oq.astype(np.float32, copy=False))

    parts = [
        jp32[:, 0:6],            # left_leg
        jp32[:, 6:12],           # right_leg
        jp32[:, 12:15],          # waist
        jp32[:, 15:22],          # left_arm
        jp32[:, 22:29],          # right_arm
        bits[:, 0:1],            # left_hand
        bits[:, 1:2],            # right_hand
        pg[:, 0:3],              # projected_gravity
    ]
    proprio = np.concatenate(parts, axis=-1).astype(np.float32)
    if proprio.shape[1] != 34:
        raise ValueError(f"assembled proprio shape {proprio.shape} != (*, 34) at {proprio_path}")
    atomic_save_npy(proprio, proprio_path)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_main(
    gpu_id: int | None,
    work_q: "mp.Queue",
    result_q: "mp.Queue",
    encoder_path: str,
    out_root: str,
    proprio_root: str,
    data_root: str,
    source_fps: float,
    target_fps: float,
) -> None:
    os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    import numpy as np
    import torch  # preload cuDNN before ORT picks it up
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
    from src.tokenizer.sonic_encoder import SonicEncoder

    try:
        encoder = SonicEncoder(encoder_path, device_id=gpu_id)
    except Exception:
        result_q.put(("fatal", os.getpid(), traceback.format_exc()))
        return

    # Cache one Subset object per name; constructed lazily from the dispatched item.
    subsets_by_name: dict[str, Subset] = {s.name: s for s in list_subsets(data_root)}

    while True:
        item = work_q.get()
        if item is None:
            return
        subset_name, episode_index, data_chunk, data_file, hand_type, need_sonic = item
        try:
            sub = subsets_by_name[subset_name]
            rows = load_episode_rows(data_root, sub, data_chunk, data_file, episode_index)
            tokens, jp, op, oq, hand_bits, n_tgt = _encode_one(
                encoder, rows, hand_type, source_fps, target_fps, need_sonic,
            )
            if need_sonic:
                out_dir = os.path.join(out_root, subset_name)
                final = os.path.join(out_dir, f"episode_{episode_index:06d}.npy")
                atomic_save_npy(tokens, final)

            proprio_dir = os.path.join(proprio_root, subset_name)
            os.makedirs(proprio_dir, exist_ok=True)
            proprio_path = os.path.join(proprio_dir, f"episode_{episode_index:06d}.npy")
            if not os.path.exists(proprio_path):
                _dump_proprio(
                    jp, oq, hand_bits, proprio_path,
                )
            result_q.put(("ok", subset_name, episode_index, int(n_tgt)))
        except Exception:
            result_q.put((
                "err", subset_name, episode_index, traceback.format_exc()
            ))


def _producer_main(work_q: "mp.Queue", pending: list[tuple], n_workers: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for item in pending:
            work_q.put(item)
        for _ in range(n_workers):
            work_q.put(None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_episode_meta(data_root: str, subsets: list[Subset]):
    """Concatenate per-subset episode metadata into a single DataFrame and
    assign a stable global ``episode_id`` (subset-name-sorted, then
    ``episode_index``)."""
    import polars as pl
    from tqdm import tqdm

    frames: list[pl.DataFrame] = []
    for sub in tqdm(subsets, desc="Reading episode meta"):
        df = load_subset_episode_meta(sub, data_root)
        df = df.with_columns([
            pl.lit(sub.name).alias("subset"),
            pl.lit(sub.hand_type).alias("hand_type"),
            pl.lit(sub.video_key).alias("video_key"),
            pl.lit(sub.nested_dir or "").alias("nested_dir"),
        ])
        frames.append(df)
    df = pl.concat(frames).sort(["subset", "episode_index"])
    df = df.with_row_index("episode_id")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="resource/dataset/G1_WBT")
    parser.add_argument(
        "--cache-root", default=None,
        help="Writable output dir. Defaults to <data-root> if writable else <data-root>-cache.",
    )
    parser.add_argument("--sonic-model-dir", default="resource/model/sonic")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs (default: all visible; 0 = CPU).")
    parser.add_argument("--workers-per-gpu", type=int, default=8)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=50.0,
                        help="Resample rate before encoding (matches HE/EgoDex 50 Hz timeline).")
    parser.add_argument("--subsets", nargs="*", default=None,
                        help="Restrict to these subset directory names. Default: all discovered.")
    parser.add_argument("--log-failures", default=None)
    args = parser.parse_args()

    import polars as pl
    import torch
    from tqdm import tqdm

    cache_root = args.cache_root
    if cache_root is None:
        cache_root = (
            args.data_root if os.access(args.data_root, os.W_OK)
            else f"{args.data_root}-cache"
        )

    subsets = list_subsets(args.data_root)
    if args.subsets:
        wanted = set(args.subsets)
        subsets = [s for s in subsets if s.name in wanted]
        if not subsets:
            raise SystemExit(
                f"No subsets matched --subsets={args.subsets}. Available: "
                f"{[s.name for s in list_subsets(args.data_root)]}"
            )

    print(f"Subsets ({len(subsets)}):")
    for s in subsets:
        print(f"  {s.name}  hand_type={s.hand_type}  video={s.video_key}"
              + (f"  nested={s.nested_dir}" if s.nested_dir else ""))

    meta_df = _build_episode_meta(args.data_root, subsets)
    meta_path = os.path.join(cache_root, "meta", "episodes_meta.parquet")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    tmp_meta = f"{meta_path}.tmp.{os.getpid()}"
    meta_df.write_parquet(tmp_meta)
    os.replace(tmp_meta, meta_path)
    print(f"\nWrote episodes_meta.parquet ({meta_df.height} episodes) -> {meta_path}")

    encoder_path = os.path.join(args.sonic_model_dir, "model_encoder.onnx")
    out_root = os.path.join(cache_root, "sonic_tokens")
    # ``g1_proprio/`` references the G1 *robot's* joint state cache; the
    # directory name is preserved across the G1-expert removal.
    proprio_root = os.path.join(cache_root, "g1_proprio")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(proprio_root, exist_ok=True)

    # Build pending list. An episode is pending when either of {sonic, proprio}
    # is missing; ``need_sonic`` is False when only the lightweight
    # parquet-derived proprio dump is missing — the worker can skip the ONNX
    # session.
    pending: list[tuple] = []
    skipped = 0
    for row in meta_df.iter_rows(named=True):
        sub = row["subset"]
        ep_fn = f"episode_{row['episode_index']:06d}.npy"
        sonic_path = os.path.join(out_root, sub, ep_fn)
        proprio_path = os.path.join(proprio_root, sub, ep_fn)
        need_sonic = not os.path.exists(sonic_path)
        need_proprio = not os.path.exists(proprio_path)
        if not (need_sonic or need_proprio):
            skipped += 1
            continue
        pending.append((
            sub, int(row["episode_index"]),
            int(row["data_chunk"]), int(row["data_file"]),
            row["hand_type"], need_sonic,
        ))
    print(f"{meta_df.height} episodes total; {skipped} already done; {len(pending)} pending.")
    if not pending:
        print(f"Nothing to do. Output: {out_root}")
        return

    if args.gpus is None:
        n_gpus = torch.cuda.device_count()
    else:
        n_gpus = max(0, args.gpus)
    if n_gpus == 0:
        gpu_ids: list[int | None] = [None]
    else:
        gpu_ids = list(range(n_gpus))
    workers_per_gpu = max(1, args.workers_per_gpu)
    n_workers = len(gpu_ids) * workers_per_gpu

    print(f"Spawning {n_workers} workers ("
          f"{'CPU' if n_gpus == 0 else f'{n_gpus} GPU x {workers_per_gpu}'}).")

    ctx = mp.get_context("spawn")
    work_q: "mp.Queue" = ctx.Queue(maxsize=max(4 * n_workers, 32))
    result_q: "mp.Queue" = ctx.Queue()

    workers: list[mp.Process] = []
    for w in range(n_workers):
        gpu = gpu_ids[w % len(gpu_ids)]
        p = ctx.Process(
            target=worker_main,
            args=(
                gpu, work_q, result_q, encoder_path, out_root,
                proprio_root, args.data_root,
                args.source_fps, args.target_fps,
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
            print("\n[!] SIGINT received, draining queue and stopping workers...",
                  file=sys.stderr)

    signal.signal(signal.SIGINT, _handle_sigint)

    ok = 0
    errs: list[tuple[str, int, str]] = []
    fatal: list[tuple[int, str]] = []
    pbar = tqdm(total=len(pending), desc="Encoding episodes", unit="ep")
    done = 0
    while done < len(pending):
        if stop_flag["stop"]:
            break
        try:
            msg = result_q.get(timeout=1.0)
        except Exception:
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
            errs.append((msg[1], msg[2], msg[3]))
            done += 1
            pbar.update(1)
        elif kind == "fatal":
            fatal.append((msg[1], msg[2]))
    pbar.close()

    if stop_flag["stop"]:
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
        log_path = args.log_failures or os.path.join(out_root, "failed_episodes.log")
        with open(log_path, "w") as fh:
            for sub, ep, tb in errs:
                fh.write(f"--- subset={sub} episode={ep} ---\n{tb}\n")
            for pid, tb in fatal:
                fh.write(f"--- worker pid {pid} (fatal) ---\n{tb}\n")
        print(f"{len(errs)} failed episode(s), {len(fatal)} fatal worker(s). "
              f"See {log_path}", file=sys.stderr)

    print(f"\nDone. {ok} encoded, {skipped} skipped, {len(errs)} failed. "
          f"Output: {out_root}")


if __name__ == "__main__":
    main()

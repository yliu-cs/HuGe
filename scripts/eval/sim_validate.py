#!/usr/bin/env python3
"""Launch deploy-faithful Kappa simulation through the GR00T stack.

This entry point intentionally shares the same Kappa server, bridge, WBC
deploy binary, and GR00T MuJoCo renderer path as ``run_kappa_sim_test.sh``.
By default it runs an offline closed-loop simulation: Kappa receives
prerecorded dataset images plus cached robot proprio, while its actions are
executed by the real deployment bridge / WBC / MuJoCo sim loop.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "GR00T-WholeBodyControl") not in sys.path:
    sys.path.insert(0, str(ROOT / "GR00T-WholeBodyControl"))

from src.eval.prop_episode_source import PropEpisodeSource  # noqa: E402
from src.eval.episode_source import FrameStream  # noqa: E402
from gear_sonic.camera.sensor_server import ImageMessageSchema  # noqa: E402


SIM_LAUNCHER = ROOT / "scripts/eval/run_kappa_sim_test.sh"
DEFAULT_PROPRIETARY_ROOT = "resource/dataset/proprietary"
DEFAULT_PROPRIETARY_DUMP = "2026-05-16"


def _set_if_present(env: dict[str, str], key: str, value: object | None) -> None:
    if value is not None:
        env[key] = str(value)


def _resolve_proprietary_root(path: str | Path) -> Path:
    root = Path(path)
    if (root / "meta" / "episodes.jsonl").exists():
        return root
    dated = root / DEFAULT_PROPRIETARY_DUMP
    if (dated / "meta" / "episodes.jsonl").exists():
        return dated
    raise FileNotFoundError(
        f"Could not find proprietary episodes.jsonl under {root} or {dated}"
    )


def _publish_state(socket: zmq.Socket, proprio_64: np.ndarray, frame: int) -> None:
    p = np.asarray(proprio_64, dtype=np.float32)
    quat_xyzw = p[60:64]
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]].astype(np.float32)
    # Forward the cache's binary hand bits as a direct `hand_bits` field so
    # the bridge's _hand_bits_from_state sees the same training-side signal
    # it would derive from `*_q_measured` in a real-deploy g1_debug. Without
    # this the bridge falls back to zeros in offline-cache mode and the
    # policy's proprio[29:31] diverges from training distribution.
    msg = {
        "body_q": p[0:29].astype(np.float32),
        "base_quat": quat_wxyz,
        "hand_bits": p[29:31].astype(np.float32),
        "frame_index": int(frame),
        "timestamp": time.time(),
    }
    socket.send(b"g1_debug" + msgpack.packb(msg, use_bin_type=True), flags=zmq.NOBLOCK)


def _publish_camera(socket: zmq.Socket, image: Image.Image) -> None:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    now = time.time()
    payload = ImageMessageSchema(
        timestamps={"ego_view": now},
        images={"ego_view": arr},
    ).serialize()
    socket.send(msgpack.packb(payload, use_bin_type=True), flags=zmq.NOBLOCK)


def _wait_for_start_file(path: str | None, poll_s: float = 0.05) -> None:
    if not path:
        return
    start_path = Path(path)
    print(f"[offline-cache] waiting for start signal: {start_path}", flush=True)
    while not start_path.exists():
        time.sleep(poll_s)
    print(f"[offline-cache] start signal received: {start_path}", flush=True)


def run_offline_cache_publisher(args: argparse.Namespace) -> None:
    """Publish prerecorded camera/state cache in the same ZMQ formats the
    deployment bridge consumes. This is intentionally a small internal helper
    used by ``run_kappa_sim_test.sh`` so the bridge remains unchanged.
    """
    data_root = _resolve_proprietary_root(args.data_root or DEFAULT_PROPRIETARY_ROOT)
    source = PropEpisodeSource(
        data_root=data_root,
        proprio_cache_root=args.proprio_cache_root or f"{data_root}/g1_proprio",
        seed=args.seed,
    )
    episode = source.by_index(args.episode_index)
    stream = FrameStream(episode.video_path, fps=episode.video_fps)

    mnp.patch()
    ctx = zmq.Context()
    cam_socket = ctx.socket(zmq.PUB)
    cam_socket.setsockopt(zmq.SNDHWM, 5)
    cam_socket.setsockopt(zmq.LINGER, 0)
    cam_socket.bind(f"tcp://*:{args.camera_port}")
    state_socket = ctx.socket(zmq.PUB)
    state_socket.setsockopt(zmq.SNDHWM, 20)
    state_socket.setsockopt(zmq.LINGER, 0)
    state_socket.bind(f"tcp://*:{args.state_port}")
    time.sleep(0.2)

    print(
        f"[offline-cache] dataset={data_root} episode={episode.episode_index} "
        f"fps={args.ctrl_hz} camera=: {args.camera_port} state=: {args.state_port}"
    )
    _wait_for_start_file(args.start_file)
    period = 1.0 / float(args.ctrl_hz)
    start = time.perf_counter()
    frame = 0
    try:
        while True:
            t = frame * period
            if args.loop:
                t = t % max(episode.duration_s, period)
            elif t >= episode.duration_s:
                print("[offline-cache] episode finished", flush=True)
                break
            img = stream.frame_at(max(0.0, t))
            proprio = source.proprio_at(episode.episode_index, max(0.0, t))
            _publish_camera(cam_socket, img)
            _publish_state(state_socket, proprio, frame)

            if frame % int(max(args.ctrl_hz, 1)) == 0:
                print(f"[offline-cache] frame={frame} t={t:.2f}s")
            frame += 1
            target = start + frame * period
            sleep_for = target - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                start = time.perf_counter() - frame * period
    except KeyboardInterrupt:
        print("[offline-cache] interrupted")
    finally:
        stream.close()
        cam_socket.close()
        state_socket.close()
        ctx.term()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--offline-cache-publisher",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--dataset",
        choices=["proprietary"],
        default="proprietary",
        help="Offline input dataset. Currently only proprietary is supported.",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="Dataset root. Defaults to resource/dataset/proprietary/2026-05-16.",
    )
    ap.add_argument("--proprio-cache-root", default=None)
    ap.add_argument("--episode-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loop", action="store_true", default=False)
    ap.add_argument(
        "--start-file",
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--checkpoint",
        help="Fine-Tune safetensors checkpoint. Maps to KAPPA_CKPT.",
    )
    ap.add_argument("--backbone", help="Backbone path. Maps to KAPPA_BACKBONE.")
    ap.add_argument(
        "--proprio-stats",
        help="Optional proprio stats JSON override. Maps to KAPPA_PROPRIO_STATS. "
             "When unset, the inner server auto-discovers proprio_stats.json "
             "bundled next to the checkpoint (written by Trainer.save_checkpoint).",
    )
    ap.add_argument("--device", help="Inference device. Maps to KAPPA_DEVICE.")
    ap.add_argument("--ctrl-hz", type=float, help="Control rate. Maps to CTRL_HZ.")
    ap.add_argument("--camera-host", help="Bridge camera host. Maps to CAMERA_HOST.")
    ap.add_argument("--camera-port", type=int, help="Bridge camera port. Maps to CAMERA_PORT.")
    ap.add_argument(
        "--state-port",
        type=int,
        help="Bridge state input port. Maps to STATE_PORT in offline-cache mode.",
    )
    ap.add_argument(
        "--sim-camera-port",
        type=int,
        help="GR00T rendered camera publish port. Maps to SIM_CAMERA_PORT.",
    )
    ap.add_argument("--session", help="tmux session name. Maps to SESSION.")
    args = ap.parse_args()

    if args.offline_cache_publisher:
        run_offline_cache_publisher(args)
        return

    env = os.environ.copy()
    env["OFFLINE_CACHE_INPUT"] = "1"
    env["OFFLINE_DATASET"] = args.dataset
    env["OFFLINE_DATA_ROOT"] = args.data_root or DEFAULT_PROPRIETARY_ROOT
    env["OFFLINE_PROPRIO_CACHE_ROOT"] = args.proprio_cache_root or ""
    env["OFFLINE_EPISODE_INDEX"] = str(args.episode_index)
    env["OFFLINE_SEED"] = str(args.seed)
    if args.camera_port is not None:
        env["OFFLINE_CAMERA_PORT"] = str(args.camera_port)
    if args.state_port is not None:
        env["OFFLINE_STATE_PORT"] = str(args.state_port)
    _set_if_present(env, "KAPPA_CKPT", args.checkpoint)
    _set_if_present(env, "KAPPA_BACKBONE", args.backbone)
    _set_if_present(env, "KAPPA_PROPRIO_STATS", args.proprio_stats)
    _set_if_present(env, "KAPPA_DEVICE", args.device)
    _set_if_present(env, "CTRL_HZ", args.ctrl_hz)
    _set_if_present(env, "CAMERA_HOST", args.camera_host)
    _set_if_present(env, "CAMERA_PORT", args.camera_port)
    _set_if_present(env, "STATE_PORT", args.state_port)
    _set_if_present(env, "SIM_CAMERA_PORT", args.sim_camera_port)
    _set_if_present(env, "SESSION", args.session)

    os.execve(str(SIM_LAUNCHER), [str(SIM_LAUNCHER)], env)


if __name__ == "__main__":
    main()

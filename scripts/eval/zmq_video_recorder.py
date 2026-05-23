"""Record a MuJoCo sim camera ZMQ stream to mp4.

The publisher in gear_sonic.utils.mujoco_sim.image_publish_utils sends a
msgpack payload of the form

    {
        "timestamps": {cam_name: float, ...},
        "images":     {cam_name: <base64 JPEG str>, ...},
        cam_name: <base64 JPEG str>,   # also pushed flat at top-level
        ...
    }

over a zmq.PUB socket bound to ``tcp://*:<port>``. We subscribe with no
topic filter, decode the chosen camera (default ``ego_view``, matching the
bridge in scripts/deploy/run_kappa_inference.py), convert BGR->RGB, and
write frames to an mp4 via imageio-ffmpeg.

The encoder MUST be closed cleanly on SIGTERM/SIGINT so the moov atom is
written; otherwise the file is unplayable. We install signal handlers
that flip a stop flag, let the main loop drain, and close the writer in
a ``finally`` block.
"""

from __future__ import annotations

import argparse
import base64
import signal
import sys
import time
from typing import Optional

import cv2
import imageio.v2 as imageio
import msgpack
import numpy as np
import zmq


_STOP = False


def _request_stop(signum, _frame):  # noqa: ANN001 - signal handler
    global _STOP
    _STOP = True
    print(f"[recorder] received signal {signum}; finalizing mp4", flush=True)


def _decode_jpeg_b64(payload) -> Optional[np.ndarray]:
    """Decode the publisher's per-camera payload to a BGR uint8 frame."""
    if payload is None:
        return None
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    elif isinstance(payload, str):
        raw = base64.b64decode(payload)
    elif isinstance(payload, np.ndarray):
        return payload  # already decoded upstream
    else:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _pick_camera(images: dict, preferred: str) -> tuple[Optional[str], object]:
    if preferred in images:
        return preferred, images[preferred]
    if images:
        name = next(iter(images))
        return name, images[name]
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--output", required=True, help="Output mp4 path")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--camera-key",
        default="ego_view",
        help="Camera name in the publisher's images dict (fallback: first key)",
    )
    ap.add_argument(
        "--idle-timeout-s",
        type=float,
        default=30.0,
        help="Exit if no frame arrives within this many seconds",
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=8,
        help="imageio quality 0-10 (higher=better, larger files)",
    )
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 20)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(f"tcp://{args.host}:{args.port}")

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    # macro_block_size=1 keeps non-16-multiple frame dims (e.g. 480x640) intact.
    writer = imageio.get_writer(
        args.output,
        fps=args.fps,
        codec="libx264",
        quality=args.quality,
        macro_block_size=1,
    )
    print(
        f"[recorder] subscribed tcp://{args.host}:{args.port} -> {args.output} "
        f"@ {args.fps} fps (camera={args.camera_key})",
        flush=True,
    )

    frames = 0
    cam_used: Optional[str] = None
    last_frame_t = time.monotonic()
    last_log_t = last_frame_t
    try:
        while not _STOP:
            socks = dict(poller.poll(timeout=200))
            if sub in socks:
                try:
                    raw = sub.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                try:
                    msg = msgpack.unpackb(raw, raw=False)
                except Exception as e:  # noqa: BLE001
                    print(f"[recorder] msgpack decode failed: {e}", flush=True)
                    continue

                images = msg.get("images") if isinstance(msg, dict) else None
                if not images:
                    # Fall back to flat top-level cam_name keys.
                    images = {
                        k: v
                        for k, v in (msg or {}).items()
                        if k not in ("images", "timestamps")
                    }
                name, payload = _pick_camera(images or {}, args.camera_key)
                frame = _decode_jpeg_b64(payload)
                if frame is None:
                    continue
                if cam_used is None:
                    cam_used = name
                    print(f"[recorder] first frame from camera '{name}' shape={frame.shape}", flush=True)
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames += 1
                last_frame_t = time.monotonic()

                now = last_frame_t
                if now - last_log_t >= 5.0:
                    print(f"[recorder] frames={frames} (cam={cam_used})", flush=True)
                    last_log_t = now
            else:
                idle = time.monotonic() - last_frame_t
                if idle > args.idle_timeout_s:
                    print(
                        f"[recorder] no frames for {idle:.1f}s (> {args.idle_timeout_s}s); exiting",
                        flush=True,
                    )
                    break
    finally:
        try:
            writer.close()
        except Exception as e:  # noqa: BLE001
            print(f"[recorder] writer.close() failed: {e}", flush=True)
        try:
            sub.close(linger=0)
            ctx.term()
        except Exception:
            pass
        print(f"[recorder] done. frames={frames} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Subscribe to a headless sim's ZMQ image-publish stream (see
launch_headless_sim.py --enable-image-publish) and dump frames to an mp4.

Wire format matches GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/sensor_server.py's
SensorServer/ImageMessageSchema (PUB, msgpack, base64-JPEG per camera key) --
reimplemented here (not imported) to keep this Kappa-side script off the
GR00T-WholeBodyControl Python path, matching align_sim_replay.py's precedent.
"""

from __future__ import annotations

import argparse
import base64
import time

import cv2
import msgpack
import numpy as np
import zmq


def decode_image(b64: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--camera", default="ego_view")
    parser.add_argument("--out", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{args.host}:{args.port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.setsockopt(zmq.CONFLATE, True)
    sock.setsockopt(zmq.RCVTIMEO, 3000)

    writer = None
    n = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.duration:
        try:
            packed = sock.recv()
        except zmq.error.Again:
            continue
        data = msgpack.unpackb(packed, raw=False)
        img_b64 = data.get("images", {}).get(args.camera)
        if img_b64 is None:
            continue
        img = decode_image(img_b64)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(img)
        n += 1

    if writer is not None:
        writer.release()
    sock.close()
    ctx.term()
    print(f"wrote {n} frames -> {args.out}")


if __name__ == "__main__":
    main()

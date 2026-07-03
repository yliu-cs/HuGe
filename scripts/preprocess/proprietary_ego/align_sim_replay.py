"""Anchor + positional alignment between captured g1_debug ZMQ frames and the
original latent-token .npy that drove a sim replay. See
resource/info/ego_cola_full_stack_verification_and_headless_render.md and
ego_cola_sim_replay_pseudo_proprio_pipeline.md for the full pipeline this feeds.

token_state jitters at per-frame granularity (a race between the ZMQ-receive
thread and the debug-publish thread, not a real skipped control tick -- verified
by checking body_q, a continuous physical quantity, across the same transitions).
So content-matching is used only once, to find the anchor frame where real
token-driven control begins (after the TRT-loading / safety-reset startup
period); every frame after the anchor is mapped to a token index by pure
sequential position.

`capture` needs pyzmq+msgpack against a live sim session (either venv works);
`align` is offline, numpy-only, and can run standalone.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5557
DEFAULT_TOPIC = "g1_debug"
ANCHOR_TOL = 1e-5
WARN_TAIL_THRESHOLD = 1  # a ~1-frame tail shortfall is expected (deploy exits before its last flush)
ANCHOR_OFFSET_WINDOW = 5  # ZMQ slow-joiner can drop the first few published tokens; search a small window
ALIGN_FIELDS = ("token_state", "body_q", "left_hand_q", "right_hand_q", "base_quat")


def capture_g1_debug_frames(
    duration: float,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    topic: str = DEFAULT_TOPIC,
    recv_timeout_ms: int = 3000,
) -> list[dict[str, Any]]:
    """SUB to a live g1_debug ZMQ stream and collect every frame for `duration` seconds."""
    import msgpack
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)

    frames: list[dict[str, Any]] = []
    t0 = time.time()
    while time.time() - t0 < duration:
        try:
            msg = sock.recv()
        except zmq.error.Again:
            continue
        payload = msg[len(topic):]
        data = msgpack.unpackb(payload, raw=False)
        data["_recv_time"] = time.time() - t0
        frames.append(data)
    sock.close()
    ctx.term()
    return frames


def find_anchor(frames: list[dict[str, Any]], token_row: np.ndarray, tol: float = ANCHOR_TOL) -> int:
    """Index of the first frame whose token_state matches the given original npy row
    (max-abs-diff < tol). This is the only content-matching step in the whole
    alignment -- everything after the anchor is positional. Raises if nothing matches."""
    for i, frame in enumerate(frames):
        state = frame.get("token_state")
        if state is None:
            continue
        if np.max(np.abs(np.asarray(state, dtype=np.float64) - token_row)) < tol:
            return i
    raise ValueError(f"no frame matched token row within tol={tol}")


def find_anchor_with_offset(
    frames: list[dict[str, Any]],
    tokens: np.ndarray,
    tol: float = ANCHOR_TOL,
    max_offset: int = ANCHOR_OFFSET_WINDOW,
) -> tuple[int, int]:
    """Same as `find_anchor`, but tries token rows 0, 1, 2, ... in turn and returns
    the first one that matches, as `(anchor_idx, token_offset)`.

    Needed because the deploy binary's ZMQ SUB socket can miss the first one or
    two published tokens (slow-joiner) -- when that happens, no captured frame
    ever matches token row 0 even though the replay otherwise ran fine. Episodes
    where token row 0 happens to duplicate row 1 (common in this dataset) hide
    this; episodes where it doesn't raise a spurious "no frame matched" error
    unless we search forward a few rows."""
    for offset in range(min(max_offset, tokens.shape[0])):
        try:
            idx = find_anchor(frames, tokens[offset].astype(np.float64), tol=tol)
            return idx, offset
        except ValueError:
            continue
    raise ValueError(f"no frame matched token rows 0..{max_offset - 1} within tol={tol}")


def align_frames_to_tokens(
    frames: list[dict[str, Any]],
    num_tokens: int,
    tokens: np.ndarray | None = None,
    anchor_idx: int | None = None,
    token_offset: int = 0,
    tol: float = ANCHOR_TOL,
    max_offset: int = ANCHOR_OFFSET_WINDOW,
) -> dict[str, Any]:
    """Anchor + positional alignment: frame `anchor_idx + k` is token index
    `token_offset + k`. `token_offset` is normally 0, but see
    `find_anchor_with_offset` for when the leading token(s) never got driven."""
    if anchor_idx is None:
        if tokens is None:
            raise ValueError("must supply tokens or anchor_idx")
        anchor_idx, token_offset = find_anchor_with_offset(frames, tokens, tol=tol, max_offset=max_offset)

    matched = min(len(frames) - anchor_idx, num_tokens - token_offset)
    token_index_by_frame = [-1] * len(frames)
    for k in range(matched):
        token_index_by_frame[anchor_idx + k] = token_offset + k

    return {
        "anchor_idx": anchor_idx,
        "token_offset": token_offset,
        "num_tokens": num_tokens,
        "num_frames": len(frames),
        "matched": matched,
        "missing_head": token_offset,
        "missing_tail": num_tokens - token_offset - matched,
        "token_index_by_frame": token_index_by_frame,
    }


def build_aligned_arrays(
    frames: list[dict[str, Any]],
    alignment: dict[str, Any],
    fields: tuple[str, ...] = ALIGN_FIELDS,
) -> dict[str, np.ndarray]:
    """Reindex `fields` into arrays of shape [num_tokens, ...]. Token indices with
    no captured frame (expected only at the tail) are filled with NaN."""
    num_tokens = alignment["num_tokens"]
    by_token: dict[int, dict[str, Any]] = {}
    for frame_idx, token_idx in enumerate(alignment["token_index_by_frame"]):
        if token_idx >= 0:
            by_token[token_idx] = frames[frame_idx]

    out: dict[str, np.ndarray] = {}
    for field in fields:
        sample = next((f[field] for f in by_token.values() if field in f), None)
        if sample is None:
            continue
        shape = np.asarray(sample, dtype=np.float64).shape
        arr = np.full((num_tokens, *shape), np.nan, dtype=np.float64)
        for token_idx, frame in by_token.items():
            if field in frame:
                arr[token_idx] = np.asarray(frame[field], dtype=np.float64)
        out[field] = arr
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="SUB to a live g1_debug stream and dump raw frames to a pickle")
    cap.add_argument("--duration", type=float, required=True)
    cap.add_argument("--host", default=DEFAULT_HOST)
    cap.add_argument("--port", type=int, default=DEFAULT_PORT)
    cap.add_argument("--topic", default=DEFAULT_TOPIC)
    cap.add_argument("--out", required=True, help="output pickle path for raw frames")

    aln = sub.add_parser("align", help="anchor + positional alignment against a token .npy")
    aln.add_argument("--frames", required=True, help="pickle produced by `capture`")
    aln.add_argument("--tokens", required=True, help="original *_latent_tokens_64d.npy")
    aln.add_argument("--tol", type=float, default=ANCHOR_TOL)
    aln.add_argument("--out", required=True, help="output .npz path for aligned arrays")
    aln.add_argument("--report", default=None, help="output JSON path (default: <out> with .report.json suffix)")

    args = parser.parse_args()

    if args.cmd == "capture":
        frames = capture_g1_debug_frames(args.duration, host=args.host, port=args.port, topic=args.topic)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump(frames, f)
        os.replace(tmp, out)
        print(f"captured {len(frames)} frames -> {out}")
        return

    with open(args.frames, "rb") as f:
        frames = pickle.load(f)
    tokens = np.load(args.tokens)
    alignment = align_frames_to_tokens(frames, num_tokens=tokens.shape[0], tokens=tokens, tol=args.tol)
    arrays = build_aligned_arrays(frames, alignment)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, out)

    report = {k: v for k, v in alignment.items() if k != "token_index_by_frame"}
    report_path = Path(args.report) if args.report else out.with_suffix(".report.json")
    tmp_report = report_path.with_suffix(report_path.suffix + f".tmp.{os.getpid()}")
    tmp_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp_report, report_path)

    print(f"aligned {alignment['matched']}/{alignment['num_tokens']} tokens -> {out}")
    if alignment["token_offset"] > 0:
        print(f"NOTE: token_offset={alignment['token_offset']} -- leading token row(s) were dropped by ZMQ slow-joiner")
    if alignment["missing_tail"] > WARN_TAIL_THRESHOLD:
        print(
            f"WARNING: missing_tail={alignment['missing_tail']} exceeds the expected "
            f"~{WARN_TAIL_THRESHOLD}-frame shortfall -- investigate before trusting this episode"
        )


if __name__ == "__main__":
    main()

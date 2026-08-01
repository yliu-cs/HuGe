"""Capture G1 debug frames and align them positionally to SONIC motion tokens.

Content matching is used only to find the first replayed token. Frames after
that anchor are assigned to token rows by position because the debug publisher
can expose per-frame token jitter even when the physical state is continuous.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5557
DEFAULT_TOPIC = "g1_debug"
ANCHOR_TOL = 1e-5
ANCHOR_OFFSET_WINDOW = 5
WARN_TAIL_THRESHOLD = 1
ALIGN_FIELDS = ("token_state", "body_q", "left_hand_q", "right_hand_q", "base_quat")


def capture_g1_debug_frames(
    duration: float,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    topic: str = DEFAULT_TOPIC,
    recv_timeout_ms: int = 3000,
    stop_event: threading.Event | None = None,
) -> list[dict[str, Any]]:
    """Subscribe to g1_debug and collect frames for ``duration`` seconds."""
    import msgpack
    import zmq

    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{host}:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    socket.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
    frames: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration and not (stop_event and stop_event.is_set()):
            try:
                message = socket.recv()
            except zmq.error.Again:
                continue
            payload = message[len(topic) :]
            frame = msgpack.unpackb(payload, raw=False)
            if not isinstance(frame, dict):
                raise ValueError(f"expected a mapping in g1_debug payload, got {type(frame).__name__}")
            received = time.monotonic()
            frame["_recv_time"] = received - started
            frame["_recv_monotonic"] = received
            frames.append(frame)
    finally:
        socket.close()
        context.term()
    return frames


def find_anchor(frames: list[dict[str, Any]], token_row: np.ndarray, tol: float = ANCHOR_TOL) -> int:
    """Return the first frame whose token_state matches ``token_row``."""
    expected = np.asarray(token_row, dtype=np.float64)
    for index, frame in enumerate(frames):
        state = frame.get("token_state")
        if state is None:
            continue
        actual = np.asarray(state, dtype=np.float64)
        if actual.shape == expected.shape and np.max(np.abs(actual - expected)) < tol:
            return index
    raise ValueError(f"no frame matched a token row within tol={tol}")


def find_anchor_with_offset(
    frames: list[dict[str, Any]],
    tokens: np.ndarray,
    tol: float = ANCHOR_TOL,
    max_offset: int = ANCHOR_OFFSET_WINDOW,
) -> tuple[int, int]:
    """Return ``(frame_index, token_offset)`` for the first matching leading row."""
    if max_offset <= 0:
        raise ValueError(f"max_offset must be positive, got {max_offset}")
    limit = min(max_offset, tokens.shape[0])
    for offset in range(limit):
        try:
            return find_anchor(frames, tokens[offset], tol=tol), offset
        except ValueError:
            pass
    raise ValueError(f"no frame matched token rows 0..{limit - 1} within tol={tol}")


def align_frames_to_tokens(
    frames: list[dict[str, Any]],
    num_tokens: int,
    tokens: np.ndarray | None = None,
    anchor_idx: int | None = None,
    token_offset: int = 0,
    tol: float = ANCHOR_TOL,
    max_offset: int = ANCHOR_OFFSET_WINDOW,
) -> dict[str, Any]:
    """Map captured frames to token rows after a single content anchor."""
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if not frames:
        raise ValueError("capture contains no g1_debug frames")
    if anchor_idx is None:
        if tokens is None:
            raise ValueError("must supply tokens or anchor_idx")
        anchor_idx, token_offset = find_anchor_with_offset(frames, tokens, tol=tol, max_offset=max_offset)
    if not 0 <= anchor_idx < len(frames):
        raise ValueError(f"anchor_idx {anchor_idx} is outside {len(frames)} frames")

    matched = min(len(frames) - anchor_idx, num_tokens - token_offset)
    token_index_by_frame = [-1] * len(frames)
    for index in range(matched):
        token_index_by_frame[anchor_idx + index] = token_offset + index

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
    """Build one NaN-padded array per required debug field."""
    matched_frames = [
        (token_index, frames[frame_index])
        for frame_index, token_index in enumerate(alignment["token_index_by_frame"])
        if token_index >= 0
    ]
    if not matched_frames:
        raise ValueError("alignment produced no matched frames")

    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        missing = [index for index, frame in matched_frames if field not in frame]
        if missing:
            raise ValueError(f"required g1_debug field {field!r} is missing at token rows {missing[:5]}")
        shape = np.asarray(matched_frames[0][1][field]).shape
        array = np.full((alignment["num_tokens"], *shape), np.nan, dtype=np.float64)
        for token_index, frame in matched_frames:
            value = np.asarray(frame[field], dtype=np.float64)
            if value.shape != shape:
                raise ValueError(
                    f"g1_debug field {field!r} changed shape at token {token_index}: {value.shape} != {shape}"
                )
            array[token_index] = value
        arrays[field] = array
    return arrays


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="capture raw g1_debug frames")
    capture.add_argument("--duration", type=float, required=True)
    capture.add_argument("--host", default=DEFAULT_HOST)
    capture.add_argument("--port", type=int, default=DEFAULT_PORT)
    capture.add_argument("--topic", default=DEFAULT_TOPIC)
    capture.add_argument("--recv-timeout-ms", type=int, default=3000)
    capture.add_argument("--out", required=True)

    align = commands.add_parser("align", help="align a capture to a [T,64] token array")
    align.add_argument("--frames", required=True)
    align.add_argument("--tokens", required=True)
    align.add_argument("--tol", type=float, default=ANCHOR_TOL)
    align.add_argument("--max-offset", type=int, default=ANCHOR_OFFSET_WINDOW)
    align.add_argument("--out", required=True)
    align.add_argument("--report", default=None)
    args = parser.parse_args()

    if args.command == "capture":
        frames = capture_g1_debug_frames(
            args.duration,
            host=args.host,
            port=args.port,
            topic=args.topic,
            recv_timeout_ms=args.recv_timeout_ms,
        )
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
        with temporary.open("wb") as stream:
            pickle.dump(frames, stream)
        os.replace(temporary, output)
        print(f"captured {len(frames)} frames -> {output}")
        return

    with Path(args.frames).open("rb") as stream:
        frames = pickle.load(stream)
    if not isinstance(frames, list):
        raise ValueError(f"capture must contain a frame list, got {type(frames).__name__}")
    tokens = np.load(args.tokens)
    if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] != 64:
        raise ValueError(f"expected non-empty [T,64] motion tokens, got {tokens.shape}")
    if not np.isfinite(tokens).all():
        raise ValueError("motion tokens contain NaN or Inf")

    alignment = align_frames_to_tokens(
        frames,
        num_tokens=tokens.shape[0],
        tokens=tokens,
        tol=args.tol,
        max_offset=args.max_offset,
    )
    arrays = build_aligned_arrays(frames, alignment)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, output)

    report = {key: value for key, value in alignment.items() if key != "token_index_by_frame"}
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")
    _atomic_json(report_path, report)
    print(f"aligned {alignment['matched']}/{alignment['num_tokens']} tokens -> {output}")
    if alignment["token_offset"]:
        print(f"NOTE: token_offset={alignment['token_offset']} due to leading ZMQ frame loss")
    if alignment["missing_tail"] > WARN_TAIL_THRESHOLD:
        print(f"WARNING: missing_tail={alignment['missing_tail']} exceeds {WARN_TAIL_THRESHOLD}")


if __name__ == "__main__":
    main()

"""Record an X11 window (e.g. the MuJoCo on-screen viewer) to mp4 via ffmpeg.

Why not the ZMQ recorder (scripts/eval/zmq_video_recorder.py)?
    The ZMQ recorder consumes frames published by the MuJoCo offscreen EGL
    renderer. On NVIDIA hosts, EGL and GLX cannot coexist in the same process
    (eglMakeCurrent fails when a GLFW/GLX window is already open), so the
    offscreen renderer cannot run alongside the on-screen viewer. This
    recorder sidesteps the issue entirely by grabbing the on-screen viewer
    window through X11.

What it does
    1. Polls ``xwininfo -root -tree`` until a window matching ``--window-name``
       (substring match against the title) and class ``--window-class``
       appears. We prefer the inner application window (class ``MuJoCo``)
       over the WM frame (class ``mutter-x11-frames``) so the grab excludes
       the decoration bar.
    2. Reads the matched window's geometry and X resource id.
    3. Launches ffmpeg with ``-f x11grab -window_id <id>``. ffmpeg's x11grab
       follows the window if it moves; closing the window or sending SIGINT
       triggers a clean shutdown that writes the mp4 moov atom.

Signal handling
    SIGTERM / SIGINT to this script -> SIGINT to ffmpeg. ffmpeg returns 0 (or
    255 on older builds) on SIGINT and finalizes the mp4 trailer in the
    process. We wait for it to exit before returning so the orchestrator can
    observe a complete file.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


# Sample tree line (note tab + 2-space indent):
#   0x520000b "MuJoCo : MuJoCo Model": ("MuJoCo" "MuJoCo")  2560x1440+28+98  +182+188


def _xwininfo_tree(display: str) -> str:
    env = os.environ.copy()
    env["DISPLAY"] = display
    return subprocess.check_output(["xwininfo", "-root", "-tree"], env=env).decode(
        "utf-8", errors="replace"
    )


def _find_window(tree: str, name_pat: str, klass: str) -> Optional[dict]:
    name_lc = name_pat.lower()
    for line in tree.splitlines():
        # The class section looks like: ("MuJoCo" "MuJoCo")
        if klass not in line:
            continue
        if name_lc not in line.lower():
            continue
        m = re.search(r"(0x[0-9a-fA-F]+)\s+\"([^\"]*)\"", line)
        if not m:
            continue
        wid_hex = m.group(1)
        # xwininfo emits one "WxH+X+Y" (size + parent-relative offset) and
        # one trailing "+X+Y" (root-relative offset) per child line, e.g.:
        #   0x520000b "MuJoCo ...": ("MuJoCo" "MuJoCo")  2560x1440+28+98  +182+188
        # ffmpeg with -window_id follows the window regardless of position,
        # so we only need the size. Take the first (and only) WxH+X+Y match.
        gm = re.search(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", line)
        if not gm:
            continue
        w, h = int(gm.group(1)), int(gm.group(2))
        x, y = int(gm.group(3)), int(gm.group(4))
        if w <= 0 or h <= 0:
            continue
        return {"id_hex": wid_hex, "id_int": int(wid_hex, 16), "w": w, "h": h, "x": x, "y": y}
    return None


def _wait_for_window(
    display: str,
    name_pat: str,
    klass: str,
    timeout_s: float,
    poll_s: float = 0.5,
    should_stop=None,
) -> dict:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        if should_stop is not None and should_stop():
            # Caller signalled cancellation; return a sentinel that the caller
            # treats as "exit cleanly without recording."
            return {"id_hex": "", "id_int": 0, "w": 0, "h": 0, "x": 0, "y": 0}
        try:
            tree = _xwininfo_tree(display)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(poll_s)
            continue
        win = _find_window(tree, name_pat, klass)
        if win is not None:
            return win
        time.sleep(poll_s)
    raise TimeoutError(
        f"window matching name~='{name_pat}' class='{klass}' did not appear within "
        f"{timeout_s:.1f}s on {display}"
        + (f" (last xwininfo error: {last_err})" if last_err else "")
    )


def _even(n: int) -> int:
    """libx264 requires even dimensions; round DOWN so we never exceed the
    real window size (otherwise ffmpeg reads garbage off the edge)."""
    return n - (n & 1)


def _wait_for_start_file(
    path: str,
    poll_s: float = 0.05,
    should_stop=None,
) -> bool:
    """Block until ``path`` exists. Returns True if start was observed, False
    if should_stop() fired first. The orchestrator touches this file at the
    same moment the offline-cache publisher is released, so gating ffmpeg
    here lines the mp4 up with the actual policy/sim activity instead of
    capturing the WBC Init Done warmup period."""
    start_path = Path(path)
    print(f"[x11rec] waiting for start signal: {start_path}", flush=True)
    while not start_path.exists():
        if should_stop is not None and should_stop():
            return False
        time.sleep(poll_s)
    print(f"[x11rec] start signal received: {start_path}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="Output mp4 path")
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":0"))
    ap.add_argument(
        "--window-name",
        default="MuJoCo",
        help="Case-insensitive substring of the window title to match",
    )
    ap.add_argument(
        "--window-class",
        default="MuJoCo",
        help="Window class to match (use to exclude WM frame windows)",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--window-timeout-s",
        type=float,
        default=120.0,
        help="How long to wait for the target window to appear",
    )
    ap.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg binary (path or in $PATH)",
    )
    ap.add_argument(
        "--preset",
        default="veryfast",
        help="libx264 preset (default veryfast for low CPU overhead)",
    )
    ap.add_argument(
        "--crf",
        type=int,
        default=23,
        help="libx264 CRF quality (lower = better, default 23)",
    )
    ap.add_argument(
        "--start-file",
        default=None,
        help="Optional sentinel path. If set, the window is located first but "
             "ffmpeg only launches once this file appears (kept in sync with "
             "the offline-cache publisher so the mp4 starts when frames do).",
    )
    args = ap.parse_args()

    # Install signal handlers BEFORE the window-wait so an early SIGTERM
    # (e.g. orchestrator teardown while the viewer is still starting up)
    # exits cleanly instead of inheriting the default fatal handler.
    early_stop = False

    def _early_stop_handler(signum, _frame):  # noqa: ANN001
        nonlocal early_stop
        early_stop = True
        print(f"[x11rec] received signal {signum} before ffmpeg launched; exiting", flush=True)

    signal.signal(signal.SIGTERM, _early_stop_handler)
    signal.signal(signal.SIGINT, _early_stop_handler)

    print(
        f"[x11rec] waiting up to {args.window_timeout_s:.0f}s for window "
        f"name~='{args.window_name}' class='{args.window_class}' on {args.display}",
        flush=True,
    )
    try:
        win = _wait_for_window(
            args.display,
            args.window_name,
            args.window_class,
            args.window_timeout_s,
            should_stop=lambda: early_stop,
        )
    except TimeoutError as e:
        print(f"[x11rec] ERROR: {e}", file=sys.stderr, flush=True)
        return 2
    if early_stop:
        return 0

    w_even = _even(win["w"])
    h_even = _even(win["h"])
    print(
        f"[x11rec] found window {win['id_hex']} at +{win['x']},{win['y']} "
        f"{win['w']}x{win['h']} (encoding {w_even}x{h_even})",
        flush=True,
    )

    # Gate ffmpeg launch on the same sentinel the offline-cache publisher
    # waits on, so the mp4 starts the instant data publishing starts.
    if args.start_file:
        started = _wait_for_start_file(
            args.start_file, should_stop=lambda: early_stop
        )
        if not started or early_stop:
            print("[x11rec] cancelled before start signal arrived", flush=True)
            return 0

    cmd = [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel", "warning",
        "-f", "x11grab",
        "-framerate", str(args.fps),
        "-video_size", f"{w_even}x{h_even}",
        "-window_id", str(win["id_int"]),
        "-i", args.display,
        "-c:v", "libx264",
        "-preset", args.preset,
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        args.output,
    ]
    print(f"[x11rec] launching: {shlex.join(cmd)}", flush=True)

    # Inherit the caller's DISPLAY but force the chosen one (in case env
    # differs). stdin must remain a pipe so we can send 'q' for clean exit;
    # ffmpeg ALSO finalizes on SIGINT, so 'q' is a redundancy.
    env = os.environ.copy()
    env["DISPLAY"] = args.display
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        env=env,
    )

    stopping = False

    def _request_stop(signum, _frame):  # noqa: ANN001 - signal handler
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print(f"[x11rec] received signal {signum}; finalizing mp4", flush=True)
        # 'q' is ffmpeg's documented graceful-stop key. We send it AND SIGINT
        # so whichever path the build supports kicks in.
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.write(b"q")
                proc.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:  # noqa: BLE001
            pass

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    try:
        rc = proc.wait()
    finally:
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass

    # ffmpeg returns 255 on SIGINT in some builds; treat that as success.
    if rc in (0, 255):
        if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            print(
                f"[x11rec] done. output={args.output} ({os.path.getsize(args.output)} bytes)",
                flush=True,
            )
            return 0
        print(f"[x11rec] WARNING: ffmpeg exit {rc} but {args.output} is missing/empty", flush=True)
        return 3
    print(f"[x11rec] ffmpeg exited with {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

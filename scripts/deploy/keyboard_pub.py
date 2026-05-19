"""Standalone keyboard publisher for the Kappa deploy stack.

Binds a ZMQ PUB on ``tcp://0.0.0.0:5580`` (matches
``gear_sonic.utils.data_collection.keyboard_subscriber.DEFAULT_ZMQ_KEYBOARD_PORT``)
and forwards single keystrokes to the bridge. Press ``t`` to enter a new
language prompt; the script briefly drops out of cbreak mode for line
editing and publishes ``prompt:<text>``.

Keys:
    p   pause / resume bridge inference loop
    k   start / stop C++ control loop
    i   send initial pose + switch to POSE mode
    [   toggle initial-pose left hand closed
    ]   toggle initial-pose right hand closed
    c   start recording (handled by data exporter, if running)
    s   stop recording success
    f   stop recording failure
    t   enter prompt edit (line entry; publishes "prompt:<text>")
    q   quit this publisher (other processes keep running)
"""
from __future__ import annotations

import argparse
import os
import sys
import termios
import time
import tty

import zmq

SINGLE_KEYS = set("pki[]csf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5580)
    args = parser.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.host}:{args.port}")
    # Brief sleep so SUB sockets connected after us can subscribe before
    # the first message lands (avoids the ZMQ slow-joiner drop).
    time.sleep(0.2)
    print(
        f"[kbd] PUB bound on tcp://{args.host}:{args.port}\n"
        "      keys: p k i [ ] c s f t q  ('t' for prompt entry)"
    )

    if not sys.stdin.isatty():
        print("[kbd] stdin is not a TTY; aborting", file=sys.stderr)
        sys.exit(2)

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = os.read(fd, 1).decode("utf-8", errors="ignore")
            if not ch:
                continue
            if ch == "q":
                print("[kbd] quit")
                break
            if ch == "t":
                # Restore cooked mode for line entry.
                termios.tcsetattr(fd, termios.TCSADRAIN, saved)
                try:
                    sys.stdout.write("new prompt> ")
                    sys.stdout.flush()
                    text = sys.stdin.readline().strip()
                finally:
                    tty.setcbreak(fd)
                if text:
                    sock.send_string(f"prompt:{text}")
                    print(f"[kbd] sent prompt:{text}")
                else:
                    print("[kbd] empty prompt -- ignored")
                continue
            if ch in SINGLE_KEYS:
                sock.send_string(ch)
                print(f"[kbd] sent {ch!r}")
            elif ch in ("\n", "\r"):
                # Ignore stray newlines (common after pasting).
                continue
            else:
                # Echo unknown keys silently — avoids spamming on arrow-key
                # escape sequences.
                pass
    except KeyboardInterrupt:
        print("[kbd] interrupted")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()

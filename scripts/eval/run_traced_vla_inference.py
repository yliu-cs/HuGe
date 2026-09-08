"""Optional timing entry point for the existing VLA client (Python 3.10).

--trace-dir, --validate-only and --hand-state-compat belong to this wrapper; remaining arguments
are forwarded to the original InferenceConfig. This entry starts paused like the
original client. --validate-only inspects imports/interfaces without connecting.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from vla_timing import TraceRecorder, timing_hooks, validate_api
from hand_state_compat import COMPAT_NOTE, FORMULA_VERSION, LEGACY_MODE, hand_state_compat_hooks, validate_compat_api


ROOT = Path(__file__).resolve().parents[2]
GROOT = ROOT / "GR00T-WholeBodyControl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--hand-state-compat", choices=("off", LEGACY_MODE), default="off",
                        help="Opt in to the current cola model's training hand labels; requires --hand-type inspire")
    args, remaining = parser.parse_known_args()
    from gear_sonic.scripts import run_vla_inference as api
    import tyro

    config = tyro.cli(api.InferenceConfig, args=remaining)
    if args.hand_state_compat != "off":
        if config.hand_type != "inspire":
            parser.error("--hand-state-compat requires explicit --hand-type inspire")
        validate_compat_api(api)
    policy = validate_api(api)
    directory = (args.trace_dir or ROOT / ".runs/deploy_timing" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if directory == GROOT or GROOT in directory.parents:
        parser.error("--trace-dir must be outside GR00T-WholeBodyControl")
    if args.validate_only:
        print(f"Policy client: {policy.__module__}.{policy.__name__}")
        print(f"Hand-state compatibility: {args.hand_state_compat}")
        print("Preflight passed: imports and timing interfaces; no sockets, camera, model or controller started.")
        return
    recorder = TraceRecorder(directory, {"client_class": f"{policy.__module__}.{policy.__name__}",
                                         "config": asdict(config),
                                         "hand_state_compat": {"mode": args.hand_state_compat,
                                                               "formula_version": FORMULA_VERSION if args.hand_state_compat != "off" else None,
                                                               "note": COMPAT_NOTE}})
    print(f"VLA timing directory: {directory}", flush=True)
    try:
        with ExitStack() as stack:
            if args.hand_state_compat != "off":
                stack.enter_context(hand_state_compat_hooks(api))
            stack.enter_context(timing_hooks(api, recorder, policy))
            api.main(config)
    except BaseException as exc:
        recorder.emit("client_exit", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        recorder.emit("client_end")
        recorder.close()


if __name__ == "__main__":
    main()

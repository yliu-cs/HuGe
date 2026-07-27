"""Replay a 64d latent-token .npy through the calc_proprio sim + WBC stack,
but capture an ego-view video instead of numeric g1_debug state. This is for
visual inspection, not pseudo-proprio export. Sim-side steps use
.venv_sim/bin/python, never bare python3.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GR00T_DIR = REPO_ROOT / "GR00T-WholeBodyControl"
VENV_SIM_PYTHON = GR00T_DIR / ".venv_sim" / "bin" / "python"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
LAUNCH_SIM_SCRIPT = REPO_ROOT / "scripts" / "preprocess" / "calc_proprio" / "launch_headless_sim.py"
RECORD_SCRIPT = REPO_ROOT / "scripts" / "preprocess" / "proprietary_ego" / "record_sim_video.py"
DEPLOY_BINARY = GR00T_DIR / "gear_sonic_deploy" / "target" / "release" / "g1_deploy_onnx_ref"
PLAYER_SCRIPT = "gear_sonic/scripts/sonic_encoder_input_player.py"

DEPLOY_ARGS = [
    "lo",
    "policy/release/model_decoder.onnx",
    "reference/example/",
    "--input-type", "zmq_manager",
    "--output-type", "all",
    "--hand-type", "dex3",
    "--disable-crc-check",
    "--obs-config", "policy/release/observation_config.yaml",
    "--encoder-file", "policy/release/model_encoder.onnx",
    "--planner-file", "planner/target_vel/V2/planner_sonic.onnx",
]

REQUIRED_PORTS = (5555, 5556, 5557)
DEPLOY_INIT_TIMEOUT_S = 150.0
DEPLOY_INIT_PATTERN = re.compile(r"Init Done")
RECORD_STARTUP_MARGIN_S = 2.0
RECORD_TAIL_MARGIN_S = 15.0
DEPLOY_STARTUP_STARTUP_S = 11.0  # cold-start fall/reset settling period, see handoff doc section 3


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def check_ports_free() -> None:
    busy = [p for p in REQUIRED_PORTS if not port_free(p)]
    if busy:
        raise RuntimeError(f"port(s) {busy} already in use -- kill leftover sim/deploy processes first")


def wait_for_log_pattern(log_path: Path, pattern: re.Pattern, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if log_path.exists() and pattern.search(log_path.read_text(errors="ignore")):
            return
        time.sleep(1.0)
    raise TimeoutError(f"timed out after {timeout_s}s waiting for {pattern.pattern!r} in {log_path}")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", required=True, help="path to a [T,64] latent-token .npy")
    parser.add_argument("--out", required=True, help="output .mp4 path")
    parser.add_argument("--work-dir", default=None, help="scratch dir for logs; default resource/tmp/render_video")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--fps", type=float, default=30.0, help="output video fps (camera publish rate)")
    parser.add_argument("--token-hz", type=float, default=50.0, help="native token capture rate; drives real-time playback speed")
    args = parser.parse_args()

    tokens_path = Path(args.tokens).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    num_tokens = int(np.load(tokens_path, mmap_mode="r").shape[0])
    episode_duration_s = num_tokens / args.token_hz

    work_dir = Path(args.work_dir).resolve() if args.work_dir else REPO_ROOT / "resource" / "tmp" / "render_video"
    work_dir.mkdir(parents=True, exist_ok=True)
    sim_log = work_dir / "sim.log"
    deploy_log = work_dir / "deploy.log"
    record_log = work_dir / "record.log"
    player_log = work_dir / "player.log"

    print(f"tokens={tokens_path} ({num_tokens} tokens, {episode_duration_s:.1f}s) work_dir={work_dir} out={out_path}")

    check_ports_free()

    sim_proc = None
    deploy_proc = None
    try:
        env = {**os.environ, "MUJOCO_GL": "egl"}
        with open(sim_log, "wb") as f:
            sim_proc = subprocess.Popen(
                [str(VENV_SIM_PYTHON), str(LAUNCH_SIM_SCRIPT), "--enable-image-publish", "--camera-port", str(args.camera_port)],
                cwd=GR00T_DIR, stdout=f, stderr=subprocess.STDOUT, env=env,
            )
        time.sleep(3.0)
        if sim_proc.poll() is not None:
            raise RuntimeError(f"sim process exited early (code {sim_proc.returncode}); see {sim_log}")

        with open(deploy_log, "wb") as f:
            deploy_proc = subprocess.Popen(
                [str(DEPLOY_BINARY), *DEPLOY_ARGS],
                cwd=GR00T_DIR / "gear_sonic_deploy", stdout=f, stderr=subprocess.STDOUT,
            )
        wait_for_log_pattern(deploy_log, DEPLOY_INIT_PATTERN, DEPLOY_INIT_TIMEOUT_S)
        print("deploy: Init Done")

        # Recorder and player must run back-to-back with no intervening delay
        # (same slow-joiner concern as align_sim_replay.py's capture step).
        record_duration = episode_duration_s + DEPLOY_STARTUP_STARTUP_S + RECORD_TAIL_MARGIN_S
        shell_cmd = (
            f"nohup {VENV_PYTHON} {RECORD_SCRIPT} --port {args.camera_port} --fps {args.fps} "
            f"--duration {record_duration:.1f} --out {out_path} > {record_log} 2>&1 &\n"
            f"RECORD_PID=$!\n"
            f"sleep {RECORD_STARTUP_MARGIN_S}\n"
            f"cd {GR00T_DIR}\n"
            f"{VENV_SIM_PYTHON} {PLAYER_SCRIPT} --latent-input-file {tokens_path} "
            f"--hz {args.token_hz} --source-hz {args.token_hz} --duration 0 > {player_log} 2>&1\n"
            f"wait $RECORD_PID\n"
        )
        run(["bash", "-c", shell_cmd])

        print(f"done -> {out_path}")
    finally:
        for proc, name in ((deploy_proc, "deploy"), (sim_proc, "sim")):
            if proc is not None and proc.poll() is None:
                print(f"terminating {name} (pid {proc.pid})")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()

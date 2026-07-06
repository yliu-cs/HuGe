"""Single-episode orchestrator for the ego cola pseudo-proprio pipeline.

Wraps resource/info/ego_cola_pseudo_proprio_handoff.md section 7 (stages 0-6)
into one script: launch headless sim -> launch WBC deploy binary -> capture
g1_debug frames + replay the token stream (back-to-back in one subprocess
call, see note below) -> align capture to tokens -> assemble + hand-correct
the pseudo proprio.

Hard constraints (see handoff doc section 0):
  - never touches any file under GR00T-WholeBodyControl/
  - Huge-side steps run under sys.executable (invoke this script itself with
    .venv/bin/python); sim-side steps run under
    GR00T-WholeBodyControl/.venv_sim/bin/python, never bare `python3`
  - single-episode only; this script does not loop over episodes itself

Known pitfall this script exists to avoid: capture and player must run
back-to-back with no intervening delay (a ZMQ slow-joiner / fixed-duration
capture window problem), so both are launched from one `bash -c` string
instead of two separate subprocess calls.
"""

from __future__ import annotations

import argparse
import glob
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
ALIGN_SCRIPT = REPO_ROOT / "scripts" / "preprocess" / "proprietary_ego" / "align_sim_replay.py"
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "preprocess" / "proprietary_ego" / "export_pseudo_proprio.py"
LAUNCH_SIM_SCRIPT = REPO_ROOT / "scripts" / "preprocess" / "proprietary_ego" / "launch_headless_sim.py"
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

REQUIRED_PORTS = (5556, 5557)
DEPLOY_INIT_TIMEOUT_S = 150.0
DEPLOY_INIT_PATTERN = re.compile(r"Init Done")
CAPTURE_STARTUP_MARGIN_S = 2.0
CAPTURE_TAIL_MARGIN_S = 15.0
DEPLOY_STARTUP_STARTUP_S = 11.0  # cold-start fall/reset noise period, see handoff doc section 3


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
        raise RuntimeError(
            f"port(s) {busy} already in use -- kill leftover sim/deploy processes first "
            f"(ps aux | grep, not pkill exit code -- see handoff doc section 9)"
        )


def wait_for_log_pattern(log_path: Path, pattern: re.Pattern, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    seen = 0
    while time.monotonic() < deadline:
        if log_path.exists():
            text = log_path.read_text(errors="ignore")
            if pattern.search(text):
                return
            seen = len(text)
        time.sleep(1.0)
    raise TimeoutError(f"timed out after {timeout_s}s waiting for {pattern.pattern!r} in {log_path} (last read {seen} bytes)")


def resolve_tokens(cola_root: Path, episode_num: int) -> tuple[Path, Path]:
    ep_dir = cola_root / str(episode_num)
    tok64_matches = sorted(glob.glob(str(ep_dir / "*_latent_tokens_64d.npy")))
    tok66_matches = sorted(glob.glob(str(ep_dir / "*_latent_tokens_66d.npy")))
    if len(tok64_matches) != 1:
        raise FileNotFoundError(f"expected exactly one 64d token file under {ep_dir}, found {tok64_matches}")
    if len(tok66_matches) != 1:
        raise FileNotFoundError(f"expected exactly one 66d token file under {ep_dir}, found {tok66_matches}")
    return Path(tok64_matches[0]), Path(tok66_matches[0])


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode-num", type=int, required=True)
    parser.add_argument("--cola-root", default="resource/dataset/proprietary/ego/cola")
    parser.add_argument("--out-root", default="resource/dataset/proprietary/ego/cola_pseudo_proprio")
    parser.add_argument("--stats", default="resource/dataset/proprietary/real/cola_merged_cleaned_0530/meta/proprio_stats_union_sonic.json")
    parser.add_argument("--work-dir", default=None, help="scratch dir for pkl/npz/logs; default resource/tmp/ego_pipeline/ep_<n>")
    args = parser.parse_args()

    cola_root = (REPO_ROOT / args.cola_root).resolve()
    tok64, tok66 = resolve_tokens(cola_root, args.episode_num)
    num_tokens = int(__import__("numpy").load(tok64, mmap_mode="r").shape[0])

    work_dir = Path(args.work_dir) if args.work_dir else REPO_ROOT / "resource" / "tmp" / "ego_pipeline" / f"ep_{args.episode_num}"
    work_dir.mkdir(parents=True, exist_ok=True)
    sim_log = work_dir / "sim.log"
    deploy_log = work_dir / "deploy.log"
    capture_log = work_dir / "capture.log"
    player_log = work_dir / "player.log"
    frames_pkl = work_dir / "frames.pkl"
    aligned_npz = work_dir / "aligned.npz"

    print(f"episode {args.episode_num}: tok64={tok64} ({num_tokens} tokens), tok66={tok66}, work_dir={work_dir}")

    check_ports_free()

    sim_proc = None
    deploy_proc = None
    try:
        # Stage 1: headless sim (persistent for the duration of this episode's replay)
        with open(sim_log, "wb") as f:
            sim_proc = subprocess.Popen(
                [str(VENV_SIM_PYTHON), str(LAUNCH_SIM_SCRIPT)],
                cwd=GR00T_DIR, stdout=f, stderr=subprocess.STDOUT,
            )
        time.sleep(3.0)  # let BaseSimulator construct + disable elastic band before deploy connects
        if sim_proc.poll() is not None:
            raise RuntimeError(f"sim process exited early (code {sim_proc.returncode}); see {sim_log}")

        # Stage 2: WBC deploy binary
        with open(deploy_log, "wb") as f:
            deploy_proc = subprocess.Popen(
                [str(DEPLOY_BINARY), *DEPLOY_ARGS],
                cwd=GR00T_DIR / "gear_sonic_deploy", stdout=f, stderr=subprocess.STDOUT,
            )
        wait_for_log_pattern(deploy_log, DEPLOY_INIT_PATTERN, DEPLOY_INIT_TIMEOUT_S)
        print("deploy: Init Done")

        # Stages 3+4: capture and player MUST run back-to-back in one shell call
        # (see module docstring / handoff doc section 9 -- splitting this across
        # two subprocess calls loses frames to inter-call latency).
        capture_duration = num_tokens / 50.0 + DEPLOY_STARTUP_STARTUP_S + CAPTURE_TAIL_MARGIN_S
        shell_cmd = (
            f"nohup {VENV_PYTHON} {ALIGN_SCRIPT} capture --duration {capture_duration:.1f} "
            f"--out {frames_pkl} > {capture_log} 2>&1 &\n"
            f"CAPTURE_PID=$!\n"
            f"sleep {CAPTURE_STARTUP_MARGIN_S}\n"
            f"cd {GR00T_DIR}\n"
            f"{VENV_SIM_PYTHON} {PLAYER_SCRIPT} --latent-input-file {tok64} "
            f"--hz 50 --source-hz 50 --duration 0 > {player_log} 2>&1\n"
            f"wait $CAPTURE_PID\n"
        )
        run(["bash", "-c", shell_cmd])

        # Stage 5: align
        run([
            str(VENV_PYTHON), str(ALIGN_SCRIPT), "align",
            "--frames", str(frames_pkl), "--tokens", str(tok64), "--out", str(aligned_npz),
        ])
        report = (aligned_npz.parent / (aligned_npz.stem + ".report.json"))
        if report.exists():
            print(report.read_text())

        # Stage 6: assemble + hand-correct + self-check + write out
        run([
            str(VENV_PYTHON), str(EXPORT_SCRIPT),
            "--aligned", str(aligned_npz), "--episode-num", str(args.episode_num),
            "--cola-root", str(cola_root), "--out-root", args.out_root, "--stats", args.stats,
        ])
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

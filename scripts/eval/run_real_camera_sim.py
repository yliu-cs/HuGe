"""Launch real-camera / simulated-proprioception preview without editing GR00T.

Use --validate-only for a read-only preflight. A normal invocation starts the
model server, loopback WBC, MuJoCo and an OpenCV viewer; it never starts real WBC.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GROOT = ROOT / "GR00T-WholeBodyControl"
DEPLOY = GROOT / "gear_sonic_deploy"
EVENT_PREFIX = "HUGE_PREVIEW_EVENT "
PORTS = {"server": 5550, "action": 5556, "state": 5557, "video": 15555}
VIDEO_FPS = 50
DEFAULT_CKPT = (
    ".runs/finetune_real/qwenpi_v3/full/stage3/"
    "20260905_181319_qwenpi_v3_R110_E0_none/step_20000"
)


def event(name: str, **data: Any) -> None:
    print(EVENT_PREFIX + json.dumps({"event": name, **data}), flush=True)


def command_reader() -> queue.Queue[str]:
    """Local parent/child pipe only; unrelated to robot communication."""
    commands: queue.Queue[str] = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            commands.put(line.strip())
        commands.put("stop")

    threading.Thread(target=read, daemon=True).start()
    return commands


def worker_main(function: Any) -> None:
    parser = argparse.ArgumentParser(description=function.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    def interrupt(*_: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        function(json.loads(args.config.read_text()), args.viewer)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        event("error", reason=f"{type(exc).__name__}: {exc}")
        raise


def child_environment(*, python310: bool = False, egl: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if python310:
        env.pop("PYTHONPATH", None)
        env.pop("LD_LIBRARY_PATH", None)
    else:
        env["PYTHONPATH"] = str(ROOT / "src")
    if egl:
        env["MUJOCO_GL"] = "egl"
        env["PYOPENGL_PLATFORM"] = "egl"
        # Mesa otherwise selects llvmpipe on this RTX 5090 host, blocking
        # physics for ~250 ms per image. Scope vendor selection to simulation.
        nvidia_egl = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
        if nvidia_egl.is_file():
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(nvidia_egl)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path(DEFAULT_CKPT))
    parser.add_argument("--camera-host", default="192.168.123.164")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--prompt", default="Walk forward, grab the cola and throw into the trash bin")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-horizon", type=int, choices=[50], default=50)
    parser.add_argument("--action-publish-rate", type=int, choices=[50], default=50)
    parser.add_argument("--rate", type=float, default=2.5,
                        help="Existing scheduler: wait 1/rate seconds after each result")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--data-timeout", type=float, default=2.0)
    parser.add_argument("--inference-timeout", type=float, default=60.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    for key in ("rate", "startup_timeout", "data_timeout", "inference_timeout"):
        value = getattr(args, key)
        if not 0 < value < float("inf"):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if not 1 <= args.camera_port <= 65535:
        parser.error("--camera-port must be between 1 and 65535")
    if args.camera_host in {"localhost", "127.0.0.1", "::1"} and args.camera_port == PORTS["video"]:
        parser.error("The third-person preview stream cannot be used as the real-camera input")
    args.ckpt = args.ckpt.resolve()
    args.log_dir = (args.log_dir or ROOT / ".runs/sim_preview" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if args.log_dir == GROOT or GROOT in args.log_dir.parents:
        parser.error("--log-dir must be outside GR00T-WholeBodyControl")
    return args


def preflight(args: argparse.Namespace) -> None:
    required = [
        ROOT / ".venv/bin/python", GROOT / ".venv_sim/bin/python",
        GROOT / ".venv_inference/bin/python",
        DEPLOY / "target/release/g1_deploy_onnx_ref",
        DEPLOY / "policy/release/model_decoder.onnx",
        DEPLOY / "policy/release/model_encoder.onnx",
        DEPLOY / "planner/target_vel/V2/planner_sonic.onnx",
        DEPLOY / "policy/release/observation_config.yaml",
        args.ckpt / "config.json", args.ckpt / "proprio_stats_sonic.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise RuntimeError(f"Unresolved LFS asset: {path}")
    if not any(args.ckpt.glob("*.safetensors")):
        raise FileNotFoundError(f"No safetensors checkpoint in {args.ckpt}")
    motion = DEPLOY / "reference/example"
    csvs = list(motion.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No reference motion CSVs in {motion}")
    for path in csvs:
        with path.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise RuntimeError(f"Unresolved reference motion: {path}")
    # Read /proc, rather than probing by connecting to another controller.
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            port = int(fields[1].split(":")[1], 16)
            if fields[3] == "0A" and port in PORTS.values():
                raise RuntimeError(f"Local port {port} is already listening; stop the conflicting stack first")
    own_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == own_pid:
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        names = {Path(os.fsdecode(a)).name for a in argv[:3] if a}
        if names & {"g1_deploy_onnx_ref", "run_sim_loop.py", "launch_headless_sim.py", "real_camera_sim_env.py"}:
            raise RuntimeError(f"Existing WBC/simulator process {proc.name}; do not run two stacks together")
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("OpenCV needs a graphical session (DISPLAY/WAYLAND_DISPLAY is absent)")
    for python, code in (
        (GROOT / ".venv_inference/bin/python",
         "from gear_sonic.scripts import run_vla_inference; print('VLA imports OK')"),
        (GROOT / ".venv_sim/bin/python",
         "from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator; "
         "from gear_sonic.scripts import run_camera_viewer; import cv2; "
         "gui = next(s.strip() for s in cv2.getBuildInformation().splitlines() if 'GUI:' in s); "
         "assert 'NONE' not in gui, 'OpenCV has no GUI backend'; print(gui)"),
    ):
        check = subprocess.run([str(python), "-c", code], cwd=ROOT,
                               env=child_environment(python310=True), capture_output=True, text=True, timeout=30)
        if check.returncode:
            raise RuntimeError(f"Runtime import check failed ({python}): {check.stderr}")
        print(check.stdout.strip(), flush=True)
    print("Preflight: paths, assets, imports, GUI backend and process/port conflicts OK", flush=True)


def prepare_models(run_dir: Path) -> dict[str, str]:
    result = {}
    for name, relative, prefix in (
        ("decoder", "policy/release/model_decoder.onnx", "policy_"),
        ("encoder", "policy/release/model_encoder.onnx", "encoder_"),
        ("planner", "planner/target_vel/V2/planner_sonic.onnx", "planner_"),
    ):
        source = DEPLOY / relative
        # WBC derives the planner version from the path (target_vel/V2).
        # Preserve that layout when relocating assets away from GR00T.
        target_dir = run_dir / "wbc_models" / Path(relative).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        shutil.copy2(source, target)
        cache = source.with_name(prefix + source.stem + ".trt")
        if cache.is_file():
            shutil.copy2(cache, target_dir / cache.name)
        result[name] = str(target)
    return result


def wbc_command(models: dict[str, str], run_dir: Path, cycle: int) -> list[str]:
    return [
        str(DEPLOY / "target/release/g1_deploy_onnx_ref"), "lo", models["decoder"],
        str(DEPLOY / "reference/example"),
        "--obs-config", str(DEPLOY / "policy/release/observation_config.yaml"),
        "--encoder-file", models["encoder"], "--planner-file", models["planner"],
        "--input-type", "zmq_manager", "--output-type", "zmq",
        "--zmq-host", "127.0.0.1", "--zmq-port", str(PORTS["action"]),
        "--zmq-out-port", str(PORTS["state"]), "--hand-type", "dex3",
        "--disable-crc-check", "--logs-dir", str(run_dir / f"wbc_{cycle}"),
    ]


@dataclass
class Child:
    name: str
    process: subprocess.Popen[str]
    reader: threading.Thread
    flags: set[str] = field(default_factory=set)

    def send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()


class Supervisor:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.children: dict[str, Child] = {}
        self.messages: queue.Queue[tuple[Child, dict[str, Any]]] = queue.Queue()

    def start(self, name: str, command: list[str], env: dict[str, str], suffix: str = "",
              cwd: Path | None = None) -> Child:
        print(f"Starting {name}{suffix}", flush=True)
        # WBC polls raw stdin in addition to ZMQ; an open, empty pipe could
        # block that reader. Only our two adapters need a parent command pipe.
        stdin = subprocess.PIPE if name in {"sim", "client"} else subprocess.DEVNULL
        process = subprocess.Popen(command, cwd=cwd or self.run_dir, env=env, stdin=stdin,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors="replace", bufsize=1, start_new_session=True)
        child = Child(name, process, threading.Thread())

        def read_output() -> None:
            assert process.stdout is not None
            with (self.run_dir / f"{name}{suffix}.log").open("w") as log:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(EVENT_PREFIX):
                        self.messages.put((child, json.loads(line[len(EVENT_PREFIX):])))
                    elif name == "wbc" and line.strip() == "Init Done":
                        self.messages.put((child, {"event": "initialized"}))
                    elif name == "wbc" and "[ZMQManager] ZMQ streaming enabled" in line:
                        self.messages.put((child, {"event": "streaming"}))
                    elif name == "wbc" and "[ERROR] Safety check failed" in line:
                        self.messages.put((child, {"event": "error", "reason": line.strip()}))

        child.reader = threading.Thread(target=read_output, daemon=True)
        self.children[name] = child
        child.reader.start()
        return child

    def poll(self) -> tuple[str, dict[str, Any]] | None:
        try:
            child, message = self.messages.get(timeout=0.05)
        except queue.Empty:
            return None
        if self.children.get(child.name) is not child:
            return None  # An old cycle's final output is not a new cycle's event.
        child.flags.add(message["event"])
        if message["event"] == "error":
            raise RuntimeError(f"{child.name}: {message['reason']} (see {self.run_dir})")
        return child.name, message

    def stop(self, name: str) -> None:
        child = self.children.pop(name, None)
        if child is None:
            return
        process = child.process
        # Always signal the group: simulator image workers can outlive their parent.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        if process.stdin:
            process.stdin.close()
        child.reader.join(timeout=2)
        if process.stdout:
            process.stdout.close()

    def close(self) -> None:
        for name in ("client", "wbc", "viewer", "sim", "server"):
            self.stop(name)


def run(args: argparse.Namespace) -> None:
    # Suppress crash dumps anywhere, including the read-only external project.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    args.log_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(root=str(ROOT), groot=str(GROOT), ports=PORTS)
    config_path = args.log_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    models = prepare_models(args.log_dir)
    base_env = child_environment()
    base_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    python310_env = child_environment(python310=True)
    worker = ROOT / "scripts/eval/real_camera_sim_client.py"
    sim_worker = ROOT / "scripts/eval/real_camera_sim_env.py"
    client_command = [str(GROOT / ".venv_inference/bin/python"), str(worker), "--config", str(config_path)]
    supervisor = Supervisor(args.log_dir)
    try:
        supervisor.start("server", [str(ROOT / ".venv/bin/python"), str(ROOT / "src/huge/inference/run_huge_server.py"),
            "--ckpt", str(args.ckpt), "--proprio-stats-path", str(args.ckpt / "proprio_stats_sonic.json"),
            "--host", "127.0.0.1", "--port", str(PORTS["server"]), "--device", args.device,
            "--action-horizon", "50", "--no-state-tap-enable", "--no-closed-loop-enable",
            "--log-dir", str(args.log_dir / "server")], base_env, cwd=ROOT)
        supervisor.start("sim", [str(GROOT / ".venv_sim/bin/python"), str(sim_worker), "--config", str(config_path)],
                         child_environment(python310=True, egl=True))
        cycle = 0
        phase = "sim"
        deadline = time.monotonic() + args.startup_timeout
        while True:
            update = supervisor.poll()
            if update:
                name, message = update
                kind = message["event"]
                print(f"[{name}] {kind}", flush=True)
                if name == "viewer" and kind == "user_stop":
                    return
                if name == "sim" and kind == "fell":
                    supervisor.stop("client")
                    supervisor.stop("wbc")
                    cycle += 1
                    supervisor.children["sim"].send("reset")
                    phase = "reset"
                    deadline = time.monotonic() + args.startup_timeout
                elif name == "sim" and kind in {"ready", "reset_ready"}:
                    if "viewer" not in supervisor.children:
                        # The inference environment's OpenCV is headless. The
                        # existing simulation environment has a Qt GUI backend.
                        viewer_command = [str(GROOT / ".venv_sim/bin/python"), *client_command[1:], "--viewer"]
                        supervisor.start("viewer", viewer_command, python310_env)
                    supervisor.start("wbc", wbc_command(models, args.log_dir, cycle), os.environ.copy(), f"_{cycle}")
                    supervisor.start("client", client_command, python310_env, f"_{cycle}")
                    phase = "initialize"
            if phase == "initialize":
                wbc = supervisor.children["wbc"]
                client = supervisor.children["client"]
                if "initialized" in wbc.flags and "inputs_ready" in client.flags:
                    client.send("initialize")
                    phase = "warmup"
            if phase == "warmup":
                client = supervisor.children["client"]
                if ("policy_ready" in client.flags and "streaming" in supervisor.children["wbc"].flags
                        and "ready" in supervisor.children["viewer"].flags):
                    supervisor.children["sim"].send("release")
                    phase = "release"
            if phase == "release" and update == ("sim", {"event": "released"}):
                supervisor.children["client"].send("run")
                phase = "running"
                print("Preview running: real RGB + sim proprio; elastic band OFF. Ctrl+C to stop.", flush=True)
            for name, child in list(supervisor.children.items()):
                code = child.process.poll()
                if code is not None:
                    if child.reader.is_alive() or not supervisor.messages.empty():
                        continue  # Consume the final reason/user-stop event first.
                    if name == "viewer" and code == 0 and "user_stop" in child.flags:
                        return
                    raise RuntimeError(f"{name} exited with code {code}; see {args.log_dir}")
            if phase != "running" and time.monotonic() > deadline:
                raise TimeoutError(f"Startup timed out in phase {phase}; see {args.log_dir}")
    finally:
        supervisor.close()


def main() -> None:
    args = parse_args()
    try:
        preflight(args)
        if args.validate_only:
            print("Read-only preflight complete; no models, DDS participants, sockets or windows started.")
            return
        run(args)
    except KeyboardInterrupt:
        print("Preview stopped by user.")
    except Exception as exc:
        print(f"Preview failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

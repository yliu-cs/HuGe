"""Kappa real-robot bridge: WebSocket inference server <-> WBC ZMQ pose stream.

This script is the Kappa-flavoured analogue of
``GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py`` for the
NVIDIA GR00T stack. Differences:

* The Kappa server speaks WebSocket JSON (see ``src/inference/kappa_serve_rtc.py``)
  instead of Isaac-GR00T's ZMQ REQ/REP ``PolicyClient``.
* The server's ``RealTimeChunkController`` already runs the RTC chunker
  internally at ``ctrl_hz`` (default 50 Hz). The bridge runs three
  concurrent asyncio tasks sharing a single ``SharedSlot``: ``obs_publisher``
  paces obs sends at ``tick_hz``, ``action_consumer`` drains server-pushed
  actions into the slot, and ``wbc_publisher`` fires the latest slot value
  to WBC at strict ``tick_hz`` cadence. Decoupling obs send from action
  recv means WBC sees a steady 50 Hz frame_index stream even when an
  individual obs round-trip takes longer than the tick period.
* The server returns ``[64 SONIC latent | 2 hand bits]``; the bridge
  thresholds the 2 hand bits at 0.5 and expands them to 7-DoF left/right
  hand joints using ``G1GripperInverseKinematicsSolver`` (open = zeros,
  closed = ``_get_middle_close_q_desired()``).

Run via the orchestrator (``scripts/deploy/run_kappa_deploy.sh``) or
manually after the server, WBC ``deploy.sh``, and keyboard publisher
are up.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tyro
import websockets
import zmq

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Cfg:
    """CLI config for the Kappa bridge."""

    # Kappa WebSocket server
    ws_host: str = "localhost"
    ws_port: int = 8765
    ws_path: str = "/predict"

    # Camera (ZMQ from camera server on robot)
    camera_host: str = "localhost"
    camera_port: int = 5555

    # Robot state (ZMQ SUB from C++ deploy, topic g1_debug)
    state_zmq_host: str = "localhost"
    state_zmq_port: int = 5557

    # Action out (ZMQ PUB consumed by C++ deploy's zmq_manager)
    action_zmq_host: str = "0.0.0.0"
    action_zmq_port: int = 5556

    # Keyboard (ZMQ SUB)
    keyboard_zmq_host: str = "localhost"
    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT

    # Inference
    prompt: str = "perform the task"
    tick_hz: float = 50.0
    hand_bit_threshold: float = 0.5

    # Debug: print actions instead of publishing ZMQ
    dry_run: bool = False

    # Per-tick dump of (image, proprio, predicted sonic) under this dir,
    # mirroring the training-dataset layout so the run can be diffed against
    # offline data via ``PropEpisodeSource``. Empty = disabled, zero overhead.
    log_dir: str = ""


# ---------------------------------------------------------------------------
# Action packing (Protocol v4 — token_state + frame_index + hand joints)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4)."""
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data: dict[str, np.ndarray] = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }
    if left_hand_joints is not None:
        lh = np.asarray(left_hand_joints, dtype=np.float32)
        if lh.ndim == 1:
            lh = lh.reshape(1, 7)
        pose_data["left_hand_joints"] = lh
    if right_hand_joints is not None:
        rh = np.asarray(right_hand_joints, dtype=np.float32)
        if rh.ndim == 1:
            rh = rh.reshape(1, 7)
        pose_data["right_hand_joints"] = rh

    return pack_pose_message(pose_data, topic="pose", version=4)


# ---------------------------------------------------------------------------
# Hand-bit expansion
# ---------------------------------------------------------------------------


LEFT_CLOSED_Q = (
    G1GripperInverseKinematicsSolver("left")._get_middle_close_q_desired().astype(np.float32)
)
RIGHT_CLOSED_Q = (
    G1GripperInverseKinematicsSolver("right")._get_middle_close_q_desired().astype(np.float32)
)
HAND_OPEN_Q = np.zeros(7, dtype=np.float32)


def hand_bits_to_joints(bits: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    left = LEFT_CLOSED_Q if bits[0] > threshold else HAND_OPEN_Q
    right = RIGHT_CLOSED_Q if bits[1] > threshold else HAND_OPEN_Q
    return left, right


# ---------------------------------------------------------------------------
# Hand-state -> 1-bit detector (mirror of training-side observation.hand_binary_state)
# ---------------------------------------------------------------------------
#
# Dex3 ``_get_middle_close_q_desired()`` puts q[1] at ±0.7 (closed) and 0
# (open), with the right hand mirroring the left's sign convention. abs()
# collapses the L/R polarity into a single 0..0.7 magnitude, and 0.35 is the
# midpoint — wide hysteresis-free detector that matches the training-side
# binary state without needing the IK solver at runtime.
_HAND_CLOSED_Q1_THRESHOLD = 0.35

# Once-per-process diagnostic flags so we don't spam every tick.
_HAND_SOURCE_LOGGED: set[str] = set()


def _q7_to_hand_bit(q7) -> float | None:
    """Threshold a Dex3 7-DoF joint vector into a binary closed bit, or
    return ``None`` if the array is missing / wrong-shape so the caller can
    fall back to the next preferred source."""
    if q7 is None:
        return None
    arr = np.asarray(q7, dtype=np.float32)
    if arr.shape != (7,):
        return None
    return float(abs(arr[1]) > _HAND_CLOSED_Q1_THRESHOLD)


def _hand_bits_from_state(state_msg: dict) -> np.ndarray:
    """Resolve ``[left_closed, right_closed]`` (float32, 0/1) from the
    g1_debug ZMQ message.

    Preference order (each tier is taken only if BOTH L+R fields are
    populated and well-shaped):

      1. ``left_hand_q_measured`` / ``right_hand_q_measured``  — real Dex3
         encoder readback, matches training ``observation.hand_binary_state``
         semantics. Always present on real-robot C++ deploys built with the
         ``output_data_map_`` visualisation block.
      2. ``left_hand_q`` / ``right_hand_q``                    — commanded
         Dex3 setpoints from the C++ state logger. Tracks (1) within one
         tick on real hardware; what the sim-mode C++ deploy publishes.
      3. ``last_left_hand_action`` / ``last_right_hand_action`` — last
         action the C++ deploy executed. Same polarity as (1)/(2).
      4. Direct ``hand_bits`` field — used by the offline-cache publisher
         in ``scripts/eval/sim_validate.py`` to forward the cache's already-
         binary bits when it doesn't synthesise the 7-DoF q vector.

    Any successful tier below "measured" prints a one-shot info line so a
    real-deploy operator notices a stale C++ build or a degenerate cache.
    """
    measured_l, measured_r = state_msg.get("left_hand_q_measured"), state_msg.get("right_hand_q_measured")
    lb, rb = _q7_to_hand_bit(measured_l), _q7_to_hand_bit(measured_r)
    if lb is not None and rb is not None:
        return np.array([lb, rb], dtype=np.float32)

    cmd_l, cmd_r = state_msg.get("left_hand_q"), state_msg.get("right_hand_q")
    lb, rb = _q7_to_hand_bit(cmd_l), _q7_to_hand_bit(cmd_r)
    if lb is not None and rb is not None:
        if "commanded" not in _HAND_SOURCE_LOGGED:
            _HAND_SOURCE_LOGGED.add("commanded")
            print(
                "[bridge] hand-state: *_q_measured missing in g1_debug; "
                "thresholding commanded *_q instead. Fine in sim, "
                "may indicate a stale C++ build on real hardware."
            )
        return np.array([lb, rb], dtype=np.float32)

    last_l, last_r = state_msg.get("last_left_hand_action"), state_msg.get("last_right_hand_action")
    lb, rb = _q7_to_hand_bit(last_l), _q7_to_hand_bit(last_r)
    if lb is not None and rb is not None:
        if "last_action" not in _HAND_SOURCE_LOGGED:
            _HAND_SOURCE_LOGGED.add("last_action")
            print(
                "[bridge] hand-state: only last_*_hand_action available; "
                "using it as a proxy for current hand position."
            )
        return np.array([lb, rb], dtype=np.float32)

    direct = state_msg.get("hand_bits")
    if direct is not None:
        arr = np.asarray(direct, dtype=np.float32)
        if arr.shape == (2,):
            if "direct" not in _HAND_SOURCE_LOGGED:
                _HAND_SOURCE_LOGGED.add("direct")
                print("[bridge] hand-state: using direct 'hand_bits' field from state_msg.")
            return arr

    if "zeros" not in _HAND_SOURCE_LOGGED:
        _HAND_SOURCE_LOGGED.add("zeros")
        print(
            "[bridge] WARNING: no usable hand state in g1_debug; "
            "writing proprio[29:31]=0. Policy will see 'hands always open' "
            "in proprio — output bits can still be correct but the "
            "feedback loop is broken."
        )
    return np.zeros(2, dtype=np.float32)


# ---------------------------------------------------------------------------
# Proprio assembly (64-D RAW; server normalizes via q01/q99)
# ---------------------------------------------------------------------------


def assemble_proprio(
    state_msg: dict,
    prev_jp: np.ndarray | None,
    tick_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 64-D proprio vector from a g1_debug state message.

    Layout (see README.md and src/inference/kappa_serve_rtc.py):
        [ 0:29]  jp        — MuJoCo joint order, default_angles offset baked in by C++
        [29:31]  hand_bits — binary closed/open per hand, derived from the
                             best hand-state field available in state_msg
                             (see ``_hand_bits_from_state`` for the fallback
                             chain). Must match the training-side
                             ``observation.hand_binary_state`` semantics.
        [31:60]  jv        — finite-diff of jp at tick_hz Hz
        [60:64]  quat_xyzw — pelvis orientation (C++ publishes wxyz; reorder)
    """
    jp = np.asarray(state_msg["body_q"], dtype=np.float32)
    if jp.shape != (29,):
        raise ValueError(f"body_q expected shape (29,), got {jp.shape}")
    if prev_jp is None:
        jv = np.zeros(29, dtype=np.float32)
    else:
        jv = (jp - prev_jp) * float(tick_hz)
    bq = np.asarray(state_msg["base_quat"], dtype=np.float32)
    if bq.shape != (4,):
        raise ValueError(f"base_quat expected shape (4,), got {bq.shape}")
    quat_xyzw = bq[[1, 2, 3, 0]]

    out = np.empty(64, dtype=np.float32)
    out[0:29] = jp
    out[29:31] = _hand_bits_from_state(state_msg)
    out[31:60] = jv
    out[60:64] = quat_xyzw
    return out, jp


# ---------------------------------------------------------------------------
# Per-loop mutable state
# ---------------------------------------------------------------------------


class LoopState:
    def __init__(self, prompt: str):
        self.pause: bool = True
        self.cpp_loop_running: bool = False
        self.cpp_mode: str = "OFF"  # "OFF" | "PLANNER" | "POSE"
        self.initial_pose_left_hand_closed: bool = False
        self.initial_pose_right_hand_closed: bool = False
        self.frame: int = 0
        self.prev_jp: np.ndarray | None = None
        self.prompt: str = prompt


# ---------------------------------------------------------------------------
# Keyboard handler (ported from run_vla_inference.py:465-536)
# ---------------------------------------------------------------------------


def _green(msg: str) -> None:
    print(f"\033[92m{msg}\033[0m")


def make_keyboard_handler(
    kb: ZMQKeyboardSubscriber,
    zmq_socket: zmq.Socket,
    state: LoopState,
):
    PROMPT_PREFIX = "prompt:"

    def publish_initial_pose() -> None:
        print("Moving to initial pose")
        lh = LEFT_CLOSED_Q if state.initial_pose_left_hand_closed else HAND_OPEN_Q
        rh = RIGHT_CLOSED_Q if state.initial_pose_right_hand_closed else HAND_OPEN_Q
        msg = pack_latent_action_message(
            motion_token=LATENT_INITIAL_MOTION_TOKEN,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=lh,
            right_hand_joints=rh,
        )
        zmq_socket.send(msg)
        _green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def send_cpp_command(start: bool, planner: bool = False) -> bool:
        try:
            cmd = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd)
            time.sleep(0.01)
            state.cpp_loop_running = start
            if start:
                state.cpp_mode = "PLANNER" if planner else "POSE"
            else:
                state.cpp_mode = "OFF"
            mode_str = "planner" if planner else "pose"
            _green(f"Sent ZMQ command: {'start' if start else 'stop'} control loop ({mode_str} mode)")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"Warning: failed to send command: {e}")
            return False

    def handle() -> None:
        key = kb.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_PREFIX):
            new_prompt = key[len(PROMPT_PREFIX):]
            if new_prompt:
                old = state.prompt
                state.prompt = new_prompt
                _green(f'Inference prompt changed: "{old}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            state.frame = 0
            print("Reset ZMQ frame counter")
            publish_initial_pose()
            state.prev_jp = None
            if state.cpp_loop_running and state.cpp_mode == "PLANNER":
                if send_cpp_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
            elif not state.cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")
        elif key == "p":
            state.pause = not state.pause
            print(f"{'Paused' if state.pause else 'Resumed'} policy loop")
            if not state.pause:
                state.prev_jp = None  # suppress velocity spike on resume
        elif key == "k":
            if state.cpp_loop_running:
                planner = state.cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {state.cpp_mode} mode)...")
                if send_cpp_command(start=False, planner=planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_command(start=True, planner=True):
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if state.pause:
                        print("Note: Policy loop paused - press 'p' to resume")
        elif key == "[":
            state.initial_pose_left_hand_closed = not state.initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: "
                f"{'closed' if state.initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            state.initial_pose_right_hand_closed = not state.initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if state.initial_pose_right_hand_closed else 'open'}"
            )

    return handle


# ---------------------------------------------------------------------------
# Concurrent producer/consumer tasks
# ---------------------------------------------------------------------------
#
# We split the bridge into three independent asyncio tasks sharing a single
# ``SharedSlot`` for the latest action. This decouples WBC pacing from the
# WS round-trip and obs-encode cost so the ZMQ action stream to WBC fires
# at a strict ``cfg.tick_hz`` cadence even when an individual obs send is
# slow. The server's ``_send_loop`` already pushes a new action version per
# server tick — we simply consume them as they arrive.


class SharedSlot:
    """Latest action from the server. Single-writer/single-reader via the
    asyncio event loop, so no lock is needed."""

    __slots__ = ("action", "version")

    def __init__(self) -> None:
        self.action: np.ndarray | None = None
        self.version: int = -1


# ---------------------------------------------------------------------------
# Per-tick logger for training-dataset-format dumps
# ---------------------------------------------------------------------------


class BridgeLogger:
    """Per-tick logger that mirrors the training-dataset layout.

    Output under ``log_dir`` (loadable by ``PropEpisodeSource``):
      ``videos/chunk-000/observation.images.ego_view/episode_NNNNNN.mp4``
      ``g1_proprio/episode_NNNNNN.npy``         (T, 64) float32
      ``sonic_tokens/episode_NNNNNN.npy``       (T, 66) float32, NaN for invalid rows
      ``meta/tasks.jsonl``                      dedup-append-by-prompt
      ``meta/episodes.jsonl``                   one line per closed episode

    Plus a deploy-only sidecar:
      ``bridge_log/episode_NNNNNN.npz``         t_obs_send, version, valid_action

    One episode per bridge process invocation. Episode index = next free
    slot under ``g1_proprio/``, so repeated bridge runs into the same
    ``log_dir`` append episodes.

    Trade-offs / edge cases:
      * Numpy buffers live in memory until ``close()``. A SIGKILL between
        ``log_obs`` ticks loses every row of this episode; the mp4 (streamed
        via ffmpeg subprocess) survives but lacks moov — recover with
        ``ffmpeg -i bad.mp4 -c copy fixed.mp4``. Per-row disk flush at 50 Hz
        was rejected (~1 ms/tick is too costly on the bridge loop).
      * WS reconnect mid-episode: ``SharedSlot`` is rebuilt per session, so
        ``slot.action is None`` again until the new server's first reply
        arrives. Those rows are NaN-filled with ``valid_action=False``.
      * Pause ('p') freezes ``obs_publisher`` before the logger call, so no
        rows are written during pause — logged length == deployed ticks.
      * Mid-episode prompt changes via ``prompt:NEW`` keyboard hotkey are
        NOT reflected in ``episodes.jsonl`` — that file records only the
        prompt set at logger construction. Restart between prompts if you
        need per-episode prompt labels.
    """

    def __init__(self, log_dir: str, prompt: str, fps: float) -> None:
        self._closed = False
        if not log_dir:
            self.enabled = False
            return
        self.enabled = True
        self._prompt = prompt
        root = Path(log_dir)
        self._video_dir = root / "videos" / "chunk-000" / "observation.images.ego_view"
        self._proprio_dir = root / "g1_proprio"
        self._sonic_dir = root / "sonic_tokens"
        self._sidecar_dir = root / "bridge_log"
        self._meta_dir = root / "meta"
        for d in (
            self._video_dir,
            self._proprio_dir,
            self._sonic_dir,
            self._sidecar_dir,
            self._meta_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Next free episode index = max(existing g1_proprio/episode_*.npy) + 1.
        existing = sorted(self._proprio_dir.glob("episode_*.npy"))
        if existing:
            last_idx = max(int(p.stem.split("_")[-1]) for p in existing)
            self._ep = last_idx + 1
        else:
            self._ep = 0
        self._stem = f"episode_{self._ep:06d}"

        tasks_path = self._meta_dir / "tasks.jsonl"
        task_lines: list[dict] = []
        if tasks_path.exists():
            for raw in tasks_path.read_text().splitlines():
                raw = raw.strip()
                if raw:
                    try:
                        task_lines.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
        if not any(t.get("task") == prompt for t in task_lines):
            new_idx = max((t.get("task_index", -1) for t in task_lines), default=-1) + 1
            with tasks_path.open("a") as f:
                f.write(json.dumps({"task_index": new_idx, "task": prompt}) + "\n")

        self._mp4_path = self._video_dir / f"{self._stem}.mp4"
        # macro_block_size=1 keeps non-16-multiple frame dims (480x640) intact.
        self._writer = imageio.get_writer(
            str(self._mp4_path),
            fps=float(fps),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        self._proprio: list[np.ndarray] = []
        self._sonic: list[np.ndarray] = []
        self._t_obs: list[float] = []
        self._version: list[int] = []
        self._valid: list[bool] = []

        print(f"[bridge-log] writing {self._stem} to {root}")

    def log_obs(
        self,
        img: np.ndarray,
        proprio: np.ndarray,
        slot: SharedSlot,
        t_wall: float,
    ) -> None:
        """Append one row. Caller wraps this in ``asyncio.to_thread`` so the
        mp4 ffmpeg write doesn't stall the obs loop."""
        if not self.enabled or self._closed:
            return
        self._writer.append_data(img)
        self._proprio.append(proprio.astype(np.float32, copy=True))
        if slot.action is None:
            self._sonic.append(np.full(66, np.nan, dtype=np.float32))
            self._version.append(-1)
            self._valid.append(False)
        else:
            self._sonic.append(slot.action.astype(np.float32, copy=True))
            self._version.append(int(slot.version))
            self._valid.append(True)
        self._t_obs.append(float(t_wall))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return

        # Close the writer first to finalize the moov atom.
        try:
            self._writer.close()
        except Exception as e:  # noqa: BLE001
            print(f"[bridge-log] writer.close() failed: {e!r}")

        T = len(self._proprio)
        if T == 0:
            try:
                self._mp4_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            print(f"[bridge-log] empty {self._stem}; nothing saved")
            return

        proprio = np.stack(self._proprio, axis=0).astype(np.float32, copy=False)
        sonic = np.stack(self._sonic, axis=0).astype(np.float32, copy=False)
        np.save(self._proprio_dir / f"{self._stem}.npy", proprio)
        np.save(self._sonic_dir / f"{self._stem}.npy", sonic)
        np.savez(
            self._sidecar_dir / f"{self._stem}.npz",
            t_obs_send=np.array(self._t_obs, dtype=np.float64),
            version=np.array(self._version, dtype=np.int64),
            valid_action=np.array(self._valid, dtype=bool),
        )

        episodes_path = self._meta_dir / "episodes.jsonl"
        entry = {"episode_index": self._ep, "tasks": [self._prompt], "length": int(T)}
        with episodes_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())

        print(f"[bridge-log] closed {self._stem}: T={T}, mp4={self._mp4_path.name}")


async def obs_publisher(
    cfg: Cfg,
    ws,
    state_sub: ZMQStateSubscriber,
    cam_sub: ComposedCameraClientSensor,
    state: LoopState,
    slot: SharedSlot,
    logger: BridgeLogger,
) -> None:
    """Pace obs at ``tick_hz`` and push to the server WS without waiting for
    a reply. Replies are drained by ``action_consumer``. After the send, if
    a logger is enabled, snapshot (img, proprio, latest slot.action) for the
    training-format dump — done via ``asyncio.to_thread`` so the mp4 frame
    write can't stall this loop."""
    period = 1.0 / float(cfg.tick_hz)
    next_t = time.perf_counter()
    while True:
        if state.pause:
            await asyncio.sleep(0.05)
            next_t = time.perf_counter()
            continue

        state_msg = await asyncio.to_thread(state_sub.get_msg)
        cam_msg = await asyncio.to_thread(cam_sub.read)
        if state_msg is None or cam_msg is None:
            await asyncio.sleep(period)
            next_t = time.perf_counter()
            continue

        img = cam_msg["images"]["ego_view"]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        proprio, state.prev_jp = assemble_proprio(state_msg, state.prev_jp, cfg.tick_hz)

        # Binary WS frame: 4-byte BE header length + UTF-8 JSON header + raw
        # image bytes. Skips the PNG encode that previously cost ~5-15ms/tick
        # on 480x640 frames; server decodes via np.frombuffer + reshape.
        header = {
            "text": state.prompt,
            "proprio": proprio.tolist(),
            "img_shape": list(img.shape),
            "img_dtype": str(img.dtype),
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        payload = (
            len(header_bytes).to_bytes(4, "big") + header_bytes + img.tobytes()
        )
        await ws.send(payload)

        if logger.enabled:
            await asyncio.to_thread(logger.log_obs, img, proprio, slot, time.time())

        next_t += period
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_t = time.perf_counter()


async def action_consumer(ws, slot: SharedSlot) -> None:
    """Drain server-pushed actions and update ``slot`` with the latest one."""
    while True:
        raw = await ws.recv()
        try:
            reply = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[bridge] bad reply: {e!r}")
            continue
        sonic = np.asarray(reply.get("sonic", []), dtype=np.float32)
        if sonic.shape != (66,):
            print(f"[bridge] unexpected sonic shape {sonic.shape}; skipping")
            continue
        slot.action = sonic
        slot.version = int(reply.get("version", slot.version + 1))


async def wbc_publisher(
    cfg: Cfg,
    zmq_socket: zmq.Socket,
    state: LoopState,
    slot: SharedSlot,
    handle_kb,
) -> None:
    """At strict ``tick_hz`` cadence, pack the latest action in ``slot`` and
    publish it to WBC over ZMQ. Repeats the latest action if no new version
    has arrived since the last tick — WBC keys on ``frame_index`` and the
    SONIC decoder expects a steady 50 Hz pulse."""
    period = 1.0 / float(cfg.tick_hz)
    next_t = time.perf_counter()
    last_log = next_t
    while True:
        handle_kb()

        if state.pause:
            await asyncio.sleep(0.05)
            next_t = time.perf_counter()
            continue

        sonic = slot.action
        if sonic is None:
            # Server hasn't pushed a chunk yet (warm-up). Idle until it does.
            await asyncio.sleep(period)
            next_t = time.perf_counter()
            continue

        motion_token = sonic[:64]
        hand_bits = sonic[64:66]
        lh, rh = hand_bits_to_joints(hand_bits, cfg.hand_bit_threshold)
        version = slot.version

        msg = pack_latent_action_message(
            motion_token=motion_token,
            frame_index=np.array([state.frame], dtype=np.int64),
            left_hand_joints=lh,
            right_hand_joints=rh,
        )
        if cfg.dry_run:
            print(
                f"[dry-run] frame={state.frame} version={version} "
                f"motion[:4]={motion_token[:4]} bits={hand_bits}"
            )
        else:
            zmq_socket.send(msg)
            now = time.perf_counter()
            if now - last_log > 1.0:
                _green(
                    f"ZMQ: sent action frame={state.frame} version={version} "
                    f"bits={hand_bits.round(2).tolist()}"
                )
                last_log = now
        state.frame += 1

        next_t += period
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        else:
            next_t = time.perf_counter()


async def run_session(
    cfg: Cfg,
    ws,
    state_sub: ZMQStateSubscriber,
    cam_sub: ComposedCameraClientSensor,
    zmq_socket: zmq.Socket,
    state: LoopState,
    handle_kb,
    logger: BridgeLogger,
) -> None:
    """Run the three concurrent tasks for the lifetime of one WS connection.
    Returns (and propagates) when any task raises — the outer reconnect loop
    handles WS exceptions."""
    slot = SharedSlot()
    tasks = [
        asyncio.create_task(obs_publisher(cfg, ws, state_sub, cam_sub, state, slot, logger)),
        asyncio.create_task(action_consumer(ws, slot)),
        asyncio.create_task(wbc_publisher(cfg, zmq_socket, state, slot, handle_kb)),
    ]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc is not None:
                raise exc
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Outer run loop (reconnect on WS drop)
# ---------------------------------------------------------------------------


async def run(cfg: Cfg) -> None:
    state_sub = ZMQStateSubscriber(host=cfg.state_zmq_host, port=cfg.state_zmq_port)
    cam_sub = ComposedCameraClientSensor(server_ip=cfg.camera_host, port=cfg.camera_port)
    kb = ZMQKeyboardSubscriber(host=cfg.keyboard_zmq_host, port=cfg.keyboard_zmq_port)

    ctx = zmq.Context()
    zmq_socket = ctx.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{cfg.action_zmq_host}:{cfg.action_zmq_port}")
    time.sleep(0.1)
    _green(f"ZMQ action socket bound to tcp://{cfg.action_zmq_host}:{cfg.action_zmq_port}")

    state = LoopState(prompt=cfg.prompt)
    handle_kb = make_keyboard_handler(kb, zmq_socket, state)

    logger = BridgeLogger(cfg.log_dir, cfg.prompt, cfg.tick_hz)
    if not logger.enabled:
        print("[bridge-log] disabled (no --log-dir set)")

    url = f"ws://{cfg.ws_host}:{cfg.ws_port}{cfg.ws_path}"
    backoff = 0.5
    try:
        while True:
            try:
                async with websockets.connect(
                    url,
                    max_size=2**24,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    _green(f"[bridge] connected to {url}")
                    backoff = 0.5
                    await run_session(cfg, ws, state_sub, cam_sub, zmq_socket, state, handle_kb, logger)
            except (websockets.WebSocketException, OSError, ConnectionRefusedError) as e:
                # WebSocketException covers ConnectionClosed *and* handshake
                # failures like InvalidMessage (raised when the server's TCP
                # port is open but ASGI lifespan hasn't finished — common
                # while kappa_serve_rtc.py is still loading the Qwen3-VL
                # backbone). Without this, the bridge dies on its first
                # connect attempt during serve warm-up.
                print(f"[bridge] WS dropped: {e!r}; reconnecting in {backoff:.1f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
    except KeyboardInterrupt:
        print("[bridge] interrupted by user")
    finally:
        try:
            logger.close()
        except Exception as e:  # noqa: BLE001
            print(f"[bridge-log] close failed: {e!r}")
        zmq_socket.close()
        ctx.term()
        state_sub.close()
        kb.close()
        print("[bridge] shutdown complete")


def main() -> None:
    cfg = tyro.cli(Cfg)
    print(f"[bridge] starting with prompt: {cfg.prompt!r}")
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()

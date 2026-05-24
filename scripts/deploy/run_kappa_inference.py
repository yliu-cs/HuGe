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
import time
from dataclasses import dataclass

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
        [29:31]  hand_bits — 0,0 for fake-hand training data (placeholder)
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
    out[29:31] = 0.0
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


async def obs_publisher(
    cfg: Cfg,
    ws,
    state_sub: ZMQStateSubscriber,
    cam_sub: ComposedCameraClientSensor,
    state: LoopState,
) -> None:
    """Pace obs at ``tick_hz`` and push to the server WS without waiting for
    a reply. Replies are drained by ``action_consumer``."""
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
) -> None:
    """Run the three concurrent tasks for the lifetime of one WS connection.
    Returns (and propagates) when any task raises — the outer reconnect loop
    handles WS exceptions."""
    slot = SharedSlot()
    tasks = [
        asyncio.create_task(obs_publisher(cfg, ws, state_sub, cam_sub, state)),
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
                    await run_session(cfg, ws, state_sub, cam_sub, zmq_socket, state, handle_kb)
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

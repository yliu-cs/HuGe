"""Reuse SONIC observation/action helpers for automatic simulation-only preview."""

from __future__ import annotations

import copy
import queue
import threading
import time
from typing import Any

import numpy as np

from run_real_camera_sim import VIDEO_FPS, command_reader, event, worker_main


class Snapshot:
    """Feed an already received observation to the upstream assembly helpers."""
    def __init__(self, value: Any):
        self.value = value

    def read(self) -> Any:
        return self.value

    def get_msg(self) -> Any:
        return self.value


def validate_state(state: dict[str, Any]) -> None:
    for key, size in (("body_q", 29), ("left_hand_q", 7), ("right_hand_q", 7), ("base_quat", 4)):
        arr = np.asarray(state[key])
        if arr.shape != (size,) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid simulated state {key}: expected finite [{size}]")
    norm = np.linalg.norm(state["base_quat"])
    if not 0.9 <= norm <= 1.1:
        raise ValueError(f"Invalid simulated base quaternion norm {norm}")


def action_frames(action: dict[str, Any], api: Any, horizon: int) -> list[tuple[Any, Any, Any]]:
    tokens = np.asarray(api.get_action_field(action, "motion_token"), dtype=np.float32)
    if tokens.shape == (1, horizon, 64):
        tokens = tokens[0]
    if tokens.shape != (horizon, 64) or not np.isfinite(tokens).all():
        raise ValueError(f"Expected finite motion_token [1,{horizon},64], got {tokens.shape}")
    if np.abs(tokens).max() > 1.25:
        raise ValueError("SONIC token exceeds existing +/-1.25 bound")
    # The upstream per-frame helper clamps short hand arrays; enforce matching
    # horizons here so a malformed chunk cannot silently repeat a hand command.
    hand_fields = ("left_hand_joints", "right_hand_joints") if "left_hand_joints" in action else (
        "hand_binary_action" if "hand_binary_action" in action else "hand_binary",
    )
    for key in hand_fields:
        width = 7 if key.endswith("joints") else 2
        values = np.asarray(api.get_action_field(action, key))
        if values.shape not in ((horizon, width), (1, horizon, width)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key}: expected finite [1,{horizon},{width}], got {values.shape}")
    frames = []
    for index, token in enumerate(tokens):
        signals = api._extract_hand_debug_signals(action, index, "dex3")
        raw_binary = signals["raw_hand_binary"]
        if raw_binary is not None and not np.isfinite(raw_binary).all():
            raise ValueError(f"Non-finite hand binary at frame {index}")
        left, right = signals["sent_left_hand_joints"], signals["sent_right_hand_joints"]
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Non-finite hand joints at frame {index}")
        frames.append((token, left, right))
    return frames


def camera_class() -> Any:
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    class FreshCamera(ComposedCameraClientSensor):
        def __init__(self, *args: Any, **kwargs: Any):
            self.last_received = 0.0
            super().__init__(*args, **kwargs)

        def receive_message_nonblocking(self, timeout_ms: int = 0) -> Any:
            message = super().receive_message_nonblocking(timeout_ms)
            if message is not None:
                self.last_received = time.monotonic()
            return message

    return FreshCamera


def view(config: dict[str, Any]) -> None:
    """Use the existing OpenCV viewer, with recording disabled and close detection."""
    import cv2
    from gear_sonic.scripts import run_camera_viewer as viewer

    FreshCamera = camera_class()
    startup = time.monotonic()

    class PreviewCamera(FreshCamera):
        def read(self, *args: Any, **kwargs: Any) -> Any:
            result = super().read(*args, **kwargs)
            timeout = config["data_timeout"] if self.last_received else min(9.0, config["startup_timeout"])
            if time.monotonic() - (self.last_received or startup) > timeout:
                raise TimeoutError("Third-person image stream stopped or never produced an image")
            return result

    class Display:
        window = None

        def __getattr__(self, name: str) -> Any:
            return getattr(cv2, name)

        def imshow(self, name: str, image: Any) -> None:
            cv2.imshow(name, image)
            if self.window is None:
                self.window = name
                event("ready")

        def waitKey(self, delay: int) -> int:
            key = cv2.waitKey(delay)
            closed = False
            if self.window is not None:
                try:
                    closed = cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1
                except cv2.error:
                    # Qt can destroy the last window before this property query.
                    closed = True
            if closed or key & 0xFF == ord("q"):
                event("user_stop")
                return ord("q")
            return -1 if key & 0xFF == ord("r") else key

    # Local module references only: no changes to GR00T files or global cv2.
    viewer.ComposedCameraClientSensor = PreviewCamera
    viewer.cv2 = Display()
    viewer.main(viewer.CameraViewerConfig(camera_host="127.0.0.1", camera_port=config["ports"]["video"], fps=VIDEO_FPS,
                                         output_path=config["log_dir"] + "/unused_recordings"))


def infer(config: dict[str, Any]) -> None:
    import zmq
    from gear_sonic.scripts import run_vla_inference as api

    commands = command_reader()
    camera = camera_class()(server_ip=config["camera_host"], port=config["camera_port"])
    state_sub = api.ZMQStateSubscriber(host="127.0.0.1", port=config["ports"]["state"])
    robot = api.instantiate_g1_robot_model(waist_location="lower_and_upper_body")
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.bind(f"tcp://127.0.0.1:{config['ports']['action']}")
    requests: queue.Queue[Any] = queue.Queue(maxsize=1)
    results: queue.Queue[Any] = queue.Queue(maxsize=1)
    errors: queue.Queue[BaseException] = queue.Queue()
    stop = threading.Event()
    busy = threading.Event()
    reachable = threading.Event()
    startup_deadline = time.monotonic() + config["startup_timeout"]

    def worker() -> None:
        # REQ socket belongs entirely to this thread; never share it with main.
        policy = api._LocalPolicyClient(host="127.0.0.1", port=config["ports"]["server"], timeout_ms=1000)
        try:
            while not stop.is_set() and not policy.ping():
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("Huge server did not become ready")
                stop.wait(0.2)
            policy.timeout_ms = int(config["inference_timeout"] * 1000)
            policy._init_socket()
            reachable.set()
            while not stop.is_set():
                try:
                    observation = requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                start = time.monotonic()
                try:
                    # Use the existing policy protocol and action transformation,
                    # but let exceptions propagate instead of the upstream catch-and-skip.
                    action, _ = policy.get_action(observation)
                    action.pop("task_progress", None)
                    action.pop("action.task_progress", None)
                    processed = api.concat_action(robot, action)
                    frames = action_frames(processed, api, config["action_horizon"])
                    results.put((frames, start))
                finally:
                    busy.clear()
        except BaseException as exc:
            errors.put(exc)
        finally:
            policy.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    state = None
    last_state = 0.0
    image = None
    phase = "waiting"
    inputs_announced = False
    initialize_at = 0.0
    command_at = 0.0
    cache = None
    index = 0
    frame_number = 0
    last_result = 0.0
    chunk_deadline = 0.0
    initial_token = np.asarray(api.LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32)
    zero_hand = np.zeros(7, dtype=np.float32)

    def publish(token: Any, left: Any, right: Any) -> None:
        nonlocal frame_number
        publisher.send(api.pack_latent_action_message(token, np.array([frame_number], dtype=np.int64), left, right))
        frame_number += 1

    def submit() -> None:
        observation = api.prepare_observation_from_sensors(
            Snapshot(image), Snapshot(copy.deepcopy(state)), robot, config["prompt"], hand_type="dex3")
        if observation is None:
            raise RuntimeError("Could not assemble real-image / simulated-state observation")
        busy.set()
        requests.put_nowait(observation)

    try:
        while True:
            tick = time.monotonic()
            if not errors.empty():
                raise RuntimeError(f"Policy worker failed: {errors.get_nowait()}")
            image = camera.read(blocking=False)
            new_state = state_sub.get_msg()
            if new_state is not None:
                validate_state(new_state)
                state, last_state = new_state, tick
            if camera.last_received and tick - camera.last_received > config["data_timeout"]:
                raise TimeoutError("Real camera stopped sending messages")
            if last_state and tick - last_state > config["data_timeout"]:
                raise TimeoutError("Simulated g1_debug stopped sending messages")
            if image is not None:
                rgb = np.asarray(image["images"]["ego_view"])
                if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
                    raise ValueError("Real camera ego_view must be a nonempty RGB image")
            if reachable.is_set() and image is not None and not inputs_announced:
                event("inputs_ready")
                inputs_announced = True
            while True:
                try:
                    command = commands.get_nowait()
                except queue.Empty:
                    break
                if command == "stop":
                    return
                if command == "initialize":
                    phase, initialize_at = "starting_planner", tick
                elif command == "run":
                    if cache is None:
                        raise RuntimeError("Run requested before successful policy warmup")
                    phase, index = "running", 0
                    last_result = tick
                    chunk_deadline = tick + config["action_horizon"] / config["action_publish_rate"]
                    event("running")
            if phase != "running" and tick > startup_deadline:
                raise TimeoutError(f"Client initialization timed out in phase {phase}")
            if phase in {"starting_planner", "initializing", "warming", "armed"}:
                publish(initial_token, zero_hand, zero_hand)
                if phase in {"starting_planner", "initializing"} and tick - command_at > 0.25:
                    # ZMQManager consumes start_control_ only in planner mode.
                    # Match the upstream k -> i sequence before streaming tokens.
                    publisher.send(api.build_command_message(start=True, stop=False, planner=phase == "starting_planner"))
                    command_at = tick
                if phase == "starting_planner" and last_state > initialize_at:
                    phase, initialize_at, command_at = "initializing", tick, 0.0
                if phase == "initializing" and last_state > initialize_at and tick - initialize_at > 1.0:
                    submit()
                    phase = "warming"
            try:
                frames, started = results.get_nowait()
            except queue.Empty:
                pass
            else:
                cache = frames
                last_result = time.monotonic()
                latency = last_result - started
                if phase == "warming":
                    # Startup is held by the band and initial pose. Begin the first
                    # valid chunk at frame zero when the parent releases the band.
                    phase = "armed"
                    event("policy_ready", latency=latency)
                else:
                    index = api.calculate_latency_compensated_index(latency, config["action_publish_rate"], config["action_horizon"])
                    chunk_deadline = started + config["action_horizon"] / config["action_publish_rate"]
                    if latency >= config["action_horizon"] / config["action_publish_rate"]:
                        raise TimeoutError(f"Returned action chunk already expired (inference {latency:.3f}s)")
            if phase == "running":
                now = time.monotonic()
                if index >= len(cache) or now >= chunk_deadline:
                    raise TimeoutError("Action chunk exhausted before replacement; last token will not be repeated")
                if api.should_trigger_new_inference(cache is not None, busy.is_set(), now - last_result, 1 / config["rate"]):
                    submit()
                publish(*cache[index])
                index += 1
            time.sleep(max(0.0, 1 / config["action_publish_rate"] - (time.monotonic() - tick)))
    finally:
        stop.set()
        publisher.send(api.build_command_message(start=False, stop=True, planner=False))
        thread.join(timeout=0.2)
        publisher.close(linger=0)
        context.term()
        camera.close()
        state_sub.close()


def run(config: dict[str, Any], viewer: bool) -> None:
    """Run the preview client, or the existing OpenCV viewer with --viewer."""
    if viewer:
        view(config)
    else:
        infer(config)


if __name__ == "__main__":
    worker_main(run)

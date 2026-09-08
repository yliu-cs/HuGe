"""Client-local timing hooks; network payloads and control scheduling stay upstream."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import queue
import socket
import sys
import threading
import time
from typing import Any, Iterator
import uuid


class TraceRecorder:
    """Nonblocking producer queue with a single file owner and explicit loss accounting."""

    def __init__(self, directory: Path, metadata: dict[str, Any], capacity: int = 4096):
        directory.mkdir(parents=True, exist_ok=False)
        self.session_id = uuid.uuid4().hex
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.dropped = 0
        self.failure: str | None = None
        self.closed = False
        self.warned = False
        self.metadata = {
            "schema_version": 1, "trace_session_id": self.session_id,
            "pid": os.getpid(), "hostname": socket.gethostname(),
            "clock": "CLOCK_MONOTONIC", "time_unit": "ns",
            "started_wall_ns": time.time_ns(), "started_mono_ns": time.monotonic_ns(),
            **metadata,
        }
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        self.stream = (directory / "events.jsonl").open("w", encoding="utf-8")
        self.thread = threading.Thread(target=self._write, name="vla-timing-writer", daemon=True)
        self.thread.start()

    def _lost(self, reason: str) -> None:
        with self.lock:
            self.dropped += 1
            warn = not self.warned
            self.warned = True
        if warn:
            print(f"VLA timing incomplete: {reason}; control behavior is unchanged.", file=sys.stderr)

    def emit(self, event: str, **fields: Any) -> None:
        if self.closed or self.failure is not None:
            self._lost(self.failure or "recorder closed")
            return
        record = {"event": event, "mono_ns": time.monotonic_ns(), **fields}
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self._lost("writer queue full")

    def _write(self) -> None:
        flushed = time.monotonic()
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=0.1)
                except queue.Empty:
                    record = None
                if record is not None:
                    self.stream.write(json.dumps(record, allow_nan=False) + "\n")
                if time.monotonic() - flushed >= 0.5:
                    self.stream.flush()
                    flushed = time.monotonic()
            self.stream.write(json.dumps({
                "event": "trace_end", "mono_ns": time.monotonic_ns(),
                "dropped_events": self.dropped, "complete": self.dropped == 0,
            }) + "\n")
            self.stream.flush()
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            self._lost(self.failure)
        finally:
            try:
                self.stream.close()
            except OSError:
                pass

    def close(self) -> None:
        self.closed = True
        self.stop.set()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            self._lost("writer did not finish before shutdown")


@dataclass(frozen=True)
class RequestTag:
    request_id: int


class TracedAction(dict):
    def __init__(self, action: dict[str, Any], tag: RequestTag):
        super().__init__(action)
        self.tag = tag


class TracedMessage(bytes):
    def __new__(cls, payload: bytes, fields: dict[str, Any]) -> TracedMessage:
        value = super().__new__(cls, payload)
        value.fields = fields
        return value


class Proxy:
    def __init__(self, wrapped: Any):
        self.wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


def validate_api(api: Any) -> type:
    """Inspect hooks without constructing clients, sockets or cameras."""
    policy = api._resolve_policy_client_class()
    for name in ("get_action", "call_endpoint", "_init_socket"):
        if not callable(getattr(policy, name, None)):
            raise RuntimeError(f"Unsupported policy client: missing {name}")
    for name in ("main", "concat_action", "run_policy_inference_and_process",
                 "prepare_observation_from_sensors", "_extract_hand_debug_signals",
                 "pack_latent_action_message", "build_command_message"):
        if not callable(getattr(api, name, None)):
            raise RuntimeError(f"Unsupported VLA module: missing {name}")
    return policy


@contextmanager
def timing_hooks(api: Any, recorder: TraceRecorder, policy_class: type) -> Iterator[None]:
    """Install hooks only on the imported VLA module; restore its references on exit."""
    originals: dict[str, Any] = {}
    local = threading.local()
    request_ids = itertools.count(1)
    publish_ids = itertools.count(1)
    camera_ids = itertools.count(1)

    def replace(name: str, value: Any) -> Any:
        original = getattr(api, name)
        originals[name] = original
        setattr(api, name, value)
        return original

    class RequestSocket(Proxy):
        def send(self, data: Any, *args: Any, **kwargs: Any) -> Any:
            tag = getattr(local, "request", None)
            begin = time.monotonic_ns()
            try:
                result = self.wrapped.send(data, *args, **kwargs)
            except Exception as exc:
                if tag is not None:
                    recorder.emit("request_send", request_id=tag.request_id,
                                  request_send_begin_ns=begin, request_send_done_ns=None,
                                  error=f"{type(exc).__name__}: {exc}")
                raise
            done = time.monotonic_ns()
            if tag is not None:
                recorder.emit("request_send", request_id=tag.request_id,
                              request_send_begin_ns=begin, request_send_done_ns=done)
            return result

        def recv(self, *args: Any, **kwargs: Any) -> Any:
            try:
                result = self.wrapped.recv(*args, **kwargs)
            except Exception as exc:
                tag = getattr(local, "request", None)
                if tag is not None:
                    recorder.emit("response_error", request_id=tag.request_id,
                                  error=f"{type(exc).__name__}: {exc}")
                raise
            done = time.monotonic_ns()
            tag = getattr(local, "request", None)
            if tag is not None:
                recorder.emit("response_recv", request_id=tag.request_id, response_recv_done_ns=done)
            return result

    class TimedPolicy(policy_class):
        def _init_socket(self) -> None:
            super()._init_socket()
            self.socket = RequestSocket(self.socket)

        def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> Any:
            tag = RequestTag(next(request_ids))
            local.request = tag
            local.last_request = tag
            camera = getattr(local, "observation", {})
            recorder.emit("request", request_id=tag.request_id, **camera)
            traced_options = dict(options or {})
            traced_options["huge_timing"] = {
                "trace_session_id": recorder.session_id, "request_id": tag.request_id,
            }
            try:
                action, info = super().get_action(observation, traced_options)
                return TracedAction(action, tag), info
            except Exception as exc:
                recorder.emit("request_error", request_id=tag.request_id,
                              error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                local.request = None

    class CameraSocket(Proxy):
        def recv(self, *args: Any, **kwargs: Any) -> Any:
            begin = time.monotonic_ns()
            result = self.wrapped.recv(*args, **kwargs)
            done = time.monotonic_ns()
            self.latest = {"camera_message_seq": next(camera_ids),
                           "camera_read_begin_ns": begin, "camera_read_done_ns": done}
            return result

    class TimedCamera(api.ComposedCameraClientSensor):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.socket = CameraSocket(self.socket)

        def read(self, *args: Any, **kwargs: Any) -> Any:
            message = super().read(*args, **kwargs)
            local.camera = dict(getattr(self.socket, "latest", {})) if message is not None else {}
            if message is not None:
                stamp = message.get("timestamps", {}).get("ego_view")
                local.camera["camera_source_timestamp"] = float(stamp) if stamp is not None else None
            return message

    original_prepare = api.prepare_observation_from_sensors

    def prepare(*args: Any, **kwargs: Any) -> Any:
        local.camera = {}
        local.observation = {}
        observation = original_prepare(*args, **kwargs)
        if observation is not None:
            local.observation = {**local.camera, "observation_ready_ns": time.monotonic_ns()}
            audit = getattr(observation, "hand_state_compat", None)
            if audit is not None:
                local.observation["hand_state_compat"] = audit
        return observation

    original_concat = api.concat_action

    def concat(robot: Any, action: Any) -> Any:
        result = original_concat(robot, action)
        return TracedAction(result, action.tag) if isinstance(action, TracedAction) else result

    original_infer = api.run_policy_inference_and_process

    def infer(*args: Any, **kwargs: Any) -> Any:
        local.last_request = None
        try:
            result = original_infer(*args, **kwargs)
        except Exception as exc:
            tag = local.last_request
            recorder.emit("result_error", request_id=tag.request_id if tag else None,
                          error=f"{type(exc).__name__}: {exc}")
            raise
        ready = time.monotonic_ns()
        if isinstance(result, TracedAction):
            token = api.get_action_field(result, "motion_token")
            horizon = int(token.shape[-2]) if token.ndim >= 2 else 1
            recorder.emit("result_ready", request_id=result.tag.request_id,
                          result_ready_ns=ready, horizon=horizon)
        elif local.last_request is not None:
            recorder.emit("result_unavailable", request_id=local.last_request.request_id)
        return result

    original_extract = api._extract_hand_debug_signals

    def extract(processed_action: Any, current_idx: int, hand_type: str) -> Any:
        local.pending_action = None
        result = original_extract(processed_action, current_idx, hand_type)
        tag = processed_action.tag if isinstance(processed_action, TracedAction) else None
        local.pending_action = {"source": "model" if tag else "unattributed",
                                "request_id": tag.request_id if tag else None,
                                "chunk_index": int(current_idx)}
        return result

    original_pack = api.pack_latent_action_message

    def pack(motion_token: Any, frame_index: Any, *args: Any, **kwargs: Any) -> bytes:
        fields = getattr(local, "pending_action", None)
        local.pending_action = None
        initial = getattr(local, "initial_pose_pending", False)
        local.initial_pose_pending = False
        payload = original_pack(motion_token, frame_index, *args, **kwargs)
        if fields is None:
            fields = {"source": "initial_pose" if initial else "unattributed",
                      "request_id": None, "chunk_index": None}
        return TracedMessage(payload, {**fields, "frame_index": int(frame_index.reshape(-1)[0])})

    original_command = api.build_command_message

    def command(start: bool, stop: bool, planner: bool, delta_heading: float | None = None) -> bytes:
        payload = original_command(start, stop, planner, delta_heading)
        return TracedMessage(payload, {"source": "control_command", "start": bool(start),
                                       "stop": bool(stop), "planner": bool(planner)})

    class ActionSocket(Proxy):
        def send(self, data: Any, *args: Any, **kwargs: Any) -> Any:
            fields = data.fields if isinstance(data, TracedMessage) else {"source": "unattributed"}
            payload = bytes(data) if isinstance(data, TracedMessage) else data
            seq = next(publish_ids)
            begin = time.monotonic_ns()
            try:
                result = self.wrapped.send(payload, *args, **kwargs)
            except Exception as exc:
                recorder.emit("action_send", **fields, publish_seq=seq,
                              action_send_begin_ns=begin, action_send_done_ns=None,
                              error=f"{type(exc).__name__}: {exc}")
                raise
            done = time.monotonic_ns()
            recorder.emit("action_send", **fields, publish_seq=seq,
                          action_send_begin_ns=begin, action_send_done_ns=done)
            return result

    original_zmq = api.zmq

    class ActionContext(Proxy):
        def socket(self, socket_type: Any, *args: Any, **kwargs: Any) -> Any:
            result = self.wrapped.socket(socket_type, *args, **kwargs)
            return ActionSocket(result) if socket_type == original_zmq.PUB else result

    class ZmqFacade(Proxy):
        def Context(self, *args: Any, **kwargs: Any) -> Any:
            return ActionContext(original_zmq.Context(*args, **kwargs))

    class TimedKeyboard(api.ZMQKeyboardSubscriber):
        def read_msg(self, *args: Any, **kwargs: Any) -> Any:
            key = super().read_msg(*args, **kwargs)
            if key is not None:
                local.initial_pose_pending = key == "i"
                recorder.emit("keyboard", key=key)
            return key

    try:
        replace("_resolve_policy_client_class", lambda: TimedPolicy)
        replace("ComposedCameraClientSensor", TimedCamera)
        replace("prepare_observation_from_sensors", prepare)
        replace("concat_action", concat)
        replace("run_policy_inference_and_process", infer)
        replace("_extract_hand_debug_signals", extract)
        replace("pack_latent_action_message", pack)
        replace("build_command_message", command)
        replace("zmq", ZmqFacade(original_zmq))
        replace("ZMQKeyboardSubscriber", TimedKeyboard)
        yield
    finally:
        for name, original in originals.items():
            setattr(api, name, original)

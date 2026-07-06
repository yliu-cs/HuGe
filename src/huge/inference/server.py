"""Minimal msgpack-numpy ZeroMQ PolicyServer compatible with GR00T clients."""

from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy as m
import numpy as np
import zmq

from huge.data.schemas import ACTION_HORIZON, ACTION_DIM, ActionSchema, combine_action_dict
from huge.data.state_normalizer import StateNormalizer
from huge.inference.deploy_logger import DeployCallLogger
from huge.inference.policy import HugePolicy
from huge.inference.sonic_adapter import SonicAdapter
from huge.models.qwen_sonicfm import QwenSonicFM

m.patch()


def _unpack(payload: bytes) -> dict[str, Any]:
    return m.unpackb(payload, raw=False)


def _pack(obj: Any) -> bytes:
    return m.packb(obj, use_bin_type=True)


class G1StateTap:
    """Background SUB tap for the C++ deploy debug stream."""

    TOPIC = "g1_debug"

    def __init__(self, *, host: str = "localhost", port: int = 5557, poll_hz: float = 200.0) -> None:
        self._period = 1.0 / float(poll_hz)
        self._stop = threading.Event()
        self._latest_quat: np.ndarray | None = None
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self.TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(f"tcp://{host}:{port}")
        self._thread = threading.Thread(target=self._run, daemon=True, name="huge-g1-state-tap")
        self._thread.start()

    def _run(self) -> None:
        topic_len = len(self.TOPIC)
        while not self._stop.is_set():
            try:
                raw = self._socket.recv(zmq.NOBLOCK)
                msg = msgpack.unpackb(raw[topic_len:], raw=False)
                quat = msg.get("base_quat")
                if quat is not None:
                    arr = np.asarray(quat, dtype=np.float32).reshape(-1)
                    if arr.size == 4:
                        self._latest_quat = arr.copy()
            except zmq.error.Again:
                time.sleep(self._period)
            except Exception:
                time.sleep(self._period)

    def latest_quat(self) -> np.ndarray | None:
        return None if self._latest_quat is None else self._latest_quat.copy()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._socket.close(linger=0)
        self._ctx.term()


class ClosedLoopSonicPublisher:
    """Publishes cached model chunks as SONIC latent v4 pose messages."""

    def __init__(
        self,
        *,
        bind: str,
        control_hz: float = 50.0,
        latency_skip: int = 1,
        adapter: SonicAdapter | None = None,
    ) -> None:
        self.bind = bind
        self.period = 1.0 / float(control_hz)
        self.latency_skip = max(0, int(latency_skip))
        self.adapter = adapter or SonicAdapter()
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(bind)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._chunk = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
        self._chunk_start_time = time.monotonic()
        self._last_valid = self._chunk[0].copy()
        self._thread = threading.Thread(target=self._run, daemon=True, name="huge-sonic-publisher")
        self._thread.start()

    def reset(self) -> None:
        with self._lock:
            self.adapter.reset()
            self._chunk[:] = 0.0
            self._last_valid[:] = 0.0
            self._chunk_start_time = time.monotonic()

    def set_action_dict(self, action: dict[str, np.ndarray]) -> None:
        chunk = combine_action_dict(action)
        if chunk.ndim == 3:
            chunk = chunk[0]
        if chunk.shape != (ACTION_HORIZON, ACTION_DIM):
            padded = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
            n = min(ACTION_HORIZON, chunk.shape[0])
            padded[:n] = chunk[:n]
            if n > 0:
                padded[n:] = chunk[n - 1]
            chunk = padded
        with self._lock:
            self._chunk = np.asarray(chunk, dtype=np.float32).copy()
            self._chunk_start_time = time.monotonic()

    def _select_action(self) -> np.ndarray:
        with self._lock:
            elapsed_steps = int((time.monotonic() - self._chunk_start_time) / self.period)
            idx = min(max(elapsed_steps + self.latency_skip, 0), ACTION_HORIZON - 1)
            action = self._chunk[idx].copy()
        try:
            self.adapter.validate_motion_token(action[:64])
            self._last_valid = action.copy()
            return action
        except Exception:
            return self._last_valid.copy()

    def _run(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            action = self._select_action()
            msg = self.adapter.to_protocol_v4(action.reshape(1, ACTION_DIM), 0)
            topic = str(msg.get("topic", "pose")).encode()
            self._socket.send(topic + _pack(msg))
            next_t += self.period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)
        self._socket.close(linger=0)
        self._ctx.term()


class HugePolicyServer:
    def __init__(
        self,
        policy: HugePolicy,
        *,
        host: str = "0.0.0.0",
        port: int = 5550,
        log_dir: str | None = None,
        publisher: ClosedLoopSonicPublisher | None = None,
        state_tap: G1StateTap | None = None,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = int(port)
        self._running = False
        self.log_dir = Path(log_dir) if log_dir else None
        self.logger = DeployCallLogger(self.log_dir) if self.log_dir else None
        if self.log_dir is not None:
            print(f"Huge deploy logger writing to {self.log_dir}")
        self.call_idx = 0
        self.publisher = publisher
        self.state_tap = state_tap

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        endpoint = request.get("endpoint") or request.get("cmd") or request.get("type")
        groot_style = "data" in request
        if endpoint == "ping":
            return {"status": "ok", "message": "Huge server is running"} if groot_style else {"ok": True, "endpoint": "ping"}
        if endpoint == "reset":
            if self.publisher is not None:
                self.publisher.reset()
            return {} if groot_style else {"ok": True, "endpoint": "reset"}
        if endpoint == "get_modality_config":
            return {} if groot_style else {"ok": True, "action_schema": ActionSchema().to_dict()}
        if endpoint == "get_action":
            data = request.get("data", {})
            observation = data.get("observation", request.get("observation", request.get("data", {}))) if isinstance(data, dict) else {}
            self._merge_state_tap(observation)
            action, trace = self.policy.get_action_with_trace(observation)
            if self.publisher is not None:
                self.publisher.set_action_dict(action)
            self._log_call(request, observation, action, trace)
            return (action, {}) if groot_style else {"ok": True, "action": action}
        if endpoint == "kill":
            self._running = False
            return {"status": "shutting down"} if groot_style else {"ok": True, "endpoint": "kill"}
        return {"ok": False, "error": f"unknown endpoint {endpoint!r}"}

    def _merge_state_tap(self, observation: dict[str, Any]) -> None:
        if self.state_tap is None:
            return
        quat = self.state_tap.latest_quat()
        if quat is None:
            return
        state = observation.setdefault("state", {})
        if not isinstance(state, dict):
            return
        # The C++ deploy stream is the authoritative base-orientation source, so
        # write it onto the nested path assemble_proprio actually reads and drop any
        # client-supplied projected_gravity that would otherwise shadow it.
        state["base_quat"] = quat
        state.pop("projected_gravity", None)
        observation.pop("base_quat", None)
        observation.pop("state.base_quat", None)
        observation.pop("state.projected_gravity", None)

    def _log_call(self, request: dict[str, Any], observation: dict[str, Any], action: dict[str, Any], trace: Any) -> None:
        if self.logger is None:
            return
        call_idx = self.call_idx
        self.call_idx += 1
        try:
            record = {
                "time": time.time(),
                "call_idx": call_idx,
                "request": copy.deepcopy(request),
                "observation": copy.deepcopy(observation),
                "model_inputs": {
                    "prompt": trace.prompt,
                    "input_ids": trace.input_ids,
                    "image": trace.image,
                    "state_raw": trace.state_raw,
                    "state": trace.state,
                    "state_mask": trace.state_mask,
                    "qwen_inputs": trace.qwen_inputs,
                    "action_token_mask": getattr(trace, "action_token_mask", None),
                    "qwen_text": trace.qwen_text,
                },
                "model_output": {
                    "action": action,
                    "action_array": trace.action_array,
                },
            }
            self.logger.log(call_idx=call_idx, record=record)
        except Exception as exc:
            print(f"Huge deploy logging failed for call {call_idx}: {exc!r}")

    def serve_forever(self) -> None:
        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.host}:{self.port}")
        self._running = True
        print(f"HugePolicyServer listening on tcp://{self.host}:{self.port}")
        while self._running:
            payload = socket.recv()
            try:
                response = self.handle(_unpack(payload))
            except Exception as exc:  # pragma: no cover - server boundary.
                response = {"ok": False, "error": repr(exc)}
            socket.send(_pack(response))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--action-horizon", type=int, default=ACTION_HORIZON)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--proprio-stats-path", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--closed-loop-enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sonic-pub-bind", default="tcp://*:5556")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--latency-skip", type=int, default=1)
    parser.add_argument("--state-tap-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--state-tap-host", default="localhost")
    parser.add_argument("--state-tap-port", type=int, default=5557)
    args = parser.parse_args(argv)

    model = QwenSonicFM.from_checkpoint(args.ckpt, map_location=args.device) if args.ckpt else QwenSonicFM()
    model.config.action_horizon = args.action_horizon
    normalizer = _load_normalizer(args.ckpt, args.proprio_stats_path)
    publisher = None
    if args.closed_loop_enable:
        publisher = ClosedLoopSonicPublisher(
            bind=args.sonic_pub_bind,
            control_hz=args.control_hz,
            latency_skip=args.latency_skip,
        )
        print(f"Huge SONIC v4 publisher bound on {args.sonic_pub_bind}")
    state_tap = None
    if args.state_tap_enable:
        try:
            state_tap = G1StateTap(host=args.state_tap_host, port=args.state_tap_port)
            print(f"Huge state tap connected to tcp://{args.state_tap_host}:{args.state_tap_port}")
        except Exception as exc:
            print(f"Huge state tap disabled: {exc!r}")
    server = HugePolicyServer(
        HugePolicy(model, device=args.device, normalizer=normalizer),
        host=args.host,
        port=args.port,
        log_dir=args.log_dir,
        publisher=publisher,
        state_tap=state_tap,
    )
    try:
        server.serve_forever()
    finally:
        if publisher is not None:
            publisher.close()
        if state_tap is not None:
            state_tap.close()


def _load_normalizer(ckpt: str | None, override: str | None) -> StateNormalizer | None:
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    if ckpt:
        ckpt_path = Path(ckpt)
        ckpt_dir = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
        candidates.append(ckpt_dir / "proprio_stats_sonic.json")
        config_path = ckpt_dir / "config.json"
        if config_path.exists():
            try:
                meta = json.loads(config_path.read_text(encoding="utf-8")).get("metadata", {})
                if meta.get("proprio_stats_path"):
                    candidates.append(Path(meta["proprio_stats_path"]))
            except Exception:
                pass
    for path in candidates:
        if path.exists():
            return StateNormalizer.from_json(path)
    return None


if __name__ == "__main__":
    main()

"""WebSocket server that runs a Kappa post-train RTC control loop.

Adapted from ``Psi0/src/psi/deploy/psi_serve_rtc-trainingtimertc.py``. A
single Kappa MoT model is loaded once on a chosen device. Clients connect
via WebSocket and send periodic observation packets:

    {
        "image":   <base64 PNG bytes>,
        "text":    "perform the task",
        "proprio": <list of 64 floats>,
    }

The server runs an internal control loop at ``ctrl_hz`` (default 30 Hz) and
streams the most-recent denormalized G1 action as JSON:

    {"version": <int>, "action": [31 floats]}

Run via:

    .venv/bin/python -m src.inference.kappa_serve_rtc \\
        --weights .runs/kappa/<post-train>/checkpoints/ckpt_<step>/model.safetensors \\
        --backbone resource/model/Qwen3-VL-2B-Instruct \\
        --action-stats resource/dataset/G1_WBT-cache/meta/g1_action_stats.json
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import threading
import time
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


def _load_model(args, device: torch.device):
    """Build a fresh ``KappaModelWrapper``, convert to MoT with G1 + proprio,
    and load post-train weights via ``safetensors``.
    """
    from src.config.model_kappa import KappaModelConfig
    from src.models.kappa_model import KappaModelWrapper
    from safetensors.torch import load_file

    cfg = KappaModelConfig(
        model_name_or_path=args.backbone,
        use_g1_expert=True,
        use_proprio=args.use_proprio,
        use_rtc=True,
        rtc_on_sonic=args.rtc_on_sonic,
    )
    wrapper = KappaModelWrapper(
        model_name_or_path=cfg.model_name_or_path,
        tune_vision=False,
        tune_llm=False,
        tune_mlp=False,
        device=device,
    )
    wrapper.convert_to_mot(
        sonic_dim=cfg.sonic_dim,
        action_loss_weight=cfg.action_loss_weight,
        use_g1_expert=True,
        g1_dim=cfg.g1_dim,
        g1_loss_weight=cfg.g1_loss_weight,
        use_proprio=args.use_proprio,
        proprio_dim=cfg.proprio_dim,
        use_rtc=True,
        rtc_on_sonic=args.rtc_on_sonic,
    )
    state_dict = load_file(args.weights)
    result = wrapper.model.load_state_dict(state_dict, strict=False)
    print(
        f"[serve] loaded {len(state_dict)} tensors; missing={len(result.missing_keys)} "
        f"unexpected={len(result.unexpected_keys)}"
    )
    wrapper.model.eval()
    return wrapper, cfg


def _load_json(path: Optional[str]) -> Optional[dict]:
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _decode_image(payload: dict[str, Any]) -> Image.Image:
    raw = payload["image"]
    if isinstance(raw, str):
        data = base64.b64decode(raw)
    elif isinstance(raw, (bytes, bytearray)):
        data = bytes(raw)
    else:
        raise ValueError("image must be base64 str or bytes")
    return Image.open(io.BytesIO(data)).convert("RGB")


class Server:
    """Single-client RTC server. Keeps one ``RealTimeChunkController`` alive
    once obs is first received; control loop fires at fixed rate regardless
    of inference latency.
    """

    def __init__(
        self,
        controller_factory,
        ctrl_hz: float = 30.0,
    ):
        self.controller_factory = controller_factory
        self.ctrl_period = 1.0 / float(ctrl_hz)
        self.controller = None

        self._latest_obs: dict[str, Any] | None = None
        self._latest_action: np.ndarray | None = None
        self._action_version: int = 0
        self._action_lock = threading.Lock()
        self._obs_lock = threading.Lock()
        self._action_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ctrl_thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # FastAPI / WebSocket handlers
    # ------------------------------------------------------------------

    def fastapi_app(self):
        from fastapi import FastAPI, WebSocket

        app = FastAPI()

        @app.websocket("/predict")
        async def predict(ws: WebSocket):  # noqa: D401
            await ws.accept()
            self._loop = asyncio.get_running_loop()
            self._action_event = asyncio.Event()

            recv_task = asyncio.create_task(self._recv_loop(ws))
            send_task = asyncio.create_task(self._send_loop(ws))
            try:
                await asyncio.gather(recv_task, send_task)
            except Exception as e:  # noqa: BLE001
                print(f"[serve] websocket closed: {e!r}")
            finally:
                self._stop.set()
                if self.controller is not None:
                    self.controller.shutdown()

        return app

    async def _recv_loop(self, ws):
        while True:
            text = await ws.receive_text()
            try:
                payload = json.loads(text)
            except Exception as e:  # noqa: BLE001
                print(f"[serve] bad obs payload: {e!r}")
                continue
            obs = self._parse_obs(payload)
            with self._obs_lock:
                self._latest_obs = obs
            if self.controller is None:
                self._start_control_loop(obs)

    async def _send_loop(self, ws):
        last_version = -1
        while True:
            assert self._action_event is not None
            await self._action_event.wait()
            self._action_event.clear()
            with self._action_lock:
                if self._latest_action is None:
                    continue
                if self._action_version == last_version:
                    continue
                version = self._action_version
                action = self._latest_action.tolist()
            await ws.send_text(json.dumps({"version": version, "action": action}))
            last_version = version

    # ------------------------------------------------------------------
    # Control thread
    # ------------------------------------------------------------------

    def _start_control_loop(self, first_obs: dict[str, Any]) -> None:
        if self.controller is not None:
            return
        print("[serve] starting controller (warm-up + inference thread)")
        self.controller = self.controller_factory()
        self.controller.warm_start(first_obs)
        self._ctrl_thread = threading.Thread(target=self._ctrl_loop, daemon=True)
        self._ctrl_thread.start()

    def _ctrl_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop.is_set():
            with self._obs_lock:
                obs = self._latest_obs
            if obs is None:
                time.sleep(self.ctrl_period)
                continue
            try:
                action = self.controller.step(obs)
            except Exception as e:  # noqa: BLE001
                print(f"[serve] controller.step error: {e!r}")
                time.sleep(self.ctrl_period)
                continue
            with self._action_lock:
                self._latest_action = action
                self._action_version += 1
            if self._action_event is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(self._action_event.set)
            next_tick += self.ctrl_period
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Missed tick — re-anchor instead of accumulating debt.
                next_tick = time.perf_counter()

    # ------------------------------------------------------------------
    # Payload parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_obs(payload: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["image"] = _decode_image(payload)
        out["text"] = str(payload.get("text", "perform the task"))
        if "proprio" in payload:
            out["proprio"] = np.asarray(payload["proprio"], dtype=np.float32)
        return out


def _build_controller(args, wrapper, cfg, device):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from src.inference.kappa_rtc_controller import RealTimeChunkController

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=getattr(cfg, "num_train_timesteps", 1000),
    )
    g1_action_stats = _load_json(args.action_stats)

    def factory():
        return RealTimeChunkController(
            model=wrapper.model,
            processor=wrapper.processor,
            scheduler=scheduler,
            cfg=cfg,
            g1_action_stats=g1_action_stats,
            ctrl_hz=args.ctrl_hz,
            s_min=args.s_min,
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help="Post-train safetensors checkpoint.")
    parser.add_argument("--backbone", default="resource/model/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--action-stats", default="resource/dataset/G1_WBT-cache/meta/g1_action_stats.json",
    )
    parser.add_argument("--use-proprio", action="store_true", default=True)
    parser.add_argument("--no-proprio", dest="use_proprio", action="store_false")
    parser.add_argument("--rtc-on-sonic", action="store_true", default=True)
    parser.add_argument("--no-rtc-on-sonic", dest="rtc_on_sonic", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ctrl-hz", type=float, default=30.0)
    parser.add_argument("--s-min", type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    wrapper, cfg = _load_model(args, device)
    factory = _build_controller(args, wrapper, cfg, device)
    server = Server(factory, ctrl_hz=args.ctrl_hz)

    import uvicorn
    app = server.fastapi_app()
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=2**24)


if __name__ == "__main__":
    main()

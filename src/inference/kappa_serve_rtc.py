"""WebSocket server that runs a Kappa RTC control loop emitting SONIC tokens.

Adapted from ``Psi0/src/psi/deploy/psi_serve_rtc-trainingtimertc.py``. A
single Kappa MoT model is loaded once on a chosen device. Clients connect
via WebSocket and send periodic observation packets:

    {
        "image":   <base64 PNG bytes>,
        "text":    "perform the task",
        "proprio": <list of 64 floats — RAW, server normalizes>,
    }

Proprio layout matches the training cache (``calc_sonic_tokens.py``):

    proprio[ 0:29] = jp        (rad, MuJoCo joint order)
    proprio[29:31] = hand_bits (binary 0/1; zeroed for fake-hand datasets)
    proprio[31:60] = jv        (rad/s, finite-diff of jp at 50 Hz)
    proprio[60:64] = quat_xyzw (pelvis orientation)

The server normalizes proprio internally using the cached ``q01``/``q99``
bounds (``--proprio-stats``) — clients send raw values.

The control loop runs at ``ctrl_hz`` (default 50 Hz, matching the deploy
SONIC decoder's ``control_dt_ = 0.02 s``) and streams the most-recent SONIC
token vector as JSON:

    {"version": <int>, "sonic": [66 floats]}

The downstream client runs the SONIC ONNX decoder
(``resource/model/sonic/model_decoder.onnx``) with the live robot state
history to recover the 29-D IsaacLab action, then reorders to MuJoCo joint
order and feeds the PD controller. The 2 trailing hand bits go to a
separate gripper controller.

Run via:

    .venv/bin/python src/inference/kappa_serve_rtc.py \\
        --weights .runs/kappa/<finetune>/checkpoints/ckpt_<step>/model.safetensors \\
        --backbone resource/model/Qwen3-VL-2B-Instruct
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
from fastapi import FastAPI, WebSocket
from PIL import Image
from torchvision.transforms import v2


# Mirror src/data/transform.py:DefaultImageAugment (Resize short=224 ->
# CenterCrop(224, 320)). Training applied this in PIL space before the
# Qwen3-VL processor, so the model only ever saw 224x320 frames. Deploy
# cameras publish 480x640 raw — feeding that straight to Qwen yields a
# different image_grid_thw than training and collapses SONIC velocity to
# near-zero, so the robot stops moving. Apply the same transform here.
_TRAIN_IMG_TRANSFORM = v2.Compose([
    v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop((224, 320)),
])


def _load_model(args, device: torch.device):
    """Build a fresh ``KappaModelWrapper``, convert to MoT with proprio + RTC,
    and load Post-Train / Fine-Tune weights via ``safetensors``.
    """
    from src.config.model_kappa import KappaModelConfig
    from src.models.kappa_model import KappaModelWrapper
    from safetensors.torch import load_file

    cfg = KappaModelConfig(
        model_name_or_path=args.backbone,
        use_proprio=args.use_proprio,
        use_rtc=True,
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
        use_proprio=args.use_proprio,
        proprio_dim=cfg.proprio_dim,
        use_rtc=True,
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


def _load_proprio_bounds(
    stats_path: str, mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read q01/q99 (or min/max) for the 64-D proprio. Mirrors
    :meth:`src.eval.kappa_policy.KappaPolicy._load_stats`.
    """
    with open(stats_path) as f:
        pstats = json.load(f).get("proprio", {})
    if mode == "bounds":
        lo_key, hi_key = "min", "max"
    elif mode == "bounds_q99":
        lo_key, hi_key = "q01", "q99"
    else:
        raise ValueError(f"unknown normalize_mode {mode!r}")
    lo = np.asarray(pstats[lo_key], dtype=np.float32)
    hi = np.asarray(pstats[hi_key], dtype=np.float32)
    ill = np.abs(hi - lo) < 1e-4 * (np.abs(hi) + np.abs(lo) + 1e-8)
    hi = np.where(ill, lo + 1.0, hi)
    return lo, hi


def _decode_image(payload: dict[str, Any]) -> Image.Image:
    raw = payload["image"]
    if isinstance(raw, str):
        data = base64.b64decode(raw)
    elif isinstance(raw, (bytes, bytearray)):
        data = bytes(raw)
    else:
        raise ValueError("image must be base64 str or bytes")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return _TRAIN_IMG_TRANSFORM(img)


class Server:
    """Single-client RTC server. Keeps one ``RealTimeChunkController`` alive
    once obs is first received; control loop fires at fixed rate regardless
    of inference latency.
    """

    def __init__(
        self,
        controller_factory,
        ctrl_hz: float = 50.0,
    ):
        self.controller_factory = controller_factory
        self.ctrl_period = 1.0 / float(ctrl_hz)
        self.controller = None
        self._controller_starting: bool = False

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
            if self.controller is None and not self._controller_starting:
                # Warm-up runs 3 full forward passes (cuDNN autotune + ORT
                # init); on a fresh CUDA context this takes 20-60s and would
                # block the event loop here, freezing WS ping/pong and
                # causing the client to drop with 1006 ABNORMAL_CLOSURE.
                # Off-load to a worker thread so the loop stays responsive.
                self._controller_starting = True
                asyncio.create_task(self._warm_start_async(obs))

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
            await ws.send_text(json.dumps({"version": version, "sonic": action}))
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
        print("[serve] controller ready")

    async def _warm_start_async(self, obs: dict[str, Any]) -> None:
        """Run the blocking ``_start_control_loop`` in a worker thread so the
        event loop can keep handling WS ping/pong while warm-up is running.
        """
        try:
            await asyncio.to_thread(self._start_control_loop, obs)
        except Exception as e:  # noqa: BLE001
            print(f"[serve] controller warm-up failed: {e!r}")
            # Reset flag so the next obs can re-trigger warm-up.
            self._controller_starting = False

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

    if args.use_proprio:
        proprio_min, proprio_max = _load_proprio_bounds(
            args.proprio_stats, args.normalize_mode,
        )
        print(
            f"[serve] loaded proprio bounds ({args.normalize_mode}) from "
            f"{args.proprio_stats}"
        )
    else:
        proprio_min, proprio_max = None, None

    def factory():
        return RealTimeChunkController(
            model=wrapper.model,
            processor=wrapper.processor,
            scheduler=scheduler,
            cfg=cfg,
            ctrl_hz=args.ctrl_hz,
            s_min=args.s_min,
            proprio_min=proprio_min,
            proprio_max=proprio_max,
        )

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help="Post-Train / Fine-Tune safetensors checkpoint.")
    parser.add_argument("--backbone", default="resource/model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--use-proprio", action="store_true", default=True)
    parser.add_argument("--no-proprio", dest="use_proprio", action="store_false")
    parser.add_argument(
        "--proprio-stats",
        default="resource/dataset/G1_WBT-cache/meta/g1_proprio_stats.json",
        help="Path to the q01/q99 stats JSON used at training time. Required "
             "when --use-proprio (default) is on; the server uses it to map "
             "raw client proprio into the [-1, 1] band the model was trained on.",
    )
    parser.add_argument(
        "--normalize-mode",
        default="bounds_q99",
        choices=["bounds_q99", "bounds"],
        help="Must match the training-time data.normalize_mode. bounds_q99 "
             "reads q01/q99 (Kappa default); bounds reads min/max.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--ctrl-hz", type=float, default=50.0,
        help="Control-loop rate. Defaults to 50 Hz to match the deploy SONIC "
             "decoder's training control_dt=0.02s and the 50 Hz token timeline "
             "the model was trained on.",
    )
    parser.add_argument(
        "--s-min", type=int, default=8,
        help="Re-query the policy every this many control steps. Default 8 "
             "= KappaModelConfig.sonic_max_delay so the RTC prefix-substitution "
             "delay stays inside the training distribution (trainer samples "
             "delay in [0, sonic_max_delay)). At ctrl_hz=50 this is 6.25Hz "
             "inference; raise only if you also retrain with a larger "
             "sonic_max_delay.",
    )
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

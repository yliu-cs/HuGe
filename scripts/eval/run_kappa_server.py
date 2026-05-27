"""Kappa policy ZMQ server — Terminal 1 for the GR00T-WholeBodyControl deploy.

Drops in for ``Isaac-GR00T/gr00t/eval/run_gr00t_server.py``. Speaks the same
msgpack-numpy REQ/REP wire format on the same default port (5550), so the
existing ``gear_sonic_deploy`` Terminals 2/3/4 work unchanged.

Endpoints
---------
``ping``                  → ``{"status": "ok", ...}``
``get_modality_config``   → empty dict (Terminal 3 / _LocalPolicyClient never queries it)
``reset``                 → empty dict
``kill``                  → shut server down
``get_action``            → ``[action_dict, info_dict]`` where action_dict has::

    motion_token        np.float32 (1, T, 64)
    hand_binary_action  np.float32 (1, T, 2)

matching the keys Terminal 3 unpacks via ``_extract_hand_joint_actions``
(see ``GR00T-WholeBodyControl/gear_sonic/scripts/run_vla_inference.py``).

Usage
-----
    cd /home/kinetra-5090-01/code/liuyang/Kappa
    source .venv/bin/activate
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    python scripts/eval/run_kappa_server.py \\
        --ckpt-path .runs/<run>/checkpoints/ckpt_<step> \\
        --base-vlm-path resource/model/Qwen3-VL-2B-Instruct \\
        --port 5550 \\
        --action-horizon 50 \\
        --num-inference-steps 4

Notes
-----
* The wire protocol (``MsgSerializer`` + REQ/REP loop) is vendored here so the
  server has no runtime dependency on the Isaac-GR00T package — only the
  Kappa venv is required. Matches ``Isaac-GR00T/gr00t/policy/server_client.py``
  byte-for-byte for non-``ModalityConfig`` payloads.
* SONIC sampler: noise→clean Euler integration over ``num_inference_steps``
  steps, following the flow-matching schedule used at training time
  (``noisy = t * noise + (1 - t) * clean``; ``v = noise - clean``;
  ``x_{t - dt} = x_t - dt * v``).
* When the checkpoint was trained with ``use_proprio=True``, the server
  assembles the full 64-D proprio layout (``jp + hand_bits + jv + quat_xyzw``)
  from Terminal 3's observation plus an *auxiliary ZMQ tap* on the C++
  deploy's ``g1_debug`` topic (default ``tcp://localhost:5557``):

    - ``jp``        — ``state.{left_leg, right_leg, waist, left_arm, right_arm}``
                      from Terminal 3 (29 dims, MuJoCo body order).
    - ``hand_bits`` — ``state.hand_binary_state`` from Terminal 3 (2 dims).
    - ``jv``        — finite-difference of consecutive ``body_q`` frames from
                      the state-tap, sampled at the C++ control rate (50 Hz
                      typical), matching training.
    - ``quat_xyzw`` — ``state.base_quat`` if Terminal 3 forwards it (preferred),
                      otherwise the state-tap's ``base_quat``. Both arrive in
                      WXYZ order; the server reorders to XYZW for the proprio
                      layout.

  If the state-tap can't connect, ``jv`` and ``quat`` degrade to zeros (the
  proprio_stats normalization then maps them to the middle of the training
  range). Set ``--no-deploy-proprio`` to bypass the proprio branch entirely
  even on a proprio-trained checkpoint, or ``--no-state-tap-enable`` to keep
  proprio on but skip the tap (jp+hand only, jv/quat zeroed).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Callable

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import tyro
import zmq

# Ensure the repo root is importable (so ``src.*`` resolves when this script
# is launched directly from anywhere). The script lives at
# ``scripts/eval/run_kappa_server.py`` — go up two levels.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.tokenized_video_dataset import DEFAULT_FRAME_SIZE


# =============================================================================
# Wire protocol (mirrors Isaac-GR00T/gr00t/policy/server_client.py)
# =============================================================================


class MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return mnp.packb(data, default=mnp.encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return mnp.unpackb(data, object_hook=mnp.decode, raw=False)


# =============================================================================
# Kappa policy
# =============================================================================


def _resolve_ckpt_file(path: str | Path) -> Path:
    """Mirror ``Trainer.load_pretrained_weights`` resolution.

    Accepts a direct ``model.safetensors`` file, a ``ckpt_*`` directory, or a
    run directory (latest ``ckpt_*`` wins).
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        if (p / "model.safetensors").is_file():
            return p / "model.safetensors"
        ckpt_root = p / "checkpoints"
        if ckpt_root.is_dir():
            ckpt_dirs = [
                d for d in ckpt_root.iterdir()
                if d.is_dir() and d.name.startswith("ckpt_")
            ]
            if not ckpt_dirs:
                raise FileNotFoundError(f"No ckpt_* dirs under {ckpt_root}")
            ckpt_dirs.sort(key=lambda d: int(d.name.split("_")[1]))
            cand = ckpt_dirs[-1] / "model.safetensors"
            if not cand.is_file():
                raise FileNotFoundError(f"Expected {cand} to exist")
            return cand
    raise FileNotFoundError(f"Cannot resolve checkpoint from {path}")


def _detect_proprio_in_ckpt(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("proprio_embedder.") for k in state_dict.keys())


def _detect_sonic_dim_from_ckpt(state_dict: dict[str, torch.Tensor], fallback: int) -> int:
    """Read SONIC dim from ``velocity_head.net.{first|last}.weight`` shapes."""
    for k in ("velocity_head.net.2.weight",):
        if k in state_dict:
            return int(state_dict[k].shape[0])
    for k, v in state_dict.items():
        if k.startswith("action_embedder.") and k.endswith(".weight"):
            return int(v.shape[1])
    return fallback


def _maybe_load_proprio_bounds(
    ckpt_file: Path, dim: int, mode: str = "bounds_q99",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Look for ``proprio_stats.json`` next to the checkpoint and assemble
    (lo, hi, ill_mask) for [-1, 1] normalization. Returns ``None`` if absent.
    Matches the dataset-side ``_load_normalize_bounds`` exactly.
    """
    stats_path = ckpt_file.parent / "proprio_stats.json"
    if not stats_path.is_file():
        return None
    from src.data.tokenized_video_dataset import _load_normalize_bounds
    with open(stats_path, "r") as f:
        stats = json.load(f)
    return _load_normalize_bounds(stats, "proprio", dim, mode)


def _normalize_proprio(
    raw: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if bounds is None:
        return raw.astype(np.float32, copy=False)
    lo, hi, _ = bounds
    out = (raw - lo) / (hi - lo) * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


# =============================================================================
# Direct C++ state subscription (port 5557 `g1_debug`)
# =============================================================================


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Reorder a (..., 4) WXYZ quaternion to XYZW.

    Mirrors ``src.tokenizer.sonic_encoder._quat_wxyz_to_xyzw`` but kept local
    so the server has no extra import dependency.
    """
    q = np.asarray(q)
    out = np.empty_like(q)
    out[..., 0] = q[..., 1]
    out[..., 1] = q[..., 2]
    out[..., 2] = q[..., 3]
    out[..., 3] = q[..., 0]
    return out


class _G1StateTap:
    """Background ZMQ SUB tap on the ``g1_debug`` topic from gear_sonic_deploy.

    Maintains the last two ``(jp_29, base_quat_wxyz, t_monotonic)`` samples so
    ``KappaPolicy._assemble_proprio`` can:
      • compute joint velocities at the *C++ control rate* (instead of the
        ~2.5 Hz get_action cadence, which would alias jv by ~20×), and
      • recover the raw pelvis quaternion that Terminal 3 otherwise flattens
        into projected_gravity (losing yaw).

    The C++ publisher writes a msgpack-encoded dict prefixed with the topic
    bytes; see ``gear_sonic_deploy/.../zmq_output_handler.hpp:316-324`` for
    the wire layout. ``body_q`` is a ``double[29]`` in MuJoCo order, which is
    the exact ordering Kappa's 64-D proprio jp slot uses.

    Thread safety: the SUB socket is owned exclusively by the background
    poller. The consumer reads a single immutable snapshot tuple via the
    ``self._snapshot`` reference, which Python rebinds atomically under the
    GIL. No locks required.
    """

    TOPIC = "g1_debug"
    JP_DIM = 29
    QUAT_DIM = 4

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5557,
        poll_hz: float = 200.0,
    ):
        self._poll_period = 1.0 / float(poll_hz)
        self._stop = threading.Event()
        self._snapshot: tuple | None = None
        self._sock_ok = False

        self._ctx: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        try:
            self._ctx = zmq.Context()
            self._socket = self._ctx.socket(zmq.SUB)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, self.TOPIC)
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.setsockopt(zmq.RCVTIMEO, 0)
            self._socket.connect(f"tcp://{host}:{port}")
        except Exception as e:
            print(f"[Kappa] state-tap: failed to connect tcp://{host}:{port}: {e}")
            self._socket = None
            return

        self._sock_ok = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="kappa-state-tap",
        )
        self._thread.start()
        print(
            f"[Kappa] state-tap: subscribed to tcp://{host}:{port} "
            f"(topic={self.TOPIC!r}, poll={poll_hz:.0f} Hz)"
        )

    @property
    def connected(self) -> bool:
        return self._sock_ok

    def _run(self) -> None:
        topic_len = len(self.TOPIC)
        prev_jp: np.ndarray | None = None
        prev_t: float | None = None
        while not self._stop.is_set():
            advanced = False
            try:
                raw = self._socket.recv(zmq.NOBLOCK)
                msg = msgpack.unpackb(raw[topic_len:], raw=False)
                body_q = msg.get("body_q")
                base_quat = msg.get("base_quat")
                if body_q is not None and base_quat is not None:
                    jp_arr = np.asarray(body_q, dtype=np.float32).reshape(-1)
                    quat_arr = np.asarray(base_quat, dtype=np.float32).reshape(-1)
                    if jp_arr.size >= self.JP_DIM and quat_arr.size == self.QUAT_DIM:
                        jp_arr = jp_arr[: self.JP_DIM].copy()
                        now = time.monotonic()
                        # Atomic-ish snapshot publish.
                        self._snapshot = (prev_jp, prev_t, jp_arr, now, quat_arr.copy())
                        prev_jp, prev_t = jp_arr, now
                        advanced = True
            except zmq.error.Again:
                pass
            except Exception as e:
                # Don't tear down the tap on a single bad frame — log and keep going.
                print(f"[Kappa] state-tap: unpack error: {e}")

            if not advanced:
                time.sleep(self._poll_period)

    def latest(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(jv_29 or None, base_quat_wxyz_4 or None)``.

        ``jv`` is finite-difference of the last two C++ frames at the actual
        sample dt. ``base_quat`` is the most recent reading.
        """
        snap = self._snapshot
        if snap is None:
            return None, None
        prev_jp, prev_t, cur_jp, cur_t, cur_quat = snap
        jv: np.ndarray | None = None
        if prev_jp is not None and prev_t is not None:
            dt = cur_t - prev_t
            if dt > 1e-4:
                jv = ((cur_jp - prev_jp) / dt).astype(np.float32, copy=False)
        return jv, cur_quat

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        try:
            if self._socket is not None:
                self._socket.close(linger=0)
        except Exception:
            pass
        try:
            if self._ctx is not None:
                self._ctx.term()
        except Exception:
            pass


# =============================================================================
# Deploy logger — record every request/response for post-hoc analysis
# =============================================================================


class _DeployLogger:
    """Persists every server endpoint call to disk.

    Layout per server session::

        <log_dir>/<stamp>/
            manifest.jsonl                 # one JSON line per call (browsing)
            calls/call_<idx>.msgpack       # full payload (msgpack-numpy)

    The msgpack record carries the deserialized request, the result, and —
    for ``get_action`` — the policy's internal summary (prompt, pre/post-norm
    proprio) so downstream tooling can compare directly against the training
    set without re-running the policy. Manifest lines are JSON-friendly
    metadata only: ``idx``, timestamps, endpoint, prompt, observation shapes.
    """

    def __init__(self, root: str | Path, save_images: bool = True):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(root) / stamp
        self.calls_dir = self.session_dir / "calls"
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = open(self.session_dir / "manifest.jsonl", "w", buffering=1)
        self._idx = 0
        self._t0 = time.monotonic()
        self.save_images = bool(save_images)
        print(
            f"[Kappa] deploy logger → {self.session_dir}"
            f" (save_images={self.save_images})"
        )

    def log_call(
        self,
        endpoint: str,
        request_data: Any,
        result: Any,
        internals: dict[str, Any] | None = None,
    ) -> None:
        idx = self._idx
        self._idx += 1
        t_wall = time.time()
        t_mono = time.monotonic()

        payload: dict[str, Any] = {
            "idx": idx,
            "t_wall": t_wall,
            "t_mono": t_mono,
            "endpoint": endpoint,
            "request": request_data,
            "result": result,
        }
        if internals is not None:
            payload["internals"] = internals

        call_path = self.calls_dir / f"call_{idx:06d}.msgpack"
        call_path.write_bytes(MsgSerializer.to_bytes(payload))

        image_file: str | None = None
        if self.save_images and endpoint == "get_action":
            image_file = self._dump_first_image(idx, request_data)

        line: dict[str, Any] = {
            "idx": idx,
            "t_wall": t_wall,
            "t_mono_rel": t_mono - self._t0,
            "endpoint": endpoint,
            "file": call_path.name,
        }
        if image_file is not None:
            line["image_file"] = image_file
        line.update(self._summarize(endpoint, request_data, result, internals))
        self._manifest.write(json.dumps(line, default=str) + "\n")

    def _dump_first_image(self, idx: int, request_data: Any) -> str | None:
        """Save the first camera frame from request_data as a JPEG sibling of
        the msgpack file. Returns the basename on success, ``None`` otherwise.
        """
        if not isinstance(request_data, dict):
            return None
        obs = request_data.get("observation", {})
        video = obs.get("video", {}) if isinstance(obs, dict) else {}
        if not isinstance(video, dict) or not video:
            return None
        arr = video.get("ego_view")
        if arr is None:
            arr = next(iter(video.values()))
        arr = np.asarray(arr)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        try:
            from PIL import Image
            img_path = self.calls_dir / f"call_{idx:06d}.jpg"
            Image.fromarray(arr).save(img_path, format="JPEG", quality=92)
            return img_path.name
        except Exception as e:
            print(f"[Kappa] deploy logger: image dump failed (call {idx}): {e}")
            return None

    @staticmethod
    def _summarize(
        endpoint: str,
        request_data: Any,
        result: Any,
        internals: dict[str, Any] | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if endpoint != "get_action":
            return out
        if internals is not None:
            if "prompt" in internals:
                out["prompt"] = internals["prompt"]
            if isinstance(internals.get("proprio_raw"), np.ndarray):
                out["proprio_raw.shape"] = list(internals["proprio_raw"].shape)
            out["use_proprio_inference"] = bool(internals.get("use_proprio_inference", False))
        obs = request_data.get("observation", {}) if isinstance(request_data, dict) else {}
        video = obs.get("video", {}) if isinstance(obs, dict) else {}
        if isinstance(video, dict):
            for k, v in video.items():
                arr = np.asarray(v)
                out[f"video.{k}.shape"] = list(arr.shape)
                out[f"video.{k}.dtype"] = str(arr.dtype)
                break
        if isinstance(result, (list, tuple)) and result and isinstance(result[0], dict):
            for k, v in result[0].items():
                if isinstance(v, np.ndarray):
                    out[f"action.{k}.shape"] = list(v.shape)
        return out

    def close(self) -> None:
        try:
            self._manifest.close()
        except Exception:
            pass


class KappaPolicy:
    """Thin wrapper around ``MoTKappaModel`` exposing a get_action endpoint
    that matches what ``run_vla_inference.py`` expects on the wire."""

    def __init__(
        self,
        ckpt_path: str,
        base_vlm_path: str,
        device: str = "cuda:0",
        action_horizon: int = 50,
        num_inference_steps: int = 4,
        deploy_proprio: bool = True,
        proprio_dim: int = 64,
        proprio_normalize_mode: str = "bounds_q99",
        seed: int | None = None,
        state_tap: _G1StateTap | None = None,
    ):
        self.device = torch.device(device)
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.deploy_proprio = bool(deploy_proprio)
        self.proprio_dim = int(proprio_dim)
        self._state_tap = state_tap

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        ckpt_file = _resolve_ckpt_file(ckpt_path)
        print(f"[Kappa] resolved checkpoint: {ckpt_file}")

        # Peek into safetensors header to discover proprio + sonic_dim before
        # building the MoT wrapper (so we don't bake the wrong shape).
        from safetensors.torch import load_file
        state_dict = load_file(str(ckpt_file), device="cpu")
        has_proprio_in_ckpt = _detect_proprio_in_ckpt(state_dict)
        sonic_dim = _detect_sonic_dim_from_ckpt(state_dict, fallback=66)
        print(
            f"[Kappa] ckpt detected: sonic_dim={sonic_dim}, "
            f"use_proprio={has_proprio_in_ckpt}"
        )

        # Build wrapper + MoT.
        from src.models.kappa_model import KappaModelWrapper
        wrapper = KappaModelWrapper(
            model_name_or_path=base_vlm_path,
            tune_vision=False,
            tune_llm=False,
            tune_mlp=False,
            device=self.device,
        )
        wrapper.convert_to_mot(
            sonic_dim=sonic_dim,
            action_loss_weight=1.0,
            use_proprio=has_proprio_in_ckpt,
            proprio_dim=self.proprio_dim,
        )

        # Load weights (strict=False to tolerate optimizer-only or processor-side
        # keys that don't live in the eval model).
        result = wrapper.model.load_state_dict(state_dict, strict=False)
        n_loaded = len(state_dict) - len(result.unexpected_keys)
        print(
            f"[Kappa] loaded {n_loaded}/{len(state_dict)} tensors "
            f"(missing={len(result.missing_keys)}, "
            f"unexpected={len(result.unexpected_keys)})"
        )
        if result.unexpected_keys:
            print(f"[Kappa]   first unexpected: {result.unexpected_keys[:3]}")
        if result.missing_keys:
            print(f"[Kappa]   first missing:    {result.missing_keys[:3]}")

        wrapper.model.eval()
        wrapper.model.to(device=self.device, dtype=torch.bfloat16)
        self.wrapper = wrapper
        self.model = wrapper.model  # MoTKappaModel
        self.processor = wrapper.processor
        self.sonic_dim = sonic_dim
        self.use_proprio_inference = has_proprio_in_ckpt and self.deploy_proprio
        self.has_proprio_in_ckpt = has_proprio_in_ckpt
        self.motion_dim = 64  # first 64 dims are SONIC motion latents
        self.hand_dim = max(0, sonic_dim - self.motion_dim)  # trailing dims are hand_binary

        # Proprio normalize bounds (per-checkpoint when bundled).
        self.proprio_bounds = (
            _maybe_load_proprio_bounds(ckpt_file, self.proprio_dim, proprio_normalize_mode)
            if self.use_proprio_inference else None
        )
        if self.use_proprio_inference and self.proprio_bounds is None:
            print(
                "[Kappa] WARNING: ckpt uses proprio but no proprio_stats.json "
                "found next to model.safetensors — proprio will be passed raw. "
                "Action quality will likely degrade."
            )

        # Populated by ``get_action`` for the deploy logger to read out the
        # internals (prompt, pre/post-norm proprio) without re-running the
        # observation→proprio pipeline.
        self._last_internals: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_image(observation: dict[str, Any]) -> np.ndarray:
        """Pull a single uint8 HxWx3 image from the Terminal 3 observation.

        Terminal 3 packs ``video.ego_view`` as ``(B=1, T=1, H, W, 3) uint8``.
        Falls back to any other video key if ``ego_view`` is missing.
        """
        video = observation.get("video", {})
        if not isinstance(video, dict) or not video:
            raise ValueError("observation['video'] is empty")
        cam_arr = video.get("ego_view")
        if cam_arr is None:
            cam_arr = next(iter(video.values()))
        arr = np.asarray(cam_arr)
        # Squeeze leading B/T dims of size 1; keep H, W, 3.
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"unexpected image shape {arr.shape}; expected (..., H, W, 3) uint8"
            )
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    @staticmethod
    def _first_prompt(observation: dict[str, Any]) -> str:
        lang = observation.get("language", {})
        if not isinstance(lang, dict) or not lang:
            return "perform the task"
        # Terminal 3 uses ``annotation.human.task_description`` (B, T) list[list[str]].
        for key in ("annotation.human.task_description", "task"):
            if key in lang:
                v = lang[key]
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)) and v[0]:
                    return str(v[0][0])
                if isinstance(v, (list, tuple)) and v:
                    return str(v[0])
        # Last resort: first key.
        v = next(iter(lang.values()))
        if isinstance(v, (list, tuple)) and v:
            inner = v[0]
            if isinstance(inner, (list, tuple)) and inner:
                return str(inner[0])
            return str(inner)
        return "perform the task"

    def _assemble_proprio_raw(self, observation: dict[str, Any]) -> np.ndarray | None:
        """Assemble the 64-D proprio vector *pre-normalization*.

        Layout (mirrors ``src.data.proprio_layout``):
          [ 0:29] jp        = left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7)
          [29:31] hand_bits = state.hand_binary_state
          [31:60] jv        = state-tap finite-diff (or 0 if tap unavailable)
          [60:64] quat_xyzw = state.base_quat (WXYZ → XYZW) → state-tap fallback → 0

        Returns ``None`` if any of the *jp* joint groups are missing in the
        Terminal 3 observation (we never fabricate jp — that would silently
        corrupt the body state). ``jv`` and ``quat`` degrade gracefully to
        zeros if neither the obs nor the C++ state-tap can supply them.
        """
        state = observation.get("state", {})
        if not isinstance(state, dict):
            return None

        def _vec(key: str, expected: int) -> np.ndarray | None:
            if key not in state:
                return None
            arr = np.asarray(state[key], dtype=np.float32)
            while arr.ndim > 1 and arr.shape[0] == 1:
                arr = arr[0]
            arr = arr.reshape(-1)
            if arr.size != expected:
                return None
            return arr

        parts: list[np.ndarray] = []
        for key, dim in (
            ("left_leg", 6), ("right_leg", 6), ("waist", 3),
            ("left_arm", 7), ("right_arm", 7),
        ):
            v = _vec(key, dim)
            if v is None:
                return None
            parts.append(v)
        jp = np.concatenate(parts, axis=0)  # (29,)

        hand = _vec("hand_binary_state", 2)
        if hand is None:
            hand = np.zeros(2, dtype=np.float32)

        # jv: prefer the C++ control-rate finite-diff from the state-tap.
        # Obs-level jv would be at ~2.5 Hz which severely under-bandwidths the
        # signal vs the training-time 50 Hz finite-diff.
        jv = np.zeros(29, dtype=np.float32)
        tap_jv: np.ndarray | None = None
        tap_quat_wxyz: np.ndarray | None = None
        if self._state_tap is not None:
            tap_jv, tap_quat_wxyz = self._state_tap.latest()
        if tap_jv is not None and tap_jv.size == 29:
            jv = tap_jv.astype(np.float32, copy=False)

        # quat: prefer state.base_quat (WXYZ) from Terminal 3, fall back to
        # the state-tap reading, then zeros.
        quat_xyzw = np.zeros(4, dtype=np.float32)
        obs_quat_wxyz = _vec("base_quat", 4)
        quat_wxyz_src: np.ndarray | None = obs_quat_wxyz
        if quat_wxyz_src is None and tap_quat_wxyz is not None and tap_quat_wxyz.size == 4:
            quat_wxyz_src = tap_quat_wxyz
        if quat_wxyz_src is not None:
            quat_xyzw = _quat_wxyz_to_xyzw(quat_wxyz_src.astype(np.float32, copy=False))

        raw = np.concatenate([jp, hand, jv, quat_xyzw], axis=0).astype(np.float32)
        if raw.shape[0] != self.proprio_dim:
            return None
        return raw

    def _assemble_proprio(self, observation: dict[str, Any]) -> np.ndarray | None:
        raw = self._assemble_proprio_raw(observation)
        if raw is None:
            return None
        return _normalize_proprio(raw, self.proprio_bounds)

    def _build_vlm_inputs(self, image_np: np.ndarray, prompt: str) -> dict[str, torch.Tensor]:
        """Mirror the training-time chat-template + processor call."""
        from PIL import Image
        pil = Image.fromarray(image_np)
        # Match training pixel distribution: decord/PyAV decode every frame to
        # DEFAULT_FRAME_SIZE=(W, H) with bilinear interp. Without this the
        # processor sees the raw 640x480 camera frame, smart_resize is a no-op
        # (both edges already multiples of 32), and the VLM gets 300 vision
        # tokens instead of the 70 it was trained on.
        target_w, target_h = DEFAULT_FRAME_SIZE
        if pil.size != (target_w, target_h):
            pil = pil.resize((target_w, target_h), Image.BILINEAR)
        conv = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False,
        )
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info([conv], image_patch_size=16)
        except Exception:
            image_inputs, video_inputs = [pil], None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        out = {
            "input_ids": inputs["input_ids"].to(self.device),
            "attention_mask": inputs["attention_mask"].to(self.device),
        }
        if "pixel_values" in inputs:
            out["pixel_values"] = inputs["pixel_values"].to(self.device, dtype=torch.bfloat16)
        if "image_grid_thw" in inputs:
            out["image_grid_thw"] = inputs["image_grid_thw"].to(self.device)
        return out

    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_sonic(
        self,
        vlm_inputs: dict[str, torch.Tensor],
        proprio: np.ndarray | None,
    ) -> np.ndarray:
        """Euler-integrate the flow-matching velocity field from t=1 to t=0.

        Schedule (matches training):
            noisy_t   = t * noise + (1 - t) * clean
            v_target  = noise - clean
        ⇒  x_{t - dt} = x_t - dt * v(x_t, t)

        Returns ``(T_action, sonic_dim) float32`` SONIC tokens.
        """
        T_action = self.action_horizon
        T_p = 1 if (self.use_proprio_inference and proprio is not None) else 0

        proprio_t: torch.Tensor | None = None
        if T_p > 0:
            proprio_t = torch.from_numpy(
                np.asarray(proprio, dtype=np.float32).reshape(1, -1)
            ).to(self.device, dtype=torch.bfloat16)

        cache = self.model.encode_latent(
            input_ids=vlm_inputs["input_ids"],
            attention_mask=vlm_inputs.get("attention_mask"),
            pixel_values=vlm_inputs.get("pixel_values"),
            image_grid_thw=vlm_inputs.get("image_grid_thw"),
            T_action=T_action,
            T_proprio=T_p,
            proprio=proprio_t,
        )

        # x_1: pure Gaussian noise.
        x = torch.randn(
            1, T_action, self.sonic_dim,
            device=self.device, dtype=torch.bfloat16,
        )
        steps = max(1, self.num_inference_steps)
        dt = 1.0 / steps
        for i in range(steps):
            t = 1.0 - i * dt
            timesteps = torch.full((1,), t, device=self.device, dtype=torch.float32)
            v = self.model.denoise_step(cache, x, timesteps)
            x = x - dt * v.to(x.dtype)
        return x.float().cpu().numpy()[0]  # (T_action, sonic_dim)

    # ------------------------------------------------------------------
    # Endpoint surface
    # ------------------------------------------------------------------

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options  # unused

        image_np = self._first_image(observation)
        prompt = self._first_prompt(observation)
        vlm_inputs = self._build_vlm_inputs(image_np, prompt)

        proprio_raw: np.ndarray | None = None
        proprio: np.ndarray | None = None
        if self.use_proprio_inference:
            proprio_raw = self._assemble_proprio_raw(observation)
            if proprio_raw is None:
                # The model still needs a proprio token; fall back to zeros so
                # the velocity head sees a well-formed but uninformative input.
                proprio = np.zeros(self.proprio_dim, dtype=np.float32)
            else:
                proprio = _normalize_proprio(proprio_raw, self.proprio_bounds)

        tokens = self._sample_sonic(vlm_inputs, proprio)  # (T, sonic_dim) float32

        motion = tokens[:, : self.motion_dim][np.newaxis, ...]  # (1, T, 64)
        action: dict[str, np.ndarray] = {
            "motion_token": motion.astype(np.float32, copy=False),
        }
        if self.hand_dim > 0:
            hand = tokens[:, self.motion_dim : self.motion_dim + self.hand_dim][np.newaxis, ...]
            # Clip to [0, 1] so downstream threshold-at-0.5 stays well-defined.
            hand = np.clip(hand, 0.0, 1.0)
            action["hand_binary_action"] = hand.astype(np.float32, copy=False)

        self._last_internals = {
            "prompt": prompt,
            "use_proprio_inference": bool(self.use_proprio_inference),
            "proprio_raw": proprio_raw,
            "proprio_normalized": proprio,
        }
        return action, {}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        del options
        return {}

    def get_modality_config(self) -> dict[str, Any]:
        # Terminal 3 (and the bundled _LocalPolicyClient) never queries this,
        # but the GR00T PolicyClient does in strict mode. Return an empty dict
        # to stay protocol-clean without forcing a ModalityConfig import.
        return {}


# =============================================================================
# Server loop (vendored from gr00t.policy.server_client.PolicyServer)
# =============================================================================


class _Endpoint:
    __slots__ = ("handler", "requires_input")

    def __init__(self, handler: Callable, requires_input: bool = True):
        self.handler = handler
        self.requires_input = requires_input


class KappaPolicyServer:
    def __init__(
        self,
        policy: KappaPolicy,
        host: str = "0.0.0.0",
        port: int = 5550,
        logger: _DeployLogger | None = None,
    ):
        self.policy = policy
        self.logger = logger
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, _Endpoint] = {
            "ping": _Endpoint(self._ping, requires_input=False),
            "kill": _Endpoint(self._kill, requires_input=False),
            "get_action": _Endpoint(policy.get_action),
            "reset": _Endpoint(policy.reset),
            "get_modality_config": _Endpoint(
                policy.get_modality_config, requires_input=False,
            ),
        }

    def _ping(self) -> dict[str, str]:
        return {"status": "ok", "message": "Kappa server is running"}

    def _kill(self) -> dict[str, str]:
        self.running = False
        return {"status": "shutting down"}

    def run(self) -> None:
        endpoint_addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"[Kappa] server ready, listening on {endpoint_addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = MsgSerializer.from_bytes(message)
                name = request.get("endpoint", "get_action")
                if name not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {name}")
                ep = self._endpoints[name]
                data = request.get("data", {}) or {}
                result = ep.handler(**data) if ep.requires_input else ep.handler()
                self.socket.send(MsgSerializer.to_bytes(result))
                if self.logger is not None:
                    internals = (
                        getattr(self.policy, "_last_internals", None)
                        if name == "get_action" else None
                    )
                    try:
                        self.logger.log_call(name, data, result, internals=internals)
                    except Exception as e:
                        print(f"[Kappa] deploy logger: log_call failed: {e}")
            except Exception as e:
                print(f"[Kappa] error in server: {e}")
                traceback.print_exc()
                # Best-effort: send a structured error back so the REQ socket
                # doesn't deadlock waiting for a reply.
                try:
                    self.socket.send(MsgSerializer.to_bytes({"error": str(e)}))
                except zmq.error.ZMQError:
                    pass

        try:
            self.socket.close(linger=0)
            self.context.term()
        except Exception:
            pass


# =============================================================================
# CLI
# =============================================================================


@dataclass
class ServerConfig:
    """Configuration for the Kappa ZMQ policy server."""

    ckpt_path: str
    """Path to ``model.safetensors``, a ``ckpt_*`` dir, or a run dir."""

    base_vlm_path: str = str(_REPO_ROOT / "resource/model/Qwen3-VL-2B-Instruct")
    """Path to the Qwen3-VL-2B base used at training (processor + backbone)."""

    device: str = "cuda:0"

    host: str = "0.0.0.0"
    port: int = 5550

    action_horizon: int = 50
    """SONIC chunk length per inference (matches ``action_chunk_size`` in
    KappaModelConfig). Terminal 3 default ``action_horizon=40`` indexes into
    this chunk so any value ≥ 40 is safe."""

    num_inference_steps: int = 4
    """Number of Euler steps for the flow-matching sampler (training default)."""

    deploy_proprio: bool = True
    """If False, force-disable the proprio branch even when the checkpoint
    was trained with proprio. The model's velocity head will then receive a
    zero proprio token via the encode_latent T_proprio=0 path."""

    proprio_normalize_mode: str = "bounds_q99"
    """Match training: ``bounds`` (min/max) or ``bounds_q99`` (q01/q99)."""

    state_tap_enable: bool = True
    """Subscribe to the C++ deploy's ``g1_debug`` ZMQ topic to recover the
    raw pelvis quaternion and a control-rate finite-diff of joint velocities.
    Required to fill the ``jv`` and ``quat_xyzw`` slots of the 64-D proprio at
    fidelity matching training. Safe to leave on even when the C++ deploy
    isn't running — the tap silently degrades to zeros for those slots."""

    state_tap_host: str = "localhost"
    state_tap_port: int = 5557
    """Match ``InferenceConfig.state_zmq_host/port`` in run_vla_inference.py."""

    state_tap_poll_hz: float = 200.0
    """Background poll rate. 200 Hz comfortably oversamples a 50 Hz control
    loop; raise if your deploy publishes faster."""

    log_enable: bool = True
    """If True, persist every ZMQ request/response to disk under
    ``log_dir/<stamp>/`` for post-hoc comparison against the training set.
    Disable with ``--no-log-enable`` to skip ~1 MB/get_action disk writes."""

    log_dir: str = str(_REPO_ROOT / ".runs/deploy_log")
    """Parent directory for per-session deploy logs. A new timestamped
    subdirectory is created at server startup."""

    log_images: bool = True
    """If True, additionally dump the first camera frame of every
    ``get_action`` request as ``calls/call_<idx>.jpg`` for human inspection.
    The image is also kept inside the msgpack file regardless of this flag."""

    seed: int | None = None


def main(cfg: ServerConfig) -> None:
    # Default to the same env knobs the GR00T server expects in deploy.md.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    print("Starting Kappa policy server")
    print(f"  ckpt_path:           {cfg.ckpt_path}")
    print(f"  base_vlm_path:       {cfg.base_vlm_path}")
    print(f"  device:              {cfg.device}")
    print(f"  host:                {cfg.host}")
    print(f"  port:                {cfg.port}")
    print(f"  action_horizon:      {cfg.action_horizon}")
    print(f"  num_inference_steps: {cfg.num_inference_steps}")
    print(f"  deploy_proprio:      {cfg.deploy_proprio}")
    print(f"  state_tap_enable:    {cfg.state_tap_enable}")
    if cfg.state_tap_enable:
        print(f"  state_tap_endpoint:  tcp://{cfg.state_tap_host}:{cfg.state_tap_port}")
    print(f"  log_enable:          {cfg.log_enable}")
    if cfg.log_enable:
        print(f"  log_dir:             {cfg.log_dir}")
        print(f"  log_images:          {cfg.log_images}")

    state_tap: _G1StateTap | None = None
    if cfg.state_tap_enable and cfg.deploy_proprio:
        state_tap = _G1StateTap(
            host=cfg.state_tap_host,
            port=cfg.state_tap_port,
            poll_hz=cfg.state_tap_poll_hz,
        )

    policy = KappaPolicy(
        ckpt_path=cfg.ckpt_path,
        base_vlm_path=cfg.base_vlm_path,
        device=cfg.device,
        action_horizon=cfg.action_horizon,
        num_inference_steps=cfg.num_inference_steps,
        deploy_proprio=cfg.deploy_proprio,
        proprio_normalize_mode=cfg.proprio_normalize_mode,
        seed=cfg.seed,
        state_tap=state_tap,
    )

    logger: _DeployLogger | None = None
    if cfg.log_enable:
        logger = _DeployLogger(cfg.log_dir, save_images=cfg.log_images)

    server = KappaPolicyServer(
        policy=policy, host=cfg.host, port=cfg.port, logger=logger,
    )
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[Kappa] shutting down (Ctrl-C)")
    finally:
        if logger is not None:
            logger.close()
        if state_tap is not None:
            state_tap.close()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))

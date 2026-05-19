"""Real-Time Chunking controller for the Kappa model (SONIC token output).

Adapted from Psi0's ``RealTimeChunkController``. Two threads share state:

* **Control thread** (fixed-rate, ``CTRL_HZ`` Hz): each tick reads the next
  step out of ``A_cur`` (a chunk of SONIC tokens) and returns it; never blocks
  on the model.
* **Inference thread** (asynchronous): once at least ``s_min`` ticks have
  been consumed, runs a full RTC denoise pass to produce a new chunk
  (``A_new``). The actual delay ``d`` is estimated from past iterations as
  ``max(Q)``; the new chunk's first ``d`` positions are pinned to the
  carried-over tail of the old chunk via training-time RTC prefix
  substitution.

The chunk emitted by ``step`` is a single SONIC token (shape ``(sonic_dim,)``)
— downstream consumers run the SONIC ONNX decoder to recover joint targets.

The controller is model-agnostic *except* for the inference call signature.
We wrap a ``MoTKappaModel`` (or any object with ``encode_latent`` /
``denoise_step_rtc``).
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any, Optional

import numpy as np
import torch


class RealTimeChunkController:
    """Async chunk controller emitting SONIC token chunks.

    Parameters
    ----------
    model : MoTKappaModel-like
        Must expose ``encode_latent`` returning a cache dict and
        ``denoise_step_rtc(cache, ...)`` returning the SONIC velocity.
    processor : transformers AutoProcessor
        Used to tokenize the text + image prompt for ``encode_latent``.
    scheduler : diffusers FlowMatchEulerDiscreteScheduler-like
        Provides ``timesteps`` and a ``step(model_pred, t, sample)`` API.
    cfg : object with the SONIC inference flags (use_proprio, sonic_dim,
        action_chunk_size, num_inference_steps, ...).
    """

    def __init__(
        self,
        model,
        processor,
        scheduler,
        cfg,
        ctrl_hz: float = 50.0,
        s_min: int = 15,
        delay_buffer_size: int = 6,
        d_init: int = 6,
        proprio_min: Optional[np.ndarray] = None,
        proprio_max: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.processor = processor
        self.scheduler = scheduler
        self.cfg = cfg
        self.H = int(cfg.action_chunk_size)
        self.SONIC_DIM = int(getattr(cfg, "sonic_dim", 66))
        self.PROPRIO_DIM = int(getattr(cfg, "proprio_dim", 64))
        self.use_proprio = bool(getattr(cfg, "use_proprio", False))
        self.num_inference_steps = int(getattr(cfg, "num_inference_steps", 8))

        self.ctrl_period = 1.0 / float(ctrl_hz)
        self.s_min = int(s_min)
        self.Q: deque[int] = deque([int(d_init)] * delay_buffer_size, maxlen=delay_buffer_size)

        # Training normalized proprio to [-1, 1] via q01/q99 bounds; mirror that
        # here so the deployed model sees the same distribution. When stats are
        # not supplied, raw values pass through (only safe if the upstream
        # already normalized).
        if proprio_min is not None and proprio_max is not None:
            self.proprio_min = np.asarray(proprio_min, dtype=np.float32)
            self.proprio_max = np.asarray(proprio_max, dtype=np.float32)
            if self.proprio_min.shape != (self.PROPRIO_DIM,) or self.proprio_max.shape != (self.PROPRIO_DIM,):
                raise ValueError(
                    f"proprio bounds shape mismatch: expected ({self.PROPRIO_DIM},), "
                    f"got min={self.proprio_min.shape} max={self.proprio_max.shape}"
                )
        else:
            self.proprio_min = None
            self.proprio_max = None

        # Shared state.
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.A_cur: np.ndarray = np.zeros((self.H, self.SONIC_DIM), dtype=np.float32)
        self.t: int = 0
        self._latest_obs: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._infer_th: threading.Thread | None = None

        device = next(model.parameters()).device
        self.device = device
        self.dtype = next(model.parameters()).dtype

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_start(self, obs: dict[str, Any]) -> None:
        """Run two non-RTC predictions to autotune CUDA kernels, then start
        the async inference loop. ``A_cur`` is initialized from the second
        warm-up so the control thread has a chunk to consume immediately.
        """
        with self._cond:
            self._latest_obs = obs
        # First chunk: standard flow (no RTC); seeds ``A_cur``.
        A_first = self._predict_chunk_no_rtc(obs)
        self.A_cur = A_first
        # Two warm-up RTC predictions to amortize cuDNN benchmark + ORT
        # init costs before the live loop sees them.
        prev = self._make_prev_actions(self.A_cur, s=self.s_min)
        for _ in range(2):
            _ = self._predict_chunk_rtc(obs, prev_sonic=prev, delay=int(max(self.Q)))

        self._stop.clear()
        self._infer_th = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_th.start()

    def shutdown(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._infer_th is not None:
            self._infer_th.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Control thread API
    # ------------------------------------------------------------------

    def step(self, obs: dict[str, Any]) -> np.ndarray:
        """One control tick. Returns a single SONIC token vector.

        Always non-blocking on inference: even if the inference thread is
        still running, ``A_cur`` is consumed step-by-step; the loop's
        ``Q.append(t-s)`` makes the next ``d`` accommodate the actual delay.
        """
        with self._cond:
            self._latest_obs = obs
            self.t = min(self.t + 1, self.H)
            idx = self.t - 1
            sonic_token = self.A_cur[idx].copy()
            self._cond.notify_all()
        return sonic_token

    # ------------------------------------------------------------------
    # Inference thread
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while (not self._stop.is_set()) and (self.t < self.s_min):
                    self._cond.wait(timeout=0.5)
                if self._stop.is_set():
                    return
                s = int(self.t)
                obs = dict(self._latest_obs) if self._latest_obs is not None else {}
                A_cur_snapshot = self.A_cur.copy()

            d = int(max(self.Q))
            prev = self._make_prev_actions(A_cur_snapshot, s=s)
            try:
                A_new = self._predict_chunk_rtc(obs, prev_sonic=prev, delay=d)
            except Exception as e:  # noqa: BLE001
                print(f"[RTC] inference error: {e!r}; reusing current chunk")
                continue

            with self._cond:
                # New chunk's first ``t-s`` positions correspond to old
                # chunk's [s : s+(t-s)] which was just consumed; the
                # training-time RTC prefix guarantees they match.
                self.A_cur = A_new
                advanced = max(0, self.t - s)
                self.t = advanced
                self.Q.append(int(advanced if advanced > 0 else 1))
                self._cond.notify_all()

    @staticmethod
    def _make_prev_actions(A_cur: np.ndarray, s: int) -> np.ndarray:
        """Shift ``A_cur[s:]`` to the front and zero-pad the tail.

        The new chunk's frame ``i`` corresponds (in absolute time) to the
        old chunk's frame ``s + i``; the first ``H - s`` positions therefore
        already have a predicted value from the previous denoise pass and
        can be reused as the RTC prefix anchor. The tail (``H - s : H``)
        was not predicted by the old chunk and is left zero — those
        positions are the `s` bits that must remain free in this iteration.
        """
        H = A_cur.shape[0]
        out = np.zeros_like(A_cur)
        if s < H:
            out[: H - s] = A_cur[s:]
        return out

    # ------------------------------------------------------------------
    # Single-chunk denoise wrappers
    # ------------------------------------------------------------------

    def _build_inputs(self, obs: dict[str, Any]):
        """Tokenize the prompt + image. Returns ``(input_ids, attention_mask,
        pixel_values, image_grid_thw)``."""
        from qwen_vl_utils import process_vision_info

        text = obs.get("text", "perform the task")
        image = obs["image"]
        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }]]
        rendered = [
            self.processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False,
            )
            for conv in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=rendered, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        device = self.device
        return {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
            "pixel_values": inputs.get("pixel_values").to(device, dtype=self.dtype),
            "image_grid_thw": inputs.get("image_grid_thw").to(device),
        }

    @torch.no_grad()
    def _predict_chunk_no_rtc(self, obs: dict[str, Any]) -> np.ndarray:
        """Cold-start chunk via plain flow matching (no RTC prefix)."""
        prompt = self._build_inputs(obs)
        T_p = 1 if self.use_proprio else 0
        proprio = self._proprio_tensor(obs, B=1, dtype=self.dtype, device=self.device)
        cache = self.model.encode_latent(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            pixel_values=prompt["pixel_values"],
            image_grid_thw=prompt["image_grid_thw"],
            T_action=self.cfg.action_chunk_size,
            T_proprio=T_p,
            proprio=proprio,
        )
        B = cache["B"]
        T = self.cfg.action_chunk_size
        device = self.device
        sonic = torch.randn(B, T, self.SONIC_DIM, device=device, dtype=cache["dtype"])

        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            t_scalar = (timestep / self.scheduler.config.num_train_timesteps).to(cache["dtype"])
            t_sonic = t_scalar.expand(B)
            v_sonic = self.model.denoise_step_rtc(
                cache=cache,
                noisy_sonic_tokens=sonic,
                timesteps_sonic=t_sonic,
            )
            sonic = self.scheduler.step(v_sonic, timestep, sonic).prev_sample
        return sonic[0].float().cpu().numpy()

    @torch.no_grad()
    def _predict_chunk_rtc(
        self, obs: dict[str, Any], prev_sonic: np.ndarray, delay: int,
    ) -> np.ndarray:
        prompt = self._build_inputs(obs)
        T_p = 1 if self.use_proprio else 0
        proprio = self._proprio_tensor(obs, B=1, dtype=self.dtype, device=self.device)
        cache = self.model.encode_latent(
            input_ids=prompt["input_ids"],
            attention_mask=prompt["attention_mask"],
            pixel_values=prompt["pixel_values"],
            image_grid_thw=prompt["image_grid_thw"],
            T_action=self.cfg.action_chunk_size,
            T_proprio=T_p,
            proprio=proprio,
        )
        B = cache["B"]
        T = self.cfg.action_chunk_size
        device = self.device
        dtype = cache["dtype"]

        sonic = torch.randn(B, T, self.SONIC_DIM, device=device, dtype=dtype)
        prev_sonic_t = torch.from_numpy(prev_sonic).to(device=device, dtype=dtype).unsqueeze(0)
        delay_t = torch.tensor([int(delay)], device=device, dtype=torch.long)

        self.scheduler.set_timesteps(self.num_inference_steps, device=device)
        for timestep in self.scheduler.timesteps:
            t_scalar = (timestep / self.scheduler.config.num_train_timesteps).to(dtype)
            t_sonic = t_scalar.expand(B)
            # Hard-overwrite the prefix at the start of every step so
            # ``scheduler.step`` cannot drift it.
            arange_t = torch.arange(T, device=device)
            prefix_mask = arange_t[None, :] < delay_t[:, None]
            sonic = torch.where(prefix_mask[..., None], prev_sonic_t, sonic)

            v_sonic = self.model.denoise_step_rtc(
                cache=cache,
                noisy_sonic_tokens=sonic,
                timesteps_sonic=t_sonic,
                prev_sonic=prev_sonic_t,
                delay_sonic=delay_t,
            )
            sonic = self.scheduler.step(v_sonic, timestep, sonic).prev_sample

        # Final pin so even diffusers' last-step rounding doesn't move the
        # prefix off ``prev_sonic``.
        arange_t = torch.arange(T, device=device)
        prefix_mask = arange_t[None, :] < delay_t[:, None]
        sonic = torch.where(prefix_mask[..., None], prev_sonic_t, sonic)
        return sonic[0].float().cpu().numpy()

    def _proprio_tensor(
        self, obs: dict[str, Any], B: int, dtype, device,
    ) -> Optional[torch.Tensor]:
        if not self.use_proprio:
            return None
        if "proprio" not in obs:
            return torch.zeros(B, self.PROPRIO_DIM, device=device, dtype=dtype)
        arr = np.asarray(obs["proprio"], dtype=np.float32)
        if arr.shape != (self.PROPRIO_DIM,):
            raise ValueError(
                f"Expected proprio shape ({self.PROPRIO_DIM},), got {arr.shape}"
            )
        if self.proprio_min is not None:
            rng = self.proprio_max - self.proprio_min
            arr = (arr - self.proprio_min) / rng * 2.0 - 1.0
            np.clip(arr, -1.0, 1.0, out=arr)
        return torch.from_numpy(arr).to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1)

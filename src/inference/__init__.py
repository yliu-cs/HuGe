"""Inference / deployment helpers for Kappa models.

* ``kappa_rtc_controller.RealTimeChunkController`` — control + inference
  thread pair adapted from Psi0's ``RealTimeChunkController``. Wraps the
  Kappa MoT model's ``encode_latent`` + ``denoise_step_rtc`` for use under
  a fixed-rate control loop, emitting SONIC token chunks.
* ``kappa_serve_rtc.Server`` — WebSocket front end that takes obs+text from
  a client and streams SONIC token vectors back at the configured control
  rate. Downstream clients decode SONIC tokens via the SONIC ONNX decoder.
"""
from .kappa_rtc_controller import RealTimeChunkController
from .kappa_serve_rtc import Server

__all__ = ["RealTimeChunkController", "Server"]

"""Inference / deployment helpers for Kappa post-train models.

* ``kappa_rtc_controller.RealTimeChunkController`` — control + inference
  thread pair adapted from Psi0's ``RealTimeChunkController``. Wraps the
  Kappa MoT model's ``encode_latent`` + ``denoise_step_rtc`` for use under
  a fixed-rate control loop.
* ``kappa_serve_rtc.Server`` — WebSocket front end that takes obs+text from
  a client and streams normalized 31-D actions back at the configured
  control rate.
"""
from .kappa_rtc_controller import RealTimeChunkController
from .kappa_serve_rtc import Server

__all__ = ["RealTimeChunkController", "Server"]

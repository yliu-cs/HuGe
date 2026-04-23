"""Training step timer with CUDA event support and sliding window statistics."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional

import torch


class StepTimer:
    """Tracks elapsed time for named phases using CUDA events (GPU) or perf_counter (CPU/MPS)."""

    def __init__(self, window_size: int = 50, use_cuda: bool = True, use_mps: bool = False):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.use_mps = (not self.use_cuda) and use_mps and torch.backends.mps.is_available()
        self.window_size = window_size
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self._starts: dict[str, float | torch.cuda.Event] = {}
        self._pending_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

    def start(self, name: str) -> None:
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self._starts[name] = start_event
        else:
            if self.use_mps:
                torch.mps.synchronize()
            self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> Optional[float]:
        if name not in self._starts:
            return None
        if self.use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._pending_events[name] = (self._starts.pop(name), end_event)
            return None  # resolved later via sync
        else:
            if self.use_mps:
                torch.mps.synchronize()
            elapsed = time.perf_counter() - self._starts.pop(name)
            self._history[name].append(elapsed)
            return elapsed

    def sync(self) -> dict[str, float]:
        """Synchronize CUDA events and return elapsed times for all pending phases."""
        results = {}
        if self._pending_events:
            torch.cuda.synchronize()
            for name, (start_ev, end_ev) in self._pending_events.items():
                elapsed_ms = start_ev.elapsed_time(end_ev)
                elapsed_s = elapsed_ms / 1000.0
                self._history[name].append(elapsed_s)
                results[name] = elapsed_s
            self._pending_events.clear()
        return results

    def record_value(self, name: str, value: float) -> None:
        self._history[name].append(value)

    def averages(self) -> dict[str, float]:
        return {name: sum(dq) / len(dq) for name, dq in self._history.items() if dq}

    def last_values(self) -> dict[str, float]:
        return {name: dq[-1] for name, dq in self._history.items() if dq}

    def summary(self, prefix: str = "time") -> dict[str, float]:
        """Return metrics dict with avg and last values, suitable for logging."""
        metrics = {}
        for name, dq in self._history.items():
            if not dq:
                continue
            metrics[f"{prefix}/{name}_avg"] = sum(dq) / len(dq)
            metrics[f"{prefix}/{name}_last"] = dq[-1]
        avgs = self.averages()
        if avgs:
            total_avg = sum(v for k, v in avgs.items() if k != "step_total")
            if "step_total" in avgs:
                metrics[f"{prefix}/overhead_pct"] = max(0, (avgs["step_total"] - total_avg) / avgs["step_total"] * 100)
        return metrics

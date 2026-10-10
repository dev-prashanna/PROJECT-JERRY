"""Measured performance metrics: FPS, per-stage latency, VRAM."""

from __future__ import annotations

import time
from collections import deque

import torch


class RollingStats:
    def __init__(self, window: int = 120) -> None:
        self.window = window
        self._values: deque[float] = deque(maxlen=window)

    def add(self, value: float) -> None:
        self._values.append(float(value))

    @property
    def last(self) -> float:
        return self._values[-1] if self._values else 0.0

    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    def p95(self) -> float:
        if not self._values:
            return 0.0
        data = sorted(self._values)
        idx = min(len(data) - 1, int(round(0.95 * (len(data) - 1))))
        return data[idx]

    def clear(self) -> None:
        self._values.clear()


class FpsMeter:
    """Event counter over a sliding time window (throughput, not latency)."""

    def __init__(self, window_s: float = 1.0) -> None:
        self.window_s = window_s
        self._stamps: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._stamps.append(now)
        self._trim(now)

    def _trim(self, now: float) -> None:
        while self._stamps and now - self._stamps[0] > self.window_s:
            self._stamps.popleft()

    @property
    def fps(self) -> float:
        now = time.time()
        self._trim(now)
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        if span <= 0:
            return 0.0
        return (len(self._stamps) - 1) / span


class PerformanceMonitor:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.capture_fps = FpsMeter()
        self.depth_fps = FpsMeter()
        self.stats: dict[str, RollingStats] = {
            "capture_to_infer_ms": RollingStats(),
            "preprocess_ms": RollingStats(),
            "inference_ms": RollingStats(),
            "postprocess_ms": RollingStats(),
            "render_ms": RollingStats(),
            "capture_to_display_ms": RollingStats(),
            "depth_age_ms": RollingStats(),
        }
        self.depth_updates = 0
        self.frames_displayed = 0
        self._t_start = time.time()

    def add(self, name: str, value: float) -> None:
        stats = self.stats.get(name)
        if stats is not None:
            stats.add(value)

    def on_capture(self, frame=None) -> None:
        self.capture_fps.tick()

    def on_depth(self, result) -> None:
        self.depth_fps.tick()
        self.depth_updates += 1
        self.add("capture_to_infer_ms", (result.t_start - result.t_capture) * 1000.0)
        self.add("preprocess_ms", result.preprocess_ms)
        self.add("inference_ms", result.inference_ms)
        self.add("postprocess_ms", result.postprocess_ms)

    def on_display(self, t_capture: float | None = None) -> None:
        self.frames_displayed += 1
        if t_capture is not None:
            self.add("capture_to_display_ms", (time.time() - t_capture) * 1000.0)

    def on_render(self, ms: float) -> None:
        self.add("render_ms", ms)

    def gpu_memory_mb(self) -> float | None:
        if not self.device.startswith("cuda"):
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def reset_gpu_peak(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> dict:
        out = {
            "camera_fps": round(self.capture_fps.fps, 2),
            "depth_fps": round(self.depth_fps.fps, 2),
            "uptime_s": round(time.time() - self._t_start, 1),
            "frames_displayed": self.frames_displayed,
            "depth_updates": self.depth_updates,
        }
        for name, stats in self.stats.items():
            out[f"{name}_mean"] = round(stats.mean(), 2)
            out[f"{name}_p95"] = round(stats.p95(), 2)
        vram = self.gpu_memory_mb()
        if vram is not None:
            out["vram_mb"] = round(vram, 1)
        return out

    def reset(self) -> None:
        for stats in self.stats.values():
            stats.clear()
        self._t_start = time.time()
        self.depth_updates = 0
        self.frames_displayed = 0

"""Latency measurement helpers.

GPU operations are asynchronous, so every timed stage involving CUDA is
synchronised before the stopwatch stops.  Otherwise the reported numbers would
measure only kernel *submission* time.
"""

from __future__ import annotations

import statistics
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import torch


def synchronize(device: Optional[str] = None) -> None:
    """Block until pending CUDA work finishes (no-op on CPU)."""
    if device and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class StageTimer:
    """Collect wall-clock durations per named stage.

    >>> with StageTimer("cuda") as t:
    ...     y = model(x)
    >>> t.durations["stage"]
    """

    device: Optional[str] = None
    sync_cuda: bool = True
    durations: Dict[str, float] = field(default_factory=dict)
    started: Optional[float] = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if self.sync_cuda:
            synchronize(self.device)
        self.started = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        if self.sync_cuda:
            synchronize(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda:
                synchronize(self.device)
            elapsed = time.perf_counter() - t0
            with self._lock:
                self.durations[name] = self.durations.get(name, 0.0) + elapsed

    @property
    def total(self) -> float:
        if self.started is None:
            return 0.0
        return time.perf_counter() - self.started

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            out = dict(self.durations)
        out["total"] = self.total
        return out


@dataclass
class RollingStats:
    """Online summary of the last N measurements."""

    window: int = 60
    _values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self._values.append(float(value))
        if len(self._values) > self.window:
            self._values.pop(0)

    @property
    def count(self) -> int:
        return len(self._values)

    def summary(self) -> Dict[str, float]:
        if not self._values:
            return {"count": 0}
        ordered = sorted(self._values)
        return {
            "count": len(ordered),
            "last": ordered[-1],
            "mean": statistics.fmean(ordered),
            "min": ordered[0],
            "max": ordered[-1],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        }


def format_seconds(seconds: Optional[float]) -> str:
    """Human friendly latency rendering for the GUI."""
    if seconds is None:
        return "--"
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"


def gpu_memory_mb(device: Optional[str] = None) -> Dict[str, float]:
    """Allocated / reserved CUDA memory in MiB (empty dict on CPU-only)."""
    if not (device and str(device).startswith("cuda") and torch.cuda.is_available()):
        return {}
    idx = 0
    if ":" in str(device):
        try:
            idx = int(str(device).split(":")[1])
        except ValueError:
            idx = 0
    free, total = torch.cuda.mem_get_info(idx)
    return {
        "allocated_mb": torch.cuda.memory_allocated(idx) / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(idx) / (1024 ** 2),
        "free_mb": free / (1024 ** 2),
        "total_mb": total / (1024 ** 2),
    }


def describe_device(device: Optional[str]) -> str:
    """Short human-readable device description for status bars."""
    if device is None:
        return "not loaded"
    dev = str(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        try:
            idx = int(dev.split(":")[1]) if ":" in dev else 0
            name = torch.cuda.get_device_name(idx)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            return f"{name} ({total:.1f} GB) — {dev}"
        except Exception:  # pragma: no cover
            return dev
    return dev


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else 0.0

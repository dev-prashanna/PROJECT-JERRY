"""Per-request latency and resource metrics.

Everything recorded here comes from real measurements (``time.perf_counter``
and, when available, ``torch.cuda`` memory counters).  Nothing is estimated.
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import METRICS_DIR
from src.utils.logger import get_logger

log = get_logger("metrics")


@dataclass
class RequestMetrics:
    request_id: str
    started_at: float = field(default_factory=time.time)
    stages: dict[str, float] = field(default_factory=dict)      # duration seconds
    stage_started: dict[str, float] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    _t0: float = field(default_factory=time.perf_counter)
    _open: Optional[str] = None

    def start(self, stage: str) -> None:
        self._open = stage
        self.stage_started[stage] = time.perf_counter()

    def stop(self, stage: Optional[str] = None) -> float:
        stage = stage or self._open
        if stage is None:
            return 0.0
        t1 = self.stage_started.pop(stage, None)
        if t1 is None:
            return 0.0
        dur = time.perf_counter() - t1
        self.stages[stage] = self.stages.get(stage, 0.0) + dur
        if self._open == stage:
            self._open = None
        return dur

    @property
    def total_s(self) -> float:
        return time.perf_counter() - self._t0

    @property
    def spoken_latency_s(self) -> Optional[float]:
        """End of user speech → first audible frame."""
        return self.stages.get("time_to_first_audio")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "started_at": self.started_at,
            "total_s": round(self.total_s, 4),
            "stages": {k: round(v, 4) for k, v in self.stages.items()},
            "flags": self.flags,
            "error": self.error,
        }


class MetricsRegistry:
    """Thread-safe store of :class:`RequestMetrics` with aggregates."""

    def __init__(self, maxlen: int = 500):
        self._lock = threading.Lock()
        self._items: list[RequestMetrics] = []
        self._maxlen = maxlen

    def new(self, request_id: str) -> RequestMetrics:
        m = RequestMetrics(request_id=request_id)
        with self._lock:
            self._items.append(m)
            if len(self._items) > self._maxlen:
                del self._items[: len(self._items) - self._maxlen]
        return m

    def get(self, request_id: str) -> Optional[RequestMetrics]:
        with self._lock:
            for m in reversed(self._items):
                if m.request_id == request_id:
                    return m
        return None

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [m.to_dict() for m in self._items]

    def stage_values(self, stage: str) -> list[float]:
        with self._lock:
            return [m.stages[stage] for m in self._items if stage in m.stages]

    def summary(self, stage: str) -> dict[str, float]:
        values = self.stage_values(stage)
        if not values:
            return {"count": 0}
        values_sorted = sorted(values)
        return {
            "count": len(values),
            "mean_s": round(statistics.fmean(values), 4),
            "median_s": round(statistics.median(values), 4),
            "p95_s": round(percentile(values_sorted, 95), 4),
            "min_s": round(values_sorted[0], 4),
            "max_s": round(values_sorted[-1], 4),
        }

    def all_stage_summaries(self) -> dict[str, dict[str, float]]:
        stages: set[str] = set()
        with self._lock:
            for m in self._items:
                stages.update(m.stages)
        return {s: self.summary(s) for s in sorted(stages)}

    def export(self, path: Optional[Path] = None) -> Path:
        path = path or (METRICS_DIR / f"metrics_{time.strftime('%Y%m%d_%H%M%S')}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": time.time(),
            "stages": self.all_stage_summaries(),
            "requests": self.records(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("metrics written to %s", path)
        return path

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile on an already sorted list."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(round((pct / 100.0) * len(sorted_values))) - 1))
    return sorted_values[k]


REGISTRY = MetricsRegistry()


class StageTimer:
    """Context manager: ``with StageTimer(metrics, "whisper"): ...``"""

    def __init__(self, metrics: Optional[RequestMetrics], stage: str):
        self.metrics = metrics
        self.stage = stage
        self.elapsed = 0.0

    def __enter__(self) -> "StageTimer":
        if self.metrics:
            self.metrics.start(self.stage)
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t
        if self.metrics:
            self.metrics.stop(self.stage)

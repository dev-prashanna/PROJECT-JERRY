"""Point / region depth measurement and heuristic quality indicators."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .calibration import CalibrationProfile
from .config import AppConfig
from .temporal_filter import ScalarEMA


def sample_region(depth: np.ndarray, x: int, y: int, k: int) -> dict:
    """Median / dispersion of the k x k neighborhood around (x, y), ignoring invalid values."""
    h, w = depth.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return {"ok": False, "reason": "out_of_bounds", "median": float("nan"), "n_valid": 0, "n_total": 0}
    half = k // 2
    x0, x1 = max(0, x - half), min(w, x + half + 1)
    y0, y1 = max(0, y - half), min(h, y + half + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float64)
    flat = patch.reshape(-1)
    valid = flat[np.isfinite(flat)]
    n_total = int(flat.size)
    n_valid = int(valid.size)
    if n_valid == 0:
        return {"ok": False, "reason": "no_valid_values", "median": float("nan"), "n_valid": 0, "n_total": n_total}
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return {
        "ok": True,
        "reason": "",
        "median": median,
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "mad": mad,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "n_valid": n_valid,
        "n_total": n_total,
    }


@dataclass
class Measurement:
    x: int
    y: int
    raw: float                       # raw model value (relative inverse depth, or meters for metric models)
    sample: dict
    t: float = field(default_factory=time.time)
    metric: float | None = None      # meters, when a calibration profile applies
    extrapolated: bool = False
    smoothed_raw: float | None = None
    quality: str = "unknown"
    quality_detail: dict = field(default_factory=dict)
    valid: bool = True
    reason: str = ""

    @property
    def display_value(self) -> float:
        if self.metric is not None:
            return self.metric
        return self.smoothed_raw if self.smoothed_raw is not None else self.raw


class DistanceEstimator:
    """Converts a depth map + pixel into a displayed measurement.

    * metric checkpoint  -> meters directly
    * calibrated relative -> fitted D(R) in meters (approximate, scene dependent)
    * relative (default) -> raw affine-invariant inverse depth in arbitrary units
    """

    def __init__(self, cfg: AppConfig, profile: CalibrationProfile | None, estimation_type: str) -> None:
        self.cfg = cfg
        self.profile = profile
        self.estimation_type = estimation_type

    @property
    def mode(self) -> str:
        if self.estimation_type == "metric":
            return "native_metric"
        return "calibrated" if self.profile is not None else "relative"

    @property
    def unit(self) -> str:
        if self.mode == "native_metric":
            return "m"
        if self.mode == "calibrated":
            return "m (calibrated)"
        return "a.u. (relative)"

    def measure(self, depth: np.ndarray, x: int, y: int, ema: ScalarEMA | None = None) -> Measurement:
        s = sample_region(depth, x, y, self.cfg.measurement.region_size)
        m = Measurement(x=x, y=y, raw=float("nan"), sample=s, t=time.time())
        if not s["ok"]:
            m.valid = False
            m.reason = s["reason"]
            m.quality = "invalid"
            return m

        raw = float(s["median"])
        m.raw = raw
        if ema is not None:
            m.smoothed_raw = ema.apply(raw)

        if self.estimation_type == "metric":
            m.metric = raw
        elif self.profile is not None:
            value, extrapolated = self.profile.predict(raw)
            m.metric = value
            m.extrapolated = extrapolated

        m.quality, m.quality_detail = self._quality(s, m)
        return m

    def _quality(self, s: dict, m: Measurement) -> tuple[str, dict]:
        """Heuristic indicators only - not calibrated confidence probabilities."""
        detail: dict[str, float | bool | str] = {}
        valid_fraction = s["n_valid"] / max(1, s["n_total"])
        detail["valid_fraction"] = round(valid_fraction, 3)

        median = abs(s["median"])
        rel_std = s["std"] / median if median > 1e-9 else (0.0 if s["std"] < 1e-9 else float("inf"))
        detail["rel_local_std"] = round(float(rel_std), 4) if math.isfinite(rel_std) else -1.0

        detail["in_calibrated_range"] = bool(
            m.metric is None or not m.extrapolated
        )

        score = 0
        if valid_fraction >= 0.99:
            score += 2
        elif valid_fraction >= 0.8:
            score += 1
        if math.isfinite(rel_std):
            if rel_std <= 0.05:
                score += 2
            elif rel_std <= 0.15:
                score += 1
        if detail["in_calibrated_range"]:
            score += 1

        label = "poor" if score <= 1 else ("fair" if score <= 3 else "good")
        detail["score"] = score
        return label, detail

    def format(self, m: Measurement) -> str:
        if not m.valid:
            return f"invalid ({m.reason})"
        if self.mode == "relative":
            value = m.display_value if m.display_value is not None else m.raw
            return f"R = {value:.3f} a.u."
        flag = " ~extrapolated" if m.extrapolated else ""
        return f"{m.display_value:.2f} m{flag}"


class PointStore:
    """Multiple selectable measurement points with per-point smoothing."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.points: dict[int, dict] = {}
        self._next_id = 1

    def add(self, x: int, y: int) -> int | None:
        if len(self.points) >= self.cfg.measurement.max_points:
            return None
        pid = self._next_id
        self._next_id += 1
        alpha = self.cfg.measurement.point_smoothing
        self.points[pid] = {
            "id": pid,
            "x": int(x),
            "y": int(y),
            "ema": ScalarEMA(alpha) if alpha > 0 else None,
            "measurement": None,
        }
        return pid

    def remove_nearest(self, x: int, y: int, max_dist: float = 30.0) -> int | None:
        best_id, best_d = None, max_dist
        for pid, p in self.points.items():
            d = math.hypot(p["x"] - x, p["y"] - y)
            if d <= best_d:
                best_id, best_d = pid, d
        if best_id is not None:
            del self.points[best_id]
        return best_id

    def clear(self) -> None:
        self.points.clear()

    def update(self, depth: np.ndarray, estimator: DistanceEstimator) -> dict[int, Measurement]:
        out: dict[int, Measurement] = {}
        for pid, p in self.points.items():
            out[pid] = estimator.measure(depth, p["x"], p["y"], p["ema"])
            p["measurement"] = out[pid]
        return out

"""Empirical calibration of relative (inverse) depth to metric distance.

Relative Depth Anything output is affine-invariant inverse depth: an arbitrary per-image
scale and shift. A fitted curve D = f(R) is therefore only valid for the scene, lighting,
camera settings and framing in which the samples were collected. Every profile stores the
sampled raw range so later predictions can be flagged when extrapolating.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - scipy is a declared dependency
    least_squares = None


class CalibrationError(ValueError):
    """Raised for invalid calibration input or an unusable profile."""


@dataclass
class Sample:
    r: float                 # raw model output at the reference point (region median)
    d_true: float            # physically measured distance in meters
    scene: str = "default"
    note: str = ""
    t: float = field(default_factory=time.time)

    def validate(self) -> None:
        if not math.isfinite(self.r):
            raise CalibrationError(f"Non-finite raw depth value: {self.r}")
        if not math.isfinite(self.d_true) or self.d_true <= 0:
            raise CalibrationError(f"Ground-truth distance must be finite and > 0 m, got {self.d_true}")


@dataclass
class Metrics:
    n: int
    mae: float
    rmse: float
    bias: float
    std_err: float
    max_err: float

    def to_dict(self) -> dict:
        return asdict(self)


def _metrics(pred: np.ndarray, true: np.ndarray) -> Metrics:
    err = pred - true
    return Metrics(
        n=int(err.size),
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        bias=float(np.mean(err)),
        std_err=float(np.std(err)),
        max_err=float(np.max(np.abs(err))),
    )


def _validate_samples(samples: Iterable[Sample], min_samples: int, min_distances: int) -> list[Sample]:
    out: list[Sample] = []
    for s in samples:
        s.validate()
        out.append(s)
    if len(out) < min_samples:
        raise CalibrationError(f"Need at least {min_samples} valid samples, got {len(out)}")
    unique = {round(s.d_true, 4) for s in out}
    if len(unique) < min_distances:
        raise CalibrationError(
            f"Need at least {min_distances} distinct measured distances to fit a curve, got {sorted(unique)}"
        )
    return out


# ----------------------------------------------------------------------------- model fits

def _fit_linear_np(r: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    a, b = np.polyfit(r, d, 1)
    return float(a), float(b)


def _fit_reciprocal_np(r: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    # D = a / (R + b); linearize: 1/D = (R + b)/a  ->  1/D = (1/a) R + b/a
    inv = 1.0 / d
    m, c = np.polyfit(r, inv, 1)
    if m == 0:
        raise CalibrationError("Degenerate reciprocal fit (zero slope)")
    a = 1.0 / m
    b = c * a
    return float(a), float(b)


def _robust(fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]],
            r: np.ndarray, d: np.ndarray, residual) -> tuple[float, float] | None:
    if least_squares is None:
        return None
    a0, b0 = fn(r, d)
    try:
        res = least_squares(residual, x0=np.array([a0, b0]), loss="soft_l1", f_scale=max(1e-3, float(np.median(d)) * 0.05))
    except Exception:
        return None
    if not res.success:
        return None
    return float(res.x[0]), float(res.x[1])


def fit_linear(samples: list[Sample], robust: bool = True) -> tuple[float, float]:
    r = np.array([s.r for s in samples], dtype=np.float64)
    d = np.array([s.d_true for s in samples], dtype=np.float64)
    if robust and least_squares is not None:
        got = _robust(_fit_linear_np, r, d, lambda p: p[0] * r + p[1] - d)
        if got is not None:
            return got
    return _fit_linear_np(r, d)


def fit_reciprocal(samples: list[Sample], robust: bool = True) -> tuple[float, float]:
    r = np.array([s.r for s in samples], dtype=np.float64)
    d = np.array([s.d_true for s in samples], dtype=np.float64)
    if robust and least_squares is not None:
        got = _robust(_fit_reciprocal_np, r, d, lambda p: p[0] / (r + p[1]) - d)
        if got is not None:
            a, b = got
            denom = r + b
            if np.any(np.abs(denom) < 1e-6):
                return _fit_reciprocal_np(r, d)
            return a, b
    return _fit_reciprocal_np(r, d)


def predict_model(model: str, params: tuple[float, float], r: float | np.ndarray) -> float | np.ndarray:
    a, b = params
    r_arr = np.asarray(r, dtype=np.float64)
    if model == "linear":
        out = a * r_arr + b
    elif model == "reciprocal":
        denom = r_arr + b
        if np.any(np.abs(denom) < 1e-9):
            raise CalibrationError("Singular reciprocal model (R + b == 0)")
        out = a / denom
    else:
        raise CalibrationError(f"Unknown calibration model: {model}")
    return float(out) if np.isscalar(r) or np.asarray(r).ndim == 0 else out


def monotonic(model: str, params: tuple[float, float], r_lo: float, r_hi: float) -> bool:
    """Check the fitted curve is monotonically decreasing in R (higher inverse depth = nearer)."""
    xs = np.linspace(r_lo, r_hi, 64)
    ys = predict_model(model, params, xs)
    diffs = np.diff(ys)
    if model == "linear":
        return bool(np.all(diffs <= 0) or np.all(diffs >= 0))
    return bool(np.all(diffs < 0) or np.all(diffs > 0))


# ----------------------------------------------------------------------------- profiles

@dataclass
class CalibrationProfile:
    model: str                       # "linear" | "reciprocal"
    params: tuple[float, float]
    created_at: float
    samples: list[Sample]
    candidate_models: dict[str, dict] = field(default_factory=dict)
    train_metrics: dict = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    r_min: float = 0.0
    r_max: float = 0.0
    note: str = ""
    version: int = 1

    @property
    def unit(self) -> str:
        return "m"

    def in_range(self, r: float) -> bool:
        return self.r_min <= r <= self.r_max

    def predict(self, r: float) -> tuple[float, bool]:
        """Returns (distance_m, extrapolated)."""
        value = float(predict_model(self.model, self.params, float(r)))
        return value, not self.in_range(float(r))

    def summary(self) -> dict:
        return {
            "model": self.model,
            "params": list(self.params),
            "samples": len(self.samples),
            "distinct_distances": sorted({round(s.d_true, 3) for s in self.samples}),
            "r_range": [self.r_min, self.r_max],
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "candidate_models": self.candidate_models,
            "created_at": self.created_at,
            "note": self.note,
        }

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model": self.model,
            "params": list(self.params),
            "created_at": self.created_at,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "candidate_models": self.candidate_models,
            "note": self.note,
            "samples": [asdict(s) for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationProfile":
        try:
            samples = [Sample(**s) for s in data.get("samples", [])]
            return cls(
                model=data["model"],
                params=tuple(data["params"]),
                created_at=float(data.get("created_at", 0.0)),
                samples=samples,
                candidate_models=data.get("candidate_models", {}),
                train_metrics=data.get("train_metrics", {}),
                val_metrics=data.get("val_metrics", {}),
                r_min=float(data.get("r_min", 0.0)),
                r_max=float(data.get("r_max", 0.0)),
                note=data.get("note", ""),
                version=int(data.get("version", 1)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"Malformed calibration profile: {exc}") from exc


def _eval(model: str, params: tuple[float, float], samples: list[Sample]) -> Metrics:
    r = np.array([s.r for s in samples], dtype=np.float64)
    d = np.array([s.d_true for s in samples], dtype=np.float64)
    return _metrics(np.asarray(predict_model(model, params, r)), d)


def split_holdout(samples: list[Sample], val_fraction: float = 0.3, seed: int = 0) -> tuple[list[Sample], list[Sample]]:
    """Holds out whole distances (not individual points) so validation tests interpolation/extrapolation."""
    distances = sorted({round(s.d_true, 4) for s in samples})
    if len(distances) < 3:
        return samples, []
    n_val = max(1, int(round(len(distances) * val_fraction)))
    n_val = min(n_val, len(distances) - 2)
    rng = np.random.default_rng(seed)
    order = list(distances)
    rng.shuffle(order)
    held = set(order[:n_val])
    train = [s for s in samples if round(s.d_true, 4) not in held]
    val = [s for s in samples if round(s.d_true, 4) in held]
    return train, val


def fit_profile(
    samples: Iterable[Sample],
    min_samples: int = 4,
    min_distances: int = 3,
    val_fraction: float = 0.3,
    note: str = "",
) -> CalibrationProfile:
    clean = _validate_samples(samples, min_samples, min_distances)
    train, val = split_holdout(clean, val_fraction)

    candidates: dict[str, tuple[float, float]] = {}
    report: dict[str, dict] = {}
    for name, fn in (("linear", fit_linear), ("reciprocal", fit_reciprocal)):
        try:
            params = fn(train)
            candidates[name] = params
            report[name] = {
                "params": list(params),
                "train": _eval(name, params, train).to_dict(),
                "val": _eval(name, params, val).to_dict() if val else {},
                "monotonic": monotonic(name, params, min(s.r for s in clean), max(s.r for s in clean)),
            }
        except Exception as exc:
            report[name] = {"error": str(exc)}

    usable = {k: v for k, v in report.items() if "error" not in v and v.get("monotonic", True)}
    if not usable:
        raise CalibrationError("No usable calibration model could be fitted: " + json.dumps(report, indent=2))

    def score(entry: dict) -> float:
        m = entry.get("val") or entry.get("train")
        return float(m["rmse"])

    best_name = min(usable, key=lambda name: score(usable[name]))
    best_params = candidates[best_name]
    train_metrics = _eval(best_name, best_params, train)
    val_metrics = _eval(best_name, best_params, val) if val else Metrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return CalibrationProfile(
        model=best_name,
        params=best_params,
        created_at=time.time(),
        samples=clean,
        candidate_models=report,
        train_metrics=train_metrics.to_dict(),
        val_metrics=val_metrics.to_dict(),
        r_min=float(min(s.r for s in clean)),
        r_max=float(max(s.r for s in clean)),
        note=note,
    )


def save_profile(profile: CalibrationProfile, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(profile.to_dict(), indent=2))
    return p


def load_profile(path: str | Path) -> CalibrationProfile:
    p = Path(path)
    if not p.exists():
        raise CalibrationError(f"Calibration profile not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"Calibration profile is not valid JSON: {p}") from exc
    profile = CalibrationProfile.from_dict(data)
    if not profile.samples:
        raise CalibrationError(f"Calibration profile has no samples: {p}")
    return profile


def load_samples(path: str | Path) -> list[Sample]:
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [Sample(**s) for s in data]


def save_samples(samples: list[Sample], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(s) for s in samples], indent=2))
    return p

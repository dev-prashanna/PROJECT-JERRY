"""Temporal smoothing for depth maps and scalar measurements."""

from __future__ import annotations

from collections import deque

import numpy as np


class EmaFilter:
    """Exponential moving average: y_t = a*x_t + (1-a)*y_{t-1}."""

    def __init__(self, alpha: float, shape: tuple[int, ...] | None = None) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: np.ndarray | None = None
        self._shape = shape

    @property
    def initialized(self) -> bool:
        return self._state is not None

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self._state is None or self._state.shape != x.shape:
            self._state = x.astype(np.float32, copy=True)
        else:
            self._state = (self.alpha * x + (1.0 - self.alpha) * self._state).astype(np.float32, copy=False)
        return self._state

    def value(self) -> np.ndarray | None:
        return self._state

    def reset(self) -> None:
        self._state = None


class TemporalMedian:
    """Median over a short sliding window of depth maps (robust to single-frame noise)."""

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._frames: deque[np.ndarray] = deque(maxlen=window)

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.window == 1:
            return x
        self._frames.append(x)
        if len(self._frames) == 1:
            return x
        stack = np.stack([f for f in self._frames], axis=0)
        return np.median(stack, axis=0).astype(np.float32)

    def reset(self) -> None:
        self._frames.clear()


class DepthSmoother:
    """Combines the configured temporal filters. State resets on scene discontinuities."""

    def __init__(self, enabled: bool, alpha: float, median_window: int) -> None:
        self.enabled = enabled
        self.ema = EmaFilter(alpha)
        self.median = TemporalMedian(median_window)

    def apply(self, depth: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return depth
        out = self.ema.apply(depth)
        return self.median.apply(out)

    def reset(self) -> None:
        self.ema.reset()
        self.median.reset()


class ScalarEMA:
    """EMA for single scalar measurements (selected point depth)."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._value: float | None = None

    def apply(self, x: float) -> float:
        if self._value is None or self.alpha >= 1.0:
            self._value = float(x)
        else:
            self._value = self.alpha * float(x) + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self) -> None:
        self._value = None


class RangeEMA:
    """EMA used to stabilize visualization ranges between frames."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self._value: tuple[float, float] | None = None

    def apply(self, lo: float, hi: float) -> tuple[float, float]:
        if self._value is None or hi < lo:
            self._value = (float(lo), float(hi))
        else:
            plo, phi = self._value
            self._value = (self.alpha * lo + (1 - self.alpha) * plo, self.alpha * hi + (1 - self.alpha) * phi)
        return self._value

    def reset(self) -> None:
        self._value = None

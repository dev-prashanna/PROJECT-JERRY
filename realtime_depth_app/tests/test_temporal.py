from __future__ import annotations

import numpy as np
import pytest

from src.temporal_filter import DepthSmoother, EmaFilter, RangeEMA, ScalarEMA, TemporalMedian


def test_ema_first_frame_initializes():
    ema = EmaFilter(0.5)
    out = ema.apply(np.array([2.0, 4.0], dtype=np.float32))
    assert out.tolist() == [2.0, 4.0]
    assert ema.initialized


def test_ema_blends_with_alpha():
    ema = EmaFilter(0.5)
    ema.apply(np.array([0.0], dtype=np.float32))
    out = ema.apply(np.array([4.0], dtype=np.float32))
    assert out[0] == pytest.approx(2.0)


def test_ema_alpha_validation():
    with pytest.raises(ValueError):
        EmaFilter(0.0)
    with pytest.raises(ValueError):
        EmaFilter(1.5)


def test_ema_reset():
    ema = EmaFilter(0.5)
    ema.apply(np.array([1.0], dtype=np.float32))
    ema.reset()
    assert not ema.initialized


def test_temporal_median_window():
    med = TemporalMedian(3)
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32)
    c = np.full((4, 4), 2.0, dtype=np.float32)
    assert med.apply(a).sum() == 0
    med.apply(b)
    out = med.apply(c)
    assert out.mean() == pytest.approx(1.0)


def test_temporal_median_window_one_passthrough():
    med = TemporalMedian(1)
    x = np.arange(5, dtype=np.float32)
    assert med.apply(x) is x


def test_depth_smoother_disabled_passthrough():
    smoother = DepthSmoother(enabled=False, alpha=0.3, median_window=3)
    x = np.ones((8, 8), dtype=np.float32)
    assert smoother.apply(x) is x


def test_depth_smoother_changes_sequence():
    smoother = DepthSmoother(enabled=True, alpha=0.5, median_window=1)
    a = np.zeros((8, 8), dtype=np.float32)
    b = np.full((8, 8), 4.0, dtype=np.float32)
    smoother.apply(a)
    out = smoother.apply(b)
    assert out.mean() == pytest.approx(2.0)


def test_scalar_ema():
    ema = ScalarEMA(0.5)
    assert ema.apply(10.0) == 10.0
    assert ema.apply(0.0) == 5.0
    ema.reset()
    assert ema.value is None


def test_range_ema_converges():
    rng = RangeEMA(0.5)
    lo, hi = rng.apply(0.0, 10.0)
    assert (lo, hi) == (0.0, 10.0)
    lo, hi = rng.apply(2.0, 12.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(11.0)

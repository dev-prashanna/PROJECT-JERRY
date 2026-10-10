from __future__ import annotations

import numpy as np
import pytest

from src.calibration import Sample, fit_profile
from src.config import AppConfig
from src.distance_estimator import DistanceEstimator, PointStore, sample_region


def test_sample_region_median_ignores_nan():
    depth = np.ones((21, 21), dtype=np.float32) * 2.0
    depth[10, 10] = np.nan
    depth[10, 11] = np.inf
    s = sample_region(depth, 10, 10, 5)
    assert s["ok"]
    assert s["median"] == pytest.approx(2.0)
    assert s["n_valid"] == 23
    assert s["n_total"] == 25


def test_sample_region_out_of_bounds():
    depth = np.zeros((10, 10), dtype=np.float32)
    s = sample_region(depth, 50, 50, 3)
    assert not s["ok"]
    assert s["reason"] == "out_of_bounds"


def test_sample_region_all_invalid():
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    s = sample_region(depth, 5, 5, 3)
    assert not s["ok"]
    assert s["reason"] == "no_valid_values"


def test_sample_region_edge_clipping():
    depth = np.arange(100, dtype=np.float32).reshape(10, 10)
    s = sample_region(depth, 0, 0, 5)
    assert s["ok"]
    assert s["n_total"] == 9


def test_relative_mode_returns_raw_value(depth_map):
    cfg = AppConfig()
    est = DistanceEstimator(cfg, profile=None, estimation_type="relative")
    m = est.measure(depth_map, 80, 60)
    assert m.valid
    assert m.metric is None
    assert m.raw == pytest.approx(np.median(depth_map[57:64, 77:84]))
    assert est.mode == "relative"
    assert "a.u." in est.format(m)


def test_calibrated_mode_converts_to_meters(depth_map):
    cfg = AppConfig()
    samples = [Sample(r=1.0, d_true=1.0), Sample(r=2.0, d_true=1.5),
               Sample(r=3.0, d_true=2.0), Sample(r=4.0, d_true=2.5),
               Sample(r=5.0, d_true=3.0), Sample(r=6.0, d_true=3.5)]
    profile = fit_profile(samples, val_fraction=0.0)
    est = DistanceEstimator(cfg, profile=profile, estimation_type="relative")
    m = est.measure(depth_map, 80, 60)
    assert m.metric is not None
    assert est.mode == "calibrated"
    assert "m" in est.format(m)


def test_native_metric_mode(depth_map):
    cfg = AppConfig()
    est = DistanceEstimator(cfg, profile=None, estimation_type="metric")
    m = est.measure(depth_map, 80, 60)
    assert m.metric == pytest.approx(m.raw)
    assert est.mode == "native_metric"


def test_quality_label_is_one_of_heuristics(depth_map):
    cfg = AppConfig()
    est = DistanceEstimator(cfg, None, "relative")
    m = est.measure(depth_map, 80, 60)
    assert m.quality in {"good", "fair", "poor"}
    assert "score" in m.quality_detail


def test_quality_flags_invalid_measurement():
    cfg = AppConfig()
    est = DistanceEstimator(cfg, None, "relative")
    depth = np.full((10, 10), np.nan, dtype=np.float32)
    m = est.measure(depth, 5, 5)
    assert not m.valid
    assert m.quality == "invalid"


def test_point_store_add_remove_clear():
    cfg = AppConfig()
    store = PointStore(cfg)
    pid = store.add(10, 20)
    assert pid is not None
    assert len(store.points) == 1
    store.add(500, 500)
    removed = store.remove_nearest(505, 505)
    assert removed is not None
    assert len(store.points) == 1
    store.clear()
    assert len(store.points) == 0


def test_point_store_respects_max_points(depth_map):
    cfg = AppConfig()
    cfg.measurement.max_points = 2
    store = PointStore(cfg)
    assert store.add(1, 1) is not None
    assert store.add(2, 2) is not None
    assert store.add(3, 3) is None
    est = DistanceEstimator(cfg, None, "relative")
    measurements = store.update(depth_map, est)
    assert len(measurements) == 2
    assert all(m.valid for m in measurements.values())


def test_point_smoothing_tracks_values():
    cfg = AppConfig()
    cfg.measurement.point_smoothing = 0.5
    store = PointStore(cfg)
    store.add(80, 60)
    est = DistanceEstimator(cfg, None, "relative")
    m1 = store.update(np.full((120, 160), 2.0, dtype=np.float32), est)[1]
    m2 = store.update(np.full((120, 160), 4.0, dtype=np.float32), est)[1]
    assert m1.smoothed_raw == pytest.approx(2.0)
    assert m2.smoothed_raw == pytest.approx(3.0)  # EMA: 0.5*4 + 0.5*2

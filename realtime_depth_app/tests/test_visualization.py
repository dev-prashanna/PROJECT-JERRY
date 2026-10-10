from __future__ import annotations

import numpy as np
import pytest

from src.config import AppConfig
from src.distance_estimator import DistanceEstimator, Measurement
from src.visualization import (
    CMAP_ORDER,
    Renderer,
    colorize,
    compute_auto_range,
    draw_legend,
    map_window_point,
    normalize_depth,
)


def test_normalize_depth_clips_to_uint8():
    depth = np.array([[-10.0, 0.0], [5.0, 100.0]], dtype=np.float32)
    out = normalize_depth(depth, 0.0, 10.0)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0 and out[1, 1] == 255


def test_normalize_depth_handles_nan():
    depth = np.array([[np.nan, np.inf], [1.0, 1.0]], dtype=np.float32)
    out = normalize_depth(depth, 0.0, 2.0)
    assert np.isfinite(out).all()


def test_compute_auto_range_orders_values():
    rng = np.random.default_rng(0)
    lo, hi = compute_auto_range(rng.random(1000))
    assert lo < hi


def test_colorize_shape_and_channels():
    depth = np.random.default_rng(0).random((48, 64)).astype(np.float32)
    vis = colorize(depth, "TURBO", 0.0, 1.0)
    assert vis.shape == (48, 64, 3)
    assert vis.dtype == np.uint8


def test_colorize_grayscale_mode():
    depth = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    vis = colorize(depth, "JET", 0.0, 1.0, grayscale=True)
    assert vis.shape == (32, 32, 3)


def test_nearest_is_warmer_than_far_for_inverse_depth():
    """With an inverse-depth convention the near region must map to a higher colormap index."""

    depth = np.zeros((10, 10), dtype=np.float32)
    depth[:, :5] = 1.0   # far -> low index
    depth[:, 5:] = 9.0   # near -> high index
    vis = colorize(depth, "TURBO", 0.0, 10.0)
    near_bgr = vis[0, 7]   # BGR: near (high value) must be red dominant
    far_bgr = vis[0, 2]    # far (low value) must be blue dominant
    assert near_bgr[2] > near_bgr[0] + 50
    assert far_bgr[0] > far_bgr[2] + 50


def test_map_window_point_left_panel():
    assert map_window_point(100, 50, 640) == (100, 50)


def test_map_window_point_depth_panel():
    assert map_window_point(640 + 6 + 100, 50, 640) == (100, 50)


def test_map_window_point_divider_returns_none():
    assert map_window_point(643, 50, 640) is None


def test_map_window_point_beyond_returns_none():
    assert map_window_point(2000, 50, 640) is None


def test_map_window_point_respects_scale():
    # a window scaled by 2: click at (200, 100) is frame pixel (100, 50)
    assert map_window_point(200, 100, 640, scale=2.0) == (100, 50)


def test_renderer_output_dimensions():
    cfg = AppConfig()
    r = Renderer(cfg)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.random.default_rng(0).random((480, 640)).astype(np.float32)
    img = r.render(frame, depth, status_lines=["hello"], perf_lines=["perf"])
    assert img.shape[0] == 480
    assert img.shape[1] == 640 * 2 + 6


def test_renderer_without_depth():
    cfg = AppConfig()
    r = Renderer(cfg)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    img = r.render(frame, None)
    assert img.shape[:2] == (240, 320 * 2 + 6)


def test_renderer_stale_badge():
    cfg = AppConfig()
    r = Renderer(cfg)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.random.default_rng(0).random((240, 320)).astype(np.float32)
    plain = r.render(frame, depth)
    stale = r.render(frame, depth, depth_stale=True, stale_text="STALE")
    assert not np.array_equal(plain, stale)


def test_colormap_cycling_wraps():
    cfg = AppConfig()
    r = Renderer(cfg)
    names = {r.cycle_colormap() for _ in range(len(CMAP_ORDER) + 2)}
    assert len(names) == len(CMAP_ORDER)


def test_legend_draws_without_error():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    draw_legend(img, 0.5, 4.5, "TURBO", "a.u.", "inverse", (10, 10))
    assert img.any()


def test_fixed_range_mode():
    cfg = AppConfig()
    cfg.depth.visualization.range_mode = "fixed"
    cfg.depth.visualization.range_fixed = [1.0, 3.0]
    r = Renderer(cfg)
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    assert r.resolve_range(depth) == (1.0, 3.0)


def test_point_marker_labels_render():
    cfg = AppConfig()
    r = Renderer(cfg)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    depth = np.random.default_rng(0).random((240, 320)).astype(np.float32)
    est = DistanceEstimator(cfg, None, "relative")
    m = est.measure(depth, 100, 120)
    img = r.render(frame, depth, points={1: m})
    assert img.shape[1] == 320 * 2 + 6

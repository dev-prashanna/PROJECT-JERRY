from __future__ import annotations

import numpy as np
import pytest

from src.geometry import CameraIntrinsics, GeometryError, calibrate_checkerboard, save_intrinsics, load_intrinsics


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480,
                            dist=[0.0, 0.0, 0.0, 0.0, 0.0], source="test")


def test_back_projection_roundtrip():
    intr = make_intrinsics()
    u, v, z = 400.0, 300.0, 2.5
    x, y, z2 = intr.back_project(u, v, z)
    u2, v2 = intr.project(x, y, z2)
    assert abs(u - u2) < 1e-9
    assert abs(v - v2) < 1e-9


def test_principal_point_projects_to_center():
    intr = make_intrinsics()
    x, y, z = intr.back_project(intr.cx, intr.cy, 1.0)
    assert abs(x) < 1e-12 and abs(y) < 1e-12 and z == 1.0


def test_euclidean_distance_exceeds_axial():
    intr = make_intrinsics()
    x, y, z = intr.back_project(600.0, 400.0, 2.0)
    r = intr.euclidean_distance(x, y, z)
    assert r >= z
    assert r == pytest.approx(np.sqrt(x * x + y * y + z * z))


def test_nonpositive_depth_rejected():
    intr = make_intrinsics()
    with pytest.raises(GeometryError):
        intr.back_project(10, 10, 0.0)


def test_scaled_intrinsics_preserve_principal_point_relative_position():
    intr = make_intrinsics()
    scaled = intr.scaled(1280, 960)
    assert scaled.fx == pytest.approx(intr.fx * 2)
    assert scaled.fy == pytest.approx(intr.fy * 2)
    assert scaled.cx == pytest.approx(intr.cx * 2)
    assert scaled.width == 1280


def _synthetic_checkerboard_views(pattern=(9, 6), square=0.025, n_views=6, size=(640, 480)):
    """Render checkerboard images from known poses so calibration can be validated.

    The pattern image contains (cols+1) x (rows+1) squares; inner corner (i, j) sits at
    object coordinates (i*square, j*square) with the origin at the first inner corner.
    """
    import cv2

    cols, rows = pattern
    cell = 40
    pat = np.zeros(((rows + 1) * cell, (cols + 1) * cell), dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                pat[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = 255

    k = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    # pattern pixel -> object plane (origin = first inner corner at pattern pixel (cell, cell))
    scale = square / cell
    pix_to_obj = np.array([[scale, 0.0, -square], [0.0, scale, -square], [0.0, 0.0, 1.0]])

    images = []
    rng = np.random.default_rng(1)
    cx_obj = (cols - 1) * square / 2.0
    cy_obj = (rows - 1) * square / 2.0
    for i in range(n_views):
        rvec = np.array([[rng.uniform(-0.3, 0.3)], [rng.uniform(-0.3, 0.3)], [rng.uniform(-0.15, 0.15)]])
        tvec = np.array([[-cx_obj], [-cy_obj], [0.55 + 0.08 * i]])
        rot, _ = cv2.Rodrigues(rvec)
        plane_to_cam = np.hstack([rot[:, :2], tvec])
        h = k @ plane_to_cam @ pix_to_obj
        h = h / h[2, 2]
        warped = cv2.warpPerspective(pat, h, (size[0], size[1]), borderValue=0)
        images.append(cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR))
    return images, k


def test_checkerboard_calibration_recovers_intrinsics():
    images, true_k = _synthetic_checkerboard_views()
    intr, errors = calibrate_checkerboard(images, pattern=(9, 6), square_size_m=0.025)
    assert intr.rms is not None and intr.rms < 2.0
    assert abs(intr.fx - true_k[0, 0]) / true_k[0, 0] < 0.05
    assert abs(intr.fy - true_k[1, 1]) / true_k[1, 1] < 0.05
    assert max(errors) < 2.0


def test_too_few_views_rejected():
    images, _ = _synthetic_checkerboard_views(n_views=2)
    with pytest.raises(GeometryError):
        calibrate_checkerboard(images, pattern=(9, 6), square_size_m=0.025)


def test_intrinsics_persistence(tmp_path):
    intr = make_intrinsics()
    path = save_intrinsics(intr, tmp_path / "intrinsics.json")
    loaded = load_intrinsics(path)
    assert loaded.fx == intr.fx and loaded.cy == intr.cy
    assert loaded.dist == intr.dist


def test_load_missing_intrinsics(tmp_path):
    with pytest.raises(GeometryError, match="not found"):
        load_intrinsics(tmp_path / "missing.json")

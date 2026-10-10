from __future__ import annotations

import numpy as np
import pytest

from src.anchor import AnchorError, ArucoAnchor
from src.geometry import CameraIntrinsics


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480,
                            dist=[0.0, 0.0, 0.0, 0.0, 0.0], source="test")


def render_marker_view(anchor: ArucoAnchor, intr: CameraIntrinsics, rvec, tvec, marker_px: int = 240) -> np.ndarray:
    """Warp a generated ArUco marker into a synthetic camera view at a known pose."""
    import cv2

    marker = cv2.aruco.generateImageMarker(anchor.dictionary, 3, marker_px)
    half = anchor.marker_length_m / 2.0
    # image corner (tl, tr, br, bl) -> marker-frame corner in the camera convention
    # (camera x right, y down, z forward): top-left of the marker is (-half, -half).
    # ArucoPose in src/anchor.py pairs detection order with the equivalent y-up frame,
    # which is a 180 degree in-plane rotation of the same square.
    obj = np.array([
        [-half, -half, 0.0],
        [half, -half, 0.0],
        [half, half, 0.0],
        [-half, half, 0.0],
    ], dtype=np.float32)
    # add a white quiet zone around the marker: detectors require one
    pad = marker_px // 4
    padded = np.full((marker_px + 2 * pad, marker_px + 2 * pad), 255, dtype=np.uint8)
    padded[pad : pad + marker_px, pad : pad + marker_px] = marker
    marker = padded
    img_px = np.array([
        [float(pad), float(pad)],
        [float(pad + marker_px - 1), float(pad)],
        [float(pad + marker_px - 1), float(pad + marker_px - 1)],
        [float(pad), float(pad + marker_px - 1)],
    ], dtype=np.float32)
    rmat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    transformed = (rmat @ obj.T).T + np.asarray(tvec, dtype=np.float64).reshape(3)
    k = intr.matrix()
    proj = (k @ transformed.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    h, _ = cv2.findHomography(img_px, proj.astype(np.float32))
    warped = cv2.warpPerspective(marker, h, (intr.width, intr.height), borderValue=0)
    valid = cv2.warpPerspective(np.full_like(marker, 255), h, (intr.width, intr.height)) > 0
    canvas = np.full((intr.height, intr.width), 200, dtype=np.uint8)
    canvas[valid] = warped[valid]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_invalid_marker_length():
    with pytest.raises(AnchorError):
        ArucoAnchor(0.0)


def test_no_marker_returns_not_detected():
    anchor = ArucoAnchor(0.12)
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    res = anchor.estimate(img, make_intrinsics())
    assert not res.detected


def test_pose_recovers_known_distance():
    """Synthetic marker at a known pose must be recovered to within a few percent."""
    anchor = ArucoAnchor(0.12)
    intr = make_intrinsics()
    rvec = (0.1, -0.15, 0.05)
    tvec = (0.05, -0.02, 0.60)
    img = render_marker_view(anchor, intr, rvec, tvec)
    res = anchor.estimate(img, intr)
    assert res.detected
    assert res.reprojection_err_px < 2.0
    expected_z = 0.60
    assert abs(res.z_axial_m - expected_z) / expected_z < 0.03
    expected_dist = float(np.linalg.norm(tvec))
    assert abs(res.distance_m - expected_dist) / expected_dist < 0.05
    # axial depth is smaller than the Euclidean distance
    assert res.z_axial_m < res.distance_m

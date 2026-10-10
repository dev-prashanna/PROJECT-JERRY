"""Camera intrinsics, distortion handling and 3D back-projection.

Note: intrinsic calibration does NOT resolve monocular scale ambiguity. Z below is axial
(camera-frame) depth; the Euclidean distance to the point is sqrt(X^2 + Y^2 + Z^2).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


class GeometryError(ValueError):
    pass


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    dist: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    rms: float | None = None
    source: str = "unknown"

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def scaled(self, new_width: int, new_height: int) -> "CameraIntrinsics":
        """Rescale intrinsics when the processed image geometry differs from calibration."""
        sx = new_width / self.width
        sy = new_height / self.height
        if sx <= 0 or sy <= 0:
            raise GeometryError("Invalid target size for intrinsic scaling")
        dist = list(self.dist)
        # radial coefficients are dimensionless; tangential and thin-prism terms scale ~ with aspect
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=int(new_width),
            height=int(new_height),
            dist=dist,
            rms=self.rms,
            source=self.source,
        )

    def back_project(self, u: float, v: float, z: float) -> tuple[float, float, float]:
        """Image point (u, v) + axial depth Z -> camera-frame (X, Y, Z)."""
        if z <= 0:
            raise GeometryError(f"Axial depth must be > 0, got {z}")
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return float(x), float(y), float(z)

    @staticmethod
    def euclidean_distance(x: float, y: float, z: float) -> float:
        return float(math.sqrt(x * x + y * y + z * z))

    def project(self, x: float, y: float, z: float) -> tuple[float, float]:
        if z <= 0:
            raise GeometryError(f"Z must be > 0, got {z}")
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return float(u), float(v)

    def undistort(self, image: np.ndarray, alpha: float = 0.0) -> tuple[np.ndarray, "CameraIntrinsics"]:
        h, w = image.shape[:2]
        k = self.matrix()
        d = np.asarray(self.dist, dtype=np.float64)
        new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, (w, h), alpha)
        undistorted = cv2.undistort(image, k, d, None, new_k)
        new_intr = CameraIntrinsics(
            fx=float(new_k[0, 0]),
            fy=float(new_k[1, 1]),
            cx=float(new_k[0, 2]),
            cy=float(new_k[1, 2]),
            width=w,
            height=h,
            dist=[0.0, 0.0, 0.0, 0.0, 0.0],
            rms=self.rms,
            source=f"{self.source}+undistort",
        )
        return undistorted, new_intr

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dist"] = [float(x) for x in self.dist]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "CameraIntrinsics":
        try:
            return cls(
                fx=float(data["fx"]),
                fy=float(data["fy"]),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
                width=int(data["width"]),
                height=int(data["height"]),
                dist=[float(x) for x in data.get("dist", [0.0, 0.0, 0.0, 0.0, 0.0])],
                rms=data.get("rms"),
                source=str(data.get("source", "unknown")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GeometryError(f"Malformed intrinsics data: {exc}") from exc


def save_intrinsics(intr: CameraIntrinsics, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(intr.to_dict(), indent=2))
    return p


def load_intrinsics(path: str | Path) -> CameraIntrinsics:
    p = Path(path)
    if not p.exists():
        raise GeometryError(f"Intrinsics file not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise GeometryError(f"Intrinsics file is not valid JSON: {p}") from exc
    return CameraIntrinsics.from_dict(data)


class Undistorter:
    """Precomputes remap tables so frames and depth maps stay pixel-aligned."""

    def __init__(self, intrinsics: CameraIntrinsics, alpha: float = 0.0) -> None:
        self.intrinsics = intrinsics
        k = intrinsics.matrix()
        d = np.asarray(intrinsics.dist, dtype=np.float64)
        size = (intrinsics.width, intrinsics.height)
        self.new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, size, alpha)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(k, d, None, self.new_k, size, cv2.CV_32FC1)
        self.new_intrinsics = CameraIntrinsics(
            fx=float(self.new_k[0, 0]),
            fy=float(self.new_k[1, 1]),
            cx=float(self.new_k[0, 2]),
            cy=float(self.new_k[1, 2]),
            width=intrinsics.width,
            height=intrinsics.height,
            dist=[0.0, 0.0, 0.0, 0.0, 0.0],
            rms=intrinsics.rms,
            source=f"{intrinsics.source}+undistort",
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)

    def apply_depth(self, depth: np.ndarray) -> np.ndarray:
        # nearest-neighbour keeps measured values intact instead of blending them
        return cv2.remap(depth, self.map1, self.map2, cv2.INTER_NEAREST)


def detect_checkerboard(image: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Detect inner corners; uses the more robust SB detector when available."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            found, corners = cv2.findChessboardCornersSB(gray, pattern)
            if found:
                return corners.reshape(-1, 1, 2)
        except cv2.error:
            pass
    found, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)


def calibrate_checkerboard(
    images: list[np.ndarray],
    pattern: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> tuple[CameraIntrinsics, list[float]]:
    """Standard OpenCV intrinsic calibration. Returns intrinsics + per-view reprojection errors."""
    if len(images) < 3:
        raise GeometryError(f"Need at least 3 checkerboard views, got {len(images)}")
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_size_m

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    for img in images:
        h, w = img.shape[:2]
        image_size = (w, h)
        corners = detect_checkerboard(img, pattern)
        if corners is None:
            continue
        object_points.append(objp)
        image_points.append(corners)

    if len(object_points) < 3:
        raise GeometryError(f"Checkerboard detected in only {len(object_points)} view(s); need >= 3")

    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)
    errors: list[float] = []
    for obj, img_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
        diff = projected.reshape(-1, 2) - np.asarray(img_pts).reshape(-1, 2)
        errors.append(float(np.mean(np.linalg.norm(diff, axis=-1))))

    intr = CameraIntrinsics(
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
        width=int(image_size[0]),
        height=int(image_size[1]),
        dist=[float(x) for x in np.asarray(dist).reshape(-1)],
        rms=float(rms),
        source="checkerboard",
    )
    return intr, errors

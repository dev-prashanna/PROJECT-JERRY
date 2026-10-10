"""Optional ArUco scale anchor (Approach C).

A marker of known physical size gives a *geometrically measured* distance (solvePnP with
intrinsics). That value is independent of the depth network and is used only to validate -
or, if enabled, to locally anchor - the monocular estimate. It never turns the rest of the
scene metric by itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import CameraIntrinsics


class AnchorError(ValueError):
    pass


@dataclass
class AnchorResult:
    detected: bool
    marker_id: int = -1
    center_px: tuple[int, int] = (0, 0)
    distance_m: float = float("nan")   # Euclidean camera -> marker center
    z_axial_m: float = float("nan")    # axial depth of marker center
    reprojection_err_px: float = float("nan")
    rvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tvec: tuple[float, float, float] = (0.0, 0.0, 0.0)


class ArucoAnchor:
    def __init__(self, marker_length_m: float, dictionary: str = "DICT_4X4_50") -> None:
        if marker_length_m <= 0:
            raise AnchorError("marker_length_m must be > 0")
        if not hasattr(cv2, "aruco"):
            raise AnchorError("This OpenCV build has no aruco module")
        if not hasattr(cv2.aruco, dictionary):
            raise AnchorError(f"Unknown aruco dictionary: {dictionary}")
        self.marker_length_m = float(marker_length_m)
        dict_id = getattr(cv2.aruco, dictionary)
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        else:  # OpenCV < 4.7
            self.dictionary = cv2.aruco.Dictionary_get(dict_id)
        self._detector = cv2.aruco.ArucoDetector(self.dictionary, cv2.aruco.DetectorParameters())
        half = marker_length_m / 2.0
        self._object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

    def detect_corners(self, image: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None, None
        return corners, ids

    def estimate(self, image: np.ndarray, intrinsics: CameraIntrinsics) -> AnchorResult:
        if intrinsics is None:
            raise AnchorError("Intrinsics are required for ArUco pose estimation")
        corners, ids = self.detect_corners(image)
        if corners is None:
            return AnchorResult(detected=False)

        idx = 0
        best = None
        # use the marker closest to the image center if several are visible
        h, w = image.shape[:2]
        for i, c in enumerate(corners):
            pts = c.reshape(-1, 2)
            center = pts.mean(axis=0)
            d = math.hypot(center[0] - w / 2, center[1] - h / 2)
            if best is None or d < best:
                best, idx = d, i

        pts = corners[idx].reshape(-1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            self._object_points,
            pts,
            intrinsics.matrix(),
            np.asarray(intrinsics.dist, dtype=np.float64),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return AnchorResult(detected=False)

        projected, _ = cv2.projectPoints(self._object_points, rvec, tvec, intrinsics.matrix(), np.asarray(intrinsics.dist))
        err = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - pts, axis=-1)))
        t = tvec.reshape(3)
        center_px = pts.mean(axis=0)
        ids_flat = np.asarray(ids).reshape(-1)
        return AnchorResult(
            detected=True,
            marker_id=int(ids_flat[idx]),
            center_px=(int(center_px[0]), int(center_px[1])),
            distance_m=float(np.linalg.norm(t)),
            z_axial_m=float(np.ravel(t)[2]),
            reprojection_err_px=err,
            rvec=tuple(float(v) for v in np.ravel(rvec)),
            tvec=tuple(float(v) for v in np.ravel(t)),
        )

#!/usr/bin/env python3
"""OpenCV intrinsic calibration from checkerboard views.

Live mode (default): a window shows the camera; the script automatically records views
where the checkerboard is detected until enough diverse views are collected.

    python3 tools/calibrate_intrinsics.py --pattern 9x6 --square 0.025

Image mode:
    python3 tools/calibrate_intrinsics.py --images path/to/checkerboard_photos
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from src.camera import CameraSource  # noqa: E402
from src.config import ConfigError, load_config  # noqa: E402
from src.geometry import (  # noqa: E402
    GeometryError,
    calibrate_checkerboard,
    detect_checkerboard,
    save_intrinsics,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera intrinsic calibration (checkerboard)")
    p.add_argument("--config", default=None)
    p.add_argument("--pattern", default="9x6", help="inner corners, e.g. 9x6")
    p.add_argument("--square", type=float, default=0.025, help="square size in meters")
    p.add_argument("--count", type=int, default=20, help="number of views to capture")
    p.add_argument("--images", default=None, help="directory of images instead of the camera")
    p.add_argument("--out", default="calibration_data/intrinsics.json")
    p.add_argument("--window", action="store_true", default=True)
    return p.parse_args()


def _load_images(directory: Path) -> list[np.ndarray]:
    images = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            img = cv2.imread(str(path))
            if img is not None:
                images.append(img)
    return images


def live_capture(cfg, pattern: tuple[int, int], needed: int) -> list[np.ndarray]:
    source = CameraSource(cfg)
    window = "checkerboard capture (q = quit)"
    collected: list[np.ndarray] = []
    seen: list[np.ndarray] = []
    last_print = 0.0
    try:
        while len(collected) < needed:
            frame = None
            for _ in range(30):
                frame = source.read()
                if frame is not None:
                    break
                time.sleep(0.05)
            if frame is None:
                print("camera read failed", file=sys.stderr)
                break
            img = frame.image
            corners = detect_checkerboard(img, pattern)
            display = img.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, pattern, corners, True)
                # keep views that are spatially different from the ones already stored
                center = corners.reshape(-1, 2).mean(axis=0)
                angle = float(np.mean(corners.reshape(-1, 2)[:, 0]))
                distinct = all(
                    np.linalg.norm(center - s[0]) > 40 or abs(angle - s[1]) > 12
                    for s in seen
                )
                if distinct:
                    collected.append(img.copy())
                    seen.append((center, angle))
            status = f"views {len(collected)}/{needed}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if corners is not None else (0, 0, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            now = time.time()
            if now - last_print > 1.0:
                print(f"\r{status}", end="", flush=True)
                last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        cv2.destroyAllWindows()
        print()
    return collected


def main() -> int:
    args = parse_args()
    try:
        cols, rows = args.pattern.lower().split("x")
        pattern = (int(cols), int(rows))
    except ValueError:
        print(f"bad --pattern {args.pattern}, expected e.g. 9x6", file=sys.stderr)
        return 2

    if args.images:
        images = _load_images(Path(args.images))
        if len(images) < 3:
            print(f"found {len(images)} images in {args.images}, need >= 3", file=sys.stderr)
            return 2
    else:
        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            print(f"config error: {exc}", file=sys.stderr)
            return 2
        images = live_capture(cfg, pattern, args.count)

    if len(images) < 3:
        print(f"only {len(images)} usable views collected; need >= 3", file=sys.stderr)
        return 1

    try:
        intrinsics, errors = calibrate_checkerboard(images, pattern, args.square)
    except GeometryError as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1

    out = APP_ROOT / args.out
    save_intrinsics(intrinsics, out)
    print(f"\nviews: {len(images)} | RMS reprojection: {intrinsics.rms:.3f} px")
    print(f"per-view mean error: {np.mean(errors):.3f} px (max {np.max(errors):.3f} px)")
    print(f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} cx={intrinsics.cx:.1f} cy={intrinsics.cy:.1f} "
          f"dist={[round(float(d), 4) for d in intrinsics.dist]}")
    print(f"saved -> {out}")
    print("\nNote: intrinsics give geometry (back-projection), NOT metric scale for the depth model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

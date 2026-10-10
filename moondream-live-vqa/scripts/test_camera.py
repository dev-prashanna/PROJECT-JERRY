#!/usr/bin/env python3
"""Phase 3 — camera acquisition smoke test.

Opens the camera through :class:`src.camera.CameraSource`, measures the real
frame rate, verifies that snapshots are independent copies, exercises a
restart, and confirms the device is released on shutdown.

Usage::

    python scripts/test_camera.py                  # index 0, MJPG 640x480
    python scripts/test_camera.py --index 1 --width 1280 --height 720
    python scripts/test_camera.py --seconds 5 --save-snapshot
    python scripts/test_camera.py --list            # only list cameras
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import build_config  # noqa: E402
from src.camera.camera import CameraSource, CameraStatus, discover_cameras  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, help="camera device index")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--fourcc", help="FOURCC code, e.g. MJPG or YUYV")
    parser.add_argument("--seconds", type=float, default=4.0, help="capture duration")
    parser.add_argument("--list", action="store_true", help="list cameras and exit")
    parser.add_argument("--save-snapshot", action="store_true", help="save latest frame")
    parser.add_argument("--no-restart", action="store_true", help="skip restart test")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)

    cameras = discover_cameras(max_index=6)
    print("Discovered cameras:")
    if not cameras:
        print("  (none)")
    for info in cameras:
        print(f"  {info.label()}")

    if args.list:
        return 0 if cameras else 1

    config = build_config(
        camera_index=args.index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        load_env=True,
    )

    events = []
    camera = CameraSource(config.camera, on_event=events.append)

    print(f"\nStarting camera {config.camera.device_index} "
          f"({config.camera.width}x{config.camera.height} @ {config.camera.fps} "
          f"fps, fourcc={config.camera.fourcc or 'default'})")
    camera.start()

    deadline = time.time() + args.seconds
    last_id = -1
    copies_checked = 0
    while time.time() < deadline:
        time.sleep(0.25)
        snap = camera.get_latest_frame(copy=True)
        if snap is None:
            continue
        if snap.frame_id != last_id:
            last_id = snap.frame_id
        # Verify snapshot independence (mutation must not leak back).
        if copies_checked < 3:
            before = snap.frame_id
            snap.frame[0, 0] = (255, 0, 255) if snap.frame.ndim == 3 else 255
            fresh = camera.get_latest_frame(copy=True)
            if fresh is not None and fresh.frame_id == before:
                assert tuple(fresh.frame[0, 0]) != tuple(snap.frame[0, 0]), (
                    "buffer leaked a shared frame!"
                )
                copies_checked += 1

    streaming = camera.status is CameraStatus.STREAMING
    fps = camera.actual_fps
    width, height = camera.resolution
    latest = camera.get_latest_frame()
    print("\n--- capture summary ---")
    print(f"status        : {camera.status.value} ({camera.status_message})")
    print(f"resolution    : {width}x{height}")
    print(f"measured fps  : {fps:.1f}")
    print(f"frames read   : {camera.frames_read}")
    print(f"latest frame  : {latest.describe() if latest else 'none'}")
    print(f"backend       : {camera.backend_name}")
    print(f"copy isolation: {copies_checked}/3 checks passed")

    exit_code = 0
    if not streaming or latest is None:
        print("FAILED: camera did not reach the streaming state")
        exit_code = 1

    if args.save_snapshot and latest is not None:
        import cv2

        out = ROOT / "outputs" / "camera_snapshot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), latest.frame)
        print(f"snapshot saved -> {out}")

    if not args.no_restart:
        print("\nRestart test...")
        camera.restart()
        time.sleep(2.0)
        after = camera.get_latest_frame()
        print(f"after restart : {after.describe() if after else 'none'}")
        if after is None:
            print("FAILED: no frames after restart")
            exit_code = 1

    print("\nStopping camera (device release)...")
    camera.stop()
    print(f"final status  : {camera.status.value}")
    print(f"buffer empty  : {camera.buffer.is_empty}")
    print("\nEvents:")
    for event in events:
        print(f"  {event.status.value:<13} {event.message}")
    print("\nOK" if exit_code == 0 else "\nFAILED")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

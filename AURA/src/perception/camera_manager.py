"""Camera capture service.

* Discovers and opens a camera once; a capture thread keeps it warm.
* Publishes only the newest frame into a :class:`FrameBuffer`.
* Snapshots are immutable, id- and timestamp-tagged copies.
* Disconnects are detected and (optionally) retried without crashing.
* Resources are always released on ``close()``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from config.settings import CameraSettings, get_settings
from src.perception.frame_buffer import Frame, FrameBuffer
from src.utils.exceptions import CameraError, StaleFrameError
from src.utils.logger import get_logger

log = get_logger("camera")


class CameraManager:
    def __init__(self, settings: Optional[CameraSettings] = None):
        self.settings = settings or get_settings().camera
        self.buffer = FrameBuffer(max_age_s=self.settings.max_frame_age_s)
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._status = "stopped"
        self._error: Optional[str] = None
        self._frames_seen = 0
        self._fps = 0.0
        self._opened = False
        self._blank_streak = 0
        self._blank_warning: Optional[str] = None

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if not self.settings.enabled:
            self._status = "disabled"
            log.info("camera disabled by configuration")
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._open_locked()
            self._thread = threading.Thread(
                target=self._capture_loop, name="camera-capture", daemon=True
            )
            self._thread.start()
        log.info("camera started (index=%s)", self.settings.device_index)

    def _open_locked(self) -> None:
        import cv2

        self._release_locked()
        index = self.settings.device_index
        cap = cv2.VideoCapture(index, self._cv_backend())
        if not cap.isOpened():
            cap.release()
            self._status = "error"
            self._error = f"camera index {index} could not be opened"
            raise CameraError(self._error)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        cap.set(cv2.CAP_PROP_FPS, self.settings.fps)
        if self.settings.fourcc and len(self.settings.fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.settings.fourcc))
        # Keep the driver-side queue shallow so we always read fresh frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            self._status = "error"
            self._error = f"camera index {index} opened but delivered no frame"
            raise CameraError(self._error)

        self._cap = cap
        self._opened = True
        self._error = None
        self._status = "running"

    def _cv_backend(self) -> int:
        import cv2

        backend = (self.settings.backend or "auto").lower()
        if backend == "v4l2":
            return cv2.CAP_V4L2
        if backend == "ffmpeg":
            return cv2.CAP_FFMPEG
        return cv2.CAP_ANY

    def _capture_loop(self) -> None:
        import cv2

        last_fps_t = time.perf_counter()
        frames_in_window = 0
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                if not self._try_reopen():
                    time.sleep(self.settings.reconnect_interval_s)
                    continue
                cap = self._cap
            try:
                ok, image = cap.read()
            except Exception as exc:  # noqa: BLE001
                ok, image = False, None
                log.warning("camera read error: %s", exc)
            if not ok or image is None:
                self._status = "reconnecting"
                self._error = "frame read failed"
                if not self.settings.auto_reconnect:
                    self._status = "error"
                    break
                self._reopen()
                continue

            self.buffer.put(image, source=f"camera{self.settings.device_index}", fps=self.settings.fps)
            self._frames_seen += 1
            self._check_blank(image)
            frames_in_window += 1
            now = time.perf_counter()
            if now - last_fps_t >= 1.0:
                self._fps = frames_in_window / (now - last_fps_t)
                frames_in_window = 0
                last_fps_t = now
            self._status = "running"

    def _check_blank(self, image) -> None:
        """Detect a uniformly black frame (covered lens / privacy shutter)."""
        try:
            import numpy as np

            sample = image[::8, ::8]
            if float(np.max(sample)) - float(np.min(sample)) <= 2:
                self._blank_streak += 1
                if self._blank_streak == 15:
                    self._blank_warning = (
                        "camera frames are uniformly black — the lens may be covered "
                        "or the hardware privacy switch may be engaged"
                    )
                    log.warning(self._blank_warning)
            else:
                if self._blank_streak and self._blank_streak >= 15:
                    log.info("camera image restored (frames are no longer blank)")
                self._blank_streak = 0
                self._blank_warning = None
        except Exception:  # pragma: no cover - diagnostics must never break capture
            pass

    def _try_reopen(self) -> bool:
        with self._lock:
            if self._stop.is_set():
                return False
            try:
                self._open_locked()
                log.info("camera reopened")
                return True
            except CameraError as exc:
                self._error = str(exc)
                return False

    def _reopen(self) -> None:
        with self._lock:
            self._release_locked()
        time.sleep(self.settings.reconnect_interval_s)
        self._try_reopen()

    def _release_locked(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # pragma: no cover
                pass
            self._cap = None
        self._opened = False

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        with self._lock:
            self._release_locked()
        self._status = "stopped"
        log.info("camera released")

    def __enter__(self) -> "CameraManager":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- snapshots -------------------------------------------------------
    def snapshot(self, max_age_s: Optional[float] = None, wait_s: float = 2.0) -> Frame:
        """Return a stable, immutable frame, waiting briefly for a fresh one."""
        if self._status in {"disabled", "stopped"} and self.buffer.empty:
            raise CameraError(
                "camera is not running",
                hint="Start the camera from the dashboard or set CAMERA_ENABLED=1.",
            )
        deadline = time.perf_counter() + max(0.0, wait_s)
        while True:
            frame = self.buffer.latest_fresh(max_age_s or self.settings.max_frame_age_s)
            if frame is not None:
                return frame
            if time.perf_counter() >= deadline:
                break
            time.sleep(0.03)
        if self.buffer.empty:
            raise CameraError(
                "no frame available from the camera",
                hint="Check that no other application is using the webcam.",
            )
        raise StaleFrameError(
            f"newest frame is older than {max_age_s or self.settings.max_frame_age_s}s"
        )

    # -- status ----------------------------------------------------------
    @property
    def status(self) -> str:
        return self._status

    @property
    def running(self) -> bool:
        return self._status == "running" and self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    def info(self) -> dict[str, Any]:
        frame = self.buffer.get(copy=False)
        return {
            "status": self._status,
            "running": self.running,
            "device_index": self.settings.device_index,
            "fps": round(self._fps, 1),
            "resolution": (
                f"{frame.width}x{frame.height}" if frame else f"{self.settings.width}x{self.settings.height}"
            ),
            "frames_seen": self._frames_seen,
            "last_frame_tag": frame.tag if frame else None,
            "last_frame_age_s": round(frame.age_s, 2) if frame else None,
            "error": self._error,
            "blank_warning": self._blank_warning,
            "buffer": self.buffer.stats,
        }

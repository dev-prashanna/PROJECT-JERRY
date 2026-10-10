"""OpenCV camera acquisition worker.

A dedicated background thread owns the ``cv2.VideoCapture`` object: it reads
frames as fast as the driver delivers them, publishes the newest one into a
:class:`~src.camera.frame_buffer.LatestFrameBuffer`, and reports status
transitions through callbacks.  Nothing here touches Qt, so the module can be
unit tested headlessly.

Key behaviours
--------------
* automatic device discovery and friendly names (via sysfs on Linux)
* explicit FourCC / resolution / FPS negotiation (``MJPG`` avoids the ~5 FPS
  collapse of uncompressed ``YUYV`` on UVC webcams)
* read-failure detection, disconnection handling and automatic reconnection
* clean release of the device on shutdown
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import cv2

from config.settings import CameraConfig
from src.camera.frame_buffer import FrameSnapshot, LatestFrameBuffer
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

_IS_LINUX = os.name == "posix" and Path("/proc/self").exists() and Path("/dev").exists()


class CameraStatus(str, Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class CameraEvent:
    """Payload delivered to ``on_event`` callbacks (from the camera thread)."""

    status: CameraStatus
    message: str = ""
    camera_index: int = -1
    width: int = 0
    height: int = 0
    fps: float = 0.0
    timestamp: float = 0.0


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    path: Optional[str] = None

    def label(self) -> str:
        return f"Camera {self.index} — {self.name}"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _api_preference(backend: str) -> int:
    backend = (backend or "auto").lower()
    if backend == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    if backend == "any":
        return cv2.CAP_ANY
    if _IS_LINUX and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def _device_name(index: int) -> str:
    """Human readable device name (Linux sysfs, best effort)."""
    sysfs = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        if sysfs.is_file():
            return sysfs.read_text(encoding="utf-8").strip() or f"device {index}"
    except OSError:
        pass
    return f"device {index}"


@contextmanager
def _silence_native_stderr():
    """Hide C-level stderr (OpenCV probe warnings) for the duration of a block.

    ``cv2.VideoCapture`` prints ``VIDIOC_* failed`` warnings straight from C++
    while probing device nodes that are not video capture devices.  Those are
    expected during discovery and would only add noise; real failures are
    still reported through return values and the event callbacks.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def probe_camera(
    index: int,
    *,
    backend: str = "auto",
    timeout_frames: int = 3,
    configure: bool = False,
    quiet: bool = True,
) -> Optional[CameraInfo]:
    """Try to open ``index`` and grab a frame.  Returns ``None`` on failure."""
    if index < 0:
        return None

    def _attempt() -> Optional[CameraInfo]:
        cap = cv2.VideoCapture(index, _api_preference(backend))
        if not cap.isOpened():
            cap.release()
            return None
        try:
            if configure:
                _apply_capture_settings(cap, CameraConfig(device_index=index))
            for _ in range(max(1, timeout_frames)):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    return CameraInfo(index=index, name=_device_name(index))
                time.sleep(0.05)
            return None
        finally:
            cap.release()

    if quiet:
        with _silence_native_stderr():
            return _attempt()
    return _attempt()


def discover_cameras(
    max_index: int = 6,
    *,
    backend: str = "auto",
    active_index: Optional[int] = None,
) -> List[CameraInfo]:
    """Probe ``0..max_index-1`` and return every camera that yields a frame."""
    found: List[CameraInfo] = []
    # Prefer the physical device nodes so ordering matches the OS.
    indices: List[int] = []
    try:
        nodes = sorted(
            int(p.name.replace("video", ""))
            for p in Path("/dev").glob("video*")
            if p.name[5:].isdigit()
        )
        indices.extend(nodes)
    except OSError:
        pass
    if not indices:
        # No device nodes (macOS/Windows): fall back to index probing.
        indices.extend(range(max_index))

    seen = set()
    for index in indices:
        if index in seen or index == active_index or index > max_index + 16:
            continue
        seen.add(index)
        info = probe_camera(index, backend=backend)
        if info is not None:
            found.append(info)
    return found


# ---------------------------------------------------------------------------
# Capture settings
# ---------------------------------------------------------------------------

def _apply_capture_settings(cap: cv2.VideoCapture, config: CameraConfig) -> None:
    """Negotiate FourCC / resolution / FPS and log what the driver accepted."""
    if config.fourcc and len(config.fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc))
    if config.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    if config.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    if config.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, config.fps)
    # Small internal buffer: we only ever want the newest frame.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # pragma: no cover - not supported by every backend
        pass


def _describe_backend(cap: cv2.VideoCapture) -> str:
    try:
        return cap.getBackendName()
    except Exception:  # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# Camera worker
# ---------------------------------------------------------------------------

class CameraSource:
    """Background camera capture with a thread-safe latest-frame buffer."""

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        on_event: Optional[Callable[[CameraEvent], None]] = None,
    ) -> None:
        self.config = config or CameraConfig()
        self._on_event = on_event
        self.buffer = LatestFrameBuffer()

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._state_lock = threading.Lock()

        self._status = CameraStatus.IDLE
        self._status_message = ""
        self._width = 0
        self._height = 0
        self._requested_fps = float(self.config.fps)
        self._actual_fps = 0.0
        self._backend_name = ""
        self._device_name = ""
        self._frame_intervals: List[float] = []
        self._last_error: Optional[str] = None
        self._frames_read = 0

    # -- public properties --------------------------------------------------
    @property
    def status(self) -> CameraStatus:
        with self._state_lock:
            return self._status

    @property
    def status_message(self) -> str:
        with self._state_lock:
            return self._status_message

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_streaming(self) -> bool:
        return self.status is CameraStatus.STREAMING

    @property
    def actual_fps(self) -> float:
        with self._state_lock:
            return self._actual_fps

    @property
    def resolution(self) -> tuple[int, int]:
        with self._state_lock:
            return (self._width, self._height)

    @property
    def backend_name(self) -> str:
        with self._state_lock:
            return self._backend_name

    @property
    def last_error(self) -> Optional[str]:
        with self._state_lock:
            return self._last_error

    @property
    def frames_read(self) -> int:
        return self._frames_read

    @property
    def active_index(self) -> int:
        return self.config.device_index

    @property
    def device_name(self) -> str:
        """Friendly device name (Linux sysfs), falling back to an index label."""
        with self._state_lock:
            name = getattr(self, "_device_name", "")
        return name or f"device {self.config.device_index}"

    # -- public control -----------------------------------------------------
    def start(self) -> bool:
        """Start (or restart) the capture thread.  Returns ``False`` if it is
        already running."""
        if self.is_running:
            log.info("Camera already running on index %s", self.config.device_index)
            return False
        self._stop.clear()
        self.buffer.clear()
        self._set_status(CameraStatus.CONNECTING, f"opening camera {self.config.device_index}")
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.config.device_index}",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the worker to stop and wait for it to release the device."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.warning("Camera thread did not stop within %.1fs", timeout)
        self._thread = None
        if self.status is not CameraStatus.STOPPED:
            self._set_status(CameraStatus.STOPPED, "camera stopped")
        self.buffer.clear()

    def restart(self) -> bool:
        self.stop()
        return self.start()

    # -- public frame access ------------------------------------------------
    def get_latest_frame(self, copy: bool = True) -> Optional[FrameSnapshot]:
        """Thread-safe snapshot of the newest frame (``None`` before the first
        successful read)."""
        return self.buffer.get_latest(copy=copy)

    def snapshot(self, copy: bool = True) -> FrameSnapshot:
        """Like :meth:`get_latest_frame` but raises when no frame exists yet."""
        snap = self.buffer.get_latest(copy=copy)
        if snap is None:
            raise RuntimeError("no camera frame available yet")
        return snap

    # -- events -------------------------------------------------------------
    def _emit(self, event: CameraEvent) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(event)
        except Exception:  # pragma: no cover - callbacks must not kill capture
            log.exception("camera event callback failed")

    def _set_status(self, status: CameraStatus, message: str) -> None:
        with self._state_lock:
            changed = status is not self._status
            self._status = status
            self._status_message = message
            width, height, fps = self._width, self._height, self._actual_fps
        if changed:
            log.info("Camera status -> %s (%s)", status.value, message)
        self._emit(
            CameraEvent(
                status=status,
                message=message,
                camera_index=self.config.device_index,
                width=width,
                height=height,
                fps=fps,
                timestamp=time.time(),
            )
        )

    # -- worker -------------------------------------------------------------
    def _run(self) -> None:
        cfg = self.config
        failures = 0
        reconnect_attempt = 0

        while not self._stop.is_set():
            if not self._open():
                if not cfg.auto_reconnect:
                    self._set_status(
                        CameraStatus.ERROR,
                        f"could not open camera {cfg.device_index} "
                        f"(permission denied, busy, or absent)",
                    )
                    return
                reconnect_attempt += 1
                self._set_status(
                    CameraStatus.RECONNECTING,
                    f"camera {cfg.device_index} unavailable — retry "
                    f"{reconnect_attempt} in {cfg.reconnect_interval:.1f}s",
                )
                if self._stop.wait(cfg.reconnect_interval * min(reconnect_attempt, 4)):
                    break
                continue

            reconnect_attempt = 0
            self._set_status(
                CameraStatus.STREAMING,
                f"streaming {self._width}x{self._height} "
                f"@ {self._actual_fps:.0f} fps ({self._backend_name})",
            )
            failures = 0
            last_t = time.perf_counter()

            while not self._stop.is_set():
                ok, frame = self._capture()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= cfg.max_read_failures:
                        log.warning(
                            "Camera read failures reached %d — treating device "
                            "as disconnected",
                            failures,
                        )
                        break
                    if self._stop.wait(0.05):
                        break
                    continue

                failures = 0
                now = time.perf_counter()
                self._update_fps(now - last_t)
                last_t = now
                self._frames_read += 1
                try:
                    self.buffer.update(
                        frame,
                        source_fps=self._actual_fps,
                        camera_index=cfg.device_index,
                    )
                except ValueError:
                    log.debug("Rejected invalid frame from camera %s", cfg.device_index)

            # ---- read loop exited -----------------------------------------
            self._close()
            if self._stop.is_set():
                break
            if not cfg.auto_reconnect:
                self._set_status(
                    CameraStatus.DISCONNECTED,
                    "camera disconnected (auto-reconnect disabled)",
                )
                return
            self._set_status(
                CameraStatus.DISCONNECTED,
                "camera disconnected — attempting to reconnect",
            )

        self._close()
        self._set_status(CameraStatus.STOPPED, "camera stopped")

    def _capture(self):
        cap = self._cap
        if cap is None or not cap.isOpened():
            return False, None
        try:
            return cap.read()
        except cv2.error as exc:
            log.debug("cv2 read error: %s", exc)
            return False, None

    def _open(self) -> bool:
        cfg = self.config
        cap = cv2.VideoCapture(cfg.device_index, _api_preference(cfg.backend))
        if not cap.isOpened():
            cap.release()
            self._last_error = (
                f"camera {cfg.device_index} could not be opened "
                "(not connected, busy or permission denied)"
            )
            return False

        _apply_capture_settings(cap, cfg)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # Discard a couple of frames so auto-exposure settles.
        warm = 0
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                warm += 1
                if warm >= 2:
                    break
        if warm == 0:
            cap.release()
            self._last_error = f"camera {cfg.device_index} opened but delivered no frames"
            return False

        with self._state_lock:
            self._cap = cap
            self._width = width
            self._height = height
            self._requested_fps = float(cfg.fps or fps or 0.0)
            self._actual_fps = float(fps or 0.0)
            self._backend_name = _describe_backend(cap)
            self._device_name = _device_name(cfg.device_index)
            self._frame_intervals = []
            self._last_error = None
        log.info(
            "Camera %s opened: %sx%s @ %.0f fps requested, backend=%s, fourcc=%s",
            cfg.device_index,
            width,
            height,
            fps,
            self._backend_name,
            cfg.fourcc or "default",
        )
        return True

    def _close(self) -> None:
        with self._state_lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:  # pragma: no cover
                log.debug("error while releasing camera", exc_info=True)
            log.info("Camera %s released", self.config.device_index)

    def _update_fps(self, interval: float) -> None:
        if interval <= 0:
            return
        inst = 1.0 / interval
        with self._state_lock:
            if self._actual_fps <= 0:
                self._actual_fps = inst
            else:
                self._actual_fps = 0.85 * self._actual_fps + 0.15 * inst
            self._frame_intervals.append(interval)
            if len(self._frame_intervals) > 60:
                self._frame_intervals.pop(0)

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "CameraSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

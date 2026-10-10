"""Camera / video / image acquisition with a latest-frame strategy."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import cv2
import numpy as np

from .config import AppConfig, ConfigError


@dataclass(frozen=True)
class Frame:
    id: int
    image: np.ndarray
    t_capture: float


T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Thread-safe single-slot buffer: writers always replace, readers get the newest item.

    This is the "discard stale frames" primitive: no unbounded queue can build up.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._seq = 0
        self._closed = False
        self.dropped = 0
        self._unread = False  # an item is present and no reader has taken it yet

    def put(self, item: T) -> None:
        with self._cond:
            if self._unread:
                self.dropped += 1
            self._item = item
            self._unread = True
            self._seq += 1
            self._cond.notify_all()

    def get(self) -> T | None:
        with self._cond:
            item = self._item
            if item is not None:
                self._unread = False
            return item

    def wait_new(self, last_seq: int, timeout: float = 1.0) -> tuple[int, T | None]:
        """Block until a new item arrives (seq > last_seq). Returns (seq, item)."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > last_seq or self._closed, timeout):
                return self._seq, None
            if self._seq > last_seq:
                self._unread = False
                return self._seq, self._item
            return self._seq, None

    def seq(self) -> int:
        with self._cond:
            return self._seq

    @property
    def closed(self) -> bool:
        with self._cond:
            return self._closed

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


def list_cameras(max_index: int = 4) -> list[int]:
    """Probe a few device indices and return the ones that open and yield a frame."""
    found: list[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if Path(f"/dev/video{idx}").exists() else cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(idx)
        cap.release()
    return found


class BaseSource:
    """Uniform frame source interface."""

    name: str = "source"

    def read(self) -> Frame | None:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError

    @property
    def is_live(self) -> bool:
        return False

    @property
    def reconnectable(self) -> bool:
        return False


class CameraSource(BaseSource):
    def __init__(self, cfg: AppConfig) -> None:
        self._cfg = cfg
        self.name = f"camera{cfg.source.index}"
        self._cap: cv2.VideoCapture | None = None
        self._id = 0
        self._failures = 0
        self._last_open = 0.0
        self._props: dict[str, float] = {}
        self.open()

    def open(self) -> None:
        sc = self._cfg.source
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        cap: cv2.VideoCapture | None = None
        for backend in backends:
            candidate = cv2.VideoCapture(sc.index, backend)
            if candidate.isOpened():
                cap = candidate
                break
            candidate.release()
        if cap is None:
            raise ConfigError(f"Cannot open camera index {sc.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, sc.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, sc.height)
        cap.set(cv2.CAP_PROP_FPS, sc.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, sc.buffer_size)
        self._cap = cap
        self._failures = 0
        self._last_open = time.time()
        self._props = {
            "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": cap.get(cv2.CAP_PROP_FPS),
        }

    @property
    def properties(self) -> dict[str, float]:
        return dict(self._props)

    @property
    def backend_name(self) -> str:
        if self._cap is None:
            return "none"
        try:
            return self._cap.getBackendName()
        except cv2.error:
            return "unknown"

    @property
    def is_live(self) -> bool:
        return True

    @property
    def reconnectable(self) -> bool:
        return True

    @property
    def healthy(self) -> bool:
        return self._cap is not None and self._failures < self._cfg.camera_reconnect.max_failures

    def read(self) -> Frame | None:
        cap = self._cap
        if cap is None:
            return None
        ok, image = cap.read()
        if not ok or image is None:
            self._failures += 1
            if self._failures >= self._cfg.camera_reconnect.max_failures:
                self.reconnect()
            return None
        self._failures = 0
        self._id += 1
        return Frame(id=self._id, image=image, t_capture=time.time())

    def reconnect(self) -> bool:
        rc = self._cfg.camera_reconnect
        if not rc.enabled:
            return False
        now = time.time()
        if now - self._last_open < rc.interval_s:
            return False
        self.release()
        time.sleep(0.2)
        try:
            self.open()
            return True
        except ConfigError:
            self._cap = None
            self._last_open = time.time()
            return False

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoSource(BaseSource):
    def __init__(self, cfg: AppConfig) -> None:
        path = cfg.resolve(cfg.source.path)
        if path is None or not Path(path).exists():
            raise ConfigError(f"Video source not found: {cfg.source.path}")
        self.name = str(path)
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ConfigError(f"Cannot open video file: {path}")
        self._id = 0
        self._loop = True

    @property
    def reconnectable(self) -> bool:
        return True

    def read(self) -> Frame | None:
        ok, image = self._cap.read()
        if not ok or image is None:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, image = self._cap.read()
            if not ok or image is None:
                return None
        self._id += 1
        return Frame(id=self._id, image=image, t_capture=time.time())

    def release(self) -> None:
        self._cap.release()


class ImageSource(BaseSource):
    """Replays a single image (useful for tests and deterministic benchmarks)."""

    def __init__(self, cfg: AppConfig) -> None:
        path = cfg.resolve(cfg.source.path)
        if path is None or not Path(path).exists():
            raise ConfigError(f"Image source not found: {cfg.source.path}")
        image = cv2.imread(str(path))
        if image is None:
            raise ConfigError(f"Cannot decode image: {path}")
        self.name = str(path)
        self._image = image
        self._id = 0
        self._replicas = 0

    @property
    def reconnectable(self) -> bool:
        return False

    def read(self) -> Frame | None:
        self._id += 1
        self._replicas += 1
        return Frame(id=self._id, image=self._image.copy(), t_capture=time.time())

    def release(self) -> None:
        pass


def create_source(cfg: AppConfig) -> BaseSource:
    if cfg.source.type == "camera":
        return CameraSource(cfg)
    if cfg.source.type == "video":
        return VideoSource(cfg)
    if cfg.source.type == "image":
        return ImageSource(cfg)
    raise ConfigError(f"Unknown source type: {cfg.source.type}")


class CaptureWorker(threading.Thread):
    """Continuously reads the source into a LatestSlot (latest frame wins)."""

    def __init__(
        self,
        source: BaseSource,
        slot: LatestSlot[Frame],
        target_fps: float | None = None,
        on_frame=None,
    ) -> None:
        super().__init__(name="capture", daemon=True)
        self.source = source
        self.slot = slot
        self.target_fps = target_fps
        self.on_frame = on_frame
        self._stop_event = threading.Event()
        self.error: str | None = None
        self.frames_captured = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        period = 1.0 / self.target_fps if self.target_fps and self.target_fps > 0 else 0.0
        while not self._stop_event.is_set():
            t0 = time.time()
            frame = self.source.read()
            if frame is None:
                if not self.source.reconnectable:
                    break
                time.sleep(0.05)
                continue
            self.frames_captured += 1
            if self.on_frame is not None:
                try:
                    self.on_frame(frame)
                except Exception:
                    pass
            self.slot.put(frame)
            if period:
                elapsed = time.time() - t0
                if elapsed < period:
                    time.sleep(period - elapsed)
        self.slot.close()

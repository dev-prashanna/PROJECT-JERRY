"""Background face overlay worker: camera frames in, labelled boxes out.

Feeds off the agent's existing camera buffer (never opens a second camera)
on a daemon thread, so the GUI can poll ``latest()`` on a Qt timer and stamp
boxes + names onto the preview without blocking the vision pipeline.

Detection and matching run through the very same :class:`FaceMemoryService`
the memory engine uses, so what the preview shows is exactly what
enrollment/recognition will see.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from src.utils.logger import get_logger

log = get_logger("face.overlay")

DEFAULT_MAX_FPS = 10.0
#: How long an overlay stays drawable while the preview runs faster.
DEFAULT_MAX_AGE_S = 1.0


@dataclass(frozen=True)
class FaceBox:
    """One detected face in *source frame* coordinates."""

    bbox_xyxy: tuple[float, float, float, float]
    label: str
    matched: bool
    confidence: float


@dataclass(frozen=True)
class FaceOverlay:
    """Result of the most recent detection pass."""

    frame_id: int
    frame_tag: str
    captured_at: float
    source_age_s: float
    faces: list[FaceBox]

    @property
    def age_s(self) -> float:
        """Seconds since this overlay was computed."""
        return time.monotonic() - self.captured_at

    def drawable(self, max_age_s: float = DEFAULT_MAX_AGE_S) -> bool:
        return self.age_s <= max_age_s


class FaceOverlayService:
    """Owns the detect→match loop for the live camera preview."""

    def __init__(
        self,
        camera: Any,
        face_service: Any,
        max_fps: float = DEFAULT_MAX_FPS,
    ):
        self.camera = camera
        self.face_service = face_service
        self._lock = threading.Lock()
        self._latest: Optional[FaceOverlay] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_frame_id = -1
        self._last_run = 0.0
        self._min_interval = 1.0 / max(0.5, max_fps)
        self.frames_processed = 0
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self.camera is None or self.face_service is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="aura-face-overlay", daemon=True
        )
        self._thread.start()
        log.info("face overlay started (max %.1f fps)", 1.0 / self._min_interval)

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def latest(self) -> Optional[FaceOverlay]:
        with self._lock:
            return self._latest

    def status(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "running": self.running,
            "frames": self.frames_processed,
            "last_error": self.last_error,
            "faces": len(latest.faces) if latest else 0,
            "latest_age_s": round(latest.age_s, 3) if latest else None,
        }

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.camera.buffer.get(copy=True)
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                frame = None
            if frame is None or frame.frame_id == self._last_frame_id:
                self._stop.wait(0.02)
                continue
            now = time.perf_counter()
            if now - self._last_run < self._min_interval:
                self._stop.wait(0.01)
                continue
            self._last_run = now
            self._last_frame_id = frame.frame_id
            try:
                boxes = self._detect(frame.image)
            except Exception as exc:  # noqa: BLE001 - never kill the preview
                self.last_error = str(exc)
                log.warning("face overlay failed: %s", exc)
                self._stop.wait(0.2)
                continue
            with self._lock:
                self._latest = FaceOverlay(
                    frame_id=frame.frame_id,
                    frame_tag=frame.tag,
                    captured_at=time.monotonic(),
                    source_age_s=frame.age_s,
                    faces=boxes,
                )
            self.frames_processed += 1
            self.last_error = None
        log.info("face overlay stopped (%d frames)", self.frames_processed)

    def _detect(self, image_bgr) -> list[FaceBox]:
        faces = self.face_service.extract_embeddings(image_bgr)
        boxes: list[FaceBox] = []
        for face in faces:
            label = "unknown"
            matched = False
            if face.embedding is not None:
                match = self.face_service.match_face(face.embedding)
                if match.status.value == "matched" and match.entity_name:
                    label = match.entity_name
                    matched = True
            boxes.append(
                FaceBox(
                    bbox_xyxy=tuple(float(v) for v in face.bbox_xyxy),
                    label=label,
                    matched=matched,
                    confidence=float(face.confidence),
                )
            )
        return boxes


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

_KNOWN_BGR = (80, 200, 90)
_UNKNOWN_BGR = (180, 180, 180)


def draw_face_boxes(image_bgr, faces: list[FaceBox]):
    """Stamp boxes + labels onto a copy of ``image_bgr``."""
    import cv2
    import numpy as np

    out = np.asarray(image_bgr).copy()
    h, w = out.shape[:2]
    for box in faces:
        x1, y1, x2, y2 = box.bbox_xyxy
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _KNOWN_BGR if box.matched else _UNKNOWN_BGR
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = box.label
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + baseline - 2), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 4, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (16, 20, 26),
            2,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# singleton (one worker per camera, mirroring DepthService)
# ---------------------------------------------------------------------------

_SERVICES: dict[int, FaceOverlayService] = {}
_SERVICES_LOCK = threading.Lock()


def get_face_overlay_service(
    camera: Any, face_service: Any, max_fps: float = DEFAULT_MAX_FPS
) -> FaceOverlayService:
    key = id(camera)
    with _SERVICES_LOCK:
        svc = _SERVICES.get(key)
        if svc is None or svc.face_service is not face_service:
            if svc is not None:
                svc.stop(timeout_s=0.5)
            svc = FaceOverlayService(camera, face_service, max_fps=max_fps)
            _SERVICES[key] = svc
        return svc


def stop_face_overlay_service(camera: Any, timeout_s: float = 2.0) -> None:
    key = id(camera)
    with _SERVICES_LOCK:
        svc = _SERVICES.pop(key, None)
    if svc is not None:
        svc.stop(timeout_s=timeout_s)

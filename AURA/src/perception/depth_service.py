"""Background depth worker: AURA camera frames in, colourized depth out.

Feeds off the agent's existing camera buffer (never opens a second camera)
on a daemon thread, so the GUI can poll ``latest()`` on a Qt timer without
blocking the vision pipeline.  Exposed to the VLM via ``latest().depth``.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

import numpy as np

from config.settings import DepthSettings, get_settings
from src.perception.depth_engine import DepthEngine, DepthFrame
from src.utils.logger import get_logger

log = get_logger("depth.service")


class DepthService:
    """Owns a :class:`DepthEngine` and a capture→infer loop."""

    def __init__(self, camera: Any, settings: Optional[DepthSettings] = None):
        self.camera = camera
        self.settings = settings or get_settings().depth
        self.engine = DepthEngine(self.settings)
        self._lock = threading.Lock()
        self._latest: Optional[DepthFrame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_infer_id = -1
        self._last_submit = 0.0
        self.frames_processed = 0
        self.last_error: Optional[str] = None
        self._min_interval = 1.0 / max(0.5, self.settings.max_fps)

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.settings.enabled or self.camera is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="aura-depth", daemon=True)
        self._thread.start()
        log.info("depth service started (max %.1f fps, %s colormap)", self.settings.max_fps, self.settings.colormap)

    def wait_until_ready(self, timeout_s: float = 5.0) -> bool:
        """Block until the first depth frame is available (or timeout)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.latest() is not None:
                return True
            time.sleep(0.05)
        return self.latest() is not None

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
        self._thread = None
        try:
            self.engine.unload()
        except Exception:  # noqa: BLE001 - best-effort CUDA teardown
            pass

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def latest(self) -> Optional[DepthFrame]:
        with self._lock:
            return self._latest

    def summarize_for_vlm(self, region: Optional[tuple[int, int, int, int]] = None) -> Optional[dict[str, Any]]:
        """Compact depth summary for the VLM prompt.

        Returns inverse-depth stats (near=high) over the whole frame or a
        ``(x0, y0, x1, y1)`` region, plus the near/far layout, so the
        language model can reason about distance without seeing pixels.
        """
        depth = self.latest()
        if depth is None:
            return None
        d = depth.depth
        h, w = d.shape[:2]
        if region is not None:
            x0, y0, x1, y1 = region
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w, int(x1)), min(h, int(y1))
            patch = d[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else d
            label = f"region ({x0},{y0})-({x1},{y1})"
        else:
            patch = d
            label = "full frame"
        finite = patch[np.isfinite(patch)]
        if finite.size == 0:
            return None
        lo, hi = float(np.min(finite)), float(np.max(finite))
        span = max(1e-9, hi - lo)
        # third-split the range into near / mid / far bands (inverse depth)
        near = float(np.mean(finite >= hi - span / 3.0))
        far = float(np.mean(finite <= lo + span / 3.0))
        mid = max(0.0, 1.0 - near - far)
        cx, cy = w // 2, h // 2
        center = float(d[cy, cx]) if np.isfinite(d[cy, cx]) else float(np.mean(finite))
        return {
            "convention": "inverse",  # higher = closer
            "unit": depth.unit,
            "region": label,
            "shape": [h, w],
            "min": lo,
            "max": hi,
            "mean": float(np.mean(finite)),
            "center": center,
            "near_frac": near,
            "mid_frac": mid,
            "far_frac": far,
            "nearest_hint": "closer to camera" if center > (hi - span / 3.0) else ("farther" if center < (lo + span / 3.0) else "mid distance"),
            "inference_ms": depth.inference_ms,
            "age_s": depth.age_s,
        }

    def status(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "enabled": self.settings.enabled,
            "running": self.running,
            "loaded": self.engine.is_loaded,
            "device": self.engine.device,
            "model": self.settings.model_id,
            "colormap": self.settings.colormap,
            "frames": self.frames_processed,
            "last_error": self.last_error,
            "latest_frame_id": latest.frame_id if latest else None,
            "latest_inference_ms": round(latest.inference_ms, 1) if latest else None,
            "latest_age_s": round(latest.age_s, 3) if latest else None,
        }

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        try:
            self.engine.load()
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            log.warning("depth engine failed to load: %s", exc)
            return
        while not self._stop.is_set():
            try:
                frame = self.camera.buffer.get(copy=True)
            except Exception:  # noqa: BLE001
                frame = None
            if frame is None or frame.frame_id == self._last_infer_id:
                self._stop.wait(0.02)
                continue
            now = time.perf_counter()
            if now - self._last_submit < self._min_interval:
                self._stop.wait(0.01)
                continue
            self._last_submit = now
            self._last_infer_id = frame.frame_id
            try:
                result = self.engine.infer(
                    frame.image, frame_id=frame.frame_id, timestamp=frame.timestamp
                )
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                log.warning("depth inference failed: %s", exc)
                self._stop.wait(0.1)
                continue
            with self._lock:
                self._latest = result
            self.frames_processed += 1
            self.last_error = None
        log.info("depth service stopped (%d frames)", self.frames_processed)


_SERVICES: dict[int, DepthService] = {}
_SERVICES_LOCK = threading.Lock()


def get_depth_service(camera: Any, settings: Optional[DepthSettings] = None) -> DepthService:
    """Process-wide singleton per camera object (so the model loads once)."""
    key = id(camera)
    with _SERVICES_LOCK:
        svc = _SERVICES.get(key)
        if svc is None:
            svc = DepthService(camera, settings)
            _SERVICES[key] = svc
        return svc


def stop_depth_service(camera: Any, timeout_s: float = 2.0) -> None:
    """Stop (and forget) the depth service bound to ``camera``, if any."""
    key = id(camera)
    with _SERVICES_LOCK:
        svc = _SERVICES.pop(key, None)
    if svc is not None:
        svc.stop(timeout_s=timeout_s)


def stop_all_depth_services(timeout_s: float = 2.0) -> None:
    with _SERVICES_LOCK:
        services = list(_SERVICES.values())
        _SERVICES.clear()
    for svc in services:
        svc.stop(timeout_s=timeout_s)

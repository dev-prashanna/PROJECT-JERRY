"""High-level orchestration: camera + Moondream + inference worker.

:class:`VQAPipeline` is Qt-free.  It exposes a single callback,
``on_event(kind, payload)``, through which every asynchronous transition
(camera status, model loading, inference progress, answers, errors) is
announced.  The GUI wraps that callback in a Qt signal so slots always run on
the main thread; the CLI just prints.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from config.settings import AppConfig, build_config, resolve_model_path
from src.camera.camera import CameraEvent, CameraSource, CameraStatus, discover_cameras
from src.camera.frame_buffer import FrameSnapshot
from src.pipeline.inference_worker import InferenceJob, InferenceWorker
from src.utils.logging_utils import get_logger
from src.vlm.moondream_engine import EngineInfo, MoondreamEngine

log = get_logger(__name__)

__all__ = ["VQAPipeline", "PipelineMode", "Event"]

Event = Callable[[str, Dict[str, Any]], None]


class PipelineMode:
    """Mode A asks about the newest frame; mode B freezes one frame."""

    LIVE = "live"
    FROZEN = "frozen"


@dataclass
class FrozenInfo:
    frame_id: int
    timestamp: float
    width: int
    height: int
    camera_index: int


class VQAPipeline:
    """Application core shared by the GUI and the CLI front-ends."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        on_event: Optional[Event] = None,
        engine: Optional[MoondreamEngine] = None,
        camera: Optional[CameraSource] = None,
    ) -> None:
        self.config = config or build_config()
        self._on_event = on_event

        self.camera = camera or CameraSource(
            self.config.camera, on_event=self._on_camera_event
        )
        if camera is not None:
            # Injected cameras (tests) still report through this pipeline.
            self.camera._on_event = self._on_camera_event  # type: ignore[attr-defined]
        self.engine = engine or MoondreamEngine(self.config.model, self.config.inference)
        self.worker = InferenceWorker(
            self.engine,
            on_result=self._on_worker_result,
            on_state=self._on_worker_state,
            queue_size=self.config.inference.queue_size,
        )

        self._mode = PipelineMode.LIVE
        self._frozen: Optional[FrameSnapshot] = None
        self._frozen_info: Optional[FrozenInfo] = None
        self._model_state = "not_loaded"  # not_loaded|loading|loaded|error
        self._model_error: Optional[str] = None
        self._model_info: Optional[EngineInfo] = None
        self._loader: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._started_at = time.time()

    # ------------------------------------------------------------------
    # events
    # ------------------------------------------------------------------
    def _emit(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, dict(payload or {}))
        except Exception:  # pragma: no cover - listeners must not break pipeline
            log.exception("event listener failed for %s", kind)

    def _set_error(self, message: str, source: str = "pipeline") -> None:
        self._last_error = message
        log.error("[%s] %s", source, message)
        self._emit("error", {"message": message, "source": source})

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------
    def _on_camera_event(self, event: CameraEvent) -> None:
        self._emit(
            "camera_status",
            {
                "status": event.status.value,
                "message": event.message,
                "camera_index": event.camera_index,
                "width": event.width,
                "height": event.height,
                "fps": event.fps,
                "timestamp": event.timestamp,
            },
        )
        if event.status in (CameraStatus.ERROR, CameraStatus.DISCONNECTED):
            self._last_error = event.message

    def start_camera(self, index: Optional[int] = None) -> bool:
        """Start the camera, optionally switching to another device index."""
        if index is not None and index != self.camera.config.device_index:
            if self.camera.is_running:
                self.camera.stop()
            self.camera.config.device_index = int(index)
            self._emit(
                "camera_selected",
                {"camera_index": int(index)},
            )
        if self.camera.is_streaming:
            return True
        return self.camera.start()

    def stop_camera(self) -> None:
        self.camera.stop()

    def restart_camera(self) -> bool:
        return self.camera.restart()

    def discover_cameras(self):
        return discover_cameras(
            max_index=self.config.camera.probe_max_index,
            backend=self.config.camera.backend,
            active_index=self.camera.active_index if self.camera.is_running else None,
        )

    def get_latest_frame(self, copy: bool = True) -> Optional[FrameSnapshot]:
        return self.camera.get_latest_frame(copy=copy)

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    @property
    def model_state(self) -> str:
        return self._model_state

    @property
    def model_error(self) -> Optional[str]:
        return self._model_error

    @property
    def model_info(self) -> Optional[EngineInfo]:
        return self._model_info

    @property
    def model_loaded(self) -> bool:
        return self.engine.is_loaded

    def load_model_async(self) -> bool:
        """Load Moondream on a background thread so the UI stays responsive."""
        if self.engine.is_loaded:
            self._model_state = "loaded"
            self._emit("model_status", self._model_status_payload())
            return False
        if self._loader is not None and self._loader.is_alive():
            return False
        self._model_state = "loading"
        self._model_error = None
        self._emit("model_status", self._model_status_payload())
        self._loader = threading.Thread(
            target=self._load_model_worker, name="moondream-loader", daemon=True
        )
        self._loader.start()
        return True

    def _load_model_worker(self) -> None:
        try:
            info = self.engine.load()
        except Exception as exc:  # noqa: BLE001 - surfaced in the GUI
            self._model_state = "error"
            self._model_error = f"{type(exc).__name__}: {exc}"
            self._set_error(self._model_error, source="model")
            self._emit("model_status", self._model_status_payload())
            return
        self._model_info = info
        self._model_state = "loaded"
        self._model_error = None
        self.worker.start()
        self._emit("model_status", self._model_status_payload())

    def load_model_blocking(self) -> EngineInfo:
        """Load Moondream synchronously (CLI front-end / tests).

        Failures are published through the ``model_status`` event and then
        re-raised so the caller can decide how to exit.
        """
        try:
            info = self.engine.load()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller
            self._model_state = "error"
            self._model_error = f"{type(exc).__name__}: {exc}"
            self._set_error(self._model_error, source="model")
            self._emit("model_status", self._model_status_payload())
            raise
        self._model_info = info
        self._model_state = "loaded"
        self.worker.start()
        self._emit("model_status", self._model_status_payload())
        return info

    def _model_status_payload(self) -> Dict[str, Any]:
        info = self._model_info or self.engine.info()
        return {
            "state": self._model_state,
            "error": self._model_error,
            "device": info.device if info else "",
            "dtype": info.dtype if info else "",
            "api": info.api if info else "",
            "model_class": info.model_class if info else "",
            "model_path": info.model_path if info else "",
            "load_seconds": info.load_seconds if info else 0.0,
            "gpu_name": info.gpu_name if info else "",
            "parameter_count": info.parameter_count if info else 0,
            "describe": info.describe() if info else "",
        }

    # ------------------------------------------------------------------
    # modes / freezing
    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def frozen_info(self) -> Optional[FrozenInfo]:
        return self._frozen_info

    def freeze(self) -> Optional[FrameSnapshot]:
        """Capture the current live frame and switch to frozen mode."""
        snap = self.camera.get_latest_frame(copy=True)
        if snap is None:
            self._set_error("cannot freeze: no camera frame available yet")
            return None
        self._frozen = snap
        self._frozen_info = FrozenInfo(
            frame_id=snap.frame_id,
            timestamp=snap.timestamp,
            width=snap.width,
            height=snap.height,
            camera_index=snap.camera_index,
        )
        self._mode = PipelineMode.FROZEN
        log.info("Frozen frame #%d for repeated questions", snap.frame_id)
        self._emit("mode_changed", self._mode_payload())
        return snap

    def unfreeze(self) -> None:
        """Return to live mode; every question gets a brand new snapshot."""
        self._frozen = None
        self._frozen_info = None
        self._mode = PipelineMode.LIVE
        self.engine.clear_image_cache()
        log.info("Switched to live snapshot mode")
        self._emit("mode_changed", self._mode_payload())

    def _mode_payload(self) -> Dict[str, Any]:
        frozen = self._frozen_info
        return {
            "mode": self._mode,
            "frozen": (
                {
                    "frame_id": frozen.frame_id,
                    "timestamp": frozen.timestamp,
                    "width": frozen.width,
                    "height": frozen.height,
                    "camera_index": frozen.camera_index,
                }
                if frozen
                else None
            ),
        }

    # ------------------------------------------------------------------
    # questions
    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.worker.busy

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def ask(self, question: str) -> tuple[bool, str]:
        """Validate and queue a question.  Returns ``(accepted, message)``."""
        text = (question or "").strip()
        if not text:
            return False, "type a question first"
        if len(text) > 4000:
            return False, "question is too long (max 4000 characters)"
        if self._model_state == "error":
            return False, "the model failed to load — check the error log"
        if self._model_state != "loaded" or not self.engine.is_loaded:
            return False, "the model is still loading — try again in a moment"
        if self.worker.busy and self.config.inference.queue_size <= 0:
            return False, "an answer is already being generated"

        # ---- pick the exact snapshot this question refers to --------------
        if self._mode == PipelineMode.FROZEN:
            snap = self._frozen
            if snap is None:
                return False, "no frozen frame — freeze a frame first"
        else:
            # The buffer may be overwritten at any moment, so the question
            # receives its own copy of the newest frame.
            snap = self.camera.get_latest_frame(copy=True)
            if snap is None:
                return False, "no camera frame available yet — start the camera"

        job = InferenceJob(
            job_id=self.worker.next_job_id(),
            question=text,
            snapshot=snap,
            cache_key=f"frame:{snap.frame_id}",
            mode=self._mode,
            created_at=time.time(),
        )
        accepted, reason = self.worker.submit(job)
        if not accepted:
            self._emit(
                "rejected",
                {"job_id": job.job_id, "question": text, "reason": reason},
            )
            return False, reason

        self._emit(
            "submitted",
            {
                "job_id": job.job_id,
                "question": text,
                "mode": job.mode,
                "frame_id": job.frame_id,
                "frame_timestamp": job.snapshot.timestamp,
                "frozen": self._mode == PipelineMode.FROZEN,
            },
        )
        return True, "queued"

    # ------------------------------------------------------------------
    # worker callbacks (inference thread)
    # ------------------------------------------------------------------
    def _on_worker_result(self, result: Dict[str, Any]) -> None:
        self._emit("inference_result", result)

    def _on_worker_state(self, state: str, payload: Dict[str, Any]) -> None:
        self._emit("inference_state", dict(payload, state=state))

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop every worker and release the camera.  Safe to call twice."""
        timeout = (
            timeout
            if timeout is not None
            else self.config.inference.shutdown_timeout
        )
        log.info("Shutting down pipeline (timeout=%.1fs)", timeout)
        try:
            self.worker.stop(timeout=timeout)
        except Exception:  # pragma: no cover
            log.exception("worker stop failed")
        try:
            self.camera.stop()
        except Exception:  # pragma: no cover
            log.exception("camera stop failed")
        try:
            if self.engine.is_loaded:
                self.engine.clear_image_cache()
        except Exception:  # pragma: no cover
            log.exception("engine cleanup failed")
        self._emit("stopped", {"uptime_seconds": time.time() - self._started_at})

    @property
    def status_summary(self) -> Dict[str, Any]:
        cam = self.camera
        return {
            "camera_status": cam.status.value,
            "camera_message": cam.status_message,
            "resolution": "%dx%d" % cam.resolution,
            "fps": round(cam.actual_fps, 1),
            "model_state": self._model_state,
            "model_device": self._model_info.device if self._model_info else "",
            "inference": "running" if self.worker.busy else "idle",
            "mode": self._mode,
            "last_error": self._last_error,
        }

    def __enter__(self) -> "VQAPipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

"""Worker coordination: capture thread, inference thread, bounded latest-frame buffers."""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass

from .camera import BaseSource, CaptureWorker, Frame, LatestSlot, create_source
from .config import AppConfig
from .depth_engine import DepthEngine, DepthResult
from .temporal_filter import DepthSmoother


@dataclass
class PipelineStats:
    captured: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0


class InferenceWorker(threading.Thread):
    """Consumes the newest captured frame, skips stale ones, publishes aligned depth maps."""

    def __init__(
        self,
        engine: DepthEngine,
        frame_slot: LatestSlot[Frame],
        depth_slot: LatestSlot[DepthResult],
        smoother: DepthSmoother,
        frame_skip: int = 1,
    ) -> None:
        super().__init__(name="inference", daemon=True)
        self.engine = engine
        self.frame_slot = frame_slot
        self.depth_slot = depth_slot
        self.smoother = smoother
        self.frame_skip = max(1, frame_skip)
        self._stop_event = threading.Event()
        self.stats = PipelineStats()
        self.error: str | None = None
        self.last_result: DepthResult | None = None

    def stop(self) -> None:
        self._stop_event.set()
        self.frame_slot.close()

    def run(self) -> None:
        seq = 0
        last_processed_id = -10**9
        while not self._stop_event.is_set():
            seq, frame = self.frame_slot.wait_new(seq, timeout=0.5)
            if frame is None:
                if self.frame_slot.closed:
                    break
                continue
            if frame.id - last_processed_id < self.frame_skip:
                self.stats.skipped += 1
                continue
            try:
                result = self.engine.infer(frame.image, frame_id=frame.id, t_capture=frame.t_capture)
                if self.smoother.enabled:
                    result.depth = self.smoother.apply(result.depth)
                self.depth_slot.put(result)
                self.last_result = result
                self.stats.processed += 1
                last_processed_id = frame.id
            except Exception:
                self.stats.errors += 1
                self.error = traceback.format_exc()
                break
        self.depth_slot.close()


class Pipeline:
    """Owns the source, the two workers and their bounded slots."""

    def __init__(
        self,
        cfg: AppConfig,
        engine: DepthEngine,
        smoother: DepthSmoother | None = None,
        on_capture=None,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.source: BaseSource = create_source(cfg)
        self.frame_slot: LatestSlot[Frame] = LatestSlot()
        self.depth_slot: LatestSlot[DepthResult] = LatestSlot()
        self.smoother = smoother or DepthSmoother(
            enabled=cfg.depth.smoothing.enabled,
            alpha=cfg.depth.smoothing.alpha,
            median_window=cfg.depth.smoothing.median_window,
        )
        self.capture = CaptureWorker(self.source, self.frame_slot, target_fps=cfg.source.fps, on_frame=on_capture)
        self.inference = InferenceWorker(engine, self.frame_slot, self.depth_slot, self.smoother, cfg.runtime.frame_skip)
        self._started = False
        self.stats = PipelineStats()

    def start(self) -> None:
        self.capture.start()
        self.inference.start()
        self._started = True

    @property
    def started(self) -> bool:
        return self._started

    def latest_frame(self) -> Frame | None:
        return self.frame_slot.get()

    def latest_depth(self) -> DepthResult | None:
        return self.depth_slot.get()

    def stop(self) -> None:
        self.capture.stop()
        self.inference.stop()
        for worker in (self.capture, self.inference):
            if worker.is_alive():
                worker.join(timeout=2.0)
        self.source.release()
        self._started = False

    def snapshot(self) -> dict:
        return {
            "captured": self.capture.frames_captured,
            "processed": self.inference.stats.processed,
            "skipped": self.inference.stats.skipped,
            "errors": self.inference.stats.errors,
            "capture_alive": self.capture.is_alive(),
            "inference_alive": self.inference.is_alive(),
            "frame_slot_dropped": self.frame_slot.dropped,
            "depth_slot_dropped": self.depth_slot.dropped,
            "inference_error": self.inference.error,
        }

    def wait_for_depth(self, timeout: float = 10.0) -> DepthResult | None:
        deadline = time.time() + timeout
        seq = 0
        while time.time() < deadline:
            seq, result = self.depth_slot.wait_new(seq, timeout=0.5)
            if result is not None:
                return result
        return self.depth_slot.get()

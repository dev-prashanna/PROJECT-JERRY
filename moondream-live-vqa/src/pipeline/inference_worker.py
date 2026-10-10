"""Inference worker: a single background thread that owns all GPU work.

Requests travel through a **bounded** queue (default capacity 1, so at most
one job waits behind the active one).  Submitting beyond that is rejected
with a clear reason instead of growing an unbounded backlog, which keeps the
GUI responsive and the GPU from being flooded.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from src.camera.frame_buffer import FrameSnapshot
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

__all__ = ["InferenceJob", "InferenceWorker"]

_SENTINEL = None


@dataclass
class InferenceJob:
    """One question + the exact frame snapshot it must be answered from."""

    job_id: int
    question: str
    snapshot: FrameSnapshot
    cache_key: str
    mode: str
    created_at: float = field(default_factory=time.time)

    @property
    def frame_id(self) -> int:
        return self.snapshot.frame_id


class InferenceWorker:
    """Bounded-queue worker thread around a :class:`MoondreamEngine`."""

    def __init__(
        self,
        engine: Any,
        on_result: Callable[[Dict[str, Any]], None],
        on_state: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        queue_size: int = 1,
    ) -> None:
        self.engine = engine
        self._on_result = on_result
        self._on_state = on_state
        self._capacity = max(0, int(queue_size))
        # ``maxsize=0`` would mean "unbounded" for queue.Queue, so the physical
        # queue always has room for at least one item and the *policy* above
        # (``_capacity``) is what bounds submissions.
        self._queue: "queue.Queue[Optional[InferenceJob]]" = queue.Queue(
            maxsize=max(1, self._capacity)
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._job_counter = 0
        self._active_job: Optional[InferenceJob] = None
        self._queued_jobs = 0
        self._results_delivered = 0
        self._results_dropped = 0
        self._accept_results = True

    # -- state --------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._active_job is not None or self._queued_jobs > 0

    @property
    def active_job(self) -> Optional[InferenceJob]:
        with self._lock:
            return self._active_job

    @property
    def pending(self) -> int:
        with self._lock:
            return self._queued_jobs

    @property
    def results_delivered(self) -> int:
        return self._results_delivered

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> bool:
        if self.is_running:
            return False
        self._stop.clear()
        self._accept_results = True
        self._thread = threading.Thread(
            target=self._run, name="vqa-inference", daemon=True
        )
        self._thread.start()
        log.info("Inference worker started (queue capacity=%d)", self._capacity)
        return True

    def stop(self, timeout: float = 10.0) -> None:
        """Stop accepting work and wait for the in-flight request (if any)."""
        self._accept_results = False
        self._stop.set()
        self._drain_queue()
        self._put_sentinel()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                log.warning(
                    "Inference worker still busy after %.1fs — abandoning it "
                    "(the process will exit when the request finishes)",
                    timeout,
                )
        self._thread = None
        with self._lock:
            self._active_job = None
            self._queued_jobs = 0
        log.info(
            "Inference worker stopped (%d results delivered, %d dropped)",
            self._results_delivered,
            self._results_dropped,
        )

    @property
    def capacity(self) -> int:
        """Number of jobs that may wait behind the active one."""
        return self._capacity

    # -- submission ---------------------------------------------------------
    def submit(self, job: InferenceJob) -> tuple[bool, str]:
        """Queue ``job``.  Returns ``(accepted, reason)``.

        The queue is bounded on purpose: submitting faster than the model can
        answer is rejected instead of growing an unbounded backlog.
        """
        if self._stop.is_set() or not self._accept_results:
            return False, "inference worker is shutting down"
        if not self.is_running:
            self.start()
        with self._lock:
            pending = self._queued_jobs
            active = self._active_job is not None
        if self._capacity == 0:
            full = active or pending > 0
        else:
            full = pending >= self._capacity
        if full:
            return (
                False,
                "a request is already queued — wait for the current answer",
            )
        try:
            self._queue.put_nowait(job)
        except queue.Full:  # pragma: no cover - defensive
            return False, "inference queue is full"
        with self._lock:
            self._queued_jobs += 1
            self._job_counter = max(self._job_counter, job.job_id)
        log.debug("Queued job #%d (%d pending)", job.job_id, self._queued_jobs)
        return True, "queued"

    def next_job_id(self) -> int:
        with self._lock:
            self._job_counter += 1
            return self._job_counter

    # -- internals ----------------------------------------------------------
    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._queued_jobs = max(0, self._queued_jobs - 1)

    def _put_sentinel(self) -> None:
        while True:
            try:
                self._queue.put_nowait(_SENTINEL)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:  # pragma: no cover
                    return

    def _emit_state(self, state: str, payload: Dict[str, Any]) -> None:
        if self._on_state is None:
            return
        try:
            self._on_state(state, payload)
        except Exception:  # pragma: no cover
            log.exception("inference state callback failed")

    def _deliver(self, result: Dict[str, Any]) -> None:
        if not self._accept_results:
            self._results_dropped += 1
            return
        try:
            self._on_result(result)
            self._results_delivered += 1
        except Exception:  # pragma: no cover - never kill the worker
            log.exception("result callback failed")

    def _run(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if job is _SENTINEL:
                break

            with self._lock:
                self._queued_jobs = max(0, self._queued_jobs - 1)
                self._active_job = job

            queued_seconds = max(0.0, time.time() - job.created_at)
            self._emit_state(
                "running",
                {
                    "job_id": job.job_id,
                    "question": job.question,
                    "mode": job.mode,
                    "frame_id": job.frame_id,
                    "queued_seconds": queued_seconds,
                },
            )
            started = time.perf_counter()
            try:
                result = self.engine.ask(
                    job.question,
                    frame=job.snapshot.frame,
                    frame_id=job.frame_id,
                    frame_timestamp=job.snapshot.timestamp,
                    cache_key=job.cache_key,
                )
            except Exception as exc:  # engine should not raise; belt and braces
                log.exception("engine.ask raised")
                result = {
                    "status": "error",
                    "answer": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "question": job.question,
                    "frame_id": job.frame_id,
                    "device": getattr(self.engine, "device", ""),
                    "inference_time_seconds": time.perf_counter() - started,
                    "timestamp": time.time(),
                }

            result = dict(result)
            result.setdefault("status", "success")
            result["job_id"] = job.job_id
            result["mode"] = job.mode
            result["queued_seconds"] = queued_seconds
            result["frame_timestamp"] = job.snapshot.timestamp
            result.setdefault("frame_id", job.frame_id)
            result["camera_fps"] = job.snapshot.source_fps

            self._deliver(result)
            self._emit_state(
                "finished",
                {
                    "job_id": job.job_id,
                    "status": result.get("status", "success"),
                    "inference_time_seconds": result.get("inference_time_seconds"),
                },
            )
            with self._lock:
                self._active_job = None

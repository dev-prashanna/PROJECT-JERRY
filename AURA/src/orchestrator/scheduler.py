"""Bounded request scheduler with cancellation and clean shutdown.

One worker executes pipeline requests sequentially (local models are not
parallel-safe anyway and would thrash the 8 GB GPU).  The queue is bounded:

* ``drop_oldest`` (default) – a new command supersedes an unstarted one.
* otherwise the submitter gets :class:`QueueFullError`.

Each request carries a unique id so a slow response can never overwrite a
newer one, and a cancellation event so ``stop()``/barge-in can interrupt
between pipeline stages.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.orchestrator.schemas import new_id
from src.utils.exceptions import QueueFullError
from src.utils.logger import get_logger

log = get_logger("scheduler")


@dataclass
class Job:
    payload: Any
    request_id: str = field(default_factory=lambda: new_id("req"))
    submitted_at: float = field(default_factory=time.time)
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Optional[BaseException] = None
    dropped: bool = False

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()


class PipelineScheduler:
    def __init__(
        self,
        handler: Callable[[Job], Any],
        queue_size: int = 2,
        drop_oldest: bool = True,
        name: str = "aura-pipeline",
    ):
        self.handler = handler
        self.queue: "queue.Queue[Optional[Job]]" = queue.Queue(maxsize=max(1, queue_size))
        self.drop_oldest = drop_oldest
        self._thread: Optional[threading.Thread] = None
        self._name = name
        self._lock = threading.Lock()
        self._closed = False
        self._current: Optional[Job] = None
        self._processed = 0
        self._rejected = 0

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._closed = False
            self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
            log.info("scheduler started (queue_size=%d)", self.queue.maxsize)

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            if job is None:
                self.queue.task_done()
                break
            self._current = job
            try:
                if job.cancelled:
                    job.dropped = True
                else:
                    job.result = self.handler(job)
            except BaseException as exc:  # noqa: BLE001 - reported, never swallowed silently
                job.error = exc
                log.exception("request %s failed", job.request_id)
            finally:
                self._current = None
                job.done.set()
                self.queue.task_done()
                self._processed += 1

    # -- submission ------------------------------------------------------
    def submit(self, payload: Any, request_id: Optional[str] = None) -> Job:
        job = Job(payload=payload, request_id=request_id or new_id("req"))
        if self._closed:
            raise QueueFullError("scheduler is shut down")
        try:
            self.queue.put_nowait(job)
        except queue.Full:
            if not self.drop_oldest:
                self._rejected += 1
                raise QueueFullError("pipeline queue is full")
            # Supersede the oldest waiting request.
            try:
                old = self.queue.get_nowait()
                old.dropped = True
                old.done.set()
                log.info("request %s superseded by a newer one", old.request_id)
            except queue.Empty:
                pass
            self.queue.task_done()
            try:
                self.queue.put_nowait(job)
            except queue.Full:
                self._rejected += 1
                raise QueueFullError("pipeline queue is full")
        return job

    def wait(self, job: Job, timeout: Optional[float] = None) -> Job:
        job.done.wait(timeout)
        return job

    # -- cancellation / shutdown ----------------------------------------
    def cancel_current(self) -> bool:
        job = self._current
        if job and not job.done.is_set():
            job.cancel.set()
            log.info("cancelling in-flight request %s", job.request_id)
            return True
        return False

    def cancel_pending(self) -> int:
        cancelled = 0
        while True:
            try:
                job = self.queue.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job.cancel.set()
                job.dropped = True
                job.done.set()
                cancelled += 1
            self.queue.task_done()
        return cancelled

    def shutdown(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.cancel_current()
        self.cancel_pending()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                log.warning("scheduler thread did not stop within %.1fs", timeout_s)
        log.info(
            "scheduler stopped (processed=%d rejected=%d)",
            self._processed, self._rejected,
        )

    # -- status ----------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._current is not None and not self._current.done.is_set()

    @property
    def current_job(self) -> Optional[Job]:
        return self._current

    @property
    def depth(self) -> int:
        return self.queue.qsize()

    @property
    def processed(self) -> int:
        return self._processed

    def status(self) -> dict[str, Any]:
        return {
            "queue_depth": self.depth,
            "busy": self.busy,
            "current": self._current.request_id if self._current else None,
            "processed": self._processed,
            "rejected": self._rejected,
            "closed": self._closed,
        }

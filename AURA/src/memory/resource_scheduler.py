"""Bounded background job queue for memory operations.

Priority classes:

1. real-time speech / user-requested inference (never queued here)
2. visual recognition needed for the current request (runs inline)
3. persistent memory embedding / background fact learning
4. cache maintenance and index rebuilds

Only priority 3–4 work runs on this scheduler so the primary pipeline stays
responsive.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

from src.memory.exceptions import MemoryOperationCancelled, MemoryQueueFull
from src.utils.logger import get_logger

log = get_logger("memory.scheduler")


class MemoryJobPriority(IntEnum):
    INTERACTIVE = 1
    REQUEST_VISUAL = 2
    BACKGROUND = 3
    MAINTENANCE = 4


@dataclass(order=True)
class MemoryJob:
    priority: int
    seq: int
    job_id: str = field(compare=False, default="")
    operation: str = field(compare=False, default="")
    payload: Any = field(compare=False, default=None)
    created_at: float = field(compare=False, default_factory=time.time)
    done: threading.Event = field(compare=False, default_factory=threading.Event)
    cancel: threading.Event = field(compare=False, default_factory=threading.Event)
    result: Any = field(compare=False, default=None)
    error: Optional[BaseException] = field(compare=False, default=None)

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_set()


class MemoryJobScheduler:
    """Single worker, bounded priority queue, no unbounded growth."""

    def __init__(
        self,
        handler: Callable[[MemoryJob], Any],
        *,
        max_pending: int = 16,
        name: str = "aura-memory",
    ):
        self.handler = handler
        self.max_pending = int(max_pending)
        self.name = name
        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.max_pending)
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._running = False
        self._seq = 0
        self.processed = 0
        self.rejected = 0
        self.failed = 0

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._worker = threading.Thread(target=self._run, name=self.name, daemon=True)
            self._worker.start()

    def stop(self, timeout_s: float = 3.0) -> None:
        with self._lock:
            self._running = False
            worker = self._worker
            self._worker = None
        try:
            self._queue.put_nowait(None)  # type: ignore[arg-type]
        except queue.Full:
            pass
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout_s)

    def submit(
        self,
        operation: str,
        payload: Any = None,
        *,
        priority: MemoryJobPriority = MemoryJobPriority.BACKGROUND,
        job_id: str = "",
    ) -> MemoryJob:
        with self._lock:
            self._seq += 1
            seq = self._seq
        job = MemoryJob(
            priority=int(priority),
            seq=seq,
            job_id=job_id or f"mjob_{seq}",
            operation=operation,
            payload=payload,
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self.rejected += 1
            raise MemoryQueueFull(
                f"memory job queue is full ({self.max_pending}); job {operation} rejected"
            )
        return job

    def wait(self, job: MemoryJob, timeout: Optional[float] = None) -> Any:
        job.done.wait(timeout)
        if job.error is not None:
            raise job.error
        return job.result

    def cancel(self, job: MemoryJob) -> None:
        job.cancel.set()

    def _run(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                if not self._running:
                    return
                continue
            if item is None:
                return
            job: MemoryJob = item
            if job.cancelled:
                job.error = MemoryOperationCancelled("job cancelled before start")
                job.done.set()
                continue
            try:
                job.result = self.handler(job)
                self.processed += 1
            except Exception as exc:  # noqa: BLE001
                job.error = exc
                self.failed += 1
                log.warning("memory job %s failed: %s", job.operation, exc)
            finally:
                job.done.set()

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "pending": self._queue.qsize(),
            "max_pending": self.max_pending,
            "processed": self.processed,
            "rejected": self.rejected,
            "failed": self.failed,
        }

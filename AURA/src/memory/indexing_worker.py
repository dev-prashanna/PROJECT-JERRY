"""Background indexing worker for PECM derived indexes.

Consumes :class:`memory_index_jobs` rows and performs the *derived* work —
embedding text, building episodes — after the authoritative event write has
already been acknowledged.  Fresh recall never blocks on this worker: the
store's FTS column is written in the same transaction as the event itself
(``indexed_immediately``), and recent-context retrieval reads the events
table directly.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

log = logging.getLogger("aura.memory.indexing")


class MemoryIndexingWorker:
    """Pulls pending index jobs and runs embed/episode work off-thread."""

    def __init__(self, engine):
        self.engine = engine
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self.processed = 0
        self.failed = 0

    # -- lifecycle --------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, poll_interval_s: float = 0.25) -> None:
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, args=(poll_interval_s,),
                name="aura-memory-indexer", daemon=True,
            )
            self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        with self._lock:
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=timeout_s)
                self._thread = None

    # -- main loop --------------------------------------------------------
    def _loop(self, poll_interval_s: float) -> None:
        store = getattr(self.engine, "store", None)
        if store is None:
            return
        batch = max(1, int(getattr(self.engine.settings, "memory_index_worker_batch", 16)))
        while not self._stop.is_set():
            try:
                jobs = store.pending_index_jobs(limit=batch)
                if not jobs:
                    self._stop.wait(poll_interval_s)
                    continue
                for job in jobs:
                    if self._stop.is_set():
                        break
                    self._process(job)
            except Exception as exc:  # noqa: BLE001
                log.debug("index worker tick failed: %s", exc)
                self._stop.wait(max(1.0, poll_interval_s * 4))

    def _process(self, job) -> None:
        store = self.engine.store
        try:
            if job.job_type == "embed_event":
                self._embed_event(job.source_id)
            elif job.job_type == "build_episode":
                builder = getattr(self.engine, "episode_builder", None)
                if builder is not None:
                    builder.build_for_turn(job.source_id)
            else:
                log.debug("unknown index job type %s", job.job_type)
            store.finish_index_job(job.job_id, status="done")
            self.processed += 1
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            attempts = int(job.attempts or 0) + 1
            status = "failed" if attempts >= 3 else "pending"
            store.finish_index_job(job.job_id, status=status, error=str(exc)[:400])
            log.warning("index job %s failed (attempt %d): %s", job.job_id, attempts, exc)

    def _embed_event(self, event_id: str) -> None:
        engine = self.engine
        store = engine.store
        embedder = getattr(engine, "text_embedder", None)
        text_index = getattr(engine, "text_index", None)
        if embedder is None or text_index is None:
            return
        if not getattr(engine.settings, "memory_semantic_search_enabled", False):
            return
        event = store.get_event(event_id)
        if event is None:
            return
        existing = store.text_embeddings_for_source(event_id)
        if existing:
            return
        vec = embedder.encode(event.content)
        from src.memory.vector_index import serialize_vector

        blob, dim, _ = serialize_vector(vec)
        store.insert_text_embedding(
            f"te_{event_id}", event_id, "event",
            embedder.name, dim, blob,
        )
        text_index.add(f"te_{event_id}", vec, {"source_id": event_id})


def process_pending_jobs(engine, *, max_jobs: int = 64) -> int:
    """Synchronously drain pending index jobs (used in tests / shutdown)."""
    store = getattr(engine, "store", None)
    if store is None:
        return 0
    worker = MemoryIndexingWorker(engine)
    done = 0
    for job in store.pending_index_jobs(limit=max_jobs):
        worker._process(job)
        done += 1
    return done

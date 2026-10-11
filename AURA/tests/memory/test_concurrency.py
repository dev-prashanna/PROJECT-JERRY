"""Concurrent engine access: parallel writes must serialize cleanly."""

from __future__ import annotations

import threading

from src.memory.engine import MultimodalMemoryEngine
from src.memory.models import RememberFactRequest


def test_parallel_fact_writes(engine: MultimodalMemoryEngine):
    transcripts = [f"I built a custom widget number {i}" for i in range(24)]
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(ts: str):
        try:
            res = engine.remember_fact(RememberFactRequest(transcript=ts, turn_id=ts))
            assert res.ok
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in transcripts]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    row = engine.store.execute_fetchone("SELECT COUNT(*) FROM declarative_memory")
    assert row is not None and int(row[0]) == len(transcripts)


def test_concurrent_read_during_write(engine: MultimodalMemoryEngine):
    engine.remember_fact(
        RememberFactRequest(transcript="My project is called ORION", turn_id="t0")
    )
    stop = threading.Event()
    reads: list[str] = []
    errors: list[BaseException] = []

    def reader():
        while not stop.is_set():
            try:
                got = engine.retrieve("ORION project")
                reads.append(got.message)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                return

    def writer():
        for i in range(10):
            engine.remember_fact(
                RememberFactRequest(transcript=f"My favorite number is {i}", turn_id=f"w{i}")
            )

    rt = threading.Thread(target=reader)
    wt = threading.Thread(target=writer)
    rt.start()
    wt.start()
    wt.join(timeout=10)
    stop.set()
    rt.join(timeout=5)
    assert not errors
    assert reads


def test_shutdown_is_idempotent(memory_settings):
    eng = MultimodalMemoryEngine(memory_settings.memory)
    eng.initialize()
    eng.shutdown()
    eng.shutdown()
    assert eng.store is None
